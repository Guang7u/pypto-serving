# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-rank native session ownership, composed with the ordinary serving engine."""

import asyncio
from contextlib import aclosing
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse

from pypto_serving.config.types import GenerateConfig
from pypto_serving.pd.protocol import Component, Contract, Continuation, Node, Topology, digest
from pypto_serving.pd.stateful import INTERNAL_PREFIX, NativeObjects, authorize, owner
from pypto_serving.router.http import RouterError
from pypto_serving.serving.constraints import ConstraintSpec
from pypto_serving.serving.reasoning import OutputParserSpec
from pypto_serving.serving.sched.scheduler import RequestStatus, Scheduler
from .server import ChatCompletionRequest, FunctionCall, ServingServer, ToolCall
from .model_identity import load_model_identity


@dataclass(frozen=True)
class SingleBinding:
    request: str
    input_digest: str


class SessionScheduler(Scheduler):
    """Retain input pages on the real prefill ACK, before ordinary retirement.

    Only registered native requests acquire a session reference. The ordinary
    scheduler, including its in-flight step pins and worker release barrier,
    continues to own request execution and retirement.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.native_requests = {}
        self.native_objects = None

    def update_from_output(self, output, tokens):
        for scheduled in output.scheduled_requests:
            request = scheduled.request
            entry = self.native_requests.get(request.request_id)
            if (entry is None or entry.native is not None or entry.error is not None
                    or request.status.is_finished or request.status is RequestStatus.PREEMPTED
                    or not scheduled.is_prefill
                    or scheduled.num_computed_tokens + scheduled.num_new_tokens < request.num_prompt_tokens):
                continue
            # The snapshot is exactly the table whose successful device result
            # the engine is consuming; optimistic counters are not confirmation.
            try:
                entry.native = self.native_objects.retain(entry, SimpleNamespace(
                    final=True, end=request.num_prompt_tokens,
                    blocks=scheduled.block_ids_by_group, partition=scheduled.cache_partition))
            except Exception as exc:
                entry.error = exc
        return super().update_from_output(output, tokens)


class NativeServer(ServingServer):
    def __init__(self, engine, model_id, generate):
        self.core = engine.single_core()
        if (len(engine.config.worker_device_ids()) != 1
                or not isinstance(self.core.scheduler, SessionScheduler)
                or engine.config.executor_cls != "PyptoQwen14BExecutor"):
            raise ValueError("Native single deployment requires the Qwen3 single-rank engine")
        self.boot = uuid.uuid4().hex
        self.model_identity = load_model_identity(engine.config.model_dir, model_id, engine.tokenizer)
        self.objects = None
        super().__init__(engine, model_id, generate)

    def _register_exception_handlers(self):
        super()._register_exception_handlers()

        @self.app.exception_handler(RouterError)
        async def native_error(request, exc):
            return JSONResponse(dict(object="error", code=exc.code, message=str(exc)), status_code=exc.status)

    def _register_routes(self):
        super()._register_routes()
        for suffix, handler in (("capabilities", self.native_capabilities), ("tokenize", self.native_tokenize),
                                ("generate", self.native_generate), ("objects/status", self.native_status),
                                ("objects/release", self.native_release), ("objects/cancel", self.native_cancel)):
            self.app.add_api_route(INTERNAL_PREFIX + suffix, handler, methods=["POST"])

    def _objects(self):
        if not self.engine.is_ready():
            raise RouterError(503, "native_unavailable", "Single engine is unavailable")
        if self.objects is None:
            cache = self.core.kv_cache_manager
            runtime = self.engine.config.resolve_runtime_config()
            contract = Contract("qwen3-14b-paged", self.model_id,
                digest(dict(deployment="single", runtime=asdict(runtime), groups=[asdict(s) for s in cache.group_specs])),
                (Component("key", "kv", "kv"), Component("value", "kv", "kv")))
            node = Node("single", "single", self.boot, contract, uuid.uuid4().hex, Topology.for_tp(1, 1),
                        provider="local", prefix=cache.enable_prefix_cache)
            self.objects = NativeObjects(cache, node, SimpleNamespace(active={}), self.model_identity["fingerprint"])
            self.core.scheduler.native_objects = self.objects
        return self.objects

    async def _health(self):
        ready = self.engine.is_ready()
        return JSONResponse(dict(status="ok" if ready else "not_ready", deployment="single",
                                 replicas_ready=int(ready)), status_code=200 if ready else 503)

    async def native_capabilities(self, request: HTTPRequest):
        authorize(request.headers)
        objects = self._objects()
        if (await request.json()).get("model") != self.model_id:
            raise ValueError("model differs from the single owner")
        node = objects.node
        return JSONResponse(dict(backend="pypto", model=self.model_id, layout=node.contract.layout,
            node_id=node.node_id, boot=node.boot, registry=node.registry, deployment="single", session_kv_mode="input",
            block_size=128, durability="volatile", max_model_len=self.engine.config.resolve_runtime_config().max_seq_len,
            raw_token_input=True, output_token_ids=True, native_retain=True, native_release=True,
            native_mount=True, streaming=False, tools=False, durable=False,
            ordinary_prefix_cache=objects.cache.enable_prefix_cache, model_identity=self.model_identity))

    async def native_tokenize(self, request: HTTPRequest):
        authorize(request.headers)
        objects = self._objects()
        public = ChatCompletionRequest.model_validate(await request.json())
        if public.model != self.model_id:
            raise ValueError("model differs from the single owner")
        prompt, config, parser, constraint = await self.prepare_chat(public)
        self._validate_generation(config)
        encoded = self.engine._tokenize_prompt(prompt)
        if encoded is None:
            raise ValueError("Native generation requires a non-empty prompt")
        tokens = list(encoded)
        self._budget(tokens, config)
        return JSONResponse(dict(model=self.model_id, layout=objects.node.contract.layout,
            prompt_token_ids=tokens, generation=asdict(config), parser=asdict(parser) if parser else None,
            constraint=constraint.to_wire() if constraint else None, parallel_tool_calls=public.parallel_tool_calls,
            model_identity=self.model_identity, rendering=self.rendering_profile(public)))

    def _budget(self, tokens, config):
        if len(tokens) + config.max_new_tokens > self.engine.config.resolve_runtime_config().max_seq_len:
            raise RouterError(400, "context_length_exceeded", "Input tokens plus output budget exceed model context")

    @staticmethod
    def _validate_generation(config):
        if config.temperature != 0.0 or config.top_p != 1.0 or config.top_k is not None:
            raise ValueError("Native Qwen3 sessions require greedy sampling")
        if config.stream:
            raise ValueError("Native single streaming is not yet supported")

    async def native_generate(self, request: HTTPRequest):
        authorize(request.headers)
        objects = self._objects()
        raw = await request.json()
        if raw.get("model_identity") != self.model_identity:
            raise RouterError(409, "model_identity_mismatch", "Native generation checkpoint identity differs")
        if raw.get("model") != self.model_id or raw.get("layout") != objects.node.contract.layout:
            raise ValueError("Native generation model/layout mismatch")
        scope = owner(raw)
        tokens = raw.get("prompt_token_ids")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("prompt_token_ids must be non-empty non-negative integer IDs")
        vocab = json.loads((Path(self.engine.config.model_dir) / "config.json").read_text())["vocab_size"]
        if any(t >= vocab for t in tokens):
            raise ValueError("prompt_token_ids contains an out-of-vocabulary token")
        if not isinstance(raw.get("generation"), dict):
            raise ValueError("generation must be an object")
        config = GenerateConfig(**raw["generation"])
        self._validate_generation(config)
        if type(raw.get("stream", False)) is not bool or raw.get("stream", False) != config.stream:
            raise ValueError("Native stream flag differs from generation configuration")
        if type(raw.get("parallel_tool_calls", True)) is not bool:
            raise ValueError("Native parallel_tool_calls must be a boolean")
        self._budget(tokens, config)
        parser = OutputParserSpec(**raw["parser"]) if raw.get("parser") else None
        constraint = ConstraintSpec.from_wire(raw["constraint"]) if raw.get("constraint") else None
        if constraint is not None:
            await self._preflight_constraint(constraint)
        parent = raw.get("parent")
        prefix = objects.mount(parent, scope, tokens) if parent is not None else None
        rid = "single-" + uuid.uuid4().hex
        entry = SimpleNamespace(binding=SingleBinding(rid, digest(raw)), native_scope=scope,
            continuation=Continuation(tuple(tokens), asdict(config), None, None), native_parent=parent,
            mounted_tokens=prefix.token_count if prefix else 0, native=None, error=None, done=asyncio.Event())
        self.core.scheduler.native_requests[rid] = entry
        successful = False
        try:
            final = None
            async with aclosing(self.core.add_request(rid, "", config, prompt_token_ids=tokens,
                    output_parser_spec=parser, constraint_spec=constraint, confirmed_prefix=prefix)) as outputs:
                async for output in outputs:
                    if output.finished:
                        final = output
            if entry.error is not None:
                raise entry.error
            if (final is None or final.finish_reason not in {"FINISHED_EOS", "FINISHED_STOP", "FINISHED_LENGTH"}
                    or entry.native is None):
                raise RouterError(502, "native_retention_unconfirmed", "Single execution did not confirm a session")
            status = objects.status(entry.native)
            if status["request_active"] or status["transfer_references"]:
                raise RouterError(502, "native_retirement_unconfirmed", "Single request has not retired")
            calls = final.tool_calls if raw.get("parallel_tool_calls", True) else final.tool_calls[:1]
            result = dict(request=rid, native=entry.native, prompt_token_ids=tokens, token_ids=list(final.token_ids),
                text=final.text, finish_reason=final.finish_reason, chat_finish_reason=self._chat_finish_reason(final),
                reasoning=final.reasoning, tool_calls=[ToolCall(id=call.id, function=FunctionCall(
                    name=call.name, arguments=call.arguments)).model_dump() for call in calls])
            successful = True
            return JSONResponse(result)
        finally:
            if not successful:
                objects.discard_request(rid)
            self.core.scheduler.native_requests.pop(rid, None)
            entry.done.set()

    async def native_status(self, request: HTTPRequest):
        authorize(request.headers)
        return JSONResponse(self._objects().status((await request.json()).get("native")))

    async def native_release(self, request: HTTPRequest):
        authorize(request.headers)
        return JSONResponse(self._objects().release((await request.json()).get("native")))

    async def native_cancel(self, request: HTTPRequest):
        authorize(request.headers)
        raw = await request.json()
        if raw.get("model") != self.model_id:
            raise ValueError("model differs from the single owner")
        objects, scope = self._objects(), owner(raw)
        for rid, entry in tuple(self.core.scheduler.native_requests.items()):
            if entry.native_scope == scope:
                await self.core.abort_request(rid)
                await asyncio.wait_for(entry.done.wait(), timeout=30)
        objects.discard_owner(scope)
        return JSONResponse(dict(live=False, session_references=0))
