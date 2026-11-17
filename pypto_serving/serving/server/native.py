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

from pypto_serving.serving import request_timing as timing

from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse, StreamingResponse

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

    def __init__(self, *, session_kv_mode="input", **kwargs):
        super().__init__(**kwargs)
        self.session_kv_mode = session_kv_mode
        self.native_requests = {}
        self.native_objects = None

    def update_from_output(self, output, tokens):
        for scheduled in output.scheduled_requests:
            request = scheduled.request
            entry = self.native_requests.get(request.request_id)
            if (entry is None or entry.error is not None
                    or request.status.is_finished or request.status is RequestStatus.PREEMPTED):
                continue
            end = scheduled.num_computed_tokens + scheduled.num_new_tokens
            if end < request.num_prompt_tokens:
                continue
            if self.session_kv_mode == "input" and (entry.native is not None or not scheduled.is_prefill):
                continue
            # Consume only this successful device ACK. The token sampled by
            # this very step has no KV yet and is not in request.output_token_ids.
            try:
                computed = None
                if self.session_kv_mode == "input-output":
                    known = request.prompt_token_ids + request.output_token_ids
                    if end > len(known):
                        raise RuntimeError("Device ACK exceeds the known computed token prefix")
                    computed = known[:end]
                else:
                    end = request.num_prompt_tokens
                entry.native = self.native_objects.retain(entry, SimpleNamespace(
                    final=True, end=end, blocks=scheduled.block_ids_by_group,
                    partition=scheduled.cache_partition), computed_tokens=computed)
            except Exception as exc:
                entry.error = exc
        return super().update_from_output(output, tokens)


class NativeServer(ServingServer):
    def __init__(self, engine, model_id, generate, *, session_kv_mode="input"):
        self.core = engine.single_core()
        if (len(engine.config.worker_device_ids()) != 1
                or not isinstance(self.core.scheduler, SessionScheduler)
                or engine.config.executor_cls != "PyptoQwen14BExecutor"):
            raise ValueError("Native single deployment requires the Qwen3 single-rank engine")
        self.session_kv_mode = session_kv_mode
        if session_kv_mode != self.core.scheduler.session_kv_mode:
            raise ValueError("Server and scheduler session mode differ")
        if session_kv_mode == "input-output" and engine.config.resolve_runtime_config().num_speculative_tokens:
            raise ValueError("Native output retention requires ordinary Qwen3 decoding")
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
        for suffix, handler in (("capabilities", self.native_capabilities), ("prepare", self.native_prepare), ("tokenize", self.native_tokenize),
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
                digest(dict(deployment="single", session_kv_mode=self.session_kv_mode,
                            runtime=asdict(runtime), groups=[asdict(s) for s in cache.group_specs])),
                (Component("key", "kv", "kv"), Component("value", "kv", "kv")))
            node = Node("single", "single", self.boot, contract, uuid.uuid4().hex, Topology.for_tp(1, 1),
                        provider="local", prefix=cache.enable_prefix_cache)
            self.objects = NativeObjects(cache, node, SimpleNamespace(active={}), self.model_identity["fingerprint"],
                                         session_kv_mode=self.session_kv_mode)
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
            node_id=node.node_id, boot=node.boot, registry=node.registry, deployment="single", session_kv_mode=self.session_kv_mode,
            block_size=128, durability="volatile", max_model_len=self.engine.config.resolve_runtime_config().max_seq_len,
            raw_token_input=True, output_token_ids=True, native_retain=True, native_release=True,
            native_mount=True, streaming=True, tools=True, durable=False,
            ordinary_prefix_cache=objects.cache.enable_prefix_cache, model_identity=self.model_identity))

    @timing.timed("native.prepare")
    async def native_prepare(self, request: HTTPRequest):
        authorize(request.headers)
        objects = self._objects()
        raw = await request.json()
        public = ChatCompletionRequest.model_validate(raw)
        if public.model != self.model_id:
            raise ValueError("model differs from the single owner")
        prompt, config, parser, constraint = await self.prepare_chat(public)
        self._validate_generation(config)
        encoded = self.engine._tokenize_prompt(prompt)
        if encoded is None:
            raise ValueError("Native generation requires a non-empty prompt")
        tokens = list(encoded)
        self._budget(tokens, config)
        return JSONResponse(objects.prepare_contract(dict(model=self.model_id, layout=objects.node.contract.layout,
            prompt_token_ids=tokens, generation=asdict(config), parser=asdict(parser) if parser else None,
            constraint=constraint.to_wire() if constraint else None, parallel_tool_calls=public.parallel_tool_calls,
            model_identity=self.model_identity, rendering=self.rendering_profile(public),
            session_kv_mode=self.session_kv_mode), raw))

    @timing.timed("native.tokenize")
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
        return JSONResponse(objects.seal_prepared(dict(model=self.model_id, layout=objects.node.contract.layout,
            prompt_token_ids=tokens, generation=asdict(config), parser=asdict(parser) if parser else None,
            constraint=constraint.to_wire() if constraint else None, parallel_tool_calls=public.parallel_tool_calls,
            model_identity=self.model_identity, rendering=self.rendering_profile(public),
            session_kv_mode=self.session_kv_mode)))

    def _budget(self, tokens, config):
        if len(tokens) + config.max_new_tokens > self.engine.config.resolve_runtime_config().max_seq_len:
            raise RouterError(400, "context_length_exceeded", "Input tokens plus output budget exceed model context")

    @staticmethod
    def _validate_generation(config):
        if config.temperature != 0.0 or config.top_p != 1.0 or config.top_k is not None:
            raise ValueError("Native Qwen3 sessions require greedy sampling")

    async def native_generate(self, request: HTTPRequest):
        timing.mark("native.generate.validation.begin")
        authorize(request.headers)
        objects = self._objects()
        raw = await request.json()
        objects.check_prepared(raw)
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
        with timing.span("native.check_prepared"):
            objects.check_prepared(raw)
        if raw.get("session_kv_mode") != self.session_kv_mode:
            raise RouterError(409, "native_identity_mismatch", "Session mode differs from the configured owner")
        parent = raw.get("parent")
        with timing.span("native.mount"):
            prefix = objects.mount(parent, scope, tokens, mount_tokens=raw.get("mount_tokens")) if parent is not None else None
        rid = "single-" + uuid.uuid4().hex
        entry = SimpleNamespace(binding=SingleBinding(rid, digest(raw)), native_scope=scope,
            continuation=Continuation(tuple(tokens), asdict(config), None, None), native_parent=parent,
            mounted_tokens=prefix.token_count if prefix else 0, native=None, error=None, done=asyncio.Event())
        self.core.scheduler.native_requests[rid] = entry
        successful = False
        timing.mark("native.generate.validation.end")
        timing.mark("engine.submit")
        outputs = self.core.add_request(rid, "", config, prompt_token_ids=tokens,
            output_parser_spec=parser, constraint_spec=constraint, confirmed_prefix=prefix)

        async def cleanup():
            try:
                await outputs.aclose()
            finally:
                if not successful:
                    objects.discard_request(rid)
                self.core.scheduler.native_requests.pop(rid, None)
                entry.done.set()

        async def frames(first=None):
            nonlocal successful
            final = None
            previous_text = ""
            previous_reasoning = ""
            try:
                output = first
                while True:
                    if output is None:
                        try:
                            output = await anext(outputs)
                        except StopAsyncIteration:
                            break
                    tool_deltas = [self._tool_delta(item).model_dump(exclude_none=True)
                                   for item in output.tool_call_deltas
                                   if raw.get("parallel_tool_calls", True) or item.index == 0]
                    if config.stream:
                        text_delta = output.text_delta
                        reasoning_delta = output.reasoning_delta
                        if parser is None:
                            # Match ordinary Chat: finalization can flush text
                            # withheld by incremental decoding without updating
                            # the engine's pre-finalization delta.
                            text_delta = output.text[len(previous_text):] if output.text else ""
                            previous_text = output.text or previous_text
                            reasoning_delta = (output.reasoning[len(previous_reasoning):]
                                               if output.reasoning else "")
                            previous_reasoning = output.reasoning or previous_reasoning
                        if text_delta or reasoning_delta or tool_deltas:
                            timing.mark("server.first_content")
                            yield dict(request=rid, delta=text_delta, reasoning=reasoning_delta,
                                       tool_calls=tool_deltas)
                    if output.finished:
                        final = output
                    output = None
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
                result["terminal_evidence"] = objects.terminal_evidence(entry.native, raw)
                successful = True
                yield result
            finally:
                await cleanup()

        if config.stream:
            # Start the existing admission path immediately after the read-only mount.
            # It acquires the request prefix reference before its first await; no raw
            # page IDs cross the HTTP-header scheduling boundary without that pin.
            try:
                first = await anext(outputs)
            except BaseException:
                await cleanup()
                raise

            async def stream():
                try:
                    async with aclosing(frames(first)) as source:
                        async for frame in source:
                            yield "data: " + json.dumps(frame) + "\n\n"
                except Exception as exc:
                    code = exc.code if isinstance(exc, RouterError) else "native_execution_failed"
                    yield "data: " + json.dumps(dict(error=dict(code=code, message=str(exc)))) + "\n\n"
                yield "data: [DONE]\n\n"

            class NativeStreamingResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Header send can fail before the body generator starts.
                        await cleanup()

            return NativeStreamingResponse(stream(), media_type="text/event-stream")
        result = None
        async with aclosing(frames()) as source:
            async for result in source:
                pass
        return JSONResponse(result)

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
