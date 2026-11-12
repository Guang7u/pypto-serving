# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single native streaming uses real scheduler/ownership with a CPU worker IPC peer."""
import asyncio
from functools import partial
import hashlib
import json
from queue import Queue
import threading
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from pypto_serving.config.types import GenerateConfig, KVCacheGroupSpec, KVCacheSpec, RuntimeConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, ReplicaEngineCore
from pypto_serving.serving.server.ipc import ShutdownCommand, StepResult, decode_command, encode_result
from pypto_serving.serving.server.native import NativeServer, SessionScheduler


class Tokenizer:
    eos_token_id = None
    bos_token_id = None

    def __init__(self, prompt_tokens=147):
        self.prompt_tokens = prompt_tokens
        self.tokenizer = SimpleNamespace(chat_template="cpu-template",
            backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(dict(type='cpu-test', base_tokens=self.prompt_tokens))),
            get_chat_template=lambda **kw: "cpu-template")

    def encode(self, text):
        messages = json.loads(text)
        tokens = list(range(self.prompt_tokens))
        return tokens if len(messages) == 1 else tokens + [7, 8] + list(range(200, 221))

    def decode(self, ids):
        return " ".join(str(value) for value in ids)

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


class Worker:
    """External worker IPC peer: only the last decode ACK is controlled."""
    def __init__(self):
        self.alive = True
        self.pid = 4242
        self.released = threading.Event()
        self.pending = Queue()
        self.last = {}
        self.commands = []
        self.fail_last = False

    def factory(self, config):
        ready = threading.Event()
        ready.set()
        return self, SimpleNamespace(put=self.put), SimpleNamespace(get=self.get), Queue(), ready, [16, 16]

    def put(self, raw):
        command = decode_command(raw)
        if isinstance(command, ShutdownCommand):
            self.alive = False
            return
        self.commands.append(command)
        tokens = {}
        for item in command.prefill_requests:
            self.last[item.request_id] = 7
            tokens[item.request_id] = [7]
        for item in command.decode_requests:
            value = self.last[item.request_id] + 1
            self.last[item.request_id] = value
            tokens[item.request_id] = [value]
        self.pending.put((encode_result(StepResult(new_tokens=tokens, step_id=command.step_id,
            error="owned worker fault" if self.fail_last and command.decode_requests else None)), bool(command.decode_requests)))

    def get(self, timeout=None):
        value, held = self.pending.get(timeout=timeout)
        if held and not self.released.wait(5):
            raise TimeoutError("CPU worker ACK was not released")
        return value

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def close(self):
        pass

    def terminate(self):
        self.alive = False

    kill = terminate


def checkpoint(path):
    weight = b"CPU fixture weights"
    (path / "model.safetensors").write_bytes(weight)
    documents = {
        "config.json": {"vocab_size": 1024}, "tokenizer.json": {"type": "cpu-test"},
        "tokenizer_config.json": {"chat_template": "cpu-template"},
        "model.safetensors.index.json": {"weight_map": {"weight": "model.safetensors"}},
        "download-verification.json": {"repo": "cpu-fixture", "revision": "revision-one", "files": [
            {"file": "model.safetensors", "bytes": len(weight), "sha256": hashlib.sha256(weight).hexdigest(),
             "upstream_lfs_sha256_verified": True}]},
    }
    for name, value in documents.items():
        (path / name).write_text(json.dumps(value))


def incoming(path, body):
    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}
    return Request({"type": "http", "method": "POST", "path": path,
        "headers": [(b"authorization", b"Bearer single-stream-credential")]}, receive)


@pytest.mark.parametrize("mode", ["input", "input-output"])
def test_single_stream_delivers_delta_before_last_ack_and_confirms_ownership(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "single-stream-credential")
    checkpoint(tmp_path)

    async def run():
        worker = Worker()
        runtime = RuntimeConfig(page_size=128, max_batch_size=2, max_seq_len=512, max_num_batched_tokens=512,
            kv_cache_groups=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 4, num_blocks=16),))
        config = EngineConfig(model_id="Qwen3-14B", model_dir=str(tmp_path), runtime_config=runtime,
            max_num_running_reqs=2, max_num_scheduled_tokens=512, enable_prefix_cache=False)
        engine = AsyncLLMEngine(config, Tokenizer(127 if mode == "input-output" else 147), core_factory=partial(ReplicaEngineCore,
            scheduler_factory=partial(SessionScheduler, session_kv_mode=mode), worker_factory=worker.factory))
        await engine.start()
        try:
            server = NativeServer(engine, "Qwen3-14B", GenerateConfig(), session_kv_mode=mode)
            tokenized = await server.native_tokenize(incoming("/tokenize", dict(model="Qwen3-14B",
                messages=[dict(role="user", content="cpu prompt")], max_completion_tokens=2, stream=True,
                temperature=0, top_p=1, chat_template_kwargs={"enable_thinking": False})))
            prepared = json.loads(tokenized.body)
            payload = dict(prepared, tenant_id="tenant", cache_salt="salt", version_id="stream-one", stream=True)
            response = await asyncio.wait_for(server.native_generate(incoming("/generate", payload)), 2)
            assert response.media_type == "text/event-stream"
            first = json.loads((await anext(response.body_iterator)).removeprefix("data: "))
            assert first["delta"] == "7" and "native" not in first
            pending = asyncio.create_task(anext(response.body_iterator))
            await asyncio.sleep(.01)
            assert not pending.done()
            worker.released.set()
            frames = [first]
            while True:
                try:
                    raw = await pending if pending is not None else await anext(response.body_iterator)
                    pending = None
                except StopAsyncIteration:
                    break
                if raw == "data: [DONE]\n\n":
                    continue
                frames.append(json.loads(raw.removeprefix("data: ")))
            assert "".join(frame.get("delta", "") for frame in frames) == "7 8"
            [final] = [frame for frame in frames if "native" in frame]
            assert final["token_ids"] == [7, 8] and final["native"]["retained_tokens"] == 128
            assert final["native"]["confirmed_tokens"] == (147 if mode == "input" else 128)
            status = await server.native_status(incoming("/objects/status", {"native": final["native"]}))
            assert json.loads(status.body)["live"] is True
            capabilities = await server.native_capabilities(incoming("/capabilities", {"model": "Qwen3-14B"}))
            assert json.loads(capabilities.body)["streaming"] is True
            await server.native_release(incoming("/objects/release", {"native": final["native"]}))
            assert engine.single_core().kv_cache_manager.group_num_free_blocks("kv", 0) == 16
        finally:
            worker.released.set()
            assert await engine.stop()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["input", "input-output"])
def test_single_stream_worker_failure_never_publishes_a_native_success(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "single-stream-credential")
    checkpoint(tmp_path)

    async def run():
        worker = Worker()
        worker.fail_last = True
        runtime = RuntimeConfig(page_size=128, max_batch_size=2, max_seq_len=512, max_num_batched_tokens=512,
            kv_cache_groups=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 4, num_blocks=16),))
        config = EngineConfig(model_id="Qwen3-14B", model_dir=str(tmp_path), runtime_config=runtime,
            max_num_running_reqs=2, max_num_scheduled_tokens=512, enable_prefix_cache=False)
        engine = AsyncLLMEngine(config, Tokenizer(127 if mode == "input-output" else 147),
            core_factory=partial(ReplicaEngineCore, scheduler_factory=partial(SessionScheduler, session_kv_mode=mode),
                                 worker_factory=worker.factory))
        await engine.start()
        try:
            server = NativeServer(engine, "Qwen3-14B", GenerateConfig(), session_kv_mode=mode)
            response = await server.native_tokenize(incoming("/tokenize", dict(model="Qwen3-14B",
                messages=[dict(role="user", content="cpu prompt")], max_completion_tokens=2, stream=True,
                temperature=0, top_p=1)))
            prepared = json.loads(response.body)
            response = await server.native_generate(incoming("/generate", dict(prepared, tenant_id="tenant",
                cache_salt="salt", version_id="stream-failure", stream=True)))
            first = json.loads((await anext(response.body_iterator)).removeprefix("data: "))
            assert first["delta"] == "7" and "native" not in first
            worker.released.set()
            frames = [json.loads(raw.removeprefix("data: ")) async for raw in response.body_iterator
                      if raw != "data: [DONE]\n\n"]
            assert len([frame for frame in frames if "error" in frame]) == 1
            assert not any("native" in frame for frame in frames)
            assert len({item.request_id for cmd in worker.commands for item in cmd.prefill_requests}) == 1
            assert engine.single_core().kv_cache_manager.group_num_free_blocks("kv", 0) == 16
        finally:
            worker.released.set()
            assert await engine.stop()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["input", "input-output"])
@pytest.mark.parametrize("close_stage", ["delta", "headers"])
def test_single_stream_close_retires_child_and_preserves_parent(tmp_path, monkeypatch, mode, close_stage):
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "single-stream-credential")
    checkpoint(tmp_path)

    async def run():
        worker = Worker()
        runtime = RuntimeConfig(page_size=128, max_batch_size=2, max_seq_len=512, max_num_batched_tokens=512,
            kv_cache_groups=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 4, num_blocks=16),))
        config = EngineConfig(model_id="Qwen3-14B", model_dir=str(tmp_path), runtime_config=runtime,
            max_num_running_reqs=2, max_num_scheduled_tokens=512, enable_prefix_cache=False)
        engine = AsyncLLMEngine(config, Tokenizer(127 if mode == "input-output" else 147),
            core_factory=partial(ReplicaEngineCore, scheduler_factory=partial(SessionScheduler, session_kv_mode=mode),
                                 worker_factory=worker.factory))
        await engine.start()
        try:
            server = NativeServer(engine, "Qwen3-14B", GenerateConfig(), session_kv_mode=mode)
            messages = [dict(role="user", content="cpu prompt")]
            async def prepare(stream):
                response = await server.native_tokenize(incoming("/tokenize", dict(model="Qwen3-14B",
                    messages=messages, max_completion_tokens=2, stream=stream, temperature=0, top_p=1)))
                return json.loads(response.body)
            worker.released.set()
            parent_response = await server.native_generate(incoming("/generate", dict(await prepare(False),
                tenant_id="tenant", cache_salt="salt", version_id="parent", stream=False)))
            parent = json.loads(parent_response.body)["native"]
            before = json.loads((await server.native_status(incoming("/objects/status", {"native": parent}))).body)
            worker.released.clear()
            messages.extend([dict(role="assistant", content="7 8"), dict(role="user", content="continue")])
            scope = dict(model="Qwen3-14B", tenant_id="tenant", cache_salt="salt", version_id="child")
            response = await server.native_generate(incoming("/generate", dict(await prepare(True), **scope,
                parent=parent, mount_tokens=128, stream=True)))
            if close_stage == "delta":
                first = json.loads((await anext(response.body_iterator)).removeprefix("data: "))
                assert first["delta"] == "7" and "native" not in first
                worker.released.set()
                await asyncio.wait_for(response.body_iterator.aclose(), 2)
            else:
                async def receive():
                    await asyncio.Event().wait()
                async def send(message):
                    assert message["type"] == "http.response.start"
                    worker.released.set()
                    raise OSError("owned disconnect before body iteration")
                with pytest.raises(BaseException):
                    await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            cancelled = await asyncio.wait_for(server.native_cancel(incoming("/objects/cancel", scope)), 2)
            assert json.loads(cancelled.body) == dict(live=False, session_references=0)
            async def retired():
                while engine.single_core().kv_cache_manager.group_num_free_blocks("kv", 0) != 15:
                    await asyncio.sleep(.01)
            await asyncio.wait_for(retired(), 2)
            after = json.loads((await server.native_status(incoming("/objects/status", {"native": parent}))).body)
            assert after == before
            assert engine.single_core().kv_cache_manager.group_num_free_blocks("kv", 0) == 15
            await server.native_release(incoming("/objects/release", {"native": parent}))
            assert engine.single_core().kv_cache_manager.group_num_free_blocks("kv", 0) == 16
        finally:
            worker.released.set()
            assert await engine.stop()
    asyncio.run(run())
