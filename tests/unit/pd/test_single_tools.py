# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native single tools cross the real parser/scheduler seam with a scripted worker IPC peer."""
import asyncio
from functools import partial
import json

import pytest

from pypto_serving.config.types import GenerateConfig, KVCacheGroupSpec, KVCacheSpec, RuntimeConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, ReplicaEngineCore
from pypto_serving.serving.server.ipc import ShutdownCommand, StepResult, decode_command, encode_result
from pypto_serving.serving.server.native import NativeServer, SessionScheduler
from tests.unit.model.qwen.test_tool_parser import QwenToolTokenizer
from tests.unit.pd.test_single_stream import Tokenizer, Worker, checkpoint, incoming


class ToolTokenizer(Tokenizer, QwenToolTokenizer):
    output_parser_id = "qwen3"
    output_parser_requires_tool_context = True
    eos_token_id = QwenToolTokenizer.vocab["<eos>"]

    def encode(self, text):
        return Tokenizer.encode(self, text) if text.startswith("[") else QwenToolTokenizer.encode(self, text)

    def decode(self, ids, **kwargs):
        return QwenToolTokenizer.decode(self, ids, **kwargs)


class ToolWorker(Worker):
    def __init__(self, ids):
        super().__init__()
        self.ids = ids

    def put(self, raw):
        command = decode_command(raw)
        if isinstance(command, ShutdownCommand):
            self.alive = False
            return
        self.commands.append(command)
        tokens = {}
        for item in command.prefill_requests:
            self.last[item.request_id] = 0
            tokens[item.request_id] = [self.ids[0]]
        for item in command.decode_requests:
            self.last[item.request_id] += 1
            tokens[item.request_id] = [self.ids[min(self.last[item.request_id], len(self.ids) - 1)]]
        self.pending.put((encode_result(StepResult(new_tokens=tokens, step_id=command.step_id)), False))


@pytest.mark.parametrize("mode", ["input", "input-output"])
@pytest.mark.parametrize("count,parallel,stream", [(1, True, True), (2, True, True), (2, False, True), (2, True, False)])
def test_single_stream_tool_delta_matches_confirmed_final_call(tmp_path, monkeypatch, mode, count, parallel, stream):
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "single-stream-credential")
    checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(dict(vocab_size=4096)))

    async def run():
        tokenizer = ToolTokenizer()
        output = "".join('<tool_call>' + json.dumps(dict(name="lookup", arguments=dict(city=city)),
            separators=(",", ":")) + '</tool_call>' for city in ["Oslo", "Berlin"][:count]) + "<eos>"
        ids = tokenizer.encode(output)
        worker = ToolWorker(ids)
        runtime = RuntimeConfig(page_size=128, max_batch_size=2, max_seq_len=512, max_num_batched_tokens=512,
            kv_cache_groups=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 4, num_blocks=16),))
        config = EngineConfig(model_id="Qwen3-14B", model_dir=str(tmp_path), runtime_config=runtime,
            max_num_running_reqs=2, max_num_scheduled_tokens=512, enable_prefix_cache=False)
        engine = AsyncLLMEngine(config, tokenizer, core_factory=partial(ReplicaEngineCore,
            scheduler_factory=partial(SessionScheduler, session_kv_mode=mode), worker_factory=worker.factory))
        await engine.start()
        try:
            server = NativeServer(engine, "Qwen3-14B", GenerateConfig(), session_kv_mode=mode)
            tools = [dict(type="function", function=dict(name="lookup", parameters=dict(type="object",
                properties=dict(city=dict(type="string")), required=["city"])))]
            prepared = await server.native_prepare(incoming("/prepare", dict(model="Qwen3-14B", tenant_id="tenant", cache_salt="salt", version_id="tool-child",
                messages=[dict(role="user", content="Look up Oslo")], tools=tools, tool_choice="auto",
                stream=stream, parallel_tool_calls=parallel, max_completion_tokens=128, temperature=0, top_p=1,
                chat_template_kwargs={"enable_thinking": False})))
            response = await server.native_generate(incoming("/generate", dict(json.loads(prepared.body),
                tenant_id="tenant", cache_salt="salt", version_id="tool-child", stream=stream)))
            if stream:
                frames = [json.loads(raw.removeprefix("data: ")) async for raw in response.body_iterator
                          if raw != "data: [DONE]\n\n"]
                assert not any("error" in frame for frame in frames)
                [final] = [frame for frame in frames if "native" in frame]
                deltas = [call for frame in frames if "native" not in frame for call in frame.get("tool_calls", [])]
                assert len(deltas) == (count if parallel else 1)
                assert [call["id"] for call in final["tool_calls"]] == [call["id"] for call in deltas]
                assert [call["function"] for call in final["tool_calls"]] == [call["function"] for call in deltas]
                assert frames[-1] is final
            else:
                final = json.loads(response.body)
            assert len(final["tool_calls"]) == (count if parallel else 1)
            assert all(call["function"]["name"] == "lookup" for call in final["tool_calls"])
            assert [json.loads(call["function"]["arguments"])["city"] for call in final["tool_calls"]] == ["Oslo", "Berlin"][:count if parallel else 1]
            assert final["chat_finish_reason"] == "tool_calls" and final["token_ids"] == ids
            caps = await server.native_capabilities(incoming("/capabilities", dict(model="Qwen3-14B")))
            assert json.loads(caps.body)["tools"] is True
            await server.native_release(incoming("/objects/release", dict(native=final["native"])))
            assert engine.single_core().kv_cache_manager.group_num_free_blocks("kv", 0) == 16
        finally:
            assert await engine.stop()
    asyncio.run(run())
