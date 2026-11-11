# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal Stateful requests use the engine operation and identity seams."""

import asyncio
import json

import httpx
import pytest

from pypto_serving.router.app import ServingRouter
from tests.unit.pd.test_coordination import config


def test_native_stream_forwards_delta_before_save_ack_and_guards_terminal(monkeypatch):
    from starlette.requests import Request
    from tests.unit.pd.test_coordination import Peers, assert_idle
    from pypto_serving.pd.stateful import digest, owner

    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "native-stream-test-credential")

    async def run():
        cfg = config()
        peers = Peers(cfg)
        held = asyncio.Event()
        original = peers.handle
        payload = dict(model="model", tenant_id="tenant", cache_salt="salt", version_id="version",
                       prompt_token_ids=list(range(147)), stream=True)

        async def handle(req):
            if not req.url.path.endswith("/generate"):
                return await original(req)
            context = json.loads(req.headers["x-pd-context"])
            binding = context["binding"]
            node = peers.nodes[req.url.host]
            if node.role == "prefill":
                await held.wait()
                receipt = dict(owner(payload), node_id=node.node_id, boot=node.boot, registry=node.registry,
                    model=node.contract.model, layout=node.contract.layout, request_id=binding["request"],
                    durability="volatile", retained_tokens=128, confirmed_tokens=147, block_size=128,
                    prefix_digest=digest(payload["prompt_token_ids"][:128]), object_id="external-owner-receipt")
                return httpx.Response(200, json=dict(request=binding["request"], status="ready", native=receipt))
            delta = dict(request=binding["request"], delta="answer")
            terminal = dict(request=binding["request"], prompt_token_ids=payload["prompt_token_ids"],
                            token_ids=[42], text="answer", finish_reason="stop")
            return httpx.Response(200, text="".join(
                "data: " + json.dumps(value) + "\n\n" for value in (delta, terminal)) + "data: [DONE]\n\n",
                headers={"Content-Type": "text/event-stream"})

        router = ServingRouter(cfg, client_factory=lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(handle), timeout=1))
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": json.dumps(payload).encode()}
            await asyncio.Future()
            return {"type": "http.disconnect"}

        incoming = Request({"type": "http", "method": "POST", "path": "/internal/stateful/v1/generate",
                            "headers": [(b"authorization", b"Bearer native-stream-test-credential")]}, receive)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(incoming, "/internal/stateful/v1/generate")
            assert response.status_code == 200, response.body
            first = await anext(response.body_iterator)
            assert json.loads(first.removeprefix("data: "))["delta"] == "answer"
            pending = asyncio.create_task(anext(response.body_iterator))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not pending.done()
            held.set()
            terminal = json.loads((await pending).removeprefix("data: "))
            assert terminal["native"]["object_id"] == "external-owner-receipt"
            assert terminal["token_ids"] == [42]
            assert "".join([chunk async for chunk in response.body_iterator]) == "data: [DONE]\n\n"
            assert_idle(router)

    asyncio.run(run())


def test_internal_generation_requires_an_internal_credential(monkeypatch):
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "stateful-internal-test-credential-20261110")

    async def run():
        router = ServingRouter(config(), client_factory=lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(503)), timeout=1))
        async with router.app.router.lifespan_context(router.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=router.app),
                                         base_url="http://router") as client:
                response = await client.post("/internal/stateful/v1/generate", json={
                    "model": "model", "tenant_id": "tenant-a", "prompt_token_ids": [42]})
            assert response.status_code == 401

    asyncio.run(run())


def native_owner():
    from types import SimpleNamespace

    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
    from pypto_serving.pd.protocol import Component, Contract, Node, Topology
    from pypto_serving.pd.stateful import NativeObjects
    from pypto_serving.serving.memory.kv_cache import KvCacheManager

    cache = KvCacheManager(enable_prefix_cache=False)
    cache.init_groups((KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 16, num_blocks=16),),
                      max_batch_size=2)
    contract = Contract("qwen3-14b-paged", "Qwen3-14B", "test-layout", (Component("key", "kv", "kv"),))
    p = Node("p", "prefill", "boot-p", contract, "registry-p", Topology.for_tp(1, 1))
    d = Node("d", "decode", "boot-d", contract, "registry-d", Topology.for_tp(1, 1))
    return NativeObjects(cache, p, SimpleNamespace(active={})), p, d


def native_request(objects, p, d):
    from types import SimpleNamespace

    from pypto_serving.pd.integration import Chunk
    from pypto_serving.pd.protocol import Binding, Continuation
    from pypto_serving.pd.stateful import owner

    binding = Binding.create(p, d, {})
    tables = objects.cache.ensure_group_blocks(binding.request, 147, partition=0)
    entry = SimpleNamespace(binding=binding, continuation=Continuation(tuple(range(147)), {}, None, None),
                            native_scope=owner(dict(tenant_id="tenant", cache_salt="salt", version_id=binding.request)))
    chunk = Chunk(binding.request, 0, 0, 147, True, 7, tables, 0, {})
    return entry, chunk


def test_actual_retain_failure_does_not_publish_or_leak_a_native_object(monkeypatch):
    import pytest

    from pypto_serving.pd import stateful

    objects, p, d = native_owner()
    entry, chunk = native_request(objects, p, d)
    before = objects.cache.group_num_free_blocks("kv", 0)

    def unavailable_identity():
        raise RuntimeError("object identity allocation failed")

    # Fail after the real cache retain; rollback must undo its actual page pin.
    monkeypatch.setattr(stateful.uuid, "uuid4", unavailable_identity)
    with pytest.raises(RuntimeError, match="identity allocation failed"):
        objects.retain(entry, chunk)
    assert not objects.objects and not objects.requests
    objects.cache.release_all_group_requests(entry.binding.request)
    assert objects.cache.group_num_free_blocks("kv", 0) == before + 2
    # A physical cache rejection must likewise never issue a positive receipt.
    with pytest.raises(RuntimeError, match="Cannot retain unowned"):
        objects.retain(entry, chunk)
    assert not objects.objects and not objects.requests


def test_execution_abort_after_request_retirement_releases_only_its_native_pin():
    from types import SimpleNamespace

    from pypto_serving.pd.config import Config, Endpoint
    from pypto_serving.pd.service import Service

    objects, p, d = native_owner()
    original, original_chunk = native_request(objects, p, d)
    original_receipt = objects.retain(original, original_chunk)
    objects.cache.release_all_group_requests(original.binding.request)
    failed, failed_chunk = native_request(objects, p, d)
    failed_receipt = objects.retain(failed, failed_chunk)
    objects.cache.release_all_group_requests(failed.binding.request)
    core = SimpleNamespace(kv_cache_manager=objects.cache)
    settings = Config((Endpoint("p", 8100, "p"),), (Endpoint("d", 8101, "d"),), log_enabled=False)
    service = Service(settings, settings.prefill[0], "prefill", None, core, None)
    service.node, service.native_objects = p, objects
    # Router aborts P when D execution fails, even if P's HTTP entry is retired.
    asyncio.run(service.cancel_binding(failed.binding))
    assert objects.status(failed_receipt)["live"] is False
    assert objects.status(original_receipt)["live"] is True
    asyncio.run(service.cancel_binding(failed.binding))
    objects.release(original_receipt)
    assert not objects.objects and not objects.requests
    assert objects.cache.group_num_free_blocks("kv", 0) == 16


@pytest.mark.parametrize("remembered", [False, True])
def test_mismatched_abort_cannot_release_a_retired_requests_native_object(remembered):
    from types import SimpleNamespace

    import msgspec
    import pytest

    from pypto_serving.pd.config import Config, Endpoint
    from pypto_serving.pd.service import Service

    objects, p, d = native_owner()
    original, chunk = native_request(objects, p, d)
    receipt = objects.retain(original, chunk)
    objects.cache.release_all_group_requests(original.binding.request)
    settings = Config((Endpoint("p", 8100, "p"),), (Endpoint("d", 8101, "d"),), log_enabled=False)
    service = Service(settings, settings.prefill[0], "prefill", None,
                      SimpleNamespace(kv_cache_manager=objects.cache), None)
    service.node, service.native_objects = p, objects
    if remembered:
        service.remember_terminal(original.binding)
    invalid = msgspec.structs.replace(original.binding, input_digest="wrong-input")
    with pytest.raises(ValueError, match="binding mismatch"):
        asyncio.run(service.cancel_binding(invalid))
    assert objects.status(receipt)["live"] is True
    asyncio.run(service.cancel_binding(original.binding))
    assert objects.status(receipt)["live"] is False


def test_native_capabilities_require_the_internal_credential(monkeypatch):
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "native-capabilities-test-credential")

    async def run():
        router = ServingRouter(config(), client_factory=lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(503)), timeout=1))
        async with router.app.router.lifespan_context(router.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=router.app),
                                         base_url="http://router") as client:
                response = await client.post("/internal/stateful/v1/capabilities", json={"model": "model"})
            assert response.status_code == 401

    asyncio.run(run())


def test_live_native_mount_attaches_parent_pages_before_scheduling_suffix():
    from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig

    objects, p, d = native_owner()
    parent, parent_chunk = native_request(objects, p, d)
    receipt = objects.retain(parent, parent_chunk)
    objects.cache.release_all_group_requests(parent.binding.request)
    child_ids = list(parent.continuation.tokens) + list(range(147, 170))
    child_scope = dict(parent.native_scope, version_id="child")
    prefix = objects.mount(receipt, child_scope, child_ids)

    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=False), objects.cache)
    scheduler.add_request(Request("child", child_ids, 8, confirmed_prefix=prefix))
    [scheduled] = scheduler.schedule().scheduled_requests
    assert scheduled.num_computed_tokens == 128
    assert scheduled.num_new_tokens == 42
    assert scheduled.block_ids_by_group["kv"][0] == objects.status(receipt)["pages"]["kv"][0]
    scheduler.abort_request("child")
    assert objects.status(receipt)["live"] is True
    assert objects.status(receipt)["free_blocks"] == 15
    objects.release(receipt)
    assert objects.cache.group_num_free_blocks("kv", 0) == 16
