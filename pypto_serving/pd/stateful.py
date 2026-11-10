# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Volatile Qwen3 complete-page ownership, independent of request/transfer pins.

This module has no model-loading dependency: Router uses its credential check.
Only the P owner can issue receipts, after retaining its confirmed cache pages.
"""

import hashlib
import hmac
import os
import re
import secrets
import uuid
from dataclasses import dataclass

from pypto_serving.router.http import RouterError
from .protocol import Binding, digest

INTERNAL_PREFIX = "/internal/stateful/v1/"


def authorize(headers):
    token = os.getenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "")
    if not token or not secrets.compare_digest(headers.get("authorization", "").encode(),
                                                ("Bearer " + token).encode()):
        raise RouterError(401, "invalid_internal_credentials", "Invalid internal credentials")


def owner(payload):
    values = {}
    for name in ("tenant_id", "cache_salt", "version_id"):
        value = payload.get(name)
        if not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError(f"{name} must be a non-empty string of at most 256 characters")
        values[name] = value
    values["cache_salt_digest"] = hashlib.sha256(values.pop("cache_salt").encode()).hexdigest()
    return values


@dataclass
class _Object:
    receipt: dict
    blocks: dict
    partition: int
    binding: Binding


class NativeObjects:
    """One independent cache pin per live object; authenticated release receipts.

    Receipts are scoped to this boot, registry, model/layout and owner. Once a
    receipt is released its signature proves an idempotent repeat without an
    unbounded table of released objects or reuse of bare block IDs as identity.
    """

    def __init__(self, cache, node, pins):
        if (node.contract.adapter != "qwen3-14b-paged" or cache.group_names != ("kv",)
                or cache.group_specs[0].spec.token_capacity != 128):
            raise ValueError("Stateful retention requires the Qwen3-14B paged layout")
        self.cache, self.node, self.pins = cache, node, pins
        self.secret = secrets.token_bytes(32)
        self.objects: dict[str, _Object] = {}
        self.requests: dict[str, str] = {}

    def _signature(self, receipt, nonce):
        return hmac.new(self.secret, (nonce + ":" + digest(receipt)).encode(), hashlib.sha256).hexdigest()

    def retain(self, entry, chunk):
        if entry.binding.request in self.requests:
            raise RuntimeError("Request already owns a native object")
        if not chunk.final or chunk.end != len(entry.continuation.tokens):
            raise ValueError("Native retain requires the confirmed final prompt chunk")
        boundary = chunk.end // 128 * 128
        tables = {"kv": list(chunk.blocks["kv"][:boundary // 128])}
        if len(tables["kv"]) != boundary // 128:
            raise ValueError("Confirmed prompt does not have enough complete physical pages")
        receipt = dict(entry.native_scope, node_id=self.node.node_id, boot=self.node.boot,
                       registry=self.node.registry, model=self.node.contract.model, layout=self.node.contract.layout,
                       request_id=entry.binding.request, partition=chunk.partition,
                       durability="volatile", retained_tokens=boundary, confirmed_tokens=chunk.end,
                       block_size=128, prefix_digest=digest(list(entry.continuation.tokens[:boundary])))
        self.cache.retain_group_block_snapshot(tables, chunk.partition)
        key = None
        try:
            nonce = uuid.uuid4().hex
            key = nonce + "." + self._signature(receipt, nonce)
            receipt["object_id"] = key
            self.objects[key] = _Object(receipt, tables, chunk.partition, entry.binding)
            self.requests[entry.binding.request] = key
        except BaseException:
            if key is not None:
                self.objects.pop(key, None)
                self.requests.pop(entry.binding.request, None)
            self.cache.release_group_block_snapshot(tables, chunk.partition)
            raise
        return dict(receipt)

    def _check(self, receipt):
        if not isinstance(receipt, dict):
            raise ValueError("native must be an object receipt")
        expected = dict(node_id=self.node.node_id, boot=self.node.boot, registry=self.node.registry,
                        model=self.node.contract.model, layout=self.node.contract.layout)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise RouterError(409, "native_identity_mismatch", "Native object belongs to a different node identity")
        unsigned = dict(receipt)
        key = unsigned.pop("object_id", "")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}\.[0-9a-f]{64}", key):
            raise RouterError(404, "native_object_not_found", "Unknown native object")
        nonce, signature = key.split(".")
        if not hmac.compare_digest(signature, self._signature(unsigned, nonce)):
            raise RouterError(403, "native_owner_mismatch", "Native object receipt does not match its owner")
        return key

    def status(self, receipt):
        key = self._check(receipt)
        record = self.objects.get(key)
        request = receipt["request_id"]
        return dict(live=record is not None, session_references=int(record is not None),
                    request_active=self.cache.group_request_partition(request) is not None,
                    transfer_references=int(request in self.pins.active),
                    free_blocks=self.cache.group_num_free_blocks("kv", receipt["partition"]),
                    pages={name: list(ids) for name, ids in record.blocks.items()} if record is not None else {})

    def release(self, receipt):
        key = self._check(receipt)
        record = self.objects.get(key)
        if record is not None:
            self.cache.release_group_block_snapshot(record.blocks, record.partition)
            del self.objects[key]
            self.requests.pop(record.receipt["request_id"], None)
        return self.status(receipt)

    def validate_binding(self, binding):
        key = self.requests.get(binding.request)
        if key is not None and self.objects[key].binding != binding:
            raise ValueError("native binding mismatch")

    def discard_request(self, request, *, binding=None):
        key = self.requests.get(request)
        if key is not None:
            record = self.objects[key]
            if binding is not None:
                self.validate_binding(binding)
            self.release(record.receipt)

    def discard_owner(self, scope):
        for record in tuple(self.objects.values()):
            if all(record.receipt.get(key) == value for key, value in scope.items()):
                self.release(record.receipt)

    def close(self):
        for record in list(self.objects.values()):
            self.release(record.receipt)
