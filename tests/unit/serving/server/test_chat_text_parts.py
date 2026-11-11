# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Gateway-restored Responses histories contain Chat text content parts."""

import pytest
from pydantic import ValidationError

from pypto_serving.serving.server.server import ChatCompletionRequest


def test_restored_assistant_text_parts_are_normalized_for_chat_template():
    request = ChatCompletionRequest.model_validate({
        "messages": [
            {"role": "user", "content": "Remember ORCHID."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "O"}, {"type": "text", "text": "K."},
            ]},
            {"role": "user", "content": "What was my code?"},
        ],
    })
    assert request.messages[1].content == "OK."
    assert request.messages[1].model_dump()["content"] == "OK."


@pytest.mark.parametrize("content", [
    [{"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}],
    [{"type": "text"}],
    [{"type": "text", "text": 123}],
    ["text"],
])
def test_non_text_or_malformed_parts_are_rejected(content):
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": content}]})


@pytest.mark.parametrize("content", [None, "plain text", ""])
def test_existing_content_values_are_preserved(content):
    request = ChatCompletionRequest.model_validate({"messages": [{"role": "assistant", "content": content}]})
    assert request.messages[0].content == content


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_restored_reasoning_alias_is_preserved_for_the_actual_template(field):
    request = ChatCompletionRequest.model_validate({"messages": [
        {"role": "assistant", "content": "READY", field: "Saved explanation"},
        {"role": "user", "content": "Continue"},
    ]})
    assert request.messages[0].reasoning == "Saved explanation"
    assert request.messages[0].model_dump()["reasoning"] == "Saved explanation"
