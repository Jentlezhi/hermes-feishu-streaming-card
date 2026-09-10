"""Unit tests for the CardKit element-level streaming transport.

These cover the safety contract, not just the happy path: a message this
transport owns must never fall back to ``PATCH`` (Feishu rejects it for a
``card_id`` message), and any failure has to degrade to a full-card CardKit
update instead of dropping the card.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from hermes_feishu_card import cardkit_stream
from hermes_feishu_card.cardkit_stream import (
    MAX_STREAMABLE_CHARS,
    STREAMING_ELEMENT_ID,
    CardKitRegistry,
    collapse_main_content,
    deliver_card_update,
    registry,
    send_cardkit_delivery,
    streaming_enabled,
    structure_fingerprint,
    with_body_text,
    with_streaming_config,
)
from hermes_feishu_card.feishu_client import FeishuAPIError, FeishuSendResult


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    cardkit_stream.registry().reset()


def make_card(
    text: str,
    *,
    footer: str = "生成中",
    chunks: Optional[List[str]] = None,
    header_template: Optional[str] = None,
) -> Dict[str, Any]:
    parts = chunks if chunks is not None else [text]
    elements: List[Dict[str, Any]] = []
    for index, part in enumerate(parts):
        element_id = "main_content" if index == 0 else f"main_content_{index}"
        elements.append({"tag": "markdown", "element_id": element_id, "content": part})
    elements.append({"tag": "hr", "element_id": "main_divider"})
    elements.append({"tag": "markdown", "element_id": "footer", "content": footer})
    card: Dict[str, Any] = {
        "schema": "2.0",
        "config": {"update_multi": True},
        "body": {"elements": elements},
    }
    if header_template is not None:
        card["header"] = {
            "template": header_template,
            "title": {"tag": "plain_text", "content": "Hermes Agent"},
        }
    return card


class FakeClient:
    """Records CardKit calls and can be told to fail specific ones."""

    def __init__(
        self,
        *,
        create_error: Optional[Exception] = None,
        send_error: Optional[Exception] = None,
        stream_errors: Optional[List[Optional[Exception]]] = None,
        update_error: Optional[Exception] = None,
    ) -> None:
        self.calls: List[tuple] = []
        self.create_error = create_error
        self.send_error = send_error
        self.stream_errors = list(stream_errors or [])
        self.update_error = update_error

    async def cardkit_create_card(self, card: Dict[str, Any]) -> str:
        self.calls.append(("create", with_streaming_config(card).get("config", {})))
        if self.create_error is not None:
            raise self.create_error
        return "card-1"

    async def cardkit_set_streaming(
        self, card_id: str, *, enabled: bool, sequence: int = 0, **_: Any
    ) -> None:
        self.calls.append(("set_streaming", enabled, sequence))

    async def cardkit_update_card(
        self, card_id: str, card: Dict[str, Any], *, sequence: int = 0, **_: Any
    ) -> None:
        body = card["body"]["elements"][0].get("content")
        self.calls.append(("update_card", card_id, sequence, body))
        if self.update_error is not None:
            raise self.update_error

    async def cardkit_stream_element(
        self,
        card_id: str,
        element_id: str,
        content: str,
        *,
        sequence: int = 0,
        **_: Any,
    ) -> None:
        self.calls.append(("stream_element", element_id, sequence, content))
        if self.stream_errors:
            error = self.stream_errors.pop(0)
            if error is not None:
                raise error

    async def send_card_delivery(
        self, chat_id: str, card: Dict[str, Any], **kwargs: Any
    ) -> FeishuSendResult:
        self.calls.append(("send_card_delivery", chat_id, kwargs.get("content_override")))
        if self.send_error is not None:
            raise self.send_error
        return FeishuSendResult(message_id="om-1", retry_count=0)

    async def update_card_message(self, message_id: str, card: Dict[str, Any]) -> None:
        self.calls.append(("patch_message", message_id))

    def names(self) -> List[str]:
        return [call[0] for call in self.calls]


def streamed_closed() -> FeishuAPIError:
    return FeishuAPIError(
        "Feishu API application failure",
        api_code=cardkit_stream.STREAMING_CLOSED_API_CODE,
        retryable=False,
        outcome="not_sent",
    )


# --- configuration ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("on", True),
        ("no", False),
        (None, False),
        ("", False),
    ],
)
def test_streaming_enabled_truth_table(value: Any, expected: bool) -> None:
    assert streaming_enabled({"cardkit_streaming": value}) is expected


def test_streaming_disabled_without_config() -> None:
    assert streaming_enabled(None) is False
    assert streaming_enabled({}) is False


def test_with_streaming_config_preserves_existing_keys() -> None:
    card = {"schema": "2.0", "config": {"streaming_mode": False, "summary": {"content": "x"}}}
    prepared = with_streaming_config(card)
    assert prepared["config"]["streaming_mode"] is False
    assert prepared["config"]["summary"] == {"content": "x"}
    assert prepared is not card
    # untouched original
    assert card["config"] == {"streaming_mode": False, "summary": {"content": "x"}}


# --- card shaping ----------------------------------------------------------


def test_collapse_merges_chunked_body_into_single_element() -> None:
    card = make_card("", chunks=["第一段", "第二段", "第三段"])
    merged, text = collapse_main_content(card)
    assert text == "第一段\n\n第二段\n\n第三段"
    ids = [element["element_id"] for element in merged["body"]["elements"]]
    assert ids == ["main_content", "main_divider", "footer"]
    assert merged["body"]["elements"][0]["content"] == text
    # the input card is not mutated
    assert len(card["body"]["elements"]) == 5


def test_collapse_returns_none_when_too_long() -> None:
    huge = "x" * (MAX_STREAMABLE_CHARS + 1)
    merged, text = collapse_main_content(make_card(huge))
    assert text is None
    assert merged["body"]["elements"][0]["content"] == huge


def test_collapse_without_body_element() -> None:
    merged, text = collapse_main_content({"body": {"elements": [{"tag": "hr"}]}})
    assert text is None
    assert merged["body"]["elements"] == [{"tag": "hr"}]


def test_fingerprint_ignores_body_text_but_tracks_structure() -> None:
    first = structure_fingerprint(make_card("答案 A", footer="生成中"))
    second = structure_fingerprint(make_card("答案 B", footer="生成中"))
    assert first == second
    changed = structure_fingerprint(make_card("答案 A", footer="已完成"))
    assert changed != first


def test_with_body_text_replaces_only_body() -> None:
    card = make_card("新答案", footer="生成中")
    replaced = with_body_text(card, "旧答案")
    assert replaced["body"]["elements"][0]["content"] == "旧答案"
    assert card["body"]["elements"][0]["content"] == "新答案"


# --- update delivery -------------------------------------------------------


@pytest.mark.asyncio
async def test_update_returns_false_for_unknown_message() -> None:
    client = FakeClient()
    assert await deliver_card_update(client, "om-unknown", make_card("hi")) is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_update_streams_body_when_structure_unchanged() -> None:
    client = FakeClient()
    state = registry().bind("om-1", "card-1")
    state.fingerprint = structure_fingerprint(make_card(""))
    state.streaming_open = True

    assert await deliver_card_update(client, "om-1", make_card("正在生成")) is True

    assert client.names() == ["stream_element"]
    _, element_id, sequence, content = client.calls[0]
    assert element_id == STREAMING_ELEMENT_ID
    assert content == "正在生成"
    assert sequence == 1


@pytest.mark.asyncio
async def test_update_pushes_structure_with_previous_text_then_streams() -> None:
    client = FakeClient()
    state = registry().bind("om-1", "card-1")
    state.fingerprint = structure_fingerprint(make_card("", footer="生成中"))
    state.streaming_open = True
    state.streamed_text = "第一段"
    state.has_body = True

    card = make_card("第一段第二段", footer="已完成")
    assert await deliver_card_update(client, "om-1", card) is True

    names = client.names()
    assert names == ["update_card", "stream_element"]
    # the full-card push carries the text the client already shows, so the
    # following element stream is a visible growth
    assert client.calls[0][3] == "第一段"
    assert client.calls[1][3] == "第一段第二段"


@pytest.mark.asyncio
async def test_unchanged_text_and_structure_costs_no_api_call() -> None:
    client = FakeClient()
    state = registry().bind("om-1", "card-1")
    card = make_card("同一段文字")
    state.fingerprint = structure_fingerprint(card)
    state.streaming_open = True
    state.streamed_text = "同一段文字"
    state.has_body = True

    assert await deliver_card_update(client, "om-1", card) is True
    assert client.calls == []


@pytest.mark.asyncio
async def test_closed_streaming_is_reopened_and_retried() -> None:
    client = FakeClient(stream_errors=[streamed_closed(), None])
    state = registry().bind("om-1", "card-1")
    state.fingerprint = structure_fingerprint(make_card(""))
    state.streaming_open = True

    assert await deliver_card_update(client, "om-1", make_card("答案")) is True

    assert client.names() == ["stream_element", "set_streaming", "stream_element"]
    assert client.calls[1][1] is True


@pytest.mark.asyncio
async def test_other_stream_errors_degrade_to_full_card_update() -> None:
    client = FakeClient(stream_errors=[FeishuAPIError("boom", outcome="not_sent")])
    state = registry().bind("om-1", "card-1")
    state.fingerprint = structure_fingerprint(make_card(""))
    state.streaming_open = True

    assert await deliver_card_update(client, "om-1", make_card("答案")) is True

    assert client.names() == ["stream_element", "update_card"]
    assert client.calls[-1][3] == "答案"
    # never PATCH a card_id message
    assert "patch_message" not in client.names()


@pytest.mark.asyncio
async def test_long_body_never_streams_but_keeps_updating() -> None:
    client = FakeClient()
    state = registry().bind("om-1", "card-1")
    state.fingerprint = structure_fingerprint(make_card("", footer="生成中"))
    state.streaming_open = True
    huge = "y" * (MAX_STREAMABLE_CHARS + 1)

    assert await deliver_card_update(client, "om-1", make_card(huge)) is True
    assert "stream_element" not in client.names()
    assert "update_card" in client.names()


@pytest.mark.asyncio
async def test_terminal_card_closes_streaming_after_final_text() -> None:
    """A finished card must not keep the streaming cursor ("still thinking")."""

    client = FakeClient()
    state = registry().bind("om-1", "card-1")
    state.fingerprint = structure_fingerprint(make_card("", footer="已完成"))
    state.streaming_open = True
    state.streamed_text = "第一段"
    state.has_body = True

    card = make_card("第一段第二段", footer="已完成", header_template="green")
    assert await deliver_card_update(client, "om-1", card) is True

    names = client.names()
    assert names[-1] == "set_streaming"
    assert client.calls[-1][1] is False  # streaming closed
    assert state.streaming_open is False


@pytest.mark.asyncio
async def test_terminal_card_without_text_growth_still_closes_streaming() -> None:
    client = FakeClient()
    state = registry().bind("om-1", "card-1")
    card = make_card("同一段", footer="已完成", header_template="green")
    state.fingerprint = structure_fingerprint(card)
    state.streaming_open = True
    state.streamed_text = "同一段"
    state.has_body = True

    assert await deliver_card_update(client, "om-1", card) is True
    assert client.names() == ["set_streaming"]
    assert client.calls[0][1] is False


@pytest.mark.asyncio
async def test_in_progress_card_keeps_streaming_open() -> None:
    client = FakeClient()
    state = registry().bind("om-1", "card-1")
    state.fingerprint = structure_fingerprint(make_card(""))
    state.streaming_open = True

    card = make_card("生成中", header_template="blue")
    assert await deliver_card_update(client, "om-1", card) is True
    assert "set_streaming" not in client.names()
    assert state.streaming_open is True


# --- send delivery ---------------------------------------------------------


@pytest.mark.asyncio
async def test_send_skips_cardkit_when_disabled() -> None:
    client = FakeClient()
    result = await send_cardkit_delivery(
        client=client,
        chat_id="oc-1",
        card=make_card("hi"),
        card_config={"cardkit_streaming": False},
    )
    assert result is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_send_creates_entity_and_binds_message() -> None:
    client = FakeClient()
    result = await send_cardkit_delivery(
        client=client,
        chat_id="oc-1",
        card=make_card("hi"),
        card_config={"cardkit_streaming": True},
    )
    assert result is not None and result.message_id == "om-1"
    assert client.names() == ["create", "send_card_delivery", "set_streaming"]
    override = client.calls[1][2]
    assert override is not None and "card-1" in override
    state = registry().get("om-1")
    assert state is not None and state.card_id == "card-1" and state.streaming_open is True


@pytest.mark.asyncio
async def test_send_falls_back_when_entity_creation_fails() -> None:
    client = FakeClient(create_error=RuntimeError("api down"))
    result = await send_cardkit_delivery(
        client=client,
        chat_id="oc-1",
        card=make_card("hi"),
        card_config={"cardkit_streaming": True},
    )
    assert result is None
    assert client.names() == ["create"]


@pytest.mark.asyncio
async def test_send_falls_back_when_message_send_fails() -> None:
    client = FakeClient(send_error=RuntimeError("send down"))
    result = await send_cardkit_delivery(
        client=client,
        chat_id="oc-1",
        card=make_card("hi"),
        card_config={"cardkit_streaming": True},
    )
    assert result is None
    assert registry().get("om-1") is None


def test_registry_is_bounded() -> None:
    registry_ = CardKitRegistry(max_entries=2)
    registry_.bind("a", "card-a")
    registry_.bind("b", "card-b")
    registry_.bind("c", "card-c")
    assert len(registry_) == 2
    assert registry_.get("a") is None
    assert registry_.get("c") is not None
