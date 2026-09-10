"""CardKit element-level streaming transport for Feishu card delivery.

This plugin delivers its cards as plain ``interactive`` messages and keeps them
fresh with a full-card ``PATCH``. No CardKit ``streaming_mode`` is involved, so
the Feishu client renders每个更新 as a block-wise jump instead of a typewriter
crawl.

This module adds an opt-in transport (``card.cardkit_streaming: true``) that
keeps the existing card structure and content but routes the growing answer text
through the CardKit element content API, which the Feishu client animates.

Design contract:

* The legacy ``PATCH`` path stays untouched for every message this transport
  does not own (registry miss, feature disabled, limits exceeded).
* Anything that goes wrong inside the CardKit path degrades to a full-card
  CardKit update — never to a silent drop, and never back to ``PATCH`` for a
  message that references a ``card_id`` (Feishu rejects that).
* Sequence numbers are monotonic per card and shared by every CardKit call for
  that card (create/update/settings/element content).

Verified against the live Feishu API: card entity creation, referencing it from
``/im/v1/messages``, element content streaming, and full-card updates.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .card_limits import serialize_card_for_delivery
from .feishu_client import FeishuAPIError, FeishuSendResult

logger = logging.getLogger(__name__)

#: Element id the growing answer text is streamed into. Matches the id used by
#: ``render._render_main_content_elements`` for the first content chunk.
STREAMING_ELEMENT_ID = "main_content"
_MAIN_CONTENT_PREFIX = "main_content"

#: Feishu error code returned by the element content API when the card's
#: streaming mode has been closed (segment boundary / earlier finalise).
STREAMING_CLOSED_API_CODE = 300309

#: Print tuning handed to CardKit when streaming mode is (re)opened. Mirrors the
#: values proven by the CardKit-based predecessor plugin.
STREAMING_CONFIG: Dict[str, Any] = {
    "print_frequency_ms": {"default": 15},
    "print_step": {"default": 1},
    "print_strategy": "fast",
}

#: Answers longer than this stay chunked across several elements, and element
#: streaming is skipped for that card: updates keep flowing through the
#: full-card CardKit update (correct content, no typewriter).
MAX_STREAMABLE_CHARS = 8000

_REGISTRY_MAX_ENTRIES = 512


def streaming_enabled(card_config: Optional[Mapping[str, Any]]) -> bool:
    """Return True when ``card.cardkit_streaming`` is explicitly enabled."""

    if not isinstance(card_config, Mapping):
        return False
    value = card_config.get("cardkit_streaming")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def with_streaming_config(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``card`` with CardKit streaming mode switched on.

    Existing ``config`` values win; only the streaming keys are filled in.
    """

    prepared = copy.deepcopy(dict(card))
    config = prepared.get("config")
    if not isinstance(config, dict):
        config = {}
        prepared["config"] = config
    config.setdefault("streaming_mode", True)
    config.setdefault("streaming_config", copy.deepcopy(STREAMING_CONFIG))
    return prepared


def collapse_main_content(card: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Merge chunked ``main_content*`` elements into a single element.

    Returns ``(card, text)``. ``text`` is ``None`` when the merged answer is too
    long to stream (caller keeps chunked elements and skips element streaming),
    otherwise it is the merged main-content text — possibly ``""`` when the card
    has a body element that is still empty.
    """

    prepared = copy.deepcopy(dict(card))
    body = prepared.get("body")
    elements = body.get("elements") if isinstance(body, dict) else None
    if not isinstance(elements, list):
        return prepared, None

    indexes: List[int] = []
    contents: List[str] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        element_id = element.get("element_id")
        if not isinstance(element_id, str) or not element_id.startswith(_MAIN_CONTENT_PREFIX):
            continue
        indexes.append(index)
        content = element.get("content")
        contents.append(content if isinstance(content, str) else "")

    if not indexes:
        return prepared, None

    merged = "\n\n".join(part for part in contents if part)
    if len(merged) > MAX_STREAMABLE_CHARS:
        return prepared, None

    first = indexes[0]
    merged_element = dict(elements[first])
    merged_element["element_id"] = STREAMING_ELEMENT_ID
    merged_element["content"] = merged
    rebuilt: List[Any] = []
    for index, element in enumerate(elements):
        if index == first:
            rebuilt.append(merged_element)
            continue
        if index in indexes:
            continue
        rebuilt.append(element)
    if isinstance(body, dict):
        body["elements"] = rebuilt
    return prepared, merged


def structure_fingerprint(card: Mapping[str, Any]) -> str:
    """Fingerprint of everything except the growing main-content text.

    Used to decide whether an update needs a full-card push (structure changed)
    or can be satisfied by streaming the body element alone.
    """

    stripped = copy.deepcopy(dict(card))
    body = stripped.get("body")
    elements = body.get("elements") if isinstance(body, dict) else None
    if isinstance(elements, list):
        for element in elements:
            if isinstance(element, dict):
                element_id = element.get("element_id")
                if isinstance(element_id, str) and element_id.startswith(_MAIN_CONTENT_PREFIX):
                    element["content"] = ""
    payload = json.dumps(stripped, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()


def with_body_text(card: Mapping[str, Any], text: str) -> Dict[str, Any]:
    """Return a copy of ``card`` whose main-content element carries ``text``."""

    prepared = copy.deepcopy(dict(card))
    body = prepared.get("body")
    elements = body.get("elements") if isinstance(body, dict) else None
    if isinstance(elements, list):
        for element in elements:
            if isinstance(element, dict) and element.get("element_id") == STREAMING_ELEMENT_ID:
                element["content"] = text
                break
    return prepared


@dataclass
class CardState:
    """Per-message CardKit bookkeeping."""

    card_id: str
    sequence: int = 0
    streamed_text: str = ""
    streaming_open: bool = False
    fingerprint: str = ""
    has_body: bool = False

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


class CardKitRegistry:
    """Bounded ``message_id -> CardState`` map for CardKit-backed cards."""

    def __init__(self, max_entries: int = _REGISTRY_MAX_ENTRIES) -> None:
        self._states: "OrderedDict[str, CardState]" = OrderedDict()
        self._max_entries = max(1, int(max_entries))

    def bind(self, message_id: str, card_id: str) -> CardState:
        state = CardState(card_id=card_id)
        self._states[message_id] = state
        self._states.move_to_end(message_id)
        while len(self._states) > self._max_entries:
            self._states.popitem(last=False)
        return state

    def get(self, message_id: str) -> Optional[CardState]:
        state = self._states.get(message_id)
        if state is not None:
            self._states.move_to_end(message_id)
        return state

    def forget(self, message_id: str) -> None:
        self._states.pop(message_id, None)

    def reset(self) -> None:
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)


_registry = CardKitRegistry()


def registry() -> CardKitRegistry:
    """Process-wide registry (the sidecar serves a single gateway)."""

    return _registry


def _safe_error(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    return text[:200]


def is_streaming_closed(exc: BaseException) -> bool:
    return isinstance(exc, FeishuAPIError) and exc.api_code == STREAMING_CLOSED_API_CODE


async def send_cardkit_delivery(
    *,
    client: Any,
    chat_id: str,
    card: Mapping[str, Any],
    card_config: Optional[Mapping[str, Any]],
    thread_id: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    delivery_uuid: Optional[str] = None,
    reply_in_thread: bool = False,
) -> Optional[FeishuSendResult]:
    """Create a CardKit entity and send a message that references it.

    Returns the send result, or ``None`` when the caller should use the legacy
    plain-card delivery (feature disabled, client without the CardKit methods,
    or any failure while creating the entity).
    """

    if not streaming_enabled(card_config):
        return None
    if not all(
        callable(getattr(client, name, None))
        for name in ("cardkit_create_card", "cardkit_set_streaming", "cardkit_update_card")
    ):
        return None

    try:
        # Validate limits before touching the API: a card that cannot be
        # delivered as a plain message must not be created as an entity either.
        serialize_card_for_delivery(card)
        card_id = await client.cardkit_create_card(with_streaming_config(card))
    except Exception as exc:  # noqa: BLE001 - fail-open to the legacy path
        logger.warning("CardKit entity creation failed, using plain card: %s", _safe_error(exc))
        return None

    try:
        result = await client.send_card_delivery(
            chat_id,
            card,
            thread_id=thread_id,
            reply_to_message_id=reply_to_message_id,
            delivery_uuid=delivery_uuid,
            reply_in_thread=reply_in_thread,
            content_override=json.dumps(
                {"type": "card", "data": {"card_id": card_id}},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - message not sent: fall back
        logger.warning("CardKit message send failed, using plain card: %s", _safe_error(exc))
        return None

    message_id = str(getattr(result, "message_id", "") or "")
    if not message_id:
        return None

    state = registry().bind(message_id, card_id)
    state.fingerprint = structure_fingerprint(card)
    try:
        await client.cardkit_set_streaming(
            card_id, enabled=True, sequence=state.next_sequence()
        )
        state.streaming_open = True
    except Exception as exc:  # noqa: BLE001 - animated update is best-effort
        logger.warning("CardKit streaming could not be opened: %s", _safe_error(exc))
    return result


async def deliver_card_update(client: Any, message_id: str, card: Mapping[str, Any]) -> bool:
    """Deliver a card update, preferring element-level streaming.

    Returns ``True`` when this transport owns the message (the caller must not
    fall back to ``PATCH``), ``False`` when the caller should use the legacy
    path.
    """

    state = registry().get(message_id)
    if state is None:
        return False

    try:
        await _deliver(client, state, card)
    except Exception as exc:  # noqa: BLE001 - never lose the card
        logger.warning("CardKit update failed, using full card update: %s", _safe_error(exc))
        try:
            await client.cardkit_update_card(
                state.card_id,
                _prepare(card),
                sequence=state.next_sequence(),
            )
        except Exception as inner:  # noqa: BLE001
            logger.warning("CardKit full card update failed: %s", _safe_error(inner))
    return True


def _prepare(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Collapse the body into one element and validate delivery limits."""

    prepared, _ = collapse_main_content(card)
    serialize_card_for_delivery(prepared)
    return prepared


async def _deliver(client: Any, state: CardState, card: Mapping[str, Any]) -> None:
    prepared, text = collapse_main_content(card)
    serialize_card_for_delivery(prepared)
    fingerprint = structure_fingerprint(prepared)
    structure_changed = fingerprint != state.fingerprint or not state.streaming_open

    if structure_changed:
        # Full-card push for structure, carrying the text the client already
        # shows so the following element stream is a visible growth.
        await client.cardkit_update_card(
            state.card_id,
            with_body_text(prepared, state.streamed_text if state.has_body else ""),
            sequence=state.next_sequence(),
        )
        state.fingerprint = fingerprint
        if not state.streaming_open:
            await client.cardkit_set_streaming(
                state.card_id, enabled=True, sequence=state.next_sequence()
            )
            state.streaming_open = True

    if text is None:
        # Too long to stream (or no body element): the full push above already
        # carried the content, so only remember what the client displays.
        if not structure_changed:
            await client.cardkit_update_card(
                state.card_id, prepared, sequence=state.next_sequence()
            )
        state.streamed_text = _main_text(prepared)
        state.has_body = True
    elif text == state.streamed_text and state.has_body:
        pass
    else:
        try:
            await client.cardkit_stream_element(
                state.card_id,
                STREAMING_ELEMENT_ID,
                text,
                sequence=state.next_sequence(),
            )
        except FeishuAPIError as exc:
            if not is_streaming_closed(exc):
                raise
            # Segment boundary (or an earlier finalise) closed streaming mode.
            await client.cardkit_set_streaming(
                state.card_id, enabled=True, sequence=state.next_sequence()
            )
            state.streaming_open = True
            await client.cardkit_stream_element(
                state.card_id,
                STREAMING_ELEMENT_ID,
                text,
                sequence=state.next_sequence(),
            )
        state.streamed_text = text
        state.has_body = True

    if _card_is_terminal(prepared):
        # A completed/failed card must not keep the streaming cursor (and must
        # be reproducible by the user's screenshot: text final but the card
        # still "thinking"). Close streaming mode explicitly.
        if state.streaming_open:
            await client.cardkit_set_streaming(
                state.card_id, enabled=False, sequence=state.next_sequence()
            )
            state.streaming_open = False


def _card_is_terminal(card: Mapping[str, Any]) -> bool:
    """True when the card JSON marks a completed or failed turn.

    ``render._render_status`` maps completed to a green header and failed to a
    red one, so the header template is the reliable terminal signal.
    """

    header = card.get("header") if isinstance(card, dict) else None
    if isinstance(header, dict):
        template = header.get("template")
        if template in {"green", "red"}:
            return True
    return False


def _main_text(card: Mapping[str, Any]) -> str:
    body = card.get("body")
    elements = body.get("elements") if isinstance(body, dict) else None
    if not isinstance(elements, list):
        return ""
    parts: List[str] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        element_id = element.get("element_id")
        if isinstance(element_id, str) and element_id.startswith(_MAIN_CONTENT_PREFIX):
            content = element.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n\n".join(part for part in parts if part)
