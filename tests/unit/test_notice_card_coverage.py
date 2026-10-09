"""Card coverage for every operational notice Hermes can push to Feishu.

The classifier matches notice text by marker, and Hermes renders notices through its i18n
layer — so a marker that only knows one language silently demotes the card to a bare text
line on every other install (a zh user lost the 「自我改进」card while en users kept it).

The strings below are copied verbatim from Hermes' ``locales/<lang>.yaml``. They are embedded
rather than loaded so the suite stays hermetic and reproducible; when Hermes rewords a notice,
refresh them from the locale file. ``scripts``-side discovery lives in the HFC install skill.
"""
import pytest

from hermes_feishu_card import hook_runtime

# (notice key, language, verbatim wording, expected notice_kind)
LOCALIZED_NOTICES = [
    ("gateway.notify.no_home_channel", "en",
     "📬 No home channel is set for Feishu. A home channel is where Hermes delivers cron job "
     "results and cross-platform messages.\n\nType /sethome to make this chat your home "
     "channel, or ignore to skip.", "home-channel"),
    ("gateway.notify.no_home_channel", "zh",
     "📬 Feishu 尚未设置主频道。主频道是 Hermes 投递定时任务结果和跨平台消息的地方。\n\n"
     "输入 /sethome 将此聊天设为主频道，或忽略以跳过。", "home-channel"),

    ("gateway.notify.turn_error", "en", "Sorry, I encountered an error.", "turn-error"),
    ("gateway.notify.turn_error", "zh", "抱歉，我遇到了一个错误。", "turn-error"),
    ("gateway.notify.turn_error_detailed", "zh",
     "抱歉，我遇到了一个错误（APIError）。\nboom\n请重试，或用 /reset 开始新会话。", "turn-error"),

    ("gateway.notify.context_injection_refused", "en",
     "Context injection refused.", "context-injection"),
    ("gateway.notify.context_injection_refused", "zh",
     "上下文注入已被拒绝。", "context-injection"),

    ("gateway.notify.media_delivery_failed", "en",
     "⚠️ Couldn't deliver the file attachment (a.pdf).", "media-delivery"),
    ("gateway.notify.media_delivery_failed", "zh",
     "⚠️ 无法发送file附件（a.pdf）。", "media-delivery"),

    ("gateway.notify.delivery_failed_retry", "en",
     "⚠️ Message delivery failed after multiple attempts. Please try again — your request was "
     "processed but the response could not be sent.", "delivery-failure"),
    ("gateway.notify.delivery_failed_retry", "zh",
     "⚠️ 多次尝试后消息仍发送失败。请重试——你的请求已处理，但响应无法送达。", "delivery-failure"),

    ("gateway.notify.plain_fallback_prefix", "en",
     "(Response formatting failed, plain text:)\n\nX", "plain-fallback"),
    ("gateway.notify.plain_fallback_prefix", "zh",
     "（响应格式化失败，以下为纯文本：）\n\nX", "plain-fallback"),

    ("gateway.startup.db_corrupt", "en",
     "⚠️ Session database corruption detected. Messages may not be persisted. Recovery options:",
     "database"),
    ("gateway.startup.db_corrupt", "zh",
     "⚠️ 检测到会话数据库损坏。消息可能无法持久化。恢复方案：", "database"),
    ("gateway.startup.db_unavailable", "zh",
     "⚠️ 会话数据库不可用——消息可能无法保存，/resume 将为空。原因：c。a", "database"),

    ("gateway.compress.summary_failed", "en",
     "⚠️ Summary generation failed (e). 2 historical message(s) were removed and replaced with "
     "a placeholder; earlier context is no longer recoverable. Consider checking your "
     "auxiliary.compression model configuration.", "compression"),
    ("gateway.compress.summary_failed", "zh",
     "⚠️ 摘要生成失败（e）。2 条历史消息已被移除并替换为占位符；之前的上下文已无法恢复。"
     "建议检查 auxiliary.compression 模型配置。", "compression"),
    ("gateway.compress.aborted", "zh",
     "⚠️ 压缩已中止 (e)。未删除任何消息 — 对话保持不变。运行 /compress 重试，运行 /reset "
     "开始新会话，或检查你的 auxiliary.compression 模型配置。", "compression"),
    ("gateway.compress.aux_failed", "zh",
     "ℹ️ 配置的压缩模型 `m` 失败（e）。已使用主模型恢复 — 上下文完好 — 但您可能想检查 "
     "config.yaml 中的 `auxiliary.compression.model`。", "compression"),
    ("gateway.compress.turnhold_deferred", "zh",
     "ℹ️ 上下文压缩已推迟 — 摘要仍在生成中。本回合将不压缩继续。", "compression"),
    ("gateway.compress.turnhold_deferred", "en",
     "ℹ️ Context compression deferred — summary still streaming. Continuing without compression "
     "this turn.", "compression"),

    ("display.review.summary_callback", "en",
     "💾 Self-improvement review: Skill 'x' patched", "self-improvement"),
    ("display.review.summary_callback", "zh",
     "💾 自我改进回顾：技能 'x' 已修补", "self-improvement"),

    ("display.review.routing_fallback_warning", "en",
     "⚠ auxiliary.background_review.provider='p' (model 'm') could not be routed (e); falling "
     "back to p2/m2", "review-config"),
    ("display.review.routing_fallback_warning", "zh",
     "⚠ auxiliary.background_review.provider='p'（模型 'm'）无法路由（e）；回退到 p2/m2",
     "review-config"),
]

# Notices Hermes emits as English literals — never routed through the i18n layer, so a
# locale-keyed table alone would miss them.
LITERAL_NOTICES = [
    ("⚠ Context is over the compression threshold (~200,000 tokens >= 150,000) but compression "
     "is currently blocked (cooldown). The model may stop responding.", "compression"),
    ("⚠ Compression summary failed: boom. Inserted a fallback context marker.", "compression"),
    ("⚠ Codex app-server compaction failed: boom", "compression"),
    ("⚠ Subagent failed: 'do the thing' (APIError after 12s)", "operational"),
    ("ℹ Codex gpt-5 caps context at 258000, so auto-compaction was raised to 0.9.", "context-cap"),
    ("📬 No home channel is set for OtherPlatform. Type /sethome to set one.", "home-channel"),
]


@pytest.mark.parametrize("key,lang,text,kind", LOCALIZED_NOTICES,
                         ids=[f"{k}-{l}" for k, l, _, _ in LOCALIZED_NOTICES])
def test_localized_notice_becomes_a_card(key, lang, text, kind):
    """Both locales must classify — a marker keyed to one language drops the card in the other."""
    result = hook_runtime._hfc_classify_system_notice(text)
    assert result is not None, f"{key} [{lang}] fell through to plain text"
    assert result["notice_kind"] == kind


@pytest.mark.parametrize("text,kind", LITERAL_NOTICES,
                         ids=[t[:44] for t, _ in LITERAL_NOTICES])
def test_literal_notice_becomes_a_card(text, kind):
    result = hook_runtime._hfc_classify_system_notice(text)
    assert result is not None, f"fell through to plain text: {text[:60]}"
    assert result["notice_kind"] == kind


@pytest.mark.parametrize("text,expected_level", [
    ("⚠ Something new Hermes started emitting", "warning"),
    ("❌ Something new failed", "error"),
    ("⛔ Something new is blocking", "error"),
    ("ℹ️ A new informational line", "info"),
    ("🔄 A new rotating status line", "info"),
])
def test_unknown_glyph_notice_still_becomes_a_card(text, expected_level):
    """Coverage safety net for notices the marker table does not name yet.

    Hermes gains notices faster than any marker table can track, and an unmapped notice
    silently degrading to a bare text line beside carded ones is the failure being guarded.
    """
    result = hook_runtime._hfc_classify_system_notice(text)
    assert result is not None, text
    assert result["notice_kind"] == "operational"
    assert result["level"] == expected_level


@pytest.mark.parametrize("text", [
    # Transient per-update process state: the turn's own card already carries progress, so a
    # second card would appear and refresh continuously (explicit product decision).
    "⏳ Working — loading the model…",
    "⏳ 正在压缩上下文",
    # A translated install sends its own wording for the restart line, which is deliberately
    # unmatched so the CORE emits the already-localized plain line instead of HFC's English
    # `_HFC_GATEWAY_ONLINE_TEXT`. The English wording IS matched (see the dedicated test below).
    "♻ 网关已成功重启。你的会话继续有效。",
    "♻️ 网关已上线——Hermes 已恢复就绪。",
    # Ordinary answer text must never become a notice card.
    "这是一条普通回复正文，不该被当成系统通知。",
    "prefix ⚠ glyph not at the start is not a notice",
    "plain assistant answer with no glyph",
])
def test_deliberately_plain_lines_stay_plain(text):
    assert hook_runtime._hfc_classify_system_notice(text) is None


def test_english_restart_completion_is_classified_as_plain_text():
    """The English restart line keeps its contract: classified, but delivered as text.

    A translated install sends a different wording, which is deliberately unmatched — see
    `test_deliberately_plain_lines_stay_plain`. Either way the notice is PLAIN TEXT: the point
    is that no locale turns it into a card.
    """
    notice = hook_runtime._hfc_classify_system_notice(
        "♻ Gateway restarted successfully. Your session continues.")
    assert notice is not None
    assert notice["notice_kind"] == "gateway-restart"
    assert hook_runtime._hfc_notice_plain_text(notice) == hook_runtime._HFC_GATEWAY_ONLINE_TEXT


@pytest.mark.parametrize("text", [
    # Malformed members of the background family must fail OPEN to plain text: the card would
    # carry no parsed process/task id. Pinned upstream in test_hook_runtime.
    '🔄 Background task started: "Check the release"\nTask ID: bad-id\n'
    "You can keep chatting — results will appear when done.",
    "❌ Background task failed without an id",
    "✅ Background task",
])
def test_malformed_structured_notice_fails_open(text):
    assert hook_runtime._hfc_classify_system_notice(text) is None, text


@pytest.mark.parametrize("text,expected_kind", [
    # Wording from Hermes' locale pack for languages the hand-written table does not name.
    # Derived at runtime from `locales/*.yaml`, so these stay classified as Hermes rewords them.
    ("Désolé, j'ai rencontré une erreur.", "turn-error"),
    ("Entschuldigung, es ist ein Fehler aufgetreten.", "turn-error"),
    ("Injection de contexte refusée.", "context-injection"),
    ("Kontext-Injektion abgelehnt.", "context-injection"),
    ("💾 Revue d'auto-amélioration : x", "self-improvement"),
    ("💾 Selbstverbesserungs-Review: x", "self-improvement"),
])
def test_locale_pack_wording_is_carded(text, expected_kind):
    result = hook_runtime._hfc_classify_system_notice(text)
    assert result is not None, f"locale wording fell through: {text}"
    assert result["notice_kind"] == expected_kind


def test_locale_markers_cache_and_fail_open(monkeypatch):
    """A missing locale pack must not break classification — it only narrows coverage."""
    saved = dict(hook_runtime._HFC_LOCALE_MARKER_CACHE)
    try:
        hook_runtime._HFC_LOCALE_MARKER_CACHE.clear()
        monkeypatch.setattr(hook_runtime, "_hfc_hermes_locale_dir", lambda: None)
        assert hook_runtime._hfc_locale_notice_markers() == {}
        # Hand-written markers and the glyph fallback still classify without the pack.
        assert hook_runtime._hfc_classify_system_notice("阅读技能") is not None
        assert hook_runtime._hfc_classify_system_notice("⚠ Something new") is not None
        # Ordinary text stays ordinary.
        assert hook_runtime._hfc_classify_system_notice("just an answer") is None
    finally:
        hook_runtime._HFC_LOCALE_MARKER_CACHE.clear()
        hook_runtime._HFC_LOCALE_MARKER_CACHE.update(saved)


def test_gateway_restart_wait_notice_is_still_a_card():
    """The "restarting, not accepting work" line is actionable and keeps its card."""
    result = hook_runtime._hfc_classify_system_notice(
        "⏳ Gateway is restarting and is not accepting new work right now.")
    assert result is not None
    assert result["notice_kind"] == "gateway-restart"
    assert result["notice_terminal"] is True


@pytest.mark.parametrize("exit_code,title", [
    ("0", "后台进程已完成"),
    ("None", "后台进程已结束"),
    ("1", "后台进程失败"),
])
def test_background_process_notices_are_carded(exit_code, title):
    text = (f"[Background process proc_0123456789ab finished with exit code {exit_code}~ "
            f"Here's the final output:\nok\n]")
    result = hook_runtime._hfc_classify_system_notice(text)
    assert result is not None
    assert result["notice_kind"] == "background-process"
    assert result["title"] == title
