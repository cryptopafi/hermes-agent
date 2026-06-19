import asyncio
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import pytest

from gateway.livekit_realtime_agent import (
    HERMES_BRAIN_UNAVAILABLE_MESSAGE,
    HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE,
    HermesRealtimeAssistant,
    build_orchestrator_task_payload,
    build_server,
    build_modular_session,
    build_hermes_brain_payload,
    build_assistant_instructions,
    create_realtime_model,
    modular_preflight,
    guard_enabled_for_run,
    hermes_live_voice,
    _install_session_telemetry,
    _build_call_context,
    _wait_for_outbound_participant_ready,
    _outbound_initial_reply_text,
    _outbound_initial_reply_instructions,
    _first_callee_reply_instructions,
    _manual_loop_response_text,
    _manual_first_speech_timeout_seconds,
    _drain_latest_final_transcript,
    _record_session_audit_event,
    _extract_call_metadata,
    is_hermes_brain_url_allowed,
    is_hermes_orchestrator_url_allowed,
    query_hermes_brain,
    sanitize_hermes_brain_answer,
    submit_to_hermes_orchestrator,
    append_concierge_call_outcome,
    audit_restaurant_transcript,
)
from gateway.livekit_voice import (
    DEFAULT_GEMINI_REALTIME_MODEL,
    DEFAULT_HERMES_BRAIN_MODEL,
    DEFAULT_HERMES_ORCHESTRATOR_URL,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_DEEPGRAM_MODEL,
    DEFAULT_DEEPGRAM_LANGUAGE,
    DEFAULT_CARTESIA_MODEL,
    DEFAULT_OPENAI_TTS_MODEL,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_XAI_REALTIME_MODEL,
    build_dispatch_rule_payload,
    build_inbound_trunk_payload,
    build_livekit_preflight,
    build_outbound_call_metadata,
    build_outbound_call_plan,
    build_outbound_sip_participant_payload,
    build_outbound_trunk_payload,
    build_realtime_room_metadata,
    build_realtime_worker_status,
    build_room_name,
    build_room_token_output,
    execute_outbound_call_plan,
    load_livekit_config,
    require_outbound_execution_authorization,
)


def test_preflight_reports_missing_number_without_blocking_web_mvp():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is True
    assert report["ready"]["web_mvp"] is True
    assert report["ready"]["sip_phone"] is False
    assert any(issue["code"] == "missing_phone_number" for issue in report["issues"])


def test_preflight_can_require_phone_number_for_sip_gate():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env, require_phone_number=True)
    assert report["ok"] is False
    assert any(
        issue["code"] == "missing_phone_number" and issue["severity"] == "error"
        for issue in report["issues"]
    )


def test_preflight_redacts_secret_values():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "lk_API_KEY_SECRET_VALUE",
        "LIVEKIT_API_SECRET": "lk_API_SECRET_VALUE",
        "OPENAI_API_KEY": "sk-test-secret-value",
        "HERMES_LIVEKIT_AGENT_NAME": "hermes-live-voice",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    rendered = json.dumps(report, sort_keys=True)
    assert "lk_API_KEY_SECRET_VALUE" not in rendered
    assert "lk_API_SECRET_VALUE" not in rendered
    assert "sk-test-secret-value" not in rendered
    assert report["config"]["livekit_api_key"] == "set"
    assert report["config"]["livekit_api_secret"] == "set"
    assert report["config"]["openai_api_key"] == "set"
    assert report["config"]["google_api_key"] == "missing"
    assert report["config"]["xai_api_key"] == "missing"


def test_preflight_reports_missing_outbound_trunk_when_requested():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env, include_outbound=True)

    assert report["ok"] is False
    assert report["ready"]["sip_outbound"] is False
    assert any(issue["code"] == "missing_outbound_trunk" for issue in report["issues"])


def test_preflight_accepts_stored_outbound_trunk_and_redacts_inline_auth():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123",
        "HERMES_LIVEKIT_OUTBOUND_SIP_AUTH_PASSWORD": "sip-secret-password",
    }
    report = build_livekit_preflight(env, include_outbound=True)
    rendered = json.dumps(report, sort_keys=True)

    assert report["ok"] is True
    assert report["ready"]["sip_outbound"] is True
    assert report["config"]["outbound_sip_trunk_id"] == "set"
    assert report["config"]["outbound_sip_auth_password"] == "set"
    assert "sip-secret-password" not in rendered


def test_hermes_brain_config_is_loaded_and_redacted():
    env = {
        "HERMES_LIVEKIT_HERMES_URL": "http://127.0.0.1:8646/v1/chat/completions",
        "HERMES_LIVEKIT_HERMES_API_KEY": "hermes-brain-secret",
        "HERMES_LIVEKIT_HERMES_MODEL": "voice",
        "HERMES_LIVEKIT_HERMES_TIMEOUT_SECONDS": "9.5",
        "HERMES_LIVEKIT_HERMES_MAX_TOKENS": "320",
        "HERMES_LIVEKIT_HERMES_ALLOWED_HOSTS": "brain.example.com, api.example.net",
    }
    cfg = load_livekit_config(env)
    rendered = json.dumps(cfg.public_dict(), sort_keys=True)
    assert cfg.hermes_brain_url.endswith("/v1/chat/completions")
    assert cfg.hermes_brain_api_key == "hermes-brain-secret"
    assert cfg.hermes_brain_model == "voice"
    assert cfg.hermes_brain_timeout_seconds == 9.5
    assert cfg.hermes_brain_max_tokens == 320
    assert cfg.hermes_brain_allowed_hosts == (
        "brain.example.com",
        "api.example.net",
    )
    assert cfg.has_brain_credentials is True
    assert cfg.public_dict()["hermes_brain_api_key"] == "set"
    assert "hermes-brain-secret" not in rendered


def test_hermes_brain_config_defaults_are_phone_safe():
    cfg = load_livekit_config({})
    assert cfg.hermes_brain_model == DEFAULT_HERMES_BRAIN_MODEL
    assert cfg.hermes_brain_timeout_seconds <= 10
    assert cfg.hermes_brain_max_tokens <= 500
    assert cfg.hermes_brain_allow_remote is False
    assert cfg.has_brain_credentials is False
    assert cfg.hermes_orchestrator_url == DEFAULT_HERMES_ORCHESTRATOR_URL
    assert cfg.hermes_orchestrator_allow_remote is False
    assert cfg.has_orchestrator_credentials is False
    assert cfg.deepgram_language == DEFAULT_DEEPGRAM_LANGUAGE == "en-US"


def test_hermes_brain_url_allows_only_trusted_hosts_by_default():
    assert is_hermes_brain_url_allowed("http://127.0.0.1:8646/v1/chat/completions")
    assert not is_hermes_brain_url_allowed("http://10.0.0.5:8646/v1/chat/completions")
    assert not is_hermes_brain_url_allowed("https://brain.example.com/v1/chat/completions")
    assert not is_hermes_brain_url_allowed(
        "https://brain.example.com/v1/chat/completions",
        allow_remote=True,
    )
    assert is_hermes_brain_url_allowed(
        "https://brain.example.com/v1/chat/completions",
        allow_remote=True,
        allowed_hosts=("brain.example.com",),
    )
    assert not is_hermes_brain_url_allowed("http://brain.example.com/v1/chat/completions", allow_remote=True)


def test_hermes_orchestrator_config_is_loaded_and_redacted():
    env = {
        "HERMES_LIVEKIT_ORCHESTRATOR_URL": "http://127.0.0.1:8642",
        "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY": "orchestrator-secret",
        "HERMES_LIVEKIT_ORCHESTRATOR_TIMEOUT_SECONDS": "12.5",
        "HERMES_LIVEKIT_ORCHESTRATOR_ALLOWED_HOSTS": "orch.example.com",
    }
    cfg = load_livekit_config(env)
    rendered = json.dumps(cfg.public_dict(), sort_keys=True)

    assert cfg.hermes_orchestrator_url == "http://127.0.0.1:8642"
    assert cfg.hermes_orchestrator_api_key == "orchestrator-secret"
    assert cfg.hermes_orchestrator_timeout_seconds == 12.5
    assert cfg.hermes_orchestrator_allowed_hosts == ("orch.example.com",)
    assert cfg.has_orchestrator_credentials is True
    assert cfg.public_dict()["hermes_orchestrator_api_key"] == "set"
    assert "orchestrator-secret" not in rendered


def test_hermes_orchestrator_url_allows_only_trusted_hosts_by_default():
    assert is_hermes_orchestrator_url_allowed("http://127.0.0.1:8642")
    assert not is_hermes_orchestrator_url_allowed("http://10.0.0.5:8642")
    assert not is_hermes_orchestrator_url_allowed("https://orch.example.com")
    assert is_hermes_orchestrator_url_allowed(
        "https://orch.example.com",
        allow_remote=True,
        allowed_hosts=("orch.example.com",),
    )


def test_orchestrator_task_payload_routes_full_work_to_orchestrator():
    cfg = load_livekit_config({})
    payload = build_orchestrator_task_payload(
        "Use LLM-Wiki and Cortex to research the peptide protocol.",
        profile_hint="hermes-research",
        response_mode="telegram",
        config=cfg,
    )

    assert "PHONE VOICE HANDOFF FROM PAFI" in payload["input"]
    assert "LLM-Wiki" in payload["input"]
    assert "Cortex" in payload["input"]
    assert "hermes-research" in payload["input"]
    assert "profile router" in payload["instructions"]
    assert "source-backed research" in payload["instructions"]


def test_orchestrator_task_payload_rejects_empty_task():
    cfg = load_livekit_config({})
    with pytest.raises(ValueError, match="task"):
        build_orchestrator_task_payload("   ", config=cfg)


def test_hermes_brain_payload_is_concise_and_non_streaming():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_MODEL": "voice",
        "HERMES_LIVEKIT_HERMES_MAX_TOKENS": "321",
    })
    payload = build_hermes_brain_payload(
        "Explain the Hermes phone architecture in depth.",
        config=cfg,
    )
    assert payload["model"] == "voice"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 321
    assert payload["temperature"] <= 0.3
    assert "live phone call" in payload["messages"][0]["content"]
    assert "Hermes phone architecture" in payload["messages"][1]["content"]


def test_hermes_brain_payload_rejects_empty_questions():
    cfg = load_livekit_config({})
    with pytest.raises(ValueError, match="question"):
        build_hermes_brain_payload("   ", config=cfg)


def test_query_hermes_brain_returns_assistant_text():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_API_KEY": "fake-brain-key",
        "HERMES_LIVEKIT_HERMES_MODEL": "voice",
    })

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "Use the fast voice model first."}}
                ]
            }

    class FakeClient:
        def __init__(self):
            self.posted = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            self.posted = (url, headers, json)
            return FakeResponse()

    fake_client = FakeClient()
    answer = asyncio.run(
        query_hermes_brain(
            "Should this use deeper reasoning?",
            config=cfg,
            client_factory=lambda **_: fake_client,
        )
    )
    assert answer == "Use the fast voice model first."
    assert fake_client.posted[1]["Authorization"] == "Bearer fake-brain-key"


def test_sanitize_hermes_brain_answer_redacts_and_clamps():
    raw = (
        "Here is the answer. API_KEY=secret-value "
        "Bearer abcdefghijklmnopqrstuvwxyz "
        "eyJaaaaaaaaaaa.bbbbbbbbbbbb.cccccccccccc "
        "xai-abcdefghijklmnopqrstuvwxyz "
        + ("x" * 2000)
    )
    clean = sanitize_hermes_brain_answer(raw)
    assert "secret-value" not in clean
    assert "abcdefghijklmnopqrstuvwxyz" not in clean
    assert "API_KEY=[redacted]" in clean
    assert "Bearer [redacted]" in clean
    assert "[redacted-jwt]" in clean
    assert "[redacted-token]" in clean
    assert len(clean) <= 1203
    assert clean.endswith("...")


def test_query_hermes_brain_returns_safe_message_on_error(caplog):
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_API_KEY": "fake-brain-key",
    })

    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            raise RuntimeError("upstream leaked detail")

    answer = asyncio.run(
        query_hermes_brain(
            "Need deep answer.",
            config=cfg,
            client_factory=lambda **_: FailingClient(),
        )
    )
    assert answer == HERMES_BRAIN_UNAVAILABLE_MESSAGE
    assert "Hermes brain query failed: RuntimeError" in caplog.text
    assert "fake-brain-key" not in caplog.text


def test_query_hermes_brain_does_not_send_key_to_untrusted_url():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_URL": "https://brain.example.com/v1/chat/completions",
        "HERMES_LIVEKIT_HERMES_API_KEY": "fake-brain-key",
    })

    class FailingIfCalledClient:
        async def __aenter__(self):
            raise AssertionError("client must not be opened for untrusted brain URL")

    answer = asyncio.run(
        query_hermes_brain(
            "Need deep answer.",
            config=cfg,
            client_factory=lambda **_: FailingIfCalledClient(),
        )
    )
    assert answer == HERMES_BRAIN_UNAVAILABLE_MESSAGE


def test_submit_to_hermes_orchestrator_returns_run_ack():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY": "fake-orchestrator-key",
    })

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"run_id": "run_abc123", "status": "started"}

    class FakeClient:
        def __init__(self):
            self.posted = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            self.posted = (url, headers, json)
            return FakeResponse()

    fake_client = FakeClient()
    answer = asyncio.run(
        submit_to_hermes_orchestrator(
            "Research with Wiki and send the summary.",
            profile_hint="hermes-research",
            config=cfg,
            client_factory=lambda **_: fake_client,
        )
    )

    assert "Submitted the task internally as run_abc123" in answer
    assert fake_client.posted[0] == "http://127.0.0.1:8642/v1/runs"
    assert fake_client.posted[1]["Authorization"] == "Bearer fake-orchestrator-key"
    assert "Research with Wiki" in fake_client.posted[2]["input"]


def test_submit_to_hermes_orchestrator_does_not_send_key_to_untrusted_url():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_ORCHESTRATOR_URL": "https://orch.example.com",
        "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY": "fake-orchestrator-key",
    })

    class FailingIfCalledClient:
        async def __aenter__(self):
            raise AssertionError("client must not be opened for untrusted URL")

    answer = asyncio.run(
        submit_to_hermes_orchestrator(
            "Submit this.",
            config=cfg,
            client_factory=lambda **_: FailingIfCalledClient(),
        )
    )
    assert answer == HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE


def test_realtime_assistant_registers_hermes_brain_tool():
    cfg = load_livekit_config({})
    assistant = HermesRealtimeAssistant(cfg)
    tool_names = {
        getattr(getattr(tool, "_info", None), "name", None)
        for tool in assistant._tools
    }
    assert "ask_hermes_brain" in tool_names
    assert "submit_to_hermes_orchestrator" in tool_names
    assert "record_concierge_call_outcome" in tool_names


def test_concierge_outcome_tool_persists_structured_json_without_transcript(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("HERMES_CONCIERGE_LEDGER_PATH", str(ledger))
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assistant = HermesRealtimeAssistant(
        cfg,
        call_metadata={
            "call_profile": "concierge",
            "task_id": "reservation-123",
            "purpose": "Book Anima for Bogdan Rosu at 16:00",
            "restaurant_address": "Bucharest, Romania",
        },
        room_name="hermes-call-test",
    )

    answer = asyncio.run(
        assistant.record_concierge_call_outcome(
            "alternative_time_offered_owner_decision_required",
            requested_time="16:00",
            offered_times="17:30, 18:00",
            payment_requested="none_requested",
            next_action="Ask Pafi which alternative to accept",
            confirmation_name="Bogdan Roșu",
            confirmation_contact_provided="Leonardo.ai@voxsolutions.co",
            confirmation_contact_channel="email",
            notes="Venue cannot do 16:00",
        )
    )

    assert answer == "Structured Concierge call outcome recorded without transcript."
    rows = ledger.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record["task_id"] == "reservation-123"
    assert record["status"] == "alternative_time_offered_owner_decision_required"
    assert record["offered_times"] == ["17:30", "18:00"]
    assert record["confirmation_contact_provided"] == "Leonardo.ai@voxsolutions.co"
    assert record["confirmation_contact_channel"] == "email"
    assert record["transcript_persisted"] is False
    assert record["audio_persisted"] is False
    assert "Venue cannot do 16:00" in record["notes_summary"]


def test_concierge_outcome_tool_uses_approved_metadata_ledger_path(monkeypatch, tmp_path):
    fallback_ledger = tmp_path / "fallback-ledger.jsonl"
    monkeypatch.setenv("HERMES_CONCIERGE_LEDGER_PATH", str(fallback_ledger))
    approved_ledger = Path("/home/pafi/hermes-agent/.tmp-test-ledger/approved-ledger.jsonl")
    if approved_ledger.exists():
        approved_ledger.unlink()
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assistant = HermesRealtimeAssistant(
        cfg,
        call_metadata={
            "call_profile": "concierge",
            "task_id": "reservation-123",
            "purpose": "Book Anima for Bogdan Rosu at 16:00",
            "ledger_path": str(approved_ledger),
        },
        room_name="hermes-call-test",
    )

    try:
        answer = asyncio.run(assistant.record_concierge_call_outcome("confirmed"))

        assert answer == "Structured Concierge call outcome recorded without transcript."
        assert approved_ledger.exists()
        assert not fallback_ledger.exists()
        record = json.loads(approved_ledger.read_text(encoding="utf-8").splitlines()[-1])
        assert record["task_id"] == "reservation-123"
        assert record["status"] == "confirmed"
    finally:
        if approved_ledger.exists():
            approved_ledger.unlink()
        try:
            approved_ledger.parent.rmdir()
        except OSError:
            pass


def test_concierge_outcome_tool_refuses_non_concierge_calls(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("HERMES_CONCIERGE_LEDGER_PATH", str(ledger))
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assistant = HermesRealtimeAssistant(cfg, call_metadata={"call_profile": "pa"})

    answer = asyncio.run(assistant.record_concierge_call_outcome("confirmed"))

    assert answer == "Outcome not recorded: this is not a Concierge call."
    assert not ledger.exists()


def test_append_concierge_call_outcome_creates_parent_directory(tmp_path):
    ledger = tmp_path / "nested" / "ledger.jsonl"

    append_concierge_call_outcome({"status": "confirmed"}, ledger_path=ledger)

    assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "confirmed"


def test_session_telemetry_logs_redacted_events(caplog):
    caplog.set_level(logging.INFO, logger="gateway.livekit_realtime_agent")

    class FakeSession:
        def __init__(self):
            self.callbacks = {}

        def on(self, event_name, callback):
            self.callbacks[event_name] = callback

    session = FakeSession()
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    _install_session_telemetry(
        session,
        config=cfg,
        room_name="room one",
        started_at=time.monotonic(),
    )

    session.callbacks["user_input_transcribed"](
        types.SimpleNamespace(transcript="secret user words", is_final=True)
    )
    session.callbacks["agent_state_changed"](
        types.SimpleNamespace(old_state="thinking", new_state="speaking")
    )
    session.callbacks["close"](
        types.SimpleNamespace(reason=types.SimpleNamespace(value="done"), error=None)
    )

    assert "hermes_call event=transcript" in caplog.text
    assert "chars=17" in caplog.text
    assert "secret user words" not in caplog.text
    assert "hermes_call event=close" in caplog.text


def test_brain_tool_logs_start_and_done(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gateway.livekit_realtime_agent")
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assistant = HermesRealtimeAssistant(cfg)

    async def fake_query(question, *, config):
        return "answer"

    monkeypatch.setattr(
        "gateway.livekit_realtime_agent.query_hermes_brain",
        fake_query,
    )

    answer = asyncio.run(assistant.ask_hermes_brain("question"))

    assert answer == "answer"
    assert "hermes_call event=brain_tool_start" in caplog.text
    assert "hermes_call event=brain_tool_done" in caplog.text


def test_orchestrator_tool_logs_start_and_done(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gateway.livekit_realtime_agent")
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assistant = HermesRealtimeAssistant(cfg)

    async def fake_submit(task, *, profile_hint, response_mode, config):
        return "submitted"

    monkeypatch.setattr(
        "gateway.livekit_realtime_agent.submit_to_hermes_orchestrator",
        fake_submit,
    )

    answer = asyncio.run(
        assistant.submit_to_hermes_orchestrator(
            "task",
            profile_hint="hermes-research",
        )
    )

    assert answer == "submitted"
    assert "hermes_call event=orchestrator_submit_start" in caplog.text
    assert "hermes_call event=orchestrator_submit_done" in caplog.text


def test_hermes_live_voice_logs_job_and_session(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gateway.livekit_realtime_agent")

    class FakeSession:
        def __init__(self, *, llm, **kwargs):
            self.llm = llm
            self.kwargs = kwargs
            self.callbacks = {}

        def on(self, event_name, callback):
            self.callbacks[event_name] = callback

        async def start(self, *, room, agent, **kwargs):
            self.room = room
            self.agent = agent
            self.start_kwargs = kwargs

    monkeypatch.setattr("gateway.livekit_realtime_agent.AgentSession", FakeSession)
    monkeypatch.setattr(
        "gateway.livekit_realtime_agent.create_realtime_model",
        lambda cfg, **_: object(),
    )
    monkeypatch.setenv("HERMES_LIVEKIT_PIPELINE_MODE", "realtime")
    monkeypatch.setenv("HERMES_LIVEKIT_REALTIME_PROVIDER", "xai")
    ctx = types.SimpleNamespace(room=types.SimpleNamespace(name="bench-room"))

    asyncio.run(hermes_live_voice(ctx))

    assert "hermes_call event=job_start" in caplog.text
    assert "room=bench-room" in caplog.text
    assert "hermes_call event=session_started" in caplog.text


def test_outbound_call_context_from_dispatch_metadata():
    metadata = {
        "mode": "sip-outbound",
        "call_profile": "pa",
        "purpose": "Confirm the outbound call works.",
        "restaurant_address": "Calle de Serrano 1, Madrid, Spain",
        "max_duration_seconds": "900",
    }

    context = _build_call_context(metadata)

    assert "outbound phone call initiated by Leonardo" in context
    assert "you called them" in context
    assert "Active outbound profile: pa" in context
    assert "Confirm the outbound call works" in context
    assert "Calle de Serrano 1, Madrid, Spain" in context
    assert "Start by briefly identifying" in context
    assert "Use standard English first" not in context
    assert "record_concierge_call_outcome" not in context


def test_outbound_concierge_context_requires_structured_outcome_tool():
    context = _build_call_context({
        "mode": "sip-outbound",
        "call_profile": "concierge",
        "purpose": "Book a restaurant.",
    })

    assert "do not store raw transcript" in context
    assert "record_concierge_call_outcome" in context
    assert "alternative_time_offered_owner_decision_required" in context
    assert "do not accept an alternative automatically" in context
    assert "owner_followup_required" in context


def test_outbound_concierge_context_offers_confirmation_contact_from_metadata():
    context = _build_call_context({
        "mode": "sip-outbound",
        "call_profile": "concierge",
        "purpose": "Book a restaurant.",
        "confirmation_phone": "+12025550199",
        "confirmation_email": "pafi@example.com",
    })

    assert "proactively provide Leonardo/Concierge's dedicated operational confirmation contact details" in context
    assert "dedicated confirmation phone:" in context
    assert "email: pafi@example.com" in context
    assert "not the guest's personal contacts" in context
    assert "Never provide the owner's personal phone number or personal email" in context
    assert "Record which confirmation contact was provided" in context
    assert "Pafi" not in context


def test_outbound_initial_reply_instructions_force_first_speech_for_gemini():
    metadata = {
        "mode": "sip-outbound",
        "call_profile": "concierge",
        "restaurant_address": "Restaurant Anima, Romania",
        "purpose": "Book Restaurant Anima",
    }
    text = _outbound_initial_reply_text(metadata)
    instructions = _outbound_initial_reply_instructions(metadata)

    assert text == "Hello, this is Leonardo."
    assert "deterministic and already scheduled" in instructions
    assert "Hello, this is Leonardo" in instructions
    assert "Pafi" not in instructions
    assert "Call goal for later turns" in instructions
    assert "Book Restaurant Anima" in instructions



def test_manual_restaurant_confirmation_reply_does_not_proactively_send_contact_details():
    text, signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="yes, all good",
        turn_index=2,
        confirmation_phone="+100****0001",
        confirmation_email="reservations@leonardoboutique.ro",
    )

    assert signal == "reservation_confirmed_acknowledged"
    assert text == "Perfect, thank you. Please keep the reservation for two people today at 3 PM."
    assert "+100" not in text
    assert "reservations@" not in text


def test_manual_restaurant_answers_contact_only_when_asked():
    email_text, email_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Can you repeat the email address?",
        turn_index=6,
        confirmation_email="reservations@leonardoboutique.ro",
    )
    phone_text, phone_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Can you repeat the phone number?",
        turn_index=6,
        confirmation_phone="+100****0001",
    )

    assert email_signal == "confirmation_email_answered"
    assert email_text == "The email is reservations@leonardoboutique.ro."
    assert phone_signal == "confirmation_phone_answered"
    assert phone_text == "The phone number is +100****0001."


def test_manual_restaurant_handles_no_availability_and_alternative_time_without_confirming():
    no_avail_text, no_avail_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Don't have availability at two. Three PM?",
        turn_index=2,
    )
    only_text, only_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Only one PM is possible.",
        turn_index=3,
    )

    assert no_avail_signal == "alternative_requested"
    assert no_avail_text == "I understand. Is there any time close to 3 PM available today for two people?"
    assert only_signal == "alternative_closer_time_requested"
    assert only_text == "I understand. Is there anything closer to 3 PM available for two people today?"
    assert "Please keep the reservation" not in only_text

    no_closer_text, no_closer_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="No closer, only at one PM.",
        turn_index=4,
    )
    assert no_closer_signal == "alternative_requires_owner_approval"
    assert no_closer_text == "Understood. I cannot change the time without approval first, so I will check and follow up. Thank you."
    assert "Please keep the reservation" not in no_closer_text


def test_manual_restaurant_asks_repeat_on_unclear_audio_and_closes_on_thanks():
    unclear_text, unclear_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="A tu pie.",
        turn_index=3,
    )
    thanks_text, thanks_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="K. Bye.",
        turn_index=5,
    )
    waiting_thanks_text, waiting_thanks_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Thank you.",
        turn_index=3,
        previous_signals=["restaurant_waiting"],
    )

    assert unclear_signal == "clarification_requested"
    assert unclear_text == "Sorry, could you repeat that please?"
    assert thanks_signal == "polite_close_after_confirmation"
    assert thanks_text == "Thank you. Have a good day."
    assert waiting_thanks_signal == "availability_check_prompted"
    assert waiting_thanks_text == "Were you able to check availability for two people at 3 PM today?"


def test_manual_restaurant_spells_name_when_asked_and_closes_on_final_okay():
    spell_text, spell_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Can you spell the name?",
        turn_index=3,
    )
    close_text, close_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Okay.",
        turn_index=4,
    )

    assert spell_signal == "name_spelled"
    assert spell_text == "Of course: Bogdan Rosu. B as in Bravo, O, G, D, A, N. Last name Rosu: R, O, S, U."
    assert close_signal == "polite_close_after_confirmation"
    assert close_text == "Perfect, thank you. Have a good day."
    assert "Please keep the reservation" not in close_text


def test_manual_restaurant_answers_basic_venue_questions():
    people_text, people_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="How many persons?",
        turn_index=2,
    )
    time_text, time_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="What time?",
        turn_index=2,
    )
    name_text, name_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Under what name?",
        turn_index=2,
    )

    assert people_text == "Two people, please."
    assert people_signal == "party_size_answered"
    assert time_text == "Today at 3 PM, please."
    assert time_signal == "time_answered"
    assert name_text == "Under the name Bogdan Rosu, please."
    assert name_signal == "name_answered"


def test_manual_restaurant_wait_reply_does_not_repeat_booking_request():
    text, signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="one second, let me check",
        turn_index=1,
    )

    assert signal == "restaurant_waiting"
    assert text == "Of course, I will wait."

    text, signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="I don't know. I need to check. Wait.",
        turn_index=2,
    )

    assert signal == "restaurant_waiting"
    assert text == "Of course, I will wait."


def test_manual_restaurant_first_request_is_short():
    text, signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="hello",
        turn_index=0,
    )

    assert signal == "reservation_requested"
    assert text == "Hello, this is Leonardo. I'd like to arrange a table for two people today at 3 PM, please. Would that be possible?"
    assert len(text) < 120


def test_restaurant_transcript_audit_allows_clean_fast_intro():
    result = audit_restaurant_transcript(
        [
            {"speaker": "assistant", "text": "Hello, this is Leonardo. I'd like to arrange a table for two people today at 3 PM, please. Would that be possible?"},
            {"speaker": "callee", "text": "No. Only at two PM."},
            {"speaker": "assistant", "text": "I understand. Is there anything closer to 3 PM available for two people today?"},
            {"speaker": "callee", "text": "No. Only two, we have nothing."},
            {"speaker": "assistant", "text": "Understood. I cannot change the time without approval first, so I will check and follow up. Thank you."},
        ]
    )

    assert result["passed"] is True
    assert result["checks"]["venue_first"] is True


def test_manual_restaurant_alternative_followup_closes_after_no_closer():
    first_text, first_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Only at one PM.",
        turn_index=2,
        previous_signals=["alternative_requested"],
    )
    close_text, close_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="No. He's not available. Only at one o'clock.",
        turn_index=3,
        previous_signals=["alternative_requested", "alternative_closer_time_requested"],
    )

    assert first_signal == "alternative_closer_time_requested"
    assert first_text == "I understand. Is there anything closer to 3 PM available for two people today?"
    assert close_signal == "alternative_requires_owner_approval"
    assert close_text == "Understood. I cannot change the time without approval first, so I will check and follow up. Thank you."


def test_manual_loop_drains_to_latest_final_transcript_before_reply():
    queue = asyncio.Queue()
    queue.put_nowait("Yes.")
    queue.put_nowait("You spell the name, please?")

    assert _drain_latest_final_transcript(queue, "Yeah. We have.") == "You spell the name, please?"


def test_restaurant_transcript_audit_flags_delayed_spell_and_consecutive_assistant_turns():
    result = audit_restaurant_transcript(
        [
            {"speaker": "callee", "text": "Hello?"},
            {"speaker": "assistant", "text": "Hello, this is Leonardo. I'd like to arrange a table for two people today at 3 PM, please. Would that be possible?"},
            {"speaker": "callee", "text": "You spell the name, please?"},
            {"speaker": "assistant", "text": "Perfect, thank you. Please keep the reservation for two people today at 3 PM."},
            {"speaker": "assistant", "text": "Of course: Bogdan Rosu. B as in Bravo, O, G, D, A, N. Last name Rosu: R, O, S, U."},
        ]
    )

    assert result["passed"] is False
    assert "answered_venue_questions" in result["failure_reasons"]
    assert "no_consecutive_assistant_turns" in result["failure_reasons"]


def test_manual_restaurant_clarifies_address_before_email():
    address_text, address_signal = _manual_loop_response_text(
        purpose_text="restaurant-reservation-test",
        user_text="Address?",
        turn_index=4,
        confirmation_email="reservations@leonardoboutique.ro",
    )
    assert address_signal == "confirmation_address_clarified"
    assert address_text == "Do you mean the email address for confirmation?"
    assert "reservations@" not in address_text


def test_first_callee_reply_instructions_respect_non_booking_test_purpose():
    instructions = _first_callee_reply_instructions(
        {
            "call_profile": "concierge",
            "purpose": "English full-audit test call. No booking or payment request.",
        },
        "What reservation?",
    )

    assert "non-booking test call" in instructions
    assert "do not ask for a reservation" in instructions
    assert "Get the booking outcome" not in instructions


def test_record_session_audit_event_records_exact_assistant_text():
    calls = []

    class FakeSession:
        pass

    session = FakeSession()

    def recorder(**kwargs):
        calls.append(kwargs)

    setattr(session, "_hermes_record_audit_transcript_event", recorder)

    _record_session_audit_event(
        session,
        speaker="assistant",
        text="Hello, this is Leonardo.",
        source="outbound_initial_say",
        turn=0,
    )

    assert calls == [
        {
            "speaker": "assistant",
            "text": "Hello, this is Leonardo.",
            "source": "outbound_initial_say",
            "turn": 0,
        }
    ]


def test_restaurant_transcript_audit_passes_natural_flow():
    result = audit_restaurant_transcript(
        [
            {"speaker": "callee", "text": "Calla Blanco, hello."},
            {"speaker": "assistant", "text": "Hello, this is Leonardo. I'd like to arrange a table for two people today at 3 PM, please. Would that be possible?"},
            {"speaker": "callee", "text": "Yes, available. Under what name?"},
            {"speaker": "assistant", "text": "Under the name Bogdan Rosu, please."},
            {"speaker": "callee", "text": "Ok, confirmed."},
            {"speaker": "assistant", "text": "Thank you. Have a good day."},
        ]
    )

    assert result["passed"] is True
    assert result["human_naturalness"] == 5
    assert result["failure_reasons"] == []


def test_restaurant_transcript_audit_flags_fake_or_invented_flow():
    result = audit_restaurant_transcript(
        [
            {"speaker": "assistant", "text": "Hello, this is a brief test of your restaurant's communication system."},
            {"speaker": "callee", "text": "How many persons?"},
            {"speaker": "assistant", "text": "Perfect, thank you. Please keep the reservation for two people today at 3 PM."},
        ]
    )

    assert result["passed"] is False
    assert "venue_first" in result["failure_reasons"]
    assert "asked_exact_booking" in result["failure_reasons"]
    assert "answered_venue_questions" in result["failure_reasons"]
    assert "did_not_invent_confirmation" in result["failure_reasons"]
    assert result["human_naturalness"] < 5


def test_restaurant_transcript_audit_flags_close_before_late_venue_answer():
    result = audit_restaurant_transcript(
        [
            {"speaker": "callee", "text": "Hello?"},
            {"speaker": "assistant", "text": "Hello, this is Leonardo. I'd like to arrange a table for two people today at 3 PM, please. Would that be possible?"},
            {"speaker": "callee", "text": "I don't know. Let me check."},
            {"speaker": "assistant", "text": "Of course, I will wait."},
            {"speaker": "callee", "text": "Thank you."},
            {"speaker": "assistant", "text": "Thank you. Have a good day."},
            {"speaker": "callee", "text": "So, no, unfortunately, it's impossible."},
        ]
    )

    assert result["passed"] is False
    assert "closed_cleanly" in result["failure_reasons"]


def test_restaurant_transcript_audit_allows_owner_followup_close_acknowledgment():
    result = audit_restaurant_transcript(
        [
            {"speaker": "callee", "text": "Hello?"},
            {"speaker": "assistant", "text": "Hello, this is Leonardo. I'd like to arrange a table for two people today at 3 PM, please. Would that be possible?"},
            {"speaker": "callee", "text": "No. Only at two PM."},
            {"speaker": "assistant", "text": "I understand. Is there anything closer to 3 PM available for two people today?"},
            {"speaker": "callee", "text": "No. No. Only two, we have nothing."},
            {"speaker": "assistant", "text": "Understood. I cannot change the time without approval first, so I will check and follow up. Thank you."},
            {"speaker": "callee", "text": "Okay. Thank you. Call me back."},
        ]
    )

    assert result["passed"] is True
    assert result["failure_reasons"] == []


def test_wait_for_outbound_participant_ready_waits_for_sip_media(monkeypatch):
    calls = []

    class FakePublication:
        source = "SOURCE_MICROPHONE"
        kind = "KIND_AUDIO"
        subscribed = True
        track = object()

    class FakeParticipant:
        identity = "sip-concierge-test"
        track_publications = {"track-1": FakePublication()}

    class FakeCtx:
        async def wait_for_participant(self, *, identity=None):
            calls.append(identity)
            return FakeParticipant()

    async def fake_sleep(seconds):
        calls.append(f"sleep:{seconds}")

    monkeypatch.setattr("gateway.livekit_realtime_agent.asyncio.sleep", fake_sleep)

    identity = asyncio.run(
        _wait_for_outbound_participant_ready(
            FakeCtx(),
            {"mode": "sip-outbound", "participant_identity": "sip-concierge-test"},
            room_name="room-test",
            settle_seconds=1.2,
        )
    )

    assert identity == "sip-concierge-test"
    assert calls == ["sip-concierge-test", "sleep:1.2"]


def test_outbound_call_metadata_accepts_confirmation_contact_fields():
    metadata = build_outbound_call_metadata(
        profile="concierge",
        purpose="Book a restaurant",
        confirmation_phone="+12025550111",
        confirmation_email="pafi@example.com",
    )

    assert metadata["confirmation_phone"] == "+12025550111"
    assert metadata["confirmation_email"] == "pafi@example.com"
    assert metadata["confirmation_contact_owner"] == "leonardo_concierge"


def test_outbound_call_metadata_uses_dedicated_concierge_contact_from_config():
    cfg = load_livekit_config({
        "HERMES_CONCIERGE_CONFIRMATION_PHONE": "+12025550198",
        "HERMES_CONCIERGE_CONFIRMATION_EMAIL": "leonardo@example.com",
    })

    metadata = build_outbound_call_metadata(
        profile="concierge",
        purpose="Book a restaurant",
        config=cfg,
    )

    assert metadata["confirmation_phone"] == "+12025550198"
    assert metadata["confirmation_email"] == "leonardo@example.com"
    assert metadata["confirmation_contact_owner"] == "leonardo_concierge"


def test_outbound_call_metadata_does_not_use_livekit_phone_as_concierge_phone_fallback():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_PHONE_NUMBER": "+100000000001",
        "HERMES_CONCIERGE_CONFIRMATION_EMAIL": "leonardo@example.com",
    })

    metadata = build_outbound_call_metadata(
        profile="concierge",
        purpose="Book a restaurant",
        config=cfg,
    )

    assert "confirmation_phone" not in metadata
    assert metadata["confirmation_email"] == "leonardo@example.com"
    assert metadata["confirmation_contact_owner"] == "leonardo_concierge"


def test_livekit_public_dict_redacts_dedicated_concierge_contact_values():
    cfg = load_livekit_config({
        "HERMES_CONCIERGE_CONFIRMATION_PHONE": "+12025550198",
        "HERMES_CONCIERGE_CONFIRMATION_EMAIL": "leonardo@example.com",
    })

    public = cfg.public_dict()
    rendered = json.dumps(public, sort_keys=True)

    assert public["concierge_confirmation_phone"] == "set"
    assert public["concierge_confirmation_email"] == "set"
    assert "+12025550198" not in rendered
    assert "leonardo@example.com" not in rendered


def test_outbound_call_metadata_rejects_invalid_confirmation_contact_fields():
    with pytest.raises(ValueError, match="confirmation_phone"):
        build_outbound_call_metadata(
            profile="concierge",
            purpose="Book a restaurant",
            confirmation_phone="0700000000",
        )
    with pytest.raises(ValueError, match="confirmation_email"):
        build_outbound_call_metadata(
            profile="concierge",
            purpose="Book a restaurant",
            confirmation_email="not-an-email",
        )


def test_outbound_call_metadata_rejects_protected_extra_overrides():
    with pytest.raises(ValueError, match="protected outbound keys"):
        build_outbound_call_metadata(
            profile="concierge",
            purpose="Book a restaurant",
            task_id="reservation-123",
            idempotency_key="reservation-123-call-001",
            ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
            extra={"task_id": "tampered-task"},
        )

    metadata = build_outbound_call_metadata(
        profile="concierge",
        purpose="Book a restaurant",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
        extra={"restaurant_name": "Calla Blanco"},
    )
    assert metadata["task_id"] == "reservation-123"
    assert metadata["restaurant_name"] == "Calla Blanco"


def test_outbound_call_metadata_restricts_ledger_path_roots():
    with pytest.raises(ValueError, match="approved Concierge call ledger directory"):
        build_outbound_call_metadata(
            profile="concierge",
            purpose="Book a restaurant",
            task_id="reservation-123",
            idempotency_key="reservation-123-call-001",
            ledger_path="/home/pafi/.hermes/random-ledger.jsonl",
        )

    metadata = build_outbound_call_metadata(
        profile="concierge",
        purpose="Book a restaurant",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
    )
    assert metadata["ledger_path"] == "/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl"


def test_extract_call_metadata_reads_job_and_participant_metadata():
    ctx = types.SimpleNamespace(
        room=types.SimpleNamespace(
            name="room",
            metadata="{}",
            remote_participants={
                "sip": types.SimpleNamespace(
                    metadata=json.dumps({"purpose": "participant purpose"}),
                    attributes={"hermes.profile": "concierge"},
                )
            },
        ),
        _info=types.SimpleNamespace(
            accept_arguments=types.SimpleNamespace(
                metadata=json.dumps({
                    "mode": "sip-outbound",
                    "purpose": "dispatch purpose",
                    "call_profile": "pa",
                })
            )
        ),
    )

    metadata = _extract_call_metadata(ctx)

    assert metadata["mode"] == "sip-outbound"
    assert metadata["purpose"] == "participant purpose"
    assert metadata["call_profile"] == "pa"


def test_hermes_live_voice_injects_outbound_context(monkeypatch):
    captured = {}

    class FakeSession:
        def __init__(self, *, llm, **kwargs):
            self.llm = llm
            self.kwargs = kwargs
            captured["session_kwargs"] = kwargs
            self.callbacks = {}

        def on(self, event_name, callback):
            self.callbacks[event_name] = callback

        async def start(self, *, room, agent, **kwargs):
            captured["agent_instructions"] = agent._instructions
            captured["start_kwargs"] = kwargs

        async def generate_reply(self, *, instructions):
            captured["initial_reply_instructions"] = instructions

        def say(self, text, *, allow_interruptions=None, add_to_chat_ctx=True):
            captured["initial_say_text"] = text
            captured["initial_say_allow_interruptions"] = allow_interruptions
            captured["initial_say_add_to_chat_ctx"] = add_to_chat_ctx

            class FakeSpeech:
                async def wait_for_playout(self):
                    captured["initial_say_playout_waited"] = True

            return FakeSpeech()

    class FakeModel:
        def __init__(self, instructions):
            self.instructions = instructions

    def fake_create_model(cfg, *, call_context=""):
        captured["model_call_context"] = call_context
        return FakeModel(call_context)

    monkeypatch.setattr("gateway.livekit_realtime_agent.AgentSession", FakeSession)
    monkeypatch.setattr(
        "gateway.livekit_realtime_agent.create_realtime_model",
        fake_create_model,
    )
    monkeypatch.setenv("HERMES_LIVEKIT_PIPELINE_MODE", "realtime")
    monkeypatch.setenv("HERMES_LIVEKIT_REALTIME_PROVIDER", "gemini")
    async def fake_sleep(_seconds):
        captured["media_settle_waited"] = True

    monkeypatch.setattr("gateway.livekit_realtime_agent.asyncio.sleep", fake_sleep)
    class ReadyPublication:
        source = "SOURCE_MICROPHONE"
        kind = "KIND_AUDIO"
        subscribed = True
        track = object()

    class ReadyParticipant:
        identity = "sip-test"
        track_publications = {"track-1": ReadyPublication()}

    ctx = types.SimpleNamespace(
        room=types.SimpleNamespace(name="outbound-room", remote_participants={}),
        wait_for_participant=lambda **_: ReadyParticipant(),
        _info=types.SimpleNamespace(
            accept_arguments=types.SimpleNamespace(
                metadata=json.dumps({
                    "mode": "sip-outbound",
                    "call_profile": "pa",
                    "purpose": "Greet Pafi and confirm this was an outbound call.",
                })
            )
        ),
    )

    asyncio.run(hermes_live_voice(ctx))

    assert "outbound phone call initiated by Leonardo" in captured["model_call_context"]
    assert "you called them" in captured["agent_instructions"]
    assert captured["initial_say_text"] == "Hello, this is Leonardo."
    assert captured["initial_say_allow_interruptions"] is True
    assert captured["initial_say_add_to_chat_ctx"] is True
    assert captured["initial_say_playout_waited"] is True
    assert "initial_reply_instructions" not in captured
    assert captured["session_kwargs"]["min_endpointing_delay"] == 0.9
    assert captured["session_kwargs"]["min_interruption_words"] == 2
    assert captured["media_settle_waited"] is True
    assert "allow_interruptions" not in captured["session_kwargs"]


def test_realtime_preflight_reports_missing_gemini_key_by_default():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert report["ready"]["realtime_agent"] is False
    assert report["config"]["pipeline_mode"] == "realtime"
    assert report["config"]["realtime_provider"] == "gemini"
    assert any(issue["code"] == "missing_google_api_key" for issue in report["issues"])


def test_livekit_preflight_rejects_remote_ws_url():
    env = {
        "LIVEKIT_URL": "ws://livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is False
    assert any(issue["code"] == "invalid_livekit_url" for issue in report["issues"])


def test_livekit_preflight_allows_loopback_ws_url():
    env = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is True
    assert not any(issue["code"] == "invalid_livekit_url" for issue in report["issues"])


def test_livekit_preflight_rejects_invalid_agent_name():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_AGENT_NAME": "../bad agent",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is False
    assert any(issue["code"] == "invalid_agent_name" for issue in report["issues"])


def test_gemini_realtime_preflight_accepts_gemini_api_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini",
        "GEMINI_API_KEY": "gemini-secret-value",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    rendered = json.dumps(report, sort_keys=True)
    assert report["ok"] is True
    assert report["ready"]["realtime_agent"] is True
    assert report["config"]["google_api_key"] == "set"
    assert "gemini-secret-value" not in rendered
    assert report["worker"]["model"] == DEFAULT_GEMINI_REALTIME_MODEL


def test_gemini_realtime_preflight_reports_missing_google_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert report["ready"]["realtime_agent"] is False
    assert any(issue["code"] == "missing_google_api_key" for issue in report["issues"])


def test_xai_realtime_preflight_accepts_xai_api_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "xai",
        "XAI_API_KEY": "xai-secret-value",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    rendered = json.dumps(report, sort_keys=True)
    assert report["ok"] is True
    assert report["ready"]["realtime_agent"] is True
    assert report["config"]["xai_api_key"] == "set"
    assert "xai-secret-value" not in rendered
    assert report["worker"]["model"] == DEFAULT_XAI_REALTIME_MODEL
    assert report["worker"]["voice"] == "ara"


def test_xai_realtime_preflight_reports_missing_xai_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "xai",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert report["ready"]["realtime_agent"] is False
    assert any(issue["code"] == "missing_xai_api_key" for issue in report["issues"])


def test_realtime_config_defaults_and_status_are_operator_safe():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "GEMINI_API_KEY": "gemini-secret-value",
        "HERMES_LIVEKIT_REALTIME_ENABLED": "true",
    }
    cfg = load_livekit_config(env)
    status = build_realtime_worker_status(config=cfg)
    assert cfg.pipeline_mode == "realtime"
    assert cfg.realtime_provider == "gemini"
    assert cfg.realtime_model == DEFAULT_GEMINI_REALTIME_MODEL
    assert cfg.realtime_voice == "Puck"
    assert status["enabled"] is True
    assert status["mode"] == "manual"
    assert "gateway.livekit_realtime_agent" in status["run"]


def test_realtime_room_metadata_is_stable_for_webrtc_dispatch():
    cfg = load_livekit_config({})
    assert build_realtime_room_metadata(mode="webrtc", config=cfg) == {
        "mode": "webrtc",
        "route": "hermes-main",
        "voice_version": "v02",
        "pipeline_mode": "realtime",
        "realtime_provider": "gemini",
        "stt_provider": "none",
        "tts_provider": "none",
    }


def test_realtime_room_metadata_uses_configured_provider():
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini"})
    assert build_realtime_room_metadata(mode="sip", config=cfg)["realtime_provider"] == "gemini"

    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assert build_realtime_room_metadata(mode="sip", config=cfg)["realtime_provider"] == "xai"


def test_assistant_instructions_are_short_and_english_for_concierge():
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_INSTRUCTIONS": "Be concise."})
    instructions = build_assistant_instructions(cfg)
    assert "Be concise." in instructions
    assert "speak US English unless the callee explicitly asks" in instructions
    assert "do not mix languages unless the callee switches first" in instructions
    assert "use the reservation name only if the venue asks or needs it" in instructions
    assert "Speak slowly and clearly" in instructions
    assert "written confirmation by email or WhatsApp using the dedicated Concierge contact details" in instructions
    assert "unverified messaging channel" in instructions


def test_assistant_instructions_include_outbound_call_context():
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_INSTRUCTIONS": "Be concise."})
    instructions = build_assistant_instructions(
        cfg,
        call_context="Call context: this is an outbound phone call initiated by Leonardo.",
    )

    assert "Be concise." in instructions
    assert "outbound phone call initiated by Leonardo" in instructions


def test_realtime_worker_start_guard_requires_explicit_enable():
    cfg = load_livekit_config({})

    with pytest.raises(SystemExit, match="disabled"):
        guard_enabled_for_run(["dev"], cfg)

    guard_enabled_for_run(["--help"], cfg)
    enabled_cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_ENABLED": "true"})
    guard_enabled_for_run(["dev"], enabled_cfg)


def test_create_realtime_model_rejects_unknown_provider():
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "bogus"})
    with pytest.raises(RuntimeError, match="openai.*gemini.*xai"):
        create_realtime_model(cfg)


def test_openai_realtime_model_uses_config_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeRealtimeModel:
        def __init__(self, *, model, voice):
            self.model = model
            self.voice = voice

    fake_openai = types.SimpleNamespace(
        realtime=types.SimpleNamespace(RealtimeModel=FakeRealtimeModel)
    )
    fake_plugins = types.SimpleNamespace(openai=fake_openai)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.openai", fake_openai)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "openai",
        "OPENAI_API_KEY": "cfg-openai-key",
    })
    model = create_realtime_model(cfg)

    assert os.environ["OPENAI_API_KEY"] == "cfg-openai-key"
    assert model.model == DEFAULT_REALTIME_MODEL
    assert model.voice == "coral"


def test_gemini_realtime_model_uses_config_key_and_instructions(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    class FakeRealtimeModel:
        def __init__(self, *, model, voice, instructions, temperature=None, max_output_tokens=None):
            self.model = model
            self.voice = voice
            self.instructions = instructions
            self.temperature = temperature
            self.max_output_tokens = max_output_tokens

    fake_google = types.SimpleNamespace(
        realtime=types.SimpleNamespace(RealtimeModel=FakeRealtimeModel)
    )
    fake_plugins = types.SimpleNamespace(google=fake_google)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.google", fake_google)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini",
        "GEMINI_API_KEY": "cfg-gemini-key",
    })
    model = create_realtime_model(
        cfg,
        call_context="Call context: this is an outbound phone call initiated by Leonardo.",
    )

    assert os.environ["GOOGLE_API_KEY"] == "cfg-gemini-key"
    assert model.model == DEFAULT_GEMINI_REALTIME_MODEL
    assert model.voice == "Puck"
    assert "live phone call" in model.instructions
    assert "one short sentence only" in model.instructions
    assert "Never speak two sentences" in model.instructions
    assert "outbound phone call initiated by Leonardo" in model.instructions
    assert model.temperature == 0.2
    assert model.max_output_tokens == 45


def test_xai_realtime_model_uses_config_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    class FakeRealtimeModel:
        def __init__(self, *, model, voice):
            self.model = model
            self.voice = voice

    fake_xai = types.SimpleNamespace(
        realtime=types.SimpleNamespace(RealtimeModel=FakeRealtimeModel)
    )
    fake_plugins = types.SimpleNamespace(xai=fake_xai)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.xai", fake_xai)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "xai",
        "XAI_API_KEY": "cfg-xai-key",
    })
    model = create_realtime_model(cfg)

    assert os.environ["XAI_API_KEY"] == "cfg-xai-key"
    assert model.model == DEFAULT_XAI_REALTIME_MODEL
    assert model.voice == "ara"



def test_modular_provider_env_parsing_and_public_dict_redaction():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
        "HERMES_LIVEKIT_STT_PROVIDER": "deepgram",
        "HERMES_LIVEKIT_TTS_PROVIDER": "cartesia",
        "HERMES_LIVEKIT_DEEPGRAM_MODEL": "nova-3",
        "HERMES_LIVEKIT_DEEPGRAM_LANGUAGE": "ro",
        "HERMES_LIVEKIT_CARTESIA_MODEL": "sonic-2",
        "HERMES_LIVEKIT_CARTESIA_VOICE": "voice-id",
        "DEEPGRAM_API_KEY": "deepgram-secret",
        "CARTESIA_API_KEY": "cartesia-secret",
    })
    public = cfg.public_dict()
    rendered = json.dumps(public, sort_keys=True)
    assert cfg.uses_modular_pipeline is True
    assert cfg.has_modular_credentials is True
    assert cfg.deepgram_model == "nova-3"
    assert cfg.deepgram_language == "ro"
    assert cfg.cartesia_model == "sonic-2"
    assert public["deepgram_api_key"] == "set"
    assert public["cartesia_api_key"] == "set"
    assert "deepgram-secret" not in rendered
    assert "cartesia-secret" not in rendered


def test_modular_preflight_reports_missing_dependency_or_key(monkeypatch):
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert any(issue["code"] == "missing_deepgram_api_key" for issue in report["issues"])
    assert any(issue["code"] == "missing_cartesia_api_key" for issue in report["issues"])

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    preflight = modular_preflight(load_livekit_config({"HERMES_LIVEKIT_PIPELINE_MODE": "modular"}))
    assert preflight["dependencies_ready"] is False
    assert any("deepgram" in warning for warning in preflight["warnings"])
    assert preflight["cartesia_voice"] in {"set", "missing"}


def test_modular_preflight_accepts_openai_tts_with_openai_key(monkeypatch):
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
        "HERMES_LIVEKIT_STT_PROVIDER": "deepgram",
        "HERMES_LIVEKIT_TTS_PROVIDER": "openai",
        "HERMES_LIVEKIT_OPENAI_TTS_MODEL": "gpt-4o-mini-tts",
        "HERMES_LIVEKIT_OPENAI_TTS_VOICE": "verse",
        "DEEPGRAM_API_KEY": "deepgram-secret",
        "OPENAI_API_KEY": "openai-secret",
    }
    cfg = load_livekit_config(env)

    assert cfg.has_modular_credentials is True
    assert cfg.openai_tts_model == "gpt-4o-mini-tts"
    assert cfg.openai_tts_voice == "verse"
    rendered = json.dumps(cfg.public_dict(), sort_keys=True)
    assert cfg.public_dict()["openai_api_key"] == "set"
    assert "openai-secret" not in rendered

    report = build_livekit_preflight(env, include_realtime=True)
    assert not any(
        issue["code"] == "unsupported_modular_tts_provider"
        for issue in report["issues"]
    )
    assert not any(
        issue["code"] == "missing_openai_api_key" and "TTS" in issue["message"]
        for issue in report["issues"]
    )

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    preflight = modular_preflight(cfg)
    assert preflight["tts_provider"] == "openai"
    assert preflight["openai_tts_model"] == "gpt-4o-mini-tts"
    assert preflight["openai_tts_voice"] == "verse"
    assert preflight["credentials_ready"] is True
    assert preflight["dependencies_ready"] is True


def test_build_modular_session_uses_lazy_livekit_plugin_apis(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)

    class FakeSession:
        def __init__(self, *, stt, tts):
            self.stt = stt
            self.tts = tts

    class FakeSTT:
        def __init__(self, *, model, language, **_kwargs):
            self.model = model
            self.language = language

    class FakeTTS:
        def __init__(self, *, model, voice, **_kwargs):
            self.model = model
            self.voice = voice

    monkeypatch.setattr("gateway.livekit_realtime_agent.AgentSession", FakeSession)
    fake_deepgram = types.SimpleNamespace(STT=FakeSTT)
    fake_cartesia = types.SimpleNamespace(TTS=FakeTTS)
    fake_plugins = types.SimpleNamespace(deepgram=fake_deepgram, cartesia=fake_cartesia)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.deepgram", fake_deepgram)
    monkeypatch.setitem(sys.modules, "livekit.plugins.cartesia", fake_cartesia)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
        "DEEPGRAM_API_KEY": "deepgram-key",
        "CARTESIA_API_KEY": "cartesia-key",
    })
    session = build_modular_session(cfg)

    assert session.stt.model == DEFAULT_DEEPGRAM_MODEL
    assert session.tts.model == DEFAULT_CARTESIA_MODEL


def test_build_modular_session_supports_openai_tts(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeSession:
        def __init__(self, *, stt, tts):
            self.stt = stt
            self.tts = tts

    class FakeSTT:
        def __init__(self, *, model, language, **_kwargs):
            self.model = model
            self.language = language

    class FakeOpenAITTS:
        def __init__(self, *, model, voice, api_key):
            self.model = model
            self.voice = voice
            self.api_key = api_key

    monkeypatch.setattr("gateway.livekit_realtime_agent.AgentSession", FakeSession)
    fake_deepgram = types.SimpleNamespace(STT=FakeSTT)
    fake_openai = types.SimpleNamespace(TTS=FakeOpenAITTS)
    fake_plugins = types.SimpleNamespace(deepgram=fake_deepgram, openai=fake_openai)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.deepgram", fake_deepgram)
    monkeypatch.setitem(sys.modules, "livekit.plugins.openai", fake_openai)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
        "HERMES_LIVEKIT_STT_PROVIDER": "deepgram",
        "HERMES_LIVEKIT_TTS_PROVIDER": "openai",
        "DEEPGRAM_API_KEY": "deepgram-key",
        "OPENAI_API_KEY": "openai-key",
    })
    session = build_modular_session(cfg)

    assert session.stt.model == DEFAULT_DEEPGRAM_MODEL
    assert session.tts.model == DEFAULT_OPENAI_TTS_MODEL
    assert session.tts.voice == DEFAULT_OPENAI_TTS_VOICE
    assert session.tts.api_key == "openai-key"

def test_dispatch_rule_payload_uses_explicit_agent_dispatch():
    payload = build_dispatch_rule_payload(
        agent_name="hermes-live-voice",
        room_prefix="hermes-call-",
        metadata={"route": "hermes-main", "mode": "sip"},
        trunk_ids=["ST_123"],
    )
    assert payload == {
        "name": "Leonardo live voice dispatch",
        "trunkIds": ["ST_123"],
        "rule": {"dispatchRuleIndividual": {"roomPrefix": "hermes-call-"}},
        "roomConfig": {
            "agents": [
                {
                    "agentName": "hermes-live-voice",
                    "metadata": '{"mode":"sip","route":"hermes-main"}',
                }
            ]
        },
    }


def test_dispatch_rule_payload_rejects_invalid_agent_name():
    with pytest.raises(ValueError, match="agent_name"):
        build_dispatch_rule_payload(agent_name="../bad agent")


def test_dispatch_rule_payload_rejects_invalid_trunk_id():
    with pytest.raises(ValueError, match="trunk_ids"):
        build_dispatch_rule_payload(trunk_ids=["ST_good", "../bad"])


def test_build_server_registers_validated_agent_name(monkeypatch):
    calls = []

    class FakeAgentServer:
        def rtc_session(self, entrypoint, *, agent_name):
            calls.append((entrypoint, agent_name))

    fake_agents = types.SimpleNamespace(AgentServer=FakeAgentServer)
    monkeypatch.setitem(sys.modules, "livekit.agents", fake_agents)
    monkeypatch.setenv("HERMES_LIVEKIT_AGENT_NAME", "safe-agent")

    server = build_server()

    assert isinstance(server, FakeAgentServer)
    assert calls[0][1] == "safe-agent"


def test_build_server_rejects_invalid_agent_name(monkeypatch):
    class FakeAgentServer:
        def rtc_session(self, entrypoint, *, agent_name):
            raise AssertionError("invalid agent name should fail before registration")

    fake_agents = types.SimpleNamespace(AgentServer=FakeAgentServer)
    monkeypatch.setitem(sys.modules, "livekit.agents", fake_agents)
    monkeypatch.setenv("HERMES_LIVEKIT_AGENT_NAME", "../bad agent")

    with pytest.raises(ValueError, match="agent_name"):
        build_server()


def test_inbound_trunk_payload_requires_e164_number():
    with pytest.raises(ValueError, match="E.164"):
        build_inbound_trunk_payload("0740000000")
    payload = build_inbound_trunk_payload(
        "+40740000000", allowed_numbers=["+40741111111"]
    )
    assert payload == {
        "trunk": {
            "name": "Leonardo live voice inbound trunk",
            "numbers": ["+40740000000"],
            "krispEnabled": True,
            "allowedNumbers": ["+40741111111"],
        }
    }


def test_outbound_trunk_payload_requires_provider_number():
    with pytest.raises(ValueError, match="E.164"):
        build_outbound_trunk_payload(address="sip.telnyx.com", numbers=["4842079980"])

    payload = build_outbound_trunk_payload(
        address="sip.telnyx.com",
        numbers=["+14842079980"],
        destination_country="US",
    )

    assert payload == {
        "trunk": {
            "name": "Leonardo live voice outbound trunk",
            "address": "sip.telnyx.com",
            "numbers": ["+14842079980"],
            "destinationCountry": "US",
        }
    }


def test_outbound_sip_participant_payload_requires_trunk_or_from_number():
    with pytest.raises(ValueError, match="trunk_id or inline from_number"):
        build_outbound_sip_participant_payload(
            to_number="+15551234567",
            room_name="hermes-call-pa-test",
            participant_identity="callee",
            metadata={"call_profile": "pa", "purpose": "test"},
        )


def test_outbound_call_plan_is_dry_run_safe_and_profiled():
    cfg = load_livekit_config({
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123",
        "HERMES_LIVEKIT_AGENT_NAME": "hermes-live-voice",
    })

    plan = build_outbound_call_plan(
        to_number="+15551234567",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        restaurant_address="Calle de Serrano 1, Madrid, Spain",
        room_name="hermes-call-concierge-test",
        participant_identity="restaurant",
        config=cfg,
    )

    assert plan["mode"] == "dry_run"
    assert plan["requires_execute"] is True
    assert plan["room"] == "hermes-call-concierge-test"
    assert plan["metadata"]["call_profile"] == "concierge"
    assert plan["metadata"]["task_id"] == "reservation-123"
    assert plan["metadata"]["restaurant_address"] == "Calle de Serrano 1, Madrid, Spain"
    assert plan["agent_dispatch"]["agent_name"] == "hermes-live-voice"
    assert plan["sip_participant"]["sip_trunk_id"] == "ST_safe123"
    assert plan["sip_participant"]["sip_call_to"] == "+15551234567"
    assert "Calle de Serrano 1, Madrid, Spain" in plan["sip_participant"]["participant_metadata"]
    assert plan["sip_participant"]["participant_attributes"] == {
        "hermes.profile": "concierge",
        "hermes.purpose": "Book a restaurant callback",
        "hermes.max_duration_seconds": "900",
    }


def test_outbound_execute_authorization_requires_approval_and_allowlist():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123",
        "HERMES_LIVEKIT_OUTBOUND_ALLOWED_NUMBERS": "+15555554567",
        "HERMES_LIVEKIT_OUTBOUND_APPROVAL_IDS": "approval-reservation-123",
    })
    plan = build_outbound_call_plan(
        to_number="+15555554567",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
        config=cfg,
    )

    with pytest.raises(ValueError, match="approval-id"):
        require_outbound_execution_authorization(plan, approval_id="", config=cfg)
    with pytest.raises(ValueError, match="not authorized"):
        require_outbound_execution_authorization(plan, approval_id="wrong", config=cfg)

    denied_plan = dict(plan)
    denied_plan["to_number"] = "+15555559999"
    with pytest.raises(ValueError, match="manual allowlist"):
        require_outbound_execution_authorization(
            denied_plan,
            approval_id="approval-reservation-123",
            config=cfg,
        )

    auth = require_outbound_execution_authorization(
        plan,
        approval_id="approval-reservation-123",
        config=cfg,
    )
    assert auth["approval_id"] == "approval-reservation-123"
    assert auth["to_number"] == plan["to_number"]
    assert auth["task_id"] == "reservation-123"
    assert auth["idempotency_key"] == "reservation-123-call-001"
    assert auth["ledger_path"] == "/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl"


def test_outbound_execute_authorization_requires_task_id():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123",
        "HERMES_LIVEKIT_OUTBOUND_ALLOWED_NUMBERS": "+100000000001",
        "HERMES_LIVEKIT_OUTBOUND_APPROVAL_IDS": "approval-reservation-123",
    })
    plan = build_outbound_call_plan(
        to_number="+100000000001",
        profile="concierge",
        purpose="Book a restaurant callback",
        idempotency_key="reservation-123-call-001",
        ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
        config=cfg,
    )

    with pytest.raises(ValueError, match="task_id"):
        require_outbound_execution_authorization(
            plan,
            approval_id="approval-reservation-123",
            config=cfg,
        )


def test_outbound_execute_authorization_requires_idempotency_and_ledger_metadata():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123",
        "HERMES_LIVEKIT_OUTBOUND_ALLOWED_NUMBERS": "+100000000001",
        "HERMES_LIVEKIT_OUTBOUND_APPROVAL_IDS": "approval-reservation-123",
    })
    plan = build_outbound_call_plan(
        to_number="+100000000001",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        config=cfg,
    )
    with pytest.raises(ValueError, match="idempotency_key"):
        require_outbound_execution_authorization(plan, approval_id="approval-reservation-123", config=cfg)

    plan_with_key = build_outbound_call_plan(
        to_number="+100000000001",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        config=cfg,
    )
    with pytest.raises(ValueError, match="ledger_path or call_notes_path"):
        require_outbound_execution_authorization(plan_with_key, approval_id="approval-reservation-123", config=cfg)

    plan_with_notes = build_outbound_call_plan(
        to_number="+100000000001",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        call_notes_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.md",
        config=cfg,
    )
    auth = require_outbound_execution_authorization(plan_with_notes, approval_id="approval-reservation-123", config=cfg)
    assert auth["idempotency_key"] == "reservation-123-call-001"
    assert auth["ledger_path"].endswith("reservation-123.md")


def test_outbound_execute_authorization_fails_closed_without_env_gates():
    cfg = load_livekit_config({"HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123"})
    plan = build_outbound_call_plan(
        to_number="+15555554567",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
        config=cfg,
    )

    with pytest.raises(ValueError, match="APPROVAL_IDS"):
        require_outbound_execution_authorization(
            plan,
            approval_id="approval-reservation-123",
            config=cfg,
        )


def test_execute_outbound_call_plan_fails_closed_without_authorization():
    cfg = load_livekit_config({
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123",
        "HERMES_LIVEKIT_OUTBOUND_ALLOWED_NUMBERS": "+15555554567",
        "HERMES_LIVEKIT_OUTBOUND_APPROVAL_IDS": "approval-reservation-123",
    })
    plan = build_outbound_call_plan(
        to_number="+15555554567",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
        config=cfg,
    )

    with pytest.raises(ValueError, match="execution_authorization"):
        asyncio.run(execute_outbound_call_plan(plan, config=cfg))


def test_execute_outbound_call_plan_revalidates_authorization_matches_plan():
    cfg = load_livekit_config({
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123",
        "HERMES_LIVEKIT_OUTBOUND_ALLOWED_NUMBERS": "+15555554567",
        "HERMES_LIVEKIT_OUTBOUND_APPROVAL_IDS": "approval-reservation-123",
    })
    plan = build_outbound_call_plan(
        to_number="+15555554567",
        profile="concierge",
        purpose="Book a restaurant callback",
        task_id="reservation-123",
        idempotency_key="reservation-123-call-001",
        ledger_path="/home/pafi/.hermes/profiles/hermes-concierge/workspace/calls/reservation-123.jsonl",
        config=cfg,
    )
    plan["execution_authorization"] = {
        "approval_id": "approval-reservation-123",
        "to_number": "+10000000002",
        "task_id": "reservation-123",
    }

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(execute_outbound_call_plan(plan, config=cfg))


def test_outbound_call_plan_rejects_unsupported_profile_and_long_purpose():
    cfg = load_livekit_config({"HERMES_LIVEKIT_OUTBOUND_TRUNK_ID": "ST_safe123"})
    with pytest.raises(ValueError, match="profile"):
        build_outbound_call_plan(
            to_number="+15555554567",
            profile="sales",
            purpose="Call",
            config=cfg,
        )
    with pytest.raises(ValueError, match="purpose"):
        build_outbound_call_plan(
            to_number="+15555554567",
            profile="pa",
            purpose="x" * 300,
            config=cfg,
        )
    with pytest.raises(ValueError, match="restaurant_address"):
        build_outbound_call_plan(
            to_number="+15555554567",
            profile="concierge",
            purpose="Call restaurant",
            restaurant_address="x" * 400,
            config=cfg,
        )
    with pytest.raises(ValueError, match="task_id"):
        build_outbound_call_plan(
            to_number="+15555554567",
            profile="concierge",
            purpose="Call restaurant",
            task_id="bad task id with spaces",
            config=cfg,
        )


def test_room_name_is_stable_safe_and_prefixed():
    assert (
        build_room_name("Hermes Call ", "Pafi Main Chat", suffix="abc123")
        == "hermes-call-pafi-main-chat-abc123"
    )


def test_room_name_preserves_suffix_when_prefix_is_long():
    room = build_room_name("x" * 200, "session", suffix="abc123")
    assert len(room) <= 96
    assert room.endswith("-abc123")


def test_room_token_output_redacts_jwt_by_default():
    output = build_room_token_output(
        livekit_url="wss://livekit.example.com",
        room="room",
        identity="pafi",
        token="jwt-secret",
    )
    assert output["token"] == "redacted"
    assert output["token_sensitive"] is True
    assert "jwt-secret" not in json.dumps(output)


def test_room_token_output_can_show_jwt_explicitly():
    output = build_room_token_output(
        livekit_url="wss://livekit.example.com",
        room="room",
        identity="pafi",
        token="jwt-secret",
        show_token=True,
    )
    assert output["token"] == "jwt-secret"
