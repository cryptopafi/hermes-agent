import json

import pytest

from gateway import vapi_voice


def test_build_assistant_payload_uses_gpt41_mini_and_elevenlabs_flash():
    payload = vapi_voice.build_gpt41_elevenlabs_assistant(
        {
            "HERMES_VAPI_WEBHOOK_URL": "http://127.0.0.1:11437/vapi/tool",
            "HERMES_VAPI_WEBHOOK_TOKEN": "secret-token",
        }
    )

    assert payload["model"]["provider"] == "openai"
    assert payload["model"]["model"] == "gpt-4.1-mini"
    assert payload["voice"]["provider"] == "11labs"
    assert payload["voice"]["model"] == "eleven_flash_v2_5"
    assert payload["transcriber"]["provider"] == "deepgram"
    assert payload["transcriber"]["language"] == "multi"
    assert payload["model"]["tools"][0]["function"]["name"] == "submit_to_hermes_orchestrator"


def test_build_concierge_assistant_uses_leonardo_identity_and_policy():
    payload = vapi_voice.build_concierge_gpt41_elevenlabs_assistant(
        {
            "HERMES_VAPI_WEBHOOK_URL": "http://127.0.0.1:11437/vapi/tool",
            "HERMES_VAPI_WEBHOOK_TOKEN": "secret-token",
        }
    )

    system_prompt = payload["model"]["messages"][0]["content"]
    assert payload["name"] == vapi_voice.DEFAULT_CONCIERGE_ASSISTANT_NAME
    assert payload["firstMessage"] == "I'm Leonardo Concierge. How can I help you?"
    assert payload["artifactPlan"]["transcriptPlan"]["assistantName"] == "Leonardo"
    assert "Never introduce yourself as Hermes" in system_prompt
    assert "Use English by default with venues and external parties" in system_prompt
    assert "Switch to the venue's local language only if the venue" in system_prompt
    assert "Warm Concierge style by default" in system_prompt
    assert "written confirmation by email or WhatsApp" in system_prompt
    assert "do not accept an alternative automatically" in system_prompt
    assert "do not ask a generic 'How can I help you?'" in system_prompt
    assert "profile_hint concierge" in system_prompt
    assert "results will continue via Telegram or Hermes" in system_prompt
    assert "is already finished unless the tool result explicitly says it is finished" in system_prompt
    assert "payments" in system_prompt
    assert payload["maxDurationSeconds"] == vapi_voice.DEFAULT_MAX_CALL_DURATION_SECONDS
    assert payload["silenceTimeoutSeconds"] == vapi_voice.DEFAULT_SILENCE_TIMEOUT_SECONDS


def test_assistant_guardrail_duration_env_is_bounded():
    payload = vapi_voice.build_gpt41_elevenlabs_assistant(
        {
            "HERMES_VAPI_WEBHOOK_URL": "http://127.0.0.1:11437/vapi/tool",
            "HERMES_VAPI_WEBHOOK_TOKEN": "secret-token",
            "HERMES_VAPI_MAX_DURATION_SECONDS": "9999",
            "HERMES_VAPI_SILENCE_TIMEOUT_SECONDS": "1",
        }
    )

    assert payload["maxDurationSeconds"] == vapi_voice.MAX_CALL_DURATION_SECONDS
    assert payload["silenceTimeoutSeconds"] == 4


def test_rendered_assistant_redacts_webhook_token():
    payload = vapi_voice.build_gpt41_elevenlabs_assistant(
        {
            "HERMES_VAPI_WEBHOOK_URL": "http://127.0.0.1:11437/vapi/tool",
            "HERMES_VAPI_WEBHOOK_TOKEN": "secret-token",
        }
    )

    rendered = json.dumps(vapi_voice.redact_structure(payload), sort_keys=True)

    assert "secret-token" not in rendered
    assert "X-Hermes-Vapi-Token" in rendered
    assert '"maxTokens": 220' in rendered


def test_handle_tool_calls_submits_to_orchestrator(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"run_id": "run_vapi123", "status": "started"}

        return Response()

    monkeypatch.setattr(vapi_voice.httpx, "post", fake_post)
    monkeypatch.setattr(vapi_voice, "DEFAULT_EVENTS_PATH", tmp_path / "events.jsonl")

    response = vapi_voice.handle_vapi_webhook(
        {
            "message": {
                "type": "tool-calls",
                "toolCallList": [
                    {
                        "id": "tool_1",
                        "name": "submit_to_hermes_orchestrator",
                        "parameters": {
                            "task": "Research this with LLM-Wiki and Cortex.",
                            "profile_hint": "hermes-research",
                        },
                    }
                ],
            }
        },
        env={
            "HERMES_LIVEKIT_ORCHESTRATOR_URL": "http://127.0.0.1:8642",
            "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY": "orchestrator-secret",
        },
    )

    result = json.loads(response["results"][0]["result"])
    assert result["run_id"] == "run_vapi123"
    assert captured["url"] == "http://127.0.0.1:8642/v1/runs"
    assert captured["headers"]["Authorization"] == "Bearer orchestrator-secret"
    assert "PHONE VOICE HANDOFF FROM PAFI" in captured["json"]["input"]
    assert "LLM-Wiki" in captured["json"]["input"]


def test_submit_to_orchestrator_rejects_unallowed_url_before_auth_post(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("post should not be called")

    monkeypatch.setattr(vapi_voice.httpx, "post", fake_post)
    monkeypatch.setattr(vapi_voice, "is_hermes_orchestrator_url_allowed", lambda *args, **kwargs: False)

    result = vapi_voice.submit_task_to_orchestrator(
        "Run research.",
        env={
            "HERMES_LIVEKIT_ORCHESTRATOR_URL": "https://evil.example",
            "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY": "orchestrator-secret",
        },
    )

    assert result["status"] == "unavailable"
    assert "url not allowed" in result["message"]
    assert calls == []


def test_handle_tool_calls_rejects_empty_task(monkeypatch, tmp_path):
    monkeypatch.setattr(vapi_voice, "DEFAULT_EVENTS_PATH", tmp_path / "events.jsonl")

    response = vapi_voice.handle_vapi_webhook(
        {
            "message": {
                "type": "tool-calls",
                "toolCallList": [
                    {
                        "id": "tool_1",
                        "name": "submit_to_hermes_orchestrator",
                        "parameters": {"task": "  "},
                    }
                ],
            }
        },
        env={
            "HERMES_LIVEKIT_ORCHESTRATOR_URL": "http://127.0.0.1:8642",
            "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY": "orchestrator-secret",
        },
    )

    result = json.loads(response["results"][0]["result"])
    assert result["status"] == "rejected"


def test_tool_call_parameters_accepts_arguments_dict():
    params = vapi_voice.tool_call_parameters(
        {
            "arguments": {
                "task": "Create a short PA follow-up.",
                "profile_hint": "pa",
            }
        }
    )

    assert params["task"] == "Create a short PA follow-up."
    assert params["profile_hint"] == "pa"


def test_tool_call_parameters_accepts_arguments_json_string():
    params = vapi_voice.tool_call_parameters(
        {
            "arguments": '{"task":"Run Hermes research","profile_hint":"research"}',
        }
    )

    assert params["task"] == "Run Hermes research"
    assert params["profile_hint"] == "research"


def test_tool_call_parameters_accepts_vapi_nested_function_arguments():
    params = vapi_voice.tool_call_parameters(
        {
            "function": {
                "name": "submit_to_hermes_orchestrator",
                "arguments": '{"task":"Find a restaurant in Ibiza","profile_hint":"concierge"}',
            }
        }
    )

    assert params["task"] == "Find a restaurant in Ibiza"
    assert params["profile_hint"] == "concierge"


def test_tool_call_name_accepts_vapi_nested_function_name():
    name = vapi_voice.tool_call_name(
        {
            "function": {
                "name": "submit_to_hermes_orchestrator",
                "arguments": "{}",
            }
        }
    )

    assert name == "submit_to_hermes_orchestrator"


def test_create_outbound_call_requires_e164():
    with pytest.raises(ValueError, match="E.164"):
        vapi_voice.create_outbound_call("0758400900", env_file="/missing")


def test_update_env_file_replaces_and_adds(tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text("A=1\nHERMES_VAPI_WEBHOOK_TOKEN=old\n", encoding="utf-8")

    vapi_voice.update_env_file(
        env_file,
        {
            "HERMES_VAPI_WEBHOOK_TOKEN": "new",
            "HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID": "asst_123",
        },
    )

    text = env_file.read_text(encoding="utf-8")
    assert "HERMES_VAPI_WEBHOOK_TOKEN=new" in text
    assert "HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID=asst_123" in text


def test_deploy_concierge_assistant_writes_concierge_env_key(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\nHERMES_VAPI_WEBHOOK_URL=http://127.0.0.1:11437/vapi/tool\n",
        encoding="utf-8",
    )
    calls = []

    def fake_request(method, path, *, api_key, payload=None, timeout=30.0):
        calls.append((method, path, payload))
        if method == "GET" and path == "/assistant":
            return []
        if method == "POST" and path == "/assistant":
            return {"id": "asst_concierge_123"}
        raise AssertionError((method, path))

    monkeypatch.setattr(vapi_voice, "vapi_request", fake_request)

    result = vapi_voice.deploy_concierge_assistant(env_file)

    assert result["assistant_id"] == "asst_concierge_123"
    assert "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge_123" in env_file.read_text(encoding="utf-8")
    assert calls[-1][2]["artifactPlan"]["transcriptPlan"]["assistantName"] == "Leonardo"


def test_ensure_phone_number_includes_area_code(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID=asst_123\n"
        "HERMES_VAPI_NUMBER_AREA_CODE=484\n",
        encoding="utf-8",
    )
    calls = []

    def fake_request(method, path, *, api_key, payload=None, timeout=30.0):
        calls.append((method, path, payload))
        if method == "GET" and path == "/phone-number":
            return []
        if method == "POST" and path == "/phone-number":
            return {"id": "pn_123", "number": "+14842070000"}
        raise AssertionError((method, path))

    monkeypatch.setattr(vapi_voice, "vapi_request", fake_request)

    result = vapi_voice.ensure_phone_number(env_file)

    assert result["phone_number_id"] == "pn_123"
    assert calls[-1][2]["numberDesiredAreaCode"] == "484"
    assert calls[-1][2]["assistantId"] == "asst_123"


def test_create_outbound_call_uses_concierge_assistant_for_concierge_profile(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID=asst_generic\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+13022045511\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(vapi_voice, "DEFAULT_IDEMPOTENCY_DIR", tmp_path / "idempotency")

    def fake_request(method, path, *, api_key, payload=None, timeout=30.0):
        captured["payload"] = payload
        return {"id": "call_123", "status": "queued"}

    monkeypatch.setattr(vapi_voice, "vapi_request", fake_request)

    result = vapi_voice.create_outbound_call(
        "+13022045511",
        env_file,
        profile="concierge",
        require_public_webhook=False,
        approval_artifact_id="approval-test-1",
        idempotency_key="idem-test-1",
        purpose="Book a table today at 15:00.",
        venue_name="Destino Five",
        reservation_name="Bogdan Roșu",
        party_size="2",
        requested_time="15:00",
        acceptable_window="exactly 15:00 only",
    )

    assert result["profile"] == "concierge"
    assert result["target"] == "+1302***5511"
    assert result["preflight_ok"] is None
    assert result["approval_artifact_id"] == "approval-test-1"
    assert result["idempotency_hash"]
    assert captured["payload"]["name"] == "Leonardo Concierge Vapi test"
    assert captured["payload"]["assistantId"] == "asst_concierge"
    assert captured["payload"]["phoneNumberId"] == "pn_123"
    assert (
        captured["payload"]["assistantOverrides"]["firstMessage"]
        == "Hello, this is Leonardo. I'd like to arrange a table for 2 people today at three PM, please. Would that be possible?"
    )
    variables = captured["payload"]["assistantOverrides"]["variableValues"]
    assert variables["venue_name"] == "Destino Five"
    assert variables["reservation_name"] == "Bogdan Roșu"
    assert variables["party_size"] == "2"
    assert variables["requested_time"] == "15:00"
    assert variables["requested_time_spoken"] == "three PM"
    assert variables["acceptable_window"] == "exactly 15:00 only"


def test_create_outbound_call_passes_operational_confirmation_contacts(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+13022045511\n"
        "HERMES_CONCIERGE_CONFIRMATION_EMAIL=leonardo@example.com\n"
        "HERMES_CONCIERGE_CONFIRMATION_PHONE=+14842079980\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(vapi_voice, "DEFAULT_IDEMPOTENCY_DIR", tmp_path / "idempotency")

    def fake_request(method, path, *, api_key, payload=None, timeout=30.0):
        captured["payload"] = payload
        return {"id": "call_123", "status": "queued"}

    monkeypatch.setattr(vapi_voice, "vapi_request", fake_request)

    vapi_voice.create_outbound_call(
        "+13022045511",
        env_file,
        profile="concierge",
        require_public_webhook=False,
        approval_artifact_id="approval-test-contacts",
        idempotency_key="idem-test-contacts",
        purpose="Book a table.",
        party_size="2",
        requested_time="15:00",
    )

    variables = captured["payload"]["assistantOverrides"]["variableValues"]
    assert variables["confirmation_email"] == "leonardo@example.com"
    assert variables["confirmation_phone"] == "+14842079980"


def test_create_outbound_call_blocks_when_public_webhook_preflight_fails(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+13022045511\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(vapi_voice, "DEFAULT_CALL_LEDGER_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(vapi_voice, "public_webhook_preflight", lambda env_file: {"ok": False, "error": "offline"})

    def fake_request(method, path, *, api_key, payload=None, timeout=30.0):
        calls.append((method, path, payload))
        return {"id": "call_123", "status": "queued"}

    monkeypatch.setattr(vapi_voice, "vapi_request", fake_request)

    with pytest.raises(RuntimeError, match="preflight failed"):
        vapi_voice.create_outbound_call(
            "+13022045511",
            env_file,
            profile="concierge",
            approval_artifact_id="approval-test-1",
            idempotency_key="idem-test-1",
        )

    assert calls == []
    ledger = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert "blocked_preflight" in ledger
    assert "+1302***5511" in ledger


def test_create_outbound_call_requires_manual_allowlist_and_approval(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+40758400900\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(vapi_voice, "public_webhook_preflight", lambda env_file: {"ok": True})

    with pytest.raises(PermissionError, match="not in HERMES_VAPI_ALLOWED_NUMBERS"):
        vapi_voice.create_outbound_call(
            "+13022045511",
            env_file,
            profile="concierge",
            approval_artifact_id="approval-test-1",
            idempotency_key="idem-test-1",
        )

    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+13022045511\n",
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="approval_artifact_id"):
        vapi_voice.create_outbound_call("+13022045511", env_file, profile="concierge")


def test_create_outbound_call_rejects_duplicate_idempotency(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+13022045511\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "calls.jsonl"
    idem_hash = vapi_voice.hash_identifier("idem-test-1")
    ledger.write_text(
        json.dumps({"status": "created", "authorization": {"idempotency_hash": idem_hash}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(vapi_voice, "DEFAULT_CALL_LEDGER_PATH", ledger)
    monkeypatch.setattr(vapi_voice, "DEFAULT_IDEMPOTENCY_DIR", tmp_path / "idempotency")
    monkeypatch.setattr(vapi_voice, "public_webhook_preflight", lambda env_file: {"ok": True})

    with pytest.raises(FileExistsError, match="already used"):
        vapi_voice.create_outbound_call(
            "+13022045511",
            env_file,
            profile="concierge",
            approval_artifact_id="approval-test-1",
            idempotency_key="idem-test-1",
        )

    assert "blocked_duplicate_idempotency" in ledger.read_text(encoding="utf-8")


def test_create_outbound_call_reserves_idempotency_before_vapi_post(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+13022045511\n",
        encoding="utf-8",
    )
    idempotency_dir = tmp_path / "idempotency"
    monkeypatch.setattr(vapi_voice, "DEFAULT_IDEMPOTENCY_DIR", idempotency_dir)
    monkeypatch.setattr(vapi_voice, "DEFAULT_CALL_LEDGER_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(vapi_voice, "public_webhook_preflight", lambda env_file: {"ok": True})
    monkeypatch.setattr(
        vapi_voice,
        "vapi_request",
        lambda method, path, *, api_key, payload=None, timeout=30.0: {"id": "call_123", "status": "queued"},
    )

    vapi_voice.create_outbound_call(
        "+13022045511",
        env_file,
        profile="concierge",
        approval_artifact_id="approval-test-1",
        idempotency_key="idem-test-1",
    )

    with pytest.raises(FileExistsError, match="reserved"):
        vapi_voice.create_outbound_call(
            "+13022045511",
            env_file,
            profile="concierge",
            approval_artifact_id="approval-test-1",
            idempotency_key="idem-test-1",
        )

    assert len(list(idempotency_dir.glob("*.json"))) == 1


def test_parse_content_length_rejects_missing_negative_and_oversized():
    with pytest.raises(vapi_voice.RequestBodyError) as missing_error:
        vapi_voice.parse_content_length(None)
    with pytest.raises(vapi_voice.RequestBodyError) as negative_error:
        vapi_voice.parse_content_length("-1")
    with pytest.raises(vapi_voice.RequestBodyError) as oversized_error:
        vapi_voice.parse_content_length("1001", max_bytes=1000)

    assert missing_error.value.status == 411
    assert negative_error.value.status == 400
    assert oversized_error.value.status == 413
    assert vapi_voice.parse_content_length("42") == 42


def test_rotate_event_log_moves_existing_file_when_limit_would_be_exceeded(tmp_path):
    event_path = tmp_path / "events.jsonl"
    event_path.write_text("old\n", encoding="utf-8")

    vapi_voice.rotate_event_log(event_path, incoming_bytes=10, max_bytes=5)
    vapi_voice.append_event({"type": "status-update"}, path=event_path)

    assert event_path.exists()
    assert (tmp_path / "events.jsonl.1").read_text(encoding="utf-8") == "old\n"
    assert "status-update" in event_path.read_text(encoding="utf-8")


def test_handle_vapi_webhook_summarizes_raw_events_by_default(monkeypatch, tmp_path):
    event_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(vapi_voice, "DEFAULT_EVENTS_PATH", event_path)

    response = vapi_voice.handle_vapi_webhook(
        {
            "message": {
                "type": "end-of-call-report",
                "transcript": "Pafi: private call text",
                "durationSeconds": 12,
                "call": {"id": "call_123"},
                "artifact": {"messages": [{"role": "user", "message": "private call text"}]},
            }
        },
        env={},
    )

    payload = json.loads(event_path.read_text(encoding="utf-8").splitlines()[0])
    assert response == {"ok": True}
    assert payload["message_summary"]["call_id"] == "call_123"
    assert payload["message_summary"]["transcript_chars"] == len("Pafi: private call text")
    assert "private call text" not in event_path.read_text(encoding="utf-8")


def test_handle_vapi_webhook_saves_aftercall_transcript_for_allowed_test(monkeypatch, tmp_path):
    event_path = tmp_path / "events.jsonl"
    transcript_dir = tmp_path / "transcripts"
    monkeypatch.setattr(vapi_voice, "DEFAULT_EVENTS_PATH", event_path)
    monkeypatch.setattr(vapi_voice, "DEFAULT_AFTERCALL_TRANSCRIPT_DIR", transcript_dir)

    response = vapi_voice.handle_vapi_webhook(
        {
            "message": {
                "type": "end-of-call-report",
                "transcript": "Leonardo: test transcript",
                "durationSeconds": 12,
                "endedReason": "customer-ended-call",
                "call": {
                    "id": "019edfae-c457-700e-9bf5-3a5808a79d6b",
                    "customer": {"number": "+13022045511"},
                },
            }
        },
        env={
            "HERMES_VAPI_SAVE_AFTERCALL_TRANSCRIPT": "true",
            "HERMES_VAPI_ALLOWED_NUMBERS": "+13022045511",
        },
    )

    payload = json.loads(event_path.read_text(encoding="utf-8").splitlines()[0])
    transcript_files = list(transcript_dir.glob("*.json"))
    assert response == {"ok": True}
    assert payload["aftercall_transcript"]["saved"] is True
    assert len(transcript_files) == 1
    assert "test transcript" in transcript_files[0].read_text(encoding="utf-8")


def test_normalize_webhook_url_accepts_root_or_tool_path():
    assert (
        vapi_voice.normalize_webhook_url("https://example.trycloudflare.com")
        == "https://example.trycloudflare.com/vapi/tool"
    )
    assert (
        vapi_voice.normalize_webhook_url("http://127.0.0.1:11437")
        == "http://127.0.0.1:11437/vapi/tool"
    )
    assert (
        vapi_voice.normalize_webhook_url("https://example.trycloudflare.com/vapi/tool/")
        == "https://example.trycloudflare.com/vapi/tool"
    )
    with pytest.raises(ValueError, match="http"):
        vapi_voice.normalize_webhook_url("example.trycloudflare.com")
    with pytest.raises(ValueError, match="/vapi/tool"):
        vapi_voice.normalize_webhook_url("https://example.trycloudflare.com/other")
    with pytest.raises(ValueError, match="https"):
        vapi_voice.normalize_webhook_url("http://example.trycloudflare.com")


def test_get_call_status_validates_call_id_before_api(monkeypatch, tmp_path):
    calls = []

    def fake_request(method, path, *, api_key, payload=None, timeout=30.0):
        calls.append(path)
        return {}

    monkeypatch.setattr(vapi_voice, "vapi_request", fake_request)
    with pytest.raises(ValueError, match="call_id"):
        vapi_voice.get_call_status("../secret", tmp_path / "provider.env")

    assert calls == []


def test_sync_webhook_url_updates_env_without_deploy(tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text("HERMES_VAPI_WEBHOOK_URL=https://old.example/vapi/tool\n", encoding="utf-8")

    result = vapi_voice.sync_webhook_url("https://new.example", env_file=env_file, deploy=False)

    assert result == {
        "webhook_url": "https://new.example/vapi/tool",
        "assistant_deployed": False,
        "concierge_assistant_deployed": False,
    }
    assert "HERMES_VAPI_WEBHOOK_URL=https://new.example/vapi/tool" in env_file.read_text(encoding="utf-8")


def test_sync_webhook_url_deploys_generic_and_concierge(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    calls = []

    def fake_deploy_assistant(path):
        calls.append(("generic", path))
        return {"assistant_id": "asst_generic"}

    def fake_deploy_concierge(path):
        calls.append(("concierge", path))
        return {"assistant_id": "asst_concierge"}

    monkeypatch.setattr(vapi_voice, "deploy_assistant", fake_deploy_assistant)
    monkeypatch.setattr(vapi_voice, "deploy_concierge_assistant", fake_deploy_concierge)

    result = vapi_voice.sync_webhook_url("https://new.example", env_file=env_file, deploy=True)

    assert result["assistant_deployed"] is True
    assert result["concierge_assistant_deployed"] is True
    assert calls == [("generic", env_file), ("concierge", env_file)]


def test_public_webhook_preflight_posts_status_update(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "HERMES_VAPI_WEBHOOK_URL=https://public.example/vapi/tool\n"
        "HERMES_VAPI_WEBHOOK_TOKEN=secret-token\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json

        class Response:
            status_code = 200
            text = '{"ok": true}'

        return Response()

    monkeypatch.setattr(vapi_voice.httpx, "post", fake_post)

    result = vapi_voice.public_webhook_preflight(env_file)

    assert result["ok"] is True
    assert captured["url"] == "https://public.example/vapi/tool"
    assert captured["headers"]["X-Hermes-Vapi-Token"] == "secret-token"
    assert captured["json"]["message"]["type"] == "status-update"
    assert captured["json"]["message"]["status"] == "preflight"


def test_call_cli_bypass_requires_test_env_and_reason(monkeypatch, tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "VAPI_API_KEY=vapi-key\n"
        "HERMES_VAPI_ALLOWED_NUMBERS=+13022045511\n"
        "HERMES_VAPI_CONCIERGE_ASSISTANT_ID=asst_concierge\n"
        "HERMES_VAPI_PHONE_NUMBER_ID=pn_123\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        vapi_voice.main(
            [
                "--env-file",
                str(env_file),
                "call",
                "--to",
                "+13022045511",
                "--profile",
                "concierge",
                "--skip-webhook-preflight",
            ]
        )


def test_run_bridge_rejects_non_loopback_without_explicit_env(tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text("HERMES_VAPI_WEBHOOK_TOKEN=secret-token\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-loopback"):
        vapi_voice.run_bridge(host="0.0.0.0", port=0, env_file=env_file)
