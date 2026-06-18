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


def test_normalize_webhook_url_accepts_root_or_tool_path():
    assert (
        vapi_voice.normalize_webhook_url("https://example.trycloudflare.com")
        == "https://example.trycloudflare.com/vapi/tool"
    )
    assert (
        vapi_voice.normalize_webhook_url("https://example.trycloudflare.com/vapi/tool/")
        == "https://example.trycloudflare.com/vapi/tool"
    )
    with pytest.raises(ValueError, match="http"):
        vapi_voice.normalize_webhook_url("example.trycloudflare.com")
    with pytest.raises(ValueError, match="/vapi/tool"):
        vapi_voice.normalize_webhook_url("https://example.trycloudflare.com/other")


def test_sync_webhook_url_updates_env_without_deploy(tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text("HERMES_VAPI_WEBHOOK_URL=https://old.example/vapi/tool\n", encoding="utf-8")

    result = vapi_voice.sync_webhook_url("https://new.example", env_file=env_file, deploy=False)

    assert result == {"webhook_url": "https://new.example/vapi/tool", "assistant_deployed": False}
    assert "HERMES_VAPI_WEBHOOK_URL=https://new.example/vapi/tool" in env_file.read_text(encoding="utf-8")
