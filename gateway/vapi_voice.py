"""Vapi deployment and webhook bridge helpers for Hermes phone voice.

The active LiveKit route remains independent. This module builds a Vapi
assistant candidate and exposes a token-protected webhook that can submit
selected voice tasks to Hermes Orchestrator.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import secrets
import shlex
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import httpx

from gateway.livekit_realtime_agent import build_orchestrator_task_payload


LOGGER = logging.getLogger(__name__)

DEFAULT_ENV_FILE = Path("/home/pafi/.hermes/secrets/provider.env")
DEFAULT_EVENTS_PATH = Path("/home/pafi/.hermes/vapi_voice_events.jsonl")
DEFAULT_ASSISTANT_NAME = "Hermes Vapi GPT41 11Labs"
DEFAULT_CONCIERGE_ASSISTANT_NAME = "Leonardo Concierge Vapi GPT41 11Labs"
DEFAULT_PHONE_NUMBER_NAME = "Hermes Vapi Test"
DEFAULT_VAPI_AREA_CODE = "484"
DEFAULT_WEBHOOK_PORT = 11437
MAX_TASK_CHARS = 8000
MAX_EVENT_BYTES = 750_000
MAX_EVENT_LOG_BYTES = 5_000_000
MAX_REQUEST_BYTES = 1_000_000

_SAFE_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|authorization|credential|bearer)"
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk_car_[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bxai-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"),
)


class RequestBodyError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path).expanduser()
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _SAFE_ENV_KEY_RE.match(key):
            continue
        try:
            parts = shlex.split(value, comments=True, posix=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = value.strip().strip('"').strip("'")
        loaded[key] = value
    return loaded


def merged_env(env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    env = load_env_file(env_file)
    env.update({key: value for key, value in os.environ.items() if value})
    return env


def update_env_file(path: str | Path, updates: Mapping[str, str]) -> None:
    env_path = Path(path).expanduser()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        clean = line.strip()
        prefix = "export " if clean.startswith("export ") else ""
        key = clean[len(prefix) :].split("=", 1)[0].strip() if "=" in clean else ""
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    if remaining and out and out[-1].strip():
        out.append("")
    for key, value in remaining.items():
        if not _SAFE_ENV_KEY_RE.match(key):
            raise ValueError(f"unsafe env key: {key}")
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    env_path.chmod(0o600)


def ensure_webhook_token(env_file: str | Path = DEFAULT_ENV_FILE) -> str:
    env = load_env_file(env_file)
    token = env.get("HERMES_VAPI_WEBHOOK_TOKEN", "").strip()
    if token:
        return token
    token = secrets.token_urlsafe(32)
    update_env_file(env_file, {"HERMES_VAPI_WEBHOOK_TOKEN": token})
    return token


def normalize_webhook_url(url: str) -> str:
    clean = url.strip()
    if not clean:
        raise ValueError("webhook URL is required")
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("webhook URL must be an absolute http(s) URL")
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        path = "/vapi/tool"
    elif path != "/vapi/tool":
        raise ValueError("webhook URL path must be /vapi/tool or a tunnel/domain root")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def sync_webhook_url(
    webhook_url: str,
    *,
    env_file: str | Path = DEFAULT_ENV_FILE,
    deploy: bool = True,
) -> dict[str, Any]:
    normalized = normalize_webhook_url(webhook_url)
    update_env_file(env_file, {"HERMES_VAPI_WEBHOOK_URL": normalized})
    result: dict[str, Any] = {"webhook_url": normalized, "assistant_deployed": False}
    if deploy:
        result["assistant"] = deploy_assistant(env_file)
        result["assistant_deployed"] = True
    return result


def redact_structure(value: Any, key: str = "") -> Any:
    if key.lower() != "maxtokens" and _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact_structure(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_structure(item, key) for item in value]
    if isinstance(value, tuple):
        return [redact_structure(item, key) for item in value]
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_VALUE_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        return text
    return value


def build_system_prompt() -> str:
    return (
        "You are Hermes on a live phone call with Pafi. Keep replies brief and natural. "
        "Understand Romanian and English. Reply in the caller's language unless asked otherwise. "
        "Do not narrate model internals. Do not say only 'understood' for real tasks. "
        "If the caller asks for research, LLM-Wiki, Cortex, coding, file work, PA/concierge actions, "
        "cross-references, or anything that should continue after the call, call "
        "submit_to_hermes_orchestrator with the full task. After the tool returns, tell the caller "
        "the task was submitted and will continue through Hermes. For simple questions that can be "
        "answered safely in one short turn, answer directly."
    )


def build_concierge_system_prompt() -> str:
    return (
        "You are Leonardo, the Concierge voice assistant for Pafi, on a live phone call. "
        "Answer new calls with: I'm Leonardo Concierge. How can I help you? "
        "Never introduce yourself as Hermes. Keep replies brief, natural, and useful. "
        "Use English by default with venues and external parties. "
        "For restaurants, hotels, clubs, vendors, transport, support desks, or external parties, "
        "start and continue in English. Switch to the venue's local language only if the venue "
        "cannot or will not continue in English, or explicitly asks to use the local language. "
        "If you switch, keep the local-language exchange simple and return to English when possible. "
        "You may gather information, coordinate non-payment logistics, and prepare call notes. "
        "You must not accept or offer payments, deposits, card guarantees, purchases, bids, "
        "subscriptions, penalties, or irreversible commitments. If one is requested, stop and say "
        "you need owner approval. Never provide Pafi's personal phone or email; use operational "
        "Concierge contact details only if already provided in the task. For any task that must "
        "continue after the call, including bookings, vendor follow-up, research, LLM-Wiki, Cortex, "
        "files, coding, or PA/concierge execution, call submit_to_hermes_orchestrator with "
        "profile_hint concierge and the complete task. Do not respond only with 'understood' for "
        "real tasks."
    )


def build_orchestrator_tool(*, webhook_url: str, webhook_token: str) -> dict[str, Any]:
    if not webhook_url:
        raise ValueError("webhook_url is required")
    if not webhook_token:
        raise ValueError("webhook_token is required")
    return {
        "type": "function",
        "async": False,
        "server": {
            "url": webhook_url,
            "timeoutSeconds": 20,
            "headers": {"X-Hermes-Vapi-Token": webhook_token},
        },
        "function": {
            "name": "submit_to_hermes_orchestrator",
            "description": (
                "Submit a task from the live phone call to Hermes Orchestrator. "
                "Use for research, LLM-Wiki/Cortex access, coding, file work, PA/concierge tasks, "
                "or anything requiring tools beyond the voice model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The complete task or request exactly as the caller asked it.",
                    },
                    "profile_hint": {
                        "type": "string",
                        "description": "Optional Hermes profile hint such as hermes-research, pa, concierge, or coding.",
                    },
                    "response_mode": {
                        "type": "string",
                        "enum": ["telegram", "call_summary", "background"],
                        "description": "Where Hermes should deliver the result. Default telegram.",
                    },
                },
                "required": ["task"],
            },
        },
    }


def build_vapi_assistant(
    env: Mapping[str, str],
    *,
    name: str,
    first_message: str,
    system_prompt: str,
    transcript_assistant_name: str,
) -> dict[str, Any]:
    webhook_url = env.get("HERMES_VAPI_WEBHOOK_URL", "").strip()
    webhook_token = env.get("HERMES_VAPI_WEBHOOK_TOKEN", "").strip()
    tools = []
    server: dict[str, Any] | None = None
    if webhook_url and webhook_token:
        tools.append(build_orchestrator_tool(webhook_url=webhook_url, webhook_token=webhook_token))
        server = {
            "url": webhook_url,
            "timeoutSeconds": 20,
            "headers": {"X-Hermes-Vapi-Token": webhook_token},
        }
    payload: dict[str, Any] = {
        "name": name,
        "firstMessage": first_message,
        "firstMessageMode": "assistant-speaks-first",
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-3",
            "language": "multi",
            "smartFormat": True,
            "endpointing": 100,
        },
        "model": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.2,
            "maxTokens": 220,
            "messages": [{"role": "system", "content": system_prompt}],
            "tools": tools,
        },
        "voice": {
            "provider": "11labs",
            "voiceId": env.get("HERMES_VAPI_ELEVENLABS_VOICE_ID", "mark"),
            "model": "eleven_flash_v2_5",
        },
        "maxDurationSeconds": 600,
        "silenceTimeoutSeconds": 12,
        "backgroundDenoisingEnabled": True,
        "startSpeakingPlan": {"waitSeconds": 0.25},
        "stopSpeakingPlan": {"numWords": 0, "voiceSeconds": 0.2, "backoffSeconds": 0.6},
        "artifactPlan": {
            "recordingEnabled": False,
            "loggingEnabled": True,
            "transcriptPlan": {
                "enabled": True,
                "assistantName": transcript_assistant_name,
                "userName": "Pafi",
            },
        },
        "serverMessages": [
            "tool-calls",
            "end-of-call-report",
            "status-update",
            "transcript[transcriptType=\"final\"]",
            "hang",
            "user-interrupted",
        ],
        "clientMessages": ["transcript", "status-update", "tool-calls", "user-interrupted"],
        "endCallMessage": "I am ending the call now.",
    }
    if server:
        payload["server"] = server
    return payload


def build_gpt41_elevenlabs_assistant(env: Mapping[str, str]) -> dict[str, Any]:
    return build_vapi_assistant(
        env,
        name=env.get("HERMES_VAPI_ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME),
        first_message="Hermes on Vapi. Tell me what you want me to do.",
        system_prompt=build_system_prompt(),
        transcript_assistant_name="Hermes",
    )


def build_concierge_gpt41_elevenlabs_assistant(env: Mapping[str, str]) -> dict[str, Any]:
    return build_vapi_assistant(
        env,
        name=env.get("HERMES_VAPI_CONCIERGE_ASSISTANT_NAME", DEFAULT_CONCIERGE_ASSISTANT_NAME),
        first_message="I'm Leonardo Concierge. How can I help you?",
        system_prompt=build_concierge_system_prompt(),
        transcript_assistant_name="Leonardo",
    )


def vapi_request(
    method: str,
    path: str,
    *,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    if not api_key:
        raise ValueError("VAPI_API_KEY is required")
    url = f"https://api.vapi.ai{path}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        response = client.request(method, url, headers=headers, json=payload)
    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"Vapi {method} {path} failed HTTP {response.status_code}: {detail}")
    if response.text:
        return response.json()
    return {}


def find_named(items: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    for item in items:
        if str(item.get("name") or "").strip() == name:
            return item
    return None


def deploy_assistant_payload(
    env_file: str | Path,
    *,
    payload_builder: Any,
    env_id_key: str,
) -> dict[str, Any]:
    token = ensure_webhook_token(env_file)
    env = load_env_file(env_file)
    env["HERMES_VAPI_WEBHOOK_TOKEN"] = token
    api_key = env.get("VAPI_API_KEY", "")
    payload = payload_builder(env)
    name = str(payload["name"])
    existing = find_named(vapi_request("GET", "/assistant", api_key=api_key), name)
    if existing and existing.get("id"):
        assistant = vapi_request(
            "PATCH",
            f"/assistant/{existing['id']}",
            api_key=api_key,
            payload=payload,
        )
        action = "updated"
    else:
        assistant = vapi_request("POST", "/assistant", api_key=api_key, payload=payload)
        action = "created"
    assistant_id = str(assistant.get("id") or (existing.get("id") if existing else "")).strip()
    if not assistant_id:
        raise RuntimeError("Vapi assistant response missing id")
    update_env_file(env_file, {env_id_key: assistant_id})
    return {"action": action, "assistant_id": assistant_id, "name": name}


def deploy_assistant(env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, Any]:
    return deploy_assistant_payload(
        env_file,
        payload_builder=build_gpt41_elevenlabs_assistant,
        env_id_key="HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID",
    )


def deploy_concierge_assistant(env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, Any]:
    return deploy_assistant_payload(
        env_file,
        payload_builder=build_concierge_gpt41_elevenlabs_assistant,
        env_id_key="HERMES_VAPI_CONCIERGE_ASSISTANT_ID",
    )


def ensure_phone_number(env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, Any]:
    env = load_env_file(env_file)
    api_key = env.get("VAPI_API_KEY", "")
    assistant_id = env.get("HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID", "")
    if not assistant_id:
        raise ValueError("HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID is required")
    name = env.get("HERMES_VAPI_PHONE_NUMBER_NAME", DEFAULT_PHONE_NUMBER_NAME)
    numbers = vapi_request("GET", "/phone-number", api_key=api_key)
    existing = find_named(numbers, name)
    if existing and existing.get("id"):
        phone = vapi_request(
            "PATCH",
            f"/phone-number/{existing['id']}",
            api_key=api_key,
            payload={"provider": str(existing.get("provider") or "vapi"), "assistantId": assistant_id},
        )
        action = "updated"
    else:
        area_code = env.get("HERMES_VAPI_NUMBER_AREA_CODE", DEFAULT_VAPI_AREA_CODE)
        phone = vapi_request(
            "POST",
            "/phone-number",
            api_key=api_key,
            payload={
                "provider": "vapi",
                "name": name,
                "assistantId": assistant_id,
                "numberDesiredAreaCode": area_code,
            },
        )
        action = "created"
    phone_id = str(phone.get("id") or (existing.get("id") if existing else "")).strip()
    if not phone_id:
        raise RuntimeError("Vapi phone-number response missing id")
    updates = {"HERMES_VAPI_PHONE_NUMBER_ID": phone_id}
    if phone.get("number"):
        updates["HERMES_VAPI_PHONE_NUMBER"] = str(phone["number"])
    update_env_file(env_file, updates)
    return {"action": action, "phone_number_id": phone_id, "number": phone.get("number", ""), "name": name}


def create_outbound_call(
    target_number: str,
    env_file: str | Path = DEFAULT_ENV_FILE,
    *,
    profile: str = "default",
    customer_name: str = "Pafi",
) -> dict[str, Any]:
    if not _E164_RE.match(target_number):
        raise ValueError("target_number must be E.164, for example +40758400900")
    env = load_env_file(env_file)
    api_key = env.get("VAPI_API_KEY", "")
    if profile == "default":
        assistant_id = env.get("HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID", "")
        call_name = "Hermes Vapi test"
    elif profile == "concierge":
        assistant_id = env.get("HERMES_VAPI_CONCIERGE_ASSISTANT_ID", "")
        call_name = "Leonardo Concierge Vapi test"
    else:
        raise ValueError("profile must be default or concierge")
    phone_number_id = env.get("HERMES_VAPI_PHONE_NUMBER_ID", "")
    if not assistant_id or not phone_number_id:
        raise ValueError("assistant and phone number ids are required before placing calls")
    payload = {
        "name": call_name,
        "assistantId": assistant_id,
        "phoneNumberId": phone_number_id,
        "customer": {"number": target_number, "name": customer_name},
    }
    call = vapi_request("POST", "/call", api_key=api_key, payload=payload)
    return {
        "call_id": call.get("id", ""),
        "status": call.get("status", ""),
        "ended_reason": call.get("endedReason", ""),
        "profile": profile,
    }


def append_event(event: Mapping[str, Any], *, path: str | Path | None = None) -> None:
    event_path = Path(path or DEFAULT_EVENTS_PATH)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    payload = redact_structure({"ts": time.time(), **dict(event)})
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    if len(text.encode("utf-8")) > MAX_EVENT_BYTES:
        text = text[:MAX_EVENT_BYTES] + '"...[truncated]"}'
    rotate_event_log(event_path, incoming_bytes=len(text.encode("utf-8")) + 1)
    with event_path.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    event_path.chmod(0o600)


def rotate_event_log(
    event_path: Path,
    *,
    incoming_bytes: int,
    max_bytes: int = MAX_EVENT_LOG_BYTES,
) -> None:
    if max_bytes <= 0 or not event_path.exists():
        return
    try:
        current_size = event_path.stat().st_size
    except OSError:
        return
    if current_size + max(incoming_bytes, 0) <= max_bytes:
        return
    rotated = event_path.with_suffix(event_path.suffix + ".1")
    try:
        if rotated.exists():
            rotated.unlink()
        event_path.replace(rotated)
        rotated.chmod(0o600)
    except OSError as exc:
        LOGGER.warning("Failed to rotate Vapi event log: %s", exc.__class__.__name__)


def parse_content_length(raw_value: str | None, *, max_bytes: int = MAX_REQUEST_BYTES) -> int:
    if raw_value is None or raw_value.strip() == "":
        raise RequestBodyError(411, "content-length required")
    try:
        length = int(raw_value)
    except ValueError as exc:
        raise RequestBodyError(400, "invalid content-length") from exc
    if length < 0:
        raise RequestBodyError(400, "invalid content-length")
    if length > max_bytes:
        raise RequestBodyError(413, "request too large")
    return length


def submit_task_to_orchestrator(
    task: str,
    *,
    profile_hint: str = "",
    response_mode: str = "telegram",
    env: Mapping[str, str],
) -> dict[str, str]:
    clean_task = task.strip()[:MAX_TASK_CHARS]
    if not clean_task:
        return {"status": "rejected", "message": "empty task"}
    url = env.get("HERMES_LIVEKIT_ORCHESTRATOR_URL", "http://127.0.0.1:8642").rstrip("/") + "/v1/runs"
    api_key = env.get("HERMES_LIVEKIT_ORCHESTRATOR_API_KEY", "")
    if not api_key:
        return {"status": "unavailable", "message": "orchestrator credentials missing"}
    payload = build_orchestrator_task_payload(
        clean_task,
        profile_hint=profile_hint,
        response_mode=response_mode,
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=12.0)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        LOGGER.warning("Vapi bridge Orchestrator submission failed: %s", exc.__class__.__name__)
        return {"status": "unavailable", "message": "orchestrator submission failed"}
    run_id = str(data.get("run_id") or "").strip()
    status = str(data.get("status") or "started").strip()
    if not run_id:
        return {"status": "unavailable", "message": "orchestrator response missing run id"}
    return {"status": status, "run_id": run_id, "message": f"submitted to Hermes Orchestrator as {run_id}"}


def tool_call_parameters(tool_call: Mapping[str, Any]) -> Mapping[str, Any]:
    params = tool_call.get("parameters")
    if isinstance(params, Mapping):
        return params
    args = tool_call.get("arguments")
    if isinstance(args, Mapping):
        return args
    if isinstance(args, str):
        try:
            decoded = json.loads(args)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, Mapping):
            return decoded
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        function_params = function.get("parameters")
        if isinstance(function_params, Mapping):
            return function_params
        function_args = function.get("arguments")
        if isinstance(function_args, Mapping):
            return function_args
        if isinstance(function_args, str):
            try:
                decoded = json.loads(function_args)
            except json.JSONDecodeError:
                return {}
            if isinstance(decoded, Mapping):
                return decoded
    return {}


def tool_call_name(tool_call: Mapping[str, Any]) -> str:
    name = str(tool_call.get("name") or "").strip()
    if name:
        return name
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "").strip()
    return ""


def handle_tool_calls(message: Mapping[str, Any], *, env: Mapping[str, str]) -> dict[str, Any]:
    results: list[dict[str, str]] = []
    tool_calls = message.get("toolCallList") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        tool_id = str(tool_call.get("id") or "")
        name = tool_call_name(tool_call)
        params = tool_call_parameters(tool_call)
        if name != "submit_to_hermes_orchestrator":
            result = {"status": "rejected", "message": f"unknown tool {name}"}
        else:
            result = submit_task_to_orchestrator(
                str(params.get("task") or ""),
                profile_hint=str(params.get("profile_hint") or ""),
                response_mode=str(params.get("response_mode") or "telegram"),
                env=env,
            )
        results.append({"toolCallId": tool_id, "name": name, "result": json.dumps(result, ensure_ascii=False)})
    return {"results": results}


def handle_vapi_webhook(body: Mapping[str, Any], *, env: Mapping[str, str]) -> dict[str, Any]:
    message = body.get("message") if isinstance(body.get("message"), Mapping) else {}
    message_type = str(message.get("type") or "")
    if message_type == "tool-calls":
        response = handle_tool_calls(message, env=env)
        append_event({"type": "tool-calls", "response": response})
        return response
    if message_type in {"end-of-call-report", "status-update", "hang", "transcript[transcriptType=\"final\"]", "user-interrupted"}:
        append_event({"type": message_type, "message": message})
        return {"ok": True}
    append_event({"type": message_type or "unknown", "message": message})
    return {"ok": True}


class VapiWebhookHandler(BaseHTTPRequestHandler):
    server_version = "HermesVapiBridge/1.0"

    def _json_response(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json_response(200, {"ok": True, "service": "hermes-vapi-bridge"})
            return
        self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/vapi/tool", "/vapi/events"}:
            self._json_response(404, {"error": "not found"})
            return
        token = self.server.env.get("HERMES_VAPI_WEBHOOK_TOKEN", "")  # type: ignore[attr-defined]
        provided = self.headers.get("X-Hermes-Vapi-Token", "")
        if not token or not hmac.compare_digest(token, provided):
            self._json_response(401, {"error": "unauthorized"})
            return
        try:
            length = parse_content_length(self.headers.get("Content-Length"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except RequestBodyError as exc:
            self._json_response(exc.status, {"error": exc.message})
            return
        except Exception:
            self._json_response(400, {"error": "invalid json"})
            return
        response = handle_vapi_webhook(body, env=self.server.env)  # type: ignore[attr-defined]
        self._json_response(200, response)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("vapi_bridge " + fmt, *args)


class HermesVapiServer(ThreadingHTTPServer):
    env: Mapping[str, str]


def run_bridge(*, host: str, port: int, env_file: str | Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = merged_env(env_file)
    if not env.get("HERMES_VAPI_WEBHOOK_TOKEN"):
        raise RuntimeError("HERMES_VAPI_WEBHOOK_TOKEN is required")
    server = HermesVapiServer((host, port), VapiWebhookHandler)
    server.env = env
    LOGGER.info("Hermes Vapi bridge listening on %s:%s", host, port)
    server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage Hermes Vapi voice integration.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("render-assistant")
    sub.add_parser("render-concierge-assistant")
    sub.add_parser("deploy-assistant")
    sub.add_parser("deploy-concierge-assistant")
    sub.add_parser("ensure-phone-number")
    sync = sub.add_parser("sync-webhook-url")
    sync.add_argument("--url", required=True)
    sync.add_argument("--no-deploy", action="store_true")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--json", action="store_true")
    bridge = sub.add_parser("bridge")
    bridge.add_argument("--host", default="127.0.0.1")
    bridge.add_argument("--port", type=int, default=DEFAULT_WEBHOOK_PORT)
    call = sub.add_parser("call")
    call.add_argument("--to", required=True)
    call.add_argument("--profile", choices=("default", "concierge"), default="default")
    call.add_argument("--customer-name", default="Pafi")

    args = parser.parse_args(argv)
    env_file = Path(args.env_file)
    env = merged_env(env_file)
    if args.cmd == "render-assistant":
        ensure_webhook_token(env_file)
        env = merged_env(env_file)
        print(json.dumps(redact_structure(build_gpt41_elevenlabs_assistant(env)), indent=2, sort_keys=True))
        return 0
    if args.cmd == "render-concierge-assistant":
        ensure_webhook_token(env_file)
        env = merged_env(env_file)
        print(json.dumps(redact_structure(build_concierge_gpt41_elevenlabs_assistant(env)), indent=2, sort_keys=True))
        return 0
    if args.cmd == "deploy-assistant":
        print(json.dumps(deploy_assistant(env_file), indent=2, sort_keys=True))
        return 0
    if args.cmd == "deploy-concierge-assistant":
        print(json.dumps(deploy_concierge_assistant(env_file), indent=2, sort_keys=True))
        return 0
    if args.cmd == "ensure-phone-number":
        print(json.dumps(ensure_phone_number(env_file), indent=2, sort_keys=True))
        return 0
    if args.cmd == "sync-webhook-url":
        print(
            json.dumps(
                sync_webhook_url(args.url, env_file=env_file, deploy=not args.no_deploy),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.cmd == "preflight":
        required = [
            "VAPI_API_KEY",
            "HERMES_VAPI_WEBHOOK_URL",
            "HERMES_VAPI_WEBHOOK_TOKEN",
            "HERMES_LIVEKIT_ORCHESTRATOR_URL",
            "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY",
            "HERMES_VAPI_GPT41_ELEVENLABS_ASSISTANT_ID",
            "HERMES_VAPI_CONCIERGE_ASSISTANT_ID",
        ]
        report = {
            "ok": all(env.get(key) for key in required),
            "required": {key: ("set" if env.get(key) else "missing") for key in required},
            "phone_number_id": "set" if env.get("HERMES_VAPI_PHONE_NUMBER_ID") else "missing",
            "phone_number": env.get("HERMES_VAPI_PHONE_NUMBER", "missing"),
        }
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("Hermes Vapi preflight " + ("OK" if report["ok"] else "BLOCKED"))
            for key, value in report["required"].items():
                print(f"- {key}: {value}")
            print(f"- HERMES_VAPI_PHONE_NUMBER_ID: {report['phone_number_id']}")
            print(f"- HERMES_VAPI_PHONE_NUMBER: {report['phone_number']}")
        return 0 if report["ok"] else 2
    if args.cmd == "bridge":
        run_bridge(host=args.host, port=args.port, env_file=env_file)
        return 0
    if args.cmd == "call":
        print(
            json.dumps(
                create_outbound_call(
                    args.to,
                    env_file,
                    profile=args.profile,
                    customer_name=args.customer_name,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
