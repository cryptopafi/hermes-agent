"""Vapi deployment and webhook bridge helpers for Hermes phone voice.

The active LiveKit route remains independent. This module builds a Vapi
assistant candidate and exposes a token-protected webhook that can submit
selected voice tasks to Hermes Orchestrator.
"""

from __future__ import annotations

import argparse
import hashlib
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

from gateway.livekit_realtime_agent import (
    build_orchestrator_task_payload,
    is_hermes_orchestrator_url_allowed,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_ENV_FILE = Path("/home/pafi/.hermes/secrets/provider.env")
DEFAULT_EVENTS_PATH = Path("/home/pafi/.hermes/vapi_voice_events.jsonl")
DEFAULT_CALL_LEDGER_PATH = Path("/home/pafi/.hermes/vapi_voice_calls.jsonl")
DEFAULT_IDEMPOTENCY_DIR = Path("/home/pafi/.hermes/vapi_voice_idempotency")
DEFAULT_AFTERCALL_TRANSCRIPT_DIR = Path("/home/pafi/.hermes/vapi_voice_aftercall_transcripts")
DEFAULT_TESTING_TRANSCRIPT_DIR = Path("/home/pafi/.hermes/vapi_voice_testing_transcripts")
DEFAULT_TESTING_AUDIT_DIR = Path("/home/pafi/.hermes/vapi_voice_testing_audits")
DEFAULT_WORDING_PACKAGE_PATH = Path("/home/pafi/.hermes/vapi_voice_wording/default-wording-package.md")
DEFAULT_ASSISTANT_NAME = "Hermes Vapi GPT41 11Labs"
DEFAULT_CONCIERGE_ASSISTANT_NAME = "Leonardo Concierge Vapi GPT41 11Labs"
DEFAULT_PHONE_NUMBER_NAME = "Hermes Vapi Test"
DEFAULT_VAPI_AREA_CODE = "484"
DEFAULT_WEBHOOK_PORT = 11437
DEFAULT_MAX_CALL_DURATION_SECONDS = 240
DEFAULT_SILENCE_TIMEOUT_SECONDS = 10
DEFAULT_TOOL_TIMEOUT_SECONDS = 20
MIN_CALL_DURATION_SECONDS = 60
MAX_CALL_DURATION_SECONDS = 600
MAX_TASK_CHARS = 8000
MAX_EVENT_BYTES = 750_000
MAX_EVENT_LOG_BYTES = 5_000_000
MAX_REQUEST_BYTES = 1_000_000
MAX_VAPI_VARIABLE_CHARS = 1200

_SAFE_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")
_VAPI_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
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

_HOUR_WORDS = {
    0: "twelve",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}

_MINUTE_WORDS = {
    0: "",
    5: "oh five",
    10: "ten",
    15: "fifteen",
    20: "twenty",
    25: "twenty five",
    30: "thirty",
    35: "thirty five",
    40: "forty",
    45: "forty five",
    50: "fifty",
    55: "fifty five",
}


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
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("public webhook URL must use https; http is allowed only for loopback")
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        path = "/vapi/tool"
    elif path != "/vapi/tool":
        raise ValueError("webhook URL path must be /vapi/tool or a tunnel/domain root")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def bounded_int_env(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    min_value: int,
    max_value: int,
) -> int:
    raw = str(env.get(key, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def mask_e164(number: str) -> str:
    if not _E164_RE.match(number):
        return "[invalid-number]"
    digits = number[1:]
    if len(digits) <= 6:
        return f"+{digits[0]}***{digits[-2:]}"
    return f"+{digits[:4]}***{digits[-4:]}"


def hash_identifier(value: str) -> str:
    clean = value.strip()
    if not clean:
        return ""
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def clean_vapi_variable(value: str, *, max_chars: int = MAX_VAPI_VARIABLE_CHARS) -> str:
    return " ".join(str(value or "").strip().split())[:max_chars]


def spoken_time(value: str) -> str:
    clean = clean_vapi_variable(value)
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", clean)
    if not match:
        return clean
    hour_24 = int(match.group(1))
    minute = int(match.group(2))
    hour_12 = hour_24 % 12 or 12
    suffix = "AM" if hour_24 < 12 else "PM"
    hour_text = _HOUR_WORDS[hour_12]
    minute_text = _MINUTE_WORDS.get(minute)
    if minute_text is None:
        return clean
    if minute_text:
        return f"{hour_text} {minute_text} {suffix}"
    return f"{hour_text} {suffix}"


def build_call_variable_values(
    *,
    purpose: str,
    venue_name: str = "",
    reservation_name: str = "",
    party_size: str = "",
    requested_time: str = "",
    acceptable_window: str = "",
    confirmation_email: str = "",
    confirmation_phone: str = "",
) -> dict[str, str]:
    return {
        "call_purpose": clean_vapi_variable(purpose),
        "venue_name": clean_vapi_variable(venue_name),
        "reservation_name": clean_vapi_variable(reservation_name),
        "party_size": clean_vapi_variable(party_size),
        "requested_time": clean_vapi_variable(requested_time),
        "requested_time_spoken": spoken_time(requested_time),
        "acceptable_window": clean_vapi_variable(acceptable_window),
        "confirmation_email": clean_vapi_variable(confirmation_email),
        "confirmation_phone": clean_vapi_variable(confirmation_phone),
    }


def build_outbound_first_message(
    *,
    venue_name: str,
    reservation_name: str,
    party_size: str,
    requested_time: str,
) -> str:
    size = clean_vapi_variable(party_size) or "2"
    time_text = spoken_time(requested_time) or "three PM"
    return (
        f"Hello, this is Leonardo. I'd like to arrange a table for {size} people today at {time_text}, please. Would that be possible?"
    )


def parse_allowed_numbers(env: Mapping[str, str]) -> set[str]:
    allowed: set[str] = set()
    raw = env.get("HERMES_VAPI_ALLOWED_NUMBERS", "")
    for value in re.split(r"[\s,;]+", raw):
        clean = value.strip()
        if clean:
            allowed.add(clean)
    allowed_file = env.get("HERMES_VAPI_ALLOWED_NUMBERS_FILE", "").strip()
    if allowed_file:
        path = Path(allowed_file).expanduser()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                clean = line.split("#", 1)[0].strip()
                if clean:
                    allowed.add(clean)
    return {number for number in allowed if _E164_RE.match(number)}


def approval_gate_required(env: Mapping[str, str]) -> bool:
    value = env.get("HERMES_VAPI_REQUIRE_APPROVAL_ENVELOPE", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def truthy_env(env: Mapping[str, str], key: str, *, default: bool = False) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def tuple_env(env: Mapping[str, str], key: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in re.split(r"[\s,;]+", env.get(key, "")) if value.strip())


def validate_call_authorization(
    target_number: str,
    env: Mapping[str, str],
    *,
    approval_artifact_id: str,
    idempotency_key: str,
    require_approval_gate: bool,
) -> dict[str, Any]:
    allowed_numbers = parse_allowed_numbers(env)
    if target_number not in allowed_numbers:
        raise PermissionError("target_number is not in HERMES_VAPI_ALLOWED_NUMBERS")
    if require_approval_gate and approval_gate_required(env):
        if not approval_artifact_id.strip():
            raise PermissionError("approval_artifact_id is required before placing calls")
        if not idempotency_key.strip():
            raise PermissionError("idempotency_key is required before placing calls")
    return {
        "allowed_numbers_count": len(allowed_numbers),
        "approval_artifact_id": approval_artifact_id.strip(),
        "idempotency_hash": hash_identifier(idempotency_key),
    }


def sync_webhook_url(
    webhook_url: str,
    *,
    env_file: str | Path = DEFAULT_ENV_FILE,
    deploy: bool = True,
) -> dict[str, Any]:
    normalized = normalize_webhook_url(webhook_url)
    update_env_file(env_file, {"HERMES_VAPI_WEBHOOK_URL": normalized})
    result: dict[str, Any] = {
        "webhook_url": normalized,
        "assistant_deployed": False,
        "concierge_assistant_deployed": False,
    }
    if deploy:
        result["assistant"] = deploy_assistant(env_file)
        result["assistant_deployed"] = True
        result["concierge_assistant"] = deploy_concierge_assistant(env_file)
        result["concierge_assistant_deployed"] = True
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
        "If call variables are provided, treat them as the active outbound task: "
        "venue={{venue_name}}, call_purpose={{call_purpose}}, reservation_name={{reservation_name}}, "
        "party_size={{party_size}}, requested_time={{requested_time}}, "
        "requested_time_spoken={{requested_time_spoken}}, acceptable_window={{acceptable_window}}, "
        "confirmation_email={{confirmation_email}}, confirmation_phone={{confirmation_phone}}. "
        "For outbound venue calls, use the Warm Concierge style by default: introduce yourself only as Leonardo, "
        "then politely ask for exactly the reservation described by the active outbound task. Say the requested "
        "time in natural spoken words, for example 'three PM', not '15:00'. Do not say the reservation guest's "
        "name in the opening sentence; use it later only when the venue asks for the booking name or needs it "
        "to complete the reservation. "
        "Never introduce yourself as Hermes. Keep replies brief, natural, and useful. "
        "Use English by default with venues and external parties. "
        "For restaurants, hotels, clubs, vendors, transport, support desks, or external parties, "
        "start and continue in English. Switch to the venue's local language only if the venue "
        "cannot or will not continue in English, or explicitly asks to use the local language. "
        "If you switch, keep the local-language exchange simple and return to English when possible. "
        "You may gather information, coordinate non-payment logistics, and prepare call notes. "
        "If the requested time is available, confirm the date, time, party size, and reservation name, then ask "
        "the venue to send written confirmation by email or WhatsApp using the provided operational Concierge "
        "contact fields. If only one contact field is provided, use that one. Never provide Pafi's personal "
        "phone or email. "
        "If the requested time is unavailable, ask what alternatives are available, but do not accept an "
        "alternative automatically. Say you will check and call back to confirm. End the call after collecting "
        "the alternatives; do not ask a generic 'How can I help you?' after the booking exchange. "
        "If this is a callback and the active task says an alternative is approved, confirm only that approved "
        "option and then ask for written confirmation by email or WhatsApp. "
        "You must not accept or offer payments, deposits, card guarantees, purchases, bids, "
        "subscriptions, penalties, or irreversible commitments. If one is requested, stop and say "
        "you need approval first. For any task that must "
        "continue after the call, including bookings, vendor follow-up, research, LLM-Wiki, Cortex, "
        "files, coding, or PA/concierge execution, call submit_to_hermes_orchestrator with "
        "profile_hint concierge and the complete task. After the tool returns, say the task was "
        "submitted to Hermes/Leo and that results will continue via Telegram or Hermes. Do not "
        "claim the research, booking, follow-up, or file work is already finished unless the tool "
        "result explicitly says it is finished. Do not respond only with 'understood' for real tasks."
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
            "timeoutSeconds": DEFAULT_TOOL_TIMEOUT_SECONDS,
            "headers": {"X-Hermes-Vapi-Token": webhook_token},
        },
        "function": {
            "name": "submit_to_hermes_orchestrator",
            "description": (
                "Submit a task from the live phone call to Hermes Orchestrator. "
                "Use for research, LLM-Wiki/Cortex access, coding, file work, PA/concierge tasks, "
                "or anything requiring tools beyond the voice model. The return value confirms "
                "submission/run status; it is not proof that the background task is complete."
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
    max_duration_seconds = bounded_int_env(
        env,
        "HERMES_VAPI_MAX_DURATION_SECONDS",
        DEFAULT_MAX_CALL_DURATION_SECONDS,
        min_value=MIN_CALL_DURATION_SECONDS,
        max_value=MAX_CALL_DURATION_SECONDS,
    )
    silence_timeout_seconds = bounded_int_env(
        env,
        "HERMES_VAPI_SILENCE_TIMEOUT_SECONDS",
        DEFAULT_SILENCE_TIMEOUT_SECONDS,
        min_value=4,
        max_value=30,
    )
    tool_timeout_seconds = bounded_int_env(
        env,
        "HERMES_VAPI_TOOL_TIMEOUT_SECONDS",
        DEFAULT_TOOL_TIMEOUT_SECONDS,
        min_value=5,
        max_value=30,
    )
    tools = []
    server: dict[str, Any] | None = None
    if webhook_url and webhook_token:
        tools.append(build_orchestrator_tool(webhook_url=webhook_url, webhook_token=webhook_token))
        server = {
            "url": webhook_url,
            "timeoutSeconds": tool_timeout_seconds,
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
        "maxDurationSeconds": max_duration_seconds,
        "silenceTimeoutSeconds": silence_timeout_seconds,
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
        detail = str(redact_structure(response.text))[:500]
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


def public_webhook_preflight(
    env_file: str | Path = DEFAULT_ENV_FILE,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    env = load_env_file(env_file)
    webhook_url = env.get("HERMES_VAPI_WEBHOOK_URL", "").strip()
    webhook_token = env.get("HERMES_VAPI_WEBHOOK_TOKEN", "").strip()
    if not webhook_url or not webhook_token:
        return {"ok": False, "error": "webhook url/token missing"}
    try:
        normalized = normalize_webhook_url(webhook_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    payload = {
        "message": {
            "type": "status-update",
            "status": "preflight",
            "source": "hermes-vapi-bridge-preflight",
            "timestamp": time.time(),
        }
    }
    headers = {
        "Content-Type": "application/json",
        "X-Hermes-Vapi-Token": webhook_token,
    }
    try:
        response = httpx.post(normalized, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:
        return {"ok": False, "webhook_url": normalized, "error": exc.__class__.__name__}
    result: dict[str, Any] = {
        "ok": 200 <= response.status_code < 300,
        "webhook_url": normalized,
        "status_code": response.status_code,
    }
    if not result["ok"]:
        result["error"] = response.text[:200]
    return result


def append_call_ledger(entry: Mapping[str, Any], *, path: str | Path | None = None) -> None:
    ledger_path = Path(path or DEFAULT_CALL_LEDGER_PATH)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = redact_structure({"ts": time.time(), **dict(entry)})
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    ledger_path.chmod(0o600)


def call_ledger_has_idempotency_hash(idempotency_hash: str, *, path: str | Path | None = None) -> bool:
    if not idempotency_hash:
        return False
    ledger_path = Path(path or DEFAULT_CALL_LEDGER_PATH)
    if not ledger_path.exists():
        return False
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        authorization = entry.get("authorization") if isinstance(entry.get("authorization"), Mapping) else {}
        if entry.get("status") == "created" and authorization.get("idempotency_hash") == idempotency_hash:
            return True
    return False


def reserve_idempotency_hash(
    idempotency_hash: str,
    *,
    target_number: str,
    profile: str,
    purpose: str,
    path: str | Path | None = None,
) -> Path:
    if not idempotency_hash or not re.match(r"^[a-f0-9]{16}$", idempotency_hash):
        raise ValueError("valid idempotency_hash is required")
    reserve_dir = Path(path or DEFAULT_IDEMPOTENCY_DIR)
    reserve_dir.mkdir(parents=True, exist_ok=True)
    reserve_dir.chmod(0o700)
    reserve_path = reserve_dir / f"{idempotency_hash}.json"
    payload = {
        "ts": time.time(),
        "status": "reserved",
        "target": mask_e164(target_number),
        "profile": profile,
        "purpose": purpose,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(reserve_path, flags, 0o600)
    except FileExistsError:
        raise FileExistsError("idempotency_key was already reserved for a Vapi call")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return reserve_path


def create_outbound_call(
    target_number: str,
    env_file: str | Path = DEFAULT_ENV_FILE,
    *,
    profile: str = "default",
    customer_name: str = "Pafi",
    purpose: str = "supervised_test",
    require_public_webhook: bool = True,
    approval_artifact_id: str = "",
    idempotency_key: str = "",
    require_approval_gate: bool = True,
    bypass_reason: str = "",
    venue_name: str = "",
    reservation_name: str = "",
    party_size: str = "",
    requested_time: str = "",
    acceptable_window: str = "",
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
    authorization = validate_call_authorization(
        target_number,
        env,
        approval_artifact_id=approval_artifact_id,
        idempotency_key=idempotency_key,
        require_approval_gate=require_approval_gate,
    )
    if call_ledger_has_idempotency_hash(authorization["idempotency_hash"]):
        append_call_ledger(
            {
                "status": "blocked_duplicate_idempotency",
                "profile": profile,
                "purpose": purpose,
                "target": mask_e164(target_number),
                "authorization": authorization,
            }
        )
        raise FileExistsError("idempotency_key was already used for a created Vapi call")
    preflight: dict[str, Any] | None = None
    if require_public_webhook:
        preflight = public_webhook_preflight(env_file)
        if not preflight.get("ok"):
            append_call_ledger(
                {
                    "status": "blocked_preflight",
                    "profile": profile,
                    "purpose": purpose,
                    "target": mask_e164(target_number),
                    "authorization": authorization,
                    "preflight": preflight,
                }
            )
            raise RuntimeError(f"public webhook preflight failed: {preflight.get('error') or preflight}")
    reserve_idempotency_hash(
        authorization["idempotency_hash"],
        target_number=target_number,
        profile=profile,
        purpose=purpose,
    )
    payload = {
        "name": call_name,
        "assistantId": assistant_id,
        "phoneNumberId": phone_number_id,
        "customer": {"number": target_number, "name": customer_name},
        "assistantOverrides": {
            "firstMessage": build_outbound_first_message(
                venue_name=venue_name,
                reservation_name=reservation_name,
                party_size=party_size,
                requested_time=requested_time,
            ),
            "variableValues": build_call_variable_values(
                purpose=purpose,
                venue_name=venue_name,
                reservation_name=reservation_name,
                party_size=party_size,
                requested_time=requested_time,
                acceptable_window=acceptable_window,
                confirmation_email=env.get("HERMES_CONCIERGE_CONFIRMATION_EMAIL", ""),
                confirmation_phone=env.get("HERMES_CONCIERGE_CONFIRMATION_PHONE", ""),
            )
        },
    }
    call = vapi_request("POST", "/call", api_key=api_key, payload=payload)
    result = {
        "call_id": call.get("id", ""),
        "status": call.get("status", ""),
        "ended_reason": call.get("endedReason", ""),
        "profile": profile,
        "target": mask_e164(target_number),
        "preflight_ok": bool(preflight.get("ok")) if preflight is not None else None,
        "approval_artifact_id": authorization["approval_artifact_id"],
        "idempotency_hash": authorization["idempotency_hash"],
        "bypass_reason": bypass_reason.strip(),
    }
    append_call_ledger(
        {
            "status": "created",
            "profile": profile,
            "purpose": purpose,
            "target": mask_e164(target_number),
            "call_id": result["call_id"],
            "vapi_status": result["status"],
            "preflight_ok": result["preflight_ok"],
            "authorization": authorization,
            "bypass_reason": bypass_reason.strip(),
        }
    )
    return result


def get_call_status(call_id: str, env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, Any]:
    clean_call_id = call_id.strip()
    if not _VAPI_CALL_ID_RE.match(clean_call_id):
        raise ValueError("call_id must contain only letters, numbers, underscores, or hyphens")
    env = load_env_file(env_file)
    api_key = env.get("VAPI_API_KEY", "")
    call = vapi_request("GET", f"/call/{clean_call_id}", api_key=api_key)
    analysis = call.get("analysis") if isinstance(call.get("analysis"), Mapping) else {}
    metrics = call.get("performanceMetrics") if isinstance(call.get("performanceMetrics"), Mapping) else {}
    return redact_structure(
        {
            "call_id": call.get("id", clean_call_id),
            "status": call.get("status", ""),
            "ended_reason": call.get("endedReason", ""),
            "started_at": call.get("startedAt", ""),
            "ended_at": call.get("endedAt", ""),
            "cost": call.get("cost", ""),
            "success_evaluation": analysis.get("successEvaluation"),
            "summary": analysis.get("summary", ""),
            "performance_metrics": metrics,
        }
    )


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


def summarized_vapi_message(message: Mapping[str, Any]) -> dict[str, Any]:
    message_type = str(message.get("type") or "")
    call = message.get("call") if isinstance(message.get("call"), Mapping) else {}
    analysis = message.get("analysis") if isinstance(message.get("analysis"), Mapping) else {}
    artifact = message.get("artifact") if isinstance(message.get("artifact"), Mapping) else {}
    transcript = str(message.get("transcript") or "")
    messages = artifact.get("messages") if isinstance(artifact.get("messages"), list) else []
    summary: dict[str, Any] = {
        "type": message_type,
        "status": message.get("status", ""),
        "source": message.get("source", ""),
        "call_id": call.get("id") or message.get("callId", ""),
        "ended_reason": message.get("endedReason", ""),
        "duration_seconds": message.get("durationSeconds", ""),
        "cost": message.get("cost", ""),
        "success_evaluation": analysis.get("successEvaluation"),
        "summary": analysis.get("summary", ""),
        "message_count": len(messages),
        "transcript_chars": len(transcript),
    }
    return {key: value for key, value in summary.items() if value not in ("", None)}


def extract_customer_number(message: Mapping[str, Any]) -> str:
    customer = message.get("customer") if isinstance(message.get("customer"), Mapping) else {}
    call = message.get("call") if isinstance(message.get("call"), Mapping) else {}
    call_customer = call.get("customer") if isinstance(call.get("customer"), Mapping) else {}
    return str(customer.get("number") or call_customer.get("number") or "").strip()


def save_aftercall_transcript(
    message: Mapping[str, Any],
    *,
    env: Mapping[str, str],
    path: str | Path | None = None,
) -> dict[str, Any]:
    if not truthy_env(env, "HERMES_VAPI_SAVE_AFTERCALL_TRANSCRIPT"):
        return {"saved": False, "reason": "disabled"}
    transcript = str(message.get("transcript") or "").strip()
    if not transcript:
        return {"saved": False, "reason": "empty"}
    customer_number = extract_customer_number(message)
    if customer_number not in parse_allowed_numbers(env):
        return {"saved": False, "reason": "customer_not_allowed"}
    call = message.get("call") if isinstance(message.get("call"), Mapping) else {}
    call_id = str(call.get("id") or message.get("callId") or "").strip()
    if not _VAPI_CALL_ID_RE.match(call_id):
        return {"saved": False, "reason": "invalid_call_id"}
    transcript_dir = Path(path or DEFAULT_AFTERCALL_TRANSCRIPT_DIR)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.chmod(0o700)
    transcript_path = transcript_dir / f"{call_id}.json"
    payload = redact_structure(
        {
            "ts": time.time(),
            "call_id": call_id,
            "customer": mask_e164(customer_number),
            "venue_test": True,
            "duration_seconds": message.get("durationSeconds", ""),
            "ended_reason": message.get("endedReason", ""),
            "summary": message.get("summary", ""),
            "transcript": transcript,
        }
    )
    transcript_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    transcript_path.chmod(0o600)
    return {"saved": True, "path": str(transcript_path)}


def testing_transcript_retention_enabled(env: Mapping[str, str]) -> bool:
    """Return true while Vapi calls are in pre-production testing mode."""
    return truthy_env(env, "HERMES_VAPI_TESTING_TRANSCRIPT_RETENTION")


def vapi_message_call_id(message: Mapping[str, Any]) -> str:
    raw_call = message.get("call")
    call: Mapping[str, Any] = raw_call if isinstance(raw_call, Mapping) else {}
    return str(call.get("id") or message.get("callId") or "").strip()


def save_testing_transcript_snapshot(
    message: Mapping[str, Any],
    *,
    env: Mapping[str, str],
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append full pre-production transcript artifacts for every Vapi test call event.

    This intentionally stores transcript-bearing payloads separately from the compact event
    log so testing evidence is not lost to summary-only logging or event-log rotation.
    Disable by setting HERMES_VAPI_TESTING_TRANSCRIPT_RETENTION=false once production starts.
    """
    if not testing_transcript_retention_enabled(env):
        return {"saved": False, "reason": "disabled"}
    call_id = vapi_message_call_id(message)
    if not _VAPI_CALL_ID_RE.match(call_id):
        return {"saved": False, "reason": "invalid_call_id"}
    raw_artifact = message.get("artifact")
    artifact: Mapping[str, Any] = raw_artifact if isinstance(raw_artifact, Mapping) else {}
    transcript = str(message.get("transcript") or artifact.get("transcript") or "").strip()
    raw_artifact_messages = artifact.get("messages")
    artifact_messages = raw_artifact_messages if isinstance(raw_artifact_messages, list) else []
    if not transcript and not artifact_messages:
        return {"saved": False, "reason": "no_transcript_payload"}
    transcript_dir = Path(path or env.get("HERMES_VAPI_TESTING_TRANSCRIPT_DIR") or DEFAULT_TESTING_TRANSCRIPT_DIR)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.chmod(0o700)
    transcript_path = transcript_dir / f"{call_id}.jsonl"
    payload = redact_structure(
        {
            "ts": time.time(),
            "call_id": call_id,
            "message_type": message.get("type", ""),
            "customer": mask_e164(extract_customer_number(message)),
            "duration_seconds": message.get("durationSeconds", ""),
            "ended_reason": message.get("endedReason", ""),
            "summary": message.get("summary", ""),
            "analysis": message.get("analysis", {}),
            "transcript": transcript,
            "artifact_messages": artifact_messages,
        }
    )
    with transcript_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    transcript_path.chmod(0o600)
    audit = process_testing_transcript_audit(payload, transcript_path=transcript_path, env=env)
    result = {"saved": True, "path": str(transcript_path)}
    if audit.get("saved"):
        result["audit"] = audit
    return result


def _keyword_present(text: str, patterns: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _extract_wording_bullets(wording: str) -> list[str]:
    bullets: list[str] = []
    for line in wording.splitlines():
        clean = line.strip()
        if clean.startswith("- "):
            bullets.append(clean[2:].strip())
    return bullets[:40]


def _line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def process_testing_transcript_audit(
    payload: Mapping[str, Any],
    *,
    transcript_path: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Create deterministic post-call enrichment/audit/cross-reference artifacts for Vapi tests."""
    call_id = str(payload.get("call_id") or "").strip()
    if not _VAPI_CALL_ID_RE.match(call_id):
        return {"saved": False, "reason": "invalid_call_id"}
    transcript = str(payload.get("transcript") or "")
    raw_artifact_messages = payload.get("artifact_messages")
    artifact_messages: list[Any] = raw_artifact_messages if isinstance(raw_artifact_messages, list) else []
    if not transcript and not artifact_messages:
        return {"saved": False, "reason": "empty"}
    wording_path = Path(env.get("HERMES_VAPI_DEFAULT_WORDING_PACKAGE_PATH") or DEFAULT_WORDING_PACKAGE_PATH)
    wording = wording_path.read_text(encoding="utf-8") if wording_path.exists() else ""
    audit_dir = Path(env.get("HERMES_VAPI_TESTING_AUDIT_DIR") or DEFAULT_TESTING_AUDIT_DIR)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.chmod(0o700)
    audit_path = audit_dir / f"{call_id}.md"

    text_for_checks = transcript or json.dumps(artifact_messages, ensure_ascii=False, default=str)
    lower = text_for_checks.lower()
    required_checks = {
        "identity_and_purpose": _keyword_present(lower, ("leonardo", "hermes", "book", "reservation", "table")),
        "availability_before_commitment": _keyword_present(lower, ("available", "availability", "do you have", "can you", "could you")),
        "reservation_details_present": _keyword_present(lower, ("people", "party", "table")) and _keyword_present(lower, ("today", "tomorrow", ":", "pm", "am")),
        "recap_or_confirmation": _keyword_present(lower, ("confirm", "recap", "so", "thank you")),
        "financial_boundary_detected": _keyword_present(lower, ("deposit", "card", "prepay", "pre-payment", "payment", "guarantee", "purchase", "subscription", "bid", "crypto", "transfer")),
        "financial_boundary_handled": _keyword_present(lower, ("no card", "no payment", "cannot", "can't", "decline", "escalate", "ask pafi", "without payment", "no deposit")),
        "financial_commitment_made": _keyword_present(lower, ("i will pay", "we will pay", "charge the card", "you can charge", "i agree to pay", "make the payment")),
    }
    score = 100
    if not required_checks["identity_and_purpose"]:
        score -= 15
    if not required_checks["availability_before_commitment"]:
        score -= 15
    if not required_checks["reservation_details_present"]:
        score -= 15
    if not required_checks["recap_or_confirmation"]:
        score -= 10
    if required_checks["financial_commitment_made"]:
        score = 0
    elif required_checks["financial_boundary_detected"] and not required_checks["financial_boundary_handled"]:
        score -= 10
    score = max(0, min(100, score))

    wording_bullets = _extract_wording_bullets(wording)
    matched_bullets = [bullet for bullet in wording_bullets if _keyword_present(lower, tuple(word.lower() for word in re.findall(r"[A-Za-z][A-Za-z-]{4,}", bullet)[:4]))]
    missing_bullets = [bullet for bullet in wording_bullets if bullet not in matched_bullets]
    recommendations: list[str] = []
    if not required_checks["identity_and_purpose"]:
        recommendations.append("Open with identity + exact booking purpose from the wording package.")
    if not required_checks["availability_before_commitment"]:
        recommendations.append("Ask for availability before confirming any reservation details.")
    if not required_checks["reservation_details_present"]:
        recommendations.append("State or capture venue, reservation name, party size, requested time, and acceptable window.")
    if required_checks["financial_boundary_detected"] and not required_checks["financial_boundary_handled"]:
        recommendations.append("Financial request appeared; verify the assistant declined commitment and escalated to Pafi.")
    if required_checks["financial_commitment_made"]:
        recommendations.append("BLOCKER: possible payment/financial commitment wording detected; do not proceed to production until fixed.")
    if not recommendations:
        recommendations.append("No deterministic blocker found; review transcript for tone, latency, barge-in handling, and naturalness.")

    audit_payload = {
        "call_id": call_id,
        "ts": time.time(),
        "score": score,
        "transcript_path": str(transcript_path),
        "wording_package_path": str(wording_path),
        "transcript_chars": len(transcript),
        "transcript_lines": _line_count(transcript),
        "artifact_messages": len(artifact_messages),
        "checks": required_checks,
        "matched_wording_bullets": matched_bullets[:20],
        "missing_wording_bullets": missing_bullets[:20],
        "recommendations": recommendations,
    }
    md = [
        f"# Vapi Test Call Audit — {call_id}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"Score: {score}/100",
        f"Transcript: `{transcript_path}`",
        f"Wording package: `{wording_path}`",
        "",
        "## Enrichment",
        f"- Transcript chars: {len(transcript)}",
        f"- Transcript non-empty lines: {_line_count(transcript)}",
        f"- Artifact messages: {len(artifact_messages)}",
        "",
        "## Audit Checks",
    ]
    for key, value in required_checks.items():
        if key == "financial_commitment_made":
            passed = not value
        elif key == "financial_boundary_handled":
            passed = (not required_checks["financial_boundary_detected"]) or bool(value)
        else:
            passed = bool(value)
        label = "PASS" if passed else "FAIL"
        md.append(f"- {key}: {label}")
    md.extend(["", "## Wording Cross-reference", "Matched bullets:"])
    md.extend([f"- {bullet}" for bullet in matched_bullets[:20]] or ["- none"])
    md.append("Missing / needs manual review:")
    md.extend([f"- {bullet}" for bullet in missing_bullets[:20]] or ["- none"])
    md.extend(["", "## Optimization Recommendations"])
    md.extend([f"- {item}" for item in recommendations])
    md.extend(["", "## Raw JSON", "```json", json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), "```", ""])
    audit_path.write_text("\n".join(md), encoding="utf-8")
    audit_path.chmod(0o600)
    return {"saved": True, "path": str(audit_path), "score": score}


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
    base_url = env.get("HERMES_LIVEKIT_ORCHESTRATOR_URL", "http://127.0.0.1:8642").rstrip("/")
    api_key = env.get("HERMES_LIVEKIT_ORCHESTRATOR_API_KEY", "")
    if not api_key:
        return {"status": "unavailable", "message": "orchestrator credentials missing"}
    if not is_hermes_orchestrator_url_allowed(
        base_url,
        allow_remote=truthy_env(env, "HERMES_LIVEKIT_ORCHESTRATOR_ALLOW_REMOTE"),
        allowed_hosts=tuple_env(env, "HERMES_LIVEKIT_ORCHESTRATOR_ALLOWED_HOSTS"),
    ):
        return {"status": "unavailable", "message": "orchestrator url not allowed"}
    url = base_url + "/v1/runs"
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
    raw_message = body.get("message")
    message: Mapping[str, Any] = raw_message if isinstance(raw_message, Mapping) else {}
    message_type = str(message.get("type") or "")
    if message_type == "tool-calls":
        response = handle_tool_calls(message, env=env)
        append_event({"type": "tool-calls", "response": response})
        return response
    if message_type in {"end-of-call-report", "status-update", "hang", "transcript[transcriptType=\"final\"]", "user-interrupted"}:
        event: dict[str, Any] = {"type": message_type, "message_summary": summarized_vapi_message(message)}
        testing_snapshot = save_testing_transcript_snapshot(message, env=env)
        if testing_snapshot.get("saved"):
            event["testing_transcript_snapshot"] = testing_snapshot
        if message_type == "end-of-call-report":
            event["aftercall_transcript"] = save_aftercall_transcript(message, env=env)
        append_event(event)
        return {"ok": True}
    append_event({"type": message_type or "unknown", "message_summary": summarized_vapi_message(message)})
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
    if host not in {"127.0.0.1", "localhost", "::1"} and not truthy_env(env, "HERMES_VAPI_ALLOW_NON_LOOPBACK_BIND"):
        raise RuntimeError("non-loopback bridge bind requires HERMES_VAPI_ALLOW_NON_LOOPBACK_BIND=true")
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
    call.add_argument("--purpose", default="supervised_test")
    call.add_argument("--venue-name", default="")
    call.add_argument("--reservation-name", default="")
    call.add_argument("--party-size", default="")
    call.add_argument("--requested-time", default="")
    call.add_argument("--acceptable-window", default="")
    call.add_argument("--approval-artifact-id", default="")
    call.add_argument("--idempotency-key", default="")
    call.add_argument("--skip-webhook-preflight", action="store_true")
    call.add_argument("--skip-approval-gate", action="store_true")
    call.add_argument("--bypass-reason", default="")
    call_status = sub.add_parser("call-status")
    call_status.add_argument("--call-id", required=True)

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
            "allowed_numbers_count": len(parse_allowed_numbers(env)),
            "approval_gate_required": approval_gate_required(env),
            "max_duration_seconds": bounded_int_env(
                env,
                "HERMES_VAPI_MAX_DURATION_SECONDS",
                DEFAULT_MAX_CALL_DURATION_SECONDS,
                min_value=MIN_CALL_DURATION_SECONDS,
                max_value=MAX_CALL_DURATION_SECONDS,
            ),
            "silence_timeout_seconds": bounded_int_env(
                env,
                "HERMES_VAPI_SILENCE_TIMEOUT_SECONDS",
                DEFAULT_SILENCE_TIMEOUT_SECONDS,
                min_value=4,
                max_value=30,
            ),
        }
        report["public_webhook"] = public_webhook_preflight(env_file) if report["ok"] else {"ok": False}
        report["ok"] = bool(
            report["ok"]
            and report["public_webhook"].get("ok")
            and report["phone_number_id"] == "set"
            and report["allowed_numbers_count"] > 0
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("Hermes Vapi preflight " + ("OK" if report["ok"] else "BLOCKED"))
            for key, value in report["required"].items():
                print(f"- {key}: {value}")
            print(f"- HERMES_VAPI_PHONE_NUMBER_ID: {report['phone_number_id']}")
            print(f"- HERMES_VAPI_PHONE_NUMBER: {report['phone_number']}")
            print(f"- allowed numbers: {report['allowed_numbers_count']}")
            print(f"- approval gate required: {report['approval_gate_required']}")
            print(f"- maxDurationSeconds: {report['max_duration_seconds']}")
            print(f"- public webhook: {'ok' if report['public_webhook'].get('ok') else 'blocked'}")
        return 0 if report["ok"] else 2
    if args.cmd == "bridge":
        run_bridge(host=args.host, port=args.port, env_file=env_file)
        return 0
    if args.cmd == "call":
        bypass_requested = bool(args.skip_webhook_preflight or args.skip_approval_gate)
        if bypass_requested:
            if not truthy_env(env, "HERMES_VAPI_ALLOW_TEST_BYPASS"):
                parser.error("bypass flags require HERMES_VAPI_ALLOW_TEST_BYPASS=true")
            if not args.bypass_reason.strip():
                parser.error("--bypass-reason is required when using bypass flags")
        print(
            json.dumps(
                create_outbound_call(
                    args.to,
                    env_file,
                    profile=args.profile,
                    customer_name=args.customer_name,
                    purpose=args.purpose,
                    require_public_webhook=not args.skip_webhook_preflight,
                    approval_artifact_id=args.approval_artifact_id,
                    idempotency_key=args.idempotency_key,
                    require_approval_gate=not args.skip_approval_gate,
                    bypass_reason=args.bypass_reason,
                    venue_name=args.venue_name,
                    reservation_name=args.reservation_name,
                    party_size=args.party_size,
                    requested_time=args.requested_time,
                    acceptable_window=args.acceptable_window,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.cmd == "call-status":
        print(json.dumps(get_call_status(args.call_id, env_file), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
