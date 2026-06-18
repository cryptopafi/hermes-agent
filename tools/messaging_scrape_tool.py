"""VPS-native messaging scrape tool for Hermes.

The tool wraps ``~/.nexus/scripts/hermes-messaging-intel-vps.sh``. That script
uses VPS-local Telegram MTProto state and server-side WhatsApp/Radar sources;
it never reads MacM4-only WhatsApp Desktop databases.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.registry import registry


_DEFAULT_SCRIPT = Path.home() / ".nexus/scripts/hermes-messaging-intel-vps.sh"
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_VALID_ACTIONS = {"status", "run"}
_VALID_MODES = {"dry_run", "scrape", "scrape_and_wiki"}
_VALID_PLATFORMS = {"all", "telegram", "whatsapp"}


def _script_path() -> Path:
    configured = os.getenv("HERMES_MESSAGING_SCRAPE_SCRIPT", "").strip()
    return Path(configured).expanduser() if configured else _DEFAULT_SCRIPT


def _check_requirements() -> bool:
    script = _script_path()
    return script.exists() and os.access(script, os.X_OK)


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False, indent=2)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_iso(name: str, value: Any) -> tuple[str | None, str | None]:
    if value is None or str(value).strip() == "":
        return None, None
    text = str(value).strip()
    if not _ISO_RE.match(text):
        return None, f"{name} must be an ISO timestamp with timezone, e.g. 2026-06-18T00:00:00Z"
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, f"{name} is not a valid ISO timestamp"
    return text, None


def _optional_int(name: str, value: Any, *, minimum: int, maximum: int, default: int | None = None) -> tuple[int | None, str | None]:
    if value is None or str(value).strip() == "":
        return default, None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, f"{name} must be an integer"
    if parsed < minimum or parsed > maximum:
        return None, f"{name} must be between {minimum} and {maximum}"
    return parsed, None


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, dict[str, Any]]] = []
    for idx, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((end, value))
    if not candidates:
        return {}
    return max(candidates, key=lambda item: item[0])[1]


def _run_script(cmd: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
        env={**os.environ, "PYTHONWARNINGS": os.getenv("PYTHONWARNINGS", "ignore:urllib3 v2 only supports OpenSSL")},
    )


def _status_payload(timeout_seconds: int = 60) -> str:
    script = _script_path()
    payload: dict[str, Any] = {
        "success": True,
        "script": str(script),
        "script_available": _check_requirements(),
    }
    if not _check_requirements():
        return json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        proc = _run_script([str(script), "doctor"], timeout_seconds)
    except subprocess.TimeoutExpired:
        payload["doctor_error"] = f"doctor timed out after {timeout_seconds}s"
        return json.dumps(payload, ensure_ascii=False, indent=2)
    payload["doctor"] = _json_from_stdout(proc.stdout)
    payload["returncode"] = proc.returncode
    if proc.returncode != 0:
        payload["stderr_tail"] = proc.stderr[-2000:]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _run_payload(args: dict[str, Any]) -> str:
    script = _script_path()
    if not _check_requirements():
        return _error(f"Messaging scrape script is unavailable or not executable: {script}")

    mode = str(args.get("mode") or "dry_run").strip().lower()
    if mode not in _VALID_MODES:
        return _error("mode must be one of: dry_run, scrape, scrape_and_wiki")
    platform = str(args.get("platform") or "all").strip().lower()
    if platform not in _VALID_PLATFORMS:
        return _error("platform must be one of: all, telegram, whatsapp")
    since, err = _optional_iso("since", args.get("since"))
    if err:
        return _error(err)
    until, err = _optional_iso("until", args.get("until"))
    if err:
        return _error(err)
    max_dialogs, err = _optional_int("max_dialogs", args.get("max_dialogs"), minimum=1, maximum=5000)
    if err:
        return _error(err)
    dialog_timeout, err = _optional_int("dialog_timeout", args.get("dialog_timeout"), minimum=5, maximum=300, default=25)
    if err:
        return _error(err)
    timeout_seconds, err = _optional_int("timeout_seconds", args.get("timeout_seconds"), minimum=30, maximum=7200, default=1200)
    if err:
        return _error(err)

    cmd = [str(script), "run"]
    if mode == "dry_run":
        cmd.append("--dry-run")
    if mode != "scrape_and_wiki":
        cmd.append("--skip-wiki")
    if platform == "telegram":
        cmd.append("--skip-cortex")
    elif platform == "whatsapp":
        cmd.append("--skip-telegram")
        cmd.extend(["--cortex-platform", "whatsapp"])
    if since:
        cmd.extend(["--since", since])
    if until:
        cmd.extend(["--until", until])
    if max_dialogs is not None:
        cmd.extend(["--max-dialogs", str(max_dialogs)])
    if dialog_timeout is not None:
        cmd.extend(["--dialog-timeout", str(dialog_timeout)])

    try:
        proc = _run_script(cmd, timeout_seconds or 1200)
    except subprocess.TimeoutExpired:
        return _error(f"Messaging scrape timed out after {timeout_seconds}s")

    return json.dumps(
        {
            "success": proc.returncode == 0,
            "mode": mode,
            "platform": platform,
            "returncode": proc.returncode,
            "result": _json_from_stdout(proc.stdout),
            **({"stderr_tail": proc.stderr[-2000:], "stdout_tail": proc.stdout[-2000:]} if proc.returncode != 0 else {}),
        },
        ensure_ascii=False,
        indent=2,
    )


def messaging_scrape(args: dict[str, Any], **_: Any) -> str:
    """Inspect or run the VPS-native WhatsApp/Telegram messaging intel pipeline."""
    action = str(args.get("action") or "status").strip().lower()
    if action not in _VALID_ACTIONS:
        return _error("action must be one of: status, run")
    if action == "status":
        timeout_seconds, err = _optional_int("timeout_seconds", args.get("timeout_seconds"), minimum=10, maximum=300, default=60)
        if err:
            return _error(err)
        return _status_payload(timeout_seconds or 60)
    return _run_payload(args)


MESSAGING_SCRAPE_SCHEMA = {
    "name": "messaging_scrape",
    "description": (
        "Inspect or run Hermes' VPS-native WhatsApp/Telegram messaging intel pipeline. "
        "Telegram uses the VPS MTProto user session directly. WhatsApp requires a paired "
        "VPS WhatsApp bridge; until paired, the tool can only use server-side Cortex/Radar "
        "private_messages fallback records. Use action='status' first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "run"], "description": "Inspect capability or execute a scrape."},
            "mode": {"type": "string", "enum": ["dry_run", "scrape", "scrape_and_wiki"], "description": "dry_run avoids state/wiki writes; scrape writes dedupe/export state only; scrape_and_wiki writes LLM-Wiki notes."},
            "platform": {"type": "string", "enum": ["all", "telegram", "whatsapp"], "description": "Limit source scope. whatsapp currently means Cortex/Radar fallback unless the bridge is paired."},
            "since": {"type": "string", "description": "Optional ISO timestamp with timezone."},
            "until": {"type": "string", "description": "Optional ISO timestamp with timezone."},
            "max_dialogs": {"type": "integer", "description": "Optional Telegram dialog scan cap for smoke tests."},
            "dialog_timeout": {"type": "integer", "description": "Per-dialog Telegram timeout in seconds."},
            "timeout_seconds": {"type": "integer", "description": "Overall subprocess timeout."},
        },
        "required": ["action"],
    },
}


registry.register(
    name="messaging_scrape",
    toolset="messaging_scrape",
    schema=MESSAGING_SCRAPE_SCHEMA,
    handler=messaging_scrape,
    check_fn=_check_requirements,
    emoji="",
)
