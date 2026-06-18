"""Preflight comparison for Hermes phone-call voice stacks.

This module intentionally does not change LiveKit routing, dispatch rules, or
runtime services. It gives Hermes a redacted, deterministic way to compare the
current LiveKit stack against Vapi/OpenAI/ElevenLabs/xAI candidates before any
live cutover is attempted.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


RECOMMENDATION_SCOPE = "preflight_readiness_not_performance"
BASELINE_ROUTE_ID = "livekit-current-openai-tts"

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


@dataclass(frozen=True)
class LatencyBudget:
    """Target latency envelope for a route candidate."""

    stt_ms: int
    reasoning_ms: int
    tts_ms: int
    transport_ms: int
    total_ms: int
    target_label: str


@dataclass(frozen=True)
class VoiceQualityGate:
    """Manual live-test gate required before promoting a voice route."""

    gate_id: str
    description: str
    languages: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class VoiceRouteCandidate:
    """Comparable voice route with provider handles and promotion criteria."""

    route_id: str
    priority: int
    transport: str
    stt: str
    reasoning_bridge: str
    tts_or_realtime_voice: str
    required_handles: tuple[str, ...]
    optional_handles: tuple[str, ...] = ()
    latency_budget: LatencyBudget = field(
        default_factory=lambda: LatencyBudget(
            stt_ms=700,
            reasoning_ms=2500,
            tts_ms=1200,
            transport_ms=500,
            total_ms=4900,
            target_label="default_conversational",
        )
    )
    provider_versions: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


DEFAULT_QUALITY_GATES: tuple[VoiceQualityGate, ...] = (
    VoiceQualityGate(
        gate_id="english_naturalness",
        description="English speech sounds natural, not robotic, and keeps a concise assistant tone.",
        languages=("en",),
    ),
    VoiceQualityGate(
        gate_id="romanian_naturalness",
        description="Romanian speech has acceptable pronunciation, diacritics handling, and sentence rhythm.",
        languages=("ro",),
    ),
    VoiceQualityGate(
        gate_id="names_and_numbers",
        description="Names, phone numbers, dates, and short calculations are spoken accurately.",
        languages=("en", "ro"),
    ),
    VoiceQualityGate(
        gate_id="interruptions_and_barge_in",
        description="Caller interruption stops or redirects the assistant without runaway self-talk.",
        languages=("en", "ro"),
    ),
    VoiceQualityGate(
        gate_id="silence_handling",
        description="Short silence is tolerated; long silence is handled with one concise prompt or hangup policy.",
        languages=("en", "ro"),
    ),
    VoiceQualityGate(
        gate_id="code_switching",
        description="English/Romanian code-switching preserves intent and avoids translating when not asked.",
        languages=("en", "ro"),
    ),
)


def build_default_candidates() -> tuple[VoiceRouteCandidate, ...]:
    """Return the council-approved candidate set in test priority order."""

    return (
        VoiceRouteCandidate(
            route_id=BASELINE_ROUTE_ID,
            priority=0,
            transport="LiveKit SIP/WebRTC",
            stt="Deepgram nova-3",
            reasoning_bridge="Hermes Orchestrator/current brain bridge",
            tts_or_realtime_voice="OpenAI TTS, current production fallback",
            required_handles=(
                "LIVEKIT_URL",
                "LIVEKIT_API_KEY",
                "LIVEKIT_API_SECRET",
                "DEEPGRAM_API_KEY",
                "OPENAI_API_KEY",
            ),
            optional_handles=(
                "HERMES_LIVEKIT_HERMES_API_KEY",
                "HERMES_LIVEKIT_ORCHESTRATOR_API_KEY",
                "HERMES_LIVEKIT_OUTBOUND_TRUNK_ID",
            ),
            latency_budget=LatencyBudget(700, 3000, 1400, 600, 5700, "known_baseline"),
            provider_versions={
                "stt": "deepgram-nova-3",
                "tts": "openai-current-config",
                "transport": "livekit-cloud-sip",
            },
            notes=("Baseline and rollback path. Do not alter during comparison.",),
        ),
        VoiceRouteCandidate(
            route_id="vapi-gpt41-mini-elevenlabs-flash",
            priority=1,
            transport="Vapi phone transport",
            stt="Vapi managed STT or Deepgram",
            reasoning_bridge="Vapi custom LLM/webhook to Hermes plus GPT-4.1 mini class reasoning",
            tts_or_realtime_voice="ElevenLabs Flash v2.5",
            required_handles=("VAPI_API_KEY",),
            optional_handles=("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "HERMES_ORCHESTRATOR_WEBHOOK_URL"),
            latency_budget=LatencyBudget(500, 1800, 800, 450, 3550, "fast_vapi_candidate"),
            provider_versions={
                "reasoning": "gpt-4.1-mini-compatible",
                "tts": "elevenlabs-flash-v2.5",
                "transport": "vapi-telephony",
            },
            notes=("Friend production reference path; first non-baseline route to live-test.",),
        ),
        VoiceRouteCandidate(
            route_id="vapi-openai-realtime",
            priority=2,
            transport="Vapi phone transport",
            stt="OpenAI Realtime integrated ASR",
            reasoning_bridge="OpenAI Realtime with Hermes tool/webhook bridge",
            tts_or_realtime_voice="OpenAI Realtime voice",
            required_handles=("VAPI_API_KEY", "OPENAI_API_KEY"),
            optional_handles=("HERMES_ORCHESTRATOR_WEBHOOK_URL",),
            latency_budget=LatencyBudget(450, 1500, 450, 450, 2850, "realtime_single_provider"),
            provider_versions={"voice": "openai-realtime", "transport": "vapi-telephony"},
            notes=("Strong latency candidate; quality must be judged by live bilingual call.",),
        ),
        VoiceRouteCandidate(
            route_id="livekit-xai-realtime",
            priority=3,
            transport="LiveKit SIP/WebRTC",
            stt="xAI/Grok voice ASR or LiveKit compatible STT",
            reasoning_bridge="Hermes Orchestrator bridge plus xAI realtime voice model",
            tts_or_realtime_voice="xAI realtime voice",
            required_handles=("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "XAI_API_KEY"),
            optional_handles=("HERMES_LIVEKIT_ORCHESTRATOR_API_KEY",),
            latency_budget=LatencyBudget(550, 1700, 550, 600, 3400, "direct_livekit_xai"),
            provider_versions={"voice": "xai-realtime-voice", "transport": "livekit-cloud-sip"},
            notes=("Direct-control xAI path. Useful if Vapi adds overhead or limits tool routing.",),
        ),
        VoiceRouteCandidate(
            route_id="vapi-xai-grok-voice",
            priority=4,
            transport="Vapi phone transport",
            stt="Vapi or xAI voice ASR",
            reasoning_bridge="Vapi custom LLM/webhook to Hermes plus Grok voice model",
            tts_or_realtime_voice="xAI/Grok voice",
            required_handles=("VAPI_API_KEY", "XAI_API_KEY"),
            optional_handles=("HERMES_ORCHESTRATOR_WEBHOOK_URL",),
            latency_budget=LatencyBudget(550, 1800, 650, 450, 3450, "vapi_xai_candidate"),
            provider_versions={"voice": "xai-grok-voice", "transport": "vapi-telephony"},
            notes=("Test after Vapi baseline to isolate xAI quality from transport quality.",),
        ),
    )


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load simple KEY=VALUE env files without exposing values in reports."""

    env_path = Path(path).expanduser()
    loaded: dict[str, str] = {}
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
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        try:
            parts = shlex.split(value, comments=True, posix=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = value.strip().strip('"').strip("'")
        loaded[key] = value
    return loaded


def _has_handle(env: Mapping[str, str], handle: str) -> bool:
    return bool(str(env.get(handle, "")).strip())


def _redact_text(value: str) -> str:
    text = value
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def redact_structure(value: Any, key: str = "") -> Any:
    """Redact secret-looking keys and known secret-looking string values."""

    if _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact_structure(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_structure(item, key) for item in value]
    if isinstance(value, tuple):
        return [redact_structure(item, key) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _candidate_payload(candidate: VoiceRouteCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["recommendation_scope"] = RECOMMENDATION_SCOPE
    payload["voice_quality_gates"] = [asdict(gate) for gate in DEFAULT_QUALITY_GATES]
    return payload


def evaluate_routes(
    env: Mapping[str, str],
    *,
    candidates: Sequence[VoiceRouteCandidate] | None = None,
    benchmark_results: Mapping[str, Mapping[str, Any]] | None = None,
    tool_delegation_results: Mapping[str, Mapping[str, Any]] | None = None,
    redaction_scan_passed: bool = False,
) -> dict[str, Any]:
    """Build a redacted readiness report for all route candidates."""

    route_candidates = tuple(candidates or build_default_candidates())
    benchmark_results = benchmark_results or {}
    tool_delegation_results = tool_delegation_results or {}
    routes: list[dict[str, Any]] = []

    for candidate in route_candidates:
        missing_required = [h for h in candidate.required_handles if not _has_handle(env, h)]
        present_required = [h for h in candidate.required_handles if _has_handle(env, h)]
        missing_optional = [h for h in candidate.optional_handles if not _has_handle(env, h)]
        ready = not missing_required
        benchmark = dict(benchmark_results.get(candidate.route_id, {}))
        delegation = dict(tool_delegation_results.get(candidate.route_id, {}))
        live_benchmark_ok = bool(benchmark.get("success"))
        tool_delegation_ok = bool(delegation.get("success"))

        blockers: list[str] = []
        if missing_required:
            blockers.append("missing_required_handles")
        if not live_benchmark_ok:
            blockers.append("missing_successful_live_inbound_outbound_benchmark")
        if not tool_delegation_ok:
            blockers.append("missing_verified_hermes_orchestrator_tool_delegation")
        if not redaction_scan_passed:
            blockers.append("missing_redaction_scan")
        if candidate.route_id != BASELINE_ROUTE_ID:
            blockers.append("rollback_note_required_keep_livekit_available")

        route = _candidate_payload(candidate)
        route.update(
            {
                "ready": ready,
                "present_required_handles": present_required,
                "missing_required_handles": missing_required,
                "missing_optional_handles": missing_optional,
                "readiness_status": "ready" if ready else "blocked",
                "benchmark_status": "pass" if live_benchmark_ok else "not_run",
                "benchmark_summary": redact_structure(benchmark),
                "tool_delegation_status": "pass" if tool_delegation_ok else "not_verified",
                "tool_delegation_summary": redact_structure(delegation),
                "promotion_eligible": ready and live_benchmark_ok and tool_delegation_ok and redaction_scan_passed,
                "promotion_blockers": blockers,
            }
        )
        routes.append(route)

    ranked = sorted(routes, key=lambda item: (not item["ready"], item["priority"]))
    for index, route in enumerate(ranked, start=1):
        route["readiness_rank"] = index

    routes_by_id = {route["route_id"]: route for route in ranked}
    ordered_routes = [routes_by_id[candidate.route_id] for candidate in route_candidates]
    production_cutover_allowed = any(
        route["route_id"] != BASELINE_ROUTE_ID and route["promotion_eligible"]
        for route in ordered_routes
    )
    top_ready = next((route for route in ranked if route["ready"]), None)

    report = {
        "schema_version": "2026-06-18.voice_stack_compare.v1",
        "scope": RECOMMENDATION_SCOPE,
        "production_cutover_allowed": production_cutover_allowed,
        "recommendation": {
            "label": "preflight_readiness_rank",
            "top_ready_route_id": top_ready["route_id"] if top_ready else None,
            "ranked_route_ids": [route["route_id"] for route in ranked],
            "live_test_priority_route_ids": [candidate.route_id for candidate in route_candidates],
            "livekit_baseline_route_id": BASELINE_ROUTE_ID,
            "cutover_blockers": []
            if production_cutover_allowed
            else [
                "complete_live_inbound_outbound_benchmark",
                "verify_hermes_orchestrator_tool_delegation",
                "run_redaction_scan",
                "document_livekit_rollback_path",
            ],
            "notes": [
                "This is a readiness report, not a latency or voice-quality verdict.",
                "LiveKit remains the production route until live tests pass.",
            ],
        },
        "routes": ordered_routes,
    }
    return redact_structure(report)


def render_text_report(report: Mapping[str, Any]) -> str:
    lines = [
        "Hermes voice stack comparison",
        f"Scope: {report.get('scope')}",
        f"Production cutover allowed: {report.get('production_cutover_allowed')}",
        f"Top ready route: {report.get('recommendation', {}).get('top_ready_route_id')}",
        "Routes:",
    ]
    routes = sorted(report.get("routes", []), key=lambda item: item.get("readiness_rank", 999))
    for route in routes:
        missing = ", ".join(route.get("missing_required_handles", [])) or "none"
        lines.append(
            "- {rank}. {route_id}: {status}; benchmark={benchmark}; delegation={delegation}; missing={missing}".format(
                rank=route.get("readiness_rank"),
                route_id=route.get("route_id"),
                status=route.get("readiness_status"),
                benchmark=route.get("benchmark_status"),
                delegation=route.get("tool_delegation_status"),
                missing=missing,
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Hermes voice stack readiness.")
    parser.add_argument("--env-file", action="append", default=[], help="KEY=VALUE env file to include")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    env: dict[str, str] = {}
    for env_file in args.env_file:
        env.update(load_env_file(env_file))
    report = evaluate_routes(env)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
