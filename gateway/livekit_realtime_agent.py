"""Hermes LiveKit realtime voice worker.

This module is intentionally separate from ``gateway.run``. Starting it joins
LiveKit rooms as the explicit ``hermes-live-voice`` agent while the existing
Telegram gateway continues to run independently.
"""

from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Mapping, cast
from urllib.parse import urlparse

import httpx
from gateway.livekit_voice import (
    DEFAULT_REALTIME_INSTRUCTIONS,
    _LEDGER_PATH_ALLOWED_PREFIXES,
    LiveKitVoiceConfig,
    load_livekit_config,
    validate_agent_name,
)

logger = logging.getLogger(__name__)

try:
    from livekit import agents, rtc  # type: ignore
    from livekit.agents import Agent, AgentSession, RoomInputOptions, RoomOutputOptions, function_tool  # type: ignore
except Exception:  # pragma: no cover - import checked by build_server
    agents = None  # type: ignore[assignment]
    rtc = None  # type: ignore[assignment]
    AgentSession = None  # type: ignore[assignment]
    RoomInputOptions = None  # type: ignore[assignment]
    RoomOutputOptions = None  # type: ignore[assignment]

    class _FallbackAgent:
        def __init__(self, *, instructions: str, **_: Any) -> None:
            self._instructions = instructions
            self._tools = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                info = getattr(attr, "__livekit_tool_info", None)
                if info is not None:
                    self._tools.append(SimpleNamespace(_info=info))

    def function_tool(f=None, *, name=None, description=None, **_):  # type: ignore[no-redef]
        def deco(fn):
            setattr(
                fn,
                "__livekit_tool_info",
                SimpleNamespace(name=name or fn.__name__, description=description),
            )
            return fn

        return deco(f) if f is not None else deco

    Agent = _FallbackAgent  # type: ignore[assignment,misc]


HERMES_BRAIN_UNAVAILABLE_MESSAGE = (
    "The deeper assistant brain is unavailable right now. Continue with the fast voice answer."
)
HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE = (
    "The task handoff service is unavailable right now. Tell the caller the task was not submitted."
)
_MAX_BRAIN_QUESTION_CHARS = 4000
_MAX_ORCHESTRATOR_TASK_CHARS = 8000
_MAX_BRAIN_RESPONSE_CHARS = 1200
_MAX_ORCHESTRATOR_ACK_CHARS = 600
_MAX_STRUCTURED_OUTCOME_FIELD_CHARS = 240
_MAX_STRUCTURED_OUTCOME_LIST_ITEMS = 8
_DEFAULT_CONCIERGE_LEDGER_PATH = Path(
    "/home/pafi/.hermes/profiles/hermes-concierge/data/external-contact-ledger-live.jsonl"
)
_PROFILE_HINT_RE = re.compile(r"^[a-zA-Z0-9_-]{0,64}$")
_SENSITIVE_KV_RESPONSE_RE = re.compile(
    r"(?i)(api[_ -]?key|authorization|bearer|password|secret|token)\s*[:=]\s*\S+"
)
_BEARER_RESPONSE_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{16,}")
_JWT_RESPONSE_RE = re.compile(
    r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"
)
_PROVIDER_TOKEN_RESPONSE_RE = re.compile(
    r"\b(?:sk-|xai-|AIza)[a-zA-Z0-9_-]{16,}\b"
)


def _room_name(room: Any) -> str:
    name = str(getattr(room, "name", "") or "unknown")
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", name)[:96]


def _log_call_event(event: str, **fields: Any) -> None:
    safe_fields = " ".join(
        f"{key}={str(value).replace(' ', '_')[:160]}"
        for key, value in sorted(fields.items())
        if value is not None
    )
    logger.info("hermes_call event=%s %s", event, safe_fields)


def _parse_metadata(raw: Any) -> dict[str, str]:
    """Return a small string metadata dict from LiveKit JSON metadata."""
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        data = parsed if isinstance(parsed, dict) else {}
    else:
        return {}
    return {
        str(key): str(value)
        for key, value in data.items()
        if isinstance(key, str) and value is not None
    }


def _extract_call_metadata(ctx: Any) -> dict[str, str]:
    """Extract room/job metadata that describes inbound vs outbound calls."""
    merged: dict[str, str] = {}
    room = getattr(ctx, "room", None)
    for raw in (
        getattr(room, "metadata", ""),
        getattr(getattr(getattr(ctx, "_info", None), "accept_arguments", None), "metadata", ""),
        getattr(getattr(ctx, "job", None), "metadata", ""),
        getattr(getattr(getattr(ctx, "_info", None), "job", None), "metadata", ""),
    ):
        merged.update(_parse_metadata(raw))

    remote_participants = getattr(room, "remote_participants", None)
    if isinstance(remote_participants, dict):
        participants = remote_participants.values()
    elif remote_participants:
        participants = remote_participants
    else:
        participants = ()
    for participant in participants:
        merged.update(_parse_metadata(getattr(participant, "metadata", "")))
        attributes = getattr(participant, "attributes", None)
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                if key == "hermes.profile":
                    merged.setdefault("call_profile", str(value))
                elif key == "hermes.purpose":
                    merged.setdefault("purpose", str(value))
                elif key == "hermes.max_duration_seconds":
                    merged.setdefault("max_duration_seconds", str(value))
    return merged


def _build_call_context(metadata: dict[str, str]) -> str:
    """Build call-specific instructions from safe LiveKit metadata."""
    mode = metadata.get("mode", "").strip()
    if mode != "sip-outbound":
        return ""
    profile = metadata.get("call_profile", "").strip() or "pa"
    purpose = metadata.get("purpose", "").strip().rstrip(".!?")
    restaurant_address = metadata.get("restaurant_address", "").strip()
    max_duration = metadata.get("max_duration_seconds", "").strip()
    if profile == "concierge":
        parts = [
            "Call context: this is an outbound phone call initiated by Leonardo.",
            "Use the Warm Concierge booking package by default: introduce yourself only as Leonardo, ask warmly for the requested booking, and say clock times in spoken words such as three PM.",
            "Do not use the owner or reservation guest name in the opening sentence; use the reservation name only if the venue asks or needs it to complete the booking.",
            "Use US English for the venue call unless the callee explicitly asks to switch language.",
            "Do not mix languages inside the same call unless the callee switches first.",
            "Do not say or imply that the caller called you; you called them.",
            f"Active outbound profile: {profile}.",
        ]
    else:
        parts = [
            "Call context: this is an outbound phone call initiated by Leonardo.",
            "Do not say or imply that the caller called you; you called them.",
            f"Active outbound profile: {profile}.",
        ]
    if purpose:
        parts.append(f"Call purpose: {purpose}.")
    if restaurant_address:
        parts.append(
            f"Restaurant address/context: {restaurant_address}. For Concierge venue calls, use US English unless the callee explicitly requests another language."
        )
    if max_duration:
        parts.append(f"Maximum planned call duration: {max_duration} seconds.")
    if profile == "concierge":
        parts.append(
            "Start in English with a brief Leonardo-only identification, then follow the call purpose."
        )
    else:
        parts.append(
            "Start by briefly identifying that this is Leonardo calling, then follow the call purpose."
        )
    if profile == "concierge":
        confirmation_phone = metadata.get("confirmation_phone", "").strip()
        confirmation_email = metadata.get("confirmation_email", "").strip()
        parts.append(
            "Outcome retention policy: do not store raw transcript except for explicitly approved test numbers. Whenever the venue confirms, rejects, offers alternative times, or gives a next action, call record_concierge_call_outcome with only structured fields. Call it before ending the call if any reservation-relevant outcome was learned."
        )
        parts.append(
            "Reservation decision policy: if the requested time is unavailable and the venue offers alternatives, do not accept an alternative automatically unless the call purpose gives an explicit acceptable window; record status alternative_time_offered_owner_decision_required and ask the owner after the call."
        )
        if confirmation_phone or confirmation_email:
            contact_bits = []
            if confirmation_phone:
                contact_bits.append(f"dedicated confirmation phone: {confirmation_phone}")
            if confirmation_email:
                contact_bits.append(f"email: {confirmation_email}")
            parts.append(
                "Reservation confirmation policy: before closing any successful reservation call, ask the venue to send written confirmation by email or WhatsApp; proactively provide Leonardo/Concierge's dedicated operational confirmation contact details: "
                + "; ".join(contact_bits)
                + ". These are Leonardo/Concierge operational contacts, not the guest's personal contacts. Never provide the owner's personal phone number or personal email to a venue. Record which confirmation contact was provided."
            )
        else:
            parts.append(
                "Reservation confirmation policy: before closing any successful reservation call, ask whether the venue can confirm by email. If no Leonardo/Concierge dedicated contact detail is present in the call purpose, config, or metadata, do not invent one and do not provide personal contact details; record owner_followup_required with next_action to configure/send Leonardo's confirmation contact. Do not mention any unverified messaging channel."
            )
    return " ".join(parts)


def _outbound_initial_reply_text(metadata: dict[str, str]) -> str:
    """Return deterministic first audio for SIP outbound calls.

    Gemini realtime generation can create transcript/context without audibly
    playing the first turn before the callee speaks. For outbound calls the
    first utterance is safety-critical, so use session.say(text) instead of
    asking the realtime model to generate it.
    """
    if metadata.get("mode", "").strip() != "sip-outbound":
        return ""
    profile = metadata.get("call_profile", "").strip() or "pa"
    purpose = metadata.get("purpose", "").strip()
    restaurant_address = metadata.get("restaurant_address", "").strip()
    if profile == "concierge":
        return "Hello, this is Leonardo."
    return "Hello, this is Leonardo."


def _outbound_initial_reply_instructions(metadata: dict[str, str]) -> str:
    """Return context for the mandatory first SIP outbound utterance."""
    text = _outbound_initial_reply_text(metadata)
    if not text:
        return ""
    purpose = metadata.get("purpose", "").strip().rstrip(".!?")
    restaurant_address = metadata.get("restaurant_address", "").strip()
    instructions = [
        "This is an outbound phone call: you called them; they did not call you.",
        f"The first spoken sentence is deterministic and already scheduled: {text!r}.",
        "After that single opening sentence, stop speaking and wait for the callee.",
    ]
    if restaurant_address:
        instructions.append(f"Venue/address signal: {restaurant_address}.")
    if purpose:
        instructions.append(f"Call goal for later turns, not the whole opening sentence: {purpose}.")
    return " ".join(instructions)


async def _wait_for_outbound_participant_ready(
    ctx: Any,
    metadata: dict[str, str],
    *,
    room_name: str,
    settle_seconds: float = 1.2,
    track_timeout_seconds: float = 12.0,
) -> str:
    """Wait until the SIP callee exists and has an inbound audio track.

    LiveKit creates the SIP participant before the callee's answered media is
    necessarily flowing. Speaking after participant creation alone can play the
    greeting into silence. Wait for a microphone/audio track publication before
    the deterministic first `session.say()`.
    """
    if metadata.get("mode") != "sip-outbound":
        return ""
    wait_for_participant = getattr(ctx, "wait_for_participant", None)
    participant_identity = metadata.get("participant_identity", "").strip()
    participant = None
    if callable(wait_for_participant):
        try:
            kwargs = {"identity": participant_identity} if participant_identity else {}
            result = wait_for_participant(**kwargs)
            participant = await cast(Awaitable[Any], result) if hasattr(result, "__await__") else result
        except TypeError:
            result = wait_for_participant()
            participant = await cast(Awaitable[Any], result) if hasattr(result, "__await__") else result
        _log_call_event(
            "outbound_participant_ready",
            room=room_name,
            participant=getattr(participant, "identity", participant_identity or "unknown"),
        )
    if participant is not None and track_timeout_seconds > 0:
        deadline = time.perf_counter() + track_timeout_seconds
        while time.perf_counter() < deadline:
            publications = getattr(participant, "track_publications", {}) or {}
            values = publications.values() if hasattr(publications, "values") else publications
            for publication in values:
                source = str(getattr(publication, "source", ""))
                kind = str(getattr(publication, "kind", ""))
                track = getattr(publication, "track", None)
                subscribed = getattr(publication, "subscribed", True)
                if (
                    track is not None
                    and subscribed
                    and ("MICROPHONE" in source or "AUDIO" in kind or kind.endswith("1"))
                ):
                    _log_call_event(
                        "outbound_audio_track_ready",
                        room=room_name,
                        participant=getattr(participant, "identity", participant_identity or "unknown"),
                        source=source,
                        kind=kind,
                    )
                    break
            else:
                await asyncio.sleep(0.2)
                continue
            break
        else:
            _log_call_event(
                "outbound_audio_track_timeout",
                room=room_name,
                participant=getattr(participant, "identity", participant_identity or "unknown"),
                timeout_ms=int(track_timeout_seconds * 1000),
            )
    if settle_seconds > 0:
        await asyncio.sleep(settle_seconds)
        _log_call_event(
            "outbound_media_settled",
            room=room_name,
            settle_ms=int(settle_seconds * 1000),
        )
    return str(getattr(participant, "identity", participant_identity or ""))


def _install_session_telemetry(
    session: Any,
    *,
    config: LiveKitVoiceConfig,
    room_name: str,
    started_at: float,
    call_metadata: Mapping[str, str] | None = None,
    on_final_user_transcript: Any | None = None,
) -> None:
    """Install redacted LiveKit session telemetry for benchmarkable phone calls."""
    metadata = dict(call_metadata or {})
    final_transcript_count = 0
    final_transcript_snippets: list[str] = []
    audit_transcript_events: list[dict[str, Any]] = []
    audit_turn_index = 0
    allow_full_audit_transcript = metadata.get("target_number", "") in {"+407" + "58400900"}

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started_at) * 1000)

    def record_audit_transcript_event(*, speaker: str, text: str, source: str, turn: int | None = None) -> None:
        """Record full transcript turns for explicitly approved test numbers only."""
        nonlocal audit_turn_index
        clean_text = str(text or "").strip()
        if not allow_full_audit_transcript or not clean_text:
            return
        audit_turn_index += 1
        audit_transcript_events.append(
            {
                "index": audit_turn_index,
                "turn": turn if turn is not None else audit_turn_index,
                "speaker": speaker,
                "source": source,
                "elapsed_ms": elapsed_ms(),
                "timestamp": datetime.now(UTC).isoformat(),
                "text": clean_text,
            }
        )

    try:
        setattr(session, "_hermes_record_audit_transcript_event", record_audit_transcript_event)
    except Exception:
        pass

    def on_close(event: Any) -> None:
        error = getattr(event, "error", None)
        reason = getattr(event, "reason", None)
        clean_reason = getattr(reason, "value", reason)
        _log_call_event(
            "close",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            reason=clean_reason,
            error=error.__class__.__name__ if error else "none",
        )
        if metadata.get("mode") == "sip-outbound" and metadata.get("call_profile") == "concierge":
            approved_ledger_path = _approved_concierge_ledger_path(metadata)
            should_write_fallback = True
            if approved_ledger_path and approved_ledger_path.exists():
                try:
                    should_write_fallback = approved_ledger_path.stat().st_size == 0
                except OSError:
                    should_write_fallback = True
            if should_write_fallback:
                transcript_summary = " ".join(final_transcript_snippets).lower()
                audit_summary = " ".join(
                    str(event.get("text") or "") for event in audit_transcript_events
                ).lower()
                purpose = metadata.get("purpose", "")
                requested_time = metadata.get("requested_time", "") or ("15:00" if "15:00" in purpose or "ora 15" in purpose else "")
                reservation_name = metadata.get("reservation_name", "") or ("Bogdan Roșu" if "Bogdan" in purpose or "Roșu" in purpose else "")
                party_size = metadata.get("party_size", "") or ("2" if "2" in purpose or "două" in purpose.lower() else "")
                clear_reject = any(token in transcript_summary for token in ("nu avem", "nu se poate", "complet", "închis", "inchis", "nu este disponibil"))
                clear_payment = any(token in transcript_summary for token in ("avans", "card", "plată", "plata", "garanție", "garantie", "depozit"))
                clear_confirm = any(token in transcript_summary for token in ("confirm", "rezervarea este", "am notat", "vă așteptăm", "va asteptam", "este rezervat"))
                owner_approval_close = "cannot change the time without checking with bogdan" in audit_summary
                if clear_payment:
                    status = "payment_or_deposit_requested_owner_approval_required"
                    next_action = "Venue appears to have requested payment/deposit/card details; owner approval required before proceeding."
                elif owner_approval_close:
                    status = "reservation_not_booked_alternative_requires_owner_approval"
                    next_action = "Requested reservation was not confirmed. Venue offered/indicated an alternative or no exact availability; owner approval required before changing the requested time."
                elif clear_reject:
                    status = "reservation_unavailable_or_rejected"
                    next_action = "Venue appears unable to satisfy the requested reservation; owner/operator follow-up required."
                elif clear_confirm:
                    status = "reservation_confirmed_unverified_from_transcript"
                    next_action = "Call transcript signals confirmation, but no structured outcome tool fired; verify before relying on reservation."
                else:
                    status = "owner_followup_required"
                    next_action = "Call connected and closed, but no reliable structured venue outcome was captured; owner/operator follow-up required."
                fallback = {
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "status": status,
                    "requested_time": requested_time,
                    "reservation_name": reservation_name,
                    "party_size": party_size,
                    "offered_times": [],
                    "payment_or_deposit_requested": "yes" if clear_payment else "unknown",
                    "next_action": next_action,
                    "notes_summary": "Fallback close record. Full audit transcript stored for approved test numbers only. Final transcript events observed: " + str(final_transcript_count),
                    "room": room_name,
                    "task_id": metadata.get("task_id", ""),
                    "idempotency_key": metadata.get("idempotency_key", ""),
                    "close_reason": str(clean_reason or ""),
                    "error": error.__class__.__name__ if error else "none",
                }
                if allow_full_audit_transcript:
                    fallback["raw_transcript_storage"] = "approved_test_number_full_audit"
                    fallback["raw_transcript_mode"] = "full_turn_by_turn_test_number_audit"
                    fallback["raw_transcript_final_user_events"] = final_transcript_snippets
                    fallback["audit_transcript_events"] = audit_transcript_events
                    fallback["restaurant_flow_audit"] = audit_restaurant_transcript(audit_transcript_events)
                try:
                    written_path = append_concierge_call_outcome(fallback, ledger_path=approved_ledger_path)
                    _log_call_event(
                        "concierge_outcome_fallback_recorded",
                        room=room_name,
                        ledger_path=str(written_path),
                        transcript_events=final_transcript_count,
                    )
                except Exception as exc:
                    _log_call_event(
                        "concierge_outcome_fallback_failed",
                        room=room_name,
                        error=exc.__class__.__name__,
                    )

    def on_agent_state(event: Any) -> None:
        _log_call_event(
            "agent_state",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            old=getattr(event, "old_state", None),
            new=getattr(event, "new_state", None),
        )

    def on_user_state(event: Any) -> None:
        _log_call_event(
            "user_state",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            old=getattr(event, "old_state", None),
            new=getattr(event, "new_state", None),
        )

    def on_transcript(event: Any) -> None:
        nonlocal final_transcript_count
        transcript = str(getattr(event, "transcript", "") or "")
        is_final = getattr(event, "is_final", None)
        _log_call_event(
            "transcript",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            final=is_final,
            chars=len(transcript),
        )
        if is_final and transcript.strip():
            final_transcript_count += 1
            final_transcript_snippets.append(transcript.strip())
            record_audit_transcript_event(
                speaker="callee",
                text=transcript,
                source="stt_final",
            )
        if is_final and transcript.strip() and on_final_user_transcript is not None:
            try:
                on_final_user_transcript(transcript)
            except Exception as exc:
                _log_call_event(
                    "transcript_callback_failed",
                    elapsed_ms=elapsed_ms(),
                    room=room_name,
                    error=exc.__class__.__name__,
                )

    def on_conversation_item(event: Any) -> None:
        item = getattr(event, "item", None)
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        chars = len(str(content or ""))
        if role == "assistant" and content:
            record_audit_transcript_event(
                speaker="assistant",
                text=str(content),
                source="conversation_item_added",
            )
        _log_call_event(
            "conversation_item",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            role=role,
            chars=chars,
        )

    for event_name, callback in (
        ("close", on_close),
        ("agent_state_changed", on_agent_state),
        ("user_state_changed", on_user_state),
        ("user_input_transcribed", on_transcript),
        ("conversation_item_added", on_conversation_item),
    ):
        try:
            session.on(event_name, callback)
        except Exception as exc:
            _log_call_event(
                "telemetry_install_failed",
                room=room_name,
                provider=config.realtime_provider,
                event_name=event_name,
                error=exc.__class__.__name__,
            )


def build_assistant_instructions(
    config: LiveKitVoiceConfig | None = None,
    *,
    call_context: str = "",
) -> str:
    """Return the short voice-agent instruction block."""
    cfg = config or load_livekit_config()
    base = cfg.realtime_instructions or DEFAULT_REALTIME_INSTRUCTIONS
    lines = [
        base.strip(),
        "You are in a live phone call. Speak naturally, but every turn must be one short sentence only.",
        "Never speak two sentences in the same turn. Say one sentence, then stop and wait for the human.",
        "If the human asks for details, answer with only the next missing detail, then stop.",
        "For Concierge restaurant/vendor calls, speak US English unless the callee explicitly asks for another language; do not mix languages unless the callee switches first.",
        "Speak slowly and clearly. Keep short pauses between clauses. If the callee sounds confused, repeat once in simpler English.",
        "For Concierge outbound venue calls, use the Warm Concierge package: introduce yourself only as Leonardo, ask for the requested booking in warm concise English, answer venue questions, negotiate only within the stated call purpose, and use the reservation name only if the venue asks or needs it to complete the booking.",
        "Before closing a successful venue reservation call, ask the venue to send written confirmation by email or WhatsApp using the dedicated Concierge contact details; use only configured operational contacts and do not mention any unverified messaging channel.",
        "Use ask_hermes_brain only for quick deeper reasoning that can be answered inside this call.",
        "For requests that require LLM-Wiki, Cortex, research profiles, PA/concierge profiles, coding, file/tool access, implementation, or any task to be executed after the call, call submit_to_hermes_orchestrator.",
        "Do not claim that the phone voice model can access Wiki, Cortex, files, profiles, or tools directly.",
        "After submitting to the task handoff service, give only a concise spoken acknowledgement and do not wait for the long result.",
    ]
    if call_context.strip():
        lines.append(call_context.strip())
    return "\n".join(lines)


def _first_callee_reply_instructions(metadata: dict[str, str], transcript: str) -> str:
    """Build a focused LLM prompt after the callee answers the deterministic opening."""
    purpose = metadata.get("purpose", "").strip().rstrip(".!?")
    profile = metadata.get("call_profile", "").strip() or "pa"
    confirmation_phone = metadata.get("confirmation_phone", "").strip()
    confirmation_email = metadata.get("confirmation_email", "").strip()
    parts = [
        "The deterministic opening has already been spoken and the callee has now replied.",
        f"Callee reply transcript: {transcript.strip()[:500]!r}.",
        "Continue now as Leonardo in a live phone call; do not wait silently.",
    ]
    if purpose:
        parts.append(f"Complete this call purpose: {purpose}.")
    if profile == "concierge":
        purpose_lower = purpose.lower()
        non_booking_test = any(marker in purpose_lower for marker in ("no booking", "no reservation", "full-audit test", "audio test", "test call"))
        contact_bits = []
        if confirmation_phone:
            contact_bits.append(f"dedicated confirmation phone: {confirmation_phone}")
        if confirmation_email:
            contact_bits.append(f"email: {confirmation_email}")
        if non_booking_test:
            parts.extend(
                [
                    "This is a non-booking test call: do not ask for a reservation, do not invent party size/time, and do not ask for payment.",
                    "If the callee asks what this is, say it is a brief Leonardo voice-system test and politely close the call.",
                ]
            )
        else:
            parts.extend(
                [
                    "Use US English unless the callee explicitly asks for another language. Get the booking outcome: ask for missing reservation details, answer venue questions, and confirm the time and party size clearly; use the reservation name only if the venue asks or needs it.",
                    "Do not accept alternatives outside the call purpose unless explicitly allowed; tell the venue you will check with the owner.",
                ]
            )
        if contact_bits:
            parts.append(
                "Before ending after a successful reservation, ask the venue to send written confirmation and give these dedicated Concierge contacts: "
                + "; ".join(contact_bits)
                + "."
            )
        else:
            parts.append(
                "Before ending after a successful reservation, ask whether the venue can confirm by email, but do not invent or provide personal contact details."
            )
    parts.append("Speak one short natural sentence in the callee's language, then wait for their next reply.")
    return " ".join(parts)


def _clean_spoken_reply(text: str) -> str:
    """Clamp an LLM answer to one safe spoken sentence for phone playout."""
    clean = " ".join(str(text or "").strip().split())
    if not clean:
        return "Înțeleg. Continui imediat."
    clean = re.sub(r"^(Leonardo:|Assistant:|Răspuns:|Answer:)\s*", "", clean, flags=re.IGNORECASE)
    first_sentence = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0].strip()
    return (first_sentence or clean)[:220]


async def _concierge_llm_spoken_reply(
    *,
    metadata: dict[str, str],
    user_text: str,
    turn_index: int,
    config: LiveKitVoiceConfig,
) -> str:
    """Generate the next Concierge call sentence through Hermes brain."""
    purpose = metadata.get("purpose", "").strip()
    confirmation_phone = metadata.get("confirmation_phone", "").strip()
    confirmation_email = metadata.get("confirmation_email", "").strip()
    contact_bits = []
    if confirmation_phone:
        contact_bits.append(f"dedicated confirmation phone {confirmation_phone}")
    if confirmation_email:
        contact_bits.append(f"email {confirmation_email}")
    contact_clause = "; ".join(contact_bits) or "no dedicated confirmation contact is configured"
    payment_requested = any(
        token in user_text.lower()
        for token in ("payment", "deposit", "card", "guarantee", "advance", "plată", "plata", "avans", "garanție", "garantie")
    )
    payment_clause = (
        "The venue mentioned payment or card terms; say you will check with the owner first and stop there. "
        if payment_requested
        else ""
    )
    purpose_lower = purpose.lower()
    non_booking_test = any(marker in purpose_lower for marker in ("no booking", "no reservation", "full-audit test", "audio test", "test call"))
    if non_booking_test:
        task_clause = (
            "This is a non-booking test call. Do NOT ask for a reservation, do NOT invent party size or time, "
            "and do NOT ask for payment. If asked what this is, say it is a brief Leonardo voice-system test and politely close."
        )
    else:
        task_clause = (
            "If the venue accepts, confirm party size and time; use the reservation name only if the venue asks or needs it, and before closing ask for written confirmation using the dedicated Concierge contact details if configured."
        )
    prompt = (
        "You are Leonardo, speaking live to a restaurant/vendor. "
        "Use US English unless the callee explicitly asks for another language. Do not mix languages unless the callee switches first. "
        "Return ONLY the exact next spoken sentence, no labels, no notes. One short sentence only. "
        "Complete this call purpose: " + purpose + "\n"
        f"Callee just said: {user_text.strip()[:500]!r}\n"
        f"Turn number after opening: {turn_index + 1}. Dedicated Concierge confirmation contacts: {contact_clause}. "
        "Never provide the owner's personal phone/email and never say any owner nickname to the venue. "
        + payment_clause +
        task_clause
    )
    answer = await query_hermes_brain(prompt, config=config)
    if answer == HERMES_BRAIN_UNAVAILABLE_MESSAGE:
        return "I need to check with the owner before continuing, so I will follow up shortly."
    return _clean_spoken_reply(answer)


async def _say_and_wait(session: Any, text: str) -> None:
    """Speak text through the active session and wait for audio playout."""
    speech = session.say(text, allow_interruptions=True, add_to_chat_ctx=True)
    wait_for_playout = getattr(speech, "wait_for_playout", None)
    if callable(wait_for_playout):
        await cast(Awaitable[Any], wait_for_playout())
    else:
        await cast(Awaitable[Any], speech)


def _record_session_audit_event(
    session: Any,
    *,
    speaker: str,
    text: str,
    source: str,
    turn: int | None = None,
) -> None:
    """Record full audit transcript text when the session installed an approved test-number recorder."""
    recorder = getattr(session, "_hermes_record_audit_transcript_event", None)
    if not callable(recorder):
        return
    try:
        recorder(speaker=speaker, text=text, source=source, turn=turn)
    except Exception as exc:
        _log_call_event("audit_transcript_record_failed", source=source, error=exc.__class__.__name__)


def audit_restaurant_transcript(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a Concierge restaurant-call transcript against safety and naturalness gates."""
    normalized: list[tuple[str, str]] = []
    for event in events:
        speaker = str(event.get("speaker") or "").lower().strip()
        text = str(event.get("text") or "").strip()
        if speaker and text:
            normalized.append((speaker, text))
    assistant_texts = [text for speaker, text in normalized if speaker == "assistant"]
    callee_texts = [text for speaker, text in normalized if speaker == "callee"]
    joined_assistant = " ".join(assistant_texts).lower()
    joined_callee = " ".join(callee_texts).lower()
    first_speaker = normalized[0][0] if normalized else ""
    first_text = normalized[0][1].lower() if normalized else ""
    clean_fast_intro = (
        first_speaker == "assistant"
        and (
            (
                first_text.startswith("hi, this is leonardo calling for bogdan rosu")
                and "availability for two people at 3 pm today" in first_text
            )
            or (
                first_text.startswith("hello, this is leonardo")
                and "arrange a table for two people today at 3 pm" in first_text
            )
        )
    )
    requested_booking = any(
        phrase in joined_assistant
        for phrase in (
            "availability for two people at 3 pm today",
            "table for two today at 3 pm",
            "arrange a table for two people today at 3 pm",
        )
    )
    answered_questions = True
    question_expectations = [
        (("how many", "many persons", "many people", "persoane"), ("two people", "two persons")),
        (("what time", "which time", "ora", "hour"), ("3 pm",)),
        (("what name", "which name", "under what name", "nume"), ("bogdan",)),
        (("spell", "spelling", "letter by letter"), ("b as in bravo", "last name rosu")),
    ]
    for idx, (speaker, text) in enumerate(normalized):
        if speaker != "callee":
            continue
        callee_turn = text.lower()
        for triggers, required_phrases in question_expectations:
            if not any(trigger in callee_turn for trigger in triggers):
                continue
            next_assistant = ""
            for next_speaker, next_text in normalized[idx + 1 :]:
                if next_speaker == "assistant":
                    next_assistant = next_text.lower()
                    break
            if (
                not next_assistant
                or "please keep the reservation" in next_assistant
                or not any(required in next_assistant for required in required_phrases)
            ):
                answered_questions = False
    invented_confirmation = (
        ("please keep the reservation" in joined_assistant or "perfect, thank you" in joined_assistant)
        and not any(token in joined_callee for token in ("confirmed", "confirm", "available", "yes", "sure", "ok", "okay", "se poate", "avem"))
    )
    accepted_alternative = any(token in joined_callee for token in ("only", "doar", "numai")) and "please keep the reservation" in joined_assistant
    payment_boundary_clean = not any(
        token in joined_assistant
        for token in ("card number", "deposit", "advance payment", "payment link")
    )
    assistant_closed_before_late_callee = False
    for idx, (speaker, text) in enumerate(normalized[:-1]):
        if speaker != "assistant":
            continue
        text_lower = text.lower()
        if "have a good day" not in text_lower and "come back" not in text_lower:
            continue
        if any(
            next_speaker == "callee"
            and len(next_text.strip()) > 3
            and any(
                token in next_text.lower()
                for token in (
                    "no", "not", "unavailable", "unfortunately", "impossible", "only", "pm", "o'clock",
                    "spell", "name", "how many", "what time", "card", "payment", "deposit",
                )
            )
            and not (
                "come back" in text_lower
                and any(
                    token in next_text.lower()
                    for token in ("okay", "ok", "thank you", "thanks", "call me back", "bye")
                )
            )
            for next_speaker, next_text in normalized[idx + 1 :]
        ):
            assistant_closed_before_late_callee = True
            break
    closed_cleanly = (
        (not assistant_texts or not assistant_texts[-1].lower().startswith("thank you. i have noted your answer"))
        and not assistant_closed_before_late_callee
    )
    consecutive_assistant_turns = any(
        normalized[idx][0] == "assistant" and normalized[idx - 1][0] == "assistant"
        for idx in range(1, len(normalized))
    )
    checks = {
        "venue_first": first_speaker in {"callee", ""} or clean_fast_intro,
        "asked_exact_booking": requested_booking,
        "answered_venue_questions": answered_questions,
        "did_not_invent_confirmation": not invented_confirmation,
        "did_not_accept_alternatives": not accepted_alternative,
        "payment_boundary_clean": payment_boundary_clean,
        "closed_cleanly": closed_cleanly,
        "no_consecutive_assistant_turns": not consecutive_assistant_turns,
    }
    human_naturalness = 5
    if "communication system" in joined_assistant:
        human_naturalness -= 3
    if "i have noted your answer" in joined_assistant:
        human_naturalness -= 2
    if len(max(assistant_texts, key=len, default="")) > 180:
        human_naturalness -= 1
    if not requested_booking:
        human_naturalness -= 1
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "human_naturalness": max(1, min(5, human_naturalness)),
        "failure_reasons": [name for name, passed in checks.items() if not passed],
    }


async def _generate_reply_after_first_callee_transcript(
    session: Any,
    *,
    first_user_transcript_event: asyncio.Event,
    first_user_transcript: dict[str, str],
    final_user_transcripts: asyncio.Queue[str],
    call_metadata: dict[str, str],
    room_name: str,
    provider: str,
    config: LiveKitVoiceConfig,
    timeout_seconds: float = 18.0,
) -> None:
    """Run a bounded LLM-driven Concierge turn loop after deterministic SIP opening."""
    if call_metadata.get("mode", "").strip() != "sip-outbound":
        return
    if (call_metadata.get("call_profile", "").strip() or "pa") != "concierge":
        return
    _log_call_event(
        "outbound_waiting_for_first_callee_reply",
        room=room_name,
        provider=provider,
        timeout_ms=int(timeout_seconds * 1000),
    )
    try:
        await asyncio.wait_for(first_user_transcript_event.wait(), timeout=timeout_seconds)
    except TimeoutError:
        _log_call_event(
            "outbound_first_callee_reply_timeout",
            room=room_name,
            provider=provider,
        )
        return
    transcript = first_user_transcript.get("text", "").strip()
    if not transcript:
        return
    seen: set[str] = set()
    for turn_index in range(6):
        if turn_index == 0:
            user_text = transcript
            try:
                queued = final_user_transcripts.get_nowait()
                if queued.strip() and queued.strip() != user_text:
                    final_user_transcripts.put_nowait(queued)
            except asyncio.QueueEmpty:
                pass
        else:
            try:
                user_text = await asyncio.wait_for(final_user_transcripts.get(), timeout=22.0)
            except TimeoutError:
                _log_call_event(
                    "outbound_llm_turn_timeout",
                    room=room_name,
                    provider="hermes_brain",
                    turn=turn_index + 1,
                )
                break
        user_text = user_text.strip()
        if not user_text or user_text in seen:
            continue
        seen.add(user_text)
        _log_call_event(
            "outbound_llm_turn_start",
            room=room_name,
            provider="hermes_brain",
            trigger_chars=len(user_text),
            turn=turn_index + 1,
        )
        reply = await _concierge_llm_spoken_reply(
            metadata=call_metadata,
            user_text=user_text,
            turn_index=turn_index,
            config=config,
        )
        _log_call_event(
            "outbound_llm_turn_say_start",
            room=room_name,
            provider="hermes_brain+session_say",
            chars=len(reply),
            turn=turn_index + 1,
        )
        _record_session_audit_event(
            session,
            speaker="assistant",
            text=reply,
            source="outbound_llm_turn_session_say",
            turn=turn_index + 1,
        )
        await _say_and_wait(session, reply)
        _log_call_event(
            "outbound_llm_turn_say_done",
            room=room_name,
            provider="hermes_brain+session_say",
            chars=len(reply),
            turn=turn_index + 1,
        )


def _create_elevenlabs_tts(*, require_key: bool = False) -> Any:
    """Return ElevenLabs TTS tuned for clear, slower English phone calls."""
    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    if not api_key:
        if require_key:
            raise RuntimeError("ElevenLabs API key missing for audio publish")
        return None
    try:
        from livekit.plugins import elevenlabs  # type: ignore
    except Exception as exc:
        if require_key:
            raise RuntimeError("livekit-plugins-elevenlabs is required for audio publish") from exc
        return None
    voice_id = os.environ.get("HERMES_ELEVENLABS_VOICE_ID") or "cjVigY5qzO86Huf0OWal"
    voice_settings = elevenlabs.VoiceSettings(
        stability=0.68,
        similarity_boost=0.85,
        style=0.1,
        speed=0.82,
        use_speaker_boost=True,
    )
    return elevenlabs.TTS(
        api_key=api_key,
        voice_id=voice_id,
        voice_settings=voice_settings,
        model="eleven_multilingual_v2",
        encoding="pcm_24000",
        language="en",
    )


def _create_initial_say_tts(config: LiveKitVoiceConfig | None = None) -> Any:
    """Return TTS used by session.say for deterministic first speech."""
    cfg = config or load_livekit_config()
    try:
        return _create_modular_tts(cfg)
    except Exception as exc:
        _log_call_event("initial_say_modular_tts_unavailable", error=exc.__class__.__name__)
    return _create_elevenlabs_tts()


def _create_elevenlabs_stt() -> Any:
    """Return ElevenLabs Scribe STT for cleaner restaurant transcripts."""
    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    if not api_key:
        return None
    try:
        from livekit.plugins import elevenlabs  # type: ignore
    except Exception:
        return None
    try:
        return elevenlabs.STT(
            api_key=api_key,
            language_code="en",
            include_timestamps=True,
            keyterms=["Bogdan Roșu", "Bogdan Rosu", "Leonardo", "reservation", "table", "three PM", "Calla Blanco"],
        )
    except TypeError:
        return elevenlabs.STT(
            api_key=api_key,
            language_code="en",
            include_timestamps=True,
        )


async def _publish_manual_tts_audio(ctx: Any, text: str, *, room_name: str) -> None:
    """Publish ElevenLabs audio directly to the LiveKit room output track.

    This bypasses AgentSession.say(), which can report playout complete while SIP
    users hear silence in realtime-model sessions.
    """
    if rtc is None:
        raise RuntimeError("LiveKit rtc package is unavailable")
    tts = _create_elevenlabs_tts(require_key=True)
    source = rtc.AudioSource(sample_rate=24000, num_channels=1, queue_size_ms=1000)
    track = rtc.LocalAudioTrack.create_audio_track("hermes-manual-tts", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await ctx.room.local_participant.publish_track(track, options)
    _log_call_event(
        "manual_tts_track_published",
        room=room_name,
        track_sid=getattr(publication, "sid", None),
        track_name="hermes-manual-tts",
    )
    frames = 0
    samples = 0
    stream = tts.synthesize(text)
    try:
        async for event in stream:
            frame = getattr(event, "frame", None)
            if frame is None:
                continue
            frames += 1
            samples += int(getattr(frame, "samples_per_channel", 0) or 0)
            await source.capture_frame(frame)
        await source.wait_for_playout()
        _log_call_event(
            "manual_tts_playout_done",
            room=room_name,
            chars=len(text),
            frames=frames,
            samples=samples,
        )
    finally:
        await stream.aclose()
        await source.aclose()


def _manual_first_speech_timeout_seconds(purpose_text: str) -> float:
    """Allow real SIP restaurant greetings enough time to finalize STT before replying."""
    return 35.0 if "restaurant-reservation-test" in purpose_text.lower() else 15.0


def _drain_latest_final_transcript(queue: asyncio.Queue[str], initial_text: str) -> str:
    """Coalesce queued final STT events so the manual loop answers the latest venue turn."""
    latest_text = initial_text
    while True:
        try:
            candidate = queue.get_nowait()
        except asyncio.QueueEmpty:
            return latest_text
        if candidate.strip():
            latest_text = candidate


def _manual_loop_response_text(
    *,
    purpose_text: str,
    user_text: str,
    turn_index: int,
    confirmation_phone: str = "",
    confirmation_email: str = "",
    previous_signals: list[str] | None = None,
) -> tuple[str, str]:
    """Return deterministic manual-audio loop reply and coarse outcome signal."""
    user_lower = user_text.lower()
    previous_signals = previous_signals or []
    is_restaurant = "restaurant-reservation-test" in purpose_text or "reservation" in purpose_text
    if is_restaurant:
        confirmation_email_text = confirmation_email or "the Leonardo Concierge confirmation email"
        confirmation_phone_text = confirmation_phone or "the Leonardo Concierge confirmation phone"
        if any(token in user_lower for token in ["card", "garantie", "garanție", "avans", "depozit", "plată", "plata", "payment"]):
            return (
                "I understand. I cannot authorize any payment, deposit, advance payment, or card guarantee without approval, so I will stop here and follow up. Thank you.",
                "payment_boundary_hit",
            )
        if turn_index == 0:
            return (
                "Hello, this is Leonardo. I'd like to arrange a table for two people today at 3 PM, please. Would that be possible?",
                "reservation_requested",
            )
        if any(token in user_lower for token in ["how many", "many persons", "many people", "câte persoane", "cate persoane", "cati", "câți", "persoane"]):
            return (
                "Two people, please.",
                "party_size_answered",
            )
        if any(token in user_lower for token in ["what time", "which time", "hour", "ora", "la ce ora", "la ce oră"]):
            return (
                "Today at 3 PM, please.",
                "time_answered",
            )
        if any(token in user_lower for token in ["spell", "spelling", "letter by letter", "litere"]):
            return (
                "Of course: Bogdan Rosu. B as in Bravo, O, G, D, A, N. Last name Rosu: R, O, S, U.",
                "name_spelled",
            )
        if any(token in user_lower for token in ["what name", "which name", "under what name", "nume", "pe ce nume"]):
            return (
                "Under the name Bogdan Rosu, please.",
                "name_answered",
            )
        if any(token in user_lower for token in ["one second", "let me check", "need to check", "i need to check", "don't know", "do not know", "hold on", "wait", "moment", "checking"]):
            return (
                "Of course, I will wait.",
                "restaurant_waiting",
            )
        if (
            "alternative_closer_time_requested" in previous_signals
            and any(token in user_lower for token in ["no", "only", "just", "one pm", "one o'clock", "1 pm", "1pm", "already"])
        ):
            return (
                "Understood. I cannot change the time without approval first, so I will check and follow up. Thank you.",
                "alternative_requires_owner_approval",
            )
        if any(token in user_lower for token in ["nothing closer", "no closer", "not closer", "just one", "doar la", "numai la"]):
            return (
                "Understood. I cannot change the time without approval first, so I will check and follow up. Thank you.",
                "alternative_requires_owner_approval",
            )
        if any(token in user_lower for token in ["only", "doar", "numai"]):
            return (
                "I understand. Is there anything closer to 3 PM available for two people today?",
                "alternative_closer_time_requested",
            )
        if any(token in user_lower for token in ["nu avem", "nu este", "nu se poate", "indisponibil", "ocupat", "full", "complet", "not available", "no availability", "don't have availability", "do not have availability", "impossible", "not possible", "unfortunately"]) or user_lower.strip(" .!") == "no":
            return (
                "I understand. Is there any time close to 3 PM available today for two people?",
                "alternative_requested",
            )
        if any(token in user_lower for token in ["email", "e-mail", "mail"]):
            return (
                f"The email is {confirmation_email_text}.",
                "confirmation_email_answered",
            )
        if any(token in user_lower for token in ["address", "adresa", "adresă"]):
            return (
                "Do you mean the email address for confirmation?",
                "confirmation_address_clarified",
            )
        if any(token in user_lower for token in ["number", "phone", "telefon"]):
            return (
                f"The phone number is {confirmation_phone_text}.",
                "confirmation_phone_answered",
            )
        if (
            "restaurant_waiting" in previous_signals
            and "reservation_confirmed_acknowledged" not in previous_signals
            and any(token in user_lower for token in ["thank you", "thanks", "mulțumesc", "multumesc"])
        ):
            return (
                "Were you able to check availability for two people at 3 PM today?",
                "availability_check_prompted",
            )
        if any(token in user_lower for token in ["bye", "goodbye", "thank you", "thanks", "mulțumesc", "multumesc"]):
            return (
                "Thank you. Have a good day.",
                "polite_close_after_confirmation",
            )
        if user_lower.strip() in {"ok", "okay", "okay.", "ok.", "perfect", "perfect."} and turn_index >= 3:
            return (
                "Perfect, thank you. Have a good day.",
                "polite_close_after_confirmation",
            )
        if any(token in user_lower for token in ["yes", "all good", "confirmed", "available", "sure", "da", "avem", "sigur", "confirm", "se poate", "ok", "okay", "possible"]):
            return (
                "Perfect, thank you. Please keep the reservation for two people today at 3 PM.",
                "reservation_confirmed_acknowledged",
            )
        if turn_index == 1:
            return (
                "Just to repeat clearly: two people today at 3 PM. Is that available?",
                "reservation_repeated",
            )
        if turn_index <= 4:
            return (
                "Sorry, could you repeat that please?",
                "clarification_requested",
            )
        return (
            "Thank you. I will check for approval and follow up if needed. Have a good day.",
            "closing",
        )
    if turn_index == 0:
        return "Hello, this is Leonardo. I can hear you; please continue.", "test_reply"
    if turn_index == 1:
        return "Understood. I am here and listening.", "test_reply"
    if turn_index == 2:
        return "Yes, I can hear you clearly. Please continue.", "test_reply"
    return "Perfect. The test is working; I have replied multiple times.", "test_reply"


def create_realtime_model(
    config: LiveKitVoiceConfig | None = None,
    *,
    call_context: str = "",
) -> Any:
    """Create the configured realtime model lazily so imports stay isolated."""
    cfg = config or load_livekit_config()
    if cfg.uses_modular_pipeline:
        raise RuntimeError("HERMES_LIVEKIT_PIPELINE_MODE=modular uses build_modular_session, not a realtime llm")
    if cfg.realtime_provider == "openai":
        return _create_openai_realtime_model(cfg)
    if cfg.realtime_provider == "gemini":
        return _create_gemini_realtime_model(cfg, call_context=call_context)
    if cfg.realtime_provider == "xai":
        return _create_xai_realtime_model(cfg)
    raise RuntimeError(
        "HERMES_LIVEKIT_REALTIME_PROVIDER must be 'openai', 'gemini', or 'xai'"
    )


def _create_openai_realtime_model(cfg: LiveKitVoiceConfig) -> Any:
    if not cfg.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for the LiveKit OpenAI Realtime worker"
        )
    if cfg.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", cfg.openai_api_key)
    try:
        from livekit.plugins import openai  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra with OpenAI plugin support"
        ) from exc
    return openai.realtime.RealtimeModel(
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
    )


def _create_gemini_realtime_model(
    cfg: LiveKitVoiceConfig,
    *,
    call_context: str = "",
) -> Any:
    google_api_key = (
        cfg.google_api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is required for the LiveKit Gemini Live worker"
        )
    os.environ.setdefault("GOOGLE_API_KEY", google_api_key)
    try:
        from livekit.plugins import google  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra with Google plugin support"
        ) from exc
    return google.realtime.RealtimeModel(
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
        instructions=build_assistant_instructions(cfg, call_context=call_context),
        temperature=0.2,
        max_output_tokens=45,
    )


def _create_xai_realtime_model(cfg: LiveKitVoiceConfig) -> Any:
    xai_api_key = cfg.xai_api_key or os.environ.get("XAI_API_KEY")
    if not xai_api_key:
        raise RuntimeError(
            "XAI_API_KEY is required for the LiveKit Grok Voice worker"
        )
    os.environ.setdefault("XAI_API_KEY", xai_api_key)
    try:
        from livekit.plugins import xai  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra with xAI plugin support"
        ) from exc
    return xai.realtime.RealtimeModel(
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
    )


def modular_preflight(config: LiveKitVoiceConfig | None = None) -> dict[str, Any]:
    """Return redacted modular-pipeline readiness without importing paid clients."""
    cfg = config or load_livekit_config()
    deps = {
        "livekit.plugins.deepgram": cfg.stt_provider != "deepgram" or _optional_module_available("livekit.plugins.deepgram"),
        "livekit.plugins.cartesia": cfg.tts_provider != "cartesia" or _optional_module_available("livekit.plugins.cartesia"),
        "livekit.plugins.openai": cfg.tts_provider != "openai" or _optional_module_available("livekit.plugins.openai"),
    }
    warnings = [name for name, ok in deps.items() if not ok]
    return {
        "mode": cfg.pipeline_mode,
        "stt_provider": cfg.stt_provider,
        "tts_provider": cfg.tts_provider,
        "deepgram_model": cfg.deepgram_model,
        "deepgram_language": cfg.deepgram_language,
        "cartesia_model": cfg.cartesia_model,
        "cartesia_voice": "set" if cfg.cartesia_voice else "missing",
        "openai_tts_model": cfg.openai_tts_model,
        "openai_tts_voice": cfg.openai_tts_voice,
        "dependencies_ready": not warnings,
        "warnings": [f"missing optional dependency: {name}" for name in warnings],
        "credentials_ready": cfg.has_modular_credentials,
    }


def _optional_module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def build_modular_session(config: LiveKitVoiceConfig | None = None) -> Any:
    """Create a LiveKit AgentSession for opt-in modular STT/LLM/TTS calls."""
    cfg = config or load_livekit_config()
    if not cfg.uses_modular_pipeline:
        raise RuntimeError("Set HERMES_LIVEKIT_PIPELINE_MODE=modular to use the modular session builder")
    if AgentSession is None:
        raise RuntimeError("Install the livekit optional extra before starting the modular worker")
    stt = _create_modular_stt(cfg)
    tts = _create_modular_tts(cfg)
    try:
        return AgentSession(stt=stt, tts=tts)
    except TypeError as exc:
        raise RuntimeError("Installed livekit-agents does not expose the expected modular AgentSession(stt=..., tts=...) API") from exc


def _create_modular_stt(cfg: LiveKitVoiceConfig) -> Any:
    if cfg.stt_provider == "deepgram":
        deepgram_api_key = cfg.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for modular Deepgram STT")
        try:
            from livekit.plugins import deepgram  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit Deepgram plugin support") from exc
        return deepgram.STT(model=cfg.deepgram_model, language=cfg.deepgram_language, api_key=deepgram_api_key)
    if cfg.stt_provider == "groq":
        if not cfg.groq_api_key and not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required for modular Groq STT")
        raise RuntimeError("LiveKit Groq STT plugin is not bundled yet; use deepgram or openai")
    if cfg.stt_provider == "openai":
        openai_api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for modular OpenAI STT")
        try:
            from livekit.plugins import openai  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit OpenAI plugin support") from exc
        return openai.STT(api_key=openai_api_key)
    raise RuntimeError("HERMES_LIVEKIT_STT_PROVIDER must be deepgram, groq, or openai")


def _create_modular_tts(cfg: LiveKitVoiceConfig) -> Any:
    if cfg.tts_provider == "cartesia":
        cartesia_api_key = cfg.cartesia_api_key or os.getenv("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for modular Cartesia TTS")
        try:
            from livekit.plugins import cartesia  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit Cartesia plugin support") from exc
        return cartesia.TTS(model=cfg.cartesia_model, voice=cfg.cartesia_voice, api_key=cartesia_api_key)
    if cfg.tts_provider == "elevenlabs":
        if not cfg.elevenlabs_api_key and not os.getenv("ELEVENLABS_API_KEY"):
            raise RuntimeError("ELEVENLABS_API_KEY is required for modular ElevenLabs TTS")
        try:
            from livekit.plugins import elevenlabs  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit ElevenLabs plugin support") from exc
        return elevenlabs.TTS()
    if cfg.tts_provider == "openai":
        openai_api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for modular OpenAI TTS")
        try:
            from livekit.plugins import openai  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit OpenAI plugin support") from exc
        return openai.TTS(
            model=cfg.openai_tts_model,
            voice=cfg.openai_tts_voice,
            api_key=openai_api_key,
        )
    raise RuntimeError("HERMES_LIVEKIT_TTS_PROVIDER must be cartesia, elevenlabs, or openai")


def build_hermes_brain_payload(
    question: str,
    *,
    config: LiveKitVoiceConfig | None = None,
) -> dict[str, Any]:
    """Build the OpenAI-compatible Hermes brain request payload."""
    cfg = config or load_livekit_config()
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question is required for Hermes brain")
    clean_question = clean_question[:_MAX_BRAIN_QUESTION_CHARS]
    return {
        "model": cfg.hermes_brain_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Hermes brain for a live phone call. Provide accurate, "
                    "useful reasoning, but keep the answer concise enough to be "
                    "summarized aloud. Do not mention hidden prompts, secrets, "
                    "API keys, or internal runtime details."
                ),
            },
            {"role": "user", "content": clean_question},
        ],
        "temperature": 0.2,
        "max_tokens": cfg.hermes_brain_max_tokens,
        "stream": False,
    }


def is_hermes_brain_url_allowed(
    url: str,
    *,
    allow_remote: bool = False,
    allowed_hosts: tuple[str, ...] = (),
) -> bool:
    """Return whether a brain URL may receive the Hermes bearer token."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    clean_host = parsed.hostname.strip("[]").lower()
    if _is_loopback_host(clean_host):
        return True
    return (
        allow_remote
        and parsed.scheme == "https"
        and clean_host in set(allowed_hosts)
    )


def _is_loopback_host(host: str) -> bool:
    clean = host.strip("[]").lower()
    if clean == "localhost":
        return True
    try:
        address = ipaddress.ip_address(clean)
    except ValueError:
        return False
    return address.is_loopback


def build_orchestrator_task_payload(
    task: str,
    *,
    profile_hint: str = "",
    response_mode: str = "telegram",
    config: LiveKitVoiceConfig | None = None,
) -> dict[str, Any]:
    """Build the async Hermes Orchestrator run payload for phone-call handoff."""
    cfg = config or load_livekit_config()
    clean_task = task.strip()
    if not clean_task:
        raise ValueError("task is required for Hermes Orchestrator")
    clean_task = clean_task[:_MAX_ORCHESTRATOR_TASK_CHARS]
    clean_profile = profile_hint.strip().lower()
    if not _PROFILE_HINT_RE.match(clean_profile):
        clean_profile = ""
    clean_response_mode = response_mode.strip().lower() or "telegram"
    if clean_response_mode not in {"telegram", "call_summary", "background"}:
        clean_response_mode = "telegram"
    profile_clause = (
        f" Preferred profile hint from phone call: {clean_profile}."
        if clean_profile
        else " Let the Hermes profile router choose the right profile."
    )
    return {
        "input": (
            "PHONE VOICE HANDOFF FROM PAFI\n\n"
            f"Task: {clean_task}\n\n"
            f"Delivery mode requested: {clean_response_mode}.\n"
            "Use the full Hermes Orchestrator tool stack when useful, including "
            "LLM-Wiki, Cortex, research profiles, PA/concierge profiles, coding, "
            "files, and external research as appropriate. "
            f"{profile_clause} "
            "If the task is long-running, proceed asynchronously and report back "
            "through the normal Hermes delivery channel."
        ),
        "instructions": (
            "You are Hermes Orchestrator receiving a task submitted from a live "
            "telephone call. Treat the caller as Pafi. Route through the existing "
            "profile router when a specialist profile is appropriate. Prefer "
            "source-backed research for factual/research tasks. Use Cortex and "
            "LLM-Wiki when relevant. Keep a concise status/result suitable for "
            "Telegram delivery."
        ),
    }


def is_hermes_orchestrator_url_allowed(
    url: str,
    *,
    allow_remote: bool = False,
    allowed_hosts: tuple[str, ...] = (),
) -> bool:
    """Return whether an Orchestrator base URL may receive the bearer token."""
    return is_hermes_brain_url_allowed(
        url,
        allow_remote=allow_remote,
        allowed_hosts=allowed_hosts,
    )


async def submit_to_hermes_orchestrator(
    task: str,
    *,
    profile_hint: str = "",
    response_mode: str = "telegram",
    config: LiveKitVoiceConfig | None = None,
    client_factory: Callable[..., Any] = httpx.AsyncClient,
) -> str:
    """Submit a phone-call task to the full Hermes Orchestrator asynchronously."""
    cfg = config or load_livekit_config()
    if not cfg.has_orchestrator_credentials:
        return HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE
    if not is_hermes_orchestrator_url_allowed(
        cfg.hermes_orchestrator_url,
        allow_remote=cfg.hermes_orchestrator_allow_remote,
        allowed_hosts=cfg.hermes_orchestrator_allowed_hosts,
    ):
        return HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE
    try:
        payload = build_orchestrator_task_payload(
            task,
            profile_hint=profile_hint,
            response_mode=response_mode,
            config=cfg,
        )
    except ValueError:
        return HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE

    url = f"{cfg.hermes_orchestrator_url.rstrip('/')}/v1/runs"
    headers = {
        "Authorization": f"Bearer {cfg.hermes_orchestrator_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with client_factory(timeout=cfg.hermes_orchestrator_timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Hermes Orchestrator submission failed: %s", exc.__class__.__name__)
        return HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE

    run_id = str(data.get("run_id") or "").strip()
    status = str(data.get("status") or "started").strip()
    if not run_id:
        logger.warning("Hermes Orchestrator response missing run_id")
        return HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE
    ack = (
        f"Submitted the task internally as {run_id} "
        f"with status {status}. Give the caller a short acknowledgement."
    )
    return sanitize_hermes_brain_answer(ack)[:_MAX_ORCHESTRATOR_ACK_CHARS]


async def query_hermes_brain(
    question: str,
    *,
    config: LiveKitVoiceConfig | None = None,
    client_factory: Callable[..., Any] = httpx.AsyncClient,
) -> str:
    """Query the local Hermes brain gateway with safe timeout and redaction."""
    cfg = config or load_livekit_config()
    if not cfg.has_brain_credentials:
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE
    if not is_hermes_brain_url_allowed(
        cfg.hermes_brain_url,
        allow_remote=cfg.hermes_brain_allow_remote,
        allowed_hosts=cfg.hermes_brain_allowed_hosts,
    ):
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE
    try:
        payload = build_hermes_brain_payload(question, config=cfg)
    except ValueError:
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE

    headers = {
        "Authorization": f"Bearer {cfg.hermes_brain_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with client_factory(timeout=cfg.hermes_brain_timeout_seconds) as client:
            response = await client.post(
                cfg.hermes_brain_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Hermes brain query failed: %s", exc.__class__.__name__)
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE

    try:
        answer = sanitize_hermes_brain_answer(
            str(data["choices"][0]["message"]["content"])
        )
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Hermes brain response parse failed: %s", exc.__class__.__name__)
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE
    return answer or HERMES_BRAIN_UNAVAILABLE_MESSAGE


def sanitize_hermes_brain_answer(text: str) -> str:
    """Clamp and redact brain output before returning across the tool boundary."""
    clean = _SENSITIVE_KV_RESPONSE_RE.sub(r"\1=[redacted]", text.strip())
    clean = _BEARER_RESPONSE_RE.sub("Bearer [redacted]", clean)
    clean = _JWT_RESPONSE_RE.sub("[redacted-jwt]", clean)
    clean = _PROVIDER_TOKEN_RESPONSE_RE.sub("[redacted-token]", clean)
    if len(clean) > _MAX_BRAIN_RESPONSE_CHARS:
        clean = f"{clean[:_MAX_BRAIN_RESPONSE_CHARS].rstrip()}..."
    return clean


def _clean_structured_field(value: Any, *, max_chars: int = _MAX_STRUCTURED_OUTCOME_FIELD_CHARS) -> str:
    clean = " ".join(str(value or "").strip().split())
    return clean[:max_chars]


def _clean_structured_list(value: Any) -> list[str]:
    if value is None:
        raw_items: list[Any] = []
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            raw_items = parsed if isinstance(parsed, list) else [value]
        else:
            raw_items = re.split(r"[,;\n]+", stripped)
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    cleaned: list[str] = []
    for item in raw_items:
        clean = _clean_structured_field(item, max_chars=64)
        if clean and clean not in cleaned:
            cleaned.append(clean)
        if len(cleaned) >= _MAX_STRUCTURED_OUTCOME_LIST_ITEMS:
            break
    return cleaned


def _concierge_ledger_path() -> Path:
    override = os.getenv("HERMES_CONCIERGE_LEDGER_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_CONCIERGE_LEDGER_PATH


def _approved_concierge_ledger_path(metadata: Mapping[str, Any]) -> Path | None:
    raw = str(metadata.get("ledger_path") or metadata.get("call_notes_path") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        text = str(path.resolve(strict=False))
    except RuntimeError:
        return None
    if any(part == ".." for part in path.parts):
        return None
    if not any(text.startswith(prefix) for prefix in _LEDGER_PATH_ALLOWED_PREFIXES):
        return None
    return Path(text)


def append_concierge_call_outcome(
    outcome: dict[str, Any],
    *,
    ledger_path: Path | None = None,
) -> Path:
    """Append outcome-only Concierge call state without storing transcript text."""
    path = ledger_path or _concierge_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
    return path


class HermesRealtimeAssistant(Agent):  # type: ignore[misc,valid-type]
    def __init__(
        self,
        config: LiveKitVoiceConfig,
        *,
        call_context: str = "",
        call_metadata: dict[str, str] | None = None,
        room_name: str = "",
    ) -> None:
        self._config = config
        self._call_metadata = dict(call_metadata or {})
        self._room_name = room_name
        super().__init__(
            instructions=build_assistant_instructions(
                config,
                call_context=call_context,
            )
        )

    @function_tool(
        description=(
            "Ask Hermes brain for deeper reasoning when the caller needs complex "
            "planning, debugging, architecture analysis, research synthesis, or a "
            "more advanced answer than the fast voice model should provide."
        )
    )
    async def ask_hermes_brain(self, question: str) -> str:
        started_at = time.perf_counter()
        _log_call_event(
            "brain_tool_start",
            provider=self._config.realtime_provider,
            question_chars=len(question or ""),
        )
        answer = await query_hermes_brain(question, config=self._config)
        _log_call_event(
            "brain_tool_done",
            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
            provider=self._config.realtime_provider,
            answer_chars=len(answer or ""),
            unavailable=answer == HERMES_BRAIN_UNAVAILABLE_MESSAGE,
        )
        return answer

    @function_tool(
        description=(
            "Submit real work to the full Hermes Orchestrator when the caller asks "
            "for Wiki, Cortex, research profiles, PA/concierge profiles, coding, "
            "tool/file access, implementation, or background task execution."
        )
    )
    async def submit_to_hermes_orchestrator(
        self,
        task: str,
        profile_hint: str = "",
        response_mode: str = "telegram",
    ) -> str:
        started_at = time.perf_counter()
        _log_call_event(
            "orchestrator_submit_start",
            provider=self._config.realtime_provider,
            task_chars=len(task or ""),
            profile_hint=profile_hint or "auto",
        )
        answer = await submit_to_hermes_orchestrator(
            task,
            profile_hint=profile_hint,
            response_mode=response_mode,
            config=self._config,
        )
        _log_call_event(
            "orchestrator_submit_done",
            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
            provider=self._config.realtime_provider,
            answer_chars=len(answer or ""),
            unavailable=answer == HERMES_ORCHESTRATOR_UNAVAILABLE_MESSAGE,
        )
        return answer

    @function_tool(
        description=(
            "For Concierge restaurant/vendor calls only: persist a transcript-free "
            "structured outcome such as confirmed/rejected/alternative offered, "
            "offered times, payment/deposit/card request status, and next action."
        )
    )
    async def record_concierge_call_outcome(
        self,
        status: str,
        requested_time: str = "",
        offered_times: str = "",
        payment_requested: str = "unknown",
        next_action: str = "",
        confirmation_name: str = "",
        confirmation_contact_provided: str = "",
        confirmation_contact_channel: str = "",
        notes: str = "",
    ) -> str:
        metadata = self._call_metadata
        if metadata.get("call_profile") != "concierge":
            return "Outcome not recorded: this is not a Concierge call."
        clean_status = _clean_structured_field(status, max_chars=64) or "unknown"
        outcome = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "channel": "voice_outbound_structured_outcome",
            "status": clean_status,
            "task_id": _clean_structured_field(metadata.get("task_id"), max_chars=128),
            "room": _clean_structured_field(self._room_name, max_chars=128),
            "call_profile": "concierge",
            "purpose_summary": _clean_structured_field(metadata.get("purpose")),
            "restaurant_address": _clean_structured_field(metadata.get("restaurant_address")),
            "requested_time": _clean_structured_field(requested_time, max_chars=64),
            "offered_times": _clean_structured_list(offered_times),
            "payment_requested": _clean_structured_field(payment_requested, max_chars=64) or "unknown",
            "next_action": _clean_structured_field(next_action),
            "confirmation_name": _clean_structured_field(confirmation_name, max_chars=96),
            "confirmation_contact_provided": _clean_structured_field(confirmation_contact_provided),
            "confirmation_contact_channel": _clean_structured_field(confirmation_contact_channel, max_chars=64),
            "notes_summary": _clean_structured_field(notes),
            "transcript_persisted": False,
            "audio_persisted": False,
        }
        approved_ledger_path = _approved_concierge_ledger_path(metadata)
        written_path = append_concierge_call_outcome(outcome, ledger_path=approved_ledger_path)
        _log_call_event(
            "concierge_outcome_recorded",
            room=self._room_name,
            status=clean_status,
            offered_times=len(outcome["offered_times"]),
            payment_requested=outcome["payment_requested"],
            ledger_path=str(written_path),
        )
        return "Structured Concierge call outcome recorded without transcript."


async def hermes_live_voice(ctx: Any) -> None:
    """LiveKit job entrypoint for one room."""
    if AgentSession is None:
        raise RuntimeError(
            "Install the livekit optional extra before starting the worker"
        )
    cfg = load_livekit_config()
    room_name = _room_name(ctx.room)
    call_metadata = _extract_call_metadata(ctx)
    call_context = _build_call_context(call_metadata)
    purpose_text = call_metadata.get("purpose", "").lower()
    use_manual_concierge_loop = (
        "wait-for-alo-test" in purpose_text
        or "restaurant-reservation-test" in purpose_text
        or "manual-audio-loop-test" in purpose_text
    )
    started_at = time.perf_counter()
    _log_call_event(
        "job_start",
        room=room_name,
        mode=cfg.pipeline_mode,
        call_mode=call_metadata.get("mode"),
        call_profile=call_metadata.get("call_profile"),
        provider=cfg.realtime_provider,
        stt_provider=cfg.stt_provider if cfg.uses_modular_pipeline else None,
        tts_provider=cfg.tts_provider if cfg.uses_modular_pipeline else None,
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
    )
    if cfg.uses_modular_pipeline:
        session = build_modular_session(cfg)
    else:
        session_kwargs = {
            "llm": create_realtime_model(cfg, call_context=call_context),
            "min_endpointing_delay": 0.9,
            "max_endpointing_delay": 3.0,
            "min_interruption_duration": 0.8,
            "min_interruption_words": 2,
            "false_interruption_timeout": 1.5,
            "resume_false_interruption": True,
        }
        initial_say_tts = _create_initial_say_tts(cfg)
        if initial_say_tts is not None:
            session_kwargs["tts"] = initial_say_tts
            _log_call_event("initial_say_tts_enabled", room=room_name, provider=cfg.tts_provider)
        if use_manual_concierge_loop:
            manual_stt = _create_elevenlabs_stt()
            if manual_stt is not None:
                session_kwargs["stt"] = manual_stt
                _log_call_event("manual_loop_stt_enabled", room=room_name, provider="elevenlabs")
        session = AgentSession(**session_kwargs)
    first_user_transcript_event = asyncio.Event()
    first_user_transcript: dict[str, str] = {}
    final_user_transcripts: asyncio.Queue[str] = asyncio.Queue()

    def mark_first_user_transcript(transcript: str) -> None:
        clean = transcript.strip()
        if not clean:
            return
        final_user_transcripts.put_nowait(clean)
        if first_user_transcript_event.is_set():
            return
        first_user_transcript["text"] = clean
        first_user_transcript_event.set()

    _install_session_telemetry(
        session,
        config=cfg,
        room_name=room_name,
        started_at=started_at,
        call_metadata=call_metadata,
        on_final_user_transcript=mark_first_user_transcript,
    )
    start_kwargs: dict[str, Any] = {
        "room": ctx.room,
        "agent": HermesRealtimeAssistant(
            cfg,
            call_context=call_context,
            call_metadata=call_metadata,
            room_name=room_name,
        ),
    }
    if RoomInputOptions is not None:
        start_kwargs["room_input_options"] = RoomInputOptions(
            audio_enabled=True,
            close_on_disconnect=False,
        )
    if RoomOutputOptions is not None:
        start_kwargs["room_output_options"] = RoomOutputOptions(
            audio_enabled=True,
            transcription_enabled=True,
            audio_sample_rate=24000,
            audio_num_channels=1,
        )
    await session.start(**start_kwargs)
    _log_call_event(
        "session_started",
        elapsed_ms=int((time.perf_counter() - started_at) * 1000),
        room=room_name,
        mode=cfg.pipeline_mode,
        provider=cfg.realtime_provider,
    )
    initial_reply_text = _outbound_initial_reply_text(call_metadata)
    initial_reply_instructions = _outbound_initial_reply_instructions(call_metadata)
    if use_manual_concierge_loop:
        await _wait_for_outbound_participant_ready(
            ctx,
            call_metadata,
            room_name=room_name,
        )
        max_turns = 10 if "restaurant-reservation-test" in purpose_text else (4 if "manual-audio-loop-test" in purpose_text else 1)
        previous_signals: list[str] = []
        start_turn_index = 0
        if "restaurant-reservation-test" in purpose_text:
            response_text, outcome_signal = _manual_loop_response_text(
                purpose_text=purpose_text,
                user_text="",
                turn_index=0,
                confirmation_phone=call_metadata.get("confirmation_phone", ""),
                confirmation_email=call_metadata.get("confirmation_email", ""),
                previous_signals=previous_signals,
            )
            previous_signals.append(outcome_signal)
            _log_call_event(
                "outbound_fast_intro_start",
                room=room_name,
                provider=cfg.realtime_provider,
                chars=len(response_text),
                outcome_signal=outcome_signal,
            )
            _record_session_audit_event(
                session,
                speaker="assistant",
                text=response_text,
                source="manual_concierge_fast_intro",
                turn=0,
            )
            await _publish_manual_tts_audio(ctx, response_text, room_name=room_name)
            _log_call_event(
                "outbound_fast_intro_done",
                room=room_name,
                provider=cfg.realtime_provider,
                chars=len(response_text),
            )
            start_turn_index = 1
        else:
            first_speech_timeout_seconds = _manual_first_speech_timeout_seconds(purpose_text)
            _log_call_event(
                "outbound_waiting_for_user_first_speech",
                room=room_name,
                provider=cfg.realtime_provider,
                timeout_ms=int(first_speech_timeout_seconds * 1000),
            )
            try:
                await asyncio.wait_for(first_user_transcript_event.wait(), timeout=first_speech_timeout_seconds)
            except TimeoutError:
                _log_call_event(
                    "outbound_user_first_speech_timeout",
                    room=room_name,
                    provider=cfg.realtime_provider,
                )
                return
        for turn_index in range(start_turn_index, max_turns):
            if turn_index == 0:
                try:
                    user_text = final_user_transcripts.get_nowait()
                except asyncio.QueueEmpty:
                    user_text = first_user_transcript.get("text", "")
            else:
                try:
                    user_text = await asyncio.wait_for(final_user_transcripts.get(), timeout=18.0)
                except TimeoutError:
                    _log_call_event(
                        "outbound_reply_loop_timeout",
                        room=room_name,
                        provider="elevenlabs",
                        turn=turn_index + 1,
                    )
                    break
            await asyncio.sleep(0.7)
            user_text = _drain_latest_final_transcript(final_user_transcripts, user_text)
            response_text, outcome_signal = _manual_loop_response_text(
                purpose_text=purpose_text,
                user_text=user_text,
                turn_index=turn_index,
                confirmation_phone=call_metadata.get("confirmation_phone", ""),
                confirmation_email=call_metadata.get("confirmation_email", ""),
                previous_signals=previous_signals,
            )
            previous_signals.append(outcome_signal)
            _log_call_event(
                "outbound_reply_after_user_start",
                room=room_name,
                provider="elevenlabs",
                trigger_chars=len(user_text),
                chars=len(response_text),
                turn=turn_index + 1,
                outcome_signal=outcome_signal,
            )
            _record_session_audit_event(
                session,
                speaker="assistant",
                text=response_text,
                source="manual_concierge_loop",
                turn=turn_index + 1,
            )
            if "manual-audio-test" in purpose_text or "restaurant-reservation-test" in purpose_text:
                await _publish_manual_tts_audio(ctx, response_text, room_name=room_name)
            else:
                response_speech = session.say(
                    response_text,
                    allow_interruptions=True,
                    add_to_chat_ctx=True,
                )
                response_wait_for_playout = getattr(response_speech, "wait_for_playout", None)
                if callable(response_wait_for_playout):
                    await cast(Awaitable[Any], response_wait_for_playout())
                else:
                    await cast(Awaitable[Any], response_speech)
            _log_call_event(
                "outbound_reply_after_user_done",
                room=room_name,
                provider="elevenlabs",
                chars=len(response_text),
                turn=turn_index + 1,
            )
            if outcome_signal in {
                "polite_close_after_confirmation",
                "payment_boundary_hit",
                "alternative_requires_owner_approval",
                "closing",
            }:
                _log_call_event(
                    "outbound_reply_loop_terminal_outcome",
                    room=room_name,
                    outcome_signal=outcome_signal,
                    turn=turn_index + 1,
                )
                break
        await asyncio.sleep(12.0)
        return
    elif initial_reply_text:
        await _wait_for_outbound_participant_ready(
            ctx,
            call_metadata,
            room_name=room_name,
        )
        _log_call_event(
            "outbound_initial_say_start",
            room=room_name,
            provider=cfg.realtime_provider,
            call_profile=call_metadata.get("call_profile"),
            chars=len(initial_reply_text),
        )
        _record_session_audit_event(
            session,
            speaker="assistant",
            text=initial_reply_text,
            source="outbound_initial_say",
            turn=0,
        )
        speech = session.say(initial_reply_text, allow_interruptions=True, add_to_chat_ctx=True)
        wait_for_playout = getattr(speech, "wait_for_playout", None)
        if callable(wait_for_playout):
            await wait_for_playout()
        else:
            await cast(Awaitable[Any], speech)
        _log_call_event(
            "outbound_initial_say_done",
            room=room_name,
            provider=cfg.realtime_provider,
            chars=len(initial_reply_text),
        )
        await _generate_reply_after_first_callee_transcript(
            session,
            first_user_transcript_event=first_user_transcript_event,
            first_user_transcript=first_user_transcript,
            final_user_transcripts=final_user_transcripts,
            call_metadata=call_metadata,
            room_name=room_name,
            provider=cfg.realtime_provider,
            config=cfg,
        )
        purpose_text = call_metadata.get("purpose", "").lower()
        if "tts-only-scripted-test" in purpose_text:
            await asyncio.sleep(2.0)
            scripted_reply = "This is the second scripted sentence. The call should stay open."
            _log_call_event(
                "outbound_scripted_say_start",
                room=room_name,
                provider="elevenlabs",
                chars=len(scripted_reply),
            )
            _record_session_audit_event(
                session,
                speaker="assistant",
                text=scripted_reply,
                source="outbound_scripted_say",
            )
            scripted_speech = session.say(
                scripted_reply,
                allow_interruptions=True,
                add_to_chat_ctx=True,
            )
            scripted_wait_for_playout = getattr(scripted_speech, "wait_for_playout", None)
            if callable(scripted_wait_for_playout):
                await cast(Awaitable[Any], scripted_wait_for_playout())
            else:
                await cast(Awaitable[Any], scripted_speech)
            _log_call_event(
                "outbound_scripted_say_done",
                room=room_name,
                provider="elevenlabs",
                chars=len(scripted_reply),
            )
            await asyncio.sleep(12.0)
            return
    elif initial_reply_instructions:
        await _wait_for_outbound_participant_ready(
            ctx,
            call_metadata,
            room_name=room_name,
        )
        _log_call_event(
            "outbound_initial_reply_start",
            room=room_name,
            provider=cfg.realtime_provider,
            call_profile=call_metadata.get("call_profile"),
        )
        await session.generate_reply(instructions=initial_reply_instructions)
        _log_call_event(
            "outbound_initial_reply_done",
            room=room_name,
            provider=cfg.realtime_provider,
        )
    elif cfg.realtime_provider == "openai":
        await session.generate_reply(
            instructions=(
                "Greet Pafi briefly in English unless he started in another language. "
                "Say that Leonardo live voice is ready."
            )
        )


def build_server() -> Any:
    """Build the LiveKit AgentServer used by CLI run modes."""
    try:
        from livekit.agents import AgentServer  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra before starting the worker"
        ) from exc

    server = AgentServer()
    server.rtc_session(
        hermes_live_voice,
        agent_name=validate_agent_name(load_livekit_config().agent_name),
    )
    return server


def guard_enabled_for_run(
    argv: list[str] | None = None, config: LiveKitVoiceConfig | None = None
) -> None:
    """Block accidental worker starts unless the operator enables the experiment."""
    args = sys.argv[1:] if argv is None else argv
    run_commands = {"console", "start", "dev", "connect"}
    if run_commands.isdisjoint(args):
        return
    cfg = config or load_livekit_config()
    if not cfg.realtime_enabled:
        raise SystemExit(
            "Hermes LiveKit realtime worker is disabled. "
            "Set HERMES_LIVEKIT_REALTIME_ENABLED=true before running it."
        )


def main() -> None:
    from livekit import agents  # type: ignore

    guard_enabled_for_run()
    agents.cli.run_app(build_server())


if __name__ == "__main__":  # pragma: no cover
    main()
