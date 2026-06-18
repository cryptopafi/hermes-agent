import json

from gateway.voice_stack_compare import (
    BASELINE_ROUTE_ID,
    evaluate_routes,
    load_env_file,
    render_text_report,
)


def test_load_env_file_strips_quotes_comments_and_exports(tmp_path):
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        """
# comment
export VAPI_API_KEY='vapi-test-value'
OPENAI_API_KEY="sk-test-value"
BAD LINE
1INVALID=ignored
XAI_API_KEY=xai-test-value # removed by shlex comment parsing
""".strip(),
        encoding="utf-8",
    )

    env = load_env_file(env_file)

    assert env["VAPI_API_KEY"] == "vapi-test-value"
    assert env["OPENAI_API_KEY"] == "sk-test-value"
    assert env["XAI_API_KEY"] == "xai-test-value"
    assert "1INVALID" not in env


def test_report_never_contains_secret_values():
    env = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "livekit-secret-value-1234567890",
        "LIVEKIT_API_SECRET": "livekit-api-secret-value-1234567890",
        "DEEPGRAM_API_KEY": "deepgram-secret-value-1234567890",
        "OPENAI_API_KEY": "sk-proj-secret-value-1234567890",
        "VAPI_API_KEY": "vapi-secret-value-1234567890",
        "XAI_API_KEY": "xai-secret-value-1234567890",
        "ELEVENLABS_API_KEY": "eleven-secret-value-1234567890",
    }

    rendered = json.dumps(evaluate_routes(env), sort_keys=True)

    for value in env.values():
        if value.startswith("wss://"):
            continue
        assert value not in rendered
    assert "LIVEKIT_API_KEY" in rendered
    assert "VAPI_API_KEY" in rendered


def test_livekit_baseline_ready_when_required_handles_exist():
    report = evaluate_routes(
        {
            "LIVEKIT_URL": "wss://example.livekit.cloud",
            "LIVEKIT_API_KEY": "set",
            "LIVEKIT_API_SECRET": "set",
            "DEEPGRAM_API_KEY": "set",
            "OPENAI_API_KEY": "set",
        }
    )
    baseline = next(route for route in report["routes"] if route["route_id"] == BASELINE_ROUTE_ID)

    assert baseline["ready"] is True
    assert baseline["readiness_status"] == "ready"
    assert baseline["missing_required_handles"] == []
    assert report["recommendation"]["top_ready_route_id"] == BASELINE_ROUTE_ID


def test_vapi_routes_blocked_without_vapi_key():
    report = evaluate_routes({"OPENAI_API_KEY": "set", "XAI_API_KEY": "set"})
    vapi_routes = [route for route in report["routes"] if route["route_id"].startswith("vapi-")]

    assert vapi_routes
    assert all(route["ready"] is False for route in vapi_routes)
    assert all("VAPI_API_KEY" in route["missing_required_handles"] for route in vapi_routes)


def test_recommendation_is_readiness_not_performance():
    report = evaluate_routes({"VAPI_API_KEY": "set"})

    assert report["scope"] == "preflight_readiness_not_performance"
    assert report["recommendation"]["label"] == "preflight_readiness_rank"
    assert report["recommendation"]["live_test_priority_route_ids"][:2] == [
        "livekit-current-openai-tts",
        "vapi-gpt41-mini-elevenlabs-flash",
    ]
    assert "latency" not in report["recommendation"]["label"]
    assert report["production_cutover_allowed"] is False


def test_route_ids_are_not_redacted_as_provider_tokens():
    report = evaluate_routes({"VAPI_API_KEY": "set", "XAI_API_KEY": "set"})
    rendered = json.dumps(report, sort_keys=True)

    assert "livekit-xai-realtime" in rendered
    assert "vapi-xai-grok-voice" in rendered


def test_latency_budget_and_bilingual_quality_gates_are_present():
    report = evaluate_routes({"VAPI_API_KEY": "set"})
    route = next(route for route in report["routes"] if route["route_id"] == "vapi-gpt41-mini-elevenlabs-flash")
    gate_ids = {gate["gate_id"] for gate in route["voice_quality_gates"]}
    languages = {language for gate in route["voice_quality_gates"] for language in gate["languages"]}

    assert route["latency_budget"]["stt_ms"] > 0
    assert route["latency_budget"]["reasoning_ms"] > 0
    assert route["latency_budget"]["tts_ms"] > 0
    assert route["latency_budget"]["transport_ms"] > 0
    assert route["latency_budget"]["total_ms"] > 0
    assert {"english_naturalness", "romanian_naturalness", "interruptions_and_barge_in"} <= gate_ids
    assert {"en", "ro"} <= languages


def test_promotion_blocked_without_live_benchmark_and_tool_delegation():
    report = evaluate_routes({"VAPI_API_KEY": "set"})
    route = next(route for route in report["routes"] if route["route_id"] == "vapi-gpt41-mini-elevenlabs-flash")

    assert route["ready"] is True
    assert route["promotion_eligible"] is False
    assert "missing_successful_live_inbound_outbound_benchmark" in route["promotion_blockers"]
    assert "missing_verified_hermes_orchestrator_tool_delegation" in route["promotion_blockers"]
    assert "missing_redaction_scan" in route["promotion_blockers"]


def test_promotion_possible_only_after_required_live_evidence():
    report = evaluate_routes(
        {"VAPI_API_KEY": "set"},
        benchmark_results={"vapi-gpt41-mini-elevenlabs-flash": {"success": True, "total_ms": 3200}},
        tool_delegation_results={"vapi-gpt41-mini-elevenlabs-flash": {"success": True, "tool": "wiki"}},
        redaction_scan_passed=True,
    )
    route = next(route for route in report["routes"] if route["route_id"] == "vapi-gpt41-mini-elevenlabs-flash")

    assert route["promotion_eligible"] is True
    assert report["production_cutover_allowed"] is True


def test_text_report_contains_handle_names_not_values():
    report = evaluate_routes({"VAPI_API_KEY": "vapi-secret-value-1234567890"})
    text = render_text_report(report)

    assert "vapi-gpt41-mini-elevenlabs-flash" in text
    assert "vapi-secret-value" not in text
