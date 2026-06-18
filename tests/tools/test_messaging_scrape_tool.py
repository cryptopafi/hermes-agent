import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from tools import messaging_scrape_tool as tool


def test_messaging_scrape_rejects_bad_since():
    payload = json.loads(tool.messaging_scrape({"action": "run", "since": "bad"}))

    assert payload["success"] is False
    assert "since must be an ISO timestamp" in payload["error"]


def test_status_invokes_doctor(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout='{"ok": true, "telegram": {"authorized": true}}\n', stderr="")

    monkeypatch.setattr(tool, "_script_path", lambda: Path("/tmp/hermes-messaging-intel-vps.sh"))
    monkeypatch.setattr(tool, "_check_requirements", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = json.loads(tool.messaging_scrape({"action": "status"}))

    assert payload["success"] is True
    assert payload["doctor"]["telegram"]["authorized"] is True


def test_telegram_dry_run_command(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"ok": true, "run_id": "r1"}\n', stderr="")

    monkeypatch.setattr(tool, "_script_path", lambda: Path("/tmp/hermes-messaging-intel-vps.sh"))
    monkeypatch.setattr(tool, "_check_requirements", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = json.loads(
        tool.messaging_scrape(
            {
                "action": "run",
                "mode": "dry_run",
                "platform": "telegram",
                "since": "2026-06-18T00:00:00Z",
                "max_dialogs": 2,
            }
        )
    )

    assert payload["success"] is True
    assert captured["cmd"][:2] == ["/tmp/hermes-messaging-intel-vps.sh", "run"]
    assert "--dry-run" in captured["cmd"]
    assert "--skip-wiki" in captured["cmd"]
    assert "--skip-cortex" in captured["cmd"]
    assert "--max-dialogs" in captured["cmd"]
    assert captured["kwargs"]["timeout"] == 1200


def test_scrape_and_wiki_whatsapp_uses_cortex_fallback(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout='{"ok": true, "wiki_writes": 2}\n', stderr="")

    monkeypatch.setattr(tool, "_script_path", lambda: Path("/tmp/hermes-messaging-intel-vps.sh"))
    monkeypatch.setattr(tool, "_check_requirements", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = json.loads(tool.messaging_scrape({"action": "run", "mode": "scrape_and_wiki", "platform": "whatsapp"}))

    assert payload["success"] is True
    assert "--skip-wiki" not in captured["cmd"]
    assert "--skip-telegram" in captured["cmd"]
    idx = captured["cmd"].index("--cortex-platform")
    assert captured["cmd"][idx : idx + 2] == ["--cortex-platform", "whatsapp"]
