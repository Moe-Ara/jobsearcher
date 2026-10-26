"""Background task bookkeeping: progress stages, and letter runs killed by a restart are reported, not lost silently."""

import json
import threading

from fastapi.testclient import TestClient

from app.llm.local_provider import GPU_LOCK
from app.main import app
from app.services import letters, tasks
from app.services.letters import generate_letter
from tests.test_letters import job_id, make_setup  # noqa: F401 - fixture re-export


def test_tracked_run_is_on_disk_only_while_running(monkeypatch, tmp_path):
    monkeypatch.setattr(tasks, "_state_file", lambda: tmp_path / "running_tasks.json")
    seen = {}

    def work():
        seen["state"] = json.loads((tmp_path / "running_tasks.json").read_text())
        tasks.progress("Writing draft")
        seen["stage"] = tasks.task_status()["letter-7"]["stage"]
        return 1

    tasks.run_exclusive("letter-7", work)
    assert "letter-7" in seen["state"] and seen["stage"] == "Writing draft"
    assert json.loads((tmp_path / "running_tasks.json").read_text()) == {}
    tasks.run_exclusive("analysis-x", lambda: 1)  # untracked tasks never touch the file
    assert json.loads((tmp_path / "running_tasks.json").read_text()) == {}


def test_restart_reports_interrupted_letter(job_id, monkeypatch, tmp_path):  # noqa: F811
    state = tmp_path / "running_tasks.json"
    monkeypatch.setattr(tasks, "_state_file", lambda: state)
    state.write_text(json.dumps({f"letter-{job_id}": "2026-10-07T12:57:00"}))
    assert tasks.recover_interrupted() == [f"letter-{job_id}"]
    assert json.loads(state.read_text()) == {}
    st = tasks.task_status()[f"letter-{job_id}"]
    assert st["interrupted"] and not st["running"] and "restarted" in st["error"]
    page = TestClient(app).get(f"/jobs/{job_id}").text
    assert "Last run was interrupted" in page and "please start it again" in page
    tasks._last.pop(f"letter-{job_id}", None)


def test_letter_reports_stages_and_gpu_wait(job_id, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(tasks, "_state_file", lambda: tmp_path / "running_tasks.json")
    stages = []
    real = tasks.progress
    monkeypatch.setattr(tasks, "progress", lambda s: stages.append(s) or real(s))
    setup = make_setup()
    holder_ready, release = threading.Event(), threading.Event()

    def hold_gpu():
        with GPU_LOCK:
            holder_ready.set()
            release.wait(5)

    t = threading.Thread(target=hold_gpu)
    t.start()
    holder_ready.wait(5)
    threading.Timer(0.3, release.set).start()
    tasks.run_exclusive(f"letter-{job_id}", lambda: generate_letter(job_id, setup=setup))
    t.join()
    assert stages[0].startswith("Waiting for the GPU")
    assert any(s.startswith("Writing draft") for s in stages)
    assert any(s.startswith("German style editing") for s in stages)
    assert any(s.startswith("Checking facts") for s in stages)
    assert letters.letter_groups  # pipeline completed after the wait
    tasks._last.pop(f"letter-{job_id}", None)
