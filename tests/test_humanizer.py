"""Humanizer: exact prompt + raw Ollama call, chunking, letter frame, language guard, drift checks, letter stage,
standalone page/API, and that it is never used outside explicit letter/humanize actions."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config_files import get_prefs
from app.db import session_scope
from app.llm.local_provider import LocalLLMProvider
from app.llm.roles import LLMSetup, Role, build_setup
from app.main import app
from app.models import ApplicationDocument, JobAnalysis
from app.services import analysis, humanizer, letters, tasks
from app.services.humanizer import (
    INSTR,
    SEP,
    HumanizerUnavailable,
    build_prompt,
    chunk_paragraphs,
    fact_drift,
    humanize,
    split_letter_frame,
    text_language,
)
from app.services.letters import generate_letter, run_humanizer
from app.settings import Settings
from app.web import routes
from tests.conftest import FakeProvider
from tests.test_letters import DRAFT_DE, MISTRAL, Q14, job_id, make_setup  # noqa: F401 - fixture re-export

HUM = "hf.co/jialinyyzz/humanizer:Q6_K"

BODY_DE = (
    "bei Nordlicht Mobility Software habe ich ein Java/Spring-basiertes Validierungsframework mitentwickelt und "
    "mehr als 200 automatisierte Testszenarien umgesetzt. Das ist genau die Arbeit, die Sie in der Anzeige beschreiben."
)


def _body(prompt: str) -> str:
    assert prompt.startswith(INSTR + "\n\n") and prompt.endswith(SEP)
    return prompt[len(INSTR) + 2:-len(SEP)]


def hum_setup(raw_fn=None) -> tuple[LLMSetup, FakeProvider]:
    setup = make_setup()
    if raw_fn:
        setup.provider.raw_fn = raw_fn
    return setup, setup.provider


# --------------------------------------------------------------------------- prompt & provider


def test_prompt_is_byte_exact():
    assert build_prompt("  Hallo Welt.\n") == INSTR + "\n\nHallo Welt.\n\n### Rewritten:\n\n"
    assert INSTR.startswith("Rewrite the text below so it reads like a person wrote it, not a language model.\n\n")
    assert INSTR.endswith("Every fact, number, unit, date, name and quotation must survive unchanged.")


def test_raw_completion_request_body():
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"model": HUM, "response": "  Umgeschrieben.  ", "done": True,
                                         "prompt_eval_count": 50, "eval_count": 7})

    p = LocalLLMProvider("http://ollama:11434", transport=httpx.MockTransport(handler))
    opts = get_prefs().local_llm.options(Role.HUMANIZER).model_dump()
    r = p.complete_raw(model=HUM, prompt=build_prompt("Text."), max_tokens=2048, options=opts)
    assert r.text == "Umgeschrieben." and r.output_tokens == 7
    path, body = seen[0]
    assert path == "/api/generate" and body["raw"] is True and body["stream"] is False
    assert body["prompt"] == build_prompt("Text.")
    o = body["options"]
    assert (o["temperature"], o["top_p"], o["top_k"], o["min_p"], o["repeat_penalty"]) == (1.0, 0.95, 0, 0, 1.0)
    assert o["num_ctx"] == 8192 and o["num_predict"] == 2048 and "stop" not in o  # EOS-only stop


def test_humanizer_is_local_only():
    s = Settings(_env_file=None, local_llm_enabled=True)
    setup = build_setup(s, get_prefs(), local=FakeProvider(name="local"), cloud=FakeProvider(name="anthropic"),
                        cloud_models=("strong", "cheap"))
    t = setup.target(Role.HUMANIZER)
    assert t.model == HUM and t.fallback is None
    off = build_setup(Settings(_env_file=None, local_llm_enabled=True, local_humanizer_enabled=False), get_prefs(),
                      local=FakeProvider(name="local"), cloud=FakeProvider(name="anthropic"), cloud_models=("s", "c"))
    assert off.target(Role.HUMANIZER) is None
    with pytest.raises(HumanizerUnavailable):
        humanize("Text", setup=off)


# --------------------------------------------------------------------------- deterministic helpers


def test_letter_frame_and_chunking():
    head, body, tail = split_letter_frame(DRAFT_DE)
    assert head == "Sehr geehrte Damen und Herren,"
    assert tail == "Mit freundlichen Grüßen\n[Name]"
    assert body.startswith("bei Nordlicht Mobility") and body.endswith("freue ich mich.")
    assert split_letter_frame("Nur ein Absatz.") == ("", "Nur ein Absatz.", "")
    paras = "\n\n".join(" ".join(["wort"] * 150) for _ in range(5))
    chunks = chunk_paragraphs(paras, max_words=350)
    assert [len(c.split()) for c in chunks] == [300, 300, 150]


def test_language_and_drift():
    assert text_language(BODY_DE) == "de"
    assert text_language("I have built a test framework with Java and this is the work you describe.") == "en"
    assert text_language("Java Spring") is None
    w = fact_drift(BODY_DE, BODY_DE.replace("200", "300").replace("Java/Spring", "Kotlin"))
    assert any("200" in x and "missing" in x for x in w) and any("300" in x and "added" in x for x in w)
    assert any("Java/Spring" in x for x in w)
    assert fact_drift(BODY_DE, BODY_DE) == []


# --------------------------------------------------------------------------- humanize()


def test_humanize_keeps_frame_and_rewrites_body_only():
    setup, p = hum_setup(lambda prompt: _body(prompt).replace("Das ist genau die Arbeit", "Genau das"))
    r = humanize(DRAFT_DE, setup=setup)
    assert r.text.startswith("Sehr geehrte Damen und Herren,\n\n") and r.text.endswith("Mit freundlichen Grüßen\n[Name]")
    assert "Sehr geehrte" not in _body(p.last_user["raw"]) and "Grüßen" not in _body(p.last_user["raw"])
    assert p.calls == [(HUM, "raw")] and r.chunks == 1 and r.fallbacks == 0


def test_language_switch_falls_back_to_original_paragraph():
    english = "I helped build a Java/Spring validation framework and more than 200 automated test scenarios for you."
    setup, _ = hum_setup(lambda prompt: english)
    r = humanize(BODY_DE, keep_letter_frame=False, setup=setup)
    assert r.text == BODY_DE and r.fallbacks == 1 and any("English" in w for w in r.warnings)


def test_long_text_is_chunked_under_one_gpu_session():
    para = " ".join(["Ich", "teste", "Software", "und", "die", "Tests", "laufen"] * 30)  # 210 words
    text = "\n\n".join([para] * 4)
    setup, p = hum_setup()
    r = humanize(text, keep_letter_frame=False, setup=setup)
    assert r.chunks == 4 and len(p.calls) == 4 and r.text == text


def test_usage_is_recorded_at_zero_cost():
    setup, _ = hum_setup()
    with session_scope() as s:
        humanize(BODY_DE, setup=setup, session=s)
    with session_scope() as s:
        rec = s.query(JobAnalysis).filter_by(kind="humanize").one()
        assert rec.cost_usd == 0 and rec.model == HUM and rec.job_id is None


# --------------------------------------------------------------------------- letter pipeline


def test_letter_pipeline_with_humanizer_stage(job_id):  # noqa: F811
    setup, p = hum_setup()
    generate_letter(job_id, setup=setup, use_humanizer=True)
    with session_scope() as s:
        docs = s.query(ApplicationDocument).order_by(ApplicationDocument.id).all()
        assert [d.stage for d in docs] == ["draft", "edited", "humanized", "final"]
        draft, edited, hum, final = docs
        assert hum.model == HUM and hum.source_document_id == edited.id and hum.prompt_version == "humanizer-v1"
        assert final.source_document_id == hum.id and final.meta["validated_stage"] == "humanized"
        assert final.validation["validator"].endswith(Q14)  # validated by an instruction model, never the humanizer
    assert [m for m, _ in p.calls] == [Q14, Q14, MISTRAL, HUM, Q14, Q14]  # … validator, then the personalization review


def test_humanizer_off_by_default(job_id):  # noqa: F811
    setup, p = hum_setup()
    generate_letter(job_id, setup=setup)
    assert all(kind != "raw" for _, kind in p.calls)
    with session_scope() as s:
        assert "humanized" not in {d.stage for d in s.query(ApplicationDocument)}


def test_run_humanizer_on_final_letter_revalidates(job_id):  # noqa: F811
    setup, p = hum_setup(lambda prompt: _body(prompt).replace("200", "250"))
    generate_letter(job_id, setup=setup, use_editor=False)
    with session_scope() as s:
        first_final = s.query(ApplicationDocument).filter_by(stage="final").one().id
    p.calls.clear()
    final_id = run_humanizer(job_id, first_final, setup=setup)
    with session_scope() as s:
        final = s.get(ApplicationDocument, final_id)
        hum = s.get(ApplicationDocument, final.source_document_id)
        assert hum.stage == "humanized" and hum.source_document_id == first_final
        assert final.validation["validator"].endswith(Q14)  # writer-only letter → validated with the writer model
        assert any(w["type"] == "humanizer" and "250" in w["detail"] for w in final.validation["warnings"])
    assert [m for m, _ in p.calls][0] == HUM


def test_screening_and_analysis_never_use_humanizer(job_id, monkeypatch):  # noqa: F811
    setup, p = hum_setup()
    monkeypatch.setattr(analysis, "get_llm", lambda: setup)
    analysis.analyze_pending()
    assert p.calls and all(kind != "raw" and model != HUM for model, kind in p.calls)


# --------------------------------------------------------------------------- UI & API


def test_job_page_offers_humanizer_and_routes(job_id, monkeypatch):  # noqa: F811
    setup, p = hum_setup()
    monkeypatch.setattr(letters, "get_llm", lambda: setup)
    monkeypatch.setattr(routes, "get_llm", lambda: setup)
    monkeypatch.setattr(tasks, "run_in_background", lambda name, fn: fn() or True)
    client = TestClient(app)
    client.post(f"/jobs/{job_id}/letters/generate", data={"language": "de", "use_editor": "1", "use_humanizer": "1"})
    page = client.get(f"/jobs/{job_id}").text
    assert "Humanizer pass" in page and "Run humanizer" in page
    with session_scope() as s:
        assert {d.stage for d in s.query(ApplicationDocument)} == {"draft", "edited", "humanized", "final"}
        final_id = max(d.id for d in s.query(ApplicationDocument).filter_by(stage="final"))
    client.post(f"/jobs/{job_id}/letters/{final_id}/humanize")
    with session_scope() as s:
        assert s.query(ApplicationDocument).filter_by(stage="humanized").count() == 2


def test_standalone_page_and_api(monkeypatch):
    setup, p = hum_setup(lambda prompt: _body(prompt).replace("Das ist genau die Arbeit", "Genau diese Arbeit"))
    monkeypatch.setattr(humanizer, "get_llm", lambda: setup)
    monkeypatch.setattr(routes, "get_llm", lambda: setup)
    monkeypatch.setattr(tasks, "run_in_background", lambda name, fn: tasks.run_exclusive(name, fn) is not None)
    client = TestClient(app)
    assert "Humanize text" in client.get("/humanize").text
    client.post("/humanize", data={"text": DRAFT_DE, "keep_frame": "1"})
    page = client.get("/humanize").text
    assert "Genau diese Arbeit" in page and "Copy result" in page and HUM in page

    r = client.post("/api/humanize", json={"text": BODY_DE, "keep_letter_frame": False})
    assert r.status_code == 200 and r.json()["text"].startswith("bei Nordlicht Mobility") and r.json()["chunks"] == 1
    assert client.post("/api/humanize", json={"text": ""}).status_code == 422
    assert "Paste some text" in client.post("/humanize", data={"text": "  "}).text


def test_humanized_text_is_not_rewritten_by_llm_correction(job_id):  # noqa: F811
    from app.llm.schemas import ClaimCheck, FactCheckResult

    setup, p = hum_setup()
    p.responses["FactCheckResult"] = FactCheckResult(claims=[ClaimCheck(
        sentence="Ich habe fünf Jahre Berufserfahrung als Teamleiter.", claim="5 years team lead",
        claimed_level="professional", status="UNSUPPORTED", profile_evidence="", note="not in profile")])
    generate_letter(job_id, setup=setup, use_editor=False, use_humanizer=True)
    kinds = [k for _, k in p.calls]
    assert kinds[kinds.index("raw"):].count("text") == 0  # no LLM correction after the humanizer
    with session_scope() as s:
        final = s.query(ApplicationDocument).filter_by(stage="final").one()
        hum = s.query(ApplicationDocument).filter_by(stage="humanized").one()
        assert "Teamleiter" in hum.content and "Teamleiter" not in final.content  # removed deterministically
        assert "Über ein Gespräch freue ich mich." in final.content  # the rest is the humanizer's text, untouched
        assert not final.validation["corrected"] and final.validation["removed_sentences"]
