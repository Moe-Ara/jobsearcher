"""Every letter for a job in Germany states German citizenship (no work permit) and willingness to relocate —
the writer is asked to, and a deterministic step inserts whatever it forgot. Never for jobs abroad."""

from types import SimpleNamespace

from app.config_files import get_prefs, get_profile
from app.db import session_scope
from app.llm.letter_prompts import writer_system_de, writer_system_en
from app.models import ApplicationDocument, Job
from app.services.letters import ensure_mobility_statement, generate_letter, mobility_applies
from tests.test_letters import job_id, make_setup  # noqa: F401 - fixture re-export

LETTER = ("Sehr geehrte Damen und Herren,\n\nich entwickle Software mit Java.\n\n"
          "Über ein Gespräch freue ich mich.\n\nMit freundlichen Grüßen\nalex beispiel")


def test_both_sentences_inserted_before_the_closing_paragraph():
    w = get_prefs().writing
    out, added = ensure_mobility_statement(LETTER, "de", SimpleNamespace(country="Germany"), w)
    assert added == [w.work_permit_sentence_de, w.relocation_sentence_de]
    paragraphs = out.split("\n\n")
    assert paragraphs[2] == " ".join(added) and paragraphs[3].startswith("Über ein Gespräch")
    assert out.endswith("Mit freundlichen Grüßen\nalex beispiel")


def test_only_the_missing_part_is_added_and_never_twice():
    w = get_prefs().writing
    job = SimpleNamespace(country="Germany")
    has_citizenship = LETTER.replace("ich entwickle", "Als deutscher Staatsbürger brauche ich keine Arbeitserlaubnis. Ich entwickle")
    out, added = ensure_mobility_statement(has_citizenship, "de", job, w)
    assert added == [w.relocation_sentence_de]
    assert ensure_mobility_statement(out, "de", job, w)[1] == []


def test_english_letters_get_english_sentences():
    w = get_prefs().writing
    letter = "Dear Hiring Team,\n\nI build software.\n\nI look forward to a conversation.\n\nKind regards\nalex beispiel"
    out, added = ensure_mobility_statement(letter, "en", SimpleNamespace(country="Germany"), w)
    assert added == [w.work_permit_sentence_en, w.relocation_sentence_en] and "German citizen" in out


def test_not_for_jobs_abroad():
    w = get_prefs().writing
    for country in ("UAE", "Saudi Arabia", None):
        job = SimpleNamespace(country=country)
        assert not mobility_applies(job, w)
        assert ensure_mobility_statement(LETTER, "de", job, w) == (LETTER, [])


def test_writer_is_asked_for_it_only_when_it_applies():
    w = get_prefs().writing
    assert w.work_permit_sentence_de in writer_system_de(w, mobility=True)  # your own configured facts
    assert "ARBEITSERLAUBNIS UND MOBILITÄT" not in writer_system_de(w, mobility=False)
    assert w.relocation_sentence_en in writer_system_en(w, mobility=True)
    lu = writer_system_en(w, mobility=True, country="Luxembourg")
    assert w.country_sentences["Luxembourg"]["work_permit_en"] in lu and "(the role is in Luxembourg)" in lu
    blank = w.model_copy(update={"work_permit_sentence_en": "", "relocation_sentence_en": "", "country_sentences": {}})
    assert "WORK PERMIT AND MOBILITY" not in writer_system_en(blank, mobility=True)  # nothing configured: no claim


def test_profile_carries_the_facts_for_the_fact_check():
    text = get_profile().as_prompt_text()
    assert "German citizen" in text and "relocate anywhere in Germany" in text


def test_pipeline_final_letter_always_has_it(job_id):  # noqa: F811
    with session_scope() as s:
        assert s.get(Job, job_id).country == "Germany"
    setup = make_setup()
    generate_letter(job_id, setup=setup)
    with session_scope() as s:
        final = s.query(ApplicationDocument).filter_by(stage="final").one()
        assert "Staatsbürger" in final.content and "Umzug" in final.content
        assert final.validation.get("mobility_added"), "the fixture draft lacks it, so the guarantee step added it"
