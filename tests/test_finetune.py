"""Fine-tuning pipeline helpers: evaluation metrics, label correction, public-letter quality gates."""

import json

import pytest

from finetune.evaluate import auc, letter_metrics
from finetune.prepare_screen import correct


def test_auc():
    assert auc([9, 8], [2, 1]) == 1.0
    assert auc([5], [5]) == 0.5
    assert auc([1], [9]) == 0.0


def test_letter_penalty_weights_unsupported_claims_most():
    clean = letter_metrics({"rounds": [{"claims": []}], "warnings": []}, "word " * 250, "de")
    bad = letter_metrics({"rounds": [{"claims": [{"status": "UNSUPPORTED"}, {"status": "AMBIGUOUS"}]}],
                          "warnings": [{"type": "tone"}, {"type": "salutation"}], "outline_added": ["degree", "closing"]},
                         "I am writing to express my interest " + "word " * 250, "en")
    assert clean["penalty"] == 0
    assert bad["unsupported"] == 1 and bad["invented_salutation"] == 1 and bad["outline_added"] == 1
    assert bad["penalty"] == 3 + 1.5 + 1 + 1 + 2 + 0.5


@pytest.mark.parametrize("label,teacher,expected", [
    ("GREAT", {"relevance_score": 5, "likely_relevant": False, "reason": "x"}, (8, True)),
    ("BAD", {"relevance_score": 7, "likely_relevant": True, "reason": "x"}, (3, False)),
    ("Potential Fit", {"relevance_score": 9, "likely_relevant": True, "reason": "keep"}, (6, True)),
])
def test_teacher_answers_are_corrected_to_the_human_label(label, teacher, expected):
    out = correct(teacher, label)
    assert (out["relevance_score"], out["likely_relevant"]) == expected
    assert (out["reason"] == "keep") == (teacher["likely_relevant"] == expected[1])


def test_public_letters_must_pass_the_pipeline_gates(tmp_path, monkeypatch):
    from finetune import prepare_writer

    raw = tmp_path / "raw"
    raw.mkdir()
    base = {"Job Title": "Data Engineer", "Preferred Qualifications": "Python, SQL", "Hiring Company": "Acme",
            "Applicant Name": "Sam Doe", "Past Working Experience": "Analyst at X", "Current Working Experience": "",
            "Skillsets": "Python, SQL", "Qualifications": "BSc"}
    good = " ".join(["I build data pipelines in Python and SQL and enjoy clean, tested code."] * 10)
    rows = [{**base, "Cover Letter": "Dear Hiring Manager,\n\n" + good + "\n\nSincerely,\nSam"},
            {**base, "Cover Letter": "I am writing to express my interest. " + good},  # stock phrase
            {**base, "Cover Letter": good + " I also run Kubernetes clusters."},       # tool not in the inputs
            {**base, "Cover Letter": good + " I have 12 years of experience."}]        # invented number
    (raw / "coverletters.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(prepare_writer, "FT", tmp_path)
    out = prepare_writer.public_examples()
    assert len(out) == 1
    answer = json.loads(out[0]["messages"][2]["content"])
    assert answer["letter"].startswith("Dear Hiring Team,") and answer["letter"].endswith("Kind regards\nSam Doe")
    assert "Sincerely" not in answer["letter"] and "<JOB_DESCRIPTION>" in out[0]["messages"][1]["content"]
