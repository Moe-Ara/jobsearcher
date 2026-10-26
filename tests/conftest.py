import os

from tests import _TMP  # noqa: F401 - sets the temporary DATABASE_URL before app modules create the engine

os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["LLM_PROVIDER"] = "heuristic"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["IMAP_HOST"] = ""
os.environ["JOBSEARCHER_DISABLE_LATEX"] = "1"  # fast built-in renderer; tests/test_latex.py enables LaTeX itself

import pytest  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.config_files import config_store  # noqa: E402
from app.db import run_migrations, session_scope  # noqa: E402
from app.llm.base import LLMError, LLMProvider, LLMResponse, TextResponse  # noqa: E402
from app.llm.factory import LLMSetup, reset_llm  # noqa: E402
from app.llm.schemas import JobFitAnalysis, JobScreening  # noqa: E402
from app.models import (  # noqa: E402
    ApplicationDocument,
    ApplicationNotes,
    ApplicationPackage,
    EmailMessage,
    Job,
    JobAnalysis,
    JobSource,
    PurgedPosting,
    SourceRun,
    StatusEvent,
    TargetCompany,
)
from app.services.ingest import invalidate_alias_cache  # noqa: E402

config_store.reload()
run_migrations()


@pytest.fixture(autouse=True)
def clean_db():
    with session_scope() as s:
        s.execute(Job.__table__.update().values(current_analysis_id=None))
        for model in (ApplicationPackage, ApplicationNotes, ApplicationDocument, StatusEvent, JobAnalysis, JobSource, Job, EmailMessage, SourceRun, TargetCompany, PurgedPosting):
            s.execute(delete(model))
    invalidate_alias_cache()
    reset_llm()
    yield


@pytest.fixture(autouse=True)
def templates_dir(tmp_path, monkeypatch):
    """Each test gets an empty letter-template folder (never the user's real config/letter_templates)."""
    from app.services import letter_templates

    d = tmp_path / "letter_templates"
    monkeypatch.setattr(letter_templates, "_override_dir", d)
    return d


@pytest.fixture(autouse=True)
def no_web_research(monkeypatch):
    """The research agent reads real websites; tests never do (tests/test_research.py uses a fake web)."""
    monkeypatch.setattr(config_store.prefs().application, "research_enabled", False)


@pytest.fixture(autouse=True)
def no_liveness_network(monkeypatch, request):
    """The "is the posting still open?" check reads real job pages; tests never do (tests/test_liveness.py tests it)."""
    if request.module.__name__.endswith("test_liveness"):
        return
    from app.services import liveness

    monkeypatch.setattr(liveness, "check", lambda job, http=None: liveness.Liveness("unknown", "not checked in tests"))


@pytest.fixture
def prefs():
    return config_store.prefs()


@pytest.fixture
def profile():
    return config_store.profile()


def fit_result(**overrides) -> JobFitAnalysis:
    data = dict(
        role_family="Industrial AI", sector="Industrial manufacturing", seniority="Mid",
        technical_match=8, experience_match=7, career_match=9, location_match=9, overall_score=8.3, confidence=0.9,
        strong_matches=[
            {"skill": "Python", "evidence": "direct_professional", "profile_evidence": "Python"},
            {"skill": "Machine learning", "evidence": "academic", "profile_evidence": "Machine learning"},
            {"skill": "OpenShift", "evidence": "direct_professional", "profile_evidence": "OpenShift operator"},
        ],
        partial_matches=[{"skill": "Azure", "evidence": "transferable", "profile_evidence": "Docker"}],
        critical_gaps=[], nice_to_have_gaps=["Azure"], why_good="good", why_bad="bad",
        summary="Strong overlap.", recommendation="PRIORITY_APPLY", defence_related=False, defence_reason="",
        years_experience_required=3, language_requirements=[], remote_status="hybrid", suspicious_instructions=False,
    )
    data.update(overrides)
    return JobFitAnalysis.model_validate(data)


def screening_result(**overrides) -> JobScreening:
    data = dict(
        likely_relevant=True, relevance_score=8, role_family="Industrial AI", sector="Industrial software",
        seniority="Junior/Mid", candidate_overlap=["Python", "Machine learning", "Docker"],
        possible_critical_gaps=["Azure"], profession_unrelated=False, defence_related=False, defence_reason="",
        confidence=0.87, reason="Applied AI role with strong Python/ML overlap.",
    )
    data.update(overrides)
    return JobScreening.model_validate(data)


class FakeProvider(LLMProvider):
    """Deterministic stand-in for any backend. Responses by schema; text responses from a queue/callable."""

    def __init__(self, fit: JobFitAnalysis | None = None, triage: JobScreening | None = None, *, name: str = "fake",
                 is_paid: bool = True, fail: bool | Exception = False):
        self.name = name
        self.is_paid = is_paid
        self.calls: list[tuple[str, str]] = []
        self.fit = fit or fit_result()
        self.triage = triage or screening_result()
        self.responses: dict[str, object] = {}  # schema name → parsed object (or callable(user) → object)
        self.texts: list[str] = []
        self.fail = fail
        self.last_user: dict[str, str] = {}

    def default_models(self):
        return "fake-strong", "fake-cheap"

    def _maybe_fail(self):
        if self.fail:
            raise self.fail if isinstance(self.fail, Exception) else LLMError(f"{self.name} failure")

    def structured(self, *, model, system, user, schema, max_tokens=4000, effort=None, options=None):
        self.calls.append((model, schema.__name__))
        self.last_user[schema.__name__] = user
        self._maybe_fail()
        if schema.__name__ in self.responses:
            r = self.responses[schema.__name__]
            parsed = r(user) if callable(r) else r
        else:
            parsed = self.fit if schema is JobFitAnalysis else self.triage
        return LLMResponse(parsed=parsed, model=model, input_tokens=1000, output_tokens=200)

    def text(self, *, model, system, user, max_tokens=2000, effort=None, options=None):
        self.calls.append((model, "text"))
        self.last_user["text"] = user
        self._maybe_fail()
        text = self.texts.pop(0) if self.texts else "Sehr geehrte Damen und Herren,\n\nText.\n\nMit freundlichen Grüßen\n[Name]"
        return TextResponse(text=text, model=model, input_tokens=800, output_tokens=300)

    def complete_raw(self, *, model, prompt, max_tokens=2048, options=None):
        """Humanizer stand-in: `raw_fn(prompt)` → completion (default: echo the text between instruction and separator)."""
        self.calls.append((model, "raw"))
        self.last_user["raw"] = prompt
        self._maybe_fail()
        fn = getattr(self, "raw_fn", None)
        text = fn(prompt) if fn else prompt.split("\n\n", 3)[-1].rsplit("\n\n### Rewritten:", 1)[0]
        return TextResponse(text=text, model=model, input_tokens=400, output_tokens=380)


@pytest.fixture
def fake_llm():
    provider = FakeProvider()
    setup = LLMSetup.single(provider, "fake-strong", "fake-cheap")
    setup.provider = provider  # convenience handle for assertions
    return setup
