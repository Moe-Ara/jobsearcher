"""Local multi-model pipeline: configuration, routing, LocalLLMProvider (mocked Ollama), fallback, batching."""

import json

import httpx
import pytest

from app.config_files import get_prefs
from app.db import session_scope
from app.enums import AnalysisStatus, Recommendation, SourceType
from app.llm.base import LLMError, LLMUnavailable
from app.llm.local_provider import HEALTH, LocalLLMProvider
from app.llm.roles import Role, build_setup
from app.llm.schemas import JobScreening
from app.models import ApplicationDocument, Job, JobAnalysis
from app.services import analysis
from app.services.ingest import ingest_posting
from app.services.llm_calls import llm_call
from app.settings import Settings
from app.sources.base import JobPosting
from tests.conftest import FakeProvider, screening_result

Q8 = "hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M"
Q14 = "hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M"
MISTRAL = "mistral-small-3.1-text:q4_k_s"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **{"local_llm_enabled": True, **kw})


def local_setup(local=None, cloud=None, **kw):
    local = local or FakeProvider(name="local", is_paid=False)
    return build_setup(settings(**kw), get_prefs(), local=local, cloud=cloud,
                       cloud_models=("cloud-strong", "cloud-cheap") if cloud else None)


# --------------------------------------------------------------------------- configuration & routing


def test_configuration_loading(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_ENABLED", "true")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://gpu-box:11434")
    monkeypatch.setenv("LOCAL_GERMAN_EDITOR_ENABLED", "false")
    monkeypatch.setenv("CLOUD_FALLBACK_ENABLED", "true")
    s = Settings(_env_file=None)
    assert s.local_llm_enabled and s.local_llm_base_url == "http://gpu-box:11434"
    assert not s.local_german_editor_enabled and s.cloud_fallback_enabled
    assert (s.local_screen_model, s.local_analysis_model, s.local_writer_model, s.local_german_editor_model) == (Q8, Q14, Q14, MISTRAL)
    assert not Settings(_env_file=None, local_llm_enabled=False).local_llm_enabled
    opts = get_prefs().local_llm.options("editor")
    assert opts.num_ctx <= 8192  # 24B editor kept at ≤ 8K context


def test_profile_keeps_extra_fields(profile):
    text = profile.as_prompt_text()
    assert "Hochschule Nordstadt" in text and "location_preferences" in text  # fields not in the Pydantic model survive


def test_model_routing_local():
    setup = local_setup()
    route = {r: (setup.target(r).provider.name, setup.target(r).model) for r in Role if setup.target(r)}
    assert route[Role.SCREEN] == ("local", Q8) and route[Role.EXTRACT] == ("local", Q8)
    assert route[Role.ANALYSIS] == ("local", Q14) and route[Role.WRITER] == ("local", Q14)
    assert route[Role.EDITOR] == ("local", MISTRAL)
    # roles sharing a model share num_ctx (no reload when switching analysis ↔ writing)
    assert setup.target(Role.ANALYSIS).options["num_ctx"] == setup.target(Role.WRITER).options["num_ctx"] == 12288
    assert setup.target(Role.EDITOR).options["num_ctx"] == 8192
    assert setup.target(Role.SCREEN).options["think"] is False
    assert all(setup.target(r).fallback is None for r in Role)


def test_model_routing_disabled_roles_and_cloud():
    cloud = FakeProvider(name="cloud")
    setup = local_setup(cloud=cloud, local_german_editor_enabled=False, local_analysis_enabled=False)
    assert setup.target(Role.EDITOR).provider is cloud and setup.target(Role.EDITOR).model == "cloud-strong"
    assert setup.target(Role.ANALYSIS).provider is cloud
    assert setup.target(Role.SCREEN).provider.name == "local" and setup.target(Role.SCREEN).fallback is None
    no_cloud = local_setup(local_german_editor_enabled=False)
    assert no_cloud.target(Role.EDITOR) is None
    with_fallback = local_setup(cloud=cloud, cloud_fallback_enabled=True)
    assert with_fallback.target(Role.SCREEN).fallback.provider is cloud
    assert with_fallback.target(Role.SCREEN).fallback.model == "cloud-cheap"
    cloud_only = build_setup(settings(local_llm_enabled=False), get_prefs(), local=None, cloud=cloud, cloud_models=("s", "c"))
    assert {cloud_only.target(r).provider.name for r in Role if r != Role.HUMANIZER} == {"cloud"}
    assert cloud_only.target(Role.HUMANIZER) is None  # local-only role: no cloud equivalent


def test_validator_defaults_to_writer_model():
    setup = local_setup()
    v = setup.retarget(setup.target(Role.EDITOR), Role.VALIDATOR)
    assert v.model == Q14 and v.role == Role.VALIDATOR  # not the slow, partly CPU-offloaded 24B editor
    assert v.options["temperature"] == 0.0 and v.options["think"] is True
    assert setup.retarget(setup.target(Role.WRITER), Role.VALIDATOR).model == Q14
    explicit = local_setup(local_validator_model=Q8)
    assert explicit.retarget(explicit.target(Role.EDITOR), Role.VALIDATOR).model == Q8


# --------------------------------------------------------------------------- LocalLLMProvider (mocked Ollama)


def _screen_json(**kw) -> str:
    return screening_result(**kw).model_dump_json()


def provider_with(handler) -> LocalLLMProvider:
    return LocalLLMProvider("http://ollama:11434", transport=httpx.MockTransport(handler))


def ok(content: str, **extra) -> httpx.Response:
    return httpx.Response(200, json={"model": extra.get("model", Q8), "message": {"role": "assistant", "content": content},
                                     "done": True, "prompt_eval_count": 900, "eval_count": 120})


def test_local_structured_screener_parsing():
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return ok("<think>\n\n</think>\n" + _screen_json())

    p = provider_with(handler)
    resp = p.structured(model=Q8, system="S", user="U", schema=JobScreening, max_tokens=1200,
                        options={"num_ctx": 8192, "keep_alive": "10m", "temperature": 0.1, "think": False})
    assert isinstance(resp.parsed, JobScreening) and resp.parsed.relevance_score == 8
    assert resp.input_tokens == 900 and resp.output_tokens == 120
    body = seen[0]
    assert body["format"]["type"] == "object" and "$defs" not in json.dumps(body["format"])
    assert body["keep_alive"] == "10m" and body["think"] is False and body["stream"] is False
    assert body["options"] == {"num_ctx": 8192, "num_predict": 1200, "temperature": 0.1}


def test_local_think_param_unsupported_falls_back_to_soft_switch():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": f'"{Q8}" does not support thinking'})
        return ok(_screen_json())

    p = provider_with(handler)
    p.structured(model=Q8, system="S", user="U", schema=JobScreening, options={"think": False})
    assert "think" not in bodies[1] and bodies[1]["messages"][-1]["content"].endswith("/no_think")
    p.structured(model=Q8, system="S", user="U2", schema=JobScreening, options={"think": False})
    assert len(bodies) == 3 and "think" not in bodies[2]  # remembered per model


def test_qwen3_text_calls_use_non_thinking_prefill():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return ok("<think>\n\n</think>\n\nSehr geehrte Damen und Herren,")

    p = provider_with(handler)
    out = p.text(model=Q14, system="S", user="U", options={"think": False})
    assert out.text == "Sehr geehrte Damen und Herren,"
    assert bodies[0]["messages"][-1] == {"role": "assistant", "content": "<think>\n\n</think>\n\n"}
    p.text(model=MISTRAL, system="S", user="U", options={"think": None})
    assert bodies[1]["messages"][-1]["role"] == "user" and "think" not in bodies[1]  # editor: no Qwen-specific tricks


def test_local_invalid_json_gets_one_repair_round():
    replies = iter(["not json at all", _screen_json(relevance_score=3)])
    p = provider_with(lambda req: ok(next(replies)))
    assert p.structured(model=Q8, system="S", user="U", schema=JobScreening).parsed.relevance_score == 3


def test_local_provider_failures():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailable, match="unreachable"):
        provider_with(refuse).structured(model=Q8, system="S", user="U", schema=JobScreening)
    assert "unreachable" in HEALTH.last_error

    with pytest.raises(LLMUnavailable, match="not installed"):
        provider_with(lambda r: httpx.Response(404, json={"error": "model not found"})).text(model=Q14, system="S", user="U")

    with pytest.raises(LLMError) as exc:
        provider_with(lambda r: httpx.Response(500, json={"error": "CUDA out of memory"})).text(model=MISTRAL, system="S", user="U")
    assert not isinstance(exc.value, LLMUnavailable) and "out of memory" in str(exc.value)

    with pytest.raises(LLMError, match="Invalid structured output"):
        provider_with(lambda r: ok("still not json")).structured(model=Q8, system="S", user="U", schema=JobScreening)


def test_local_health():
    def handler(request):
        return {
            "/api/version": httpx.Response(200, json={"version": "0.12.0"}),
            "/api/tags": httpx.Response(200, json={"models": [{"name": Q8}, {"name": Q14}]}),
            "/api/ps": httpx.Response(200, json={"models": [{"name": Q8, "size_vram": 7_000_000_000}]}),
        }[request.url.path]

    p = provider_with(handler)
    h = p.health()
    assert h["reachable"] and h["version"] == "0.12.0" and h["loaded"][0]["size_vram_gb"] == 7.0
    assert p.is_installed(Q8, h["installed"]) and not p.is_installed(MISTRAL, h["installed"])


# --------------------------------------------------------------------------- cost accounting & fallback


def _job(s, title="Applied AI Engineer", description="Python, machine learning and Docker for industrial AI. " * 6,
         company="Acme"):
    return ingest_posting(s, JobPosting(source="manual", source_type=SourceType.MANUAL, title=title,
                                        company=company, location="Stuttgart", description=description)).job


def test_local_calls_cost_nothing_but_record_tokens_and_latency():
    setup = local_setup()
    with session_scope() as s:
        job = _job(s)
        res = llm_call(s, setup.target(Role.SCREEN), kind="screen", job_id=job.id, prompt_version="v", profile_hash="p",
                       system="S", user="U", schema=JobScreening)
        assert res.record.provider == "local" and res.record.model == Q8
        assert res.record.cost_usd == 0 and res.record.input_tokens == 1000 and res.record.duration_ms >= 0


def test_cloud_fallback_on_local_failure(prefs, monkeypatch):
    monkeypatch.setitem(prefs.llm.prices, "cloud-cheap", [1.0, 5.0])
    local = FakeProvider(name="local", is_paid=False, fail=LLMUnavailable("server down"))
    cloud = FakeProvider(name="cloud")
    setup = local_setup(local=local, cloud=cloud, cloud_fallback_enabled=True)
    with session_scope() as s:
        job = _job(s)
        res = llm_call(s, setup.target(Role.SCREEN), kind="screen", job_id=job.id, prompt_version="v", profile_hash="p",
                       system="S", user="U", schema=JobScreening)
        assert res.target.provider is cloud and res.record.cost_usd > 0
        recs = s.query(JobAnalysis).order_by(JobAnalysis.id).all()
        assert [(r.provider, bool(r.error)) for r in recs] == [("local", True), ("cloud", False)]
    no_fallback = local_setup(local=local, cloud=cloud, cloud_fallback_enabled=False)
    with session_scope() as s, pytest.raises(LLMUnavailable):
        llm_call(s, no_fallback.target(Role.SCREEN), kind="screen", job_id=None, prompt_version="v", profile_hash="p",
                 system="S", user="U2", schema=JobScreening)
    assert cloud.calls == [("cloud-cheap", "JobScreening")]  # cloud only used as fallback


# --------------------------------------------------------------------------- two-phase batch pipeline


def test_batch_screens_everything_first_then_deep_analyses_promising(monkeypatch):
    local = FakeProvider(name="local", is_paid=False)
    # screener: the clerk job is a poor match, the others are promising
    local.responses["JobScreening"] = lambda user: screening_result(
        relevance_score=2, likely_relevant=False, candidate_overlap=["Python", "OpenShift"]
    ) if "Logistics Clerk" in user else screening_result()
    setup = local_setup(local=local)
    monkeypatch.setattr(analysis, "get_llm", lambda: setup)
    with session_scope() as s:
        for i, t in enumerate(("AI Engineer", "Logistics Clerk Data Entry", "Robotics Engineer", "MLOps Engineer")):
            _job(s, title=t, company=f"Company {i}", description=f"{t}: Python, machine learning, Docker for industrial AI. " * 5)

    result = analysis.analyze_pending()
    kinds = [c[1] for c in local.calls]
    assert kinds == ["JobScreening"] * 4 + ["JobFitAnalysis"] * 3  # one model swap per run, no alternation
    assert {c[0] for c in local.calls if c[1] == "JobScreening"} == {Q8}
    assert {c[0] for c in local.calls if c[1] == "JobFitAnalysis"} == {Q14}
    assert all(m != MISTRAL for m, _ in local.calls)  # the German editor is never part of screening/analysis
    assert result["screened_promising"] == 3
    with session_scope() as s:
        low = s.query(Job).filter(Job.title.like("Logistics%")).one()
        assert low.analysis_status == AnalysisStatus.SCREENED and low.recommendation == Recommendation.LOW_PRIORITY
        assert low.current_analysis.kind == "screen"
        assert low.current_analysis.result["candidate_overlap"] == ["Python"]  # "OpenShift" not in profile → dropped
        assert low.fit_score < 6.5
        assert s.query(ApplicationDocument).count() == 0  # no cover letters as a side effect
        # deep analysis received the screener output as delimited notes
    assert "<screening_notes>" in local.last_user["JobFitAnalysis"]


def test_unavailable_backend_stops_batch_without_burning_attempts(monkeypatch):
    local = FakeProvider(name="local", is_paid=False, fail=LLMUnavailable("Local inference server unreachable"))
    setup = local_setup(local=local)
    monkeypatch.setattr(analysis, "get_llm", lambda: setup)
    with session_scope() as s:
        job_id = _job(s).id
        _job(s, title="Robotics Engineer", company="Other GmbH")
    result = analysis.analyze_pending()
    assert "unreachable" in result["stopped"] and len(local.calls) == 1
    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.analysis_status == AnalysisStatus.PENDING and job.analysis_attempts == 0


def test_preflight_skips_run_when_local_model_missing(monkeypatch):
    class Local(FakeProvider):
        def cached_health(self):
            return {"reachable": True, "installed": [Q8], "loaded": [], "error": None}

        def is_installed(self, model, installed):
            return model in installed

    local = Local(name="local", is_paid=False)
    setup = local_setup(local=local)
    monkeypatch.setattr(analysis, "get_llm", lambda: setup)
    with session_scope() as s:
        _job(s)
    result = analysis.analyze_pending()
    assert "not installed" in result["stopped"] and Q14 in result["stopped"] and local.calls == []


def test_fine_tuned_models_get_the_qwen3_think_switch():
    from app.llm.local_provider import is_qwen3

    assert is_qwen3("hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M") and is_qwen3("jobsearcher-screen-ft:latest")
    assert not is_qwen3("mistral-small-3.1-text:q4_k_s")
