"""Tailored CV — part of the application agent.

The master CV (config/cv.yaml) holds the facts. Tailoring for one posting only SELECTS, ORDERS and EMPHASISES:
  - bullets within each role and the projects are ordered by relevance to the posting's keywords (roles and
    education stay reverse-chronological — recruiters and ATS parsers expect that);
  - skill rows and items are ordered by relevance;
  - posting keywords the candidate genuinely has (profile.yaml wording) but the CV never names are surfaced in a
    "further relevant skills" row — gaps are never stuffed in;
  - the profile statement is written by the writer model from the master CV and checked deterministically
    (unknown technologies, numbers that are not in the CV, years-of-experience claims → sentence dropped).
The result is a plain dict (stored with the application package) that the renderers turn into PDF / LaTeX / text.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.config_files import CandidateProfile
from app.llm.application_prompts import CV_STATEMENT_VERSION, cv_statement_system, cv_statement_user
from app.llm.factory import LLMSetup, Role
from app.llm.schemas import CvProfileStatement
from app.services import tasks
from app.services.evidence import TECH_TERMS, split_sentences
from app.services.keywords import Keyword, job_keywords, literal_in, relevance, term_in, usable_term
from app.services.llm_calls import CallResult, llm_call
from app.settings import get_settings

log = logging.getLogger(__name__)

MAX_BULLETS_PER_ROLE = 5
MAX_EXTRA_SKILLS = 8


# --------------------------------------------------------------------------- master CV model

Text = dict[str, str] | str


def tr(value: Text | None, lang: str) -> str:
    """Text in `lang` from a {de, en} mapping (falls back to the other language), or a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.get(lang) or value.get("en") or value.get("de") or ""


def item_text(item: str, lang: str) -> str:
    """Skill item 'Klassifikation|Classification' → German or English part; plain items are language-neutral."""
    if "|" in item:
        de, en = item.split("|", 1)
        return de.strip() if lang == "de" else en.strip()
    return item.strip()


class Bullet(BaseModel):
    id: str
    de: str = ""
    en: str = ""
    core: bool = False
    keywords: list[str] = Field(default_factory=list)


class Role_(BaseModel):  # "Role" is the LLM role enum
    id: str
    title: Text
    org: str
    place: str = ""
    start: str
    end: str
    detail: Text | None = None
    bullets: list[Bullet] = Field(default_factory=list)


class Thesis(BaseModel):
    title: str
    label: Text = "Thesis"
    summary: Text | None = None


class EduProject(BaseModel):
    label: Text = "Project"
    title: Text


class EducationEntry(BaseModel):
    id: str
    title: Text
    org: str
    start: str
    end: str
    grade: Text | None = None
    focus: Text | None = None
    thesis: Thesis | None = None
    project: EduProject | None = None  # e.g. the study project from the transcript
    modules: Text | None = None  # selected modules with grades
    keywords: list[str] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    de: str = ""
    en: str = ""
    keywords: list[str] = Field(default_factory=list)


class SkillRow(BaseModel):
    id: str
    label: Text
    items: list[str]


class Link(BaseModel):
    label: str
    url: str


class Contact(BaseModel):
    email: str = ""
    phone: str = ""
    street: str = ""  # only on the letter (sender block); empty = left out
    postal_code: str = ""
    location: str = ""
    links: list[Link] = Field(default_factory=list)


class PersonalLine(BaseModel):
    de: str = ""
    en: str = ""
    only: str | None = None  # "de" → shown on German CVs only


class MasterCV(BaseModel):
    name: str
    headline: Text = ""
    contact: Contact = Field(default_factory=Contact)
    personal: list[PersonalLine] = Field(default_factory=list)
    sections: dict[str, Text] = Field(default_factory=dict)
    experience: list[Role_] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[SkillRow] = Field(default_factory=list)
    extra_skills_label: Text = {"de": "Weitere Kenntnisse", "en": "Further skills"}
    languages: list[Text] = Field(default_factory=list)

    def vocabulary(self) -> list[str]:
        """Skill terms the CV names (both languages) — the literal vocabulary for keyword matching."""
        out: list[str] = []
        for row in self.skills:
            for item in row.items:
                out += [item_text(item, "de"), item_text(item, "en")]
        for r in self.experience:
            for b in r.bullets:
                out += b.keywords
        for p in self.projects:
            out += p.keywords
        for e in self.education:
            out += e.keywords
        return [x for x in dict.fromkeys(out) if x]

    def full_text(self, lang: str) -> str:
        """Everything the CV can say, in one language (used for grounding checks and the writer prompt)."""
        return render_text(build_cv(self, lang, [], statement=""), with_extras=False)


class CvError(ValueError):
    pass


def cv_path() -> Path:
    return get_settings().config_dir / "cv.yaml"


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> MasterCV:
    with open(path, encoding="utf-8") as fh:
        return MasterCV.model_validate(yaml.safe_load(fh) or {})


def get_master_cv() -> MasterCV:
    path = cv_path()
    if not path.is_file():
        raise CvError(f"No master CV found at {path} — create config/cv.yaml first")
    return _load(str(path), path.stat().st_mtime)


# --------------------------------------------------------------------------- tailoring


def _score(text: str, keywords: list[Keyword]) -> int:
    return relevance(text, keywords) if keywords else 0


# Relevance-weighted cutting, least valuable first, until the CV fits its page budget (never mechanical "oldest
# first"): (optional bullets per role, show the modules line, number of projects).
FIT_LEVELS = ((3, "relevant", 3), (2, "relevant", 3), (1, "relevant", 3), (0, "relevant", 3), (0, "relevant", 2),
              (0, "never", 2), (0, "never", 1))
MAX_PAGES = 1


_GENERIC = {"projekt", "projekte", "project", "projects", "erfahrung", "experience", "kenntnisse", "knowledge", "skills",
            "skill", "arbeit", "work", "the", "der", "die", "das", "mit", "with", "and", "und", "bei", "at", "my", "mein"}


def core_words(term: str) -> list[str]:
    """"Unity-Projekt" → ["unity"], "Robotik-Erfahrung" → ["robotik"] (the words that identify a CV item)."""
    return [w for w in re.findall(r"[a-zäöüß0-9+#.]+", term.lower()) if w not in _GENERIC and len(w) > 1]


def _excluded(text: str, exclude: list[str]) -> bool:
    low = text.lower()
    for term in exclude:
        words = core_words(term)
        if words and all(re.search(rf"(?<![a-zäöüß0-9]){re.escape(w)}", low) for w in words):
            return True
    return False


def build_cv(master: MasterCV, lang: str, keywords: list[Keyword], statement: str,
             extra_skills: list[str] | None = None, level: int = 0, exclude: list[str] | None = None) -> dict[str, Any]:
    """Select + order the master CV for these keywords. Pure function (no LLM). `level` = how much optional content
    is cut to fit the page (see FIT_LEVELS). `exclude`: bullets, projects and skills containing one of these terms are
    left out (your feedback / learned rules — even core bullets)."""
    max_optional, modules_mode, max_projects = FIT_LEVELS[min(level, len(FIT_LEVELS) - 1)]
    exclude = [t for t in (exclude or []) if t and t.strip()]
    sec = {k: tr(v, lang) for k, v in master.sections.items()}
    experience = []
    for role in master.experience:
        scored = [(b, _score(" ".join([b.de, b.en, *b.keywords]), keywords)) for b in role.bullets
                  if not _excluded(f"{b.de} {b.en} {' '.join(b.keywords)}", exclude)]
        optional = sorted(((b, s) for b, s in scored if not b.core and s > 0), key=lambda bs: -bs[1])[:max_optional]
        keep = {id(b) for b, _ in optional}
        chosen = [(b, s) for b, s in scored if b.core or id(b) in keep]
        chosen.sort(key=lambda bs: -bs[1])  # stable: master order among equals
        experience.append({
            "id": role.id, "title": tr(role.title, lang), "org": role.org,
            "place": "" if role.place.lower() in role.org.lower() else role.place,  # "Seeblick Kommunal-IT, Seeblick"
            "dates": f"{role.start} - {role.end}", "detail": tr(role.detail, lang),
            "bullets": [getattr(b, lang) or b.en or b.de for b, _ in chosen[:MAX_BULLETS_PER_ROLE]],
            "relevance": sum(s for _, s in chosen),
        })
    education = [{
        "id": e.id, "title": tr(e.title, lang), "org": e.org, "dates": f"{e.start} - {e.end}",
        "grade": tr(e.grade, lang), "focus": tr(e.focus, lang),
        "thesis": ({"label": tr(e.thesis.label, lang), "title": e.thesis.title, "summary": tr(e.thesis.summary, lang)}
                   if e.thesis else None),
        "project": {"label": tr(e.project.label, lang), "title": tr(e.project.title, lang)} if e.project else None,
        "modules": ({"label": "Module" if lang == "de" else "Modules", "text": tr(e.modules, lang)}
                    if e.modules and modules_mode != "never"
                    and (not keywords or _score(f"{tr(e.modules, 'de')} {tr(e.modules, 'en')}", keywords) > 0) else None),
    } for e in master.education]
    projects = sorted(((p, _score(" ".join([p.de, p.en, *p.keywords]), keywords)) for p in master.projects
                       if not _excluded(f"{p.de} {p.en} {' '.join(p.keywords)}", exclude)), key=lambda ps: -ps[1])
    skill_rows = []
    for row in master.skills:
        items = [(item_text(i, lang), _score(f"{item_text(i, 'de')} {item_text(i, 'en')}", keywords)) for i in row.items
                 if not _excluded(f"{item_text(i, 'de')} {item_text(i, 'en')}", exclude)]
        items.sort(key=lambda it: -it[1])
        skill_rows.append({"id": row.id, "label": tr(row.label, lang), "items": [i for i, _ in items],
                           "relevance": sum(s for _, s in items)})
    skill_rows.sort(key=lambda r: -r["relevance"])
    extra_skills = [x for x in (extra_skills or []) if not _excluded(x, exclude)]
    if extra_skills:
        skill_rows.append({"id": "extra", "label": tr(master.extra_skills_label, lang), "items": extra_skills,
                           "relevance": 0})
    personal = [getattr(p, lang) for p in master.personal if getattr(p, lang) and (p.only in (None, lang))]
    return {
        "language": lang, "name": master.name, "headline": tr(master.headline, lang),
        "contact": master.contact.model_dump(), "personal": personal, "sections": sec,
        "profile_statement": statement, "experience": experience, "education": education,
        "projects": [getattr(p, lang) or p.en or p.de for p, _ in projects[:max_projects]],
        "skills": skill_rows, "languages": [tr(x, lang) for x in master.languages],
    }


def fit_to_page(master: MasterCV, lang: str, keywords: list[Keyword], statement: str,
                extras: list[str] | None = None, exclude: list[str] | None = None) -> dict[str, Any]:
    """The fullest CV that fits MAX_PAGES pages (cuts the least relevant optional content first), measured on the
    real typeset PDF (toolbox.fit_cv_to_page)."""
    from app.services import toolbox

    return toolbox.fit_cv_to_page(lambda level: build_cv(master, lang, keywords, statement, extras, level, exclude),
                                  len(FIT_LEVELS), MAX_PAGES)


def render_text(cv: dict[str, Any], with_extras: bool = True) -> str:
    """Plain-text CV in reading order (what the reviewer reads; also the base of the ATS keyword check)."""
    s = cv["sections"]
    c = cv["contact"]
    lines = [cv["name"], cv["headline"],
             " | ".join(x for x in (c.get("email"), c.get("phone"), c.get("location")) if x),
             " | ".join(link["label"] for link in c.get("links", [])),
             " | ".join(cv.get("personal", []))]
    if cv.get("profile_statement"):
        lines += ["", s.get("profile", "Profile"), cv["profile_statement"]]
    lines += ["", s.get("experience", "Experience")]
    for r in cv["experience"]:
        lines.append(f"{r['title']} | {r['org']}{', ' + r['place'] if r['place'] else ''} | {r['dates']}"
                     + (f" | {r['detail']}" if r.get("detail") else ""))
        lines += [f"- {b}" for b in r["bullets"]]
    lines += ["", s.get("education", "Education")]
    for e in cv["education"]:
        lines.append(f"{e['title']} | {e['org']} | {e['dates']}" + (f" | {e['grade']}" if e.get("grade") else ""))
        if e.get("focus"):
            lines.append(f"  {e['focus']}")
        if e.get("thesis"):
            t = e["thesis"]
            lines.append(f"  {t['label']}: {t['title']}" + (f" - {t['summary']}" if t.get("summary") else ""))
        for extra in ("project", "modules"):
            if e.get(extra):
                lines.append(f"  {e[extra]['label']}: {e[extra].get('title') or e[extra].get('text')}")
    if cv.get("projects"):
        lines += ["", s.get("projects", "Projects")] + [f"- {p}" for p in cv["projects"]]
    lines += ["", s.get("skills", "Skills")]
    for row in cv["skills"]:
        if row["id"] == "extra" and not with_extras:
            continue
        lines.append(f"{row['label']}: {', '.join(row['items'])}")
    lines += ["", s.get("languages", "Languages"), " | ".join(cv["languages"])]
    return "\n".join(x for x in lines if x is not None).strip()


def profile_skill_for(term: str, profile: CandidateProfile) -> str | None:
    """The profile's own wording of a skill matching this posting term (None if the profile lacks it)."""
    best = None
    for skill, _level in profile.skills.all():
        if skill.strip().lower() == term.strip().lower():
            return skill
        if not usable_term(skill):  # "C" would match "C++"
            continue
        if literal_in(term, skill) or literal_in(skill, term):
            if best is None or len(skill) < len(best):
                best = skill
    return best


def extra_skills(cv_text: str, keywords: list[Keyword], profile: CandidateProfile, wanted: list[str] | None = None) -> list[str]:
    """Posting keywords the candidate has (per profile.yaml) that the CV does not name yet — profile wording."""
    out: list[str] = []
    candidates = [k.term for k in keywords] + list(wanted or [])
    for term in candidates:
        if term_in(term, cv_text):
            continue
        skill = profile_skill_for(term, profile)
        if skill and not term_in(skill, cv_text) and skill not in out:
            out.append(skill)
        if len(out) >= MAX_EXTRA_SKILLS:
            break
    return out


_YEARS_CLAIM = re.compile(r"\b\d+\+?\s*(jahre|jahren|years?)\b|\bjahrelang|\bmehrjährig|\bmany years|\bextensive experience", re.I)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


_WORDS = re.compile(r"[a-zäöüß][a-zäöüß-]{4,}")
# Ordinary CV vocabulary that may appear in a statement without being in the CV ("experienced", "focus", ...).
_COMMON = {
    "experienced", "experience", "expertise", "skilled", "proven", "strong", "solid", "focus", "focused", "focusing",
    "building", "including", "within", "across", "through", "using", "based", "driven", "oriented", "passion",
    "background", "knowledge", "practical", "hands-on", "engineer", "engineering", "developer", "development",
    "software", "systems", "system", "solutions", "solution", "robust", "scalable", "modern", "complex", "technical",
    "projects", "project", "professional", "academic", "applied", "application", "applications", "tools", "tooling",
    "environment", "environments", "workflows", "workflow", "processes", "process", "teams", "delivering", "designing",
    "erfahrung", "erfahren", "kenntnisse", "fundierte", "praktische", "schwerpunkt", "schwerpunkten", "bereich",
    "bereichen", "entwicklung", "entwickler", "softwareentwickler", "systeme", "systemen", "lösungen", "projekten",
    "anwendungen", "werkzeuge", "umgebungen", "prozesse", "insbesondere", "sowie", "zudem", "durch", "inklusive",
    "berufserfahrung", "abschluss", "studium", "masterstudium", "kombiniert", "verbindet", "verbindung", "fokus",
    "schwerpunkt", "praxis", "praxisnah", "fundiert", "einsatz", "eingesetzt", "umfassende", "vertiefte", "sicherer",
    "combines", "combining", "combined", "together", "hands", "particular", "particularly", "especially", "areas",
}


def _unbacked_words(sentence: str, corpus_words: set[str]) -> list[str]:
    """Content words of the sentence that appear nowhere in your evidence (CV text, profile skills, jobs, degrees —
    prefix match, so inflections count). Any such word = the model is describing work you never did, typically copied
    from the posting ("virtual commissioning", "CAD-Daten") — in any language, also when the posting is English
    and the CV German."""
    prefixes = {w[:6] for w in corpus_words}
    return [w for w in _WORDS.findall(sentence.lower())
            if w not in _COMMON and w[:6] not in prefixes and not any(p in w for p in corpus_words if len(p) >= 7)]


def evidence_text(profile: CandidateProfile) -> str:
    """What you have actually done: skills, employment, education — NOT career goals or target role families
    (wishes are no evidence)."""
    parts = [s for s, _ in profile.skills.all()]
    parts += [str(e.model_dump()) for e in profile.experience]
    parts += [str(e.model_dump()) for e in profile.education]
    return "\n".join(parts)


def check_statement(statement: str, master_text: str, profile: CandidateProfile, job_text: str = "") -> tuple[str, list[str]]:
    """Drop sentences that name a technology absent from the profile, a number absent from the CV, a
    years-of-experience claim, or a content word your evidence does not back (copied duties such as
    "virtual commissioning", "kinematization"). Returns (clean statement, removed sentences)."""
    corpus = (evidence_text(profile) + "\n" + master_text).lower()
    corpus_words = set(_WORDS.findall(corpus))
    unknown = [t for t in TECH_TERMS if t.lower() not in profile.evidence_corpus() + "\n" + master_text.lower()]
    kept, removed = [], []
    for sentence in split_sentences(statement):
        bad_tech = [t for t in unknown if literal_in(t, sentence)]
        bad_numbers = [n for n in _NUMBER.findall(sentence) if n not in master_text]
        unbacked = _unbacked_words(sentence, corpus_words)
        if bad_tech or bad_numbers or _YEARS_CLAIM.search(sentence) or unbacked:
            removed.append(sentence)
        else:
            kept.append(sentence)
    return " ".join(kept).strip(), removed


def fallback_statement(lang: str, master: MasterCV | None = None) -> str:
    """Deterministic statement when no writer model is available (or its text did not survive the checks): the
    CV's own headline (the applicant's words — nothing added), else the profile summary's first sentence."""
    try:
        master = master or get_master_cv()
        headline = tr(master.headline, lang).replace(" | ", " · ").strip()
    except CvError:
        headline = ""
    if headline:
        return headline.rstrip(".") + "."
    from app.config_files import get_profile

    first = split_sentences(get_profile().summary or "")
    return first[0] if first else ""


def cv_vocabulary(master: MasterCV, profile: CandidateProfile) -> list[str]:
    return master.vocabulary() + [s for s, _ in profile.skills.all()]


def tailor_cv(session, job, lang: str, setup: LLMSetup | None, profile: CandidateProfile,
              revision: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[CallResult]]:
    """Tailored CV for this job. `revision` (reviewer agent or your feedback): {"add_keywords": [...],
    "cv_changes": [...], "exclude": [...], "emphasize": [...]}. Your learned rules (agent_memory) always apply."""
    from app.services import agent_memory
    from app.services.keywords import Keyword

    master = get_master_cv()
    revision = revision or {}
    keywords = job_keywords(job, cv_vocabulary(master, profile))
    emphasize = [" ".join(core_words(t)) for t in revision.get("emphasize") or [] if core_words(t)]
    keywords = [*(Keyword(t.strip(), "required", "applicant") for t in emphasize), *keywords]  # put first
    exclude = [*agent_memory.cv_excludes(), *(revision.get("exclude") or [])]
    instructions = [*(r.text for r in agent_memory.rules_for("cv")), *(revision.get("cv_changes") or [])]
    master_text = master.full_text(lang)
    calls: list[CallResult] = []
    a = job.current_analysis.result if job.current_analysis and job.current_analysis.result else {}
    strong = [str(m.get("skill")) for m in (a.get("strong_matches") or []) if m.get("skill")]

    statement, removed, source = "", [], "fallback"
    writer = setup.target(Role.WRITER) if setup else None
    if writer is not None:
        tasks.progress("Application agent: writing the CV profile statement")
        try:
            res = llm_call(session, writer, kind="cv_statement", job_id=job.id, prompt_version=CV_STATEMENT_VERSION,
                           profile_hash=profile.fingerprint(), system=cv_statement_system(lang),
                           user=cv_statement_user(master_text, job, strong, instructions,
                                                  gaps=[k.term for k in keywords if not term_in(k.term, master_text)
                                                        and not profile_skill_for(k.term, profile)]),
                           schema=CvProfileStatement, use_cache=not revision)
            calls.append(res)
            job_text = " ".join(x for x in (job.title, job.description, job.requirements, job.responsibilities) if x)
            statement, removed = check_statement(str(res.data.get("statement", "")), master_text, profile, job_text)
            source = "writer"
        except Exception as exc:  # noqa: BLE001 - the deterministic statement stands in
            log.warning("CV statement generation failed (%s) — using the fallback", exc)
    if len(statement.split()) < 8:
        statement, source = fallback_statement(lang), "fallback"

    base = build_cv(master, lang, keywords, statement, exclude=exclude)  # (extras are judged against the full CV)
    extras = extra_skills(render_text(base), keywords, profile, list(revision.get("add_keywords") or []))
    cv = fit_to_page(master, lang, keywords, statement, extras, exclude)
    cv["meta"] = {
        "keywords": [k.as_dict() for k in keywords], "extra_skills": extras, "statement_source": source,
        "statement_removed": removed, "revision": revision or None, "excluded": exclude or None,
    }
    return cv, calls
