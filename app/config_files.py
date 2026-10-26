"""Typed access to the user-editable YAML config (profile + preferences)."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.settings import get_settings

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- profile


class _Open(BaseModel):
    """Profile sections keep any extra keys (institution, period, location...) so prompts see everything."""

    model_config = ConfigDict(extra="allow")


class Education(_Open):
    degree: str
    focus: str = ""
    thesis: str = ""


class Experience(_Open):
    employer: str
    role: str = ""
    period: str = ""
    highlights: list[str] = Field(default_factory=list)


class Skills(_Open):
    """Skills grouped by evidence level. Extra groups (e.g. `certifications`) are kept with their own label."""

    direct_professional: list[str] = Field(default_factory=list)
    academic: list[str] = Field(default_factory=list)
    project: list[str] = Field(default_factory=list)

    def all(self) -> list[tuple[str, str]]:
        out = [(s, "direct_professional") for s in self.direct_professional]
        out += [(s, "academic") for s in self.academic]
        out += [(s, "project") for s in self.project]
        for group, items in (self.model_extra or {}).items():
            if isinstance(items, list):
                out += [(str(s), group) for s in items]
        return out

    def levels(self) -> dict[str, set[str]]:
        """skill (lowercase) → evidence levels it appears under, e.g. 'c++' → {'academic', 'project'}."""
        out: dict[str, set[str]] = {}
        for skill, level in self.all():
            out.setdefault(skill.strip().lower(), set()).add(level)
        return out


class Language(_Open):
    language: str
    level: str = ""


class CareerGoals(_Open):
    positioning: str = ""
    primary_role_families: list[str] = Field(default_factory=list)
    secondary_sectors: list[str] = Field(default_factory=list)
    acceptable_sectors: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    target_countries: list[str] = Field(default_factory=list)


class CandidateProfile(_Open):
    headline: str = ""
    summary: str = ""
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    years_professional_experience: float | None = None
    skills: Skills = Field(default_factory=Skills)
    languages: list[Language] = Field(default_factory=list)
    work_authorization: list[str] = Field(default_factory=list)
    career_goals: CareerGoals = Field(default_factory=CareerGoals)
    # Countries whose citizenship you hold: postings reserved for nationals of these stay open to you.
    citizenships: list[str] = Field(default_factory=list)
    # Currently enrolled at a university: student jobs and student-only internships are shown.
    enrolled_student: bool = False

    def as_prompt_text(self) -> str:
        """Stable, deterministic text rendering for prompts (stable => cacheable)."""
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True, width=100)

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:16]

    def evidence_corpus(self) -> str:
        """All text in the profile, lowercased — used to verify LLM skill claims."""
        return json.dumps(self.model_dump(), ensure_ascii=False).lower()

    def professional_corpus(self) -> str:
        """Text that evidences PAID professional work: professional skills + employment entries."""
        parts = list(self.skills.direct_professional)
        for e in self.experience:
            parts.append(json.dumps(e.model_dump(), ensure_ascii=False))
        return "\n".join(parts).lower()

    def evidence_map_text(self) -> str:
        """Compact 'skill: levels' listing for prompts, so models see how each skill is evidenced."""
        names = {"direct_professional": "professional", "academic": "academic", "project": "project"}
        lines = []
        for skill, levels in sorted(self.skills.levels().items()):
            label = ", ".join(sorted(names.get(lv, lv) for lv in levels))
            original = next(s for s, _ in self.skills.all() if s.strip().lower() == skill)
            lines.append(f"- {original}: {label}")
        return "\n".join(lines)

    def completeness_warnings(self) -> list[str]:
        """Fields that are missing/incomplete. Surfaced in the UI — never filled in by guessing."""
        w = []
        if self.years_professional_experience is None:
            w.append("years_professional_experience is not set")
        if not self.languages:
            w.append("languages is empty (language requirements cannot be matched)")
        if not self.skills.direct_professional:
            w.append("skills.direct_professional is empty")
        for e in self.experience:
            if not e.role:
                w.append(f"experience '{e.employer}': role missing")
            if not e.period:
                w.append(f"experience '{e.employer}': period missing")
            if not e.highlights:
                w.append(f"experience '{e.employer}': no highlights")
        for ed in self.education:
            if not (ed.model_extra or {}).get("period"):
                w.append(f"education '{ed.degree}': period missing")
        if not self.career_goals.target_countries:
            w.append("career_goals.target_countries is empty")
        return w


# --------------------------------------------------------------------------- preferences


class LocationPrefs(BaseModel):
    target_countries: list[str] = Field(default_factory=lambda: ["Germany"])
    country_scores: dict[str, float] = Field(default_factory=dict)
    unknown_country_score: float = 5
    remote_other_country: str = "reject"
    unknown_country: str = "flag"  # flag | pass | reject — when no country can be determined at all
    excluded_locations: list[str] = Field(default_factory=list)
    preferred_cities: dict[str, float] = Field(default_factory=dict)
    # Where you live: drive times for side jobs and internships, the Bundesagentur's "near me" searches.
    # Coordinates are looked up from the town automatically when left empty.
    home: str = ""
    home_lat: float | None = None
    home_lon: float | None = None


class HardFilterPrefs(BaseModel):
    excluded_title_patterns: list[str] = Field(default_factory=list)
    max_years_experience_required: int = 10
    unrelated_profession_patterns: list[str] = Field(default_factory=list)
    relevant_title_patterns: list[str] = Field(default_factory=list)
    mandatory_qualification_patterns: list[str] = Field(default_factory=list)
    triage_reject_unrelated: bool = True


class SideJobPrefs(BaseModel):
    """Part-time, mini-jobs and temporary work of any kind (preferences.yaml → side_jobs). Own filters, no LLM analysis,
    ranked by distance, flexibility and how well the work fits you."""

    enabled: bool = True
    # Deprecated (kept for older preferences.yaml files): your home is now locations.home / home_lat / home_lon.
    home: str = ""
    # Side jobs must be reachable by car within this time (real road routing from home; remote jobs are exempt).
    max_drive_minutes: int = 60
    home_lat: float | None = None
    home_lon: float | None = None
    radius_km: int = 80  # straight-line search radius for the sources — an upper bound; drive time decides
    # Student jobs need a Studienbescheinigung: hidden unless your profile says enrolled_student: true.
    student_title_patterns: list[str] = Field(default_factory=lambda: [
        r"werkstudent", r"studentenjob", r"student(in)?\b", r"studierende"])
    # Never side jobs: school pupils, vocational training, internships (own page), voluntary service.
    excluded_title_patterns: list[str] = Field(default_factory=lambda: [
        r"schüler", r"azubi", r"ausbildung", r"praktik", r"trainee", r"duales? studium", r"\bfsj\b", r"bundesfreiwillig"])
    # Work that needs a licence or training you don't have (truck licence classes, nursing, …) → rejected.
    excluded_requirement_patterns: list[str] = Field(default_factory=lambda: [
        r"führerschein (der )?(klasse )?c1?e?\b|\bc1?e?-führerschein|lkw-führerschein|berufskraftfahrer", r"pflegefach|examinierte",
        r"approbation", r"meisterbrief|meisterprüfung", r"staplerschein erforderlich"])
    # Regulated professions / trades you are not qualified for (matched in the title) → rejected.
    excluded_profession_patterns: list[str] = Field(default_factory=lambda: [
        r"rechtsanw|steuerberat|steuerfachangest|notar|wirtschaftsprüf", r"\bärzt|\barzt|apothek|zahnarzt|zahntechn",
        r"pflege(fach|kraft|helfer)|altenpfleg|krankenpfleg|hebamme|physiotherap|ergotherap|logopäd|psychotherap",
        r"erzieher|heilerzieh|sozialpädagog|sozialarbeit", r"elektroniker|elektrofachkraft|mechatroniker|kfz.?mechan",
        r"fachverkäufer|bäcker|konditor|fleischer|koch\b|köchin"])
    # Your field (matched in the title / occupation): the strongest side-job fit.
    strong_fit_keywords: list[str] = Field(default_factory=lambda: [
        "it ", "it support", "software", "informatik", "programmier", "entwickler", "daten", "data", "datenbank",
        "systemadministr", "netzwerk", "computer", "ki ", "künstliche intelligenz", "machine learning", "web",
        "digital", "devops", "python", "java"])
    # Related work (matched in the title / occupation): tech, support, teaching, research, languages.
    good_fit_keywords: list[str] = Field(default_factory=lambda: [
        "support", "administr", "technik", "techniker", "elektronik", "nachhilfe", "tutor", "dozent", "lehrkraft",
        "mathe", "physik", "labor", "forschung", "wissenschaft", "übersetz", "dolmetsch", "arabisch", "englisch",
        "test", "qualitätssicherung", "büro", "sachbearbeit", "assistenz", "kundenservice", "freelance", "freiberuf"])


class SearchPrefs(BaseModel):
    """What the job platforms are searched for (Workable, Careerjet, Jooble — sources without their own `queries`)."""

    queries: list[str] = Field(default_factory=lambda: [
        "software engineer", "machine learning", "AI engineer", "data scientist", "data engineer", "backend developer",
        "full stack developer", "python developer", "computer vision", "robotics", "devops engineer", "MLOps"])


class InternshipPrefs(BaseModel):
    """Internships and trainee programmes (job.track == "intern", own page). Stricter than the career search: only your
    field, close to home (real drive time, see side_jobs.home_lat/lon) or fully remote, and never one that needs you
    to be enrolled — you have graduated. Ranked by the same AI analysis as your career jobs."""

    enabled: bool = True
    max_drive_minutes: int = 45  # by car from home (side_jobs.home_lat / home_lon); fully remote ones are exempt
    # Internships whose text requires university enrollment: reject (hidden) | flag (shown with a warning — e.g. once
    # you are enrolled as a doctoral student).
    student_only: str = "reject"
    # Title regex that makes a posting an internship / trainee programme (it leaves the career search for this page).
    posting_pattern: str = (
        r"praktik|\binterns?\b|internship|trainee|volontariat|einstiegsprogramm|graduate program|berufseinstiegsprogramm"
    )
    # Only for enrolled students (title) — hidden unless your profile says enrolled_student: true.
    student_title_patterns: list[str] = Field(default_factory=lambda: [
        r"werkstudent|working student|studentenjob|studentische|hiwi",
        r"pflicht-?praktik|mandatory internship|curricular internship|praxissemester|praktisches studiensemester|"
        r"vorpraktik|berufspraktikum (für|im rahmen)",
        r"^(?!.*praktik).*(abschlussarbeit|masterarbeit|bachelorarbeit|\bthesis\b)",
    ])
    # Never an internship for you (title): school internships, vocational training, voluntary service.
    excluded_title_patterns: list[str] = Field(default_factory=lambda: [
        r"schülerpraktik|schnupperpraktik|orientierungspraktik|anerkennungspraktik",
        r"ausbildung|\bazubi|duales? studium|dualer student|\bfsj\b|freiwilliges soziales|bundesfreiwillig",
    ])
    # In the posting text: the internship is only for enrolled students …
    enrollment_patterns: list[str] = Field(default_factory=lambda: [
        r"pflicht-?praktik|praxissemester|immatrikul|eingeschrieben|immatrikulationsbescheinigung|studienbescheinigung",
        r"(im rahmen|während) (deines|ihres|des|eines) (laufenden )?studiums|du studierst|sie studieren|"
        r"(laufende[ns]?|aktuelle[ns]?) studium|studierende[rn]? (der|des|im|in|eines)|du bist (eingeschrieben|student)|"
        r"(dich|sie|du bist|sie sind) (noch )?im (laufenden )?studium|student(in)? (der|des|im) (fach|studiengang)",
        r"currently (enrolled|studying|pursuing)|enrolled (student|in a|at a)|"
        r"\bpursuing (a |an |your )?(bs|ba|ms|msc|m\.sc|b\.sc|bachelor|master|ph\.?d|degree|university degree)|"
        r"you are (currently )?(a )?(university )?student|"
        r"(mandatory|compulsory|required) internship|as part of your (studies|degree)",
    ])
    # … unless it also welcomes graduates.
    graduate_patterns: list[str] = Field(default_factory=lambda: [
        r"absolvent|hochschulabsolvent|berufseinsteiger|nach (deinem|ihrem|dem) (abgeschlossenen )?studium|"
        r"abgeschlossene[ns]? (hochschul)?studium|(recent|new) graduates?|graduates? (are )?welcome|post-?graduate|"
        r"after (your|completing your) (studies|degree)|bereits abgeschlossen",
    ])
    # Your field (title): software, AI / ML, data, robotics, embedded, IT. Anything else is not shown.
    field_title_pattern: str = (
        r"software|informatik|\bit\b|\bit-|\bki\b|\bai\b|künstliche intelligenz|artificial intelligence|machine learning|"
        r"maschinelles lernen|deep learning|\bml\b|mlops|\bllm|gen ?ai|generative|data|daten|robot|autonom|embedded|"
        r"cloud|devops|computer vision|bildverarbeitung|programmier|entwickl|developer|development|"
        r"simulation|cyber|security|it-sicherheit|web|backend|frontend|full.?stack|app-|automatisierung|"
        r"automation|algorithm|elektronik.*software|mechatronik.*software|engineering (ai|software|data)"
    )
    # Not your field even if a field word appears ("Online Marketing", "HR Data", "Controlling Digital").
    unrelated_title_pattern: str = (
        r"marketing|vertrieb|sales|\bhr\b|human resources|personal|recruit|talent|finanz|finance|accounting|controlling|"
        r"\btax\b|steuer|einkauf|purchasing|procurement|beschaffung|logistik|logistics|kommunikation|communication|"
        r"\bpr\b|public relations|mediengestalt|design(er)?\b(?!.*software)|journalis|redaktion|legal|recht|compliance|"
        r"office|assistenz|event|e-?commerce|social media|^(?!.*(software|data|\bai\b|\bki\b|\bit\b)).*consult|"
        r"supply chain|leadership|manage(r|ment)s? program|nachwuchsführung"
    )


class PhdPrefs(BaseModel):
    """Doctoral / research positions: only funded ones in your field (see preferences.yaml → phd)."""

    enabled: bool = True
    # Title regex that marks a posting as a doctoral / research position.
    posting_pattern: str = (
        r"ph\.?\s?d\b|doktorand|doctoral|doctorate|promotionsstelle|promotionsmöglichkeit|zur promotion|promovier|"
        r"promotions-?stipend|promotionskolleg|graduiertenkolleg|pr(a|ä|e)-?doc|"
        r"(wissenschaftlich|akademisch)\w*(?:[*/:]\w*)?\s+mitarbeiter"
    )
    # Any of these over the full posting = paid (employment contract or stipend).
    funded_patterns: list[str] = Field(default_factory=lambda: [
        r"\btv-?l\b|\btv-?h\b|\btv[öo]d\b|\btv-?öd\b|\be\s?g?\s?13\b|entgeltgruppe|besoldungsgruppe",
        r"fully[- ]funded|funded (phd|doctoral|position)|funding (is )?(available|provided|secured)|scholarship|stipend|stipendium",
        r"\bsalary\b|\bgehalt\b|vergütung|bezahlung|remuneration|employment contract|arbeitsvertrag",
        r"(wissenschaftlich|akademisch)\w*(?:[*/:]\w*)?\s+mitarbeiter",
    ])
    # Any of these (and no funded signal) = not funded → rejected.
    unfunded_patterns: list[str] = Field(default_factory=lambda: [
        r"self[- ]funded|own funding|secure (their|your) own funding|no funding|(position|place|phd) is not funded|unfunded|without (salary|funding|pay)",
        r"unbezahlt|ohne vergütung|selbstfinanziert|eigene finanzierung|keine finanzierung|external(ly)? funded (students|candidates) only",
    ])
    # What to do when a posting says nothing about funding: flag | pass | reject.
    no_funding_info: str = "flag"
    # Your fields. A doctoral posting matching field_title_pattern in its title, or at least
    # min_field_keyword_hits distinct field_keywords in its text, is in your field; others are rejected.
    field_title_pattern: str = (
        r"informatik|computer science|künstliche intelligenz|artificial intelligence|\bai\b|\bki\b|machine learning|"
        r"maschinelles lernen|deep learning|reinforcement learning|robot learning|neural|neuronal|robot|autonom|"
        r"intelligente[nr]? systeme|intelligent systems|computer vision|bildverarbeitung|data science|cyber ?security|"
        r"it-sicherheit|informationssicherheit|digital twin|perception|\bllm|language model|\bnlp\b|software engineering|"
        r"softwaretechnik"
    )
    field_keywords: list[str] = Field(default_factory=lambda: [
        "informatik", "computer science", "künstliche intelligenz", "artificial intelligence", "machine learning",
        "maschinelles lernen", "deep learning", "neural network", "neuronale netz", "reinforcement learning", "robotik",
        "robotics", "autonomous", "autonome", "computer vision", "bildverarbeitung", "software", "data science",
        "large language model", "llm", "natural language", "intelligent systems", "it-sicherheit", "cyber security",
        "cybersecurity", "simulation", "digital twin", "perception", "embedded",
    ])
    min_field_keyword_hits: int = 3


class UntrustedSourcePrefs(BaseModel):
    """Gig / "AI-training" marketplaces and talent funnels whose postings are skipped (rejected before any LLM call).

    Matching is deliberately precise: company names exactly (or "<name> <suffix>"), domains in the job's URLs and in
    links inside the posting, title patterns on the title only, and posting phrases only together with gig signals.
    """

    companies: list[str] = Field(default_factory=lambda: [
        "micro1", "Alignerr", "Outlier", "Outlier AI", "Mercor", "DataAnnotation", "DataAnnotation.tech", "Remotasks",
        "Mindrift", "Invisible Technologies", "Crossover", "Toloka",
    ])
    domains: list[str] = Field(default_factory=lambda: [
        "micro1.ai", "alignerr.com", "outlier.ai", "mercor.com", "mercor.io", "dataannotation.tech", "remotasks.com",
        "mindrift.ai", "invisible.co", "invisibletech.ai", "crossover.com", "toloka.ai", "turing.com",
    ])
    title_patterns: list[str] = Field(default_factory=lambda: [
        r"\b(ai|llm) (trainer|tutor)s?\b",
        r"\b(ai|llm|model) (training|evaluation) (specialist|expert|contributor|freelancer|annotator)s?\b",
        r"\brlhf\b",
        r"\bdata annotat(or|ion (specialist|freelancer|contributor))s?\b",
        r"\bprompt (writer|evaluator|rater)s?\b",
        r"\b(ai|search) (quality )?raters?\b",
    ])
    # Platform-style wording. Rejects only with a gig signal or when two phrases appear — real ML jobs may say
    # "you will train large language models" and must pass.
    posting_patterns: list[str] = Field(default_factory=lambda: [
        r"help (us )?train (the next generation of |cutting[- ]edge |frontier )?(ai|large language|llm|frontier) models",
        r"(rate|evaluate|review|rank) (ai|model|chatbot)[- ]generated (responses|outputs|text)",
        r"join (our|a) (global )?(talent|expert|contributor) (network|community)",
        r"paid per (task|submission)",
        r"(train|improve) ai models (with|through) your (expertise|knowledge)",
    ])
    gig_signals: str = r"\b(freelance|freelancer|independent contractor|contract role|per hour|/ ?hr\b|hourly rate|hourly pay|flexible (hours|schedule)|work from anywhere|paid weekly)"


class DefencePrefs(BaseModel):
    excluded_companies: list[str] = Field(default_factory=list)
    mixed_companies: list[str] = Field(default_factory=list)
    very_strong_patterns: list[str] = Field(default_factory=list)
    strong_keywords: list[str] = Field(default_factory=list)
    benign_phrases: list[str] = Field(default_factory=list)


class ScoringPrefs(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "technical": 0.30,
            "experience": 0.20,
            "role_family": 0.15,
            "sector": 0.10,
            "location": 0.10,
            "career": 0.10,
            "source": 0.05,
        }
    )
    thresholds: dict[str, float] = Field(default_factory=lambda: {"priority": 8.0, "review": 6.5})
    target_company_bonus: dict[str, float] = Field(default_factory=dict)
    critical_gap_penalty: float = 0.4
    critical_gap_penalty_max: float = 1.2
    role_family_scores: dict[str, float] = Field(default_factory=dict)
    sector_scores: dict[str, float] = Field(default_factory=dict)
    default_sector_score: float = 5.5
    seniority_scores: dict[str, float] = Field(default_factory=dict)
    # subtracted from the score for levels above the candidate's (e.g. Senior with ~3 years of experience)
    seniority_penalty: dict[str, float] = Field(default_factory=dict)
    # required years beyond the candidate's years (+ tolerance) cost this much per year, up to the max
    years_gap_tolerance: float = 1.0
    years_gap_penalty_per_year: float = 0.6
    years_gap_penalty_max: float = 2.5
    source_reliability: dict[str, float] = Field(default_factory=dict)


class SchedulerPrefs(BaseModel):
    tick_minutes: int = 30
    company_check_hours: dict[str, float] = Field(default_factory=lambda: {"A": 6, "B": 12, "C": 24})
    email_check_minutes: int = 60
    analysis_minutes: int = 30
    screen_batch_size: int = 200  # jobs screened per run (cheap model, loaded once)
    analysis_batch_size: int = 40  # deep analyses per run (strong model, loaded once)


class LLMPrefs(BaseModel):
    triage_enabled: bool = True  # run the cheap screening stage before deep analysis
    # Deep analysis only runs when screening says likely_relevant and relevance >= this (0-10).
    min_screen_relevance: float = 5.0
    # Scheduled runs re-analyse jobs that so far only have the free heuristic score. Off by default so the backlog
    # isn't processed before you've calibrated on labeled jobs (use 'Analyze labeled jobs' / 'Re-analyse all').
    upgrade_heuristic_backlog: bool = False
    daily_budget_usd: float = 3.0
    max_description_chars_fit: int = 12000
    max_description_chars_triage: int = 2500
    max_attempts: int = 3
    prices: dict[str, list[float]] = Field(default_factory=dict)

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.prices.get(model)
        if not price:
            # tolerate provider-prefixed / dated model names
            for name, p in self.prices.items():
                if model.startswith(name) or name in model:
                    price = p
                    break
        if not price:
            return 0.0
        return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


class EmailPrefs(BaseModel):
    enabled_parsers: list[str] = Field(
        default_factory=lambda: ["linkedin", "indeed", "stepstone", "arbeitsagentur", "bayt", "gulftalent", "naukrigulf",
                                 "xing", "totaljobs", "cwjobs", "jobsite", "reed", "cvlibrary", "glassdoor", "monster",
                                 "jobs_ac_uk", "guardian_jobs", "wttj", "stellenanzeigen", "jobware", "get_in_it", "generic"]
    )
    generic_senders: list[str] = Field(default_factory=list)
    max_messages_per_run: int = 200
    cleanup_target: str = "trash"  # mailbox clean-up: "trash" (recoverable) or the name of an archive folder
    cleanup_min_age_days: int = 2  # only alerts the app processed at least this many days ago


class LocalRoleOptions(BaseModel):
    num_ctx: int = 8192
    keep_alive: str = "5m"
    temperature: float = 0.2
    top_p: float | None = None
    think: bool | None = False  # Qwen3 hybrid thinking; None = model default
    max_tokens: int = 2000
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None


class LocalLLMPrefs(BaseModel):
    roles: dict[str, LocalRoleOptions] = Field(default_factory=dict)

    def options(self, role: str) -> LocalRoleOptions:
        return self.roles.get(role) or LocalRoleOptions()


class WritingPrefs(BaseModel):
    target_words_min: int = 250
    target_words_max: int = 350
    german_editor_by_default: bool = True
    humanizer_by_default: bool = False  # optional extra rewrite pass after the style editor
    use_templates: bool = True  # human-written example letters in config/letter_templates/ guide structure and voice
    templates_per_letter: int = 2  # best-matching examples given to the writer (1 = structure only, 2 = + style)
    use_fit_analysis: bool = True  # build letters on the job's strong + transferable matches from the fit analysis
    applicant_name: str = ""  # used in the closing; empty → "[Name]" placeholder
    max_job_chars: int = 8000
    # Work permit + mobility: every letter for a job in these countries states both (the writer is asked to; if it
    # doesn't, the configured sentence is inserted after validation). Facts must also be in profile.yaml.
    mobility_countries: list[str] = Field(default_factory=lambda: ["Germany"])
    work_permit_sentence_de: str = ""
    relocation_sentence_de: str = ""
    work_permit_sentence_en: str = ""
    relocation_sentence_en: str = ""
    # per-country replacements for the sentences above: {Luxembourg: {work_permit_de, relocation_de, work_permit_en, relocation_en}}
    country_sentences: dict[str, dict[str, str]] = Field(default_factory=dict)


class RetentionPrefs(BaseModel):
    max_job_age_days: int = 14  # by date posted (or first seen when the source gives no date)
    keep_interacted: bool = True  # never delete jobs you saved, applied to, labeled, annotated or wrote a letter for
    tombstone_days: int = 30  # forget a deleted posting once no feed has listed it for this long
    academic_max_age_days: int = 60  # PhD / research postings (academic sources) stay open until their deadline


class ApplicationDocumentRef(BaseModel):
    """One certificate / reference in data/documents that goes into the complete application PDF."""

    file: str
    label_de: str
    label_en: str = ""
    skip_pages: list[int] = Field(default_factory=list)  # 1-based, e.g. an HR cover note before the reference


class ApplicationPrefs(BaseModel):
    """Application agent (tailored CV + cover letter + e-mail) and reviewer agent (ATS + recruiter → SEND / REVISE)."""

    max_revision_rounds: int = 2  # reviewer → revise → review again, at most this many times
    ats_send_threshold: int = 70  # ATS score (0-100) needed for SEND
    recruiter_send_threshold: float = 7.0  # recruiter quality (mean of CV and letter score, 0-10) needed for SEND
    cv_language: str = "auto"  # auto = same as the cover letter | de | en
    analyse_first: bool = True  # run the fit analysis first when the job has none (the CV builds on it)
    research_enabled: bool = True  # research agent: company website + Wikipedia brief before the letter
    max_sends_per_day: int = 0  # sending applications by e-mail (only on your command): daily cap, 0 = no limit
    send_pause_seconds: int = 30  # pause between two sent applications in a bulk run
    # "Anlagen" listed under the letter (what you attach when you send it)
    letter_attachments_de: list[str] = Field(default_factory=lambda: ["Lebenslauf", "Zeugnisse"])
    letter_attachments_en: list[str] = Field(default_factory=lambda: ["CV", "Certificates"])
    # Certificates appended (in this order) to the complete application PDF; when set, they also make up the
    # "Anlagen" list under the letter (Lebenslauf + these labels).
    documents: list[ApplicationDocumentRef] = Field(default_factory=list)


class DashboardPrefs(BaseModel):
    default_since_days: int = 7
    page_size: int = 50


class Preferences(BaseModel):
    locations: LocationPrefs = Field(default_factory=LocationPrefs)
    hard_filters: HardFilterPrefs = Field(default_factory=HardFilterPrefs)
    defence: DefencePrefs = Field(default_factory=DefencePrefs)
    untrusted_sources: UntrustedSourcePrefs = Field(default_factory=UntrustedSourcePrefs)
    phd: PhdPrefs = Field(default_factory=PhdPrefs)
    side_jobs: SideJobPrefs = Field(default_factory=SideJobPrefs)
    internships: InternshipPrefs = Field(default_factory=InternshipPrefs)
    search: SearchPrefs = Field(default_factory=SearchPrefs)
    scoring: ScoringPrefs = Field(default_factory=ScoringPrefs)
    role_family_keywords: dict[str, list[str]] = Field(default_factory=dict)
    scheduler: SchedulerPrefs = Field(default_factory=SchedulerPrefs)
    llm: LLMPrefs = Field(default_factory=LLMPrefs)
    email: EmailPrefs = Field(default_factory=EmailPrefs)
    dashboard: DashboardPrefs = Field(default_factory=DashboardPrefs)
    local_llm: LocalLLMPrefs = Field(default_factory=LocalLLMPrefs)
    writing: WritingPrefs = Field(default_factory=WritingPrefs)
    retention: RetentionPrefs = Field(default_factory=RetentionPrefs)
    application: ApplicationPrefs = Field(default_factory=ApplicationPrefs)

    @property
    def role_families(self) -> list[str]:
        return list(self.scoring.role_family_scores.keys()) or ["Other"]


# --------------------------------------------------------------------------- loading


def ensure_config() -> list[str]:
    """First start: copy the shipped examples (config.example/) for every config file you don't have yet — your own
    files are never overwritten. Returns the names copied."""
    import shutil

    from app.settings import PROJECT_ROOT

    target: Path = get_settings().config_dir
    example = PROJECT_ROOT / "config.example"
    copied = []
    if not example.is_dir():
        return copied
    target.mkdir(parents=True, exist_ok=True)
    for src in example.rglob("*"):
        dest = target / src.relative_to(example)
        if src.is_dir():
            dest.mkdir(exist_ok=True)
        elif not dest.exists():
            shutil.copy2(src, dest)
            copied.append(str(src.relative_to(example)))
    if copied:
        log.info("Created your config from config.example: %s", ", ".join(copied))
    return copied


class _ConfigStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profile: CandidateProfile | None = None
        self._prefs: Preferences | None = None

    def _read(self, name: str) -> dict[str, Any]:
        path: Path = get_settings().config_dir / name
        if not path.exists():
            log.warning("Config file %s not found — using defaults", path)
            return {}
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def reload(self) -> None:
        with self._lock:
            self._profile = CandidateProfile.model_validate(self._read("profile.yaml"))
            self._prefs = Preferences.model_validate(self._read("preferences.yaml"))
        log.info("Loaded config (profile %s)", self._profile.fingerprint())
        for warning in self.profile_warnings():
            log.warning("Profile incomplete: %s", warning)

    def profile_warnings(self) -> list[str]:
        """Profile gaps + 'VERIFY' markers still in profile.yaml + missing applicant name for letters."""
        assert self._profile is not None and self._prefs is not None
        warnings = self._profile.completeness_warnings()
        path: Path = get_settings().config_dir / "profile.yaml"
        if path.exists():
            for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "VERIFY" in line and not line.lstrip().startswith("#"):
                    warnings.append(f"profile.yaml line {no} still marked VERIFY")
        if not self._prefs.writing.applicant_name:
            warnings.append("preferences.yaml writing.applicant_name is empty (letters end with a [Name] placeholder)")
        return warnings

    def profile(self) -> CandidateProfile:
        if self._profile is None:
            self.reload()
        assert self._profile is not None
        return self._profile

    def prefs(self) -> Preferences:
        if self._prefs is None:
            self.reload()
        assert self._prefs is not None
        return self._prefs

    def set_for_tests(self, profile: CandidateProfile, prefs: Preferences) -> None:
        self._profile, self._prefs = profile, prefs


config_store = _ConfigStore()


def get_profile() -> CandidateProfile:
    return config_store.profile()


def get_prefs() -> Preferences:
    return config_store.prefs()


def load_companies_yaml() -> list[dict[str, Any]]:
    path = get_settings().config_dir / "companies.yaml"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return list(data.get("companies") or [])
