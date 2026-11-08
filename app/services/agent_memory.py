"""What the application agent learns from your feedback.

When you tell the agent what to change in an application ("too long", "don't mention the Unity project", "lead with
the robotics work"), one model call turns it into:
  - changes for THIS application: letter changes, CV statement changes, CV items to leave out or to emphasise
  - notes for THIS job (saved to the job's personal notes: emphasise / avoid)
  - general rules for EVERY future application ("Keep letters under 300 words", "Never list Unity on my CV") —
    saved in config/agent_memory.yaml and given to the writer, the CV agent and the reviewer from then on.

Rules are preferences about style, emphasis and selection. They never add facts: facts still come only from
profile.yaml / cv.yaml, and the fact check still applies. You can edit or delete every rule (More → Agent memory).
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

FEEDBACK_VERSION = "feedback-plan-v4"
Scope = Literal["letter", "cv", "all"]


class LearnedRule(BaseModel):
    id: int
    text: str
    applies_to: Scope = "all"
    cv_exclude: list[str] = Field(default_factory=list)  # terms whose CV items are always left out
    learned_from: str = ""
    added: str = ""


class GeneralRule(BaseModel):
    rule: str = Field(description="A short, general instruction that applies to EVERY future application")
    applies_to: Scope = "all"
    cv_exclude: list[str] = Field(default_factory=list, description="CV terms to always leave out (only if the rule says so)")


class ProfileFact(BaseModel):
    skill: str = Field(description="The skill / technology exactly as the applicant names it, e.g. 'TensorFlow'")
    evidence: Literal["direct_professional", "academic", "project"] = Field(
        "project", description="direct_professional only if they say they used it in a job; academic for studies / "
                               "thesis; project for personal or university projects — when unclear: project")


class FeedbackPlan(BaseModel):
    profile_facts: list[ProfileFact] = Field(default_factory=list,
                                             description="Skills the applicant says THEY HAVE (added to their profile)")
    letter_changes: list[str] = Field(default_factory=list, description="Concrete changes to this cover letter")
    cv_changes: list[str] = Field(default_factory=list, description="Changes to this CV's profile statement / wording")
    cv_exclude: list[str] = Field(default_factory=list, description="Words, as they appear on the CV, of items to leave out (e.g. 'Unity')")
    cv_emphasize: list[str] = Field(default_factory=list, description="Skills / topics, as written on the CV, to put first (e.g. 'Robotik')")
    job_emphasize: str = Field("", description="What to emphasise for THIS job (kept in the job's notes)")
    job_avoid: str = Field("", description="What to avoid for THIS job (kept in the job's notes)")
    general_rules: list[GeneralRule] = Field(default_factory=list,
                                             description="Only preferences meant for ALL applications, else empty")


FEEDBACK_SYSTEM = """You turn an applicant's feedback on one job application (cover letter + CV) into instructions.

Split it into:
- letter_changes / cv_changes: what to change in THIS application, as clear instructions in English.
- cv_exclude: words identifying CV items (bullets, projects, skills) the applicant wants left OUT of this CV.
- cv_emphasize: skills or topics the applicant wants this CV to put first.
- job_emphasize / job_avoid: short notes about THIS job only.
- general_rules: ONLY what is clearly a lasting preference for all applications (style, length, tone, things to
  always / never mention). A remark about this particular company or posting is NOT a general rule. Do not repeat a
  rule the applicant already has (listed below). When in doubt, leave general_rules empty.
- profile_facts: every skill or technology the applicant says THEY HAVE experience with ("I have experience in
  TensorFlow", "I know MySQL and Django", "I have experience in all of those" about a pasted list of skills — then
  each skill of that list). One entry per skill, named exactly as written. These are added to the applicant's
  profile, so the CV and letter may use them. Choose the evidence level honestly from their words:
  direct_professional only if they say they used it at work / in a job; academic for studies or a thesis; project
  for personal or university projects; when they don't say: project.

Lines copied from the app's own keyword table ("TensorFlow<TAB>required<TAB>gap not in your profile — never stuffed
in") are just a LIST OF SKILLS the applicant points at; the remarks in them are the app's words, never requests.

Only record facts the applicant actually states — never invent or upgrade any. The feedback may be in any language;
write English."""


def feedback_user(feedback: str, job, existing: list[LearnedRule]) -> str:
    rules = "\n".join(f"- {r.text}" for r in existing) or "(none)"
    return (f"Job: {job.title} at {job.company}\n\nThe applicant's existing general rules:\n{rules}\n\n"
            f"The applicant's feedback on this application:\n<<<\n{feedback[:4000]}\n>>>")


# --------------------------------------------------------------------------- storage


def memory_path() -> Path:
    from app.settings import get_settings

    return Path(get_settings().config_dir) / "agent_memory.yaml"


_HEADER = ("# What the application agent learned from your feedback — applied to every new application\n"
           "# (letters, CVs and the reviewer's check). Edit or delete freely; the app adds rules when you give feedback\n"
           "# with \"learn from this\" ticked. applies_to: letter | cv | all. cv_exclude: CV items with these words are\n"
           "# always left out.\n\n")


def load_rules() -> list[LearnedRule]:
    try:
        data = yaml.safe_load(memory_path().read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return []
    except yaml.YAMLError as exc:
        log.warning("agent_memory.yaml can't be read: %s", exc)
        return []
    return [LearnedRule.model_validate(r) for r in data.get("rules") or [] if isinstance(r, dict) and r.get("text")]


def save_rules(rules: list[LearnedRule]) -> None:
    text = _HEADER + yaml.safe_dump({"rules": [r.model_dump() for r in rules]}, sort_keys=False, allow_unicode=True,
                                    width=110)
    path = memory_path()
    if path.exists():  # previous version kept, like every config change made in the app
        backups = path.parent / ".backups"
        backups.mkdir(exist_ok=True)
        from datetime import datetime

        (backups / f"agent_memory-{datetime.now():%Y%m%d-%H%M%S}.yaml").write_bytes(path.read_bytes())
    path.write_text(text, encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", text.lower()).strip()


def add_rules(new: list[GeneralRule | dict[str, Any]], learned_from: str = "") -> list[LearnedRule]:
    """Append rules (duplicates by wording are skipped). Returns the rules actually added."""
    rules = load_rules()
    seen = {_norm(r.text) for r in rules}
    next_id = max((r.id for r in rules), default=0) + 1
    added = []
    for item in new:
        g = item if isinstance(item, GeneralRule) else GeneralRule.model_validate(item)
        text = g.rule.strip()
        if not text or _norm(text) in seen:
            continue
        rule = LearnedRule(id=next_id, text=text[:400], applies_to=g.applies_to,
                           cv_exclude=[t.strip() for t in g.cv_exclude if t.strip()][:10],
                           learned_from=learned_from[:200], added=date.today().isoformat())
        rules.append(rule)
        added.append(rule)
        seen.add(_norm(text))
        next_id += 1
    if added:
        save_rules(rules)
    return added


def delete_rule(rule_id: int) -> bool:
    rules = load_rules()
    kept = [r for r in rules if r.id != rule_id]
    if len(kept) != len(rules):
        save_rules(kept)
    return len(kept) != len(rules)


def update_rule(rule_id: int, text: str, applies_to: str) -> bool:
    rules = load_rules()
    for r in rules:
        if r.id == rule_id:
            r.text = text.strip()[:400] or r.text
            r.applies_to = applies_to if applies_to in ("letter", "cv", "all") else r.applies_to
            save_rules(rules)
            return True
    return False


# --------------------------------------------------------------------------- use in prompts


def rules_for(target: str) -> list[LearnedRule]:
    """target: letter | cv | review (the reviewer checks every rule)."""
    return [r for r in load_rules() if target == "review" or r.applies_to in (target, "all")]


def instructions_block(target: str) -> str:
    rules = rules_for(target)
    if not rules:
        return ""
    lines = "\n".join(f"- {r.text}" for r in rules)
    if target == "review":
        return ("\n\nTHE APPLICANT'S STANDING INSTRUCTIONS (learned from their earlier feedback). An application that "
                f"ignores one of them needs revising — name the rule in your reasons:\n{lines}\n")
    return ("\n\nTHE APPLICANT'S STANDING INSTRUCTIONS (learned from their earlier feedback) — follow them in this "
            f"document. They are preferences about style, emphasis and selection, never facts:\n{lines}\n")


def cv_excludes() -> list[str]:
    return [t for r in rules_for("cv") for t in r.cv_exclude]


# --------------------------------------------------------------------------- feedback → plan


# A line pasted from the app's keyword table: "TensorFlow\trequired\tgap not in your profile — never stuffed in"
_TABLE_LINE = re.compile(r"^\s*([^\t\n]{2,60}?)\s*\t\s*(required|preferred|nice[ -]to[ -]have|must|optional)\b", re.I | re.M)
_GAP_LINE = re.compile(r"^\s*([^\t\n]{2,60}?)(?:\s*\t\s*|\s+[|·–-]\s+).*?(not in your profile|genuine gap|never stuffed)",
                       re.I | re.M)
_I_HAVE = re.compile(r"\b(i\s+(do\s+)?have|i've|i\s+know|i\s+worked|i\s+used|ich\s+habe|ich\s+kenne|habe\s+ich|"
                     r"erfahrung\s+(mit|in)|experience\s+(with|in))\b", re.I)
_AT_WORK = re.compile(r"\b(at work|in my (job|work|company)|professional(ly)?|beruflich|im job|in der arbeit|bei der arbeit|"
                      r"working student|werkstudent|internship|praktikum)\b", re.I)
_STUDIES = re.compile(r"\b(thesis|studies|study|university|uni\b|course|masterarbeit|bachelorarbeit|studium|vorlesung)\b", re.I)


def pasted_skill_facts(feedback: str) -> list[ProfileFact]:
    """"I have experience in all of those" + lines pasted from the keyword table → one fact per listed skill (read by
    code, not the model: the table's own remarks confused it). Evidence from the wording: at work → professional,
    studies / thesis → academic, otherwise project."""
    if not _I_HAVE.search(feedback):
        return []
    terms = [m.group(1).strip() for m in _TABLE_LINE.finditer(feedback)]
    terms += [m.group(1).strip() for m in _GAP_LINE.finditer(feedback)]
    level = "direct_professional" if _AT_WORK.search(feedback) else "academic" if _STUDIES.search(feedback) else "project"
    seen, out = set(), []
    for t in terms:
        if t and t.lower() not in seen and not _I_HAVE.search(t):
            seen.add(t.lower())
            out.append(ProfileFact(skill=t, evidence=level))
    return out


def posting_spelling(skill: str, posting: str) -> str:
    """The posting's own spelling of a skill the applicant typed in lower case ("mysql" → "MySQL"); otherwise as typed."""
    skill = skill.strip()
    if not skill or skill != skill.lower():
        return skill
    m = re.search(r"(?<![\w.+#-])" + re.escape(skill) + r"(?![\w+#])", posting, re.I)
    return m.group(0) if m else skill


def plan_feedback(session, job, feedback: str, setup) -> FeedbackPlan:
    """One model call; without a model the feedback goes to letter and CV as it is (nothing is learned)."""
    from app.llm.roles import Role
    from app.services.llm_calls import llm_call

    target = setup.target(Role.VALIDATOR) or setup.target(Role.WRITER)
    if target is None:
        return FeedbackPlan(letter_changes=[feedback], cv_changes=[feedback], profile_facts=pasted_skill_facts(feedback))
    try:
        res = llm_call(session, target, kind="feedback_plan", job_id=job.id, prompt_version=FEEDBACK_VERSION,
                       profile_hash="-", system=FEEDBACK_SYSTEM, user=feedback_user(feedback, job, load_rules()),
                       schema=FeedbackPlan, use_cache=False)
        plan = FeedbackPlan.model_validate(res.data)
    except Exception as exc:  # noqa: BLE001 - the applicant's words still reach both documents
        log.warning("Feedback could not be structured (%s) — applying it as written", exc)
        return FeedbackPlan(letter_changes=[feedback], cv_changes=[feedback])
    pasted = pasted_skill_facts(feedback)
    if pasted:  # skills they point at in a pasted table: theirs, never "leave out"
        known = {f.skill.lower() for f in plan.profile_facts}
        plan.profile_facts += [f for f in pasted if f.skill.lower() not in known]
        names = {f.skill.lower() for f in pasted}
        plan.cv_exclude = [t for t in plan.cv_exclude if t.lower() not in names]
    for fact in plan.profile_facts:  # "mysql" typed in a hurry → "MySQL" as the posting writes it
        fact.skill = posting_spelling(fact.skill, getattr(job, "description", None) or "")
    if not (plan.letter_changes or plan.cv_changes or plan.cv_exclude or plan.cv_emphasize or plan.profile_facts):
        plan.letter_changes = [feedback]  # never drop the applicant's request
    return plan


LEVEL_LABEL = {"direct_professional": "professional", "academic": "academic", "project": "project"}


def add_profile_facts(facts: list[ProfileFact]) -> list[str]:
    """Skills you said you have → config/profile.yaml (your source of truth; comments kept, previous version backed
    up). Skills already in the profile at any level are left alone. Returns e.g. ["TensorFlow (project)"]."""
    from app.config_files import get_profile
    from app.services import config_writer

    profile = get_profile()
    known = {s.strip().lower() for s, _ in profile.skills.all()}
    groups = {level: list(getattr(profile.skills, level)) for level in LEVEL_LABEL}
    added = []
    for fact in facts:
        name = re.sub(r"\s+", " ", fact.skill).strip(" .,;:-")[:80]
        if not name or name.lower() in known:
            continue
        groups[fact.evidence].append(name)
        known.add(name.lower())
        added.append(f"{name} ({LEVEL_LABEL[fact.evidence]})")
    if added:
        config_writer.update("profile.yaml", {f"skills.{level}": items for level, items in groups.items()
                                              if items != list(getattr(profile.skills, level))})
    return added
