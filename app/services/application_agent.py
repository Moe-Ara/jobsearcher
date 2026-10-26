"""The three agents, end to end (modelled on the drafter → reviewer workflow of MadsLorentzen/ai-job-search):

  1. Scout agent        finds and scores jobs — the existing pipeline (feeds, e-mail alerts, academic sources →
                        hard filters → screening → deep fit analysis → score). Here it only runs the fit analysis
                        first when the job has none, because the CV and letter build on it.
  2. Application agent  tailors the CV (config/cv.yaml), writes the cover letter (the validated letter pipeline)
                        and drafts the application e-mail.
  3. Reviewer agent     checks the package as an ATS and as a recruiter → SEND or REVISE. On REVISE the application
                        agent applies the feedback (CV keywords/changes, letter rewrite + fact check) and the
                        reviewer looks again, up to application.max_revision_rounds times.

Runs only on an explicit click. Nothing is ever submitted, sent or uploaded — the package is for you to send.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config_files import get_prefs, get_profile
from app.db import session_scope
from app.llm.factory import LLMSetup, Role, get_llm
from app.models import ApplicationDocument, ApplicationPackage, Job, utcnow
from app.services import tasks
from app.services.analysis import analyze_job
from app.services.cv import CvError, cv_vocabulary, get_master_cv, render_text, tailor_cv
from app.services import toolbox
from app.services.keywords import job_keywords
from app.llm.local_provider import interactive
from app.services.letters import _gpu_guard, detect_language, generate_letter, revise_letter
from app.services.research import research_block, research_company
from app.services.reviewer import review_application

log = logging.getLogger(__name__)


class PostingClosed(ValueError):
    """The vacancy is no longer open — the agents do not run (the job is set to Expired)."""


class ApplicationError(ValueError):
    pass


def task_name(job_id: int) -> str:
    return f"apply-{job_id}"


def busy(job_id: int) -> bool:
    """The application agent and the single-letter buttons write the same letters → never both at once."""
    return tasks.is_running(task_name(job_id)) or tasks.is_running(f"letter-{job_id}")


def _needs_analysis(job: Job) -> bool:
    a = job.current_analysis
    return a is None or a.provider == "heuristic" or not a.result or "strong_matches" not in a.result


def package_letter(session: Session, job_id: int, group_id: str | None, reviews: list[dict[str, Any]]) -> ApplicationDocument | None:
    """The letter version of the best review round (falls back to the latest finished version)."""
    best = best_review(reviews)
    doc_id = best.get("letter_document_id") if best else None
    doc = session.get(ApplicationDocument, doc_id) if doc_id else None
    if doc is not None and doc.job_id == job_id:
        return doc
    return _latest_final(session, job_id, group_id) if group_id else None


def _latest_final(session: Session, job_id: int, group_id: str) -> ApplicationDocument | None:
    return session.scalars(select(ApplicationDocument).where(
        ApplicationDocument.job_id == job_id, ApplicationDocument.group_id == group_id,
        ApplicationDocument.stage == "final").order_by(ApplicationDocument.id.desc())).first()


_SALUTATIONS = ("sehr geehrte", "liebe", "hallo", "guten tag", "dear", "hello", "hi ")


def bundle_filename(job: Job, lang: str, name: str) -> str:
    company = re.sub(r"[^A-Za-z0-9]+", "_", job.company or "")[:40].strip("_")
    return f"{'Bewerbung' if lang == 'de' else 'Application'}_{name.replace(' ', '_')}{'_' + company if company else ''}.pdf"


def draft_email(job: Job, letter: str, lang: str, name: str, contact: dict[str, Any]) -> dict[str, Any]:
    """Short application e-mail (deterministic: no facts beyond the job, your name and contact details).
    Reuses the letter's own salutation, so a named contact person carries over. With certificates configured the
    attachment is the one complete application PDF (what German employers usually ask for)."""
    first = (letter or "").strip().split("\n", 1)[0].strip()
    salutation = first if first.lower().startswith(_SALUTATIONS) and len(first) < 120 else (
        "Sehr geehrte Damen und Herren," if lang == "de" else "Dear Hiring Team,")
    sign = "\n".join(x for x in (name, contact.get("phone"), contact.get("email")) if x)
    with_documents = bool(available_documents()[0])
    if with_documents:
        attachments = [bundle_filename(job, lang, name)]
        what_de = "meine vollständigen Bewerbungsunterlagen (Anschreiben, Lebenslauf und Zeugnisse) in einer PDF-Datei"
        what_en = "my complete application (cover letter, CV and certificates) as one PDF"
    else:
        attachments = [f"{'Anschreiben' if lang == 'de' else 'Cover_Letter'}_{name.replace(' ', '_')}.pdf",
                       f"{'Lebenslauf' if lang == 'de' else 'CV'}_{name.replace(' ', '_')}.pdf"]
        what_de, what_en = "mein Anschreiben und meinen Lebenslauf", "my cover letter and my CV"
    if lang == "de":
        subject = subject_line(job, "de")
        body = (f"{salutation}\n\nanbei sende ich Ihnen meine Bewerbung für die Position „{job.title}“"
                f"{' bei ' + job.company if job.company else ''}. Im Anhang finden Sie {what_de}.\n\n"
                f"Über eine Rückmeldung und die Gelegenheit zu einem persönlichen Gespräch freue ich mich.\n\n"
                f"Mit freundlichen Grüßen\n{sign}")
    else:
        subject = subject_line(job, "en")
        body = (f"{salutation}\n\nplease find attached my application for the position “{job.title}”"
                f"{' at ' + job.company if job.company else ''}: {what_en}.\n\n"
                f"I look forward to hearing from you and would welcome the opportunity to talk.\n\nKind regards\n{sign}")
    return {"subject": subject, "body": body, "attachments": attachments}


def _rank(review: dict[str, Any]) -> tuple:
    """Better review first: SEND, then recruiter quality, then ATS score, then fewer reasons."""
    return (review["verdict"] == "SEND", review.get("quality") or 0, review["ats"]["score"], -len(review["reasons"]))


def best_review(reviews: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The best round — but a round you asked for (your feedback) always wins over a higher-scoring earlier one."""
    if not reviews:
        return None
    yours = [r for r in reviews if r.get("user_revision")]
    return yours[-1] if yours else max(reviews, key=_rank)


def _save(pkg_id: int, **fields: Any) -> None:
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        for k, v in fields.items():
            setattr(pkg, k, v)


def start_package(job_id: int, language: str | None) -> int:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise ApplicationError("Job not found")
        lang = language if language in ("de", "en") else detect_language(job)[0]
        pkg = ApplicationPackage(job_id=job_id, status="running", language=lang, reviews=[])
        s.add(pkg)
        s.flush()
        return pkg.id


def ensure_open(job_id: int) -> None:
    """Right before the agents start: is the posting still open? Raises PostingClosed (job → Expired) if not."""
    from app.services import liveness

    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise ApplicationError("Job not found")
        tasks.progress("Checking that the posting is still open")
        result = liveness.check(job)
        raw = dict(job.raw_source_data or {})
        raw["liveness"] = {"status": result.status, "reason": result.reason, "at": utcnow().isoformat(timespec="seconds")}
        job.raw_source_data = raw
        if result.closed:
            liveness.mark_closed(s, job, result)
    if result.closed:  # raised after the commit — inside the session the Expired status would be rolled back
        raise PostingClosed(f"The posting is closed: {result.reason}")


def run_application(job_id: int, language: str | None = None, package_id: int | None = None,
                    setup: LLMSetup | None = None, check_open: bool = True) -> dict[str, Any]:
    """Full run → final package. Returns a short summary (shown as the task result)."""
    if check_open:
        try:
            ensure_open(job_id)
        except PostingClosed as exc:
            if package_id:  # started from the job page: the empty package is not kept
                with session_scope() as s:
                    pkg = s.get(ApplicationPackage, package_id)
                    if pkg is not None:
                        s.delete(pkg)
            raise ApplicationError(str(exc)) from exc
    setup = setup or get_llm()
    if setup.target(Role.WRITER) is None:
        raise ApplicationError("No writer model configured (enable LOCAL_WRITER or configure a cloud provider)")
    try:
        get_master_cv()
    except CvError as e:
        raise ApplicationError(str(e)) from e
    pkg_id = package_id or start_package(job_id, language)
    try:
        with interactive():  # the whole run has GPU priority over background batches (also between its stages)
            return _run(job_id, pkg_id, setup)
    except Exception as exc:
        _save(pkg_id, status="failed", error=str(exc)[:2000])
        raise


class _Stages:
    """Wall-clock time per agent stage (shown on the package page next to the tool log)."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.started = time.monotonic()

    @contextlib.contextmanager
    def __call__(self, name: str):
        t = time.monotonic()
        try:
            yield
        finally:
            self.items.append({"stage": name, "seconds": round(time.monotonic() - t, 1)})

    def total(self) -> float:
        return round(time.monotonic() - self.started, 1)


def _run(job_id: int, pkg_id: int, setup: LLMSetup) -> dict[str, Any]:
    stages = _Stages()
    meta: dict[str, Any] = {"stages": stages.items}
    with toolbox.recording() as calls:
        meta["tools"] = calls
        try:
            return _run_agents(job_id, pkg_id, setup, stages, meta)
        finally:
            meta["seconds_total"] = stages.total()
            _merge_meta(pkg_id, json.loads(json.dumps(meta, default=str)))


def _merge_meta(pkg_id: int, run_meta: dict[str, Any]) -> None:
    """The run's log (stages, tools, research) goes next to what was recorded during the run — the export folder,
    a submission — instead of replacing it (a lost "export" entry left stale folders behind on delete / re-run)."""
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is not None:
            pkg.meta = {**(pkg.meta or {}), **run_meta}


def _run_agents(job_id: int, pkg_id: int, setup: LLMSetup, stage: _Stages, meta: dict[str, Any]) -> dict[str, Any]:
    prefs, profile = get_prefs(), get_profile()
    p = prefs.application
    with session_scope() as s:
        lang = s.get(ApplicationPackage, pkg_id).language
        job = s.get(Job, job_id)
        needs = p.analyse_first and _needs_analysis(job) and setup.has_llm
    # 1. scout agent: fit analysis first, so CV and letter can build on the strong / transferable matches
    if needs:
        tasks.progress("Scout agent: analysing the job first (the CV and letter build on it)")
        with stage("scout: fit analysis"), _gpu_guard(setup.target(Role.SCREEN), setup.target(Role.ANALYSIS)), \
                session_scope() as s:
            analyze_job(s, s.get(Job, job_id), setup)
    cv_lang = p.cv_language if p.cv_language in ("de", "en") else lang

    # 2. research agent: what the company does (cached 30 days per company)
    research_text = ""
    if p.research_enabled:
        with stage("research: company & position"), _gpu_guard(setup.target(Role.VALIDATOR)), session_scope() as s:
            research = research_company(s, s.get(Job, job_id), setup, lang)
        meta["research"] = research
        research_text = research_block(research)

    # 3. application agent: cover letter (draft → editor → incremental fact check → final), then the tailored CV
    tasks.progress("Application agent: writing the cover letter")
    with stage("application: cover letter"):
        group_id = generate_letter(job_id, language=lang, setup=setup, research=research_text,
                                   personalization_review=False)
    _save(pkg_id, letter_group_id=group_id)

    revision: dict[str, Any] | None = None
    reviews: list[dict[str, Any]] = []
    cv: dict[str, Any] = {}
    review: dict[str, Any] = {}
    max_rounds = max(0, p.max_revision_rounds)
    for round_no in range(1, max_rounds + 2):
        with _gpu_guard(setup.target(Role.WRITER), setup.target(Role.VALIDATOR)):
            with stage(f"application: CV (round {round_no})"), session_scope() as s:
                job = s.get(Job, job_id)
                tasks.progress(f"Application agent: tailoring the CV (round {round_no})")
                cv, _ = tailor_cv(s, job, cv_lang, setup, profile, revision)
                pdf, engine = toolbox.render_cv(cv)
                cv["meta"]["renderer"] = engine
            # 4. reviewer agent: ATS + recruiter → SEND / REVISE
            with stage(f"reviewer: ATS + recruiter (round {round_no})"), session_scope() as s:
                job = s.get(Job, job_id)
                keywords = job_keywords(job, cv_vocabulary(get_master_cv(), profile))
                letter_doc = _latest_final(s, job_id, group_id)
                review, _ = review_application(s, job, cv, pdf, keywords, letter_doc, setup, profile, research_text)
        review["round"] = round_no
        review["at"] = utcnow().isoformat(timespec="seconds")
        review["cv_snapshot"] = {"profile_statement": cv["profile_statement"], "extra_skills": cv["meta"]["extra_skills"]}
        review["cv"] = cv
        previous = best_review(reviews)
        reviews.append(review)
        best = best_review(reviews)  # a revision can make things worse: the package always holds the best round
        review["best"] = best is review
        rec = review.get("recruiter") or {}
        _save(pkg_id, reviews=list(reviews), rounds=round_no, cv=best["cv"], ats_score=best["ats"]["score"],
              recruiter_score=best.get("quality"), verdict=best["verdict"])
        if review["verdict"] == "SEND" or round_no > max_rounds or not review.get("fixable", True):
            break
        if previous is not None and _rank(review) <= _rank(previous):
            log.info("Revision round %d did not improve the application — stopping", round_no)
            review["stopped"] = "no improvement over the previous round"
            _save(pkg_id, reviews=list(reviews))
            break
        revision = review["revision"] or {}
        letter_changes = list(revision.get("letter_changes") or [])
        if letter_changes or revision.get("letter_keywords") or rec.get("letter_score", 10) < p.recruiter_send_threshold:
            with stage(f"application: letter revision (round {round_no})"):
                revise_letter(job_id, group_id, letter_changes or ["Make the letter more specific to this posting's tasks."],
                              revision.get("letter_keywords"), setup=setup, research=research_text)

    best = best_review(reviews)
    with session_scope() as s:
        job = s.get(Job, job_id)
        letter_doc = package_letter(s, job_id, group_id, reviews)
        email = draft_email(job, letter_doc.content if letter_doc else "", lang, best["cv"]["name"], best["cv"]["contact"])
    _save(pkg_id, status="done", email=email, cv=best["cv"], error=None, verdict=best["verdict"],
          ats_score=best["ats"]["score"], recruiter_score=best.get("quality"))
    log.info("Application package %s for job %s: %s (best of %d round(s))", pkg_id, job_id, best["verdict"], len(reviews))
    try:  # the papers as files in data/applications — a failed export never fails the run
        from app.services.export import export_package

        tasks.progress("Saving the application papers to data/applications")
        export_package(pkg_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Export of package %s failed: %s", pkg_id, exc)
    return {"package": pkg_id, "verdict": best["verdict"], "ats": best["ats"]["score"],
            "recruiter": best.get("quality"), "rounds": len(reviews), "best_round": best["round"]}


def revise_with_feedback(pkg_id: int, feedback: str, learn: bool = True, setup: LLMSetup | None = None) -> dict[str, Any]:
    """Your feedback on a finished package → (1) saved with the package, (2) structured: changes for this letter and
    CV, notes for this job, general rules for every future application (when `learn`), (3) a new round: the letter
    revised, the CV re-tailored, both reviewed again. Your round becomes the package's version. A failed revision
    leaves the package as it was."""
    from app.services import agent_memory
    from app.services.application_notes import get_notes, save_notes

    setup = setup or get_llm()
    feedback = (feedback or "").strip()[:4000]
    with session_scope() as s:
        pkg = s.get(ApplicationPackage, pkg_id)
        if pkg is None or pkg.status != "done" or not pkg.cv:
            raise ApplicationError("Only a finished application can be revised")
        job_id, group_id, lang = pkg.job_id, pkg.letter_group_id, pkg.language
        reviews = list(pkg.reviews or [])
        research_text = research_block((pkg.meta or {}).get("research") or {}) if (pkg.meta or {}).get("research") else ""
        pkg.status = "running"
    entry: dict[str, Any] = {"at": utcnow().isoformat(timespec="seconds"), "text": feedback}
    try:
        with interactive(), toolbox.recording():
            tasks.progress("Reading your feedback")
            with _gpu_guard(setup.target(Role.VALIDATOR) or setup.target(Role.WRITER)), session_scope() as s:
                job = s.get(Job, job_id)
                plan = agent_memory.plan_feedback(s, job, feedback, setup)
                entry["plan"] = plan.model_dump()
                notes = get_notes(s, job_id)  # this job's notes: what to emphasise / avoid next time too
                updates = {f: "\n".join(x for x in (getattr(notes, f), new) if x)
                           for f, new in (("emphasize", plan.job_emphasize), ("avoid", plan.job_avoid)) if new}
                if updates:
                    save_notes(s, job_id, **updates)
            if learn and plan.general_rules:
                added = agent_memory.add_rules(plan.general_rules, learned_from=f"feedback on application #{pkg_id}")
                entry["learned"] = [r.text for r in added]
            if group_id and (plan.letter_changes or plan.job_avoid or plan.job_emphasize):
                tasks.progress("Application agent: revising the cover letter with your feedback")
                revise_letter(job_id, group_id, [f"The applicant asks: {c}" for c in (plan.letter_changes or [feedback])],
                              None, setup=setup, research=research_text)
            revision = {"cv_changes": plan.cv_changes, "exclude": plan.cv_exclude, "emphasize": plan.cv_emphasize}
            cv_lang = get_prefs().application.cv_language
            cv_lang = cv_lang if cv_lang in ("de", "en") else lang
            profile = get_profile()
            with _gpu_guard(setup.target(Role.WRITER), setup.target(Role.VALIDATOR)):
                with session_scope() as s:
                    job = s.get(Job, job_id)
                    tasks.progress("Application agent: re-tailoring the CV with your feedback")
                    cv, _ = tailor_cv(s, job, cv_lang, setup, profile, revision)
                    pdf, engine = toolbox.render_cv(cv)
                    cv["meta"]["renderer"] = engine
                with session_scope() as s:
                    job = s.get(Job, job_id)
                    keywords = job_keywords(job, cv_vocabulary(get_master_cv(), profile))
                    letter_doc = _latest_final(s, job_id, group_id) if group_id else None
                    review, _ = review_application(s, job, cv, pdf, keywords, letter_doc, setup, profile, research_text)
            review.update(round=len(reviews) + 1, at=utcnow().isoformat(timespec="seconds"), cv=cv, user_revision=True,
                          feedback=feedback, cv_snapshot={"profile_statement": cv["profile_statement"],
                                                          "extra_skills": cv["meta"]["extra_skills"]})
            reviews.append(review)
            for r in reviews:
                r["best"] = r is review
            with session_scope() as s:
                job = s.get(Job, job_id)
                email = draft_email(job, letter_doc.content if letter_doc else "", lang, cv["name"], cv["contact"])
        entry.update(verdict=review["verdict"], round=review["round"])
        _save(pkg_id, status="done", reviews=reviews, rounds=len(reviews), cv=cv, email=email, verdict=review["verdict"],
              ats_score=review["ats"]["score"], recruiter_score=review.get("quality"), error=None)
    except Exception as exc:
        entry["error"] = str(exc)[:500]
        _save(pkg_id, status="done")  # the package keeps its previous version
        raise
    finally:
        with session_scope() as s:
            pkg = s.get(ApplicationPackage, pkg_id)
            if pkg is not None:
                pkg.meta = {**(pkg.meta or {}), "feedback": [*((pkg.meta or {}).get("feedback") or []), entry]}
    try:
        from app.services.export import export_package

        export_package(pkg_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Export of package %s failed: %s", pkg_id, exc)
    return {"package": pkg_id, "verdict": review["verdict"], "learned": entry.get("learned", [])}


# --------------------------------------------------------------------------- queries & downloads


def mark_interrupted() -> int:
    """At startup no run can be alive: packages still 'running' were killed by a restart."""
    with session_scope() as s:
        stale = s.scalars(select(ApplicationPackage).where(ApplicationPackage.status == "running")).all()
        for pkg in stale:
            pkg.status = "failed"
            pkg.error = "Interrupted — the app was restarted during the run. Start the application agent again."
        return len(stale)



def packages_for(session: Session, job_id: int) -> list[ApplicationPackage]:
    return list(session.scalars(select(ApplicationPackage).where(ApplicationPackage.job_id == job_id)
                                .order_by(ApplicationPackage.created_at.desc(), ApplicationPackage.id.desc())))


def recent_packages(session: Session, limit: int | None = 50) -> list[ApplicationPackage]:
    q = select(ApplicationPackage).order_by(ApplicationPackage.created_at.desc())
    return list(session.scalars(q.limit(limit) if limit else q))


# "(m/w/d)", "(gn)", "[f/m/x]" — anywhere in the title ("Simulation Engineer (gn) NVIDIA Omniverse")
_GENDER_SUFFIX = re.compile(r"\s*[\(\[]\s*(m|w|f|d|x|gn|div|all genders?|alle)\s*([/|,*]\s*\w{1,3}\s*)*[\)\]]", re.I)
_CONTACT_DE = re.compile(r"^Sehr geehrte(r)?\s+(Frau|Herr)\s+(.+?),\s*$")
_CONTACT_EN = re.compile(r"^Dear\s+(Ms|Mrs|Mr|Dr)\.?\s+(.+?),\s*$")


def contact_person(letter: str) -> str:
    """'Sehr geehrte Frau Weber,' → 'Frau Weber'; 'Sehr geehrter Herr Lang,' → 'Herrn Lang'; else ''."""
    first = (letter or "").strip().split("\n", 1)[0].strip()
    if m := _CONTACT_DE.match(first):
        return f"{'Herrn' if m.group(2) == 'Herr' else 'Frau'} {m.group(3)}"
    if m := _CONTACT_EN.match(first):
        return f"{m.group(1)} {m.group(2)}"
    return ""


def subject_line(job: Job, lang: str) -> str:
    """'Bewerbung um die Stelle als Softwareentwickler' — the exact job title without the (m/w/d) suffix."""
    title = re.sub(r"\s{2,}", " ", _GENDER_SUFFIX.sub("", job.title or "")).strip(" -–|,")
    return f"Bewerbung um die Stelle als {title}" if lang == "de" else f"Application for the position of {title}"


def _recipient(job: Job, letter: str) -> list[str]:
    """Company, department (if the posting gave one), contact person, place — the address is not known, so the town
    alone stands in for 'Straße / PLZ Ort'."""
    return [job.company or "", getattr(job, "department", None) or "", contact_person(letter), job.location or ""]


def documents_dir():
    """<data dir>/documents (in Docker: /data/documents)."""
    from app.settings import data_dir

    return data_dir() / "documents"


def available_documents() -> tuple[list[tuple[Any, Any]], list[str]]:
    """([(document ref, path)] that exist, [missing file names]) for the configured certificates."""
    found, missing = [], []
    for ref in get_prefs().application.documents:
        path = documents_dir() / ref.file
        if path.is_file():
            found.append((ref, path))
        else:
            missing.append(ref.file)
    return found, missing


def attachment_labels(lang: str) -> list[str]:
    """"Anlagen" under the letter: the CV plus the certificates that will really be attached."""
    p = get_prefs().application
    docs, _ = available_documents()
    if docs:
        cv_label = "Lebenslauf" if lang == "de" else "CV"
        return [cv_label] + [(r.label_de if lang == "de" else (r.label_en or r.label_de)) for r, _ in docs]
    return p.letter_attachments_de if lang == "de" else p.letter_attachments_en


def bundle_pdf(parts: list[bytes], documents: list[tuple[Any, Any]]) -> bytes:
    """One application PDF: the given PDFs (letter, CV) followed by the certificates (minus skipped pages)."""
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for data in parts:
        for page in PdfReader(io.BytesIO(data)).pages:
            writer.add_page(page)
    for ref, path in documents:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, 1):
            if i not in ref.skip_pages:
                writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _date_line(lang: str, when: datetime) -> str:
    return f"{when:%d.%m.%Y}" if lang == "de" else f"{when:%d %B %Y}"


def current_cv(pkg: ApplicationPackage) -> dict[str, Any]:
    """The package's tailored CV with today's name/contact/personal lines from config/cv.yaml — an address or link
    you fill in later shows up in packages made before it."""
    try:
        master = get_master_cv()
    except CvError:
        return pkg.cv
    lang = pkg.cv.get("language", "de")
    personal = [getattr(x, lang) for x in master.personal if getattr(x, lang) and x.only in (None, lang)]
    return {**pkg.cv, "name": master.name, "contact": master.contact.model_dump(), "personal": personal}


def layout_text(job: Job, letter: ApplicationDocument) -> str:
    """The letter body as it goes on paper: no model-written header block, placeholder, invented contact data or
    unnamed contact person — also for letters written before those guards existed."""
    from app.services.letters import drop_placeholders, ensure_salutation, fix_degree_tense, strip_header

    text, _ = strip_header(letter.content)
    text, _ = drop_placeholders(text)
    text, _ = fix_degree_tense(text, letter.language)
    job_text = " ".join(x or "" for x in (job.title, job.company, job.description, job.requirements, job.responsibilities))
    return ensure_salutation(text, letter.language, job_text)[0]


def _letter_pdf(job: Job, letter: ApplicationDocument, pkg: ApplicationPackage) -> bytes:
    text = layout_text(job, letter)
    return toolbox.render_letter(text, subject_line(job, letter.language), current_cv(pkg),
                                 _recipient(job, text), _date_line(letter.language, utcnow()),
                                 attachment_labels(letter.language), letter.language)[0]


def package_file(session: Session, pkg: ApplicationPackage, kind: str) -> tuple[bytes, str, str]:
    """(content, filename, media type) for a package download:
    cv.pdf | cv.tex | cv.txt | letter.pdf | letter.txt | bundle.pdf (letter + CV + certificates in one file)."""
    from app.services.latex import cv_tex, letter_tex

    job = session.get(Job, pkg.job_id)
    if not pkg.cv:
        raise ApplicationError("This package has no CV yet")
    name = pkg.cv["name"].replace(" ", "_")
    de = pkg.cv["language"] == "de"
    if kind == "cv.pdf":
        return toolbox.render_cv(current_cv(pkg))[0], f"{'Lebenslauf' if de else 'CV'}_{name}.pdf", "application/pdf"
    if kind == "cv.tex":
        return cv_tex(current_cv(pkg)).encode("utf-8"), f"{'Lebenslauf' if de else 'CV'}_{name}.tex", "application/x-tex"
    if kind == "cv.txt":
        return render_text(current_cv(pkg)).encode("utf-8"), f"{'Lebenslauf' if de else 'CV'}_{name}.txt", "text/plain; charset=utf-8"
    letter = package_letter(session, pkg.job_id, pkg.letter_group_id, pkg.reviews or [])
    if letter is None:
        raise ApplicationError("This package has no finished cover letter")
    lde = letter.language == "de"
    lname = f"{'Anschreiben' if lde else 'Cover_Letter'}_{name}"
    if kind == "letter.pdf":
        return _letter_pdf(job, letter, pkg), f"{lname}.pdf", "application/pdf"
    if kind == "bundle.pdf":
        data = toolbox.merge_pdfs([_letter_pdf(job, letter, pkg), toolbox.render_cv(current_cv(pkg))[0]],
                                  available_documents()[0])
        return data, bundle_filename(job, letter.language, pkg.cv["name"]), "application/pdf"
    if kind == "letter.txt":
        return layout_text(job, letter).encode("utf-8"), f"{lname}.txt", "text/plain; charset=utf-8"
    if kind == "letter.tex":
        text = layout_text(job, letter)
        _, engine = toolbox.render_letter(text, subject_line(job, letter.language), current_cv(pkg),
                                          _recipient(job, text), _date_line(letter.language, utcnow()),
                                          attachment_labels(letter.language), letter.language)
        level = int(engine.rsplit("fit=", 1)[1]) if "fit=" in engine else 0  # the same layout as the PDF
        src = letter_tex(text, subject_line(job, letter.language), current_cv(pkg), _recipient(job, text),
                         _date_line(letter.language, utcnow()), attachment_labels(letter.language), letter.language,
                         fit=level)
        return src.encode("utf-8"), f"{lname}.tex", "application/x-tex"
    raise ApplicationError("Unknown file")
