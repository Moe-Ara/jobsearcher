"""Downloadable job-search report (PDF) for the weeks you choose, in English or German.

Every number belongs to the chosen weeks:
- a week-by-week table: jobs found, applications sent, employer answers (rejections, next steps, interviews, offers),
  receipt confirmations and packages prepared;
- what became of the applications sent in those weeks (answered, rejected, interview, still waiting, days to answer);
- a dated timeline of every employer answer and the list of applications;
- response rates by source, role family and country.

Built with reportlab (platypus + reportlab.graphics) — no extra dependencies. `build_report(s, weeks, lang)` returns
the PDF bytes; weeks are ISO weeks like "2026-W47" (none = every week with activity).
"""

from __future__ import annotations

import io
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.enums import FilterStatus, JobStatus
from app.models import ApplicationPackage, EmailMessage, Job, StatusEvent, utcnow
from app.web.templating import source_label

NAVY = colors.HexColor("#1F3A68")
ACCENT = colors.HexColor("#2F6FEB")
TEAL = colors.HexColor("#14A38B")
AMBER = colors.HexColor("#E3A008")
ROSE = colors.HexColor("#E05A5A")
INK = colors.HexColor("#1D2433")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#E5E7EB")
TINT = colors.HexColor("#F3F6FB")

# answers from an employer (after you applied), in funnel order
_ANSWERS = (JobStatus.HR_SCREEN, JobStatus.INTERVIEW, JobStatus.TECHNICAL, JobStatus.FINAL, JobStatus.OFFER, JobStatus.REJECTED)
_INTERVIEWS = (JobStatus.INTERVIEW, JobStatus.TECHNICAL, JobStatus.FINAL)

# ---------------------------------------------------------------------------------------------------- language

T: dict[str, dict[str, str]] = {
    "en": {
        "title": "Job Search Report", "all_weeks": "every week", "generated": "generated",
        "week": "W{w}", "weeks_range": "Weeks {a}–{b} {y}", "weeks_list": "Weeks {list}",
        "k_applied": "Applications sent", "k_answered": "Answers received", "k_interviews": "Interviews & next steps",
        "k_rejections": "Rejections received", "k_waiting": "Sent then, no answer yet", "k_packages": "Packages prepared",
        "found_line": "In these weeks the scout found {found} jobs that passed your filters; {strong} of them scored 8 or more.",
        "takeaways": "Key takeaways", "by_week": "Week by week",
        "c_week": "Week", "c_dates": "Dates", "c_found": "Found", "c_strong": "8+", "c_applied": "Applied",
        "c_confirm": "Receipts", "c_rejections": "Rejected", "c_next": "Next step", "c_interviews": "Interview",
        "c_offers": "Offer", "c_packages": "Packages", "total": "Total",
        "week_note": "Answers are counted in the week they arrived (also for applications sent earlier). Receipts = "
                     "confirmation e-mails. The current week is still running.",
        "chart": "Applications and answers per week", "s_applied": "Applications", "s_rejections": "Rejections",
        "s_positive": "Next steps & interviews",
        "outcome": "What became of the applications sent in these weeks",
        "outcome_note": "Each application counts once, by its furthest answer so far. Waiting = no answer yet.",
        "o_applied": "Sent", "o_answered": "Answered", "o_next": "Next step (HR)", "o_interview": "Interview",
        "o_offer": "Offer", "o_rejected": "Rejected", "o_waiting": "No answer yet",
        "of": "{n} of {total} ({pct}%)",
        "median": "Median time to an answer: {d} days (fastest {lo}, slowest {hi}).",
        "positive": "Interviews and next steps", "timeline": "Timeline of employer answers",
        "t_date": "Date", "t_company": "Company", "t_role": "Role", "t_event": "Answer", "t_after": "After",
        "days": "{d} d", "latest": "latest {n} of {total}",
        "apps": "Applications sent", "a_date": "Sent", "a_status": "Status now", "a_wait": "Waiting / answered",
        "waiting": "waiting {d} d", "answered_after": "answered after {d} d",
        "rates": "Response rates of these applications", "rates_note": "Small groups (under 5 applications) say little.",
        "r_source": "Source", "r_role": "Role family", "r_country": "Country", "r_applied": "Applied",
        "r_answered": "Answered", "r_interview": "Interview", "r_rejected": "Rejected",
        "none": "Nothing in these weeks.", "footer": "JobSearcher · job search report", "page": "Page {n}",
        "i_sent": "You sent {n} applications in {weeks} week(s) — {avg} per week on average.",
        "i_answered": "{a} employers answered so far ({pct}%): {r} rejections, {p} next steps or interviews.",
        "i_none_answered": "No employer has answered these applications yet — most answer within 2–4 weeks.",
        "i_median": "Typical time to an answer: {d} days.",
        "i_silent": "{n} applications are older than 4 weeks without any answer — most of these are silent rejections.",
        "i_best_week": "Most applications in {w} ({n}).",
        "i_source": "Best-answering source: {s} — {pct}% of {n} applications answered.",
        "i_role": "Most interviews per application: {r} ({i} of {n}).",
        "i_found": "The scout found {f} matching jobs in these weeks; {s} scored 8 or more.",
        "i_interviews": "Interview invitations in these weeks: {list}.",
        "st": {"APPLIED": "Applied", "APPLYING": "Applying", "HR_SCREEN": "Next step (HR)", "INTERVIEW": "Interview",
               "TECHNICAL": "Technical interview", "FINAL": "Final round", "OFFER": "Offer", "REJECTED": "Rejection",
               "NEW": "New", "REVIEW": "Saved", "PRIORITY": "Priority", "SKIPPED": "Skipped", "EXPIRED": "Expired"},
        "waiting_status": "Waiting",
    },
    "de": {
        "title": "Bewerbungsbericht", "all_weeks": "alle Wochen", "generated": "erstellt am",
        "week": "KW {w}", "weeks_range": "KW {a}–{b} {y}", "weeks_list": "KW {list}",
        "k_applied": "Bewerbungen gesendet", "k_answered": "Antworten erhalten", "k_interviews": "Gespräche & nächste Schritte",
        "k_rejections": "Absagen erhalten", "k_waiting": "Davon noch ohne Antwort", "k_packages": "Vorbereitete Pakete",
        "found_line": "In diesen Wochen hat der Scout {found} passende Stellen gefunden; {strong} davon mit Score 8 oder mehr.",
        "takeaways": "Das Wichtigste", "by_week": "Woche für Woche",
        "c_week": "KW", "c_dates": "Zeitraum", "c_found": "Gefunden", "c_strong": "8+", "c_applied": "Beworben",
        "c_confirm": "Eingang", "c_rejections": "Absagen", "c_next": "HR-Schritt", "c_interviews": "Gespräch",
        "c_offers": "Angebot", "c_packages": "Pakete", "total": "Summe",
        "week_note": "Antworten zählen in der Woche, in der sie kamen (auch für frühere Bewerbungen). Eingang = "
                     "Eingangsbestätigungen. Die aktuelle Woche läuft noch.",
        "chart": "Bewerbungen und Antworten pro Woche", "s_applied": "Bewerbungen", "s_rejections": "Absagen",
        "s_positive": "Nächste Schritte & Gespräche",
        "outcome": "Was aus den Bewerbungen dieser Wochen wurde",
        "outcome_note": "Jede Bewerbung zählt einmal, nach der bisher weitesten Antwort. Offen = noch keine Antwort.",
        "o_applied": "Gesendet", "o_answered": "Beantwortet", "o_next": "Nächster Schritt", "o_interview": "Gespräch",
        "o_offer": "Angebot", "o_rejected": "Absage", "o_waiting": "Noch keine Antwort",
        "of": "{n} von {total} ({pct} %)",
        "median": "Antwort im Median nach {d} Tagen (am schnellsten {lo}, am langsamsten {hi}).",
        "positive": "Gespräche und nächste Schritte", "timeline": "Zeitleiste der Antworten",
        "t_date": "Datum", "t_company": "Unternehmen", "t_role": "Stelle", "t_event": "Antwort", "t_after": "Nach",
        "days": "{d} T.", "latest": "die neuesten {n} von {total}",
        "apps": "Gesendete Bewerbungen", "a_date": "Gesendet", "a_status": "Stand", "a_wait": "Wartet / beantwortet",
        "waiting": "wartet {d} T.", "answered_after": "Antwort nach {d} T.",
        "rates": "Antwortquoten dieser Bewerbungen", "rates_note": "Kleine Gruppen (unter 5 Bewerbungen) sagen wenig aus.",
        "r_source": "Quelle", "r_role": "Berufsfeld", "r_country": "Land", "r_applied": "Beworben",
        "r_answered": "Beantwortet", "r_interview": "Gespräch", "r_rejected": "Absage",
        "none": "Nichts in diesen Wochen.", "footer": "JobSearcher · Bewerbungsbericht", "page": "Seite {n}",
        "i_sent": "Sie haben in {weeks} Woche(n) {n} Bewerbungen verschickt — im Schnitt {avg} pro Woche.",
        "i_answered": "{a} Arbeitgeber haben bisher geantwortet ({pct} %): {r} Absagen, {p} nächste Schritte oder Gespräche.",
        "i_none_answered": "Auf diese Bewerbungen kam noch keine Antwort — die meisten antworten nach 2–4 Wochen.",
        "i_median": "Typische Zeit bis zur Antwort: {d} Tage.",
        "i_silent": "{n} Bewerbungen sind älter als 4 Wochen und noch ohne Antwort — meist stille Absagen.",
        "i_best_week": "Die meisten Bewerbungen in {w} ({n}).",
        "i_source": "Quelle mit den meisten Antworten: {s} — {pct} % von {n} Bewerbungen beantwortet.",
        "i_role": "Die meisten Gespräche pro Bewerbung: {r} ({i} von {n}).",
        "i_found": "Der Scout hat in diesen Wochen {f} passende Stellen gefunden; {s} mit Score 8 oder mehr.",
        "i_interviews": "Einladungen zum Gespräch in diesen Wochen: {list}.",
        "st": {"APPLIED": "Beworben", "APPLYING": "In Bewerbung", "HR_SCREEN": "Nächster Schritt", "INTERVIEW": "Gespräch",
               "TECHNICAL": "Fachgespräch", "FINAL": "Finale Runde", "OFFER": "Angebot", "REJECTED": "Absage",
               "NEW": "Neu", "REVIEW": "Gemerkt", "PRIORITY": "Priorität", "SKIPPED": "Übersprungen", "EXPIRED": "Abgelaufen"},
        "waiting_status": "Offen",
    },
}
_MONTHS_EN = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

# ---------------------------------------------------------------------------------------------------- fonts

_FONT, _BOLD = "Helvetica", "Helvetica-Bold"


def _register_fonts() -> None:
    """DejaVu Sans (Docker image) or a Windows system font: full Unicode (umlauts, ≥, →); Helvetica covers cp1252 only."""
    global _FONT, _BOLD
    if _FONT != "Helvetica":
        return
    for regular, bold in (("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                          ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
                          ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf")):
        if Path(regular).is_file() and Path(bold).is_file():
            pdfmetrics.registerFont(TTFont("ReportSans", regular))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", bold))
            _FONT, _BOLD = "ReportSans", "ReportSans-Bold"
            return


def _safe(text: Any) -> str:
    s = str(text if text is not None else "")
    if _FONT == "Helvetica":  # cp1252 only
        s = s.replace("≥", ">=").replace("→", "->").replace("–", "-").replace("·", "-")
        s = s.encode("cp1252", "replace").decode("cp1252")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------------------------------- weeks

def week_key(dt: datetime | date) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def week_bounds(key: str) -> tuple[date, date]:
    y, w = key.split("-W")
    start = date.fromisocalendar(int(y), int(w), 1)
    return start, start + timedelta(days=6)


def _fmt_day(d: date | datetime, lang: str) -> str:
    return f"{d:%d.%m.%Y}" if lang == "de" else f"{d.day} {_MONTHS_EN[d.month - 1]} {d.year}"


def week_label(key: str, lang: str, with_dates: bool = False) -> str:
    start, end = week_bounds(key)
    label = T[lang]["week"].format(w=int(key.split("-W")[1]))
    if not with_dates:
        return label
    if lang == "de":
        return f"{label} · {start:%d.%m.}–{end:%d.%m.%Y}"
    return f"{label} · {start.day} {_MONTHS_EN[start.month - 1]} – {end.day} {_MONTHS_EN[end.month - 1]} {end.year}"


def _dates(key: str, lang: str) -> str:
    start, end = week_bounds(key)
    return f"{start:%d.%m.}–{end:%d.%m.}" if lang == "de" else f"{start.day} {_MONTHS_EN[start.month - 1]} – {end.day} {_MONTHS_EN[end.month - 1]}"


def period_label(keys: list[str], lang: str) -> str:
    """'Weeks 46–47 2026 (9 Nov – 22 Nov 2026)' for a run of weeks, otherwise the list of weeks."""
    t = T[lang]
    if not keys:
        return t["all_weeks"]
    nums = [int(k.split("-W")[1]) for k in keys]
    start, end = week_bounds(keys[0])[0], week_bounds(keys[-1])[1]
    span = f"{_fmt_day(start, lang)} – {_fmt_day(end, lang)}"
    contiguous = all(week_bounds(b)[0] - week_bounds(a)[0] == timedelta(days=7) for a, b in zip(keys, keys[1:]))
    if len(keys) == 1:
        return f"{week_label(keys[0], lang)} {start.year} ({span})"
    if contiguous:
        return f"{t['weeks_range'].format(a=nums[0], b=nums[-1], y=end.year)} ({span})"
    return t["weeks_list"].format(list=", ".join(str(n) for n in nums)) + f" {end.year}"


def available_weeks(s: Session, limit: int = 52) -> list[dict[str, Any]]:
    """Weeks with any activity (newest first) for the week picker: applications and jobs found per week."""
    applied = Counter(week_key(e) for e in s.scalars(select(StatusEvent.changed_at).where(StatusEvent.to_status == JobStatus.APPLIED)))
    found = Counter(week_key(d) for d in s.scalars(select(Job.first_seen).where(Job.track == "main")))
    keys = sorted(set(applied) | set(found) | {week_key(utcnow())}, reverse=True)[:limit]
    return [{"key": k, "applied": applied.get(k, 0), "found": found.get(k, 0), "current": k == week_key(utcnow())} for k in keys]


# ---------------------------------------------------------------------------------------------------- data

def _application(job: Job) -> dict[str, Any] | None:
    """When you applied and the employer's answers after that (None: never applied)."""
    events = sorted(job.status_events, key=lambda e: e.changed_at)
    applied_at = next((e.changed_at for e in events if e.to_status == JobStatus.APPLIED), None) or job.applied_at
    if applied_at is None:
        return None
    answers = [e for e in events if e.to_status in _ANSWERS and e.changed_at >= applied_at]
    statuses = {e.to_status for e in answers} | ({job.status} if job.status in _ANSWERS else set())
    return {"job": job, "applied_at": applied_at, "answers": answers, "first_answer": answers[0].changed_at if answers else None,
            "rejected": JobStatus.REJECTED in statuses, "interview": bool(statuses & set(_INTERVIEWS)) or JobStatus.OFFER in statuses,
            "next_step": JobStatus.HR_SCREEN in statuses, "offer": JobStatus.OFFER in statuses}


def report_data(s: Session, weeks: list[str] | None = None, lang: str = "en") -> dict[str, Any]:
    now = utcnow()
    jobs = list(s.scalars(select(Job).options(selectinload(Job.status_events), selectinload(Job.sources))))
    by_vacancy: dict[tuple, list] = defaultdict(list)
    for a in (a for j in jobs if (a := _application(j))):
        by_vacancy[(a["job"].normalized_company or a["job"].company, a["job"].normalized_title or a["job"].title)].append(a)
    apps = []
    for twins in by_vacancy.values():  # the copy with the most answers, applied when the first copy was
        best = max(twins, key=lambda a: (len(a["answers"]), -a["applied_at"].timestamp()))
        apps.append({**best, "applied_at": min(a["applied_at"] for a in twins)})
    keys = sorted(set(weeks or []) or {w["key"] for w in available_weeks(s) if w["applied"] or w["found"]})
    selected = set(keys)

    rows = {k: Counter() for k in keys}
    for j in jobs:
        k = week_key(j.first_seen)
        if k in selected and j.track == "main" and j.filter_status != FilterStatus.REJECTED:
            rows[k]["found"] += 1
            rows[k]["strong"] += (j.fit_score or 0) >= 8
    timeline = []
    for a in apps:
        if (k := week_key(a["applied_at"])) in selected:
            rows[k]["applied"] += 1
        seen = set()
        for e in a["answers"]:
            k = week_key(e.changed_at)
            group = ("rejections" if e.to_status == JobStatus.REJECTED else "offers" if e.to_status == JobStatus.OFFER
                     else "next" if e.to_status == JobStatus.HR_SCREEN else "interviews")
            if k not in selected or group in seen:
                continue
            seen.add(group)
            rows[k][group] += 1
            timeline.append({"at": e.changed_at, "company": a["job"].company, "title": a["job"].title, "status": e.to_status,
                             "after": (e.changed_at - a["applied_at"]).days})
    # answers that could not be tied to one of your jobs (e.g. you applied outside JobSearcher) count too
    kinds = {"rejection": ("rejections", JobStatus.REJECTED), "interview": ("interviews", JobStatus.INTERVIEW),
             "next_step": ("next", JobStatus.HR_SCREEN)}
    for em in s.scalars(select(EmailMessage).where(EmailMessage.job_id.is_(None), EmailMessage.reply_kind.in_(tuple(kinds)))):
        if em.received_at and (k := week_key(em.received_at)) in selected:
            group, status = kinds[em.reply_kind]
            rows[k][group] += 1
            timeline.append({"at": em.received_at, "company": (em.from_addr or "").partition("@")[2],
                             "title": em.subject or "", "status": status, "after": None})
    for p in s.scalars(select(ApplicationPackage).where(ApplicationPackage.status == "done")):
        if (k := week_key(p.created_at)) in selected:
            rows[k]["packages"] += 1
    for when in s.scalars(select(EmailMessage.received_at).where(EmailMessage.reply_kind == "received")):
        if when and (k := week_key(when)) in selected:
            rows[k]["confirm"] += 1
    total = sum(rows.values(), Counter())

    cohort = sorted((a for a in apps if week_key(a["applied_at"]) in selected), key=lambda a: a["applied_at"], reverse=True)
    answered = [a for a in cohort if a["answers"]]
    waits = sorted((a["first_answer"] - a["applied_at"]).days for a in answered)
    outcome = {"applied": len(cohort), "answered": len(answered), "rejected": sum(a["rejected"] for a in cohort),
               "next_step": sum(a["next_step"] for a in cohort), "interview": sum(a["interview"] for a in cohort),
               "offer": sum(a["offer"] for a in cohort), "waiting": len(cohort) - len(answered),
               "silent": sum(1 for a in cohort if not a["answers"] and (now - a["applied_at"]).days > 28),
               "median_days": round(statistics.median(waits)) if waits else None,
               "min_days": waits[0] if waits else None, "max_days": waits[-1] if waits else None}

    def breakdown(key) -> list[dict[str, Any]]:
        groups: dict[str, list] = defaultdict(list)
        for a in cohort:
            groups[key(a["job"]) or "—"].append(a)
        out = [{"key": k, "applied": len(v), "answered": sum(1 for a in v if a["answers"]),
                "interview": sum(a["interview"] for a in v), "rejected": sum(a["rejected"] for a in v)} for k, v in groups.items()]
        return sorted(out, key=lambda r: -r["applied"])

    data = {
        "generated": now, "lang": lang, "weeks": keys, "current_week": week_key(now),
        "rows": [(k, rows[k]) for k in keys], "total": total, "outcome": outcome,
        "timeline": sorted(timeline, key=lambda e: e["at"], reverse=True),
        "positive": sorted((e for e in timeline if e["status"] != JobStatus.REJECTED), key=lambda e: e["at"], reverse=True),
        "cohort": cohort,
        "by": {"source": breakdown(lambda j: source_label(j.source)), "role": breakdown(lambda j: j.role_family),
               "country": breakdown(lambda j: j.country)},
    }
    data["insights"] = insights(data)
    return data


def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / total)}" if total else "0"


def insights(d: dict[str, Any]) -> list[str]:
    """Plain-language takeaways for the chosen weeks (only those the numbers support)."""
    t, o, lang = T[d["lang"]], d["outcome"], d["lang"]
    out: list[str] = []
    n_weeks = max(1, len(d["weeks"]))
    if o["applied"]:
        avg = f"{o['applied'] / n_weeks:.1f}".replace(".", "," if lang == "de" else ".")
        out.append(t["i_sent"].format(n=o["applied"], weeks=n_weeks, avg=avg))
        if o["answered"]:
            out.append(t["i_answered"].format(a=o["answered"], pct=_pct(o["answered"], o["applied"]), r=o["rejected"],
                                              p=o["next_step"] + o["interview"]))
        else:
            out.append(t["i_none_answered"])
        if o["median_days"] is not None:
            out.append(t["i_median"].format(d=o["median_days"]))
        if o["silent"]:
            out.append(t["i_silent"].format(n=o["silent"]))
    busiest = max(d["rows"], key=lambda r: r[1]["applied"], default=None)
    if busiest and busiest[1]["applied"] and len(d["rows"]) > 1:
        out.append(t["i_best_week"].format(w=week_label(busiest[0], lang, with_dates=True), n=busiest[1]["applied"]))
    sources = [r for r in d["by"]["source"] if r["applied"] >= 5 and r["answered"]]
    if sources:
        best = max(sources, key=lambda r: (r["answered"] / r["applied"], r["applied"]))
        out.append(t["i_source"].format(s=best["key"], pct=_pct(best["answered"], best["applied"]), n=best["applied"]))
    roles = [r for r in d["by"]["role"] if r["applied"] >= 5 and r["interview"] >= 2]
    if roles:
        best = max(roles, key=lambda r: r["interview"] / r["applied"])
        out.append(t["i_role"].format(r=best["key"], i=best["interview"], n=best["applied"]))
    invites = [e for e in d["positive"] if e["status"] in _INTERVIEWS or e["status"] == JobStatus.OFFER]
    if invites:
        out.append(t["i_interviews"].format(list=", ".join(dict.fromkeys(e["company"] for e in invites[:8]))))
    if d["total"]["found"]:
        out.append(t["i_found"].format(f=d["total"]["found"], s=d["total"]["strong"]))
    return out


# ---------------------------------------------------------------------------------------------------- charts

def _nice_step(top: float) -> int:
    """A round axis step (1, 2, 5 × 10ⁿ) giving about four gridlines."""
    raw = max(1.0, top / 4)
    magnitude = 10 ** (len(str(int(raw))) - 1)
    return next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)


def _week_chart(d: dict[str, Any], width: float, height: float = 170) -> Drawing:
    t, lang = T[d["lang"]], d["lang"]
    drawing = Drawing(width, height)
    rows = d["rows"][-16:]  # the newest 16 weeks fit the page
    series = [([r["applied"] for _, r in rows], ACCENT, t["s_applied"]),
              ([r["rejections"] for _, r in rows], ROSE, t["s_rejections"]),
              ([r["next"] + r["interviews"] + r["offers"] for _, r in rows], TEAL, t["s_positive"])]
    top = max([max(v) for v, _, _ in series if v] or [0])
    if not rows or not top:
        drawing.add(String(width / 2, height / 2, t["none"], fontName=_FONT, fontSize=9, fillColor=MUTED, textAnchor="middle"))
        return drawing
    c = VerticalBarChart()
    c.x, c.y, c.width, c.height = 34, 34, width - 44, height - 58
    c.data = [v for v, _, _ in series]
    c.categoryAxis.categoryNames = [week_label(k, lang) for k, _ in rows]
    for i, (_, col, _) in enumerate(series):
        c.bars[i].fillColor, c.bars[i].strokeColor = col, None
    c.groupSpacing, c.barSpacing = 8, 1
    step = _nice_step(top)
    c.valueAxis.valueMin, c.valueAxis.valueStep = 0, step
    c.valueAxis.valueMax = step * (int(top // step) + 1)
    for axis in (c.categoryAxis, c.valueAxis):
        axis.labels.fontName, axis.labels.fontSize, axis.labels.fillColor = _FONT, 7, MUTED
        axis.strokeColor = LINE
    c.valueAxis.gridStrokeColor, c.valueAxis.visibleGrid = LINE, True
    c.barLabelFormat = lambda v: str(int(v)) if v else ""
    c.barLabels.fontName, c.barLabels.fontSize, c.barLabels.fillColor, c.barLabels.nudge = _FONT, 6, INK, 5
    drawing.add(c)
    x = 34
    for _, col, label in series:  # legend
        drawing.add(Rect(x, height - 12, 8, 8, fillColor=col, strokeColor=None))
        drawing.add(String(x + 12, height - 11, label, fontName=_FONT, fontSize=7.5, fillColor=INK))
        x += 24 + pdfmetrics.stringWidth(label, _FONT, 7.5)
    return drawing


def _outcome_bars(d: dict[str, Any], width: float) -> Drawing:
    t, o = T[d["lang"]], d["outcome"]
    steps = [(t["o_applied"], o["applied"], NAVY), (t["o_answered"], o["answered"], ACCENT),
             (t["o_next"], o["next_step"], TEAL), (t["o_interview"], o["interview"], TEAL), (t["o_offer"], o["offer"], AMBER),
             (t["o_rejected"], o["rejected"], ROSE), (t["o_waiting"], o["waiting"], MUTED)]
    label_w, bar_w, row = 110, width - 260, 19
    drawing = Drawing(width, row * len(steps) + 4)
    top = max(1, o["applied"])
    for i, (label, n, col) in enumerate(steps):
        y = drawing.height - (i + 1) * row
        w = bar_w * n / top
        drawing.add(Rect(label_w, y + 2, bar_w, 13, fillColor=TINT, strokeColor=None))
        if n:
            drawing.add(Rect(label_w, y + 2, max(2, w), 13, fillColor=col, strokeColor=None))
        drawing.add(String(label_w - 6, y + 5, label, fontName=_FONT, fontSize=8.5, fillColor=INK, textAnchor="end"))
        text = str(n) if i == 0 else t["of"].format(n=n, total=o["applied"], pct=_pct(n, o["applied"]))
        drawing.add(String(label_w + bar_w + 8, y + 5, text, fontName=_BOLD, fontSize=8.5, fillColor=INK))
    return drawing


# ---------------------------------------------------------------------------------------------------- layout

def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("t", fontName=_BOLD, fontSize=22, leading=26, textColor=colors.white),
        "subtitle": ParagraphStyle("st", fontName=_FONT, fontSize=10, leading=14, textColor=colors.HexColor("#DCE6F7")),
        "h2": ParagraphStyle("h2", fontName=_BOLD, fontSize=13, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("b", fontName=_FONT, fontSize=9, leading=13, textColor=INK, alignment=TA_LEFT),
        "small": ParagraphStyle("s", fontName=_FONT, fontSize=7.5, leading=10, textColor=MUTED, spaceAfter=4),
        "kpi_value": ParagraphStyle("kv", fontName=_BOLD, fontSize=17, leading=20, textColor=NAVY),
        "kpi_label": ParagraphStyle("kl", fontName=_FONT, fontSize=7.5, leading=9, textColor=MUTED),
        "cell": ParagraphStyle("c", fontName=_FONT, fontSize=7.8, leading=9.6, textColor=INK),
        "cellb": ParagraphStyle("cb", fontName=_BOLD, fontSize=7.8, leading=9.6, textColor=INK),
        "insight": ParagraphStyle("i", fontName=_FONT, fontSize=9.5, leading=14, textColor=INK, leftIndent=10, bulletIndent=0),
    }


def _header_band(st, d: dict[str, Any], name: str, width: float) -> Table:
    t = T[d["lang"]]
    sub = f"{name} · {period_label(d['weeks'], d['lang'])} · {t['generated']} {_fmt_day(d['generated'], d['lang'])}"
    table = Table([[Paragraph(t["title"], st["title"])], [Paragraph(_safe(sub), st["subtitle"])]], colWidths=[width])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 14),
                               ("TOPPADDING", (0, 0), (0, 0), 14), ("BOTTOMPADDING", (0, -1), (-1, -1), 14)]))
    return table


def _kpis(st, items: list[tuple[str, str]], width: float) -> Table:
    col = width / len(items)
    cells = [Table([[Paragraph(_safe(v), st["kpi_value"])], [Paragraph(_safe(k), st["kpi_label"])]], colWidths=[col - 8])
             for k, v in items]
    table = Table([cells], colWidths=[col] * len(items))
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                               ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def _table(st, header: list[str], rows: list[list[Any]], widths: list[float], total_row: bool = False,
           numeric_from: int = 99, none: str = "—") -> Table:
    num = ParagraphStyle("n", parent=st["cell"], alignment=2)
    numb = ParagraphStyle("nb", parent=st["cellb"], alignment=2)
    data = [[Paragraph(_safe(h), numb if i >= numeric_from else st["cellb"]) for i, h in enumerate(header)]]
    for r_i, r in enumerate(rows):
        bold = total_row and r_i == len(rows) - 1
        data.append([Paragraph(_safe(c), (numb if bold else num) if i >= numeric_from else (st["cellb"] if bold else st["cell"]))
                     for i, c in enumerate(r)])
    if not rows:
        data.append([Paragraph(_safe(none), st["cell"])] + [""] * (len(header) - 1))
    table = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if total_row and rows:
        style += [("LINEABOVE", (0, -1), (-1, -1), 0.8, NAVY), ("BACKGROUND", (0, -1), (-1, -1), TINT)]
    table.setStyle(TableStyle(style))
    return table


def _section(st, title: str, *flowables) -> KeepTogether:
    return KeepTogether([Paragraph(_safe(title), st["h2"]), *flowables])


def _footer_for(lang: str):
    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(_FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 10 * mm, T[lang]["footer"])
        canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, T[lang]["page"].format(n=doc.page))
        canvas.restoreState()
    return footer


def build_report(s: Session, weeks: list[str] | None = None, lang: str = "en", name: str = "") -> bytes:
    _register_fonts()
    lang = lang if lang in T else "en"
    t = T[lang]
    d = report_data(s, weeks, lang)
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=18 * mm, title=t["title"], author=name or "JobSearcher")
    W = doc.width
    o, tot = d["outcome"], d["total"]
    story: list = [_header_band(st, d, name or "JobSearcher", W), Spacer(1, 10)]
    answers = tot["rejections"] + tot["next"] + tot["interviews"] + tot["offers"]
    story.append(_kpis(st, [(t["k_applied"], str(tot["applied"])), (t["k_answered"], str(answers)),
                            (t["k_interviews"], str(tot["next"] + tot["interviews"] + tot["offers"])),
                            (t["k_rejections"], str(tot["rejections"])), (t["k_waiting"], str(o["waiting"])),
                            (t["k_packages"], str(tot["packages"]))], W))
    story.append(Spacer(1, 5))
    story.append(Paragraph(_safe(t["found_line"].format(found=tot["found"], strong=tot["strong"])), st["small"]))

    if d["insights"]:
        story.append(Paragraph(t["takeaways"], st["h2"]))
        story += [Paragraph(_safe(x), st["insight"], bulletText="•") for x in d["insights"]]

    # week by week: the numbers behind every chart
    rows = []
    for k, r in d["rows"]:
        label = week_label(k, lang) + (" *" if k == d["current_week"] else "")
        rows.append([label, _dates(k, lang), r["found"], r["strong"], r["applied"], r["confirm"], r["rejections"],
                     r["next"], r["interviews"], r["offers"], r["packages"]])
    rows.append([t["total"], "", tot["found"], tot["strong"], tot["applied"], tot["confirm"], tot["rejections"],
                 tot["next"], tot["interviews"], tot["offers"], tot["packages"]])
    cols = [t["c_week"], t["c_dates"], t["c_found"], t["c_strong"], t["c_applied"], t["c_confirm"], t["c_rejections"],
            t["c_next"], t["c_interviews"], t["c_offers"], t["c_packages"]]
    widths = [W * 0.09, W * 0.13] + [W * 0.0867] * 9
    story.append(Paragraph(t["by_week"], st["h2"]))
    story.append(_table(st, cols, rows, widths, total_row=True, numeric_from=2))
    story.append(Paragraph(_safe(t["week_note"]), st["small"]))
    story.append(_section(st, t["chart"], _week_chart(d, W)))

    # outcome of the applications sent in these weeks
    outcome = [Paragraph(_safe(t["outcome_note"]), st["small"]), _outcome_bars(d, W)]
    if o["median_days"] is not None:
        outcome.append(Paragraph(_safe(t["median"].format(d=o["median_days"], lo=o["min_days"], hi=o["max_days"])), st["body"]))
    story.append(_section(st, t["outcome"], *outcome))

    def event_rows(events, limit):
        return [[_fmt_day(e["at"], lang), e["company"], e["title"], t["st"].get(e["status"], e["status"]),
                 t["days"].format(d=e["after"]) if e["after"] is not None else "–"] for e in events[:limit]]

    ev_widths = [W * 0.13, W * 0.24, W * 0.39, W * 0.14, W * 0.10]
    ev_head = [t["t_date"], t["t_company"], t["t_role"], t["t_event"], t["t_after"]]
    if d["positive"]:
        story.append(_section(st, t["positive"], _table(st, ev_head, event_rows(d["positive"], 40), ev_widths, numeric_from=4)))
    n_tl = len(d["timeline"])
    title = t["timeline"] + (f" ({t['latest'].format(n=120, total=n_tl)})" if n_tl > 120 else f" ({n_tl})")
    story.append(Paragraph(_safe(title), st["h2"]))
    story.append(_table(st, ev_head, event_rows(d["timeline"], 120), ev_widths, numeric_from=4, none=t["none"]))

    # response rates of the applications sent in these weeks
    story.append(Paragraph(t["rates"], st["h2"]))
    story.append(Paragraph(_safe(t["rates_note"]), st["small"]))
    for key, head in (("source", t["r_source"]), ("role", t["r_role"]), ("country", t["r_country"])):
        rws = [[r["key"], r["applied"], f"{r['answered']} ({_pct(r['answered'], r['applied'])}%)", r["interview"],
                r["rejected"]] for r in d["by"][key][:8]]
        story.append(KeepTogether([Spacer(1, 4), _table(st, [head, t["r_applied"], t["r_answered"], t["r_interview"], t["r_rejected"]],
                                                       rws, [W * 0.40, W * 0.13, W * 0.19, W * 0.14, W * 0.14], numeric_from=1,
                                                       none=t["none"])]))

    # the applications themselves
    cohort = d["cohort"]
    title = t["apps"] + (f" ({t['latest'].format(n=150, total=len(cohort))})" if len(cohort) > 150 else f" ({len(cohort)})")
    rows = []
    for a in cohort[:150]:
        job = a["job"]
        status = t["waiting_status"] if job.status in (JobStatus.APPLIED, JobStatus.APPLYING) else t["st"].get(job.status, job.status)
        wait = (t["answered_after"].format(d=(a["first_answer"] - a["applied_at"]).days) if a["first_answer"]
                else t["waiting"].format(d=(d["generated"] - a["applied_at"]).days))
        rows.append([_fmt_day(a["applied_at"], lang), job.company, job.title, status, wait])
    story.append(Paragraph(_safe(title), st["h2"]))
    story.append(_table(st, [t["a_date"], t["t_company"], t["t_role"], t["a_status"], t["a_wait"]], rows,
                        [W * 0.13, W * 0.22, W * 0.37, W * 0.12, W * 0.16], none=t["none"]))
    doc.build(story, onFirstPage=_footer_for(lang), onLaterPages=_footer_for(lang))
    return buf.getvalue()
