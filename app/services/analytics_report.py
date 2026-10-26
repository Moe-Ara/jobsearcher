"""Downloadable analytics report (PDF): your application funnel, what the pipeline found, packages and takeaways.

Built with reportlab (platypus + reportlab.graphics charts) — no extra dependencies. `build_report(days)` returns the
PDF bytes; days=None covers everything, otherwise the last N days (funnel rates always use every application you
ever made, so small periods don't produce misleading percentages).
"""

from __future__ import annotations

import io
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.enums import FUNNEL, FilterStatus, JobStatus
from app.models import ApplicationPackage, Job, JobSource, TargetCompany, utcnow
from app.services import analytics
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
PALETTE = [ACCENT, TEAL, AMBER, ROSE, colors.HexColor("#7C5CE0"), colors.HexColor("#0EA5E9"),
           colors.HexColor("#84CC16"), colors.HexColor("#F97316"), MUTED]

# ---------------------------------------------------------------------------------------------------- fonts

_FONT, _BOLD = "Helvetica", "Helvetica-Bold"


def _register_fonts() -> None:
    """DejaVu Sans (Docker image) or a Windows system font: full Unicode (≥, →, Arabic company names' Latin parts);
    the built-in Helvetica only covers cp1252."""
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
        s = s.replace("≥", ">=").replace("≤", "<=").replace("→", "->").replace("–", "-").replace("·", "-")
        s = s.encode("cp1252", "replace").decode("cp1252")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------------------------------- data

def report_data(s: Session, days: int | None) -> dict[str, Any]:
    now = utcnow()
    since = now - timedelta(days=days) if days else None
    base = analytics.summary(s)
    in_period = (lambda dt: dt is not None and dt >= since) if since else (lambda dt: dt is not None)
    jobs = list(s.scalars(select(Job).options(selectinload(Job.status_events))))
    main = [j for j in jobs if j.track == "main"]
    found = [j for j in main if in_period(j.first_seen)]

    # discovered jobs per ISO week (main track) and where they come from
    weeks: dict[str, int] = defaultdict(int)
    for j in found:
        y, w, _ = j.first_seen.isocalendar()
        weeks[f"{y}-W{w:02d}"] += 1
    week_keys = sorted(weeks)[-12:]

    passed = [j for j in found if j.filter_status != FilterStatus.REJECTED]
    scores = [j.fit_score for j in passed if j.fit_score is not None]
    score_bins = Counter("8+" if x >= 8 else "6.5–8" if x >= 6.5 else "5–6.5" if x >= 5 else "<5" for x in scores)

    sources = Counter()
    for src, n in s.execute(select(JobSource.source, func.count()).join(Job, JobSource.job_id == Job.id)
                            .where(Job.track == "main", *( [Job.first_seen >= since] if since else []))
                            .group_by(JobSource.source)).all():
        sources[src] += n

    applied_jobs = []
    for j in jobs:
        stage = analytics._furthest_stage(j)
        if stage < 0:
            continue
        applied_at = j.applied_at or next((e.changed_at for e in j.status_events if e.to_status == JobStatus.APPLIED), None)
        applied_jobs.append({"title": j.title, "company": j.company, "country": j.country or "—",
                             "stage": FUNNEL[stage].replace("_", " ").title(), "status": j.status,
                             "applied_at": applied_at, "track": j.track,
                             "days": (now - applied_at).days if applied_at else None})
    applied_jobs.sort(key=lambda r: r["applied_at"] or now, reverse=True)
    applied_period = [r for r in applied_jobs if in_period(r["applied_at"])]

    pkgs = list(s.scalars(select(ApplicationPackage).where(ApplicationPackage.status == "done")))
    pkgs_period = [p for p in pkgs if in_period(p.created_at)]
    sent = [p for p in pkgs_period if (p.meta or {}).get("submission", {}).get("channel") == "email"]
    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731

    companies = s.scalar(select(func.count(TargetCompany.id)).where(TargetCompany.active.is_(True))) or 0
    side_visible = sum(1 for j in jobs if j.track == "side" and j.active and j.filter_status != FilterStatus.REJECTED)

    data = {
        "generated": now, "days": days, "since": since,
        "overall": base["overall"], "by": base["by"], "status_counts": base["status_counts"],
        "applied_per_week": base["per_week"],
        "found_total": len(found), "found_passed": len(passed),
        "found_per_week": [(k, weeks[k]) for k in week_keys],
        "countries": Counter((j.country or "Unknown") for j in passed).most_common(8),
        "sources": sources.most_common(10),
        "score_bins": [(b, score_bins.get(b, 0)) for b in ("8+", "6.5–8", "5–6.5", "<5")],
        "priority": sum(1 for j in passed if (j.fit_score or 0) >= 8),
        "applied": applied_jobs, "applied_period": applied_period,
        "packages": {"total": len(pkgs_period), "send": sum(p.verdict == "SEND" for p in pkgs_period),
                     "revise": sum(p.verdict != "SEND" for p in pkgs_period), "sent": len(sent),
                     "ats": avg([p.ats_score for p in pkgs_period if p.ats_score is not None]),
                     "recruiter": avg([p.recruiter_score for p in pkgs_period if p.recruiter_score is not None])},
        "companies": companies, "side_jobs": side_visible,
    }
    data["insights"] = insights(data)
    return data


def insights(d: dict[str, Any]) -> list[str]:
    """Plain-language takeaways (only those the numbers actually support)."""
    out: list[str] = []
    o = d["overall"]
    if o["applied"]:
        out.append(f"You have applied to {o['applied']} jobs so far; {o['responses']} answered "
                   f"({o['response_rate']}%), {o['interviews']} led to an interview ({o['interview_rate']}%).")
        if o["applied"] >= 10 and o["response_rate"] < 10:
            out.append("Fewer than one in ten applications gets an answer yet — most employers reply within 2–4 weeks, "
                       "so keep the status of each job up to date to see the real rate.")
    weeks = [n for _, n in d["applied_per_week"]]
    if len(weeks) >= 2:
        last, before = weeks[-1], (sum(weeks[:-1]) / len(weeks[:-1]))
        trend = "more" if last > before else "fewer"
        out.append(f"Last week you applied to {last} jobs — {trend} than your weekly average of {before:.1f}.")
    rows = [r for r in d["by"].get("Source", []) if r["applied"] >= 3]
    if rows:
        best = max(rows, key=lambda r: (r["response_rate"], r["applied"]))
        if best["response_rate"] > 0:
            out.append(f"Best-answering source: {best['key']} ({best['response_rate']}% responses from {best['applied']} applications).")
    by_score = {r["key"]: r for r in d["by"].get("Fit score", [])}
    hi, lo = by_score.get("8+"), by_score.get("<6.5")
    if hi and lo and hi["applied"] >= 3 and lo["applied"] >= 3:
        out.append(f"High-fit jobs (score 8+) answer {hi['response_rate']}% of the time vs {lo['response_rate']}% for "
                   "low-fit ones — the ranking is worth trusting." if hi["response_rate"] >= lo["response_rate"] else
                   f"Low-fit jobs answered more often ({lo['response_rate']}%) than high-fit ones ({hi['response_rate']}%) — "
                   "worth labelling a few jobs on the Calibrate page.")
    if d["found_total"]:
        share = round(100 * d["priority"] / max(1, d["found_passed"]), 1)
        out.append(f"The scout found {d['found_total']} jobs{' in this period' if d['days'] else ''}; "
                   f"{d['found_passed']} passed your filters and {d['priority']} ({share}%) scored 8 or higher.")
    gulf = sum(n for c, n in d["countries"] if c in ("UAE", "Saudi Arabia"))
    if gulf:
        out.append(f"{gulf} of the matching jobs are in the UAE or Saudi Arabia.")
    p = d["packages"]
    if p["total"]:
        out.append(f"{p['total']} application packages were prepared ({p['send']} ready to send); average ATS score "
                   f"{p['ats']}/100, recruiter score {p['recruiter']}/10.")
    return out


# ---------------------------------------------------------------------------------------------------- charts

def _bar_chart(pairs: list[tuple[str, int]], width: float, height: float, color=ACCENT, every: int = 1) -> Drawing:
    d = Drawing(width, height)
    if not pairs:
        d.add(String(width / 2, height / 2, "No data yet", fontName=_FONT, fontSize=9, fillColor=MUTED, textAnchor="middle"))
        return d
    c = VerticalBarChart()
    c.x, c.y, c.width, c.height = 28, 22, width - 40, height - 34
    c.data = [[n for _, n in pairs]]
    c.categoryAxis.categoryNames = [k.split("-W")[-1] and ("W" + k.split("-W")[-1] if "-W" in k else k) if i % every == 0 else ""
                                    for i, (k, _) in enumerate(pairs)]
    c.bars[0].fillColor = color
    c.bars[0].strokeColor = None
    c.barWidth = 6
    c.groupSpacing = 6
    c.valueAxis.valueMin = 0
    top = max(n for _, n in pairs)
    c.valueAxis.valueMax = max(4, int(top * 1.15) + 1)
    c.valueAxis.valueStep = max(1, c.valueAxis.valueMax // 4)
    for axis in (c.categoryAxis, c.valueAxis):
        axis.labels.fontName, axis.labels.fontSize, axis.labels.fillColor = _FONT, 7, MUTED
        axis.strokeColor = LINE
    c.valueAxis.gridStrokeColor, c.valueAxis.visibleGrid = LINE, True
    c.barLabelFormat = "%d"
    c.barLabels.fontName, c.barLabels.fontSize, c.barLabels.fillColor, c.barLabels.nudge = _FONT, 6.5, INK, 6
    d.add(c)
    return d


def _hbar_chart(pairs: list[tuple[str, int]], width: float, color=TEAL) -> Drawing:
    rows = max(1, len(pairs))
    height = 16 * rows + 20
    d = Drawing(width, height)
    if not pairs:
        d.add(String(width / 2, height / 2, "No data yet", fontName=_FONT, fontSize=9, fillColor=MUTED, textAnchor="middle"))
        return d
    c = HorizontalBarChart()
    c.x, c.y, c.width, c.height = 110, 8, width - 140, height - 16
    c.data = [[n for _, n in reversed(pairs)]]
    c.categoryAxis.categoryNames = [k[:22] for k, _ in reversed(pairs)]
    c.bars[0].fillColor, c.bars[0].strokeColor = color, None
    c.valueAxis.valueMin = 0
    c.valueAxis.visible = False
    c.categoryAxis.labels.fontName, c.categoryAxis.labels.fontSize, c.categoryAxis.labels.fillColor = _FONT, 7.5, INK
    c.categoryAxis.strokeColor = LINE
    c.barLabelFormat = "%d"
    c.barLabels.fontName, c.barLabels.fontSize, c.barLabels.fillColor = _FONT, 7, INK
    c.barLabels.boxAnchor, c.barLabels.dx = "w", 4
    d.add(c)
    return d


def _donut(pairs: list[tuple[str, int]], size: float, legend_width: float, pretty: bool = False) -> Drawing:
    pairs = [(k, n) for k, n in pairs if n]
    d = Drawing(size + legend_width, size)
    if not pairs:
        d.add(String(size / 2, size / 2, "No data yet", fontName=_FONT, fontSize=9, fillColor=MUTED, textAnchor="middle"))
        return d
    p = Pie()
    p.x, p.y, p.width, p.height = 6, 6, size - 12, size - 12
    p.data = [n for _, n in pairs]
    p.innerRadiusFraction = 0.58
    p.slices.strokeColor, p.slices.strokeWidth = colors.white, 1.2
    for i in range(len(pairs)):
        p.slices[i].fillColor = PALETTE[i % len(PALETTE)]
    d.add(p)
    total = sum(n for _, n in pairs)
    d.add(String(size / 2, size / 2 - 4, str(total), fontName=_BOLD, fontSize=14, fillColor=INK, textAnchor="middle"))
    y = size - 14
    for i, (k, n) in enumerate(pairs[:9]):
        d.add(Rect(size + 6, y, 8, 8, fillColor=PALETTE[i % len(PALETTE)], strokeColor=None))
        label = k.replace("_", " ").title() if pretty else k  # statuses: "HR_SCREEN" → "Hr Screen"; countries as they are
        d.add(String(size + 18, y + 1, f"{label}  {n}", fontName=_FONT, fontSize=7.5, fillColor=INK))
        y -= 13
    return d


def _funnel(o: dict[str, Any], width: float) -> Drawing:
    steps = [("Applied", o["applied"], ACCENT), ("Response", o["responses"], TEAL),
             ("Interview", o["interviews"], AMBER), ("Offer", o["offers"], ROSE)]
    height = 4 * 26 + 4
    d = Drawing(width, height)
    top = max(1, o["applied"])
    for i, (label, n, col) in enumerate(steps):
        y = height - (i + 1) * 26
        w = max(4, (width - 150) * n / top)
        d.add(Rect(70, y, w, 20, fillColor=col, strokeColor=None, rx=3, ry=3))
        d.add(String(64, y + 6, label, fontName=_FONT, fontSize=8.5, fillColor=INK, textAnchor="end"))
        pct = f"  ({round(100 * n / top)}%)" if i and o["applied"] else ""
        d.add(String(76 + w, y + 6, f"{n}{pct}", fontName=_BOLD, fontSize=8.5, fillColor=INK))
    return d


# ---------------------------------------------------------------------------------------------------- layout

def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("t", fontName=_BOLD, fontSize=22, leading=26, textColor=colors.white),
        "subtitle": ParagraphStyle("st", fontName=_FONT, fontSize=10, leading=14, textColor=colors.HexColor("#DCE6F7")),
        "h2": ParagraphStyle("h2", fontName=_BOLD, fontSize=13, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("b", fontName=_FONT, fontSize=9, leading=13, textColor=INK, alignment=TA_LEFT),
        "small": ParagraphStyle("s", fontName=_FONT, fontSize=7.5, leading=10, textColor=MUTED),
        "kpi_value": ParagraphStyle("kv", fontName=_BOLD, fontSize=18, leading=21, textColor=NAVY),
        "kpi_label": ParagraphStyle("kl", fontName=_FONT, fontSize=7.5, leading=9, textColor=MUTED),
        "cell": ParagraphStyle("c", fontName=_FONT, fontSize=8, leading=10, textColor=INK),
        "cellb": ParagraphStyle("cb", fontName=_BOLD, fontSize=8, leading=10, textColor=INK),
        "insight": ParagraphStyle("i", fontName=_FONT, fontSize=9.5, leading=14, textColor=INK, leftIndent=10, bulletIndent=0),
    }


def _header_band(st, d: dict[str, Any], name: str, width: float) -> Table:
    period = f"Last {d['days']} days" if d["days"] else "All time"
    t = Table([[Paragraph("Job Search Report", st["title"])],
               [Paragraph(_safe(f"{name} · {period} · generated {d['generated'].strftime('%d %B %Y')}"), st["subtitle"])]],
              colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 14),
                           ("TOPPADDING", (0, 0), (0, 0), 14), ("BOTTOMPADDING", (0, -1), (-1, -1), 14)]))
    return t


def _kpis(st, items: list[tuple[str, str]], width: float) -> Table:
    cells = [[Paragraph(_safe(v), st["kpi_value"]), Paragraph(_safe(k), st["kpi_label"])] for k, v in items]
    col = width / len(items)
    t = Table([[Table([[c[0]], [c[1]]], colWidths=[col - 8]) for c in cells]], colWidths=[col] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                           ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _table(st, header: list[str], rows: list[list[Any]], widths: list[float]) -> Table:
    data = [[Paragraph(_safe(h), st["cellb"]) for h in header]]
    data += [[Paragraph(_safe(c), st["cell"]) for c in r] for r in rows] or [[Paragraph("No data yet", st["cell"])] + [""] * (len(header) - 1)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
                           ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def _card(st, title: str, flowable, width: float) -> KeepTogether:
    return KeepTogether([Paragraph(_safe(title), st["h2"]), flowable])


def _cell(st, title: str, flowable) -> list:
    """Title + chart for a table cell (a KeepTogether inside a cell would measure as infinitely tall)."""
    return [Paragraph(_safe(title), st["h2"]), flowable]


def _side_by_side(left, right, width: float) -> Table:
    t = Table([[left, right]], colWidths=[width / 2, width / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(_FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 10 * mm, "JobSearcher · job search report")
    canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_report(s: Session, days: int | None, name: str = "") -> bytes:
    _register_fonts()
    d = report_data(s, days)
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=18 * mm, title="Job Search Report", author=name or "JobSearcher")
    W = doc.width
    o, p = d["overall"], d["packages"]
    story: list = [_header_band(st, d, name or "Your job search", W), Spacer(1, 10)]
    story.append(_kpis(st, [("Applications", str(o["applied"])), ("Response rate", f"{o['response_rate']}%"),
                            ("Interview rate", f"{o['interview_rate']}%"), ("Offers", str(o["offers"])),
                            ("Packages prepared", str(p["total"])), ("Jobs found", str(d["found_total"]))], W))
    story.append(Spacer(1, 6))
    story.append(Paragraph(_safe(f"Watching {d['companies']} employers and boards · {d['found_passed']} jobs passed your "
                                 f"filters · {d['priority']} scored 8+ · {d['side_jobs']} side jobs within reach"), st["small"]))

    if d["insights"]:
        story.append(Paragraph("Key takeaways", st["h2"]))
        story += [Paragraph(_safe(x), st["insight"], bulletText="•") for x in d["insights"]]

    half = W / 2 - 6
    story.append(_side_by_side(
        _cell(st, "Application funnel", _funnel(o, half)),
        _cell(st, "Applications per week", _bar_chart(d["applied_per_week"], half, 130, ACCENT)), W))
    story.append(_side_by_side(
        _cell(st, "Jobs discovered per week", _bar_chart(d["found_per_week"], half, 130, TEAL)),
        _cell(st, "Where your jobs are (passed filters)", _donut(d["countries"], 120, half - 130)), W))
    story.append(_side_by_side(
        _cell(st, "Fit-score distribution", _bar_chart(d["score_bins"], half, 120, AMBER)),
        _cell(st, "Pipeline status (all jobs)", _donut(sorted(d["status_counts"].items(), key=lambda kv: -kv[1])[:8],
                                                       120, half - 130, pretty=True)), W))
    story.append(_card(st, "Top sources of jobs", _hbar_chart([(source_label(k), n) for k, n in d["sources"]], W), W))

    story.append(PageBreak())
    story.append(Paragraph("What works — response rates", st["h2"]))
    story.append(Paragraph("A response is reaching HR screen or later, or a rejection after applying. Rates use every "
                           "application you have made.", st["small"]))
    for title in ("Source", "Country", "Fit score", "Role family"):
        rows = [[source_label(r["key"]) if title == "Source" else r["key"], r["applied"], f"{r['response_rate']}%",
                 f"{r['interview_rate']}%", r["offers"]]
                for r in d["by"].get(title, [])[:8]]
        story.append(_card(st, f"By {title.lower()}", _table(st, [title, "Applied", "Response", "Interview", "Offers"], rows,
                                                             [W * 0.4, W * 0.15, W * 0.15, W * 0.15, W * 0.15]), W))

    story.append(Paragraph("Application packages", st["h2"]))
    story.append(_kpis(st, [("Prepared", str(p["total"])), ("Ready to send", str(p["send"])), ("Need review", str(p["revise"])),
                            ("Sent by JobSearcher", str(p["sent"])), ("Avg ATS score", f"{p['ats'] or '–'}"),
                            ("Avg recruiter score", f"{p['recruiter'] or '–'}")], W))

    rows = [[(r["applied_at"].strftime("%d.%m.%y") if r["applied_at"] else "–"), r["company"], r["title"], r["country"],
             r["stage"], (f"{r['days']} d" if r["days"] is not None else "–")] for r in d["applied"][:40]]
    story.append(_card(st, f"Your applications ({len(d['applied'])}{', latest 40' if len(d['applied']) > 40 else ''})",
                       _table(st, ["Applied", "Company", "Role", "Country", "Furthest stage", "Since"], rows,
                              [W * 0.11, W * 0.22, W * 0.35, W * 0.11, W * 0.12, W * 0.09]), W))
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
