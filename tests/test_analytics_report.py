from fastapi.testclient import TestClient
from pypdf import PdfReader
import io

from app.main import app


def test_report_downloads_as_a_pdf_with_its_sections():
    for url in ("/analytics/report.pdf", "/analytics/report.pdf?days=30"):
        r = TestClient(app).get(url)
        assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
        assert "attachment" in r.headers["content-disposition"]
        text = " ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
        for section in ("Job Search Report", "Application funnel", "response rates", "Application packages"):
            assert section in text
    assert "Download report (PDF)" in TestClient(app).get("/analytics").text
