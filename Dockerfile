FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Agent tools: Tectonic (self-contained XeTeX engine for the CV / letter PDFs) and Poppler (pdftotext — the text
# layer an applicant tracking system parses).
ARG TECTONIC_VERSION=0.15.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils fontconfig curl ca-certificates \
    && curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
       | tar xz -C /usr/local/bin \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Download + cache every LaTeX package the CV and letter templates use (kept in this layer; code changes below
# don't invalidate it). Rerun the image build after changing the templates' packages.
COPY --chown=app:app docker/latex-warmup /opt/latex-warmup
USER app
RUN cd /opt/latex-warmup && for f in *.tex; do tectonic -X compile --untrusted "$f" >/dev/null || exit 1; done && rm -f *.pdf
USER root

COPY . .

RUN mkdir -p /data /db && chown -R app:app /app /data /db
USER app

# The database lives on its own Docker volume (/db, a Linux file system): SQLite on a Windows/macOS bind mount is
# ~100x slower. Documents, exports, caches and the daily backups stay in the visible ./data folder (/data).
ENV DATABASE_URL=sqlite:////db/jobsearcher.db \
    DATA_DIR=/data \
    CONFIG_DIR=/app/config

EXPOSE 8000
# Single worker on purpose: the in-process scheduler must run exactly once.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
