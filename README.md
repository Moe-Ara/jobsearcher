# JobSearcher

Your own AI job search assistant. JobSearcher watches hundreds of company career pages, job platforms and your job-alert
e-mails, filters out what doesn't fit you, ranks the rest against your real experience, and prepares tailored
applications — CV, cover letter and e-mail — that you review and send with one click.

- **Finds** jobs in Germany, the UK, the UAE and Saudi Arabia from 130+ sources: company job boards (Greenhouse, Lever,
  Workday, SmartRecruiters, Personio, Ashby, Workable, Recruitee …), job platforms with official APIs or feeds
  (Bundesagentur für Arbeit, Arbeitnow, Reed, Adzuna, Careerjet, Jooble, Workable job board, Berlin Startup Jobs),
  academic boards (academics.de, EURAXESS, jobs.ac.uk — funded PhD positions), and the job-alert e-mails of the sites
  that don't allow automated searches (LinkedIn, Indeed, StepStone, XING, Totaljobs, CWJobs, CV-Library, Glassdoor,
  Monster, Bayt, GulfTalent, NaukriGulf …). **More → Job sites** shows which site is connected how, per country.
- **Filters** with clear rules you control: countries, seniority, experience, defence employers, gig sites, roles
  reserved for nationals, distance from home (internships and side jobs).
- **Ranks** every remaining job with an AI fit analysis that may only use facts from your own profile — every match
  is shown with its evidence (professional, academic, project).
- **Prepares applications**: a tailored CV and cover letter (German or English), fact-checked against your profile and
  reviewed by an ATS check and a recruiter agent. Nothing is ever sent without your click.
- **Learns from you**: tell the agent what to change in an application ("too long", "leave out project X", "lead with
  the robotics work") — it revises the letter and CV, keeps job-specific notes, and turns lasting preferences into
  rules for every future application (**More → Agent memory**, editable).
- **Private by default**: runs on your computer with local AI models (or a cloud API key, if you prefer). Your CV and
  your search never leave your machine unless you choose a cloud model.

```
Sources ─► Ingestion ─► Dedup ─► Rule filters ─► AI screening ─► AI fit analysis ─► Ranking ─► Dashboard ─► Application agent
```

## Quick start

```bash
cp .env.example .env          # optional: cloud API key, mailbox, APP_PASSWORD — see .env.example
docker compose up -d --build
# open http://localhost:8000 → Setup
```

Then open **Setup** and work through the checklist:

1. **Import your CV** (PDF, Word or text). The AI copies its facts into your profile and master CV; you review both
   before they are saved. Nothing is invented.
2. **Say what you are looking for**: countries, roles and keywords, home town, how far you would drive for an
   internship or a side job.
3. **Personal details**: your name (signs your letters) and citizenship(s).
4. Optional: connect the mailbox your job alerts go to, add free Careerjet / Jooble API keys, set a password.

Good to know:

- **Your data**: the database lives on its own Docker volume (`appdb`, fast on Windows/macOS too); a copy is written
  to `./data/backups` every day (newest 7 kept, or **Operations → Back up database**). Documents, exported applications
  and caches are in `./data`; your settings in `./config` (every change made in the app keeps the previous version in
  `config/.backups`).
- **No AI model?** The app still works with a free keyword ranking, clearly labelled. Local models need an NVIDIA GPU
  with ≥ 12 GB (see below); a cloud key works on any machine.
- **Hosting it?** Set `APP_PASSWORD` in `.env` first — every page and the API then need a sign-in.
- Tests: `docker compose run --rm --no-deps -v "$PWD/tests:/app/tests" app python -m pytest -q` · Logs: `docker compose logs -f app`

Without Docker (Python 3.12+):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # Windows: .venv\Scripts\...
.venv/bin/uvicorn app.main:app --port 8000        # keep a single worker (the scheduler runs in-process)
.venv/bin/python -m pytest -q
```

## Local models (GPU) — one-time setup

The app runs a local multi-model pipeline through an **Ollama** container with NVIDIA GPU access. This needs Docker Desktop with WSL2 GPU support, or the NVIDIA Container Toolkit on Linux. Models are swapped on demand, and only one is ever resident (16 GB VRAM).

| Role | Model (GGUF via `hf.co/…`) | Used for |
|---|---|---|
| screen / extract | `Qwen/Qwen3-8B-GGUF:Q5_K_M` (non-thinking) | screening every job that passes the hard filters; structured JSON only |
| analysis | `Qwen/Qwen3-14B-GGUF:Q5_K_M` | deep fit analysis, **only** for jobs the screener rates as promising |
| writer | `Qwen/Qwen3-14B-GGUF:Q5_K_M` (non-thinking) | cover-letter drafts and application strategy |
| editor | `mistral-small-3.1-text:q4_k_s` — `bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_S` without its vision projector (fully on a 16 GB GPU, ~3× faster, identical text), 8K context | **GermanStyleEditor**: natural German, no new facts |
| validator | `Qwen/Qwen3-14B-GGUF:Q5_K_M` (the writer model, thinking on) | factual claim validation and correction of letters |
| humanizer | `jialinyyzz/humanizer:Q6_K` (Gemma-4-12B fine-tune, raw completion) | optional natural-rewrite pass for letters, and the standalone **Humanize** page |

Download the models once (about 40 GB, stored in the `ollama_models` Docker volume and kept across restarts):

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup-local-models.ps1
```
```bash
# Linux/macOS/WSL
sh scripts/setup-local-models.sh
```

That is equivalent to:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M
docker compose exec ollama ollama pull hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_S
# text-only copy of the editor (drops the unused vision projector so the model fits the GPU completely)
docker compose exec ollama sh -c "ollama show --modelfile 'hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_S' | awk '/^#/ {next} /^FROM / { if (seen++) next } {print}' > /tmp/Modelfile.text && ollama create mistral-small-3.1-text:q4_k_s -f /tmp/Modelfile.text"
docker compose exec ollama ollama pull hf.co/jialinyyzz/humanizer:Q6_K
docker compose up -d
```

Configuration:
- `.env` holds `LOCAL_*` model names and per-role on/off switches, plus `CLOUD_FALLBACK_ENABLED`.
- `config/preferences.yaml` → `local_llm` holds context size, keep-alive, temperature and thinking per role. Set `analysis.think: true` for Qwen3 reasoning on fit analysis.
- Roles that are disabled locally use the cloud provider when one is configured.
- With `CLOUD_FALLBACK_ENABLED=true`, failed local calls are retried in the cloud and cost-tracked as usual. Local calls cost $0, but their tokens and latency are still recorded.
- The **Operations** page shows whether Ollama is reachable, which models are installed or loaded, and the last successful request and last error.
- When running the app outside Docker, set `LOCAL_LLM_BASE_URL=http://localhost:11434`.

**How the GPU is used:**
- **Batch analysis is two-phase.** All pending jobs are screened first, with only the 8B model loaded. Then the promising ones are deep-analysed with the 14B model. That's one swap per run instead of one per job.
- **The screener's output feeds the deep analysis** as delimited, unverified notes. The deterministic weighted scoring stays authoritative. Jobs the screener rates as poor matches are kept as LOW PRIORITY with a provisional score.
- **If Ollama is unreachable or a model is missing**, the batch stops and the jobs stay queued without using up their retry attempts.

## Cover letters (application material)

On a job page, **Generate German / English Cover Letter** creates a letter, but **only** when you press it. Status changes, including APPLYING, never generate anything or load a model. Nothing is ever sent.

**Principle: the AI provides the German and the structure; you provide the motivation, the priorities and the reasoning.** The workflow on the job page has three steps:

1. **Personal application notes.** Four short, informal fields, in English, German, Arabic or mixed: *Why does this job genuinely interest you?*, *Why do you think you fit?*, *What should the letter emphasize?* and *Anything to avoid or be careful about?* The first two are strongly encouraged. Notes are stored per application (table `application_notes`), saved with **Save notes** or automatically when you generate, and never overwritten by regenerating. Each letter version keeps a snapshot of the notes it was written with.
   - Notes steer **motivation and emphasis only**. They are never evidence: "I know Kubernetes pretty well" in a note does not let a letter claim Kubernetes unless `profile.yaml` supports it. The fact check never sees the notes.
   - Without notes (quick mode), the letter stays factual and restrained. It never invents enthusiasm ("I have followed your company for years", "passion for your mission"…), and the review says so.
2. **Generate.** The writer gets the profile, the vacancy, the fit analysis, your notes, your writing preferences and the chosen language. It turns your reasons into natural professional German instead of translating them line by line. It picks 2–4 relevant pieces of evidence, keeps evidence levels apart (academic or project work never becomes "beruflich"), and bridges gaps you flag with truthful adjacent experience. Each generation picks one of several human letter shapes (work-first, overlap-first, path-first, compact), so letters don't share one machine-made structure. Shorter beats padded, and no deliberate mistakes are added.
3. **Review.** The page shows original draft → German edit → final (validated), plus **Personalization** (HIGH / MEDIUM / LOW), **Genericity** ("could most of this be sent unchanged to another employer?": NO / PARTLY / YES) and **Factual warnings**. The checks behind it: personal notes used, concrete facts about you, references to this vacancy, generic phrases, unsupported claims and evidence-level violations. Nothing is blocked; low scores only warn. This is a content-specificity check, not an AI detector.

The **letter strategy** per application (`NONE`, `AI_WITHOUT_PERSONAL_INPUT`, `PERSONAL_INPUT_AI_EDITED`, `MANUAL`) is recorded automatically. Pick "Written by me" if you send your own. It's exposed in `GET /api/jobs/{id}`, so applications with and without personal input can later be compared with responses. Nothing draws conclusions from it yet.

1. **Language.** German ads default to German. So do Germany-based roles that clearly require German. English ads get English. You can override it in the form.
2. **Writer (Qwen3-14B).** Writes the letter (about 250–350 words; configurable under `writing:` in `preferences.yaml`). The dedicated German prompt picks 2–3 concrete pieces of evidence from `profile.yaml`, avoids stock phrases, and never invents facts.
3. **GermanStyleEditor (Mistral Small 3.1, optional).** Makes the German draft sound natural while preserving your own reasoning (it sees your notes). It may shorten, vary, remove boilerplate and repetition, but never create motivation, facts, technologies or stronger claims.
4. **Factual validation.** Every factual claim about you is classified against `profile.yaml` as SUPPORTED_PROFESSIONAL, SUPPORTED_ACADEMIC, SUPPORTED_PROJECT, SUPPORTED_GENERAL, AMBIGUOUS or UNSUPPORTED. A claim above its evidence level is UNSUPPORTED, e.g. "mehrjährige berufliche C++-Erfahrung" when C++ is only `academic`/`project`. A deterministic evidence guard enforces this even if the model misses it:
   - UNSUPPORTED and AMBIGUOUS claims are corrected, then re-checked.
   - Anything still UNSUPPORTED is removed sentence by sentence.
   - AMBIGUOUS claims, numbers not found in your profile, and length issues are shown as warnings.

**Fit analysis in letters.** When the job has been analysed, the writer gets its strong matches, its partial/transferable matches and its critical gaps as a delimited `<FIT_ANALYSIS>` block. The strong matches carry the letter. One or two transferable skills are made explicitly visible as an honest bridge: the job's requirement, the related experience from your profile, and why it transfers, e.g. "Mit GitLab CI habe ich noch nicht gearbeitet, aber …". The target skill itself is never claimed. The fact check and the unknown-technology guard accept such disclaimed bridges and check only the experience they rest on. The editor keeps them. The final letter shows **Skills made visible**: which strong and transferable skills appear in the text. Switch it off with `writing.use_fit_analysis: false`.

Every stage (draft, edited, final) is stored with its model, prompt version, timestamps, tokens and latency. **Regenerate** creates a new version and never overwrites older ones. The page offers **Copy final letter**, **Run German style editor**, **Run humanizer** and **View factual validation**.

## Letter templates (human-written examples)

The **Templates** page (or the folder `config/letter_templates/`, one `.md`/`.txt` file per letter) holds cover letters written by people: your own past letters, or good examples. For each new letter, the best-matching templates go to the writer and the German editor as delimited `<EXAMPLE_LETTER>` blocks. Matching is by language first, then overlap between the job ad and the template's text and tags. Each example comes with a measured outline: paragraphs, words and sentences per paragraph, and how each paragraph opens.

What the models take from templates and what they don't:
- **Structure and voice are copied from the templates.** Example 1 sets the paragraph count, the order, what each paragraph does, the opening, the closing and the lengths. Any further examples add tone. The letter length follows the templates (their body length ±15 %).
- **Facts are not.** Names, employers, degrees, numbers and technologies come only from `profile.yaml`. A deterministic leak check flags anything that came from a template instead, such as "Contoso GmbH", "450 test cases" or eight-word phrases copied verbatim. Those flags appear as `template` warnings next to the fact check.
- The address block, date and the signature name are stripped from templates before they reach the model.
- On the job page, the **Template** selector picks Auto (best match), a specific template, or None (built-in four-paragraph structure). The final letter shows which template it was built from.

Settings: `writing.use_templates`, `writing.templates_per_letter` (default 2) in `preferences.yaml`.

## Humanizer

An optional rewrite pass with [`jialinyyzz/humanizer`](https://huggingface.co/jialinyyzz/humanizer) (Gemma-4-12B fine-tune, Apache-2.0, Q6_K ≈ 10 GB). It was the strongest humanizer on Hugging Face when this was built. No German-specific humanizer of comparable quality exists; the German candidates are small, old paraphrasers. The model was trained on English and Chinese, so German output is guarded:
- It is a completion model. It is called in raw mode (`/api/generate`) with the exact training prompt and the authors' samplers (`preferences.yaml` → `local_llm.roles.humanizer`: temperature 1.0, top_p 0.95, top_k 0, min_p 0, repeat_penalty 1.0). Don't change these.
- Text is rewritten in chunks of ~350 words. Salutation and closing of letters stay unchanged.
- A paragraph that comes back in another language is replaced by the original and reported.
- Numbers and technical terms that disappear or appear are listed as warnings.
- In letters, the humanized text always goes through the normal factual validation afterwards. The validator is the writer model (Qwen3-14B), never the humanizer.

Use it in one of three ways:
- **Letters:** tick **Humanizer pass** when generating (draft → editor → humanized → final), or press **Run humanizer** on an existing final letter. Default off: `writing.humanizer_by_default` in `preferences.yaml`.
- **Standalone:** the **Humanize** page in the nav bar. Paste any text, then copy the result and its warnings.
- **API:** `POST /api/humanize` with `{"text": "...", "keep_letter_frame": true}`.

Disable it with `LOCAL_HUMANIZER_ENABLED=false`. It is local-only (no cloud fallback) and never used for screening or analysis.

## Daily workflow

1. Let the scheduled sources fetch vacancies (company feeds and alert emails).
2. Open the dashboard and go through the top-ranked unreviewed jobs.
3. Mark recommendation quality GREAT / MAYBE / BAD when useful.
4. Move interesting jobs to PRIORITY.
5. Read the original vacancy yourself.
6. Move selected jobs to APPLYING.
7. Generate application material explicitly, if needed.
8. Review all generated material yourself (see the factual-validation warnings).
9. Apply manually.
10. Update the status when responses arrive.

## Calibration workflow

1. Open **Calibrate**. The default order is a stable shuffle, so you get a representative sample. Label about 50 jobs GREAT / MAYBE / BAD before relying on the AI ranking.
2. Press **Analyze labeled jobs**. This runs local screening and then deep analysis on exactly those jobs. Hard-filtered jobs are skipped, and unchanged jobs come from the cache.
3. Check **Precision@10** (GREAT = relevant) against the random-order baseline shown next to it, plus the average score per label.
4. Tune `profile.yaml`, the prompts or the weights in `preferences.yaml` only if recommendation quality is weak. Labels are used for evaluation only and never change scores.

## First steps

1. **Setup → Import your CV.** Your profile (`config/profile.yaml`) is the only source of truth the AI may use about
   you. After the import, check how skills are grouped (professional / academic / project / listed), your years of
   experience, and your languages.
2. **Setup → What you are looking for** covers the everyday settings; **All preferences (advanced)** opens
   `config/preferences.yaml` (filters, defence lists, scoring weights, LLM budget) in the browser.
3. **More → Job sources**: 130+ sources are pre-seeded from `config/companies.yaml`. Add your own by pasting a careers
   URL on **More → Add a job**.
4. **Jobs**: the default view is *unreviewed jobs from the last 7 days, best match first*.

## Configuration

Your configuration is **yours**: it lives in `config/` (git-ignored, never committed). On the first start the app
creates it from the shipped examples in `config.example/` — your own files are never overwritten — and the **Setup**
page edits it for you (CV import, search preferences, personal details, in-browser editors that validate every change
and keep the previous version in `config/.backups`).

| File | What |
|---|---|
| `.env` | Secrets and runtime settings: LLM provider/keys/models, IMAP, `APP_PASSWORD`, job-platform API keys (`CAREERJET_API_KEY`, `JOOBLE_API_KEY`, `REED_API_KEY`, `ADZUNA_APP_ID`/`ADZUNA_APP_KEY`). Git-ignored. See `.env.example`. |
| `config/profile.yaml` | Your candidate profile: education, experience, skills by evidence level, languages, citizenship(s), career goals. Filled by Setup → Import your CV. |
| `config/cv.yaml` | Your master CV — every tailored CV is selected from it. Filled by the CV import. |
| `config/preferences.yaml` | Countries, home town, search terms, hard filters, defence exclusion, scoring, scheduler, LLM budget, letters, e-mail parsers. |
| `config/companies.yaml` | Job sources (companies' job boards, platforms, academic boards) — imported once, then managed in the UI. |
| `config/agent_memory.yaml` | What the application agent learned from your feedback (More → Agent memory). |
| `config/letter_templates/` | Your own human-written cover letters (structure and voice only). |

Tests never read your `config/`: they run against a fictional candidate in `tests/fixtures/config/`.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to use, modify and share for any non-commercial purpose (your own job
search included). Commercial use needs a separate licence from the author.

## Sources

**Company career feeds** (highest priority; employer data becomes the canonical version of a job):

| ATS | How | Identifier |
|---|---|---|
| Greenhouse | public Job Board API | board token (`job-boards.greenhouse.io/<token>`) |
| Lever | public Postings API (US/EU) | company slug (`jobs.lever.co/<slug>`) |
| Personio | public XML feed | subdomain (`<sub>.jobs.personio.de`) |
| SmartRecruiters | public Posting API | company id (`jobs.smartrecruiters.com/<Id>`); supports `country`/`q` filters |
| Generic | schema.org `JobPosting` JSON-LD on the careers page, plus optional `job_link_pattern` to follow job links | careers URL |

Pasting a careers URL auto-detects the ATS. Workday, SuccessFactors and similar can be added later as new adapters (see *Extending*). There is no anti-bot evasion and no JS rendering.

**Job-alert emails** (LinkedIn, Indeed, StepStone, Bundesagentur für Arbeit, plus a generic fallback). `IMAP_FOLDER=*` (the default) scans every folder except trash, sent and drafts; on Gmail it scans All Mail and Spam. An alert stored in several folders is processed once. **Operations → Mailbox (IMAP)** shows whether the login works and which folders are scanned. After editing `.env`, press **Reload .env & retry connection**: no restart is needed, and on success the mailbox is polled immediately. The app reads your mailbox over IMAP in read-only mode and never changes flags. For Gmail, enable IMAP and use an App Password. You can also upload `.eml` files on the Add page. The **Preview** option shows what a parser extracts without saving anything, which makes tuning parsers against real emails easy. The boards themselves are never scraped. Jobs that come from alerts are merged with the employer's own posting when both exist.

**Manual**: paste a job URL, a description, or both. Greenhouse, Lever and SmartRecruiters job URLs go through their APIs; other pages are read via JSON-LD. LinkedIn and Indeed pages usually need the description pasted. When title or company is missing, the cheap model extracts them.

## Deduplication

1. Same source and external id: the existing record is refreshed (`last_seen`).
2. Same canonical URL (tracking parameters stripped; LinkedIn, Indeed and BA links reduced to their job id).
3. Fuzzy match: company (legal suffixes and aliases normalised), title (gender tags removed, synonyms such as *AI ⇄ Artificial Intelligence*, *KI*, *ML*, *Sr.*), compatible location, and description similarity when both sides have one.

Guards: different seniority levels, different cities, or two different ids from the same source are never merged. Every source is kept as a `JobSource` row (provenance). Preferred canonical order: employer > job board > manual > recruiter > aggregator.

## Filtering and analysis

**Untrusted sources.** Gig and "AI-training" marketplaces and talent funnels (micro1, Alignerr, Outlier, Mercor, DataAnnotation, Remotasks, Mindrift, Crossover, Toloka…) are skipped. Their postings are rejected before any LLM call and hidden from the dashboard. Matching is precise: company names exactly (or with a generic suffix like "AI"), domains in the job's URLs and in links inside the posting, gig titles ("AI Trainer", "RLHF Specialist"…) on the title only, and platform wording only when combined with gig signals. Edit `untrusted_sources` in `preferences.yaml`, then press **Rescore** on Operations to re-apply the list to stored jobs.

**Hard filters run before any LLM call**, and filtered jobs cost nothing. Rules: wrong country (remote roles abroad are flagged rather than rejected), excluded cities, unrealistic seniority or job type, unrelated profession, too many required years, and fundamentally missing qualifications. A single missing technology never rejects a job.

**Defence exclusion**:
- Pure-play defence companies are always rejected.
- Explicit defence wording ("weapon systems", "Rüstung", "armed forces", "military vehicles", ...) is rejected.
- Mixed civil/defence companies (Airbus, Thales, Rheinmetall, Rohde & Schwarz, ...) are **flagged, not rejected**. The LLM then judges the specific business unit and can reject the job with a stated reason.
- "Cyber defence", SOC and blue-team work are treated as civilian.

**LLM analysis** is two-stage:
1. A cheap triage model screens out defence work and clearly unrelated professions.
2. The strong model returns structured JSON validated with Pydantic: sub-scores, matches with evidence level (professional / academic / project / transferable), critical and nice-to-have gaps, why good / why not, defence check, and more.

**Anti-hallucination**: every claimed match must cite a profile entry. Claims that cannot be traced to `profile.yaml` are dropped, shown as "Dropped (not in your profile)", and lower the confidence.

**Prompt-injection safety**: job text is treated as untrusted data. It appears only inside `<job_posting>` delimiters in the user turn. Delimiter-like tags inside the job text are removed, and the system prompt forbids following embedded instructions. Postings that try to instruct the model are flagged in the UI.

**Cost control**:
- Every call is cached by a hash of kind, prompt version, model, profile and the exact prompt, so unchanged jobs are never re-sent.
- Hard filters run first.
- Descriptions are truncated.
- The cheap model handles triage and extraction.
- The system prompt and profile are prompt-cached on Anthropic.
- A daily USD budget stops paid calls.
- Token usage and cost are tracked per call (Operations page).

Providers (set in `.env`):
- `anthropic`: defaults `claude-opus-5-5` (analysis) and `claude-haiku-4-5` (triage). Server-side refusal fallback is on (`ANTHROPIC_REFUSAL_FALLBACK=true`).
- `openai`: OpenAI or any OpenAI-compatible endpoint, such as Ollama, vLLM, LM Studio or OpenRouter. Set `OPENAI_BASE_URL`. Local endpoints are treated as free.

Override the models with `LLM_ANALYSIS_MODEL` and `LLM_TRIAGE_MODEL`.

## Scoring

The LLM's own overall number is stored but **not used**. The fit score is a weighted average of structured components, with weights in `preferences.yaml`:

| Component | Weight | From |
|---|---|---|
| technical | 30% | LLM technical match |
| experience | 20% | LLM experience match blended with your seniority preferences |
| role family | 15% | your role-family scores |
| sector | 10% | your sector scores |
| location | 10% | your country/city scores |
| career value | 10% | LLM career match |
| source reliability | 5% | best source type |

Adjustments: a target-company priority bonus (A/B) and a small penalty per critical gap.

Recommendation bands: **≥ 8.0 PRIORITY**, **6.5–7.9 REVIEW**, **< 6.5 LOW PRIORITY**. Hard-filtered jobs are **REJECTED**. A score never rejects a job; it only ranks it.

**Rescore all** on the Operations page recomputes filters and scores for free after you change weights.

## Future ideas (not implemented)

- Use the collected labels to suggest weight changes (deliberately not automatic).
- Fetch full descriptions for alert-email jobs that only have a snippet.
- Workday / SuccessFactors adapters.

## Statuses and analytics

Status pipeline: `NEW → REVIEW / PRIORITY / SKIPPED → APPLYING → APPLIED → HR_SCREEN → INTERVIEW → TECHNICAL → FINAL → OFFER`, plus `REJECTED` and `EXPIRED`.

- Every change is stored with a timestamp.
- Clicking an action button again undoes it.
- Jobs that disappear from their feed become inactive. Unreviewed ones become `EXPIRED`.

The Analytics page derives applications per week and response, interview and offer rates. It breaks them down by source, country, sector, role family, fit-score bucket and CV version (set the CV version in a job's notes panel).

## Retention (14 days)

Only recent vacancies are kept (`retention.max_job_age_days`, default 14):
- **Age** is the posted date, or the date the job was first seen when the source gives no posted date.
- **When saving:** postings older than that are skipped. Jobs you add manually are exempt.
- **Cleanup:** every 6 hours older jobs are deleted. You can also run it from Operations.
- **Kept regardless:** jobs you saved or prioritised, anything in the application pipeline (including employer rejections after applying), labeled jobs, jobs with notes, and jobs with generated letters.
- **No re-adding:** deleted feed postings are remembered, so a still-listed posting isn't re-added as new.

## Operations and scheduling

The in-process APScheduler:
- checks companies when due (every 6 / 12 / 24 h for priority A / B / C),
- polls the mailbox hourly when IMAP is configured,
- analyses pending jobs every 30 minutes in batches.

The **Operations** page shows every source run (found / new / merged / updated / gone / errors) and background tasks. It also shows LLM spend and has buttons to run everything now. A failure in one company feed or one posting never stops the others.

JSON API: `/api/health`, `/api/stats`, `/api/jobs?q=…`, `/api/jobs/{id}`, `POST /api/jobs/{id}/status`, `POST /api/ingest/manual`, `/api/companies`, `POST /api/tasks/{companies|companies_all|email|analysis}`. Interactive docs are at `/docs`.

## Security notes

- Secrets come only from environment variables. `.env` is git-ignored.
- Scraped text is sanitised: control characters, zero-width and bidi characters are removed, and HTML is converted to text.
- Templates autoescape. Only `http(s)` links are rendered.
- URLs you paste can't point at private or internal addresses.
- Cross-origin POSTs are blocked.
- The port is bound to `127.0.0.1` (and `::1`) only. Set `APP_PASSWORD` before exposing it anywhere: every page and the
  JSON API then require a sign-in (browser: signed HttpOnly cookie; scripts: `Authorization: Bearer <APP_PASSWORD>`).
- Fonts and scripts are served by the app itself — no requests to Google Fonts or CDNs.
- There is no code execution of any input.

## Extending

- **New ATS**: subclass `JobSourceAdapter` in `app/sources/` and implement `fetch_jobs()` and `normalize_job()`, plus optionally `external_id()` and `fetch_details()`, which only runs for unseen postings. Register it in `app/sources/registry.py` (`ADAPTERS` and `detect_ats`).
- **New alert email**: subclass `EmailJobAlertParser` in `app/email_ingest/parsers/` (anchor on the board's stable job-URL shape) and add it to `all_parsers`.
- **PostgreSQL**: set `DATABASE_URL=postgresql+psycopg://…` and add `psycopg[binary]` to `requirements.txt`. Migrations are portable.
- **Schema changes**: `alembic revision --autogenerate -m "…"`. Migrations run automatically at startup.

## Layout

```
app/
  main.py               FastAPI app, lifespan (migrations, seed, scheduler), CSRF guard
  settings.py           env settings          config_files.py   profile/preferences models
  models.py / enums.py  SQLAlchemy models     db.py             engine, sessions, migrations
  core/                 text sanitising, normalisation/inference, polite HTTP client
  sources/              JobSourceAdapter + Greenhouse, Lever, Personio, SmartRecruiters, generic, JSON-LD
  email_ingest/         IMAP/.eml processing + alert parsers
  filters.py dedup.py scoring.py
  llm/                  provider interface, roles (model routing), local (Ollama), Anthropic, OpenAI-compatible,
                        prompts, letter_prompts, schemas, heuristic
  services/             ingest, analysis, llm_calls, letters, companies, manual, queries, analytics, tasks
  scheduler.py
  web/                  routes (HTML), api (JSON), templates, static
config/                 profile.yaml, preferences.yaml, companies.yaml
scripts/                setup-local-models.ps1 / .sh (one-time model download)
alembic/                migrations
tests/                  pytest suite (no network, fake LLM)
```

## Status

- **Milestone 1 (done)**: models, CandidateProfile, TargetCompany, manual URL/description entry, hard filters, LLM analysis, scoring, ranked dashboard with status controls, notes, search, filters.
- **Milestone 2 (done)**: Greenhouse, Lever, Personio and SmartRecruiters adapters, generic JSON-LD career pages, scheduled checks, dedup, source tracking, inactive marking.
- **Local LLM pipeline (done)**: Ollama service, role routing (local, then cloud, then heuristic), two-phase screening and analysis, cover letters with German style editor and factual validation.
- **Milestone 3 (done, needs tuning)**: IMAP and `.eml` ingestion; LinkedIn, Indeed, StepStone and BA alert parsers. The parsers were built without real alert emails. Use the `.eml` preview with your own alerts to verify them, especially Indeed, which hides job ids behind tracking links.
