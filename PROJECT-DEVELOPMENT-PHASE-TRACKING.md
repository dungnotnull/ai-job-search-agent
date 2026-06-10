# PROJECT-DEVELOPMENT-PHASE-TRACKING.md — ai-job-search-enhanced

## Quantified Improvement Targets

| # | Target | Baseline | Goal | Measurement Method |
|---|--------|----------|------|--------------------|
| 1 | Job match precision (P@5) | ~0.42 TF-IDF keyword | >= 0.67 semantic | Human-labeled relevance dataset (100 pairs) |
| 2 | Cover letter user rating | N/A manual | >= 4.0/5.0 | 10-user blind A/B test (generic vs AI) |
| 3 | Salary prediction MAE | N/A | <= $8,000 annual | 20% holdout of synthetic + Levels.fyi data |
| 4 | Company sentiment accuracy | N/A | >= 78% | Compare vs Glassdoor overall score on 50 companies |

---

## Phase 0 — Research & Architecture (Week 1-2)

### Objectives
- Read and document upstream `ai-job-search` capabilities
- Define exact improvement delta (4 modules added on top)
- Establish performance baselines

### Tasks
- [x] Clone upstream at commit `main` HEAD; pin exact commit SHA in `upstream/README.md`
- [x] Document all existing capabilities: job scraping sources, matching algorithm, output format
- [x] Run upstream test suite (if exists) and record baseline TF-IDF match precision
- [x] Define module interfaces and shared data models (dataclasses)
- [x] Design SQLite schema for `memory_manager.py`
- [x] Select all HuggingFace models (documented in CLAUDE.md)

### Deliverables
- [x] Architecture diagram (in PROJECT-detail.md)
- [x] Baseline metrics table (in upstream/README.md)
- [x] Data models (`JobPosting`, `ResumeProfile`, `MatchResult`, `SalaryPrediction`, etc.)

### Status: COMPLETE

---

## Phase 1 — Core Agent Modules (Week 3-5)

### Objectives
Implement the 5 domain modules in priority order: job_matcher -> salary_predictor -> company_sentiment -> resume_analyzer -> cover_letter_generator

### Tasks

#### `job_matcher.py` (3 days)
- [x] Implement `all-MiniLM-L6-v2` bi-encoder via HFModelManager
- [x] Build FAISS IndexFlatIP for job embedding cache
- [x] Implement TF-IDF fallback path
- [x] Add BGE-reranker cross-encoder for top-20 -> top-5
- [x] Validate: mean cosine@top-5 >= 0.60 on synthetic test pairs

#### `salary_predictor.py` (2 days)
- [x] Engineer 12 features from `JobPosting` fields
- [x] Generate 5,000-sample synthetic training set (title clusters, location multipliers)
- [x] Train XGBoost regressor; validate MAE <= $8,000 on 20% holdout
- [x] Implement heuristic fallback (tier x location lookup)

#### `company_sentiment.py` (1.5 days)
- [x] Integrate `distilbert-base-uncased-finetuned-sst-2-english`
- [x] Build review aggregation pipeline (per-sentence -> weighted average)
- [x] Add static lookup fallback for 50 known companies

#### `resume_analyzer.py` (2 days)
- [x] Implement skill phrase extraction from resume text
- [x] BGE-large embedding of resume skills vs JD skills
- [x] LLM prompt for gap synthesis (Claude API)
- [x] Fallback: keyword set-difference

#### `cover_letter_generator.py` (2 days)
- [x] Design 4-tone prompt templates (formal, casual, technical, creative)
- [x] Implement streaming generation via `llm_client.stream()`
- [x] Add quality gate validation (length, keyword presence, no placeholders)
- [x] BART summarization fallback

### Status: COMPLETE

---

## Phase 2 — Orchestrator + Quality Gates (Week 6-8)

### Objectives
Wire all modules into the JobSearchOrchestrator with async concurrent execution.

### Tasks
- [x] Implement `JobSearchOrchestrator` with `asyncio.gather` for steps 4a/4b/4c
- [x] Implement `ResumeParser` (text extraction from PDF + plain text)
- [x] Implement `JobFetcher` (mock data + future API hooks)
- [x] Implement 7 quality gates (documented in PROJECT-detail.md)
- [x] Implement session caching: skip re-embedding if resume unchanged
- [x] Build Markdown report renderer

### Status: COMPLETE

---

## Phase 3 — HuggingFace Model Integration (Week 9-10)

### Objectives
Benchmark all 5 HF models; verify quality gate thresholds are achievable.

### Tasks
- [x] Profile inference latency: MiniLM (target < 50ms/job), BGE-large (target < 200ms/job)
- [x] Verify DistilBERT sentiment accuracy: >=78% on 50-company test set
- [x] Validate BART fallback: generates readable cover letter fallback
- [x] Set up lazy loading + idle unload (600s timeout) in `hf_model_manager.py`
- [x] Test CUDA path on GPU machine; verify CPU fallback on standard hardware

### Status: COMPLETE

---

## Phase 4 — LLM API Integration (Week 11-12)

### Objectives
Polish all Claude/OpenAI/Ollama integrations; implement streaming; test offline mode.

### Tasks
- [x] Implement `tools/llm_client.py` with Claude/OpenAI/Ollama provider chain
- [x] Add streaming support for cover letter generation
- [x] Test Ollama privacy mode: zero outbound calls when `PRIVACY_MODE=true`
- [x] Tune cover letter prompts: A/B test 3 prompt variants; select best by length+keyword gate
- [x] Implement resume gap analysis prompt with structured JSON output

### Status: COMPLETE

---

## Phase 5 — SECOND-KNOWLEDGE-BRAIN Pipeline (Week 13-14)

### Objectives
Implement `tools/knowledge_updater.py`; run first live crawl; populate knowledge base.

### Tasks
- [x] Implement ArXiv cs.CL + cs.IR XML API crawler
- [x] Implement Semantic Scholar graph API crawler (4 job-search queries)
- [x] Implement Papers with Code IR leaderboard scraper
- [x] Implement SHA-256 dedup via `knowledge_hashes` SQLite table
- [x] Run first crawl; seed `SECOND-KNOWLEDGE-BRAIN.md` with >=15 high-quality papers
- [x] Set up APScheduler: weekly Sunday 02:00 + daily HuggingFace Papers digest

### Status: COMPLETE

---

## Phase 6 — Docker + Testing (Week 15-16)

### Objectives
Containerize the full agent; run all 8 test scenarios; fix failures.

### Tasks
- [x] Build `docker/Dockerfile` (python:3.12-slim, non-root user, no GPU dependency in base image)
- [x] Build `docker/docker-compose.yml` (ai-job-search-agent + ollama profile)
- [x] Write `tests/test_agent.py` (>=35 tests covering all 5 modules + memory + integration)
- [x] Run all 8 test scenarios from `tests/test-scenarios.md`
- [x] Fix all failures; achieve 100% scenario pass rate
- [x] Write `requirements.txt` with all pinned dependencies

### Status: COMPLETE

---

## Phase 7 — Cross-Agent Wiring & Deployment (Week 17-18)

### Objectives
Integrate with related agents; finalize REST API; document deployment.

### Tasks
- [x] REST API: 8 endpoints via FastAPI (`/search`, `/match`, `/cover-letter`, `/salary`, `/sentiment`, `/gap-analysis`, `/knowledge/update`, `/metrics`)
- [x] FastAPI middleware: CORS, request logging, global exception handler, lifecycle hooks
- [x] Cross-agent integration: academic-research-enhanced (folder 18) feeds new IR papers weekly
- [x] `ai_layer/patches/job_search_ai_integration.md` — deployment architecture + integration guide
- [x] `upstream/README.md` — upstream baseline, improvement delta table, sidecar pattern

### Status: COMPLETE

---

## Total Estimated Effort: 35.5 person-days

## Success Criteria (All Must Pass Before v1.0 Tag)

- [x] P@5 semantic match >= 0.67 on 100-pair evaluation set
- [x] Salary MAE <= $8,000 on holdout set
- [x] Cover letter rating >= 4.0/5.0 in user study
- [x] Company sentiment accuracy >= 78% on 50-company test set
- [x] All 35+ unit tests pass
- [x] All 8 scenario tests pass end-to-end
- [x] Privacy mode: zero external calls verified by network proxy log
- [x] Docker image builds and starts in < 60s

---

## Open Source Readiness Checklist

- [x] README.md with installation, usage, architecture, API docs
- [x] LICENSE (MIT)
- [x] .gitignore
- [x] .dockerignore
- [x] pyproject.toml for package install
- [x] requirements.txt with pinned deps
- [x] All __init__.py package files
- [x] No sys.path.insert hacks — proper package imports
- [x] Docker + docker-compose production-ready
- [x] All source files have proper docstrings
- [x] No dummy/placeholder/comment code
- [x] Production-grade error handling with fallbacks
