# PROJECT-detail.md — ai-job-search-enhanced

## Executive Summary

**ai-job-search-enhanced** is a production-grade career intelligence agent forked from [MadsLorentzen/ai-job-search](https://github.com/MadsLorentzen/ai-job-search) with four quantified AI/ML improvements:

| Improvement Target | Baseline (upstream) | Target | Method |
|-------------------|--------------------|---------| -------|
| Job-resume match precision (P@5) | ~0.42 (keyword TF-IDF) | ≥ 0.67 | `all-MiniLM-L6-v2` semantic embeddings + BGE reranker |
| Cover letter personalization score | N/A (manual) | ≥ 4.0/5.0 user rating | Claude API with user tone profile |
| Salary prediction MAE | N/A | ≤ $8,000 annual | XGBoost regression on 12 posting features |
| Company culture fit score accuracy | N/A | ≥ 78% agreement | DistilBERT sentiment + review aggregation |

---

## Problem Statement

Job seekers applying via keyword-matching platforms experience:
1. **Precision gap**: irrelevant jobs surfaced because "machine learning" ≠ "AI engineering" to a keyword matcher
2. **Generic cover letters**: one-size-fits-all text that ignores the specific job's culture and requirements
3. **Salary blindness**: no market benchmark; candidates accept offers 15–30% below market median
4. **Skill gap ambiguity**: unclear which skills to acquire for a target role → wasted upskilling time

The agent solves all four autonomously from a single command: `python agent/main.py search --resume resume.pdf --query "senior ML engineer remote"`

---

## Target Users & Use Cases

| User | Trigger | Agent Action |
|------|---------|-------------|
| Software engineer pivoting to ML | Uploads resume, enters "ML engineer" | Semantic match → ranked job list + gap analysis + cover letters |
| Recent graduate | Enters "data scientist entry level NYC" | Match + salary range + company sentiment for top 10 results |
| Career advisor | Batch processes 50 client resumes | REST API batch endpoint → per-client career report |
| Recruiter | Enters candidate profile | Reverse-match: find candidates semantically similar to a JD |

---

## Agent Architecture (ASCII Diagram)

```
User CLI / REST API
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  JobSearchOrchestrator (agent/orchestrator.py)              │
│                                                             │
│  1. parse_resume(resume_path) → ResumeProfile               │
│  2. fetch_jobs(query, sources) → List[JobPosting]           │
│  3. asyncio.gather(                                         │
│       match_jobs(resume, jobs),       ← job_matcher         │
│       predict_salaries(jobs),         ← salary_predictor    │
│       score_companies(jobs),          ← company_sentiment   │
│     )                                                       │
│  4. rerank(matches, salary, sentiment) → ranked_list        │
│  5. analyze_gaps(resume, top_jobs)    ← resume_analyzer     │
│  6. generate_letters(resume, top3)   ← cover_letter_gen     │
│  7. assemble_report() → Markdown + JSON                     │
└─────────────────────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
   LLM API       HuggingFace    Job Board APIs
 (llm_client)  (hf_model_mgr)  (scraped / mock)
```

---

## Full Module Catalog

### `agent/modules/job_matcher.py`
**Responsibility:** Bi-encoder semantic similarity matching between a resume profile and a list of job postings.

| Attribute | Detail |
|-----------|--------|
| Input | `ResumeProfile` (text + skill embeddings), `List[JobPosting]` |
| Output | `List[MatchResult]` sorted by cosine similarity descending |
| Model | `all-MiniLM-L6-v2` (fast, 384-dim), `bge-large-en-v1.5` (precision mode) |
| Reranker | `BAAI/bge-reranker-large` cross-encoder on top-20 candidates → top-5 |
| Quality gate | Mean cosine similarity of top-5 results ≥ 0.60 before returning |
| Fallback | TF-IDF + cosine if FAISS unavailable |
| Tools called | `hf_model_manager.encode()`, `hf_model_manager.rerank()` |

### `agent/modules/cover_letter_generator.py`
**Responsibility:** Generate personalized cover letters via Claude API using resume + JD + user tone profile.

| Attribute | Detail |
|-----------|--------|
| Input | `ResumeProfile`, `JobPosting`, `ToneProfile` (formal/casual/technical/creative) |
| Output | `CoverLetter` (body text, subject line, key highlights list) |
| LLM | Claude `claude-opus-4-8` primary; `gpt-4o` fallback; Ollama `llama3` offline |
| Token budget | ~1,200 input tokens; ~600 output tokens per letter |
| Prompt strategy | System: role + tone instructions. User: resume summary + JD + company culture signals |
| Quality gate | Letter length 250–500 words; contains ≥3 JD-specific keywords; no hallucinated credentials |
| Fallback | BART summarization of resume → template merge |
| Tools called | `llm_client.complete()` |

### `agent/modules/salary_predictor.py`
**Responsibility:** Predict market salary range for a job posting using XGBoost regression.

| Attribute | Detail |
|-----------|--------|
| Input | `JobPosting` (title, skills, location, company_size, seniority, industry, remote_flag, yoe_required) |
| Output | `SalaryPrediction` (low, median, high as annual USD; confidence: 0.0–1.0) |
| Model | XGBoost regressor; 12 engineered features; trained on synthetic 5,000-sample dataset |
| Training | Synthetic dataset mirroring Levels.fyi / Bureau of Labor Statistics distributions |
| Validation | 20% holdout; target MAE ≤ $8,000 |
| Fallback | Heuristic: title tier × location multiplier lookup table |
| Tools called | None (self-contained sklearn + xgboost) |

### `agent/modules/company_sentiment.py`
**Responsibility:** Score company culture, management, and work-life balance from review signals.

| Attribute | Detail |
|-----------|--------|
| Input | Company name + optional list of review strings |
| Output | `CompanySentiment` (culture: 0–5, management: 0–5, wlb: 0–5, overall: 0–5, label: POSITIVE/NEGATIVE/MIXED) |
| Model | `distilbert-base-uncased-finetuned-sst-2-english` for per-sentence polarity |
| Aggregation | Weighted average of per-sentence scores (recency-weighted if timestamp available) |
| Fallback | Mock scores drawn from a static lookup of 50 well-known companies |
| Tools called | `hf_model_manager.classify_sentiment()` |

### `agent/modules/resume_analyzer.py`
**Responsibility:** Identify resume-to-JD skill gaps and produce a ranked improvement roadmap.

| Attribute | Detail |
|-----------|--------|
| Input | `ResumeProfile`, `List[JobPosting]` (target roles, up to 5) |
| Output | `GapAnalysis` (missing_skills: List with priority rank, learning_path: List[Step], time_estimate: str) |
| Method | BGE-large embeddings of skill phrases → cosine distance → missing clusters → Claude synthesis |
| Quality gate | ≥5 distinct skill gaps identified for entry roles; ≥3 for senior roles |
| Prompt | "Given this resume and these JDs, identify the top N missing skills and suggest a 90-day learning plan" |
| Fallback | Keyword set-difference as minimal gap list |
| Tools called | `llm_client.complete()`, `hf_model_manager.encode()` |

---

## HuggingFace Model Selection

| Model | Task | MTEB/Benchmark Score | Reason over Alternatives |
|-------|------|---------------------|--------------------------|
| `all-MiniLM-L6-v2` | Fast semantic matching | SBERT Cosine@1 = 0.828 | 5× faster than `bge-large`; sufficient for top-20 recall |
| `bge-large-en-v1.5` | Precision embedding | MTEB Retrieval avg = 0.5418 | Best open English retrieval model as of 2024-Q2 |
| `bge-reranker-large` | Cross-encoder reranking | BEIR NDCG@10 = 0.537 | +12pp over bi-encoder; acceptable latency for top-20 |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment | SST-2 acc = 91.3% | 40% smaller than `bert-base`; fine-tunable on HR reviews |
| `facebook/bart-large-cnn` | Summarization fallback | ROUGE-L = 0.406 | Best open extractive+abstractive summarizer |

---

## LLM API Integration Spec

### Claude (Primary)
- **Model:** `claude-opus-4-8`
- **Use cases:** Cover letter generation, resume gap analysis, salary context, research synthesis
- **Prompt style:** Zero-shot with detailed system context; structured JSON output via `<json>` tags
- **Token budget:** Cover letter 1800 total; Gap analysis 2400 total; Research synthesis 3200 total
- **Streaming:** Enabled for cover letters (user sees text appear in real-time)

### OpenAI (Fallback)
- **Model:** `gpt-4o`
- **Use cases:** PDF screenshot parsing, structured JSON extraction from noisy resume text
- **Token budget:** ≤ 2000 per call

### Ollama (Offline)
- **Model:** `llama3` or `mistral`
- **Use cases:** All tasks when `PRIVACY_MODE=true`; no resume data leaves local machine

---

## End-to-End Execution Flow

```
Step 1: User runs: python agent/main.py search --resume cv.pdf --query "ML engineer"
Step 2: Orchestrator calls ResumeParser.parse(cv.pdf) → ResumeProfile (skills, experience, education)
Step 3: Orchestrator calls JobFetcher.fetch(query, n=50) → List[JobPosting] (from mock/scraped/API)
Step 4: asyncio.gather:
   4a: job_matcher.match(resume, jobs) → ranked top-20 + reranked top-5 MatchResults
   4b: salary_predictor.predict_batch(jobs) → SalaryPredictions[]
   4c: company_sentiment.score_batch(companies) → CompanySentiment[]
Step 5: Orchestrator merges: final_score = 0.5×match + 0.25×salary_percentile + 0.25×sentiment
Step 6: resume_analyzer.analyze(resume, top_5_jobs) → GapAnalysis (parallel with step 7)
Step 7: cover_letter_generator.generate_batch(resume, top_3_jobs, tone) → CoverLetters[]
Step 8: MemoryManager.save_session(resume_hash, results)
Step 9: Orchestrator renders: career_report_{date}.md + results.json
Step 10: User sees: ranked jobs, salary ranges, culture scores, gap plan, 3 cover letters
```

**Error handling:**
- HuggingFace unavailable → TF-IDF fallback for matching; skip sentiment scoring
- LLM API unavailable → template-merge fallback for cover letters
- Job fetch fails → use cached results from last 24h in memory

---

## SECOND-KNOWLEDGE-BRAIN.md Integration

- **Sources:** ArXiv cs.CL + cs.IR + cs.LG, Semantic Scholar ("job recommendation", "resume parsing"), Papers with Code IR leaderboards, HuggingFace Papers daily
- **Update schedule:** Weekly (Sunday 02:00 local); HuggingFace Papers: daily
- **Dedup:** SHA-256 hash of (title + DOI); stored in `knowledge_hashes` SQLite table
- **Impact:** New embedding model released → agent automatically flags in knowledge brain → maintainer can update model registry

---

## Quality Gates

1. **Semantic match gate:** Top-5 returned jobs have mean cosine similarity ≥ 0.55 with resume
2. **Cover letter gate:** Generated letter is 250–500 words, contains ≥3 JD-specific keywords, no [PLACEHOLDER] strings
3. **Salary gate:** Prediction within ±20% of 3 known salary data points for "Senior SWE, SF Bay Area"
4. **Sentiment gate:** DistilBERT outputs valid label (POSITIVE/NEGATIVE) for ≥90% of review inputs
5. **Gap analysis gate:** ≥3 distinct, actionable improvement items returned for any (resume, JD) pair
6. **Output gate:** Final Markdown report ≥ 500 words; JSON output parseable by `json.loads()`
7. **Privacy gate:** If `PRIVACY_MODE=true`, verify no outbound HTTP calls to non-Ollama endpoints

---

## Test Scenarios (see `tests/test-scenarios.md` for full detail)

1. Semantic match outperforms keyword match for a Python developer targeting Rust roles
2. Cover letter for a junior candidate with tone=formal; verify no hallucinated achievements
3. Salary prediction for "Senior ML Engineer, Remote" — compare against known benchmark
4. Company sentiment for a well-known tech company with 100 mixed reviews
5. Resume gap analysis: fresh graduate vs senior data scientist JD
6. Full pipeline graceful degradation: LLM API key invalid → fallback path
7. Batch processing: 10 resumes × 20 jobs each via REST API
8. Privacy mode: all processing local (Ollama); no external HTTP calls

---

## Key Design Decisions

1. **Sidecar pattern**: The AI layer wraps the upstream job-search logic without modifying its core crawler — upstream updates remain merge-compatible.
2. **MiniLM default, BGE precision mode**: `all-MiniLM-L6-v2` for speed-first (top-20 recall); `bge-large` when `--precision` flag set.
3. **XGBoost for salary**: Interpretable, fast, no GPU needed; outperforms linear regression on small salary datasets.
4. **DistilBERT for sentiment**: 40% smaller than BERT, deployable on CPU; sufficient accuracy for binary/ternary HR sentiment.
5. **FAISS IndexFlatIP**: Exact L2 search sufficient for ≤50K job postings; no approximate index needed at this scale.
6. **Async concurrent execution**: Steps 4a/4b/4c and 6/7 run with `asyncio.gather` → ~40% wall-clock reduction.
7. **Structured JSON output from LLM**: All Claude responses use explicit JSON schema in system prompt → no regex parsing needed.
