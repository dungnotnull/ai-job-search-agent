# CLAUDE.md — ai-job-search-enhanced

## Agent Identity
**Name:** ai-job-search-enhanced  
**Tagline:** Semantic career intelligence — match smarter, earn more, close gaps faster  
**Build Phase:** Phase 0 → Phase 7 (full delivery)  
**Cluster:** E — AI/ML Application Agents & Research Tools  
**Upstream Fork:** https://github.com/MadsLorentzen/ai-job-search (pinned: latest stable commit)

---

## Problem Statement
Traditional job search tools rely on brittle keyword matching that misses semantically equivalent skills (e.g., "PyTorch" ≠ "deep learning framework experience"). Candidates waste hours on poorly-matched applications, write generic cover letters, and accept below-market salaries because they have no benchmarks. This agent replaces keyword matching with dense semantic embeddings (`all-MiniLM-L6-v2`), generates personalized cover letters via Claude API, predicts market salaries using XGBoost regression on real posting features, and produces a ranked list of resume gaps so candidates can close the delta between their profile and target roles — all running autonomously from a single CLI command.

---

## Agent Architecture

```
User Input (resume PDF/text + job search query)
        ↓
┌──────────────────────────────────────────────────────────────┐
│  Orchestrator (agent/orchestrator.py)                        │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  Planner   │→ │  Executor  │→ │  Memory / Context    │   │
│  │ (job query)│  │ (parallel) │  │  (SQLite + FAISS)    │   │
│  └────────────┘  └────────────┘  └──────────────────────┘   │
│        ↕               ↕                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Domain Modules                                       │    │
│  │ job_matcher.py       cover_letter_generator.py       │    │
│  │ salary_predictor.py  company_sentiment.py            │    │
│  │ resume_analyzer.py                                   │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
        ↓              ↓              ↓
   LLM API        HuggingFace     External APIs
  (llm_client)  (hf_model_mgr) (job boards, scrapers)
        ↓
  Career Intelligence Report (Markdown + JSON)
```

---

## Module List (`agent/modules/`)

| File | Responsibility |
|------|---------------|
| `job_matcher.py` | Semantic similarity matching using `all-MiniLM-L6-v2` embeddings + FAISS ANN; replaces keyword matching |
| `cover_letter_generator.py` | Claude API cover letter generation with user-defined tone/style profile; citation-aware personalization |
| `salary_predictor.py` | XGBoost regression on 12 job posting features (title, skills, location, company size, seniority, industry) |
| `company_sentiment.py` | DistilBERT fine-tuned sentiment classifier on company review signals; outputs culture/management score |
| `resume_analyzer.py` | LLM compares resume embedding vs JD embedding → ranked skill gap list with actionable improvement steps |

---

## Tools (`agent/tools/` — NOT to be confused with `tools/`)

No agent-level tools directory; domain tools are the 5 modules above.

---

## HuggingFace Models

| Model ID | Task | Why Chosen |
|----------|------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic job matching | Best speed/quality on SBERT leaderboard; 384-dim; 14M params; runs on CPU |
| `BAAI/bge-large-en-v1.5` | Resume embedding, high-precision retrieval | #1 MTEB Retrieval English; 1024-dim; used when precision > speed |
| `BAAI/bge-reranker-large` | Post-retrieval reranking of job candidates | Cross-encoder; +12pp precision over bi-encoder alone |
| `distilbert-base-uncased-finetuned-sst-2-english` | Company sentiment classification | Fast DistilBERT baseline; fine-tunable on Glassdoor-style reviews |
| `facebook/bart-large-cnn` | Cover letter summarization / TL;DR | Strong extractive+abstractive baseline; 400M params |

---

## LLM API Integration

| Provider | Priority | Use Cases in This Agent |
|----------|----------|------------------------|
| Claude (`claude-opus-4-8`) | Primary | Cover letter generation, resume gap analysis, salary context explanation, research synthesis |
| OpenAI (`gpt-4o`) | Fallback | Multimodal: parse PDF resume screenshots if text extraction fails |
| Ollama (`llama3`) | Offline | Privacy mode: process resume + JD without sending to external APIs |

---

## Knowledge Crawl Sources

| Source | Categories/Queries | Frequency |
|--------|-------------------|-----------|
| ArXiv | cs.CL, cs.IR | Weekly (Sunday 02:00) |
| Semantic Scholar | "semantic job matching", "salary prediction NLP", "resume parsing", "career recommendation system" | Weekly |
| Papers with Code | Information retrieval + NLP leaderboards | Weekly |
| ACM Digital Library | CIKM, SIGKDD HR analytics papers | Weekly |
| HuggingFace Papers | daily digest for sentence transformers | Daily |

---

## Supporting Tools (`tools/`)

| File | Responsibility |
|------|---------------|
| `tools/knowledge_updater.py` | ArXiv cs.CL+cs.IR + Semantic Scholar + Papers with Code → `SECOND-KNOWLEDGE-BRAIN.md`; weekly cron |
| `tools/llm_client.py` | Claude/OpenAI/Ollama unified client with streaming, retry (exp backoff), cost tracking |
| `tools/hf_model_manager.py` | Lazy-loading HF model registry: MiniLM/BGE/DistilBERT/BART; auto-unload after 600s idle; CUDA auto-detect |

---

## Active Development Tasks

- [x] CLAUDE.md — agent identity, architecture
- [x] PROJECT-detail.md — full technical specification
- [x] PROJECT-DEVELOPMENT-PHASE-TRACKING.md — 7-phase roadmap
- [x] SECOND-KNOWLEDGE-BRAIN.md — initial knowledge base + crawl protocol
- [x] agent/main.py — CLI + FastAPI server
- [x] agent/orchestrator.py — core decision loop
- [x] agent/modules/job_matcher.py
- [x] agent/modules/cover_letter_generator.py
- [x] agent/modules/salary_predictor.py
- [x] agent/modules/company_sentiment.py
- [x] agent/modules/resume_analyzer.py
- [x] agent/memory/memory_manager.py
- [x] tools/knowledge_updater.py
- [x] tools/llm_client.py
- [x] tools/hf_model_manager.py
- [x] config/agent_config.yaml
- [x] config/.env.example
- [x] docker/docker-compose.yml
- [x] tests/test-scenarios.md
- [x] tests/test_agent.py
