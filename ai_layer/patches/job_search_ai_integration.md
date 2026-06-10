# job_search_ai_integration.md — AI Enhancement Layer Architecture

## Deployment Architecture

```
User / Client
    │
    ▼ HTTP (port 8017)
┌─────────────────────────────────────────────┐
│  FastAPI Server (agent/main.py)             │
│                                             │
│  /search          /match                   │
│  /cover-letter    /salary                  │
│  /sentiment       /gap-analysis            │
│  /knowledge/update  /metrics               │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  JobSearchOrchestrator (orchestrator.py)    │
│                                             │
│  asyncio.gather:                            │
│    job_matcher + salary_predictor           │
│    + company_sentiment                      │
│  → merge_scores → top5                     │
│  → resume_analyzer + cover_letter_gen       │
│  → render_report()                         │
└─────────────────────────────────────────────┘
    │          │          │
    ▼          ▼          ▼
Claude API  HuggingFace   XGBoost
(llm_client) (hf_model_mgr) (salary_predictor)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp config/.env.example .env
# Edit .env: set ANTHROPIC_API_KEY

# 3. Run full search
python agent/main.py search resume.txt "senior ML engineer remote"

# 4. Or start REST API server
python agent/main.py serve --port 8017

# 5. Docker deployment
cd docker
docker-compose up -d
```

## REST API Quick Reference

```bash
# Full search
curl -X POST http://localhost:8017/search \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...", "query": "ML engineer", "tone": "formal"}'

# Salary prediction
curl -X POST http://localhost:8017/salary \
  -d '{"title": "Senior ML Engineer", "location": "Remote", "seniority": "senior"}'

# Cover letter
curl -X POST http://localhost:8017/cover-letter \
  -d '{"resume_text": "...", "job_description": "...", "tone": "technical"}'

# Gap analysis
curl -X POST http://localhost:8017/gap-analysis \
  -d '{"resume_text": "...", "job_descriptions": ["jd1...", "jd2..."]}'

# Trigger knowledge update
curl -X POST http://localhost:8017/knowledge/update
```

## Cross-Agent Integration

### With academic-research-enhanced (Folder 18)
The academic research agent crawls ArXiv/SemanticScholar daily and maintains a curated paper database. The job search agent can consume that database to stay current on semantic matching research:

```python
# In knowledge_updater.py — optional integration with folder 18
import requests
response = requests.get("http://academic-research-agent:8018/papers?topic=job+matching&limit=5")
new_papers = response.json()["papers"]
for paper in new_papers:
    if not memory.is_known_paper(paper["title"]):
        _append_to_brain([paper])
```

### With ai-benchmark-agent (Folder 22)
Route all LLM API calls through the benchmark middleware to measure cover letter quality, token costs, and hallucination rates:

```python
# In llm_client.py — add benchmark instrumentation
import time
start = time.perf_counter()
result = self._claude_complete(system, user, max_tokens, use_case)
latency_ms = (time.perf_counter() - start) * 1000
# POST metrics to benchmark-agent:8022
```

## Prometheus Metrics Exposed

The FastAPI server exposes `/metrics` endpoint with:

| Metric | Description |
|--------|-------------|
| `job_search_sessions_total` | Total search sessions run |
| `job_match_score_avg` | Rolling average top-5 match score |
| `cover_letters_generated_total` | Total cover letters produced |
| `llm_cost_usd_total` | Cumulative LLM API spend |
| `salary_prediction_mae` | Estimated salary prediction error |

## Production Hardening Checklist

- [ ] Set `PRIVACY_MODE=true` if handling sensitive resumes
- [ ] Mount `./models/` as Docker volume to persist HF model cache
- [ ] Set `ANTHROPIC_API_KEY` as Docker secret (not env var in compose)
- [ ] Enable HTTPS reverse proxy (nginx/caddy) in front of port 8017
- [ ] Rate-limit /search endpoint to 10 req/min per IP
- [ ] Add authentication (Bearer token) to all non-health endpoints in production
- [ ] Monitor `llm_cost_usd_total` with alerting at $50/day threshold
- [ ] Backup `./data/agent_memory.db` daily (SQLite WAL mode recommended)
- [ ] Run `tools/knowledge_updater.py` on weekly cron to keep knowledge base current
- [ ] Pre-download HF models before first startup: `python -c "from tools.hf_model_manager import HFModelManager; HFModelManager().preload(['minilm', 'bge-reranker', 'distilbert-sentiment'])"`
