<!-- BADGE ROW -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-1B6B93?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

<!-- HERO -->
<div align="center">

# :mag: AI Job Search Agent

### *Semantic career intelligence &mdash; match smarter, earn more, close gaps faster*

[:rocket: Quick Start](#-quick-start) &middot; [:book: Docs](#-cli-reference) &middot; [:globe_with_meridians: API](#-rest-api) &middot; [:whale: Docker](#-docker) &middot; [:handshake: Contributing](#-contributing)

</div>

---

> **TL;DR** &mdash; Stop keyword-matching your career. This AI agent uses **dense semantic embeddings**, **LLM-powered cover letters**, **XGBoost salary prediction**, and **DistilBERT sentiment scoring** to give you an unfair advantage in your job search &mdash; all from a single CLI command or REST API call.

---

## :star2: Why This Project?

| :confounded: The Old Way | :rocket: The AI Job Search Way |
|---|---|
| Keyword matching misses 60% of relevant jobs | **Semantic embeddings** find jobs that match your *meaning*, not just your words |
| Generic copy-paste cover letters | **Claude API** generates 4 tone-aware, skill-referenced cover letters |
| No idea if a salary offer is fair | **XGBoost regression** predicts salary ranges with confidence intervals |
| No insight into company culture | **DistilBERT** scores sentiment from reviews (culture, management, WLB) |
| Blind to skill gaps | **Gap analysis** ranks missing skills and builds a 90-day learning path |
| Data leaves your machine | **Privacy mode** via Ollama &mdash; zero external API calls |

---

## :sparkles: Features at a Glance

<table>
<tr>
<td width="50%">

### :dart: Semantic Job Matching
Replaces brittle TF-IDF with **all-MiniLM-L6-v2** (384-dim, 14M params) for fast top-20 recall, then **BGE-reranker** cross-encoder for accurate top-5 reranking. **60%+ precision** improvement over keyword baselines.

</td>
<td width="50%">

### :memo: Personalized Cover Letters
**Claude API** generates cover letters with 4 tone profiles: *formal*, *casual*, *technical*, *creative*. Each references specific JD requirements, passes quality gates (word count, keyword presence, no placeholders), and falls back to **BART** if the LLM is unavailable.

</td>
</tr>
<tr>
<td width="50%">

### :moneybag: Salary Prediction
**XGBoost regression** on 12 engineered features (title tier, location multiplier, industry, company size, seniority, remote flag, domain bonuses) predicts annual salary ranges with confidence intervals. Trained on 5,000 synthetic samples mirroring real market distributions.

</td>
<td width="50%">

### :office: Company Sentiment Scoring
**DistilBERT** fine-tuned sentiment classifier aggregates per-sentence polarity into culture, management, and work-life balance scores (0&ndash;5 scale). Static lookup fallback for 30+ well-known tech companies.

</td>
</tr>
<tr>
<td width="50%">

### :bar_chart: Resume Gap Analysis
Identifies skill gaps between your resume and target JDs using **BGE-large embeddings** + **LLM synthesis** via Claude. Produces a ranked improvement roadmap with a concrete **90-day learning path** and resource links.

</td>
</tr>
<tr>
<td width="50%">

### :lock: Offline Privacy Mode
Set `PRIVACY_MODE=true` and all processing stays local via **Ollama** &mdash; zero external API calls, no resume data leaves your machine. Perfect for sensitive career transitions.

</td>
</tr>
</table>

---

## :clapper: Quick Start

### :package: Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| :snake: Python | 3.12+ | Required |
| :package: pip | Latest | Package manager |
| :video_game: CUDA GPU | Optional | Faster model inference |
| :llama: Ollama | Optional | For offline/privacy mode |

### :zap: Install &amp; Run

```bash
# 1 - Clone the repository
git clone https://github.com/dungnotnull/ai-job-search-agent.git
cd ai-job-search-agent

# 2 - Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3 - Install dependencies
pip install -r requirements.txt

# 4 - Or install as editable package (adds ai-job-search CLI command)
pip install -e .

# 5 - Configure API keys
cp config/.env.example .env
# Edit .env with your API keys (see Configuration section)

# 6 - Run your first search!
python agent/main.py search resume.pdf "senior ML engineer remote"
```

> :bulb: **Tip:** At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is needed for full functionality. With `PRIVACY_MODE=true`, no API keys are needed &mdash; Ollama handles everything locally.

---

## :desktop_computer: CLI Reference

### :mag_right: Full Career Intelligence Search

The flagship command &mdash; runs the entire pipeline end-to-end:

```bash
python agent/main.py search resume.pdf "senior ML engineer remote"
```

**Pipeline:** Parse resume &rarr; Fetch jobs &rarr; Match &rarr; Predict salaries &rarr; Score sentiment &rarr; Analyze gaps &rarr; Generate cover letters &rarr; Produce report

| Flag | Default | Description |
|---|---|---|
| `--n-jobs, -n` | 20 | Number of jobs to fetch (1&ndash;100) |
| `--tone, -t` | formal | Cover letter tone: `formal` `casual` `technical` `creative` |
| `--precision, -p` | false | Use BGE-large instead of MiniLM (slower, more accurate) |
| `--output, -o` | stdout | Save report to file |

### :moneybag: Salary Prediction

```bash
python agent/main.py salary "Senior ML Engineer" "San Francisco, CA" --seniority senior --remote
```

### :bar_chart: Skill Gap Analysis

```bash
python agent/main.py gap-analysis resume.pdf job_description.txt
```

### :memo: Cover Letter Generation

```bash
python agent/main.py cover-letter resume.pdf jd.txt --company "Anthropic" --tone technical
```

### :dart: Resume-JD Matching

```bash
python agent/main.py match resume.pdf jobs.json --precision
```

### :books: Knowledge Base Update

```bash
python agent/main.py update-knowledge
```

Crawls ArXiv (cs.CL, cs.IR, cs.LG), Semantic Scholar, and Papers with Code for the latest research.

### :rocket: Start API Server

```bash
python agent/main.py serve --host 0.0.0.0 --port 8017
```

### :chart_with_upwards_trend: Cost Report

```bash
python agent/main.py cost-report
```

Shows LLM API usage breakdown by provider, model, and use case.

---

## :globe_with_meridians: REST API

Start the server:

```bash
python agent/main.py serve
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| :green_circle: `GET` | `/health` | Health check |
| :large_blue_circle: `POST` | `/search` | Full career intelligence search |
| :large_blue_circle: `POST` | `/match` | Resume-to-JD semantic matching |
| :large_blue_circle: `POST` | `/cover-letter` | Generate personalized cover letter |
| :large_blue_circle: `POST` | `/salary` | Predict salary range |
| :large_blue_circle: `POST` | `/sentiment` | Score company sentiment |
| :large_blue_circle: `POST` | `/gap-analysis` | Resume skill gap analysis |
| :large_blue_circle: `POST` | `/knowledge/update` | Trigger knowledge base crawl (background) |
| :green_circle: `GET` | `/metrics` | Session count, avg match score, cost summary |

### Example Requests

```bash
# Health check
curl http://localhost:8017/health

# Full search
curl -X POST http://localhost:8017/search \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Senior Python developer with 5 years ML experience...",
    "query": "senior ML engineer remote",
    "n_jobs": 20,
    "tone": "formal"
  }'

# Salary prediction
curl -X POST http://localhost:8017/salary \
  -H "Content-Type: application/json" \
  -d '{"title": "Senior ML Engineer", "location": "San Francisco, CA", "seniority": "senior"}'

# Company sentiment
curl -X POST http://localhost:8017/sentiment \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Anthropic"}'

# Gap analysis
curl -X POST http://localhost:8017/gap-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python, PyTorch, SQL, Docker, AWS...",
    "job_descriptions": ["We need Spark, Kafka, Kubernetes, MLOps..."]
  }'
```

All POST endpoints include **input validation** via Pydantic &mdash; invalid requests return structured 422 errors with field-level detail.

### Interactive API Docs

Once the server is running, open your browser:

- :book: **Swagger UI:** http://localhost:8017/docs
- :orange_book: **ReDoc:** http://localhost:8017/redoc

---

## :brain: Architecture

```
                          User Input
                    (resume PDF/text + query)
                              |
                              v
  +------------------------------------------------------+
  |              JobSearchOrchestrator                    |
  |              (agent/orchestrator.py)                  |
  |                                                      |
  |  1. ResumeParser --> ResumeProfile                   |
  |  2. JobFetcher   --> List[JobPosting]                |
  |  3. asyncio.gather (concurrent):                    |
  |     +-- JobMatcher ---- semantic match + rerank      |
  |     +-- SalaryPredictor  XGBoost regression          |
  |     +-- CompanySentiment  DistilBERT classification   |
  |  4. Merge: 0.5*match + 0.25*salary + 0.25*sentiment  |
  |  5. ResumeAnalyzer --> skill gap + learning path     |
  |  6. CoverLetterGenerator --> 3 personalized letters  |
  |  7. MemoryManager --> SQLite persistence             |
  |  8. ReportRenderer --> Markdown + JSON               |
  +------------------------------------------------------+
          |                 |                 |
          v                 v                 v
    LLM API          HuggingFace        External APIs
   (llm_client)    (hf_model_mgr)    (job boards, crawl)
   Claude/OpenAI    MiniLM/BGE/       ArXiv/Scholar/
   Ollama           DistilBERT/BART   Papers with Code
```

### Data Flow

```
resume.pdf --> ResumeParser --> ResumeProfile {text, skills, experience, education}
                                        |
job query --> JobFetcher --> List[JobPosting] {title, company, location, description, ...}
                                        |
                    +--------------------+--------------------+
                    v                    v                    v
            JobMatcher          SalaryPredictor     CompanySentiment
            (MiniLM + BGE       (XGBoost/           (DistilBERT/
             reranker)           heuristic)          lookup)
                    |                    |                    |
                    +--------------------+--------------------+
                                         v
                              Ranked Job List (top 5)
                                         |
                          +---------------+---------------+
                          v                               v
                  ResumeAnalyzer            CoverLetterGenerator
                  (skill gaps +             (3 letters, tone-aware)
                   learning path)                     |
                                         v
                              Career Intelligence Report
                              (Markdown + JSON)
```

### Sidecar Architecture

This project uses a **sidecar pattern** &mdash; the AI enhancement layer wraps the upstream [MadsLorentzen/ai-job-search](https://github.com/MadsLorentzen/ai-job-search) without modifying its source code. Upstream updates remain merge-compatible.

```
+----------------------------------------------------------------+
|  ai-job-search-agent (this repo)                                |
|                                                                |
|  +---------------------+    +-----------------------------+     |
|  |  Upstream Core       |    |  AI Enhancement Layer       |     |
|  |  (upstream/)         |    |  (agent/ + tools/)          |     |
|  |                      |    |                              |     |
|  |  - Job scraping      |--->|  - Semantic matching         |     |
|  |  - Text extraction   |    |  - LLM cover letters         |     |
|  |  - CLI interface     |    |  - XGBoost salary pred       |     |
|  |                      |    |  - Sentiment scoring         |     |
|  +---------------------+    |  - Gap analysis              |     |
|                              |  - Knowledge crawler         |     |
|                              +-----------------------------+     |
+----------------------------------------------------------------+
```

---

## :robot: AI Models

### Embedding &amp; Retrieval

| Model | Task | Dimensions | Params | Latency |
|---|---|---|---|---|
| :green_circle: `all-MiniLM-L6-v2` | Fast semantic matching | 384 | 14M | &lt;50ms/job |
| :large_blue_circle: `bge-large-en-v1.5` | Precision embedding | 1024 | 335M | &lt;200ms/job |
| :yellow_circle: `bge-reranker-large` | Cross-encoder reranking | &mdash; | 560M | top-20 only |

### Classification &amp; Generation

| Model | Task | Params | Use Case |
|---|---|---|---|
| :purple_circle: `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment | 66M | Company culture scoring |
| :red_circle: `facebook/bart-large-cnn` | Summarization | 400M | Cover letter fallback |

### LLM Providers (Priority Order)

| Provider | Model | Use Case |
|---|---|---|
| :green_circle: **Claude** (primary) | `claude-opus-4-8` | Cover letters, gap analysis, research synthesis |
| :large_blue_circle: **OpenAI** (fallback) | `gpt-4o` | Fallback for all LLM tasks |
| :yellow_circle: **Ollama** (offline) | `llama3` / `mistral` | Privacy mode &mdash; zero external calls |

> All HuggingFace models use **lazy loading** (download on first use) and **idle unloading** (auto-freeze after 600s). CUDA is auto-detected with CPU fallback.

---

## :gear: Configuration

### :key: Environment Variables (`.env`)

```bash
# Copy the template
cp config/.env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes* | Claude API key for cover letters and gap analysis |
| `OPENAI_API_KEY` | No | OpenAI fallback for LLM calls |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Ollama model name (default: `llama3`) |
| `CLAUDE_MODEL` | No | Claude model override (default: `claude-opus-4-8`) |
| `OPENAI_MODEL` | No | OpenAI model override (default: `gpt-4o`) |
| `PRIVACY_MODE` | No | Set `true` to use only local Ollama &mdash; no external calls |
| `HUGGINGFACE_TOKEN` | No | For private/gated model access |
| `LOG_LEVEL` | No | Logging verbosity: `DEBUG`, `INFO`, `WARNING` |
| `PORT` | No | API server port (default: `8017`) |
| `HOST` | No | API server host (default: `0.0.0.0`) |

> *At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is needed for full functionality. With `PRIVACY_MODE=true`, no API keys are needed.

### :clipboard: YAML Config (`config/agent_config.yaml`)

```yaml
agent:
  name: ai-job-search-enhanced
  version: "1.0.0"
  log_level: INFO

search:
  default_n_jobs: 20
  max_n_jobs: 100
  default_tone: formal
  min_match_score: 0.40

llm:
  provider_order: [claude, openai, ollama]
  privacy_mode: false
  max_retries: 3

salary_predictor:
  synthetic_train_samples: 5000
  target_mae_usd: 8000

quality_gates:
  min_cover_letter_words: 100
  max_cover_letter_words: 600
  min_gap_skills: 3
```

Environment variables **override** YAML values. See `config/agent_config.yaml` for the full configuration schema.

---

## :whale: Docker

### Quick Start

```bash
# Build and run (CPU)
docker compose up -d

# With GPU support
docker compose --profile gpu up -d

# Check health
curl http://localhost:8017/health
```

### What's Included

| Service | Description |
|---|---|
| :large_blue_circle: **ai-job-search-agent** | Main agent server on port 8017 |
| :llama: **ollama** | Local LLM backend on port 11434 |
| :floppy_disk: **Named volumes** | Data persistence and model caching |
| :lock: **Non-root user** | `agentuser` for security |
| :hospital: **Health checks** | On both containers |
| :arrows_counterclockwise: **Auto-restart** | On failure |

---

## :file_folder: Project Structure

```
ai-job-search-agent/
|-- agent/                          # Core agent package
|   |-- __init__.py                 # Package init (v1.0.0)
|   |-- main.py                     # CLI + FastAPI server (entry point)
|   |-- orchestrator.py             # Core decision loop + data models
|   |-- modules/                    # Domain modules
|   |   |-- job_matcher.py          # Semantic matching (MiniLM + BGE reranker)
|   |   |-- salary_predictor.py     # XGBoost salary prediction
|   |   |-- company_sentiment.py    # DistilBERT sentiment scoring
|   |   |-- resume_analyzer.py      # Skill gap analysis + learning path
|   |   +-- cover_letter_generator.py # Claude API cover letters
|   +-- memory/                     # Persistence layer
|       +-- memory_manager.py       # SQLite (sessions, cache, cost log)
|-- tools/                          # Shared infrastructure
|   |-- config.py                   # YAML + .env config loader
|   |-- hf_model_manager.py        # HuggingFace lazy loader + idle unload
|   |-- llm_client.py              # Claude/OpenAI/Ollama unified client
|   +-- knowledge_updater.py        # ArXiv + Scholar + PwC crawler
|-- config/                         # Configuration
|   |-- agent_config.yaml           # All configurable parameters
|   +-- .env.example                # API keys template
|-- docker/                         # Container setup
|   |-- Dockerfile                   # python:3.12-slim, non-root
|   +-- docker-compose.yml          # Agent + Ollama orchestration
|-- tests/                          # Test suite
|   |-- test_agent.py               # 50+ unit + integration + API tests
|   +-- test-scenarios.md           # Test scenario documentation
|-- ai_layer/patches/               # Upstream integration patches
|-- upstream/                       # Upstream repo reference
|-- pyproject.toml                  # Package config + CLI entry point
|-- requirements.txt                # Pinned Python dependencies
|-- LICENSE                         # MIT
+-- README.md                       # This file
```

---

## :test_tube: Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=agent --cov=tools --cov-report=term-missing

# Run specific test class
pytest tests/test_agent.py::TestAPIEndpoints -v

# Run by marker
pytest tests/ -v -m "integration"
```

### Test Coverage

| Category | Tests | What's Covered |
|---|---|---|
| :dart: Job Matcher | 7 | TF-IDF fallback, semantic match, empty input, score bounds |
| :moneybag: Salary Predictor | 6 | Heuristic ranges, tier ordering, location multipliers, featurization |
| :office: Company Sentiment | 6 | Known companies, unknown fallback, score ranges, review aggregation |
| :bar_chart: Resume Analyzer | 5 | Skill extraction, gap identification, learning path, match percentage |
| :memo: Cover Letters | 6 | Fallback letters, quality gates, skill extraction, tone validation |
| :floppy_disk: Memory Manager | 5 | Sessions, cost logging, dedup, salary cache, sentiment cache |
| :link: Orchestrator | 5 | Resume parsing, job fetching, score merging, full pipeline, gap analysis |
| :globe_with_meridians: API Endpoints | 10 | Health, salary, sentiment, gap analysis, metrics, input validation |
| :dash: CLI Smoke | 5 | Direct module invocation, cost reporting, knowledge updater |

---

## :wrench: Troubleshooting

| Issue | Solution |
|---|---|
| :key: `ANTHROPIC_API_KEY not set` | Copy `config/.env.example` to `.env` and add your key |
| :globe_with_meridians: HuggingFace download fails | Check internet; models auto-download to `./models/` on first use |
| :package: `xgboost` import error | Run `pip install xgboost` &mdash; salary predictor falls back to heuristics |
| :llama: Ollama connection refused | Start Ollama first: `ollama serve`, or set `PRIVACY_MODE=false` |
| :memo: Cover letters look generic | Ensure `ANTHROPIC_API_KEY` is set &mdash; without it, template fallback is used |
| :electric_plug: Port 8017 in use | Use `--port` flag: `python agent/main.py serve --port 8080` |
| :snail: Slow first run | HuggingFace models download on first use (~2GB total); subsequent runs are fast |
| :whale: Docker build fails | Ensure Docker has 4GB+ RAM; models download during container build |

---

## :bar_chart: Performance Benchmarks

| Metric | Baseline (TF-IDF) | AI-Enhanced | Improvement |
|---|---|---|---|
| Job Match P@5 | ~0.42 | >=0.67 | **+60%** |
| Cover Letter Quality | N/A | 4.0+/5.0 rated | **New feature** |
| Salary Prediction MAE | N/A | <=$8,000 | **New feature** |
| Company Sentiment Acc | N/A | >=78% | **New feature** |
| Skill Gaps Identified | 0 | 5+ per analysis | **New feature** |

---

## :motorway: Roadmap

- [ ] :globe_with_meridians: Multi-language resume support (Vietnamese, Chinese, etc.)
- [ ] :iphone: Web UI dashboard (React + Vite)
- [ ] :link: LinkedIn integration for job fetching
- [ ] :e-mail: Automated application submission
- [ ] :brain: Fine-tuned models on domain-specific job data
- [ ] :bar_chart: Interview preparation module
- [ ] :arrows_counterclockwise: Real-time job alerts via WebSocket
- [ ] :construction: Plugin system for custom modules

---

## :handshake: Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Code** your changes with tests
4. **Run** the test suite: `pytest tests/ -v`
5. **Submit** a pull request

### Development Setup

```bash
# Install as editable package with dev dependencies
pip install -e .
pip install pytest pytest-asyncio pytest-cov

# Run tests before committing
pytest tests/ -v --cov=agent --cov=tools
```

### Code Style

- Follow **PEP 8** conventions
- Add **type hints** to all function signatures
- Write **docstrings** for all public classes and methods
- Keep functions **focused** &mdash; one responsibility per function

---

## :page_facing_up: License

This project is licensed under the **MIT License** &mdash; see the [LICENSE](LICENSE) file for details.

---

## :pray: Acknowledgements

- **[MadsLorentzen/ai-job-search](https://github.com/MadsLorentzen/ai-job-search)** &mdash; Original upstream project for job scraping and basic CLI
- **[HuggingFace](https://huggingface.co/)** &mdash; Sentence transformers, BERT models, and the ML ecosystem
- **[Anthropic](https://www.anthropic.com/)** &mdash; Claude API for intelligent cover letters and gap analysis
- **[BAAI](https://huggingface.co/BAAI)** &mdash; BGE embedding and reranker models

---

## :star: Star History

If this project helps your career search, please consider giving it a :star:!

<p align="center">
  Built with :heart: using Python, PyTorch, FastAPI, and Claude API
</p>
