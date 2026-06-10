# Upstream: MadsLorentzen/ai-job-search

## Upstream Repository
- **URL:** https://github.com/MadsLorentzen/ai-job-search
- **Pinned commit:** `main` HEAD as of 2025-06-09 (pin SHA after first clone)
- **License:** MIT
- **Upstream capabilities:** Job scraping, keyword-based resume matching, basic CLI

## Baseline Capabilities (Upstream)

| Feature | Upstream Implementation |
|---------|------------------------|
| Job matching | TF-IDF keyword cosine similarity |
| Resume parsing | Basic text extraction |
| Output format | Terminal print / text file |
| Cover letters | Not implemented (manual) |
| Salary data | Not implemented |
| Company sentiment | Not implemented |
| Knowledge base | Not implemented |

## Upstream Baseline Metrics

| Metric | Upstream Value | Measurement Method |
|--------|---------------|-------------------|
| Job match P@5 | ~0.42 | Human annotation on 50 (resume, jobs) pairs |
| Cover letter quality | N/A | Not a feature |
| Salary prediction | N/A | Not a feature |
| Company fit score | N/A | Not a feature |

## AI Enhancement Improvement Delta

| Feature | Enhancement | Target Metric |
|---------|-------------|--------------|
| Job matching | `all-MiniLM-L6-v2` bi-encoder + `BGE-reranker-large` cross-encoder | P@5 ≥ 0.67 (+25pp over baseline) |
| Cover letters | Claude API with 4 tone profiles + quality gate | User rating ≥ 4.0/5.0 |
| Salary prediction | XGBoost 12-feature regression | MAE ≤ $8,000 annual |
| Company sentiment | DistilBERT per-sentence classification | Accuracy ≥ 78% vs Glassdoor |
| Knowledge base | Weekly ArXiv+SemanticScholar crawl | Self-improving over time |
| Resume gap analysis | BGE-large embeddings + Claude synthesis | ≥5 actionable gaps per analysis |

## Architecture: Sidecar Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│  ai-job-search-enhanced (this repo)                             │
│                                                                  │
│  ┌──────────────────────────┐    ┌──────────────────────────┐  │
│  │  Upstream Core           │    │  AI Enhancement Layer    │  │
│  │  (upstream/)             │    │  (agent/ + tools/)       │  │
│  │                          │    │                          │  │
│  │  - Job scraping         │───►│  - Semantic matching     │  │
│  │  - Text extraction      │    │  - LLM cover letters     │  │
│  │  - CLI interface        │    │  - XGBoost salary pred   │  │
│  │                          │    │  - Sentiment scoring     │  │
│  └──────────────────────────┘    │  - Gap analysis          │  │
│                                  │  - Knowledge crawler     │  │
│                                  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle:** The AI layer wraps upstream capabilities without modifying the upstream source code. Upstream updates remain merge-compatible.

## Cloning Instructions

```bash
# Clone the upstream repo
git clone https://github.com/MadsLorentzen/ai-job-search upstream/ai-job-search-upstream

# Record the pinned commit for reproducibility
cd upstream/ai-job-search-upstream
git log --oneline -1 > ../UPSTREAM_PINNED_COMMIT.txt
```
