# test-scenarios.md — ai-job-search-enhanced

## Scenario 1: Semantic Match vs Keyword Match
**Goal:** Verify semantic matching outperforms TF-IDF for a Python developer targeting ML roles.

**Setup:**
- Resume: Python developer with "PyTorch, scikit-learn, ML pipelines, transformers"
- Jobs: Mix of (a) "ML engineer, PyTorch experience required", (b) "deep learning framework proficiency needed", (c) "accounting manager, QuickBooks, Excel"

**Expected:**
- Job (a) and (b) both score ≥ 0.70 with semantic matching (synonymy captured)
- Job (b) scores ≤ 0.20 with TF-IDF (no keyword overlap with "PyTorch")
- Job (c) scores ≤ 0.10 regardless of method (irrelevant domain)
- Mean top-5 cosine score ≥ 0.55 (quality gate passes)

**Pass criteria:** `mean_semantic_score_top2 > mean_tfidf_score_top2` by ≥ 0.15

---

## Scenario 2: Cover Letter Quality Gate — Formal Tone, Junior Candidate
**Goal:** Verify Claude generates a quality cover letter that passes all quality gates.

**Setup:**
- Resume: 1 year experience, Python, data analysis, some ML coursework
- Job: Junior Data Scientist at a fintech startup
- Tone: formal

**Expected:**
- Letter body: 200–500 words
- Contains at least 3 of: "data", "python", "analysis", "machine learning", "fintech"
- Subject line includes job title
- No [PLACEHOLDER] or fabricated credentials
- `quality_passed = True`

**Pass criteria:** All 4 quality gate checks pass; letter readable by human reviewer

---

## Scenario 3: Salary Prediction Benchmark
**Goal:** Salary predictions for known roles are within ±$15,000 of real-world benchmarks.

**Setup:**
| Title | Location | Seniority | Expected Median |
|-------|----------|-----------|-----------------|
| Senior ML Engineer | San Francisco, CA | senior | ~$220,000 |
| Data Scientist | Remote | mid | ~$140,000 |
| Junior SWE | Austin, TX | junior | ~$105,000 |
| Principal AI Engineer | New York, NY | principal | ~$280,000 |

**Expected:** `abs(predicted_median - expected_median) <= 15000` for each row

**Pass criteria:** 3 of 4 predictions within $15,000 of expected

---

## Scenario 4: Company Sentiment Classification — Mixed Reviews
**Goal:** Verify DistilBERT classifies mixed-sentiment company reviews correctly.

**Setup:**
- Company: "TechCorp" (not in static lookup)
- Reviews: 10 positive ("Great culture, supportive team, excellent WLB")
           10 negative ("Poor management, overworked, toxic environment")
           5 neutral ("standard office, average pay")

**Expected:**
- label = "MIXED"
- culture score: 2.5–3.5
- management score: 2.0–3.5
- overall score: 2.5–3.5

**Pass criteria:** label == "MIXED" AND 2.0 ≤ overall ≤ 4.0

---

## Scenario 5: Resume Gap Analysis — Fresh Graduate vs Senior Data Scientist JD
**Goal:** Identify real skill gaps and produce actionable learning path.

**Setup:**
- Resume: Computer science graduate, Python, numpy, pandas, intro ML course, no production experience
- JD: "Senior Data Scientist: 5+ years, Spark, Kafka, Kubernetes, MLOps, A/B testing, SQL, model serving"

**Expected:**
- missing_skills: ≥ 5 distinct gaps (Spark, Kafka, Kubernetes, MLOps, A/B testing all present)
- Each missing skill has priority (1–5) and reason
- learning_path: ≥ 4 concrete steps with resources
- time_estimate: ≥ 6 months
- match_percentage: ≤ 40% (fresh grad vs senior)

**Pass criteria:** `len(missing_skills) >= 5` AND `match_percentage <= 45`

---

## Scenario 6: Full Pipeline Graceful Degradation — LLM API Key Invalid
**Goal:** Verify agent completes pipeline without crashing when Claude API key is invalid.

**Setup:**
- Set `ANTHROPIC_API_KEY=invalid` and `OPENAI_API_KEY=invalid`
- Run full search with a test resume

**Expected:**
- job_matcher uses TF-IDF fallback (not crash)
- salary_predictor uses heuristic fallback
- company_sentiment uses static lookup
- cover_letter_generator returns fallback template letter (not empty)
- gap_analysis returns keyword set-difference result
- Final report is generated (may be lower quality but not empty)
- No unhandled exception raised

**Pass criteria:** `full_search()` returns a dict with non-empty `report_markdown` field

---

## Scenario 7: REST API Batch Processing
**Goal:** Verify FastAPI handles 10 concurrent search requests.

**Setup:**
- Start server: `python agent/main.py serve`
- Send 10 concurrent POST /search requests (asyncio.gather)
- Each with different resume_text and query

**Expected:**
- All 10 requests return HTTP 200
- Response time per request: p95 < 30 seconds (includes LLM call)
- No request returns HTTP 500

**Pass criteria:** 10/10 requests succeed; no 500 errors

---

## Scenario 8: Privacy Mode — Zero External Calls
**Goal:** Verify no HTTP calls to Anthropic/OpenAI when `PRIVACY_MODE=true`.

**Setup:**
- Set `PRIVACY_MODE=true`
- Start network proxy logging all outbound HTTP
- Run full search pipeline

**Expected:**
- All LLM calls route to `http://localhost:11434` (Ollama)
- Zero requests to `api.anthropic.com` or `api.openai.com`
- HuggingFace models load from local cache (`./models/`) only if pre-downloaded

**Pass criteria:** Proxy log shows zero calls to Anthropic/OpenAI endpoints

---

## Scenario 9: Salary Model Training and Persistence
**Goal:** Verify XGBoost salary model can be trained, saved, and reloaded correctly.

**Setup:**
- Call `salary_predictor.train_and_save(n_samples=1000)`
- Reload in a new `SalaryPredictor()` instance
- Run predictions

**Expected:**
- Model file saved to `./models/salary_xgb.pkl`
- Predictions from loaded model match predictions from just-trained model ±$100

**Pass criteria:** Model file exists; reloaded model predictions within ±$100 of original
