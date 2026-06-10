"""
test_agent.py — Automated unit + integration tests for ai-job-search-enhanced.
Run: pytest tests/test_agent.py -v
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import numpy as np
import pytest

SAMPLE_RESUME = """
John Doe | johndoe@email.com | github.com/johndoe

SKILLS: Python, PyTorch, TensorFlow, scikit-learn, pandas, numpy, SQL, Docker, AWS,
        machine learning, deep learning, NLP, transformer models, FastAPI, MLOps

EXPERIENCE:
  - 3 years Software Engineer at TechCorp — built ML pipelines, deployed models to production
  - Data Science intern at StartupXYZ — NLP text classification, BERT fine-tuning

EDUCATION: B.S. Computer Science, State University, 2021
"""

SAMPLE_JD_ML = """
Senior Machine Learning Engineer at Anthropic
We are seeking a Senior ML Engineer to join our safety team.
Requirements: 3+ years Python, PyTorch, deep learning, transformers, MLOps, Docker, AWS.
Nice to have: reinforcement learning, distributed training, LLM fine-tuning.
"""

SAMPLE_JD_ACCOUNTING = """
Accounting Manager at FinanceCorp
Manage AP/AR processes, QuickBooks, Excel pivot tables, financial reporting.
CPA preferred. 5+ years accounting experience required.
"""


class TestJobMatcher:
    def setup_method(self):
        from agent.modules.job_matcher import JobMatcher
        self.matcher = JobMatcher()

    def test_tfidf_fallback_returns_results(self):
        results = self.matcher._tfidf_match(
            SAMPLE_RESUME,
            [SAMPLE_JD_ML, SAMPLE_JD_ACCOUNTING],
            top_k=2,
        )
        assert len(results) == 2
        assert results[0].score >= 0.0

    def test_tfidf_relevant_job_ranked_higher(self):
        results = self.matcher._tfidf_match(
            SAMPLE_RESUME,
            [SAMPLE_JD_ML, SAMPLE_JD_ACCOUNTING],
            top_k=2,
        )
        ml_idx = next(i for i, r in enumerate(results) if r.job_idx == 0)
        acct_idx = next(i for i, r in enumerate(results) if r.job_idx == 1)
        assert results[ml_idx].score > results[acct_idx].score

    def test_empty_jobs_returns_empty(self):
        results = self.matcher.match(SAMPLE_RESUME, [], top_k=5)
        assert results == []

    def test_match_returns_correct_count(self):
        descs = [SAMPLE_JD_ML, SAMPLE_JD_ACCOUNTING, SAMPLE_JD_ML + " extended", SAMPLE_JD_ML + " v2"]
        results = self.matcher._tfidf_match(SAMPLE_RESUME, descs, top_k=3)
        assert len(results) <= 3

    def test_extract_highlights_finds_shared_skills(self):
        highlights = self.matcher._extract_highlights(SAMPLE_RESUME, SAMPLE_JD_ML)
        assert isinstance(highlights, list)
        assert len(highlights) <= 3

    def test_match_fallback_on_hf_error(self):
        with patch.object(self.matcher, '_semantic_match', side_effect=RuntimeError("HF unavailable")):
            results = self.matcher.match(SAMPLE_RESUME, [SAMPLE_JD_ML, SAMPLE_JD_ACCOUNTING])
            assert len(results) >= 1

    def test_scores_between_0_and_1(self):
        results = self.matcher._tfidf_match(SAMPLE_RESUME, [SAMPLE_JD_ML], top_k=1)
        for r in results:
            assert 0.0 <= r.score <= 1.0


class TestSalaryPredictor:
    def setup_method(self):
        from agent.modules.salary_predictor import SalaryPredictor
        self.predictor = SalaryPredictor()

    def test_heuristic_senior_swe_sf_range(self):
        pred = self.predictor._heuristic_predict({
            "title": "Senior Machine Learning Engineer",
            "location": "San Francisco, CA",
            "seniority": "senior",
            "industry": "technology",
            "company_size": "large",
            "remote": False,
            "years_required": 5,
        })
        assert pred.low < pred.median < pred.high
        assert pred.median >= 150000

    def test_heuristic_junior_lower_than_senior(self):
        senior = self.predictor._heuristic_predict({
            "title": "Senior Engineer", "location": "Remote",
            "seniority": "senior", "industry": "technology",
            "company_size": "mid", "remote": True, "years_required": 5,
        })
        junior = self.predictor._heuristic_predict({
            "title": "Junior Engineer", "location": "Remote",
            "seniority": "junior", "industry": "technology",
            "company_size": "mid", "remote": True, "years_required": 1,
        })
        assert senior.median > junior.median

    def test_remote_vs_sf_heuristic(self):
        sf = self.predictor._heuristic_predict({
            "title": "Software Engineer", "location": "San Francisco, CA",
            "seniority": "mid", "industry": "technology",
            "company_size": "large", "remote": False, "years_required": 3,
        })
        remote = self.predictor._heuristic_predict({
            "title": "Software Engineer", "location": "Remote",
            "seniority": "mid", "industry": "technology",
            "company_size": "large", "remote": True, "years_required": 3,
        })
        assert sf.median > remote.median

    def test_prediction_has_valid_range(self):
        pred = self.predictor.predict({
            "title": "Data Scientist", "location": "New York, NY",
            "seniority": "mid", "industry": "finance",
            "company_size": "large", "remote": False, "years_required": 3,
        })
        assert pred.low > 0
        assert pred.low < pred.median < pred.high
        assert 0.0 < pred.confidence <= 1.0

    def test_featurize_returns_12_features(self):
        features = self.predictor._featurize({
            "title": "ML Engineer", "location": "Remote",
            "seniority": "mid", "industry": "technology",
            "company_size": "mid", "remote": True, "years_required": 3,
        })
        assert len(features) == 12

    def test_synthetic_data_generation(self):
        X, y = self.predictor._generate_synthetic_data(100)
        assert X.shape == (100, 12)
        assert y.shape == (100,)
        assert all(y > 0)


class TestCompanySentiment:
    def setup_method(self):
        from agent.modules.company_sentiment import CompanySentimentScorer
        self.scorer = CompanySentimentScorer()

    def test_known_company_anthropic(self):
        result = self.scorer._lookup_score("Anthropic", [])
        assert result.overall >= 4.0
        assert result.label == "POSITIVE"

    def test_known_company_default_fallback(self):
        result = self.scorer._lookup_score("UnknownCompanyXYZ123", [])
        assert 1.0 <= result.overall <= 5.0
        assert result.label in ("POSITIVE", "NEGATIVE", "MIXED")

    def test_score_returns_all_fields(self):
        result = self.scorer.score("Anthropic")
        assert hasattr(result, "culture")
        assert hasattr(result, "management")
        assert hasattr(result, "wlb")
        assert hasattr(result, "overall")
        assert hasattr(result, "label")

    def test_scores_in_range(self):
        result = self.scorer.score("Google")
        assert 1.0 <= result.culture <= 5.0
        assert 1.0 <= result.management <= 5.0
        assert 1.0 <= result.wlb <= 5.0
        assert 1.0 <= result.overall <= 5.0

    def test_negative_company(self):
        result = self.scorer._lookup_score("x", [])
        assert result.overall < 3.5

    def test_mixed_reviews_distilbert_fallback(self):
        mixed_reviews = (
            ["Great team, good salary, flexible hours!"] * 5
            + ["Terrible management, overworked, no work-life balance at all."] * 5
        )
        with patch.object(self.scorer, 'hf', None):
            result = self.scorer.score("UnknownTestCorp", mixed_reviews)
            assert result.label in ("POSITIVE", "NEGATIVE", "MIXED")


class TestResumeAnalyzer:
    def setup_method(self):
        from agent.modules.resume_analyzer import ResumeAnalyzer
        self.analyzer = ResumeAnalyzer()

    def test_extract_skills_finds_known_keywords(self):
        skills = self.analyzer._extract_skills(SAMPLE_RESUME)
        assert "python" in skills
        assert "docker" in skills

    def test_keyword_gap_identifies_missing_skills(self):
        junior_resume = "Python, pandas, numpy, intro ML course"
        senior_jd = "Spark, Kafka, Kubernetes, MLOps, A/B testing, distributed systems"
        result = self.analyzer._keyword_gap(junior_resume, [senior_jd], [])
        assert len(result.missing_skills) >= 3
        gap_names = {g["skill"] for g in result.missing_skills}
        assert any(kw in gap_names for kw in ["spark", "kafka", "kubernetes", "mlops"])

    def test_match_percentage_low_for_junior_vs_senior(self):
        junior_resume = "Python, pandas, data visualization, junior ML"
        senior_jd = SAMPLE_JD_ML + " 5+ years, Spark, Kubernetes, MLOps, distributed training"
        result = self.analyzer._keyword_gap(junior_resume, [senior_jd], [])
        assert result.match_percentage <= 60

    def test_learning_path_has_steps(self):
        result = self.analyzer._keyword_gap(
            "Python only",
            ["spark, kafka, kubernetes, mlops, aws, docker, sql"],
            [],
        )
        assert len(result.learning_path) >= 1
        for step in result.learning_path:
            assert "step" in step
            assert "action" in step
            assert "weeks" in step

    def test_async_analyze_returns_gap_analysis(self):
        result = asyncio.run(self.analyzer.analyze(
            "Python, pandas",
            ["spark, kafka, kubernetes, docker, mlops"],
        ))
        assert hasattr(result, "missing_skills")
        assert hasattr(result, "match_percentage")


class TestCoverLetterGenerator:
    def setup_method(self):
        from agent.modules.cover_letter_generator import CoverLetterGenerator
        self.gen = CoverLetterGenerator()

    def test_fallback_letter_structure(self):
        letter = self.gen._fallback_letter(
            "Python developer 2 years",
            SAMPLE_JD_ML,
            "Anthropic",
            "formal",
            ["python", "machine learning"],
        )
        assert len(letter.body) > 50
        assert letter.subject_line != ""
        assert letter.tone == "formal"

    def test_quality_gate_passes_valid_letter(self):
        from agent.modules.cover_letter_generator import CoverLetter
        letter = CoverLetter(
            subject_line="Application for ML Engineer",
            body="I am writing to apply for this position. " * 30,
            word_count=120,
        )
        assert self.gen._quality_gate(letter, SAMPLE_JD_ML)

    def test_quality_gate_fails_on_placeholder(self):
        from agent.modules.cover_letter_generator import CoverLetter
        letter = CoverLetter(
            subject_line="Application",
            body="[PLACEHOLDER] text here " * 30,
            word_count=150,
        )
        assert not self.gen._quality_gate(letter, SAMPLE_JD_ML)

    def test_extract_matching_skills(self):
        skills = self.gen._extract_matching_skills(SAMPLE_RESUME, SAMPLE_JD_ML)
        assert isinstance(skills, list)
        assert "python" in skills or "machine learning" in skills

    def test_summarize_resume_truncates(self):
        long_resume = "Engineer\n" * 200
        summary = self.gen._summarize_resume(long_resume)
        assert len(summary) <= 900

    def test_tone_invalid_defaults_to_formal(self):
        async def _run():
            with patch.object(self.gen, 'llm', side_effect=Exception("no llm")):
                letter = await self.gen.generate(
                    SAMPLE_RESUME, SAMPLE_JD_ML, "Anthropic", "invalid_tone"
                )
            return letter
        letter = asyncio.run(_run())
        assert letter.tone == "formal"


class TestMemoryManager:
    def setup_method(self):
        from agent.memory.memory_manager import MemoryManager
        self.tmp = tempfile.mkdtemp()
        self.mem = MemoryManager(db_path=Path(self.tmp) / "test.db")

    def test_save_and_get_session_stats(self):
        self.mem.save_session("sess_001", {
            "query": "ML engineer", "n_jobs_fetched": 20,
            "top5_matches": [0.8, 0.75, 0.7, 0.65, 0.6],
            "gap_skills_count": 5, "letters_generated": 3,
        })
        stats = self.mem.get_session_stats()
        assert stats["total_sessions"] == 1
        assert stats["avg_top5_match"] > 0

    def test_cost_logging(self):
        self.mem.log_llm_cost("claude", "claude-opus-4-8", "cover_letter", 800, 600, 0.057)
        summary = self.mem.get_cost_summary()
        assert summary["total_cost_usd"] > 0

    def test_knowledge_hash_dedup(self):
        assert not self.mem.is_known_paper("Test Paper Title")
        self.mem.mark_paper_known("Test Paper Title", "arxiv")
        assert self.mem.is_known_paper("Test Paper Title")

    def test_salary_cache_roundtrip(self):
        features = {"title": "ML Engineer", "location": "Remote"}
        from agent.modules.salary_predictor import SalaryPrediction
        pred = SalaryPrediction(low=120000, median=150000, high=180000, confidence=0.75)
        self.mem.save_salary_cache(features, pred)
        cached = self.mem.get_salary_cache(features)
        assert cached is not None
        assert cached["median"] == 150000

    def test_sentiment_cache_roundtrip(self):
        from agent.modules.company_sentiment import CompanySentiment
        sentiment = CompanySentiment(
            company="TestCorp", culture=4.0, management=3.8,
            wlb=4.2, overall=4.0, label="POSITIVE"
        )
        self.mem.save_sentiment_cache("TestCorp", sentiment)
        cached = self.mem.get_sentiment_cache("TestCorp")
        assert cached is not None
        assert cached["overall"] == 4.0


class TestOrchestrator:
    def setup_method(self):
        from agent.orchestrator import JobSearchOrchestrator
        self.orc = JobSearchOrchestrator()

    def test_resume_parser_extracts_skills(self):
        profile = self.orc.resume_parser.parse(SAMPLE_RESUME)
        assert "python" in profile.skills
        assert len(profile.summary) > 10

    def test_job_fetcher_returns_jobs(self):
        jobs = self.orc.job_fetcher.fetch("ML engineer", n=10)
        assert len(jobs) == 10
        for j in jobs:
            assert j.title
            assert j.description

    def test_merge_scores_sorts_by_combined_score(self):
        from agent.modules.job_matcher import MatchResult
        from agent.modules.salary_predictor import SalaryPrediction
        from agent.modules.company_sentiment import CompanySentiment
        from agent.orchestrator import JobPosting

        matches = [
            MatchResult(rank=1, job_idx=0, title="ML Engineer", score=0.85, highlights=[]),
            MatchResult(rank=2, job_idx=1, title="Accountant", score=0.15, highlights=[]),
        ]
        salaries = [
            SalaryPrediction(low=180000, median=200000, high=230000, confidence=0.8),
            SalaryPrediction(low=50000, median=65000, high=80000, confidence=0.7),
        ]
        sentiments = [
            CompanySentiment("Anthropic", 4.8, 4.7, 4.5, 4.67, "POSITIVE"),
            CompanySentiment("OldCorp", 2.5, 2.2, 2.8, 2.5, "NEGATIVE"),
        ]
        jobs = [
            JobPosting(id="j0", title="ML Engineer", company="Anthropic", location="SF", description=""),
            JobPosting(id="j1", title="Accountant", company="OldCorp", location="NY", description=""),
        ]
        ranked = self.orc._merge_scores(matches, salaries, sentiments, jobs)
        assert ranked[0]["job_idx"] == 0
        assert ranked[0]["score"] > ranked[1]["score"]

    def test_full_search_returns_report(self):
        result = asyncio.run(self.orc.full_search(
            resume_text=SAMPLE_RESUME,
            query="ML engineer",
            n_jobs=5,
            tone="formal",
        ))
        assert "report_markdown" in result
        assert len(result["report_markdown"]) > 100
        assert "session_id" in result

    def test_match_only_returns_results(self):
        results = asyncio.run(self.orc.match_only(
            resume_text=SAMPLE_RESUME,
            job_descriptions=[SAMPLE_JD_ML, SAMPLE_JD_ACCOUNTING],
        ))
        assert len(results) >= 1

    def test_gap_analysis_returns_skills(self):
        analysis = asyncio.run(self.orc.analyze_gaps(
            resume_text="Python, pandas",
            job_descriptions=["Spark, Kafka, Kubernetes, MLOps"],
        ))
        assert hasattr(analysis, "missing_skills")
        assert len(analysis.missing_skills) >= 1


class TestCLISmoke:
    def test_salary_predictor_cli_direct(self):
        from agent.modules.salary_predictor import SalaryPredictor
        pred = SalaryPredictor().predict({
            "title": "Senior ML Engineer",
            "location": "San Francisco, CA",
            "seniority": "senior",
            "industry": "technology",
            "company_size": "large",
            "remote": False,
            "years_required": 5,
        })
        assert pred.median > 100000

    def test_company_sentiment_cli_direct(self):
        from agent.modules.company_sentiment import CompanySentimentScorer
        result = CompanySentimentScorer().score("Anthropic")
        assert result.overall >= 4.0

    def test_memory_manager_cost_report(self):
        from agent.memory.memory_manager import MemoryManager
        mem = MemoryManager(db_path=Path(tempfile.mkdtemp()) / "t.db")
        mem.log_llm_cost("claude", "claude-opus-4-8", "test", 100, 200, 0.016)
        report = mem.get_cost_summary()
        assert report["total_cost_usd"] > 0

    def test_knowledge_updater_instantiates(self):
        from tools.knowledge_updater import KnowledgeUpdater
        updater = KnowledgeUpdater()
        assert updater is not None

    def test_hf_model_manager_fallback_encode(self):
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        vec = mgr._numpy_fallback_encode("test text", dim=384)
        assert vec.shape == (384,)
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01


class TestAPIEndpoints:
    """FastAPI TestClient integration tests."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from agent.main import app_api
        self.client = TestClient(app_api)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"

    def test_salary_endpoint(self):
        resp = self.client.post("/salary", json={
            "title": "Senior ML Engineer",
            "location": "San Francisco, CA",
            "seniority": "senior",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["low"] > 0
        assert data["low"] < data["median"] < data["high"]

    def test_sentiment_endpoint(self):
        resp = self.client.post("/sentiment", json={
            "company_name": "Anthropic",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall"] >= 4.0
        assert data["label"] == "POSITIVE"

    def test_salary_validation_rejects_empty_title(self):
        resp = self.client.post("/salary", json={
            "title": "",
            "location": "Remote",
        })
        assert resp.status_code == 422

    def test_salary_validation_rejects_bad_company_size(self):
        resp = self.client.post("/salary", json={
            "title": "Engineer",
            "location": "Remote",
            "company_size": "mega",
        })
        assert resp.status_code == 422

    def test_search_validation_rejects_short_resume(self):
        resp = self.client.post("/search", json={
            "resume_text": "short",
            "query": "ML engineer",
        })
        assert resp.status_code == 422

    def test_cover_letter_validation_rejects_bad_tone(self):
        resp = self.client.post("/cover-letter", json={
            "resume_text": "Python developer with 3 years experience",
            "job_description": "We need a Python developer with ML skills",
            "tone": "aggressive",
        })
        assert resp.status_code == 422

    def test_gap_analysis_endpoint(self):
        resp = self.client.post("/gap-analysis", json={
            "resume_text": "Python, pandas, numpy developer",
            "job_descriptions": ["Spark, Kafka, Kubernetes, MLOps, AWS"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "missing_skills" in data
        assert "match_percentage" in data

    def test_metrics_endpoint(self):
        resp = self.client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_sessions" in data
