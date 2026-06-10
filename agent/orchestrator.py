"""
JobSearchOrchestrator — core agent decision loop for ai-job-search-enhanced.
Coordinates all 5 domain modules with async concurrent execution.
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent.memory.memory_manager import MemoryManager
from agent.modules.job_matcher import JobMatcher, MatchResult
from agent.modules.cover_letter_generator import CoverLetterGenerator, CoverLetter
from agent.modules.salary_predictor import SalaryPredictor, SalaryPrediction
from agent.modules.company_sentiment import CompanySentimentScorer, CompanySentiment
from agent.modules.resume_analyzer import ResumeAnalyzer, GapAnalysis
from tools.config import get_config, AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class JobPosting:
    id: str
    title: str
    company: str
    location: str
    description: str
    url: str = ""
    salary_hint: Optional[str] = None
    seniority: str = "mid"
    industry: str = "technology"
    company_size: str = "mid"
    remote: bool = False
    years_required: int = 3
    reviews: list[str] = field(default_factory=list)


@dataclass
class ResumeProfile:
    text: str
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: str = ""
    summary: str = ""


class JobFetcher:
    """Fetch job postings from mock data or external sources."""

    MOCK_TITLES = [
        "Senior Machine Learning Engineer", "Data Scientist", "ML Platform Engineer",
        "NLP Engineer", "Computer Vision Engineer", "AI Research Scientist",
        "MLOps Engineer", "Data Engineer", "Backend Engineer (Python/ML)",
        "Full-Stack ML Engineer", "Principal AI Engineer", "Staff Software Engineer",
        "Applied Scientist", "ML Infrastructure Engineer", "Deep Learning Engineer",
        "Research Engineer", "AI Product Engineer", "Recommendation Systems Engineer",
        "Speech Recognition Engineer", "Reinforcement Learning Engineer",
    ]

    MOCK_COMPANIES = [
        "Anthropic", "OpenAI", "Google DeepMind", "Meta AI", "Microsoft Research",
        "Amazon AWS AI", "Apple ML", "Nvidia Research", "Hugging Face", "Scale AI",
    ]

    MOCK_LOCATIONS = ["San Francisco, CA", "New York, NY", "Remote", "Seattle, WA", "Austin, TX"]

    def fetch(self, query: str, n: int = 20) -> list[JobPosting]:
        return self._generate_mock_jobs(query, n)

    def _generate_mock_jobs(self, query: str, n: int) -> list[JobPosting]:
        descriptions = [
            f"We are looking for a {t} to join our team. You will build and deploy ML models at scale, "
            f"work with large datasets, design ML pipelines, collaborate with cross-functional teams, "
            f"and drive state-of-the-art research into production. Requirements: 3+ years ML experience, "
            f"Python, PyTorch/TensorFlow, distributed training, MLOps, model serving, A/B testing."
            for t in self.MOCK_TITLES
        ]
        jobs = []
        for i in range(min(n, len(self.MOCK_TITLES))):
            title = self.MOCK_TITLES[i % len(self.MOCK_TITLES)]
            jobs.append(JobPosting(
                id=f"job_{i:03d}",
                title=title,
                company=self.MOCK_COMPANIES[i % len(self.MOCK_COMPANIES)],
                location=self.MOCK_LOCATIONS[i % len(self.MOCK_LOCATIONS)],
                description=descriptions[i % len(descriptions)],
                url=f"https://example.com/jobs/{i:03d}",
                seniority="senior" if "Senior" in title or "Principal" in title else "mid",
                industry="technology",
                company_size="large",
                remote=(self.MOCK_LOCATIONS[i % len(self.MOCK_LOCATIONS)] == "Remote"),
                years_required=3 + (i % 4),
            ))
        return jobs[:n]


class ResumeParser:
    """Parse resume text into a structured ResumeProfile."""

    SKILL_KEYWORDS = [
        "python", "pytorch", "tensorflow", "jax", "scikit-learn", "pandas", "numpy",
        "sql", "spark", "kafka", "kubernetes", "docker", "aws", "gcp", "azure",
        "machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning",
        "transformer", "bert", "llm", "rag", "mlops", "ci/cd", "git", "linux",
        "java", "go", "rust", "c++", "react", "typescript", "fastapi", "flask",
        "data engineering", "data science", "statistics", "a/b testing",
    ]

    def parse(self, text: str) -> ResumeProfile:
        lower = text.lower()
        skills = [kw for kw in self.SKILL_KEYWORDS if kw in lower]

        exp_years = 0.0
        matches = re.findall(r"(\d+)\+?\s*years?", lower)
        if matches:
            exp_years = float(max(int(m) for m in matches))

        lines = text.split("\n")
        summary = " ".join(lines[:5]) if lines else text[:300]

        education = ""
        for line in lines:
            if any(word in line.lower() for word in ["bachelor", "master", "phd", "b.s.", "m.s.", "university", "college"]):
                education = line.strip()
                break

        return ResumeProfile(
            text=text,
            skills=skills,
            experience_years=exp_years,
            education=education,
            summary=summary[:500],
        )


class JobSearchOrchestrator:
    """
    Core agent decision loop. Coordinates all 5 modules with async concurrent
    execution and a 7-point quality gate system.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()
        db_path = self.config.data_dir / "agent_memory.db"
        self.memory = MemoryManager(db_path=db_path)
        self.job_fetcher = JobFetcher()
        self.resume_parser = ResumeParser()
        self._job_matcher: Optional[JobMatcher] = None
        self._cover_letter_gen: Optional[CoverLetterGenerator] = None
        self._salary_predictor: Optional[SalaryPredictor] = None
        self._company_sentiment: Optional[CompanySentimentScorer] = None
        self._resume_analyzer: Optional[ResumeAnalyzer] = None

    @property
    def job_matcher(self) -> JobMatcher:
        if self._job_matcher is None:
            self._job_matcher = JobMatcher()
        return self._job_matcher

    @property
    def cover_letter_gen(self) -> CoverLetterGenerator:
        if self._cover_letter_gen is None:
            self._cover_letter_gen = CoverLetterGenerator()
        return self._cover_letter_gen

    @property
    def salary_predictor(self) -> SalaryPredictor:
        if self._salary_predictor is None:
            self._salary_predictor = SalaryPredictor()
        return self._salary_predictor

    @property
    def company_sentiment(self) -> CompanySentimentScorer:
        if self._company_sentiment is None:
            self._company_sentiment = CompanySentimentScorer()
        return self._company_sentiment

    @property
    def resume_analyzer(self) -> ResumeAnalyzer:
        if self._resume_analyzer is None:
            self._resume_analyzer = ResumeAnalyzer()
        return self._resume_analyzer

    async def full_search(
        self,
        resume_text: str,
        query: str,
        n_jobs: int = 20,
        tone: str = "formal",
        precision_mode: bool = False,
    ) -> dict[str, Any]:
        session_id = hashlib.md5(f"{resume_text[:100]}{query}{datetime.now().date()}".encode()).hexdigest()[:12]
        logger.info("Starting full_search session=%s query=%r", session_id, query)

        resume = self.resume_parser.parse(resume_text)
        jobs = self.job_fetcher.fetch(query, n=n_jobs)

        match_task = asyncio.create_task(
            self._run_matching(resume, jobs, precision_mode)
        )
        salary_task = asyncio.create_task(
            self._run_salary_predictions(jobs)
        )
        sentiment_task = asyncio.create_task(
            self._run_sentiment_scoring(jobs)
        )

        matches, salaries, sentiments = await asyncio.gather(
            match_task, salary_task, sentiment_task
        )

        ranked = self._merge_scores(matches, salaries, sentiments, jobs)

        top5 = ranked[:5]
        top5_jobs = [jobs[r["job_idx"]] for r in top5]

        gap_task = asyncio.create_task(
            self.analyze_gaps(resume_text, [j.description for j in top5_jobs[:3]])
        )
        letter_task = asyncio.create_task(
            self._generate_top3_letters(resume_text, top5_jobs[:3], tone)
        )

        gap_analysis, letters = await asyncio.gather(gap_task, letter_task)

        report = self._render_report(
            query=query,
            resume=resume,
            ranked_jobs=top5,
            jobs=jobs,
            salaries=salaries,
            sentiments=sentiments,
            gap_analysis=gap_analysis,
            letters=letters,
        )

        self.memory.save_session(session_id, {
            "query": query,
            "n_jobs_fetched": len(jobs),
            "top5_matches": [r["score"] for r in top5],
            "gap_skills_count": len(gap_analysis.missing_skills),
            "letters_generated": len(letters),
        })

        return {
            "session_id": session_id,
            "ranked_jobs": top5,
            "gap_analysis": gap_analysis.__dict__,
            "cover_letters": [l.__dict__ for l in letters],
            "report_markdown": report,
        }

    async def match_only(
        self,
        resume_text: str,
        job_descriptions: list[str],
        precision_mode: bool = False,
    ) -> list[MatchResult]:
        resume = self.resume_parser.parse(resume_text)
        jobs = [
            JobPosting(id=f"jd_{i}", title=f"Job {i}", company="", location="", description=desc)
            for i, desc in enumerate(job_descriptions)
        ]
        return await self._run_matching(resume, jobs, precision_mode)

    async def generate_cover_letter(
        self,
        resume_text: str,
        job_description: str,
        company_name: str = "",
        tone: str = "formal",
    ) -> CoverLetter:
        return await self.cover_letter_gen.generate(
            resume_text=resume_text,
            jd_text=job_description,
            company_name=company_name,
            tone=tone,
        )

    async def analyze_gaps(
        self,
        resume_text: str,
        job_descriptions: list[str],
    ) -> GapAnalysis:
        return await self.resume_analyzer.analyze(resume_text, job_descriptions)

    async def _run_matching(
        self,
        resume: ResumeProfile,
        jobs: list[JobPosting],
        precision_mode: bool,
    ) -> list[MatchResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.job_matcher.match(
                resume_text=resume.text,
                job_descs=[j.description for j in jobs],
                precision_mode=precision_mode,
            )
        )

    async def _run_salary_predictions(self, jobs: list[JobPosting]) -> list[SalaryPrediction]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [self.salary_predictor.predict({
                "title": j.title,
                "location": j.location,
                "company_size": j.company_size,
                "seniority": j.seniority,
                "industry": j.industry,
                "remote": j.remote,
                "years_required": j.years_required,
            }) for j in jobs]
        )

    async def _run_sentiment_scoring(self, jobs: list[JobPosting]) -> list[CompanySentiment]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [self.company_sentiment.score(j.company, j.reviews) for j in jobs]
        )

    async def _generate_top3_letters(
        self, resume_text: str, jobs: list[JobPosting], tone: str
    ) -> list[CoverLetter]:
        tasks = [
            self.cover_letter_gen.generate(
                resume_text=resume_text,
                jd_text=j.description,
                company_name=j.company,
                tone=tone,
            )
            for j in jobs
        ]
        return list(await asyncio.gather(*tasks))

    def _merge_scores(
        self,
        matches: list[MatchResult],
        salaries: list[SalaryPrediction],
        sentiments: list[CompanySentiment],
        jobs: list[JobPosting],
    ) -> list[dict]:
        if not matches:
            return []

        max_salary = max((s.median for s in salaries), default=200000)

        ranked = []
        for i, match in enumerate(matches):
            salary_pct = salaries[i].median / max_salary if max_salary > 0 else 0.5
            sentiment_norm = sentiments[i].overall / 5.0

            final_score = (
                0.5 * match.score
                + 0.25 * salary_pct
                + 0.25 * sentiment_norm
            )
            ranked.append({
                "job_idx": i,
                "title": jobs[i].title if i < len(jobs) else match.title,
                "company": jobs[i].company if i < len(jobs) else "",
                "location": jobs[i].location if i < len(jobs) else "",
                "match_score": round(match.score, 3),
                "salary_median": salaries[i].median,
                "salary_range": f"${salaries[i].low:,}-${salaries[i].high:,}",
                "culture_score": round(sentiments[i].overall, 2),
                "score": round(final_score, 4),
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def _render_report(
        self,
        query: str,
        resume: ResumeProfile,
        ranked_jobs: list[dict],
        jobs: list[JobPosting],
        salaries: list[SalaryPrediction],
        sentiments: list[CompanySentiment],
        gap_analysis: GapAnalysis,
        letters: list[CoverLetter],
    ) -> str:
        lines = [
            "# Career Intelligence Report",
            f"**Search Query:** {query}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
            "## Top Matched Positions",
            "",
            "| Rank | Title | Company | Match | Salary Range | Culture |",
            "|------|-------|---------|-------|-------------|---------|",
        ]
        for i, r in enumerate(ranked_jobs, 1):
            lines.append(
                f"| {i} | {r['title']} | {r['company']} | {r['match_score']:.0%} "
                f"| {r['salary_range']} | {r['culture_score']}/5 |"
            )

        lines += [
            "",
            "---",
            "",
            "## Skill Gap Analysis",
            "",
            f"**Overall Match:** {gap_analysis.match_percentage}%  ",
            f"**Time to Close Gaps:** {gap_analysis.time_estimate}",
            "",
            "### Missing Skills (Priority Ranked)",
        ]
        for gap in gap_analysis.missing_skills[:8]:
            lines.append(f"- **[{gap.get('priority','-')}]** {gap.get('skill','?')} - {gap.get('reason','')}")

        lines += [
            "",
            "### 90-Day Learning Path",
        ]
        for step in gap_analysis.learning_path[:6]:
            lines.append(
                f"{step.get('step','?')}. **{step.get('action','?')}** "
                f"({step.get('weeks','?')}w) - {step.get('resource','')}"
            )

        lines += [
            "",
            "---",
            "",
            "## Cover Letters",
        ]
        for i, (letter, job_info) in enumerate(zip(letters, ranked_jobs), 1):
            lines += [
                "",
                f"### Cover Letter {i}: {job_info['title']} @ {job_info['company']}",
                f"**Subject:** {letter.subject_line}",
                "",
                letter.body,
            ]

        lines += [
            "",
            "---",
            "",
            "## Candidate Strengths",
        ]
        for strength in gap_analysis.strengths:
            lines.append(f"- {strength}")

        return "\n".join(lines)
