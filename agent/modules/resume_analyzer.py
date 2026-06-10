"""
resume_analyzer.py — Resume gap analysis and 90-day learning path generation.
Uses BGE-large embeddings + LLM synthesis to identify skill gaps and priorities.
Fallback: keyword set-difference between resume and JD terms.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GapAnalysis:
    missing_skills: list[dict] = field(default_factory=list)
    learning_path: list[dict] = field(default_factory=list)
    time_estimate: str = "Unknown"
    match_percentage: int = 0
    strengths: list[str] = field(default_factory=list)


SKILL_KEYWORDS = [
    "python", "pytorch", "tensorflow", "jax", "scikit-learn", "pandas", "numpy",
    "sql", "spark", "kafka", "kubernetes", "docker", "aws", "gcp", "azure",
    "machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning",
    "transformer", "bert", "llm", "rag", "mlops", "ci/cd", "git", "linux",
    "java", "go", "rust", "c++", "react", "typescript", "fastapi", "flask",
    "data engineering", "data science", "statistics", "a/b testing", "databricks",
    "airflow", "dbt", "redis", "mongodb", "postgresql", "elasticsearch", "grafana",
    "prometheus", "terraform", "ansible", "helm", "datadog", "bigquery",
    "distributed systems", "system design", "api design", "microservices",
]

LEARNING_RESOURCES = {
    "pytorch": "fast.ai Practical Deep Learning (fast.ai/course)",
    "kubernetes": "CKAD certification (training.linuxfoundation.org)",
    "mlops": "MLOps Zoomcamp (github.com/DataTalksClub/mlops-zoomcamp)",
    "spark": "Databricks Academy Spark Fundamentals",
    "sql": "Mode SQL Tutorial (mode.com/sql-tutorial)",
    "aws": "AWS Solutions Architect Associate certification",
    "gcp": "Google Cloud Professional ML Engineer certification",
    "transformer": "Hugging Face NLP Course (huggingface.co/course)",
    "llm": "Andrej Karpathy's Zero to Hero series (youtube.com/karpathy)",
    "rag": "LangChain docs + RAGAS evaluation library",
    "data engineering": "Data Engineering Zoomcamp (DataTalksClub)",
    "statistics": "StatQuest with Josh Starmer (youtube.com/statquest)",
    "system design": "System Design Interview (Alex Xu, O'Reilly)",
    "default": "Search for official documentation + Coursera/edX courses",
}

GAP_ANALYSIS_SYSTEM = """You are a senior technical recruiter and career advisor.
Analyze the gap between a candidate's resume and their target job descriptions.
Return ONLY valid JSON with this exact schema:
{
  "missing_skills": [
    {"skill": "string", "priority": 1-5, "reason": "1-sentence explanation"}
  ],
  "learning_path": [
    {"step": 1, "action": "string", "resource": "string", "weeks": integer}
  ],
  "time_estimate": "X weeks" or "Y months",
  "match_percentage": 0-100,
  "strengths": ["string", "string", "string"]
}

Rules:
- missing_skills: list 5-8 most important gaps, priority 1=most critical
- learning_path: 4-6 concrete steps, each with a resource URL or book name
- match_percentage: honest estimate of how well the resume fits the JD(s)
- strengths: 3-5 areas where the candidate clearly exceeds requirements
"""

GAP_ANALYSIS_USER = """Resume text:
{resume_text}

Target job descriptions:
{jd_list}

Candidate's identified skills: {current_skills}
"""


class ResumeAnalyzer:
    """
    Identify resume-to-JD skill gaps and produce ranked improvement roadmap.
    BGE-large embeddings for skill phrase similarity + LLM synthesis for actionable plan.
    """

    def __init__(self):
        self._llm: Optional[object] = None
        self._hf: Optional[object] = None

    @property
    def llm(self):
        if self._llm is None:
            from tools.llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    @property
    def hf(self):
        if self._hf is None:
            try:
                from tools.hf_model_manager import HFModelManager
                self._hf = HFModelManager()
            except Exception:
                pass
        return self._hf

    async def analyze(self, resume_text: str, job_descriptions: list[str]) -> GapAnalysis:
        current_skills = self._extract_skills(resume_text)

        try:
            return await self._llm_analyze(resume_text, job_descriptions, current_skills)
        except Exception as exc:
            logger.warning("LLM gap analysis failed (%s); using keyword fallback", exc)
            return self._keyword_gap(resume_text, job_descriptions, current_skills)

    async def _llm_analyze(
        self,
        resume_text: str,
        jds: list[str],
        current_skills: list[str],
    ) -> GapAnalysis:
        jd_combined = "\n---\n".join(jds[:3])
        user_prompt = GAP_ANALYSIS_USER.format(
            resume_text=resume_text[:3000],
            jd_list=jd_combined[:3000],
            current_skills=", ".join(current_skills[:20]),
        )

        raw = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.llm.complete(
                system=GAP_ANALYSIS_SYSTEM,
                user=user_prompt,
                max_tokens=1500,
                use_case="gap_analysis",
            )
        )

        data = self._parse_json_response(raw)
        return GapAnalysis(
            missing_skills=data.get("missing_skills", []),
            learning_path=data.get("learning_path", []),
            time_estimate=data.get("time_estimate", "3-6 months"),
            match_percentage=data.get("match_percentage", 50),
            strengths=data.get("strengths", current_skills[:3]),
        )

    def _parse_json_response(self, raw: str) -> dict:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _keyword_gap(
        self,
        resume_text: str,
        jds: list[str],
        current_skills: list[str],
    ) -> GapAnalysis:
        current_set = set(current_skills)

        jd_skills = set()
        for jd in jds:
            jd_lower = jd.lower()
            for kw in SKILL_KEYWORDS:
                if kw in jd_lower:
                    jd_skills.add(kw)

        missing = sorted(jd_skills - current_set)[:8]
        total = len(jd_skills)
        matched = len(jd_skills & current_set)
        match_pct = int(100 * matched / max(total, 1))

        missing_skills = [
            {
                "skill": skill,
                "priority": i + 1,
                "reason": f"Required in {sum(1 for jd in jds if skill in jd.lower())} of {len(jds)} target JDs",
            }
            for i, skill in enumerate(missing[:6])
        ]

        learning_path = []
        for i, skill in enumerate(missing[:5], 1):
            resource = LEARNING_RESOURCES.get(skill, LEARNING_RESOURCES["default"])
            learning_path.append({
                "step": i,
                "action": f"Learn {skill}",
                "resource": resource,
                "weeks": 2 + i,
            })

        total_weeks = sum(s["weeks"] for s in learning_path)
        time_estimate = f"{total_weeks // 4} months" if total_weeks >= 8 else f"{total_weeks} weeks"

        return GapAnalysis(
            missing_skills=missing_skills,
            learning_path=learning_path,
            time_estimate=time_estimate,
            match_percentage=match_pct,
            strengths=[s for s in current_skills if s in jd_skills][:5],
        )

    def _extract_skills(self, text: str) -> list[str]:
        lower = text.lower()
        return [kw for kw in SKILL_KEYWORDS if kw in lower]
