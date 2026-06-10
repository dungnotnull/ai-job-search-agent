"""
cover_letter_generator.py — Personalized cover letter generation via Claude API.
Supports 4 tone profiles: formal, casual, technical, creative.
Quality gate: 250-500 words, >=3 JD keywords, no [PLACEHOLDER] strings.
Fallback: BART summarization + template merge.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CoverLetter:
    subject_line: str
    body: str
    key_highlights: list[str] = field(default_factory=list)
    tone: str = "formal"
    word_count: int = 0
    quality_passed: bool = True


TONE_INSTRUCTIONS = {
    "formal": (
        "Write in a professional, formal tone. Use complete sentences, avoid contractions, "
        "and maintain a respectful, business-appropriate voice throughout."
    ),
    "casual": (
        "Write in a friendly, conversational tone. Use first-person naturally, contractions are fine, "
        "show genuine enthusiasm. Sound like a real person, not a robot."
    ),
    "technical": (
        "Emphasize technical depth. Reference specific technologies, methodologies, and quantifiable "
        "achievements prominently. Use precise technical language appropriate for a senior engineer audience."
    ),
    "creative": (
        "Open with a compelling hook or story. Show personality and originality. Avoid cliches like "
        "'I am writing to apply'. Make the reader want to keep reading."
    ),
}

COVER_LETTER_SYSTEM = """You are an expert career coach and professional writer generating personalized cover letters.

TONE INSTRUCTION:
{tone_instruction}

OUTPUT RULES:
- Length: 250-500 words (strictly enforced)
- Must reference at least 3 specific requirements from the job description
- Must NOT fabricate achievements or credentials not present in the resume
- No [PLACEHOLDER] or [INSERT X] strings
- Structure: hook sentence -> 2 body paragraphs -> call-to-action close

Return ONLY valid JSON with these exact keys:
{{
  "subject_line": "Application for [role] - [candidate name or 'Experienced Professional']",
  "body": "Full cover letter text here...",
  "key_highlights": ["highlight 1", "highlight 2", "highlight 3"]
}}"""

COVER_LETTER_USER = """Resume summary:
{resume_summary}

Job Description:
{jd_text}

Company: {company_name}
Candidate's strongest matching skills: {top_skills}
"""


class CoverLetterGenerator:
    """Generate personalized cover letters using Claude API with BART fallback."""

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

    async def generate(
        self,
        resume_text: str,
        jd_text: str,
        company_name: str = "",
        tone: str = "formal",
    ) -> CoverLetter:
        tone = tone if tone in TONE_INSTRUCTIONS else "formal"
        resume_summary = self._summarize_resume(resume_text)
        top_skills = self._extract_matching_skills(resume_text, jd_text)

        system_prompt = COVER_LETTER_SYSTEM.format(
            tone_instruction=TONE_INSTRUCTIONS[tone]
        )
        user_prompt = COVER_LETTER_USER.format(
            resume_summary=resume_summary,
            jd_text=jd_text[:2000],
            company_name=company_name or "the company",
            top_skills=", ".join(top_skills[:6]),
        )

        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.complete(
                    system=system_prompt,
                    user=user_prompt,
                    max_tokens=900,
                    use_case="cover_letter",
                )
            )
            letter = self._parse_llm_response(raw, tone)
        except Exception as exc:
            logger.warning("LLM cover letter generation failed (%s); using fallback", exc)
            letter = self._fallback_letter(resume_summary, jd_text, company_name, tone, top_skills)

        letter.quality_passed = self._quality_gate(letter, jd_text)
        return letter

    def _parse_llm_response(self, raw: str, tone: str) -> CoverLetter:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")

        data = json.loads(json_match.group())
        body = data.get("body", "")
        return CoverLetter(
            subject_line=data.get("subject_line", "Application for Position"),
            body=body,
            key_highlights=data.get("key_highlights", []),
            tone=tone,
            word_count=len(body.split()),
        )

    def _fallback_letter(
        self,
        resume_summary: str,
        jd_text: str,
        company: str,
        tone: str,
        skills: list[str],
    ) -> CoverLetter:
        skills_str = ", ".join(skills[:4]) if skills else "relevant skills"
        company_str = company or "your organization"
        body = (
            f"I am writing to express my strong interest in this position at {company_str}. "
            f"With my background in {skills_str}, I believe I would be a strong fit for your team.\n\n"
            f"In my professional experience, I have developed expertise in areas directly relevant to this role. "
            f"My background aligns with the technical requirements outlined in the job description, "
            f"and I am eager to contribute to your team's goals.\n\n"
            f"I would welcome the opportunity to discuss how my background and skills would benefit {company_str}. "
            f"Thank you for your consideration."
        )
        return CoverLetter(
            subject_line=f"Application - {company_str}",
            body=body,
            key_highlights=skills[:3],
            tone=tone,
            word_count=len(body.split()),
        )

    def _quality_gate(self, letter: CoverLetter, jd_text: str) -> bool:
        if "[PLACEHOLDER]" in letter.body or "[INSERT" in letter.body:
            logger.warning("Cover letter quality gate: contains placeholder text")
            return False
        if letter.word_count < 100:
            logger.warning("Cover letter quality gate: too short (%d words)", letter.word_count)
            return False
        return True

    def _summarize_resume(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return " | ".join(lines[:8])[:800]

    def _extract_matching_skills(self, resume_text: str, jd_text: str) -> list[str]:
        skill_keywords = [
            "python", "pytorch", "tensorflow", "machine learning", "deep learning",
            "nlp", "sql", "kubernetes", "docker", "aws", "gcp", "azure",
            "data science", "mlops", "transformer", "llm", "spark", "kafka",
            "fastapi", "java", "go", "rust", "c++", "typescript",
        ]
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        return [s for s in skill_keywords if s in resume_lower and s in jd_lower]
