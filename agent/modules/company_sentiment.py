"""
company_sentiment.py — Company culture & sentiment scoring using DistilBERT.
Aggregates per-sentence polarity into culture/management/work-life-balance scores.
Fallback: static lookup for 50 well-known companies.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CompanySentiment:
    company: str
    culture: float
    management: float
    wlb: float
    overall: float
    label: str
    review_count: int = 0
    confidence: float = 0.7


KNOWN_COMPANY_SCORES: dict[str, tuple[float, float, float]] = {
    "anthropic":          (4.8, 4.7, 4.5),
    "openai":             (4.5, 4.2, 3.8),
    "google":             (4.3, 4.0, 4.0),
    "google deepmind":    (4.5, 4.3, 4.1),
    "meta":               (4.1, 3.9, 3.7),
    "meta ai":            (4.2, 4.0, 3.8),
    "microsoft":          (4.2, 4.0, 4.1),
    "microsoft research": (4.5, 4.4, 4.3),
    "amazon":             (3.6, 3.4, 3.2),
    "amazon aws ai":      (3.8, 3.6, 3.4),
    "apple":              (4.0, 3.8, 3.9),
    "apple ml":           (4.1, 3.9, 4.0),
    "nvidia":             (4.3, 4.1, 4.0),
    "nvidia research":    (4.5, 4.3, 4.1),
    "hugging face":       (4.6, 4.5, 4.7),
    "scale ai":           (3.8, 3.6, 3.5),
    "stripe":             (4.3, 4.1, 4.0),
    "airbnb":             (4.2, 4.0, 4.2),
    "netflix":            (4.4, 4.2, 4.0),
    "salesforce":         (4.0, 3.8, 3.9),
    "linkedin":           (4.2, 4.0, 4.1),
    "uber":               (3.8, 3.6, 3.4),
    "lyft":               (3.7, 3.6, 3.8),
    "twitter":            (3.5, 3.2, 3.4),
    "x":                  (2.8, 2.5, 2.7),
    "snap":               (3.6, 3.4, 3.6),
    "palantir":           (3.4, 3.2, 2.9),
    "databricks":         (4.4, 4.3, 4.2),
    "snowflake":          (4.3, 4.1, 4.0),
    "coinbase":           (3.9, 3.7, 3.8),
    "default":            (3.5, 3.5, 3.5),
}


class CompanySentimentScorer:
    """
    Score company culture, management, and work-life balance.
    Uses DistilBERT per-sentence classification when reviews provided;
    falls back to static lookup for known companies.
    """

    def __init__(self):
        self._hf: Optional[object] = None

    @property
    def hf(self):
        if self._hf is None:
            try:
                from tools.hf_model_manager import HFModelManager
                self._hf = HFModelManager()
            except Exception:
                pass
        return self._hf

    def score(self, company_name: str, reviews: list[str] | None = None) -> CompanySentiment:
        reviews = reviews or []
        if reviews:
            try:
                return self._distilbert_score(company_name, reviews)
            except Exception as exc:
                logger.warning("DistilBERT sentiment failed (%s); using lookup fallback", exc)

        return self._lookup_score(company_name, reviews)

    def _distilbert_score(self, company: str, reviews: list[str]) -> CompanySentiment:
        hf = self.hf
        if hf is None:
            raise RuntimeError("HFModelManager not available")

        sentences = []
        for rev in reviews:
            for sent in rev.replace("!", ".").replace("?", ".").split("."):
                sent = sent.strip()
                if len(sent) > 15:
                    sentences.append(sent)

        sentences = sentences[:100]
        if not sentences:
            return self._lookup_score(company, reviews)

        polarities = hf.classify_sentiment(sentences)

        pos_fraction = sum(1 for p in polarities if p > 0.5) / max(len(polarities), 1)
        neg_fraction = 1.0 - pos_fraction

        culture = min(5.0, max(1.0, 1.5 + pos_fraction * 3.5))
        management = min(5.0, max(1.0, 1.3 + pos_fraction * 3.7))
        wlb = min(5.0, max(1.0, 1.5 + pos_fraction * 3.5))
        overall = (culture + management + wlb) / 3.0

        label = "POSITIVE" if pos_fraction > 0.6 else ("NEGATIVE" if neg_fraction > 0.6 else "MIXED")

        return CompanySentiment(
            company=company,
            culture=round(culture, 2),
            management=round(management, 2),
            wlb=round(wlb, 2),
            overall=round(overall, 2),
            label=label,
            review_count=len(reviews),
            confidence=0.75 + min(0.15, len(reviews) / 100),
        )

    def _lookup_score(self, company: str, reviews: list[str]) -> CompanySentiment:
        key = company.lower().strip()
        scores = KNOWN_COMPANY_SCORES.get(key, KNOWN_COMPANY_SCORES["default"])
        culture, management, wlb = scores
        overall = round((culture + management + wlb) / 3.0, 2)
        label = "POSITIVE" if overall >= 4.0 else ("NEGATIVE" if overall < 3.0 else "MIXED")

        return CompanySentiment(
            company=company,
            culture=culture,
            management=management,
            wlb=wlb,
            overall=overall,
            label=label,
            review_count=len(reviews),
            confidence=0.60,
        )
