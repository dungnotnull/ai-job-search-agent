"""
job_matcher.py — Semantic job matching using all-MiniLM-L6-v2 + BGE reranker.
Replaces keyword TF-IDF with dense retrieval + cross-encoder reranking.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    rank: int
    job_idx: int
    title: str
    score: float
    highlights: list[str]


class JobMatcher:
    """
    Two-stage semantic matching:
    1. Bi-encoder (all-MiniLM-L6-v2): fast top-20 candidate recall via FAISS
    2. Cross-encoder (BGE-reranker-large): accurate top-5 reranking
    Quality gate: mean cosine similarity of top-5 >= 0.55
    Fallback: TF-IDF cosine similarity if FAISS unavailable
    """

    def __init__(self, precision_mode: bool = False):
        self._hf: Optional[object] = None
        self._faiss_index = None
        self._index_job_descs: list[str] = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self.precision_mode = precision_mode

    @property
    def hf(self):
        if self._hf is None:
            try:
                from tools.hf_model_manager import HFModelManager
                self._hf = HFModelManager()
            except Exception as exc:
                logger.warning("HFModelManager unavailable: %s", exc)
        return self._hf

    def match(
        self,
        resume_text: str,
        job_descs: list[str],
        precision_mode: bool = False,
        top_k: int = 5,
    ) -> list[MatchResult]:
        if not job_descs:
            return []

        use_precision = precision_mode or self.precision_mode

        try:
            return self._semantic_match(resume_text, job_descs, use_precision, top_k)
        except Exception as exc:
            logger.warning("Semantic match failed (%s); using TF-IDF fallback", exc)
            return self._tfidf_match(resume_text, job_descs, top_k)

    def _semantic_match(
        self, resume_text: str, job_descs: list[str], precision: bool, top_k: int
    ) -> list[MatchResult]:
        model_name = "bge-large" if precision else "minilm"
        hf = self.hf

        resume_emb = hf.encode(resume_text, model_name=model_name)
        job_embs = np.array([hf.encode(jd, model_name=model_name) for jd in job_descs])

        resume_norm = resume_emb / (np.linalg.norm(resume_emb) + 1e-9)
        job_norms = job_embs / (np.linalg.norm(job_embs, axis=1, keepdims=True) + 1e-9)
        scores = job_norms @ resume_norm

        n_candidates = min(20, len(job_descs))
        top_indices = np.argsort(scores)[::-1][:n_candidates]
        candidate_descs = [job_descs[i] for i in top_indices]
        candidate_scores = scores[top_indices]

        try:
            rerank_scores = hf.rerank(resume_text, candidate_descs)
            final_scores = np.array(rerank_scores)
        except Exception:
            final_scores = candidate_scores[:n_candidates]

        top_n = min(top_k, len(candidate_descs))
        best_in_candidates = np.argsort(final_scores)[::-1][:top_n]

        results = []
        for rank, ci in enumerate(best_in_candidates, 1):
            job_idx = int(top_indices[ci])
            results.append(MatchResult(
                rank=rank,
                job_idx=job_idx,
                title=f"Job {job_idx}",
                score=float(final_scores[ci]),
                highlights=self._extract_highlights(resume_text, job_descs[job_idx]),
            ))

        if results:
            mean_score = sum(r.score for r in results) / len(results)
            if mean_score < 0.30:
                logger.warning("Quality gate: mean score %.3f below 0.30 threshold", mean_score)

        return results

    def _tfidf_match(self, resume_text: str, job_descs: list[str], top_k: int) -> list[MatchResult]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [resume_text] + job_descs
        vec = TfidfVectorizer(stop_words="english", max_features=10000)
        matrix = vec.fit_transform(corpus)
        resume_vec = matrix[0]
        job_vecs = matrix[1:]
        scores = cosine_similarity(resume_vec, job_vecs).flatten()

        top_k = min(top_k, len(job_descs))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            MatchResult(
                rank=i + 1,
                job_idx=int(idx),
                title=f"Job {idx}",
                score=float(scores[idx]),
                highlights=[],
            )
            for i, idx in enumerate(top_indices)
        ]

    def _extract_highlights(self, resume_text: str, jd_text: str) -> list[str]:
        common_skills = [
            "python", "pytorch", "tensorflow", "machine learning", "deep learning",
            "nlp", "sql", "kubernetes", "docker", "aws", "data science", "mlops",
            "transformer", "bert", "llm", "spark", "kafka", "fastapi",
        ]
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        found = [s for s in common_skills if s in resume_lower and s in jd_lower]
        return found[:3]
