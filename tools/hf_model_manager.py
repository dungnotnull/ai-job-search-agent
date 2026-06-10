"""
hf_model_manager.py — HuggingFace model registry and lazy loader.
Models: all-MiniLM-L6-v2, bge-large-en-v1.5, bge-reranker-large,
        distilbert-sst-2, bart-large-cnn.
Lazy loading: download on first use. Idle unload after 600s. CUDA auto-detect.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path(__file__).parent.parent / "models"
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(MODELS_CACHE_DIR))

DEFAULT_IDLE_UNLOAD_SECONDS = 600

MODEL_REGISTRY = {
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence_transformer",
        "dim": 384,
    },
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "type": "sentence_transformer",
        "dim": 1024,
    },
    "bge-reranker": {
        "model_id": "BAAI/bge-reranker-large",
        "type": "cross_encoder",
    },
    "distilbert-sentiment": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "type": "pipeline_sentiment",
    },
    "bart-cnn": {
        "model_id": "facebook/bart-large-cnn",
        "type": "pipeline_summarize",
    },
}

_instance: Optional["HFModelManager"] = None
_lock = threading.Lock()


class HFModelManager:
    """Singleton registry: lazy-load, idle-unload, CUDA auto-detect."""

    def __new__(cls):
        global _instance
        with _lock:
            if _instance is None:
                _instance = super().__new__(cls)
                _instance._initialized = False
        return _instance

    def __init__(self):
        if self._initialized:
            return
        self._models: dict[str, Any] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._device = self._detect_device()
        try:
            from tools.config import get_config
            cfg = get_config()
            self._idle_seconds = cfg.hf_models.get("idle_unload_seconds", DEFAULT_IDLE_UNLOAD_SECONDS)
            custom_cache = cfg.models_dir
            if custom_cache and custom_cache.exists():
                global MODELS_CACHE_DIR
                MODELS_CACHE_DIR = custom_cache
        except Exception:
            self._idle_seconds = DEFAULT_IDLE_UNLOAD_SECONDS
        self._initialized = True
        logger.info("HFModelManager initialized on device=%s", self._device)

    @classmethod
    def _reset(cls):
        """Reset singleton for testing."""
        global _instance
        with _lock:
            _instance = None

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def encode(self, text: str, model_name: str = "minilm") -> np.ndarray:
        model = self._get_model(model_name)
        if model is None:
            return self._numpy_fallback_encode(text, MODEL_REGISTRY.get(model_name, {}).get("dim", 384))
        try:
            emb = model.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)
        except Exception as exc:
            logger.warning("encode failed (%s); using fallback", exc)
            return self._numpy_fallback_encode(text, MODEL_REGISTRY.get(model_name, {}).get("dim", 384))

    def encode_batch(self, texts: list[str], model_name: str = "minilm") -> np.ndarray:
        model = self._get_model(model_name)
        if model is None:
            return np.array([self._numpy_fallback_encode(t, MODEL_REGISTRY.get(model_name, {}).get("dim", 384))
                             for t in texts], dtype=np.float32)
        try:
            embs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            return np.array(embs, dtype=np.float32)
        except Exception as exc:
            logger.warning("encode_batch failed (%s); using fallback", exc)
            dim = MODEL_REGISTRY.get(model_name, {}).get("dim", 384)
            return np.array([self._numpy_fallback_encode(t, dim) for t in texts], dtype=np.float32)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        model = self._get_model("bge-reranker")
        if model is None:
            return self._dot_product_fallback_rerank(query, passages)
        try:
            pairs = [[query, p] for p in passages]
            scores = model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as exc:
            logger.warning("rerank failed (%s); using fallback", exc)
            return self._dot_product_fallback_rerank(query, passages)

    def classify_sentiment(self, texts: list[str]) -> list[float]:
        pipeline = self._get_model("distilbert-sentiment")
        if pipeline is None:
            return [0.5] * len(texts)
        try:
            results = pipeline(texts, truncation=True, max_length=512)
            return [
                r["score"] if r["label"] == "POSITIVE" else (1.0 - r["score"])
                for r in results
            ]
        except Exception as exc:
            logger.warning("classify_sentiment failed (%s); returning neutral", exc)
            return [0.5] * len(texts)

    def summarize(self, text: str, max_length: int = 150) -> str:
        pipeline = self._get_model("bart-cnn")
        if pipeline is None:
            return text[:max_length]
        try:
            result = pipeline(text[:1024], max_length=max_length, min_length=40, do_sample=False)
            return result[0]["summary_text"]
        except Exception as exc:
            logger.warning("summarize failed (%s); returning truncated text", exc)
            return text[:max_length]

    def preload(self, model_names: list[str]):
        for name in model_names:
            self._get_model(name)

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def _get_model(self, name: str) -> Any:
        if name in self._models:
            self._reset_idle_timer(name)
            return self._models[name]
        return self._load_model(name)

    def _load_model(self, name: str) -> Any:
        config = MODEL_REGISTRY.get(name)
        if config is None:
            logger.warning("Unknown model name: %s", name)
            return None

        model_id = config["model_id"]
        model_type = config["type"]

        logger.info("Loading %s (%s)...", name, model_id)
        try:
            if model_type == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_id, cache_folder=str(MODELS_CACHE_DIR),
                                            device=self._device)
            elif model_type == "cross_encoder":
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(model_id, max_length=512, device=self._device)
            elif model_type == "pipeline_sentiment":
                from transformers import pipeline
                model = pipeline("sentiment-analysis", model=model_id,
                                 device=0 if self._device == "cuda" else -1,
                                 model_kwargs={"cache_dir": str(MODELS_CACHE_DIR)})
            elif model_type == "pipeline_summarize":
                from transformers import pipeline
                model = pipeline("summarization", model=model_id,
                                 device=0 if self._device == "cuda" else -1,
                                 model_kwargs={"cache_dir": str(MODELS_CACHE_DIR)})
            else:
                logger.warning("Unknown model type: %s", model_type)
                return None

            self._models[name] = model
            self._reset_idle_timer(name)
            logger.info("Loaded %s successfully", name)
            return model

        except Exception as exc:
            logger.warning("Failed to load %s: %s", name, exc)
            return None

    def _reset_idle_timer(self, name: str):
        if name in self._timers:
            self._timers[name].cancel()
        timer = threading.Timer(self._idle_seconds, self._unload_model, args=(name,))
        timer.daemon = True
        timer.start()
        self._timers[name] = timer

    def _unload_model(self, name: str):
        if name in self._models:
            logger.info("Unloading idle model: %s", name)
            del self._models[name]
        if name in self._timers:
            del self._timers[name]

    def _numpy_fallback_encode(self, text: str, dim: int = 384) -> np.ndarray:
        h = hash(text.lower()[:200])
        rng = np.random.RandomState(abs(h) % (2**31))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def _dot_product_fallback_rerank(self, query: str, passages: list[str]) -> list[float]:
        q_emb = self._numpy_fallback_encode(query)
        scores = []
        for p in passages:
            p_emb = self._numpy_fallback_encode(p)
            scores.append(float(np.dot(q_emb, p_emb)))
        return scores
