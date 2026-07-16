"""Frozen embedding and educational-score inference helpers."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Any

import numpy as np


class SentenceEmbedder:
    """Reusable sentence-transformer encoder for the offline action pipeline."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self._torch: Any = torch
        self._model: Any = SentenceTransformer(model_name, device=device)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension reported by the loaded model, defaulting to 384."""

        try:
            dimension = self._model.get_sentence_embedding_dimension()
        except AttributeError:
            dimension = None
        return 384 if dimension is None else int(dimension)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 256,
        max_chars: int = 2000,
    ) -> np.ndarray:
        """Return normalized ``float32`` embeddings for ``texts``."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if len(texts) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        truncated = [_normalize_text(text)[:max_chars] for text in texts]
        with self._torch.no_grad():
            embeddings = self._model.encode(
                truncated,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, np.float32(1.0e-12))
        return (array / norms).astype(np.float32, copy=False)

    def close(self) -> None:
        """Release model memory and clear CUDA cache when available."""

        if hasattr(self, "_model"):
            del self._model
        gc.collect()
        if hasattr(self, "_torch") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class EduScorer:
    """Reusable FineWeb-Edu regression-head scorer."""

    def __init__(
        self,
        model_name: str = "HuggingFaceFW/fineweb-edu-classifier",
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self._torch: Any = torch
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self._model: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()

    def score(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 128,
        max_chars: int = 2000,
    ) -> np.ndarray:
        """Return FineWeb-Edu classifier logits ``[:, 0]`` as ``float32`` scores."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if len(texts) == 0:
            return np.empty((0,), dtype=np.float32)

        scores: list[np.ndarray] = []
        max_length = _safe_tokenizer_max_length(self._tokenizer)

        for start in range(0, len(texts), batch_size):
            batch_texts = [
                _normalize_text(text)[:max_chars] for text in texts[start : start + batch_size]
            ]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with self._torch.no_grad():
                outputs = self._model(**encoded)
                logits = outputs.logits
                head = logits if logits.ndim == 1 else logits[:, 0]
                scores.append(head.detach().float().cpu().numpy().astype(np.float32))

        return np.concatenate(scores, axis=0).astype(np.float32, copy=False)

    def close(self) -> None:
        """Release model/tokenizer memory and clear CUDA cache when available."""

        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
        gc.collect()
        if hasattr(self, "_torch") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def embed_texts(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
    batch_size: int = 256,
    max_chars: int = 2000,
) -> np.ndarray:
    """Embed texts with a frozen sentence-transformer, then free GPU memory."""

    embedder = SentenceEmbedder(model_name=model_name, device=device)
    try:
        return embedder.encode(texts, batch_size=batch_size, max_chars=max_chars)
    finally:
        embedder.close()


def edu_score_texts(
    texts: Sequence[str],
    model_name: str = "HuggingFaceFW/fineweb-edu-classifier",
    device: str = "cuda",
    batch_size: int = 128,
    max_chars: int = 2000,
) -> np.ndarray:
    """Score texts with the FineWeb-Edu classifier, then free GPU memory."""

    scorer = EduScorer(model_name=model_name, device=device)
    try:
        return scorer.score(texts, batch_size=batch_size, max_chars=max_chars)
    finally:
        scorer.close()


def _safe_tokenizer_max_length(tokenizer: Any) -> int:
    raw_max_length = getattr(tokenizer, "model_max_length", 512)
    try:
        max_length = int(raw_max_length)
    except (TypeError, ValueError):
        return 512

    if max_length <= 0 or max_length > 100_000:
        return 512
    return max_length


def _normalize_text(value: object) -> str:
    return "" if value is None else str(value)
