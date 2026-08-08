"""Shared sentence-transformer model and text chunking utilities."""

import threading
from typing import Iterable


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    """Load the model at most once per application process."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME)
    return _model


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character windows without empty chunks."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end]

        # Prefer ending on whitespace when doing so does not make the chunk tiny.
        if end < len(normalized):
            split_at = chunk.rfind(" ")
            if split_at >= int(chunk_size * 0.6):
                end = start + split_at
                chunk = normalized[start:end]

        chunks.append(chunk.strip())
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    values = list(texts)
    if not values:
        return []
    vectors = get_embedding_model().encode(
        values,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def vector_literal(vector: Iterable[float]) -> str:
    """Return pgvector's text input format without relying on an adapter."""
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"

