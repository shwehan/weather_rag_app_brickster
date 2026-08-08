"""
Central configuration for the Weather Intelligence pipeline.

Every component (Flask app, notebook, ingestion script) imports its table
names, embedding model and chunking parameters from here so the schema and
the vector dimension can never drift apart between the write path and the
read path.

All values can be overridden with environment variables, which is how
app.yaml configures the deployed Databricks App and how the notebook widgets
feed parameters into a scheduled Job.
"""

import os

# --------------------------------------------------------------------------
# Lakebase tables
# --------------------------------------------------------------------------

WEATHER_DOCUMENTS_TABLE = os.environ.get(
    "WEATHER_DOCUMENTS_TABLE", "weather_documents"
)
WEATHER_EMBEDDINGS_TABLE = os.environ.get(
    "WEATHER_EMBEDDINGS_TABLE", "weather_embeddings"
)

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# pgvector columns are declared as VECTOR(N) and N must match the model's
# output width exactly. Rather than hardcoding 384 in five places, look the
# dimension up from the model name so swapping models only requires changing
# EMBEDDING_MODEL plus re-running the DDL.
_MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


def embedding_dimension(model_name: str = EMBEDDING_MODEL_NAME) -> int:
    """Return the vector width for a supported sentence-transformers model."""
    try:
        return _MODEL_DIMENSIONS[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding model {model_name!r}. Add its output dimension "
            "to _MODEL_DIMENSIONS in config.py, then update the VECTOR(N) "
            "column width in sql/03_create_weather_embeddings.sql to match."
        ) from None


EMBEDDING_DIM = embedding_dimension()

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# NWS narratives are short (a detailedForecast is ~200-400 characters), but a
# combined alert description + instruction can run past 2,000 characters. An
# 800-character window with 100 characters of overlap keeps each chunk well
# inside the 256-token input limit of all-MiniLM-L6-v2 while making sure a
# safety instruction that straddles a boundary still appears whole in at
# least one chunk.

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# --------------------------------------------------------------------------
# Harvesting defaults
# --------------------------------------------------------------------------

DEFAULT_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get(
        "WEATHER_LOCATIONS",
        "Chicago, IL;Austin, TX;Denver, CO;Miami, FL;Seattle, WA",
    ).split(";")
    if loc.strip()
]

# Upper bound on documents written per location per sync call.
DEFAULT_SYNC_LIMIT = int(os.environ.get("WEATHER_SYNC_LIMIT", "50"))

# Retrieval guardrails for POST /weather/search.
MIN_TOP_K = 1
MAX_TOP_K = 20
DEFAULT_TOP_K = 5

# Optional Databricks model serving endpoint used by the RAG summary. When
# unset, /weather/search simply omits the summary instead of failing.
SUMMARY_ENDPOINT = os.environ.get("SUMMARY_MODEL_ENDPOINT", "").strip()
