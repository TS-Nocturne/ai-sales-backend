"""
Pinecone vector store and Google Embeddings configuration.

Provides lazy-initialized singletons for:
- Pinecone client and index
- Google Generative AI Embeddings (models/embedding-001, dimension 768)
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_vs_cache: dict = {}

# Embedding model config
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 3072


def _resolve_pinecone_api_key() -> str:
    """Resolve the Pinecone API key from environment variables."""
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise ValueError(
            "No Pinecone API key found. "
            "Set PINECONE_API_KEY in your .env file."
        )
    return key


def _resolve_index_name() -> str:
    """Resolve the Pinecone index name from environment variables."""
    name = os.getenv("INDEX_NAME", "ai-sale")
    return name


def get_embeddings():
    """Return a GoogleGenerativeAIEmbeddings instance (lazy init).

    Uses the same Google API key as the LLM.
    """
    if "embeddings" not in _vs_cache:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        from ai_sales.config.llm import resolve_api_key

        _vs_cache["embeddings"] = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=resolve_api_key(),
        )
    return _vs_cache["embeddings"]


def get_pinecone_client():
    """Return a Pinecone client instance (lazy init)."""
    if "client" not in _vs_cache:
        from pinecone import Pinecone

        _vs_cache["client"] = Pinecone(api_key=_resolve_pinecone_api_key())
    return _vs_cache["client"]


def get_pinecone_index():
    """Return the Pinecone Index object for the sales catalog (lazy init).

    Creates the index automatically if it does not exist.
    """
    if "index" not in _vs_cache:
        from pinecone import ServerlessSpec

        pc = get_pinecone_client()
        index_name = _resolve_index_name()

        # Create the index if it does not exist
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            
            # Wait for the index to be ready
            import time
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(5)

        _vs_cache["index"] = pc.Index(index_name)
    return _vs_cache["index"]
