"""Retrievers for the two Chroma collections (split from the old single retriever).

* section_vectorstore (collection: ``programme_sections``)
    -- factual QA: one document per programme section.

* summary_vectorstore (collection: ``programme_summaries``)
    -- recommendation: one summary document per programme.
"""

import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore")

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

SECTION_COLLECTION = "programme_sections"
SUMMARY_COLLECTION = "programme_summaries"
METADATA_COLLECTION = "programme_metadata"

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

summary_vectorstore = Chroma(
    persist_directory=VECTOR_PATH,
    embedding_function=embedding,
    collection_name=SUMMARY_COLLECTION,
)

section_vectorstore = Chroma(
    persist_directory=VECTOR_PATH,
    embedding_function=embedding,
    collection_name=SECTION_COLLECTION,
)

metadata_vectorstore = Chroma(
    persist_directory=VECTOR_PATH,
    embedding_function=embedding,
    collection_name=METADATA_COLLECTION,
)


def retrieve_summary(query: str, k: int = 5):
    """Recommendation retrieval: whole-programme summaries."""
    return summary_vectorstore.similarity_search(query, k=k)


def retrieve_section(query: str, k: int = 5):
    """Factual retrieval: programme section documents."""
    return section_vectorstore.similarity_search(query, k=k)


def retrieve_metadata(query: str, k: int = 5):
    """Exact-fact retrieval: structured metadata documents (one per programme)."""
    return metadata_vectorstore.similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# Unified tool output format
#
#   summary  -> {type, programme_id, name, context}
#   metadata -> {type, programme_id, field, value}
#   section  -> {type, programme_id, section, context}
# ---------------------------------------------------------------------------


def to_summary(doc) -> dict:
    m = doc.metadata
    return {
        "type": "summary",
        "programme_id": m.get("programme_id"),
        "name": m.get("programme_name"),
        "context": doc.page_content,
    }


def to_section(doc) -> dict:
    m = doc.metadata
    return {
        "type": "section",
        "programme_id": m.get("programme_id"),
        "section": m.get("section"),
        "context": doc.page_content,
    }


def to_metadata(doc) -> dict:
    m = doc.metadata
    return {
        "type": "metadata",
        "programme_id": m.get("programme_id"),
        "field": m.get("field"),
        "value": doc.page_content,
    }


def search_programmes(query: str, k: int = 5) -> str:
    """Backward-compatible formatted section search (used by test/test_rag.py)."""
    docs = retrieve_section(query, k=k)
    return "\n".join(
        f"Programme ID: {d.metadata.get('programme_id')}\nContent: {d.page_content}"
        for d in docs
    )
