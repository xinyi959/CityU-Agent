"""Retrievers for the three Chroma collections.

* section_vectorstore (collection: ``programme_sections``)
    -- detailed QA: one document per programme section.

* summary_vectorstore (collection: ``programme_summaries``)
    -- recommendation: one summary document per programme.

* metadata_vectorstore (collection: ``programme_metadata``)
    -- exact facts: one metadata document per programme.

Each ``retrieve_*`` returns a list of ``(Document, score)`` pairs so the
retriever nodes can build Evidence objects with a retrieval score.
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
    """Recommendation retrieval: whole-programme summaries (doc, score) pairs."""
    return summary_vectorstore.similarity_search_with_score(query, k=k)


def retrieve_section(query: str, k: int = 5):
    """Detailed-QA retrieval: programme section documents (doc, score) pairs."""
    return section_vectorstore.similarity_search_with_score(query, k=k)


def retrieve_metadata(query: str, k: int = 5):
    """Exact-fact retrieval: metadata documents (doc, score) pairs."""
    return metadata_vectorstore.similarity_search_with_score(query, k=k)


def search_programmes(query: str, k: int = 5) -> str:
    """Backward-compatible formatted section search (used by test/test_rag.py)."""
    pairs = retrieve_section(query, k=k)
    return "\n".join(
        f"Programme ID: {doc.metadata.get('programme_id')}\nContent: {doc.page_content}"
        for doc, _ in pairs
    )
