"""Ingest parsed programmes into Chroma (v1, no complex logic).

Pipeline:
  1. load all programme objects from ``data/programmes.json``
     (produced by ``rag/parser.py``);
  2. build child documents -- one per section
     (``rag/document_builder.py``) and one summary per programme
     (``rag/summary_builder.py``);
  3. embed with BAAI/bge-large-en-v1.5;
  4. write into Chroma at ``rag/vectorstore``:
       collection ``programme_sections``  -- 338 section documents (retrieval)
       collection ``programme_summaries`` --  64 summary documents (recommendation)

Run from the repo root:

    python rag/parser.py          # refresh data/programmes.json (optional)
    python rag/ingest.py          # rebuild rag/vectorstore
"""

import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.document_builder import build_documents, get_ids  # noqa: E402
from rag.summary_builder import build_summaries  # noqa: E402

VECTOR_PATH = Path(__file__).resolve().parent / "vectorstore"
PROGRAMMES_JSON = ROOT_DIR / "data" / "programmes.json"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

SECTION_COLLECTION = "programme_sections"
SUMMARY_COLLECTION = "programme_summaries"


def ingest(reset: bool = True) -> dict:
    # 1. load all programme JSONs
    programmes = json.loads(PROGRAMMES_JSON.read_text(encoding="utf-8"))
    print(f"[1/5] Loaded {len(programmes)} programme objects from {PROGRAMMES_JSON}")

    # 2a. build section child documents (one per section)
    section_docs = build_documents(programmes)
    section_ids = get_ids(section_docs)
    print(f"[2/5] Built {len(section_docs)} section documents")

    # 2b. build summary documents (one per programme)
    summary_docs = build_summaries(programmes)
    summary_ids = [d.metadata["id"] for d in summary_docs]
    print(f"[2/5] Built {len(summary_docs)} summary documents")

    # 3. embedding model
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"[3/5] Embedding model: {EMBEDDING_MODEL}")

    # 4. fresh Chroma database in v1
    if reset:
        shutil.rmtree(VECTOR_PATH, ignore_errors=True)
        print(f"[4/5] reset {VECTOR_PATH}")

    # 5a. write section documents
    section_store = Chroma.from_documents(
        documents=section_docs,
        embedding=embedding,
        ids=section_ids,
        collection_name=SECTION_COLLECTION,
        persist_directory=str(VECTOR_PATH),
    )

    # 5b. write summary documents
    summary_store = Chroma.from_documents(
        documents=summary_docs,
        embedding=embedding,
        ids=summary_ids,
        collection_name=SUMMARY_COLLECTION,
        persist_directory=str(VECTOR_PATH),
    )

    print(
        f"[5/5] Chroma ready -> {VECTOR_PATH}\n"
        f"      {SECTION_COLLECTION}: {section_store._collection.count()} docs\n"
        f"      {SUMMARY_COLLECTION}: {summary_store._collection.count()} docs"
    )
    return {
        SECTION_COLLECTION: section_store._collection.count(),
        SUMMARY_COLLECTION: summary_store._collection.count(),
    }


if __name__ == "__main__":
    ingest()
