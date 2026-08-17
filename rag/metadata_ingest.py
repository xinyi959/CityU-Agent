"""Ingest metadata documents into the dedicated ``programme_metadata`` collection.

Separate from ``rag/ingest.py`` on purpose: the metadata index must not mix
into the section collection. Rebuilding only touches ``programme_metadata``
and leaves ``programme_sections`` / ``programme_summaries`` untouched.

Run from the repo root:

    python rag/metadata_ingest.py
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from rag.metadata_builder import build_metadata_documents  # noqa: E402

VECTOR_PATH = Path(__file__).resolve().parent / "vectorstore"
PROGRAMMES_JSON = ROOT_DIR / "data" / "programmes.json"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION = "programme_metadata"


def ingest_metadata(reset: bool = True) -> int:
    # 1. load programmes + build metadata documents
    programmes = json.loads(PROGRAMMES_JSON.read_text(encoding="utf-8"))
    documents = build_metadata_documents(programmes)
    ids = [d.metadata["id"] for d in documents]
    print(f"[1/3] Loaded {len(programmes)} programme objects")
    print(f"[1/3] Built {len(documents)} metadata documents")

    # 2. embedding
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"[2/3] Embedding model: {EMBEDDING_MODEL}")

    # 3. write into Chroma -- replace only the metadata collection
    if reset:
        client = chromadb.PersistentClient(path=str(VECTOR_PATH))
        existing = {c.name for c in client.list_collections()}
        if COLLECTION in existing:
            client.delete_collection(COLLECTION)
            print(f"[3/3] deleted existing collection '{COLLECTION}'")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        ids=ids,
        collection_name=COLLECTION,
        persist_directory=str(VECTOR_PATH),
    )
    count = vectorstore._collection.count()
    print(f"[3/3] {COLLECTION}: {count} documents -> {VECTOR_PATH}")
    return count


if __name__ == "__main__":
    ingest_metadata()
