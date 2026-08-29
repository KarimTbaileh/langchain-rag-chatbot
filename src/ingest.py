"""Ingestion pipeline for the LangChain documentation knowledge base.

Flow:
    1. Obtain the official LangChain Python OSS docs markdown corpus
       (``llms-full.txt``, Mintlify LLMs.txt format) -- either download it or
       fall back to a cached local mirror.
    2. Parse the corpus into per-page documents (one ``Document`` per page,
       tagged with its source URL).
    3. Split each page into overlapping semantic chunks
       (``RecursiveCharacterTextSplitter``).
    4. Embed the chunks with NVIDIA embeddings and store them in a persistent
       ChromaDB vector store.
    5. Persist chunk metadata (source, page, chunk id, ...) in SQLite.

The pipeline is idempotent. The ChromaDB embedding step is skipped when the
stored corpus is unchanged and the vector store is already populated, but the
SQLite ``documents`` table (``id``, ``source``, ``chunk_id``,
``page_content``) is cleared and repopulated on every run so the metadata
always reflects the current corpus. Rerunning produces the same state.

Usage:
    uv run python -m src.ingest            # full ingestion
    uv run python -m src.ingest --dry-run  # download + split + pool, no API calls
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import config
except ModuleNotFoundError:  # when invoked as `python -m src.ingest`
    from src import config

USER_AGENT = "rag-chatbot/0.1.0 (+https://github.com/)"

# A page in the corpus looks like::
#   # <page title>
#   Source: <https url>
#
#   <page markdown content>
#
# We split on the ``Source:`` marker to isolate each page.
_PAGE_PATTERN = re.compile(r"^#\s+(.+?)\nSource:\s*(\S+)", re.MULTILINE)


# --------------------------------------------------------------------------- #
# SQLite helpers                                                                #
# --------------------------------------------------------------------------- #
def _connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS corpus_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            page_title TEXT,
            chunk_index INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            corpus_hash TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source TEXT,
            chunk_id TEXT,
            page_content TEXT
        );
        """
    )
    return conn


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM corpus_meta WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES (?, ?)",
        (key, value),
    )


# --------------------------------------------------------------------------- #
# Corpus loading & parsing                                                      #
# --------------------------------------------------------------------------- #
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_corpus(config) -> tuple[str, str]:
    """Return ``(corpus_text, corpus_hash)``.

    Downloads the corpus from the network, caching it to a local mirror. If the
    download fails and a mirror already exists, falls back to the mirror.
    """
    try:
        print(f"[1/5] Downloading LangChain docs corpus...")
        req = urllib.request.Request(config.DOCS_CORPUS_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8")
        config.DOCS_MIRROR_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.DOCS_MIRROR_PATH.write_text(text, encoding="utf-8")
        print(f"      Downloaded {len(text):,} characters.")
        return text, _sha256(text)
    except Exception as exc:  # noqa: BLE001
        if config.DOCS_MIRROR_PATH.exists():
            print(f"[1/5] Download failed ({exc}); using local mirror.")
            text = config.DOCS_MIRROR_PATH.read_text(encoding="utf-8")
            print(f"      Loaded {len(text):,} characters from mirror.")
            return text, _sha256(text)
        raise RuntimeError(
            "Unable to download the docs corpus and no local mirror exists. "
            f"Tried: {config.DOCS_CORPUS_URL}"
        ) from exc


def parse_pages(corpus: str) -> list[Document]:
    """Split the corpus markdown into one :class:`Document` per page."""
    documents: list[Document] = []
    matches = list(_PAGE_PATTERN.finditer(corpus))
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        source = match.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(corpus)
        content = corpus[match.end() : end].strip()
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": source, "page_title": title},
            )
        )
    return documents


# --------------------------------------------------------------------------- #
# Ingestion                                                                    #
# --------------------------------------------------------------------------- #
def collect_chunks(documents: list[Document], config) -> list[Document]:
    """Split each page document into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks: list[Document] = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc.page_content)):
            metadata = {
                **doc.metadata,
                "chunk_index": i,
                "chunk_id": f"{_sha256(doc.metadata['source'])}-chunk-{i:05d}",
            }
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


def _add_batch_with_retry(vector_store, *, texts, metadatas, ids, attempts: int = 4):
    """Add an embedding batch, retrying transient errors, then degrading per-chunk.

    The host model for fallback embeddings can be VLM-capable; the server may
    refuse certain image-heavy passages with a 503. In that case we embed each
    chunk individually, keep the ones that succeed, and skip only the fragments
    the server refuses.

    Returns the list of ``chunk_id`` values that could NOT be stored (empty on
    full success).
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            return []
        except Exception as exc:
            last_exc = exc
            message = str(exc)
            transient = any(
                tag in message for tag in ("[503]", "[429]", "[502]", "[504]", "[500]")
            ) or "image inputs require VLM serving" in message
            if attempt == attempts or not transient:
                break
            wait = min(2**attempt, 32)
            print(f"      Transient error on batch ({message[:70]}); retrying in {wait}s...")
            time.sleep(wait)

    print("      Batch-level embed failed; trying per-chunk (skipping refused fragments)...")
    skipped_ids: list[str] = []
    kept_texts: list[str] = []
    kept_metas: list[dict] = []
    kept_ids: list[str] = []
    for i in range(len(texts)):
        try:
            emb = vector_store._embedding_function.embed_documents([texts[i]])[0]
            kept_texts.append(texts[i])
            kept_metas.append(metadatas[i])
            kept_ids.append(ids[i])
        except Exception as exc:
            skipped_ids.append(ids[i])
            print(f"      Skipping refused chunk {ids[i]} ({str(exc)[:60]})")
    if kept_texts:
        vector_store.add_texts(texts=kept_texts, metadatas=kept_metas, ids=kept_ids)
    if skipped_ids:
        print(f"      Batch degraded: kept {len(kept_ids)}, "
              f"skipped {len(skipped_ids)} refused fragments.")
    return skipped_ids


def ingest(config, *, dry_run: bool = False) -> None:
    # ---- 1. Load corpus ------------------------------------------------ #
    corpus, corpus_hash = load_corpus(config)

    # ---- 2. Parse pages ------------------------------------------------ #
    print("[2/5] Parsing corpus into pages...")
    documents = parse_pages(corpus)
    print(f"      Parsed {len(documents):,} pages.")

    # ---- 3. Split into chunks ------------------------------------------ #
    print("[3/5] Splitting pages into chunks...")
    chunks = collect_chunks(documents, config)
    print(f"      Created {len(chunks):,} chunks.")

    if dry_run:
        print("\n-- DRY RUN: skipping embedding + persistence --")
        _print_chunk_sample(chunks)
        return

    # ---- SQLite connection (also creates the documents table) ---------- #
    conn = _connect_db(config.SQLITE_DB_PATH)
    conn.execute("DELETE FROM documents")

    # ---- 4. Embed & store in ChromaDB (skip only if already up to date) - #
    vector_store = _build_store(config)
    collection_ready = len(vector_store.get(limit=1).get("ids", [])) > 0
    stored_hash = _get_meta(conn, "corpus_hash")

    ids = [c.metadata["chunk_id"] for c in chunks]
    texts = [c.page_content for c in chunks]
    metadatas = [{k: v for k, v in c.metadata.items()} for c in chunks]

    if stored_hash != corpus_hash or not collection_ready:
        print("[4/5] Embedding chunks and storing in ChromaDB...")
        for start in range(0, len(chunks), config.EMBEDDING_BATCH_SIZE):
            end = min(start + config.EMBEDDING_BATCH_SIZE, len(chunks))
            print(
                f"      Embedding chunks {start:,}–{end:,} of {len(chunks):,} "
                f"({start / max(len(chunks), 1) * 100:.1f}%)"
            )
            _add_batch_with_retry(
                vector_store,
                texts=texts[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        print("      Vector store updated.")
        _set_meta(conn, "corpus_hash", corpus_hash)
        _set_meta(conn, "collection_ready", "1")
    else:
        print("[4/5] Vector store already populated; skipping ChromaDB embedding.")
        print("      Re-syncing SQLite metadata for the current corpus.")

    # ---- 5. Persist chunk metadata to SQLite (ALWAYS) ------------------- #
    print("[5/5] Persisting chunk metadata to SQLite...")
    for c in chunks:
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(id, source, chunk_id, page_content) VALUES (?, ?, ?, ?)",
            (
                c.metadata["chunk_id"],
                c.metadata["source"],
                c.metadata["chunk_id"],
                c.page_content,
            ),
        )
    conn.commit()

    n_documents = conn.execute(
        "SELECT count(*) FROM documents"
    ).fetchone()[0]
    print(f"Successfully saved {n_documents} records to SQLite.")
    conn.close()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _build_embeddings(config) -> NVIDIAEmbeddings:
    if not config.NVIDIA_API_KEY:
        print(
            "[WARN] NVIDIA_API_KEY is not set. Embedding will fail; "
            "set it in your .env file.",
            file=sys.stderr,
        )
    return NVIDIAEmbeddings(
        model=config.EMBEDDING_MODEL,
        nvidia_api_key=config.NVIDIA_API_KEY,
    )


def _build_store(config) -> Chroma:
    config.CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=str(config.CHROMA_PERSIST_DIRECTORY),
        embedding_function=_build_embeddings(config),
    )


def _print_chunk_sample(chunks: list[Document]) -> None:
    print("\n  Sample chunks (first 3):")
    for c in chunks[:3]:
        print(f"    - {c.metadata['source']} [chunk {c.metadata['chunk_index']}]: "
              f"{c.page_content[:60]!r}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LangChain docs into ChromaDB + SQLite.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download, parse and split only. No embedding or persistence.",
    )
    args = parser.parse_args()
    ingest(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
