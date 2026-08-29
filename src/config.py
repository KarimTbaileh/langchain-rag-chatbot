"""Centralized configuration for the RAG chatbot.

All paths and runtime settings live here. Values can be overridden through
environment variables (loaded from a local ``.env`` file, see ``.env.example``)
or by editing the defaults below.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels above this file: src/config.py -> <root>/src
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Environment                                                                    #
# --------------------------------------------------------------------------- #
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# Paths                                                                         #
# --------------------------------------------------------------------------- #
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_PERSIST_DIRECTORY = Path(
    os.getenv("CHROMA_PERSIST_DIRECTORY", PROJECT_ROOT / "chroma_db")
)
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", PROJECT_ROOT / "data" / "metadata.db"))

# --------------------------------------------------------------------------- #
# API keys                                                                      #
# --------------------------------------------------------------------------- #
NVIDIA_API_KEY: str | None = os.getenv("NVIDIA_API_KEY")

# --------------------------------------------------------------------------- #
# LangChain documentation corpus                                                #
# --------------------------------------------------------------------------- #
# Official LangChain Python OSS documentation, served as a single markdown
# corpus (Mintlify LLMs.txt format) built for RAG / LLM use.
DOCS_CORPUS_URL = os.getenv(
    "DOCS_CORPUS_URL",
    "https://docs.langchain.com/oss/python/llms-full.txt",
)
# Local mirror used as a fallback if the network is unavailable.
DOCS_MIRROR_PATH = RAW_DATA_DIR / "langchain_docs_full.txt"

# --------------------------------------------------------------------------- #
# Embeddings & vector store                                                     #
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "langchain_docs")

# --------------------------------------------------------------------------- #
# LLM (NVIDIA NIM) & retrieval                                                  #
# --------------------------------------------------------------------------- #
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
TOP_K = int(os.getenv("TOP_K", "8"))

# Retriever strategy: "mmr" (Maximal Marginal Relevance) returns a diverse set
# of chunks that are still relevant, which typically improves context recall
# while keeping precision sane. fetch_k is the candidate pool sampled before
# re-ranking; lambda_mult mixes relevance (0 = diversity, 1 = pure relevance).
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "mmr")
FETCH_K = int(os.getenv("FETCH_K", "20"))
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", "0.7"))

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# --------------------------------------------------------------------------- #
# Evaluation judge (Ragas)                                                      #
# --------------------------------------------------------------------------- #
# Ragas 0.4.3 drives its metrics with strict structured (JSON/Pydantic)
# prompts. The pipeline's generation model (nemotron-3-nano) is too small to
# satisfy them reliably, so the evaluation uses a larger NIM model as judge.
# Override with EVAL_JUDGE_MODEL in the environment / .env.
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")

# --------------------------------------------------------------------------- #
# Text splitting                                                                #
# --------------------------------------------------------------------------- #
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
