"""Conversational RAG pipeline built on LangChain + NVIDIA NIM."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

try:
    import config
except ModuleNotFoundError:
    from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a strict retrieval assistant. Your only job is to answer questions \
about the official LangChain documentation, using ONLY the retrieved context. \
You have NO prior knowledge: you do not remember anything from training data, \
pre-trained weights, general knowledge, or any earlier conversation. \
Everything you know comes exclusively from the context below.

# Retrieved context:
{context}

Follow these rules strictly:

1. THE QUESTION MUST BE ABOUT LANGCHAIN. If the user's question is not about \
LangChain concepts, APIs, or documentation -- for example sports, football, \
weather, politics, recipes, or general trivia -- REFUSE IMMEDIATELY. Do not \
answer it. The documentation sometimes contains worked examples about \
unrelated topics (e.g. an "Euro 2024" data notebook). Those examples are only \
illustrations of the framework; they are NOT grounds for answering an \
off-topic question about the example topic itself.
2. USE ONLY THE CONTEXT ABOVE. Base your ENTIRE answer exclusively on the \
retrieved context. Never use your prior knowledge, training data, or general \
knowledge about LangChain or anything else -- even when you are certain you \
know the answer. Any statement not traceable to the context is a \
hallucination and is forbidden. You MUST NOT silently fill in gaps from \
memory.
3. GROUND EVERY SENTENCE IN THE CONTEXT. Before writing each sentence, \
confirm there is a precise passage in the context that supports it. When you \
use a specific fact, definition, API name, or parameter, cite the passage \
that supports it inline, e.g. "[Source: https://docs.langchain.com/oss/...]". \
If several sentences come from the same source, cite it once per fact group. \
A sentence that cannot be backed by a quoted passage must not appear.
4. REFUSE whenever the context does not answer the question. If the retrieved \
context is empty, does not contain the specific information requested, or is \
irrelevant/off-topic for the question, you MUST say exactly: "I could not \
find information about that in the documentation." and briefly state the \
specific information that was missing. Do not guess, infer, extrapolate, or \
invent an answer.
5. NEVER answer from memory. If you catch yourself reaching for an answer \
that is not explicitly in the context, stop and refuse using the message \
above instead. When in doubt, refuse.
6. ANSWER IMMEDIATELY AND DIRECTLY. The very first sentence of your reply \
must already be part of the answer. Do NOT include any preamble, plan, or \
meta-commentary -- never write phrases such as "Based on the context", \
"According to the context above", "The question asks", "We need to answer", \
"Let me check the context", or any restatement of these instructions. Never \
describe the retrieved documents, never mention the rules, and never say \
"here is my answer". Just give the answer.
7. DO NOT SKIM OVER THE QUESTION. Answer the question completely: cover every \
part of the question (e.g. if it asks for both "how" and "why", or for two \
concepts, address each), restating the relevant context in your own words \
while staying strictly faithful to it.
8. Keep the answer concise (1-3 short paragraphs) and directly address the \
question. Do not add prose, congratulations, or extra examples beyond what \
the context supplies.
"""

NO_INFO_MESSAGE = (
    "I could not find information about that in the documentation. "
    "Please try rephrasing your question or ask about a topic "
    "covered in the LangChain docs."
)

_RELEVANCE_STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "give", "had", "has", "have", "having", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "like", "make", "many",
    "me", "might", "more", "most", "much", "must", "my", "myself", "need",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "please",
    "same", "she", "should", "show", "so", "some", "such", "tell", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "want", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "whose", "why", "will",
    "with", "would", "you", "your", "yours", "yourself", "yourselves",
}

_DOMAIN_TERMS = (
    "langchain", "lcel", "runnable", "runnables", "nvidia", "chroma",
    "langserve", "langsmith", "chatmodel", "chat model", "output parser",
    "outputparser", "vector store", "vectorstore", "embedding", "embeddings",
    "retriever", "retrieval", "ainvoke", "text splitter", "textsplitter",
    "genai", "ai endpoint",
)

_OFF_TOPIC_TOKENS = {
    "football", "soccer", "basketball", "baseball", "cricket", "tennis",
    "hockey", "golf", "rugby", "badminton", "olympic", "olympics", "sports",
    "weather", "forecast", "meteorology", "climate", "politics", "political",
    "election", "elections", "president", "congress", "senate", "government",
    "war", "economy", "inflation", "unemployment", "bitcoin",
    "cryptocurrency", "stock", "stocks", "movie", "movies", "film",
    "celebrity", "celebrities", "gossip", "recipe", "cooking", "restaurant",
    "music", "concert",
}

_OFF_TOPIC_PHRASES = (
    "world cup", "euro 2024", "premier league", "stock market",
    "video game", "video games", "real madrid", "fast food",
)

_JUDGE_CONTEXT_CHARS = 3000

_RELEVANCE_JUDGE_PROMPT = """\
A user asked a question that will be answered ONLY from the LangChain \
documentation.

Decide whether the question should be answered or refused.

Answer YES only if BOTH are true:
- The question is about the LangChain framework (its concepts, APIs, or \
documentation).
- The retrieved context below actually contains the information needed to \
answer it.

Answer NO if:
- The question is off-topic (e.g. about sports, football, weather, politics, \
recipes, or general trivia), even if the context text happens to contain \
related keywords or example data.
- The context cannot answer the question.

Decide based ONLY on the question and context below -- never from your own \
general knowledge.

Respond with exactly one word, YES or NO.

# Question:
{question}

# Retrieved context:
{context}
"""


class RAGPipeline:
    """A grounded conversational RAG pipeline over the LangChain docs."""

    def __init__(self, *, config=config, top_k: int | None = None) -> None:
        self._config = config
        # Safe default for TOP_K
        self.top_k = top_k or int(getattr(config, 'TOP_K', 4))

        self.embeddings = self._build_embeddings()
        self.vector_store = self._load_vector_store()
        self.retriever = self._build_retriever()
        self.llm = self._build_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

    def _build_embeddings(self) -> NVIDIAEmbeddings:
        return NVIDIAEmbeddings(
            model=self._config.EMBEDDING_MODEL,
            nvidia_api_key=self._config.NVIDIA_API_KEY,
        )

    def _load_vector_store(self) -> Chroma:
        if not self._config.CHROMA_PERSIST_DIRECTORY.exists():
            raise FileNotFoundError(
                "Vector store not found. Run `uv run python -m src.ingest` "
                "first to populate the ChromaDB store at "
                f"{self._config.CHROMA_PERSIST_DIRECTORY}."
            )
        return Chroma(
            collection_name=self._config.CHROMA_COLLECTION_NAME,
            persist_directory=str(self._config.CHROMA_PERSIST_DIRECTORY),
            embedding_function=self.embeddings,
        )

    def _build_retriever(self):
        """Build retriever with safe defaults."""
        # Safe default for SEARCH_TYPE
        search_type = getattr(self._config, 'SEARCH_TYPE', 'similarity') or 'similarity'
        if search_type not in ('similarity', 'mmr', 'similarity_score_threshold'):
            search_type = 'similarity'

        kwargs = {"k": self.top_k}
        if search_type == "mmr":
            kwargs.update({
                "fetch_k": int(getattr(self._config, 'FETCH_K', 20)),
                "lambda_mult": float(getattr(self._config, 'LAMBDA_MULT', 0.7)),
            })

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=kwargs,
        )

    def _build_llm(self) -> ChatNVIDIA:
        return ChatNVIDIA(
            model=self._config.LLM_MODEL,
            nvidia_api_key=self._config.NVIDIA_API_KEY,
            temperature=float(getattr(self._config, 'LLM_TEMPERATURE', 0.0)),
        )

    def _retriever_kwargs(self) -> dict:
        kwargs: dict = {"k": self.top_k}
        search_type = getattr(self._config, 'SEARCH_TYPE', 'similarity') or 'similarity'
        if search_type == "mmr":
            kwargs.update(
                {
                    "fetch_k": int(getattr(self._config, 'FETCH_K', 20)),
                    "lambda_mult": float(getattr(self._config, 'LAMBDA_MULT', 0.7)),
                }
            )
        return kwargs

    @staticmethod
    def _normalise_history(chat_history: list[tuple[str, str]]) -> list[Any]:
        turns = chat_history[-getattr(config, 'MAX_HISTORY_TURNS', 5):]
        messages: list[Any] = []
        for human, ai in turns:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        return messages

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    def _retrieve(self, question: str) -> list[Document]:
        """Return self.top_k chunks ranked by similarity to the question."""
        scored: dict[str, tuple[float, Document]] = {}
        try:
            results = self.vector_store.similarity_search_with_score(
                question, k=self.top_k * 2
            )
        except Exception:
            logger.exception("Similarity search failed; falling back to MMR")
            results = []
        for doc, distance in results:
            key = doc.page_content
            if key not in scored or distance < scored[key][0]:
                scored[key] = (distance, doc)

        ranked = sorted(scored.values(), key=lambda t: t[0])
        if len(ranked) < self.top_k:
            try:
                for doc in self.retriever.invoke(question):
                    if len(ranked) >= self.top_k:
                        break
                    key = doc.page_content
                    if key not in scored:
                        ranked.append((1e9, doc))
            except Exception:
                logger.exception("MMR back-fill failed")
        return [doc for _, doc in ranked[: self.top_k]]

    def _is_on_topic(self, question: str, docs: list[Document]) -> bool:
        lowered = question.lower()

        if any(term in lowered for term in _DOMAIN_TERMS):
            return True

        if any(phrase in lowered for phrase in _OFF_TOPIC_PHRASES):
            return False

        keywords = {
            tok
            for tok in re.findall(r"[a-zA-Z]{4,}", lowered)
            if tok not in _RELEVANCE_STOPWORDS
        }
        if keywords & _OFF_TOPIC_TOKENS:
            return False

        return self._llm_judges_relevant(question, docs)

    def _llm_judges_relevant(self, question: str, docs: list[Document]) -> bool:
        context = self._format_docs(docs)
        if len(context) > _JUDGE_CONTEXT_CHARS:
            context = context[:_JUDGE_CONTEXT_CHARS] + "\n...[truncated]"
        judge_prompt = _RELEVANCE_JUDGE_PROMPT.format(
            question=question, context=context
        )
        try:
            response = self.llm.invoke([HumanMessage(content=judge_prompt)]).content
        except Exception:
            logger.exception("Relevance judge failed; proceeding to answer")
            return True
        verdict = (response or "").strip().upper()
        return verdict.startswith("YES")

    def answer(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> tuple[str, list[Document]]:
        chat_history = chat_history or []

        docs = self._retrieve(question)

        if not docs or not self._is_on_topic(question, docs):
            return (NO_INFO_MESSAGE, [])

        context = self._format_docs(docs)
        history_messages = self._normalise_history(chat_history)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            *history_messages,
            HumanMessage(content=question),
        ]
        answer_text = self.llm.invoke(messages).content

        return (answer_text, docs)

    def build_chain(self):
        return self.chain


_pipeline: RAGPipeline | None = None


def get_pipeline(*, config=config, rebuild: bool = False) -> RAGPipeline:
    global _pipeline
    if _pipeline is None or rebuild:
        _pipeline = RAGPipeline(config=config)
    return _pipeline


def test_pipeline(config=config) -> None:
    if not config.NVIDIA_API_KEY:
        print("[test] NVIDIA_API_KEY is not set. Skipping live test.")
        return

    print("[test] Building pipeline...")
    pipeline = get_pipeline(config=config, rebuild=True)

    question = "What is a Runnable and how do I invoke one?"
    print(f"[test] Asking: {question!r}")
    answer, sources = pipeline.answer(question)

    print(f"[test] Answer:\n  {answer}\n")
    print(f"[test] Sources ({len(sources)}):")
    for s in sources[:3]:
        url = s.metadata.get("source", "?")
        print(f"  - {url}")

    history = [(question, answer)]
    answer2, _ = pipeline.answer("Can you give me a short code example?", history)
    print(f"[test] Follow-up answer:\n  {answer2}\n")
    print("[test] OK")


if __name__ == "__main__":
    test_pipeline()