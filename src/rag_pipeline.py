"""Conversational RAG pipeline built on LangChain + NVIDIA NIM.

Loads the persistent ChromaDB vector store, creates a retriever, and runs a
grounded question-answering chain using an NVIDIA NIM LLM. The chain:

    * retrieves relevant context for the question,
    * incorporates prior chat history,
    * answers strictly from the retrieved context (refusing to hallucinate),
    * returns both the answer and the source documents,
    * gracefully handles the case where no relevant context is found.

The module exposes a :class:`RAGPipeline` that is convenient to call from a
Gradio interface::

    pipeline = RAGPipeline()
    answer, sources = pipeline.answer("What is a Runnable?")

A simple self-test is provided at the bottom of the module.
"""

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
except ModuleNotFoundError:  # when invoked as `python -m src.rag_pipeline`
    from src import config

logger = logging.getLogger(__name__)

# Prompt that enforces strict grounding. Instructs the model to answer ONLY
# about the official LangChain docs and to REFUSE everything else, including
# questions the context merely "mentions" via unrelated example data. It also
# enforces a direct, citation-backed answer format: weak models are prone to
# rambling meta-commentary ("We need to answer using only the context..."),
# which is explicitly forbidden.
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

Example of good behaviour (formatting guide, do not copy the text):
Question: "How does conversation history work in LangChain?"
Good answer: "LangChain provides memory abstractions for remembering \
information about previous interactions. Short-term memory stores messages per \
conversation thread, and the ChatMessageHistory class keeps a record of the \
messages in a conversation, which can be passed back into a chain so follow-up \
questions have the needed context. [Source: \
https://docs.langchain.com/oss/python/concepts/memory]"
"""

# Message returned (without calling the LLM) when retrieval yields no
# relevant context.
NO_INFO_MESSAGE = (
    "I could not find information about that in the documentation. "
    "Please try rephrasing your question or ask about a topic "
    "covered in the LangChain docs."
)

# Small English function-word set used by the cheap keyword-overlap relevance
# gate in :meth:`RAGPipeline.answer`.
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

# Terms (substring-matched on the lowercased question) that clearly indicate a
# question is about the LangChain ecosystem. Used as the fast path in
# :meth:`RAGPipeline._is_on_topic`.
_DOMAIN_TERMS = (
    "langchain", "lcel", "runnable", "runnables", "nvidia", "chroma",
    "langserve", "langsmith", "chatmodel", "chat model", "output parser",
    "outputparser", "vector store", "vectorstore", "embedding", "embeddings",
    "retriever", "retrieval", "ainvoke", "text splitter", "textsplitter",
    "genai", "ai endpoint",
)

# Single tokens (matched against the question's keywords) that flag a clearly
# off-topic topic. Any match refuses the question without calling the LLM.
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

# Multi-word phrases (substring-matched on the lowercased question) that flag a
# clearly off-topic topic.
_OFF_TOPIC_PHRASES = (
    "world cup", "euro 2024", "premier league", "stock market",
    "video game", "video games", "real madrid", "fast food",
)

# Max length of the context fed to the cheap relevance judge (keeps the call
# fast and inexpensive).
_JUDGE_CONTEXT_CHARS = 3000

# Prompt used by the cheap LLM relevance judge. Only consulted when the keyword
# gates above are inconclusive. The model must reply with exactly one word.
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
        self.top_k = top_k or config.TOP_K

        self.embeddings = self._build_embeddings()
        self.vector_store = self._load_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type=config.SEARCH_TYPE,
            search_kwargs=self._retriever_kwargs(),
        )
        self.llm = self._build_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        # LCEL chain: format the prompt, run the LLM, return the answer string.
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

    # ------------------------------------------------------------------ #
    # Builders                                                            #
    # ------------------------------------------------------------------ #
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

    def _build_llm(self) -> ChatNVIDIA:
        return ChatNVIDIA(
            model=self._config.LLM_MODEL,
            nvidia_api_key=self._config.NVIDIA_API_KEY,
            temperature=self._config.LLM_TEMPERATURE,
        )

    def _retriever_kwargs(self) -> dict:
        """Search kwargs for ``as_retriever``, tuned per search type."""
        kwargs: dict = {"k": self.top_k}
        if self._config.SEARCH_TYPE == "mmr":
            kwargs.update(
                {
                    "fetch_k": self._config.FETCH_K,
                    "lambda_mult": self._config.LAMBDA_MULT,
                }
            )
        return kwargs

    # ------------------------------------------------------------------ #
    # Core interface                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_history(chat_history: list[tuple[str, str]]) -> list[Any]:
        """Convert ``[(human, ai), ...]`` turns into LangChain messages."""
        turns = chat_history[-config.MAX_HISTORY_TURNS:]
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

    # ------------------------------------------------------------------ #
    # Retrieval                                                           #
    # ------------------------------------------------------------------ #
    # The bare Chroma embedding search returns weak matches for many
    # questions (NVIDIA nemotron-3-embed-1b compresses all Chroma chunk
    # distances into a narrow band). We therefore widen the candidate pool
    # (similarity search over 2xTOP_K), rank by similarity distance, and use
    # the configured MMR retriever as a diversity back-fill so the final set
    # is relevant but not narrowed to a single matched page. Experiments showed
    # that LLM-based query expansion on the small NIM model injects off-topic
    # chunks, so retrieval is deliberately deterministic on the raw question.
    def _retrieve(self, question: str) -> list[Document]:
        """Return ``self.top_k`` chunks ranked by similarity to the question.

        Primary ranking comes from ``similarity_search_with_score`` over a
        widened pool; the MMR ``self.retriever`` only back-fills slots when the
        similarity pool is too small, adding diversity without dominating the
        ranking.
        """
        scored: dict[str, tuple[float, Document]] = {}
        try:
            results = self.vector_store.similarity_search_with_score(
                question, k=self.top_k * 2
            )
        except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
                logger.exception("MMR back-fill failed")
        return [doc for _, doc in ranked[: self.top_k]]

    def _is_on_topic(self, question: str, docs: list[Document]) -> bool:
        """Decide whether ``question`` should be answered using ``docs``.

        Three-stage guard run before the answering LLM call:

        1. Clearly about LangChain (domain vocabulary) -> answer.
        2. Clearly off-topic (sports, weather, politics, ...) -> refuse.
        3. Inconclusive -> ask a cheap LLM judge.

        The guard inspects the *question* rather than relying on keyword
        overlap with the docs, because the documentation contains worked
        examples about unrelated topics (e.g. an "Euro 2024" notebook) that a
        naive overlap check would wrongly treat as relevant.
        """
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
        """Cheap LLM judge used when the keyword gates are inconclusive.

        Asks the model to answer YES/NO about whether the question is on-topic
        for the LangChain docs AND answerable from the retrieved context. Fails
        open (proceeds to answer) if the judge itself errors out.
        """
        context = self._format_docs(docs)
        if len(context) > _JUDGE_CONTEXT_CHARS:
            context = context[:_JUDGE_CONTEXT_CHARS] + "\n...[truncated]"
        judge_prompt = _RELEVANCE_JUDGE_PROMPT.format(
            question=question, context=context
        )
        try:
            response = self.llm.invoke([HumanMessage(content=judge_prompt)]).content
        except Exception:  # noqa: BLE001 - fail open on judge failure
            logger.exception("Relevance judge failed; proceeding to answer")
            return True
        verdict = (response or "").strip().upper()
        return verdict.startswith("YES")

    def answer(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> tuple[str, list[Document]]:
        """Answer a question with retrieval + history.

        Returns a ``(answer, sources)`` tuple where ``sources`` is the list of
        retrieved documents (empty when no context was found). This shape is
        convenient for Gradio's UI components.
        """
        chat_history = chat_history or []

        # 1. Retrieve relevant context for the question (multi-query).
        docs = self._retrieve(question)

        # 2. Refuse early when there is no context or the question is clearly
        #    unrelated to LangChain, without spending the answering LLM call.
        if not docs or not self._is_on_topic(question, docs):
            return (NO_INFO_MESSAGE, [])

        # 3. Assemble the "stuff" prompt with context + history.
        context = self._format_docs(docs)
        history_messages = self._normalise_history(chat_history)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            *history_messages,
            HumanMessage(content=question),
        ]
        answer_text = self.llm.invoke(messages).content

        return (answer_text, docs)

    # Convenience alias showing this can also be expressed as a chain.
    def build_chain(self):
        """Return the underlying LCEL chain (context + question -> answer)."""
        return self.chain


# --------------------------------------------------------------------------- #
# Module-level singleton builder for reuse across Gradio threads.               #
# --------------------------------------------------------------------------- #
_pipeline: RAGPipeline | None = None


def get_pipeline(*, config=config, rebuild: bool = False) -> RAGPipeline:
    """Return a shared :class:`RAGPipeline` (built lazily, cached)."""
    global _pipeline
    if _pipeline is None or rebuild:
        _pipeline = RAGPipeline(config=config)
    return _pipeline


# --------------------------------------------------------------------------- #
# Simple self-test                                                             #
# --------------------------------------------------------------------------- #
def test_pipeline(config=config) -> None:
    """Verify the pipeline with a sample question (requires an API key)."""
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

    # Exercise chat-history follow-up.
    history = [(question, answer)]
    answer2, _ = pipeline.answer("Can you give me a short code example?", history)
    print(f"[test] Follow-up answer:\n  {answer2}\n")
    print("[test] OK")


if __name__ == "__main__":
    test_pipeline()
