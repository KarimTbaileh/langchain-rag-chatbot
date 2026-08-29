"""Multi-turn Gradio chatbot interface for the LangChain RAG pipeline.

Features:
    * Multi-turn conversation (chat history is maintained per session).
    * Retrieved source documents shown under each answer in a clean list.
    * Clear-conversation button.
    * Title + description.
    * Loads the shared RAG pipeline from :mod:`src.rag_pipeline`.
    * Graceful error handling (bad question, missing key, empty store).
    * No hardcoded absolute paths -> ready for Hugging Face Spaces.

Run locally:
    uv run python src/app.py
"""

from __future__ import annotations

import logging

import gradio as gr

try:
    import config
except ModuleNotFoundError:  # when invoked as `uv run python src/app.py`
    from src import config

from src.rag_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# UI copy                                                                      #
# --------------------------------------------------------------------------- #
TITLE = "LangChain Docs Assistant"
DESCRIPTION = (
    "Ask questions about the official **LangChain documentation**. "
    "Answers are grounded in the retrieved documentation, and the source pages "
    "are shown under each answer. Multi-turn context is maintained within a session."
)
PLACEHOLDER = "e.g. What is a Runnable and how do I invoke it?"
CLEAR_LABEL = "Clear"

NO_CONTEXT_MESSAGE = (
    "I could not find information about that in the documentation. "
    "Try rephrasing your question or asking about a topic covered in the LangChain docs."
)


# --------------------------------------------------------------------------- #
# Source rendering (pure Markdown, no raw HTML)                                #
# --------------------------------------------------------------------------- #
def _md_escape(text: str) -> str:
    """Escape characters that carry meaning in Markdown."""
    return (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _md_link_target(url: str) -> str:
    """Make a URL safe to embed in a Markdown link destination."""
    return (
        _md_escape(url)
        .replace(" ", "%20")
        .replace("(", "%28")
        .replace(")", "%29")
    )


def _render_sources(sources: list) -> str:
    """Render retrieved source documents as a clean Markdown list."""
    if not sources:
        return ""
    items = []
    for idx, doc in enumerate(sources, start=1):
        url = (doc.metadata or {}).get("source", "unknown")
        title = (doc.metadata or {}).get("page_title", "") or url
        snippet = " ".join(doc.page_content.split())[:200].strip()
        items.append(
            f"{idx}. [{_md_escape(title)}]({_md_link_target(url)}) "
            f"— *{_md_escape(snippet)}…*"
        )
    return "**Sources**\n\n" + "\n".join(items)


# --------------------------------------------------------------------------- #
# Chat logic                                                                   #
# --------------------------------------------------------------------------- #
def _assistant_message(answer: str, sources: list) -> gr.ChatMessage:
    content = answer or ""
    sources_md = _render_sources(sources)
    if sources_md:
        content = f"{content}\n\n---\n\n{sources_md}" if content else sources_md
    return gr.ChatMessage(role="assistant", content=content)


def _chat_value(session_history: list[dict]) -> list:
    """Convert the session history (list of turn dicts) into chatbot messages."""
    value = []
    for turn in session_history:
        value.append(gr.ChatMessage(role="user", content=turn["question"]))
        value.append(_assistant_message(turn["answer"], turn.get("sources", [])))
    return value


def respond(message: str, session_history: list[dict]) -> tuple[list, list[dict], str]:
    """Handle one user turn. Returns (chatbot_value, new_state, clear_textbox_value)."""
    logger.info("Received message: %r", message[:120])
    history = [dict(t) for t in (session_history or [])]

    now = {
        "question": message,
        "answer": "",
        "sources": [],
        "error": False,
    }
    if message and message.strip():
        try:
            pipeline = get_pipeline()
            chat_pairs = [(t["question"], t["answer"]) for t in history if t["answer"]]
            answer, sources = pipeline.answer(message.strip(), chat_pairs)
            now["answer"] = answer or NO_CONTEXT_MESSAGE
            now["sources"] = sources
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI
            logger.exception("Pipeline error")
            now["answer"] = (
                "Something went wrong while answering your question. "
                f"Details: {type(exc).__name__}: {exc}"
            )
            now["error"] = True
    else:
        now["answer"] = "Please type a question to get started."
        now["error"] = True

    history.append(now)
    return _chat_value(history), history, ""


def clear_conversation() -> tuple[list, list[dict], str]:
    """Reset the chatbot and session state."""
    return [], [], ""


# --------------------------------------------------------------------------- #
# UI                                                                            #
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE, fill_height=True) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        session_history = gr.State(value=[])
        chatbot = gr.Chatbot(
            label="Conversation",
            elem_id="chatbot",
            render_markdown=True,
            buttons=["copy"],
            height="100%",
        )
        textbox = gr.Textbox(
            label="Your message",
            placeholder=PLACEHOLDER,
            scale=1,
            lines=1,
            max_lines=5,
        )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button(CLEAR_LABEL, scale=0)

        outputs = [chatbot, session_history, textbox]

        send_btn.click(
            fn=respond,
            inputs=[textbox, session_history],
            outputs=outputs,
            show_progress="minimal",
        )
        textbox.submit(
            fn=respond,
            inputs=[textbox, session_history],
            outputs=outputs,
            show_progress="minimal",
        )
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=outputs,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
