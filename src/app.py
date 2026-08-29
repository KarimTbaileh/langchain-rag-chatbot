"""Multi-turn Gradio chatbot interface for the LangChain RAG pipeline."""
from __future__ import annotations

import html as _html
import logging

import gradio as gr

try:
    import config
except ModuleNotFoundError:
    from src import config

from src.rag_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TITLE = "LangChain Docs Assistant"
DESCRIPTION = (
    "Ask questions about the official **LangChain documentation**. "
    "Answers are grounded in the retrieved documentation."
)
PLACEHOLDER = "e.g. What is a Runnable and how do I invoke it?"
CLEAR_LABEL = "Clear"

def _render_sources(sources: list) -> str:
    if not sources:
        return ""
    items = []
    for idx, doc in enumerate(sources, start=1):
        url = (doc.metadata or {}).get("source", "unknown")
        title = (doc.metadata or {}).get("page_title", "") or url
        link = f'<a href="{_html.escape(url)}" target="_blank" rel="noopener">{_html.escape(title)}</a>'
        snippet = _html.escape(doc.page_content[:200])
        items.append(
            f"<li><b>{idx}.</b> {link}<br>"
            f"<span style='color:#666;font-size:0.9em'>{snippet}…</span></li>"
        )
    return (
        "<details style='margin-top:6px'>"
        f"<summary><b>Sources ({len(sources)})</b></summary>"
        f"<ul style='padding-left:1.2em'>{''.join(items)}</ul>"
        "</details>"
    )

def _assistant_message(answer: str, sources: list) -> str:
    sources_html = _render_sources(sources)
    content = answer
    if sources_html:
        content = f"{answer}\n\n{sources_html}"
    return content

def respond(message: str, session_history: list[dict]) -> tuple[list, list[dict], str]:
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
            now["answer"] = answer or ""
            now["sources"] = sources
        except Exception as exc:
            logger.exception("Pipeline error")
            now["answer"] = f"⚠️ Error: {type(exc).__name__}: {exc}"
            now["error"] = True
    else:
        now["answer"] = "Please type a question to get started."
        now["error"] = True

    history.append(now)
    
    # استخدام التنسيق الكلاسيكي المستقر لـ Gradio (قائمة من الأزواج)
    chatbot_value = []
    for turn in history:
        chatbot_value.append((turn["question"], _assistant_message(turn["answer"], turn.get("sources", []))))
        
    return chatbot_value, history, ""

def clear_conversation() -> tuple[list, list[dict], str]:
    return [], [], ""

def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        session_history = gr.State(value=[])
        
        # استخدام height كرقم صحيح لتجنب مشاكل الـ Schema في 4.44.1
        chatbot = gr.Chatbot(
            label="Conversation",
            elem_id="chatbot",
            height=500,
        )
        
        textbox = gr.Textbox(
            label="Your message",
            placeholder=PLACEHOLDER,
            lines=1,
            max_lines=5,
        )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button(CLEAR_LABEL)

        outputs = [chatbot, session_history, textbox]

        send_btn.click(
            fn=respond,
            inputs=[textbox, session_history],
            outputs=outputs,
        )
        textbox.submit(
            fn=respond,
            inputs=[textbox, session_history],
            outputs=outputs,
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