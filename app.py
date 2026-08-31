import os
import sys
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.rag_pipeline import get_pipeline


def _normalize_history(history):
    if not history:
        return []

    normalised = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            human, ai = item[0], item[1]
            if human is not None and ai is not None:
                normalised.append((str(human), str(ai)))
    return normalised


def _normalize_docs(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return []


def respond(message, history):
    try:
        pipeline = get_pipeline()
        docs = _normalize_docs(pipeline.retriever.invoke(message))

        if len(docs) == 0:
            return (
                "🔍 **Retrieval failed:**\n\n"
                "The system did not find matching documentation for that question. "
                "Please try a more specific LangChain question."
            )

        chat_pairs = _normalize_history(history)
        answer, sources = pipeline.answer(message, chat_pairs)

        if sources:
            sources_text = "\n\n---\n**📚 Sources:**\n" + "\n".join([
                f"- {s.metadata.get('source', 'unknown')}" for s in sources[:3]
            ])
            return f"✅ **Found {len(docs)} matches!**\n\n{answer}{sources_text}"

        return answer

    except Exception as e:
        return f"⚠️ **System error:**\n{str(e)}"

with gr.Blocks(
    title="LangChain RAG Assistant",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 900px !important; margin: auto !important;}"
) as demo:
    gr.Markdown("# 🤖 LangChain Documentation Assistant\n\nAsk questions about the official LangChain documentation.")
    
    gr.ChatInterface(
        fn=respond,
        examples=[
            "What is a Runnable in LangChain?",
            "How does memory work?",
            "Explain LCEL with an example"
        ],
        retry_btn="🔄 Retry",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)