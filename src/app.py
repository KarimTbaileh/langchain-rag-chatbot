import os
import gradio as gr
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


# ==========================================
# THE ULTIMATE PATCH: Fixes Gradio Schema Bugs in Memory
# ==========================================
import gradio_client.utils as gradio_utils

if not hasattr(gradio_utils, '_original_get_type'):
    gradio_utils._original_get_type = gradio_utils.get_type
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "any" if schema else "null"
        return gradio_utils._original_get_type(schema)
    gradio_utils.get_type = _patched_get_type

if not hasattr(gradio_utils, '_original_json_schema_to_python_type'):
    gradio_utils._original_json_schema_to_python_type = gradio_utils._json_schema_to_python_type
    def _patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return gradio_utils._original_json_schema_to_python_type(schema, defs)
    gradio_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
    
    def _patched_public_json_schema_to_python_type(schema):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return gradio_utils._original_json_schema_to_python_type(schema, schema.get("$defs") if isinstance(schema, dict) else None)
    gradio_utils.json_schema_to_python_type = _patched_public_json_schema_to_python_type
# ==========================================

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
            sources_text = "\n\n**Sources:**\n" + "\n".join([
                f"- [{s.metadata.get('source', 'unknown')}]({s.metadata.get('source', '#')})"
                for s in sources
            ])
            return answer + sources_text
        return answer
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

demo = gr.ChatInterface(
    fn=respond,
    title="LangChain RAG Assistant",
    description="Ask questions about the official LangChain documentation.",
    examples=["What is a Runnable?", "How does memory work in LangChain?"],
    cache_examples=False,
)