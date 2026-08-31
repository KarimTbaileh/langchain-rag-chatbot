# ==========================================
# THE ULTIMATE PATCH: Fixes Gradio 4.44.1 Schema Bugs in Memory
# ==========================================
import gradio_client.utils as gradio_utils

# Patch 1: Fix "TypeError: argument of type 'bool' is not iterable"
if not hasattr(gradio_utils, '_original_get_type'):
    gradio_utils._original_get_type = gradio_utils.get_type
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "any" if schema else "null"
        return gradio_utils._original_get_type(schema)
    gradio_utils.get_type = _patched_get_type

# Patch 2: Fix "APIInfoParseError: Cannot parse schema True"
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

import gradio as gr
from src.rag_pipeline import get_pipeline

def respond(message, history):
    try:
        pipeline = get_pipeline()
        chat_pairs = [(h[0], h[1]) for h in history] if history else []
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

if __name__ == "__main__":
    demo.launch()