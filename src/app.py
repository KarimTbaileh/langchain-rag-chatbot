# ==========================================
# SILVER BULLET: Monkey patch to fix Gradio 4.44.1 boolean schema bug on HF Spaces
# ==========================================
import gradio_client.utils as gradio_utils
if not hasattr(gradio_utils, '_original_get_type'):
    gradio_utils._original_get_type = gradio_utils.get_type
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "any" if schema else "null"
        return gradio_utils._original_get_type(schema)
    gradio_utils.get_type = _patched_get_type
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
            history = history + [[message, answer + sources_text]]
        else:
            history = history + [[message, answer]]
        return history, ""
    except Exception as e:
        return history + [[message, f"⚠️ Error: {str(e)}"]], ""

with gr.Blocks() as demo:
    gr.Markdown("# LangChain RAG Assistant")
    gr.Markdown("Ask questions about the official LangChain documentation.")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about LangChain...")
    
    msg.submit(respond, [msg, chatbot], [chatbot, msg])

if __name__ == "__main__":
    demo.launch()