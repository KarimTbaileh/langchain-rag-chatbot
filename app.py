import os
import sys
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_pipeline import get_pipeline

def respond(message, history):
    try:
        pipeline = get_pipeline()
        chat_pairs = [(h[0], h[1]) for h in history] if history else []
        answer, sources = pipeline.answer(message, chat_pairs)
        
        if sources:
            sources_text = "\n\n---\n**📚 Sources:**\n" + "\n".join([
                f"- {s.metadata.get('source', 'unknown')}" 
                for s in sources[:3]
            ])
            return answer + sources_text
        return answer
    except Exception as e:
        return f"⚠️ Error: {str(e)[:200]}"

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