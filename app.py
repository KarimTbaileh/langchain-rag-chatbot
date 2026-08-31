import os
import sys
import time
import threading
import subprocess
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Global state
is_initializing = True
init_progress = {"status": "starting", "message": "Initializing system..."}

def run_ingestion_in_background():
    global is_initializing, init_progress
    chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        init_progress = {"status": "downloading", "message": "Downloading documentation..."}
        try:
            subprocess.run([sys.executable, "-m", "src.ingest"], check=True)
            init_progress = {"status": "ready", "message": "System is ready! Ask me anything about LangChain."}
            is_initializing = False
        except Exception as e:
            init_progress = {"status": "error", "message": f"Error: {str(e)[:100]}"}
            is_initializing = False
    else:
        init_progress = {"status": "ready", "message": "System is ready! Ask me anything about LangChain."}
        is_initializing = False

# Start background ingestion
threading.Thread(target=run_ingestion_in_background, daemon=True).start()

def respond(message, history):
    global is_initializing, init_progress
    
    if is_initializing:
        status = init_progress.get("status", "working")
        msg = init_progress.get("message", "Working...")
        
        if status == "downloading":
            return "⏳ **Preparing the knowledge base...**\n\nI'm downloading and processing the LangChain documentation. This usually takes 5-10 minutes on first launch.\n\nPlease refresh the page in a few minutes and try again!"
        elif status == "starting":
            return "⏳ **Starting up...**\n\nPlease wait a moment and try again."
        else:
            return f"⏳ {msg}\n\nPlease try again in a few minutes."
    
    try:
        from src.rag_pipeline import get_pipeline
        pipeline = get_pipeline()
        chat_pairs = [(h[0], h[1]) for h in history] if history else []
        answer, sources = pipeline.answer(message, chat_pairs)
        
        if sources:
            sources_text = "\n\n---\n**📚 Sources:**\n" + "\n".join([
                f"- {s.metadata.get('source', 'unknown')}" 
                for s in sources[:3]  # Limit to top 3 sources
            ])
            return answer + sources_text
        return answer
    except Exception as e:
        return f"⚠️ Sorry, an error occurred: {str(e)[:200]}"

# Create a more professional UI
with gr.Blocks(
    title="LangChain RAG Assistant",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {max-width: 900px !important; margin: auto !important;}
    .markdown-text {font-size: 1.1em;}
    """
) as demo:
    gr.Markdown(
        """
        # 🤖 LangChain Documentation Assistant
        
        Ask questions about the official LangChain documentation and get accurate answers with source references.
        
        **Examples to try:**
        - "What is a Runnable in LangChain?"
        - "How does memory work?"
        - "Explain LCEL with an example"
        """,
        elem_classes="markdown-text"
    )
    
    chatbot = gr.ChatInterface(
        fn=respond,
        examples=[
            "What is a Runnable in LangChain?",
            "How does memory work in LangChain?",
            "Explain LCEL with an example",
            "What are the different types of chains?"
        ],
        title=None,
        description=None,
        retry_btn="🔄 Try Again",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear Chat",
    )
    
    gr.Markdown(
        """
        ---
        **Built with:** LangChain • NVIDIA NIM • ChromaDB • Gradio  
        **Deployment:** Render.com
        """,
        elem_classes="markdown-text"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)