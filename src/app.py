import gradio as gr
from src.rag_pipeline import get_pipeline

def predict(message: str, history: list):
    try:
        pipeline = get_pipeline()
        chat_pairs = history if history else []
        answer, sources = pipeline.answer(message, chat_pairs)
        
        sources_text = ""
        if sources:
            sources_text = "\n\n**Sources:**\n" + "\n".join([
                f"- [{s.metadata.get('source', 'unknown')}]({s.metadata.get('source', '#')})" 
                for s in sources
            ])
        return answer + sources_text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

demo = gr.ChatInterface(
    fn=predict,
    title="LangChain Docs Assistant",
    description="Ask questions about the official **LangChain documentation**.",
    examples=[
        "What is a Runnable?",
        "How does memory work in LangChain?"
    ],
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.launch()