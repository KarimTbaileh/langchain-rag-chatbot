import gradio as gr
from src.rag_pipeline import get_pipeline

def respond(message, history):
    try:
        pipeline = get_pipeline()
        # Convert Gradio history [[user, bot], ...] to pipeline's expected format
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

with gr.Blocks(title="LangChain RAG Assistant") as demo:
    gr.Markdown("# LangChain RAG Assistant")
    gr.Markdown("Ask questions about the official LangChain documentation.")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your message", placeholder="e.g., What is a Runnable?")
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        formatted_history = [(h[0], h[1]) for h in history[:-1]]
        
        try:
            pipeline = get_pipeline()
            answer, sources = pipeline.answer(user_message, formatted_history)
            
            if sources:
                sources_text = "\n\n**Sources:**\n" + "\n".join([
                    f"- [{s.metadata.get('source', 'unknown')}]({s.metadata.get('source', '#')})" 
                    for s in sources
                ])
                history[-1][1] = answer + sources_text
            else:
                history[-1][1] = answer
        except Exception as e:
            history[-1][1] = f"⚠️ Error: {str(e)}"
            
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()