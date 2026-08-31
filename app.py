import os
import sys
import threading
import subprocess
import gradio as gr

# إضافة مجلد src إلى مسار بايثون
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# متغيرات عالمية لتتبع حالة التحضير
is_initializing = True
init_message = "🚀 النظام يقوم الآن ببناء قاعدة المعرفة لأول مرة. هذه العملية قد تستغرق من 5 إلى 10 دقائق. يرجى التحقق مرة أخرى قريباً!"

def run_ingestion_in_background():
    global is_initializing, init_message
    chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # التحقق مما إذا كانت قاعدة البيانات موجودة وممتلئة
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        print("🚀 Running initial data ingestion in background...")
        try:
            subprocess.run([sys.executable, "-m", "src.ingest"], check=True)
            print("✅ Ingestion complete!")
            is_initializing = False
            init_message = "✅ تم تحضير النظام بنجاح! يمكنك الآن طرح أسئلتك."
        except Exception as e:
            print(f"⚠️ Ingestion failed: {e}")
            is_initializing = False
            init_message = f"⚠️ فشل تحضير النظام: {e}. يرجى مراجعة السجلات."
    else:
        print("✅ Knowledge base already exists. Skipping ingestion.")
        is_initializing = False
        init_message = "✅ النظام جاهز للعمل!"

# بدء عملية التحضير في خلفية التطبيق فوراً
threading.Thread(target=run_ingestion_in_background, daemon=True).start()

def respond(message, history):
    global is_initializing, init_message
    
    # إذا كان النظام لا يزال يحضر البيانات، نعيد رسالة انتظار بدلاً من المحاولة
    if is_initializing:
        return init_message + "\n\n*(جاري تحضير قاعدة البيانات، يرجى المحاولة مرة أخرى بعد بضع دقائق)*"
    
    try:
        from src.rag_pipeline import get_pipeline
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
    description="Ask questions about the official LangChain documentation. (ملاحظة: قد يستغرق التشغيل الأول بضع دقائق للتحضير)",
    examples=["What is a Runnable?", "How does memory work in LangChain?"],
    cache_examples=False,
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"🚀 Starting Gradio server on port {port}...")
    # هذا السطر هو ما يخبر Render بأن التطبيق بدأ بنجاح
    demo.launch(server_name="0.0.0.0", server_port=port)