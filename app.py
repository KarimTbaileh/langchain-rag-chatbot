import os
import sys
import subprocess

# 1. إضافة مجلد src إلى مسار بايثون لضمان استيراد الملفات منه
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 2. التحقق من وجود قاعدة البيانات، وإنشاؤها إذا لم تكن موجودة
chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
    print("🚀 Running initial data ingestion (this may take a few minutes)...")
    try:
        subprocess.run([sys.executable, "-m", "src.ingest"], check=True)
        print("✅ Ingestion complete!")
    except Exception as e:
        print(f"⚠️ Ingestion failed or skipped: {e}")
        print("The app will still start, but retrieval might be incomplete.")

# 3. استيراد واجهة التطبيق من مجلد src
from src.app import demo

# 4. تشغيل الخادم على المنفذ الذي يحدده Render (أو 7860 كافتراضي)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"🚀 Starting Gradio server on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port)