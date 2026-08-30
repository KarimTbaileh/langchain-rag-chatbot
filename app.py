import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
    print("🚀 Running initial data ingestion...")
    try:
        subprocess.run([sys.executable, "-m", "src.ingest"], check=True)
        print("✅ Ingestion complete!")
    except Exception as e:
        print(f"⚠️ Ingestion skipped (likely network timeout): {e}")
        print("The app will start anyway. You can re-run ingestion later.")

from src.app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)