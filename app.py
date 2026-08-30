"""Entry point for Hugging Face Spaces with safe ingestion."""
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
    print("🚀 Running initial data ingestion...")
    try:
        # check=True will raise an exception if it fails, which we catch below
        subprocess.run([sys.executable, "-m", "src.ingest"], check=True)
        print("✅ Ingestion complete!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Ingestion failed or interrupted (e.g., network timeout): {e}")
        print("The app will still start, but retrieval might be incomplete. You can re-run ingestion later.")
    except Exception as e:
        print(f"⚠️ Unexpected error during ingestion: {e}")

from src.app import demo

if __name__ == "__main__":
    # Gradio automatically detects Hugging Face Spaces and configures itself perfectly
    demo.launch()