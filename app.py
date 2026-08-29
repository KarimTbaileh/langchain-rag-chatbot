import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )