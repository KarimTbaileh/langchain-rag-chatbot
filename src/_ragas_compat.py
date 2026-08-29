"""
Compatibility shim for Ragas to prevent ChatVertexAI import errors 
when langchain-community moves it to a separate package.
"""
import sys
from types import ModuleType

# Create a dummy module to satisfy Ragas imports
dummy_vertexai = ModuleType("langchain_community.chat_models.vertexai")

# Add a dummy ChatVertexAI class
class DummyChatVertexAI:
    pass

dummy_vertexai.ChatVertexAI = DummyChatVertexAI

# Inject it into sys.modules before Ragas tries to import it
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertexai