"""Compatibility shim for Ragas.

Ragas < 1.0 (here 0.4.3) unconditionally imports
``langchain_community.chat_models.vertexai.ChatVertexAI`` at module load time,
but that module was removed from ``langchain-community 0.4.x`` (it moved to the
standalone ``langchain-google-vertexai`` package). Ragas only references
``ChatVertexAI`` when *instantiating* a Vertex AI model for evaluation, which we
never do (we evaluate with the NVIDIA NIM LLM). So a placeholder class that
satisfies the import is sufficient to let Ragas load cleanly.

Call :func:`install()` BEFORE ``import ragas`` anywhere Ragas is used.
"""

from __future__ import annotations

import sys
import types

_SHIM_PATH = "langchain_community.chat_models.vertexai"


def install() -> None:
    """Register a stub ``langchain_community.chat_models.vertexai`` module.

    No-op if the real module is already importable or the shim is installed.
    """
    if _SHIM_PATH in sys.modules:
        return

    try:  # pragma: no cover - only if a real module exists
        import importlib  # noqa: F401

        importlib.import_module(_SHIM_PATH)
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(_SHIM_PATH)
    module.__doc__ = "Compatibility stub for Ragas (Vertex AI is unused)."

    class ChatVertexAI:  # placeholder; never instantiated by our evaluation
        """Placeholder satisfying Ragas' import.

        Instantiating this raises ``NotImplementedError`` so misuse is obvious.
        """

        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError(
                "Vertex AI is not available in this stack. Use the NVIDIA NIM "
                "LLM for Ragas evaluation instead of ChatVertexAI."
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules[_SHIM_PATH] = module
