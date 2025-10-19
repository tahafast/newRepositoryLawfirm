"""
Unified LangChain compatibility shims.
Use these helpers instead of importing from `langchain.schema` or `langchain.text_splitter` directly.
Works with both the old monolith and the new split packages.
"""
from typing import Any, Type


def _ensure_available(name: str, obj: Any):
    if obj is None:
        raise ImportError(
            f"{name} is unavailable. Ensure compatible LangChain packages are installed. "
            f"See requirements.txt in this repo for the needed extras."
        )
    return obj


# Document
try:
    from langchain_core.documents import Document  # type: ignore
except Exception:
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        Document = None  # type: ignore


# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        RecursiveCharacterTextSplitter = None  # type: ignore


# Embeddings
try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore
    except Exception:
        OpenAIEmbeddings = None  # type: ignore


# Public helpers
def ensure_Document() -> Type[Any]:
    return _ensure_available("Document", Document)


def get_recursive_splitter(**kwargs: Any):
    splitter_cls = ensure_RecursiveCharacterTextSplitter()
    return splitter_cls(**kwargs)


def ensure_RecursiveCharacterTextSplitter() -> Type[Any]:
    return _ensure_available(
        "RecursiveCharacterTextSplitter", RecursiveCharacterTextSplitter
    )


def ensure_OpenAIEmbeddings() -> Type[Any]:
    return _ensure_available("OpenAIEmbeddings", OpenAIEmbeddings)
