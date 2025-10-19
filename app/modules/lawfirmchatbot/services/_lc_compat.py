# Centralized LangChain import compatibility for v0.1+
from typing import Any


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
def ensure_Document():
    return _ensure_available("Document", Document)


def ensure_RecursiveCharacterTextSplitter():
    return _ensure_available(
        "RecursiveCharacterTextSplitter", RecursiveCharacterTextSplitter
    )


def ensure_OpenAIEmbeddings():
    return _ensure_available("OpenAIEmbeddings", OpenAIEmbeddings)
