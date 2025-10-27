import os

_DEFAULT = "ephemeral_docs"
_ENV = "EPHEMERAL_COLLECTION_NAME"


def get_ephemeral_collection() -> str:
    """Single source of truth for the ephemeral collection name (read+write)."""
    value = os.getenv(_ENV, _DEFAULT).strip()
    return value or _DEFAULT
