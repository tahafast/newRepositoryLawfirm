from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class EphemeralUploadResponse(BaseModel):
    status: str = Field("ok")
    chunks: int
    collection: str
    tokens_used: Optional[int] = None


class EphemeralExistsResponse(BaseModel):
    exists: bool
    collection: str


class EphemeralDeleteResponse(BaseModel):
    status: str = Field("ok")
    collection: str
    deleted: bool
