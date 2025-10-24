from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EphemeralUploadFile(BaseModel):
    doc_id: str
    file_name: str
    chunks: int


class EphemeralUploadResponse(BaseModel):
    status: str = Field("ok")
    collection: str
    files: List[EphemeralUploadFile] = Field(default_factory=list)
    mode: str = Field("append")
    tokens_used: Optional[int] = None


class EphemeralExistsResponse(BaseModel):
    exists: bool
    collection: str


class EphemeralDeleteResponse(BaseModel):
    status: str = Field("ok")
    collection: str
    deleted: bool
