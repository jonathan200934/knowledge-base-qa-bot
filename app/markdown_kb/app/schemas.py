from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]


class IndexResponse(BaseModel):
    files_indexed: int
    sections_indexed: int
    changed_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    wiki_index: str | None = None


class ChatRequest(BaseModel):
    query: QueryText
    file_answer: bool = False


class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    answer_file: str | None = None


class CompareRequest(BaseModel):
    query: QueryText
    k: int = Field(default=3, ge=1, le=20)


class CompareResponse(BaseModel):
    query: str
    indexed: bool
    message: str | None = None
    bm25: list[SourceInfo]
    vector: list[SourceInfo]
    hybrid: list[SourceInfo]
    reranked: list[SourceInfo]
