import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from . import indexer
from .retrieval import compare, query
from .schemas import ChatRequest, ChatResponse, CompareRequest, CompareResponse, IndexResponse
from .security import require_index_api_key

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs(_authorized: None = Depends(require_index_api_key)):
    try:
        files_count, sections_count = indexer.build_index()
    except Exception as exc:
        logger.error("Index build failed (error_type=%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Index build failed") from exc
    return IndexResponse(
        files_indexed=files_count,
        sections_indexed=sections_count,
        changed_files=indexer.last_index_stats.get("changed_files", 0),
        skipped_files=indexer.last_index_stats.get("skipped_files", 0),
        deleted_files=indexer.last_index_stats.get("deleted_files", 0),
        wiki_index=indexer.last_index_stats.get("wiki_index"),
    )


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(
    req: ChatRequest,
    x_index_key: str | None = Header(default=None, alias="X-Index-Key"),
):
    if req.file_answer:
        require_index_api_key(x_index_key)
    return query(req.query, file_answer=req.file_answer)


@router.post("/compare", response_model=CompareResponse)
def compare_retrieval(req: CompareRequest):
    return compare(req.query, k=req.k)
