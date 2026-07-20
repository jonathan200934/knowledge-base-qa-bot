import os
import secrets

from fastapi import Header, HTTPException


def require_index_api_key(
    x_index_key: str | None = Header(default=None, alias="X-Index-Key"),
) -> None:
    """Authorize an HTTP operation that can write local application state."""
    expected_key = os.getenv("INDEX_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Protected write operations are not configured",
        )
    supplied_key = x_index_key.encode("utf-8") if x_index_key is not None else None
    expected_key_bytes = expected_key.encode("utf-8")
    if supplied_key is None or not secrets.compare_digest(supplied_key, expected_key_bytes):
        raise HTTPException(
            status_code=401,
            detail="Invalid index API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
