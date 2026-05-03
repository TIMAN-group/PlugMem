"""API key authentication middleware."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from plugmem.api.dependencies import get_config
from plugmem.config import PlugMemConfig

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
    config: PlugMemConfig = Depends(get_config),
) -> None:
    """Validate the API key if one is configured.

    If ``config.api_key`` is ``None`` or empty, authentication is disabled
    and all requests are allowed through.
    """
    if not config.api_key:
        return
    if api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
