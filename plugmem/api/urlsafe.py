"""Let opaque graph IDs travel as a single URL path segment.

Graph IDs are namespaced identifiers, not path fragments: the coding adapters
mint things like ``repo://claude-code/github.com/owner/repo``. Clients percent-
encode them (``encodeURIComponent``), but by the time an ASGI server hands the
request over, ``scope["path"]`` has already been fully percent-decoded, so
``%2F`` has become a real ``/``. Routing then sees extra path segments and every
graph-scoped endpoint 404s.

Two pieces fix that without rewriting a single route:

``EncodedSlashPathMiddleware``
    Rebuilds ``scope["path"]`` from ``scope["raw_path"]``, which the server does
    preserve verbatim, decoding every escape *except* ``%2F``. Routing then
    matches the ID as one segment, because ``{graph_id}`` compiles to ``[^/]+``.

``UnquotedPathParamsRoute``
    Undoes that on the way in, so handlers receive the real ID and echo the real
    ID back in responses. Set as ``route_class`` on the routers.

The alternative, declaring every graph-scoped route with ``{graph_id:path}``,
was rejected: it needs sixteen routes reordered across four modules so bare
``/{graph_id}`` cannot shadow ``/{graph_id}/stats``, and it leaves
``/{graph_id:path}/node/{node_type}/{node_id}`` genuinely ambiguous.
"""
from __future__ import annotations

from typing import Any, Callable, Coroutine
from urllib.parse import unquote

from fastapi import Request, Response
from fastapi.routing import APIRoute

# Stand-in for %2F while the rest of the path is decoded. A NUL byte cannot
# appear in a URL, so it cannot collide with real path content.
_SENTINEL = "\x00"


def decode_path_preserving_slashes(raw_path: str) -> str:
    """Percent-decode a path, leaving encoded slashes encoded."""
    parts = raw_path.replace("%2f", "%2F").split("%2F")
    return "%2F".join(unquote(p) for p in parts)


class EncodedSlashPathMiddleware:
    """Pure ASGI middleware; must run before routing, so add it outermost."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            raw_path = scope.get("raw_path")
            if raw_path:
                # Query string is a separate scope key; raw_path may still carry
                # it on some servers, so cut there rather than assume.
                raw = raw_path.decode("ascii", "ignore").split("?", 1)[0]
                scope["path"] = decode_path_preserving_slashes(raw)
        await self.app(scope, receive, send)


class UnquotedPathParamsRoute(APIRoute):
    """Percent-decode path params after routing, before the handler sees them."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def unquoting_handler(request: Request) -> Response:
            params = request.scope.get("path_params")
            if params:
                request.scope["path_params"] = {
                    key: unquote(value) if isinstance(value, str) else value
                    for key, value in params.items()
                }
            return await original(request)

        return unquoting_handler
