"""Promotion-gate extraction endpoint.

Takes a list of candidates (failure_delta / correction text windows)
and asks the configured LLM to emit 0–N structured memory nodes.
The endpoint does not write to a graph — the adapter is responsible
for posting the returned memories to /memories with the desired graph_id.
This keeps the contract orthogonal: the adapter owns graph routing,
the server owns LLM access.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_llm
from plugmem.api.schemas import ExtractRequest, ExtractResponse, ExtractedMemory
from plugmem.inference.promotion import extract_coding_memories

logger = logging.getLogger(__name__)
router = APIRouter(tags=["extract"], dependencies=[Depends(require_api_key)])


@router.post("/extract", response_model=ExtractResponse)
async def extract(body: ExtractRequest) -> ExtractResponse:
    if not body.candidates:
        return ExtractResponse(memories=[])

    candidates = [{"kind": c.kind, "window": c.window} for c in body.candidates]
    logger.info("extract: received %d candidates", len(candidates))
    print(f"EXTRACT REQUEST: {len(candidates)} candidates", flush=True)
    llm = get_llm()
    raw = extract_coding_memories(llm, candidates)

    memories = []
    for m in raw:
        try:
            memories.append(ExtractedMemory(**m))
        except Exception as e:  # noqa: BLE001 — log and skip malformed
            logger.info("extract: dropping invalid memory %r: %s", m, e)

    logger.info("extract: returning %d memories", len(memories))
    for mem in memories:
        logger.info("  -> type=%s, source=%s, confidence=%s", mem.type, mem.source, mem.confidence)
    print(f"EXTRACT RESULT: {len(memories)} memories", flush=True)
    return ExtractResponse(memories=memories)
