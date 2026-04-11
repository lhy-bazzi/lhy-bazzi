"""Permission filter — Redis-backed user context + dual-layer filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger

from app.services.retrieval.models import RetrievedChunk

_PERM_KEY_PREFIX = "uni_ai:user_perm:"


@dataclass
class UserContext:
    user_id: str
    kb_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    is_admin: bool = False


class PermissionFilter:
    """Redis-backed permission context + Milvus/ES filter builders."""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def get_user_context(self, user_id: str) -> UserContext:
        key = f"{_PERM_KEY_PREFIX}{user_id}"
        try:
            raw = await self.redis.get(key)
            if raw:
                data = json.loads(raw)
                return UserContext(
                    user_id=user_id,
                    kb_ids=data.get("kb_ids", []),
                    doc_ids=data.get("doc_ids", []),
                    is_admin=data.get("is_admin", False),
                )
        except Exception as exc:
            logger.warning("Failed to fetch user perm from Redis (user={}): {}", user_id, exc)
        # Fallback: no access (safe default)
        return UserContext(user_id=user_id)

    def build_milvus_filter(self, ctx: UserContext, kb_ids: list[str] | None = None) -> str:
        """Build Milvus scalar filter expression."""
        if ctx.is_admin:
            if kb_ids:
                ids = '", "'.join(kb_ids)
                return f'kb_id in ["{ids}"]'
            return ""

        allowed_kb = list(set(ctx.kb_ids) & set(kb_ids)) if kb_ids else ctx.kb_ids
        if not allowed_kb:
            return 'kb_id == "__no_access__"'

        kb_expr = '", "'.join(allowed_kb)
        expr = f'kb_id in ["{kb_expr}"]'

        if ctx.doc_ids:
            doc_expr = '", "'.join(ctx.doc_ids)
            expr += f' and doc_id in ["{doc_expr}"]'

        return expr

    def build_es_filter(self, ctx: UserContext, kb_ids: list[str] | None = None) -> dict:
        """Build ES bool filter dict."""
        if ctx.is_admin:
            if kb_ids:
                return {"bool": {"filter": [{"terms": {"kb_id": kb_ids}}]}}
            return {}

        allowed_kb = list(set(ctx.kb_ids) & set(kb_ids)) if kb_ids else ctx.kb_ids
        filters: list[dict] = [{"terms": {"kb_id": allowed_kb}}] if allowed_kb else [{"term": {"kb_id": "__no_access__"}}]

        if ctx.doc_ids:
            filters.append({"terms": {"doc_id": ctx.doc_ids}})

        return {"bool": {"filter": filters}}

    def post_filter(self, chunks: list[RetrievedChunk], ctx: UserContext) -> list[RetrievedChunk]:
        """Application-layer safety net filter."""
        if ctx.is_admin:
            return chunks
        allowed_kb = set(ctx.kb_ids)
        allowed_doc = set(ctx.doc_ids) if ctx.doc_ids else None
        return [
            c for c in chunks
            if c.kb_id in allowed_kb and (allowed_doc is None or c.doc_id in allowed_doc)
        ]
