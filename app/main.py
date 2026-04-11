"""
UniAI Python AI Service — FastAPI Application Entry Point.

Start with:
    uvicorn app.main:app --reload --port 8100
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.v1.router import api_v1_router
from app.config import get_settings
from app.utils.exceptions import UniAIBaseError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting UniAI Python AI Service v{}", settings.app_version)
    logger.info("Debug mode: {}", settings.debug)

    from app.core import init_all_services, close_all_services
    await init_all_services()

    yield

    await close_all_services()
    logger.info("UniAI Python AI Service stopped.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="UniAI Python AI Service",
        version=settings.app_version,
        description="Enterprise Knowledge Base AI Engine — document parsing, intelligent retrieval & QA",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request logging middleware ----
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "{method} {path} → {status} ({ms:.0f}ms)",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=elapsed_ms,
        )
        return response

    # ---- Global exception handler ----
    @app.exception_handler(UniAIBaseError)
    async def uni_ai_error_handler(request: Request, exc: UniAIBaseError):
        return JSONResponse(
            status_code=exc.code,
            content={
                "code": exc.code,
                "message": exc.message,
                "detail": None,
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on {} {}", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": "Internal server error",
                "detail": str(exc) if settings.debug else None,
            },
        )

    # ---- Register routers ----
    app.include_router(api_v1_router)

    return app


app = create_app()
