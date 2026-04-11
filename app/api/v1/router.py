"""V1 API router — aggregates all sub-routers."""

from fastapi import APIRouter

from app.api.v1 import chat, knowledge, parse, system

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(system.router)
api_v1_router.include_router(parse.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(knowledge.router)
