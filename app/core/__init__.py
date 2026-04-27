"""
Core infrastructure lifecycle management.

Call `init_all_services()` on startup and `close_all_services()` on shutdown.
Individual clients are importable directly from their modules.
"""

from __future__ import annotations

from uuid import uuid4

from loguru import logger

from app.core.database import close_db, init_db
from app.core.es_client import close_es, init_es
from app.core.llm_provider import init_llm
from app.core.minio_client import init_minio
from app.core.milvus_client import close_milvus, init_milvus
from app.core.mq_consumer import get_mq_consumer
from app.core.redis_client import close_redis, init_redis


async def init_all_services() -> None:
    """Initialise every infrastructure service. Called from FastAPI lifespan."""
    logger.info("Initialising infrastructure services...")

    await init_db()
    await init_redis()
    init_minio()
    init_llm()

    # Milvus and ES are optional on startup — log warnings but don't crash
    try:
        await init_milvus()
    except Exception as exc:
        logger.warning("Milvus init failed (service may be unavailable): {}", exc)

    try:
        await init_es()
    except Exception as exc:
        logger.warning("Elasticsearch init failed (service may be unavailable): {}", exc)

    # Embedding models + index service
    try:
        from app.config import get_settings
        from app.services.embedding.model_manager import init_model_manager
        from app.services.embedding.embedder import EmbeddingService
        from app.services.indexing.milvus_indexer import MilvusIndexer
        from app.services.indexing.es_indexer import ESIndexer
        from app.services.indexing.indexer import init_index_service
        from app.core.redis_client import get_redis
        from app.core.milvus_client import get_milvus
        from app.core.es_client import get_es

        settings = get_settings()
        model_mgr = await init_model_manager(settings)
        embed_svc = EmbeddingService(model_mgr, get_redis(), settings)
        init_index_service(
            MilvusIndexer(get_milvus(), embed_svc),
            ESIndexer(get_es()),
        )
        logger.info("Embedding models and index service initialised.")

        # Retrieval + QA engine
        from app.services.retrieval.vector_retriever import VectorRetriever, SparseRetriever
        from app.services.retrieval.fulltext_retriever import FulltextRetriever
        from app.services.retrieval.fusion import RRFFusion
        from app.services.retrieval.reranker import RerankerService
        from app.services.retrieval.permission_filter import PermissionFilter
        from app.services.retrieval.hybrid_retriever import HybridRetriever
        from app.core.retrieval import init_retriever
        from app.services.qa.query_understanding import QueryUnderstanding
        from app.services.qa.response_synthesizer import ResponseSynthesizer
        from app.services.qa.qa_engine import QAEngine
        from app.core.llm_provider import get_llm
        from app.core.qa import init_qa_engine

        hybrid = HybridRetriever(
            vector_retriever=VectorRetriever(get_milvus(), embed_svc),
            sparse_retriever=SparseRetriever(get_milvus(), embed_svc),
            fulltext_retriever=FulltextRetriever(get_es()),
            fusion=RRFFusion(),
            reranker=RerankerService(model_mgr),
            permission_filter=PermissionFilter(get_redis()),
            settings=settings,
        )
        init_retriever(hybrid)

        qa = QAEngine(
            query_understanding=QueryUnderstanding(get_llm()),
            hybrid_retriever=hybrid,
            response_synthesizer=ResponseSynthesizer(get_llm()),
            llm_provider=get_llm(),
            settings=settings,
        )
        init_qa_engine(qa)
        logger.info("Retrieval and QA engine initialised.")
    except Exception as exc:
        logger.warning("Embedding/index service init failed: {}", exc)

    # MQ consumer — register handlers and start consuming
    try:
        from app.config import get_settings

        consumer = get_mq_consumer()

        # Register parse queue handler → dispatches to Celery
        async def _handle_parse_message(body: dict) -> None:
            from app.core.mq_consumer import ParseTaskMessage
            from app.tasks.parse_task import process_parse_task

            payload = ParseTaskMessage.model_validate(body)
            task_id = (payload.task_id or uuid4().hex)[:64]
            message = {"task_id": task_id, "file_id": payload.file_id}

            logger.info("MQ parse message received: task_id={} file_id={}", task_id, payload.file_id)
            process_parse_task.apply_async(args=[message], task_id=task_id)

        parse_queue = get_settings().mq.parse_queue
        consumer.register_handler(parse_queue, _handle_parse_message)

        await consumer.start()
    except Exception as exc:
        logger.warning("RabbitMQ consumer init failed (service may be unavailable): {}", exc)

    logger.info("Infrastructure services initialised.")


async def close_all_services() -> None:
    """Gracefully shut down all infrastructure services."""
    logger.info("Shutting down infrastructure services...")

    try:
        consumer = get_mq_consumer()
        await consumer.stop()
    except Exception as exc:
        logger.warning("Error stopping MQ consumer: {}", exc)

    await close_es()
    await close_milvus()
    await close_redis()
    await close_db()

    logger.info("All infrastructure services stopped.")


__all__ = [
    "init_all_services",
    "close_all_services",
]
