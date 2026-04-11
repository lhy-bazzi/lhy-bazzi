"""RabbitMQ async consumer using aio-pika."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Optional

import aio_pika
from aio_pika import Message
from aio_pika.abc import AbstractIncomingMessage
from loguru import logger
from pydantic import BaseModel

from app.config import get_settings

# Type alias for message handler
MessageHandler = Callable[[dict], Awaitable[None]]


class ParseTaskMessage(BaseModel):
    """Schema for parse-task messages sent by the Java backend."""
    task_id: str
    doc_id: str
    kb_id: str
    file_path: str
    file_type: str
    callback_url: Optional[str] = None


class MQConsumer:
    """RabbitMQ async consumer.

    Usage::

        consumer = MQConsumer()
        consumer.register_handler("doc_parse_queue", my_handler)
        await consumer.start()
        ...
        await consumer.stop()
    """

    def __init__(self) -> None:
        self._connection: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.Channel | None = None
        self._handlers: dict[str, MessageHandler] = {}

    def register_handler(self, queue_name: str, handler: MessageHandler) -> None:
        """Register an async handler for a specific queue."""
        self._handlers[queue_name] = handler
        logger.debug("MQ handler registered for queue '{}'.", queue_name)

    async def start(self) -> None:
        """Connect to RabbitMQ and start consuming all registered queues."""
        cfg = get_settings().mq
        logger.info("Connecting to RabbitMQ: {}", cfg.url.split("@")[-1] if "@" in cfg.url else cfg.url)

        self._connection = await aio_pika.connect_robust(cfg.url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=10)

        for queue_name, handler in self._handlers.items():
            queue = await self._channel.declare_queue(
                queue_name, durable=True, auto_delete=False
            )
            await queue.consume(self._make_callback(handler, queue_name))
            logger.info("MQ consumer started on queue '{}'.", queue_name)

    async def stop(self) -> None:
        """Gracefully shut down the consumer."""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        logger.info("MQ consumer stopped.")

    def _make_callback(
        self, handler: MessageHandler, queue_name: str
    ) -> Callable[[AbstractIncomingMessage], Awaitable[None]]:
        async def _callback(message: AbstractIncomingMessage) -> None:
            async with message.process(requeue=False):
                try:
                    body = json.loads(message.body.decode("utf-8"))
                    logger.debug("MQ [{}] received: {}", queue_name, body)
                    await handler(body)
                    # Message is acked automatically by context manager on success
                except json.JSONDecodeError as exc:
                    logger.error("MQ [{}] invalid JSON: {}", queue_name, exc)
                    # Bad message — don't requeue, just discard
                except Exception as exc:
                    logger.exception("MQ [{}] handler error: {}", queue_name, exc)
                    # Nack without requeue to avoid poison-pill loops
                    await message.nack(requeue=False)

        return _callback


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_consumer: MQConsumer | None = None


def get_mq_consumer() -> MQConsumer:
    global _consumer
    if _consumer is None:
        _consumer = MQConsumer()
    return _consumer
