"""Celery application configuration."""

from celery import Celery

from app.config import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "uni_ai_tasks",
        broker=settings.redis.url,          # Use Redis as broker
        backend=settings.redis.url,         # Use Redis as result backend
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Shanghai",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        # Parse tasks can be slow — generous time limits
        task_time_limit=600,           # hard kill after 10 min
        task_soft_time_limit=540,      # soft timeout at 9 min
    )
    app.autodiscover_tasks(["app.tasks"])
    return app


celery_app = create_celery_app()
