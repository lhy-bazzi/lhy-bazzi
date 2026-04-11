"""
Initialize Elasticsearch index and mapping.

Run: python scripts/init_es.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from app.core.es_client import close_es, ensure_index, init_es


async def main():
    logger.info("Initialising Elasticsearch index...")
    await init_es()
    logger.info("Done.")
    await close_es()


if __name__ == "__main__":
    asyncio.run(main())
