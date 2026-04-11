"""
Initialize Milvus collection and indexes.

Run: python scripts/init_milvus.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from app.core.milvus_client import ensure_collection, init_milvus, close_milvus


async def main():
    logger.info("Initialising Milvus collection...")
    await init_milvus()
    logger.info("Done.")
    await close_milvus()


if __name__ == "__main__":
    asyncio.run(main())
