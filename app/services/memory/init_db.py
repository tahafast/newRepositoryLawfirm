"""Database initialization script for chat memory tables."""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.services.memory.models import Base
from app.services.memory.db import DATABASE_URL
import logging

logger = logging.getLogger(__name__)

async def init_database():
    """Initialize database tables if they don't exist."""
    try:
        logger.info(f"Initializing database at {DATABASE_URL}")
        engine = create_async_engine(DATABASE_URL, echo=False)
        
        async with engine.begin() as conn:
            # Create all tables defined in models
            await conn.run_sync(Base.metadata.create_all)
        
        await engine.dispose()
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(init_database())

