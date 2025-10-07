import time
import functools
import asyncio
import logging

logger = logging.getLogger(__name__)

def profile_stage(stage_name: str):
    """Decorator to log how long each stage takes (works with async or sync)."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    logger.info(f"[PERF] {stage_name}: {(time.perf_counter() - t0)*1000:.1f} ms")
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    logger.info(f"[PERF] {stage_name}: {(time.perf_counter() - t0)*1000:.1f} ms")
            return wrapper
    return decorator
