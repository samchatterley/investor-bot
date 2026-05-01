import functools
import logging
import time

logger = logging.getLogger(__name__)


def with_retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator: retry a function on failure with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator
