import functools
import logging
import time

logger = logging.getLogger(__name__)


def execution_timer(func):
    """
    Decorator that prints the total execution time of the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper
