import asyncio
import functools
import time

from src.logger import AppLogger


def execution_timer(func):
    """
    Decorator that prints the total execution time of the decorated function.
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        AppLogger.get_logger().info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        AppLogger.get_logger().info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
