"""Async utilities for LLM Gateway."""
import asyncio
import functools
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

from llm_gateway.utils import get_logger

logger = get_logger(__name__)

# Type definitions
T = TypeVar('T')
AsyncCallable = Callable[..., Any]


class RateLimiter:
    """Rate limiter for controlling request rates."""
    
    def __init__(self, max_calls: int, period: float):
        """Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make a call.
        
        This will block until a call is allowed based on the rate limit.
        """
        async with self.lock:
            now = time.time()
            
            # Remove expired timestamps
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # Check if we're under the limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
                
            # Calculate wait time
            wait_time = self.period - (now - self.calls[0])
            if wait_time > 0:
                # Release lock while waiting
                self.lock.release()
                try:
                    logger.debug(
                        f"Rate limit reached, waiting {wait_time:.2f}s",
                        emoji_key="warning"
                    )
                    await asyncio.sleep(wait_time)
                finally:
                    # Reacquire lock
                    await self.lock.acquire()
                
                # Retry after waiting
                await self.acquire()
            else:
                # Oldest call just expired, record new call
                self.calls = self.calls[1:] + [now]


@asynccontextmanager
async def timed_context(name: str):
    """Context manager for timing operations.
    
    Args:
        name: Name of the operation for logging
        
    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(
            f"{name} completed in {duration:.3f}s",
            emoji_key="time",
            time=duration
        )


async def gather_with_concurrency(
    n: int,
    *tasks,
    return_exceptions: bool = False
) -> List[Any]:
    """Run tasks with limited concurrency.
    
    This is similar to asyncio.gather but limits the number of
    concurrent tasks.
    
    Args:
        n: Maximum number of concurrent tasks
        *tasks: Tasks to run
        return_exceptions: Whether to return exceptions or raise them
        
    Returns:
        List of task results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def run_task_with_semaphore(task):
        async with semaphore:
            return await task
            
    return await asyncio.gather(
        *(run_task_with_semaphore(task) for task in tasks),
        return_exceptions=return_exceptions
    )


async def run_with_timeout(
    coro: Any,
    timeout: float,
    default: Optional[T] = None,
    log_timeout: bool = True
) -> Union[Any, T]:
    """Run a coroutine with a timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value to return on timeout
        log_timeout: Whether to log timeouts
        
    Returns:
        Coroutine result or default value on timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if log_timeout:
            logger.warning(
                f"Operation timed out after {timeout}s",
                emoji_key="time",
                time=timeout
            )
        return default


def async_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: Optional[List[Type[Exception]]] = None,
    max_backoff: Optional[float] = None
):
    """Decorator for retrying async functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by on each retry
        retry_exceptions: List of exception types to retry on (defaults to all)
        max_backoff: Maximum backoff time in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            exceptions = []
            delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if we should retry this exception
                    if retry_exceptions and not any(
                        isinstance(e, exc_type) for exc_type in retry_exceptions
                    ):
                        raise
                        
                    exceptions.append(e)
                    
                    # If this was the last attempt, reraise
                    if attempt >= max_retries:
                        if len(exceptions) > 1:
                            logger.error(
                                f"Function {func.__name__} failed after {max_retries+1} attempts",
                                emoji_key="error",
                                attempts=max_retries+1
                            )
                        raise
                    
                    # Log retry
                    logger.warning(
                        f"Retrying {func.__name__} after error: {str(e)} "
                        f"(attempt {attempt+1}/{max_retries+1})",
                        emoji_key="warning",
                        attempt=attempt+1,
                        max_attempts=max_retries+1,
                        error=str(e)
                    )
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
                    
                    # Increase delay for next retry
                    delay *= backoff_factor
                    if max_backoff:
                        delay = min(delay, max_backoff)
            
            # Shouldn't get here, but just in case
            raise exceptions[-1]
                
        return wrapper
    return decorator


async def map_async(
    func: Callable[[Any], Any],
    items: List[Any],
    concurrency: int = 10,
    chunk_size: Optional[int] = None
) -> List[Any]:
    """Map a function over items with limited concurrency.
    
    Args:
        func: Function to apply
        items: List of items to process
        concurrency: Maximum number of concurrent tasks
        chunk_size: Optional batch size for processing
        
    Returns:
        List of function results
    """
    if not items:
        return []
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    # Define task function
    async def process_item(item):
        async with semaphore:
            return await func(item)
    
    # If using chunks, process in batches
    if chunk_size:
        results = []
        # Process in chunks for better memory management
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i+chunk_size]
            chunk_results = await asyncio.gather(
                *(process_item(item) for item in chunk)
            )
            results.extend(chunk_results)
        return results
    else:
        # Process all at once with concurrency limit
        return await asyncio.gather(
            *(process_item(item) for item in items)
        )


class AsyncBatchProcessor:
    """Processor for batching async operations."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_concurrency: int = 5,
        flush_interval: Optional[float] = None
    ):
        """Initialize the batch processor.
        
        Args:
            batch_size: Maximum items per batch
            max_concurrency: Maximum concurrent batch operations
            flush_interval: Optional auto-flush interval in seconds
        """
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.flush_interval = flush_interval
        
        self.items = []
        self.flush_task = None
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
    async def add(self, item: Any):
        """Add an item to the batch.
        
        Args:
            item: Item to add
        """
        self.items.append(item)
        
        # Start flush task if needed
        if self.flush_interval and not self.flush_task:
            self.flush_task = asyncio.create_task(self._auto_flush())
            
        # Flush if batch is full
        if len(self.items) >= self.batch_size:
            await self.flush()
            
    async def flush(self) -> List[Any]:
        """Flush the current batch.
        
        Returns:
            Results from processing the batch
        """
        if not self.items:
            return []
            
        # Get current items
        items = self.items
        self.items = []
        
        # Cancel flush task if running
        if self.flush_task:
            self.flush_task.cancel()
            self.flush_task = None
            
        # Process the batch
        return await self._process_batch(items)
        
    async def _auto_flush(self):
        """Auto-flush task that runs periodically."""
        try:
            while True:
                await asyncio.sleep(self.flush_interval)
                if self.items:
                    await self.flush()
        except asyncio.CancelledError:
            # Task was cancelled, which is expected
            pass
            
    async def _process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a batch of items.
        
        Args:
            batch: Items to process
            
        Returns:
            List of processed results
        """
        # This should be overridden by subclasses
        logger.warning(
            f"Default batch processing used for {len(batch)} items",
            emoji_key="warning"
        )
        return batch
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Flush any remaining items
        if self.items:
            await self.flush()