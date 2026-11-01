# /// zerodep
# version = "0.3.1"
# deps = []
# tier = "simple"
# category = "process"
# note = "Install/update via `zerodep add retry`"
# ///

"""Zero-dependency retry with configurable backoff strategies.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Decorator-based retry with exponential / linear / fixed backoff,
jitter, exception and result filtering, and async support.

Basic usage::

    @retry(max_retries=3)
    def call_api():
        return get("https://api.example.com/data")

Async usage::

    @retry(max_retries=3, retry_on=(ConnectionError, TimeoutError))
    async def call_api():
        return await async_get("https://api.example.com/data")

Imperative usage::

    result = retry_call(call_api, max_retries=5)

HTTP status filtering::

    @retry(retry_on=retry_if_status(429, 502, 503))
    def call_api():
        resp = get("https://api.example.com/data")
        resp.raise_for_status()
        return resp
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import random
import time
from typing import Any, Callable

__all__ = [
    # Constants
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MAX_DELAY",
    "DEFAULT_BACKOFF_FACTOR",
    # Exceptions
    "RetryError",
    # Data classes
    "RetryState",
    # Predicates
    "retry_if_exception",
    "retry_if_result",
    "retry_if_status",
    # Main API
    "retry",
    "retry_call",
]

# ── Defaults ──

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_BACKOFF_FACTOR = 2.0


# ── Exceptions ──


class RetryError(Exception):
    """Raised when all retry attempts are exhausted.

    Attributes:
        last_exception: The exception from the final attempt, or ``None``
            if retries were triggered by result predicate.
        attempts: Total number of calls made (initial + retries).
    """

    def __init__(self, last_exception: BaseException | None, attempts: int) -> None:
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(
            f"Retry exhausted after {attempts} attempt(s)"
            + (f": {last_exception}" if last_exception else "")
        )


# ── Retry State ──


@dataclasses.dataclass
class RetryState:
    """Information about the current retry, passed to *on_retry* callback.

    Attributes:
        attempt: 1-based retry number (1 = first retry, not the initial call).
        exception: The exception that triggered this retry, or ``None``.
        result: The return value that triggered this retry, or ``None``.
        delay: Seconds to sleep before the next attempt.
        elapsed: Seconds elapsed since the initial call.
    """

    attempt: int
    exception: BaseException | None
    result: Any
    delay: float
    elapsed: float


# ── Retry Condition Helpers ──


def retry_if_exception(
    *exc_types: type[BaseException],
) -> Callable[[BaseException], bool]:
    """Build a predicate that matches specific exception types.

    Args:
        *exc_types: Exception classes to retry on.

    Returns:
        A callable ``(exc) -> bool``.
    """

    def predicate(exc: BaseException) -> bool:
        return isinstance(exc, exc_types)

    return predicate


def retry_if_result(predicate: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Mark a callable as a result-retry predicate (identity helper).

    Args:
        predicate: A callable ``(result) -> bool`` returning ``True`` to retry.

    Returns:
        The same callable, for self-documenting call sites.
    """
    return predicate


def retry_if_status(*status_codes: int) -> Callable[[BaseException], bool]:
    """Build a predicate that retries on HTTP status codes.

    Works with any exception carrying a ``status_code`` attribute
    (e.g. ``httpclient.HTTPError``).

    Args:
        *status_codes: HTTP status codes to retry on (e.g. 429, 502, 503).

    Returns:
        A callable ``(exc) -> bool``.
    """
    codes = set(status_codes)

    def predicate(exc: BaseException) -> bool:
        sc = getattr(exc, "status_code", None)
        return sc is not None and sc in codes

    return predicate


# ── Internal Helpers ──


# Cache frequently used functions as module-level references to avoid
# repeated global/attribute lookups on the hot path.
_random = random.random
_min = min


def _compute_delay(
    attempt: int,
    backoff: str,
    base_delay: float,
    backoff_factor: float,
    max_delay: float,
    jitter: str,
) -> float:
    """Compute the sleep duration for a given retry attempt.

    Args:
        attempt: 0-based retry index (0 = first retry).
        backoff: Strategy name.
        base_delay: Base delay in seconds.
        backoff_factor: Multiplier for exponential backoff.
        max_delay: Upper bound on delay.
        jitter: Jitter mode.

    Returns:
        Sleep duration in seconds.

    Raises:
        ValueError: On unknown *backoff* or *jitter* value.
    """
    if backoff == "exponential":
        delay = base_delay * (backoff_factor**attempt)
    elif backoff == "linear":
        delay = base_delay * (attempt + 1)
    elif backoff == "fixed":
        delay = base_delay
    else:
        raise ValueError(f"Unknown backoff strategy: {backoff!r}")

    delay = _min(delay, max_delay)

    if jitter == "full":
        delay *= _random()
    elif jitter == "equal":
        half = delay * 0.5
        delay = half + half * _random()
    elif jitter == "none":
        pass
    else:
        raise ValueError(f"Unknown jitter mode: {jitter!r}")

    return delay


def _should_retry_exception(
    exc: BaseException,
    retry_on: tuple[type[BaseException], ...] | Callable[[BaseException], bool],
) -> bool:
    """Check whether *exc* matches the retry condition."""
    if isinstance(retry_on, tuple):
        return isinstance(exc, retry_on)
    return retry_on(exc)


# ── Core Retry Loops ──


def _retry_sync(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    backoff: str,
    backoff_factor: float,
    jitter: str,
    retry_on: tuple[type[BaseException], ...] | Callable[[BaseException], bool],
    retry_on_result: Callable[[Any], bool] | None,
    on_retry: Callable[[RetryState], None] | None,
    _start_attempt: int = 0,
    _start_time: float | None = None,
) -> Any:
    start = _start_time if _start_time is not None else time.monotonic()

    for attempt in range(_start_attempt, max_retries + 1):  # 0 = initial call
        try:
            result = fn(*args, **kwargs)

            if retry_on_result is not None and retry_on_result(result):
                if attempt >= max_retries:
                    raise RetryError(None, max_retries + 1)
                delay = _compute_delay(
                    attempt, backoff, base_delay, backoff_factor, max_delay, jitter
                )
                if on_retry:
                    on_retry(
                        RetryState(
                            attempt=attempt + 1,
                            exception=None,
                            result=result,
                            delay=delay,
                            elapsed=time.monotonic() - start,
                        )
                    )
                time.sleep(delay)
                continue

            return result

        except BaseException as exc:
            if not _should_retry_exception(exc, retry_on) or attempt >= max_retries:
                raise

            delay = _compute_delay(
                attempt, backoff, base_delay, backoff_factor, max_delay, jitter
            )
            if on_retry:
                on_retry(
                    RetryState(
                        attempt=attempt + 1,
                        exception=exc,
                        result=None,
                        delay=delay,
                        elapsed=time.monotonic() - start,
                    )
                )
            time.sleep(delay)


async def _retry_async(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    backoff: str,
    backoff_factor: float,
    jitter: str,
    retry_on: tuple[type[BaseException], ...] | Callable[[BaseException], bool],
    retry_on_result: Callable[[Any], bool] | None,
    on_retry: Callable[[RetryState], None] | None,
) -> Any:
    start = time.monotonic()

    for attempt in range(max_retries + 1):
        try:
            result = await fn(*args, **kwargs)

            if retry_on_result is not None and retry_on_result(result):
                if attempt >= max_retries:
                    raise RetryError(None, max_retries + 1)
                delay = _compute_delay(
                    attempt, backoff, base_delay, backoff_factor, max_delay, jitter
                )
                if on_retry:
                    on_retry(
                        RetryState(
                            attempt=attempt + 1,
                            exception=None,
                            result=result,
                            delay=delay,
                            elapsed=time.monotonic() - start,
                        )
                    )
                await asyncio.sleep(delay)
                continue

            return result

        except BaseException as exc:
            if not _should_retry_exception(exc, retry_on) or attempt >= max_retries:
                raise

            delay = _compute_delay(
                attempt, backoff, base_delay, backoff_factor, max_delay, jitter
            )
            if on_retry:
                on_retry(
                    RetryState(
                        attempt=attempt + 1,
                        exception=exc,
                        result=None,
                        delay=delay,
                        elapsed=time.monotonic() - start,
                    )
                )
            await asyncio.sleep(delay)


# ── Public API ──


def retry(
    fn: Callable[..., Any] | None = None,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff: str = "exponential",
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    jitter: str = "full",
    retry_on: tuple[type[BaseException], ...] | Callable[[BaseException], bool] = (
        Exception,
    ),
    retry_on_result: Callable[[Any], bool] | None = None,
    on_retry: Callable[[RetryState], None] | None = None,
) -> Callable[..., Any]:
    """Decorator that retries a function on failure with configurable backoff.

    Can be used with or without arguments::

        @retry
        def f(): ...

        @retry()
        def g(): ...

        @retry(max_retries=5, backoff="linear")
        def h(): ...

    Automatically detects async functions and uses ``asyncio.sleep``.

    Args:
        fn: The function to decorate (set automatically when used as
            ``@retry`` without parentheses).
        max_retries: Maximum number of retries (not counting the initial call).
        base_delay: Base delay in seconds before the first retry.
        max_delay: Upper bound on computed delay.
        backoff: Backoff strategy — ``"exponential"``, ``"linear"``, or
            ``"fixed"``.
        backoff_factor: Multiplier for exponential backoff.
        jitter: Jitter mode — ``"full"`` (uniform [0, delay]),
            ``"equal"`` (delay/2 + uniform [0, delay/2]), or ``"none"``.
        retry_on: Exception types or a callable ``(exc) -> bool`` deciding
            whether to retry.  Defaults to ``(Exception,)``.
        retry_on_result: Optional callable ``(result) -> bool``.  When it
            returns ``True`` the call is retried.
        on_retry: Optional callback invoked before each retry sleep with a
            :class:`RetryState` instance.

    Returns:
        The decorated function (sync or async, matching the original).

    Raises:
        RetryError: When retries are exhausted due to *retry_on_result*.
        The original exception: When retries are exhausted due to exceptions.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Cache the coroutine check at decoration time so the per-call
        # wrapper never touches ``inspect``.
        is_async = inspect.iscoroutinefunction(fn)

        if is_async:
            # Async path is not fast-pathed: coroutine/event-loop overhead
            # already dwarfs the ns-level savings from skipping monotonic/range.

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _retry_async(
                    fn,
                    args,
                    kwargs,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    backoff=backoff,
                    backoff_factor=backoff_factor,
                    jitter=jitter,
                    retry_on=retry_on,
                    retry_on_result=retry_on_result,
                    on_retry=on_retry,
                )

            return async_wrapper

        # Build a fast-path wrapper: when the first call succeeds and no
        # result predicate fires, return immediately without entering
        # the full _retry_sync loop (avoids time.monotonic, range, etc.).
        _check_result = retry_on_result
        _inner_retry = _retry_sync

        if _check_result is None:
            _should_retry = _should_retry_exception

            # Hot path – no result predicate.  Try the happy path first;
            # only fall into the full retry loop on exception.
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                try:
                    return fn(*args, **kwargs)
                except BaseException as first_exc:
                    if max_retries == 0 or not _should_retry(first_exc, retry_on):
                        raise
                    delay = _compute_delay(
                        0, backoff, base_delay, backoff_factor, max_delay, jitter
                    )
                    if on_retry:
                        on_retry(
                            RetryState(
                                attempt=1,
                                exception=first_exc,
                                result=None,
                                delay=delay,
                                elapsed=time.monotonic() - start,
                            )
                        )
                    time.sleep(delay)
                    return _inner_retry(
                        fn,
                        args,
                        kwargs,
                        max_retries=max_retries,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        backoff=backoff,
                        backoff_factor=backoff_factor,
                        jitter=jitter,
                        retry_on=retry_on,
                        retry_on_result=None,
                        on_retry=on_retry,
                        _start_attempt=1,
                        _start_time=start,
                    )
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return _inner_retry(
                    fn,
                    args,
                    kwargs,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    backoff=backoff,
                    backoff_factor=backoff_factor,
                    jitter=jitter,
                    retry_on=retry_on,
                    retry_on_result=_check_result,
                    on_retry=on_retry,
                )

        return sync_wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def retry_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    **retry_kwargs: Any,
) -> Any:
    """Call *fn* with retry logic without using a decorator.

    Args:
        fn: The callable to invoke.
        args: Positional arguments for *fn*.
        kwargs: Keyword arguments for *fn*.
        **retry_kwargs: Same keyword arguments accepted by :func:`retry`.

    Returns:
        The return value of *fn* on success.
    """
    if kwargs is None:
        kwargs = {}

    opts: dict[str, Any] = dict(
        max_retries=retry_kwargs.pop("max_retries", DEFAULT_MAX_RETRIES),
        base_delay=retry_kwargs.pop("base_delay", DEFAULT_BASE_DELAY),
        max_delay=retry_kwargs.pop("max_delay", DEFAULT_MAX_DELAY),
        backoff=retry_kwargs.pop("backoff", "exponential"),
        backoff_factor=retry_kwargs.pop("backoff_factor", DEFAULT_BACKOFF_FACTOR),
        jitter=retry_kwargs.pop("jitter", "full"),
        retry_on=retry_kwargs.pop("retry_on", (Exception,)),
        retry_on_result=retry_kwargs.pop("retry_on_result", None),
        on_retry=retry_kwargs.pop("on_retry", None),
    )

    if inspect.iscoroutinefunction(fn):
        return _retry_async(fn, args, kwargs, **opts)
    return _retry_sync(fn, args, kwargs, **opts)
