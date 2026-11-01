# /// zerodep
# version = "0.2.4"
# deps = []
# tier = "subsystem"
# category = "storage"
# note = "Install/update via `zerodep add cache`"
# ///
"""Zero-dependency caching with TTL, eviction policies, and async support.

stdlib only, Python 3.10+.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Provides: LRUCache, FIFOCache, LFUCache, TTLCache cache classes (MutableMapping),
``cached`` decorator with sync/async auto-detection, and convenience decorators
``lru_cache``, ``ttl_cache``, ``lfu_cache``, ``fifo_cache``.

Does NOT implement: TLRUCache (per-item TTL function), MRUCache, RRCache,
``cachedmethod``, ``condition`` parameter, multiple backends, serialization.

Example::

    from cache import lru_cache, ttl_cache, LRUCache, cached

    # Convenience decorator (like functools.lru_cache but with more policies)
    @lru_cache(maxsize=256)
    def fibonacci(n):
        return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)

    # TTL decorator
    @ttl_cache(maxsize=100, ttl=60)
    def fetch_config(key):
        return load_from_db(key)

    # Async support (auto-detected)
    @lru_cache(maxsize=128)
    async def fetch_data(url):
        return await async_get(url)

    # Direct cache class usage
    cache = LRUCache(maxsize=1024)
    cache["key"] = "value"

    # Advanced: cached() with explicit lock
    import threading
    @cached(LRUCache(128), lock=threading.Lock(), info=True)
    def thread_safe_compute(x):
        return expensive(x)
"""

from __future__ import annotations

import collections
import collections.abc
import functools
import inspect
import time
from typing import Any, Callable

# ── Key Functions ──────────────────────────────────────────────────────────


class _HashedTuple(tuple):
    """Tuple subclass that caches its hash value for fast repeated lookups."""

    __hashvalue: int | None = None

    def __hash__(self, hash=tuple.__hash__):  # type: ignore[override]
        hashvalue = self.__hashvalue
        if hashvalue is None:
            self.__hashvalue = hashvalue = hash(self)
        return hashvalue

    def __add__(self, other, add=tuple.__add__):
        return _HashedTuple(add(self, other))

    def __radd__(self, other, add=tuple.__add__):
        return _HashedTuple(add(other, self))


_KWMARK = (_HashedTuple,)


def hashkey(*args: Any, **kwargs: Any) -> _HashedTuple:
    """Return a cache key for the given positional and keyword arguments.

    This is the default key function for all cache decorators.
    """
    if kwargs:
        return _HashedTuple(args + _KWMARK + tuple(sorted(kwargs.items())))
    return _HashedTuple(args)


def methodkey(_self: Any, *args: Any, **kwargs: Any) -> _HashedTuple:
    """Like ``hashkey`` but ignores the first positional argument (``self``).

    Use as the key function when caching instance methods.
    """
    return hashkey(*args, **kwargs)


def typedkey(*args: Any, **kwargs: Any) -> _HashedTuple:
    """Like ``hashkey`` but includes type information, so ``f(3)`` and
    ``f(3.0)`` are cached separately.
    """
    if kwargs:
        sorted_items = tuple(sorted(kwargs.items()))
        return _HashedTuple(
            args
            + _KWMARK
            + sorted_items
            + tuple(type(v) for v in args)
            + tuple(type(v) for _, v in sorted_items)
        )
    return _HashedTuple(args + tuple(type(v) for v in args))


# ── Cache Info ─────────────────────────────────────────────────────────────

CacheInfo = collections.namedtuple(
    "CacheInfo", ["hits", "misses", "maxsize", "currsize"]
)

# ── Cache Base Class ───────────────────────────────────────────────────────


class _DefaultSize:
    """Lightweight size tracker that always returns 1 for any key."""

    __slots__ = ()

    def __getitem__(self, _: Any) -> int:
        return 1

    def __setitem__(self, _: Any, __: Any) -> None:
        pass

    def pop(self, _: Any, *args: Any) -> int:
        return 1

    def clear(self) -> None:
        pass


class Cache(collections.abc.MutableMapping):
    """Mutable mapping to serve as a simple cache or cache base class.

    Subclasses must override ``popitem`` to implement a specific eviction
    policy.  The base class evicts arbitrary items (dict order).

    Args:
        maxsize: Maximum capacity (item count or weighted size).
        getsizeof: Optional callable returning the size of a value.
            Defaults to 1 per item.
    """

    __marker = object()

    def __init__(
        self,
        maxsize: int | float,
        getsizeof: Callable[[Any], int] | None = None,
    ):
        if maxsize < 0:
            raise ValueError("maxsize must be non-negative")
        self.__data: dict[Any, Any] = {}
        self.__currsize: int | float = 0
        self.__maxsize = maxsize
        if getsizeof is not None:
            self.__size: dict | _DefaultSize = {}
            self.getsizeof = getsizeof  # type: ignore[assignment]
        else:
            self.__size = _DefaultSize()

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return (
            f"{cls}({self.__data!r}, "
            f"maxsize={self.__maxsize}, currsize={self.__currsize})"
        )

    def __getitem__(self, key: Any) -> Any:
        try:
            return self.__data[key]
        except KeyError:
            return self.__missing__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        maxsize = self.__maxsize
        size = self.getsizeof(value)
        if size > maxsize:
            raise ValueError("value too large")
        if key in self.__data:
            diffsize = size - self.__size[key]
        else:
            diffsize = size
        while self.__currsize + diffsize > maxsize:
            self.popitem()
        if key in self.__data:
            # Key might have been evicted during popitem loop
            self.__currsize -= self.__size[key]
        self.__data[key] = value
        self.__size[key] = size
        self.__currsize += size

    def __delitem__(self, key: Any) -> None:
        size = self.__size.pop(key)
        del self.__data[key]
        self.__currsize -= size

    def __contains__(self, key: object) -> bool:
        return key in self.__data

    def __missing__(self, key: Any) -> Any:
        raise KeyError(key)

    def __iter__(self):
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    @staticmethod
    def getsizeof(value: Any) -> int:
        """Return the size of *value*.  Defaults to 1."""
        return 1

    @property
    def maxsize(self) -> int | float:
        """Maximum cache capacity."""
        return self.__maxsize

    @property
    def currsize(self) -> int | float:
        """Current cache size (sum of item sizes)."""
        return self.__currsize

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default

    def pop(self, key: Any, default: Any = __marker) -> Any:
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default is self.__marker:
            raise KeyError(key)
        return default

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key in self:
            return self[key]
        try:
            self[key] = default
        except ValueError:
            pass
        return default

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return an arbitrary ``(key, value)`` pair.

        Subclasses override this to implement eviction policies.
        """
        try:
            key = next(iter(self.__data))
        except StopIteration:
            raise KeyError(f"{type(self).__name__} is empty") from None
        return key, self.pop(key)

    def clear(self) -> None:
        self.__data.clear()
        self.__size.clear()
        self.__currsize = 0


# ── FIFO Cache ─────────────────────────────────────────────────────────────


class FIFOCache(Cache):
    """First In First Out (FIFO) cache implementation.

    Evicts the oldest inserted item when the cache is full.  Accessing an
    item does **not** change its eviction priority.
    """

    def __init__(
        self,
        maxsize: int | float,
        getsizeof: Callable[[Any], int] | None = None,
    ):
        super().__init__(maxsize, getsizeof)
        self.__order: collections.OrderedDict[Any, None] = collections.OrderedDict()

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            del self.__order[key]
        super().__setitem__(key, value)
        self.__order[key] = None

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        del self.__order[key]

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return the oldest inserted ``(key, value)`` pair."""
        try:
            key = next(iter(self.__order))
        except StopIteration:
            raise KeyError(f"{type(self).__name__} is empty") from None
        return key, self.pop(key)

    def clear(self) -> None:
        super().clear()
        self.__order.clear()


# ── LRU Cache ──────────────────────────────────────────────────────────────


class LRUCache(Cache):
    """Least Recently Used (LRU) cache implementation.

    Evicts the least recently accessed item when the cache is full.
    Both reads and writes count as accesses.
    """

    def __init__(
        self,
        maxsize: int | float,
        getsizeof: Callable[[Any], int] | None = None,
    ):
        super().__init__(maxsize, getsizeof)
        self.__order: collections.OrderedDict[Any, None] = collections.OrderedDict()

    def __getitem__(self, key: Any) -> Any:
        # Bypass super().__getitem__ to avoid MRO dispatch overhead;
        # call Cache.__getitem__ directly + inline move_to_end.
        value = Cache.__getitem__(self, key)
        self.__order.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        Cache.__setitem__(self, key, value)
        try:
            self.__order.move_to_end(key)
        except KeyError:
            self.__order[key] = None

    def __delitem__(self, key: Any) -> None:
        Cache.__delitem__(self, key)
        del self.__order[key]

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return the least recently used ``(key, value)`` pair."""
        try:
            key = next(iter(self.__order))
        except StopIteration:
            raise KeyError(f"{type(self).__name__} is empty") from None
        return key, self.pop(key)

    def clear(self) -> None:
        Cache.clear(self)
        self.__order.clear()


# ── LFU Cache ──────────────────────────────────────────────────────────────


class _LFUNode:
    """Doubly-linked node for the LFU frequency list."""

    __slots__ = ("count", "keys", "prev", "next")

    def __init__(self, count: int = 0):
        self.count = count
        self.keys: set[Any] = set()
        self.prev: _LFUNode = self
        self.next: _LFUNode = self

    def insert_after(self, node: _LFUNode) -> None:
        """Insert *node* after this node."""
        node.prev = self
        node.next = self.next
        self.next.prev = node
        self.next = node

    def unlink(self) -> None:
        """Remove this node from the linked list."""
        self.prev.next = self.next
        self.next.prev = self.prev


class LFUCache(Cache):
    """Least Frequently Used (LFU) cache implementation.

    Evicts the least frequently accessed item when the cache is full.
    All operations are O(1).
    """

    def __init__(
        self,
        maxsize: int | float,
        getsizeof: Callable[[Any], int] | None = None,
    ):
        super().__init__(maxsize, getsizeof)
        self.__root = _LFUNode()  # sentinel
        self.__links: dict[Any, _LFUNode] = {}

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self.__touch(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        existed = key in self
        super().__setitem__(key, value)
        if existed:
            self.__touch(key)
        else:
            # New key starts at count=1
            root = self.__root
            first = root.next
            if first is root or first.count != 1:
                node = _LFUNode(count=1)
                root.insert_after(node)
            else:
                node = first
            node.keys.add(key)
            self.__links[key] = node

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        node = self.__links.pop(key)
        node.keys.discard(key)
        if not node.keys:
            node.unlink()

    def __touch(self, key: Any) -> None:
        """Increment frequency count for *key*."""
        node = self.__links[key]
        new_count = node.count + 1
        nxt = node.next
        if nxt is self.__root or nxt.count != new_count:
            new_node = _LFUNode(count=new_count)
            node.insert_after(new_node)
        else:
            new_node = nxt
        new_node.keys.add(key)
        node.keys.discard(key)
        self.__links[key] = new_node
        if not node.keys:
            node.unlink()

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return the least frequently used ``(key, value)`` pair."""
        root = self.__root
        node = root.next
        if node is root:
            raise KeyError(f"{type(self).__name__} is empty")
        key = next(iter(node.keys))
        # Bypass self.pop → self[key] → __touch to avoid wasteful
        # frequency increment on a key that is about to be deleted.
        value = Cache.__getitem__(self, key)
        del self[key]
        return key, value

    def clear(self) -> None:
        super().clear()
        self.__links.clear()
        self.__root.prev = self.__root
        self.__root.next = self.__root


# ── TTL Cache ──────────────────────────────────────────────────────────────


class _Timer:
    """Wraps a timer function with nestable time-freezing.

    Used by TTLCache to ensure time consistency within a single operation.
    """

    __slots__ = ("_timer", "_nesting", "_time")

    def __init__(self, timer: Callable[[], float] = time.monotonic):
        self._timer = timer
        self._nesting = 0
        self._time: float = 0.0

    def __call__(self) -> float:
        if self._nesting == 0:
            return self._timer()
        return self._time

    def __enter__(self) -> float:
        if self._nesting == 0:
            self._time = self._timer()
        self._nesting += 1
        return self._time

    def __exit__(self, *exc: Any) -> None:
        self._nesting -= 1


class _TTLLink:
    """Doubly-linked node for TTLCache expiry ordering."""

    __slots__ = ("key", "expires", "prev", "next")

    def __init__(
        self,
        key: Any = None,
        expires: float = 0.0,
    ):
        self.key = key
        self.expires = expires
        self.prev: _TTLLink = self
        self.next: _TTLLink = self

    def unlink(self) -> None:
        self.prev.next = self.next
        self.next.prev = self.prev


class TTLCache(LRUCache):
    """LRU cache with per-item time-to-live (TTL).

    Items are evicted when they expire (checked lazily on access) or when
    the cache is full (LRU order).

    Args:
        maxsize: Maximum cache capacity.
        ttl: Time-to-live in seconds for each item.
        timer: Callable returning the current time (default: ``time.monotonic``).
        getsizeof: Optional callable returning the size of a value.
    """

    def __init__(
        self,
        maxsize: int | float,
        ttl: float,
        *,
        timer: Callable[[], float] = time.monotonic,
        getsizeof: Callable[[Any], int] | None = None,
    ):
        LRUCache.__init__(self, maxsize, getsizeof)
        self.__ttl = ttl
        self.__timer = _Timer(timer)
        self.__root = _TTLLink()  # sentinel for expiry list
        self.__links: dict[Any, _TTLLink] = {}

    @property
    def ttl(self) -> float:
        """Time-to-live in seconds."""
        return self.__ttl

    @property
    def timer(self) -> _Timer:
        """The timer function."""
        return self.__timer

    def __contains__(self, key: object) -> bool:
        link = self.__links.get(key)
        if link is None:
            return False
        return self.__timer() < link.expires

    def __getitem__(self, key: Any) -> Any:
        link = self.__links.get(key)
        if link is not None and self.__timer() >= link.expires:
            del self[key]
            return self.__missing__(key)
        # Bypass LRUCache.__getitem__ → Cache.__getitem__ call chain;
        # inline both to save one method-call and one dict lookup.
        # Same direct-call pattern used by LFUCache.popitem.
        value = Cache.__getitem__(self, key)
        self._LRUCache__order.move_to_end(key)  # type: ignore[attr-defined]
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        with self.__timer as now:
            self.expire(now)
            LRUCache.__setitem__(self, key, value)
            # Update or create expiry link
            link = self.__links.get(key)
            if link is not None:
                link.unlink()
            else:
                link = _TTLLink()
                self.__links[key] = link
            link.key = key
            link.expires = now + self.__ttl
            # Append to end of expiry list (newest)
            tail = self.__root.prev
            link.prev = tail
            link.next = self.__root
            tail.next = link
            self.__root.prev = link

    def __delitem__(self, key: Any) -> None:
        LRUCache.__delitem__(self, key)
        link = self.__links.pop(key)
        link.unlink()

    def __iter__(self):
        now = self.__timer()
        for key in super().__iter__():
            link = self.__links.get(key)
            if link is not None and now < link.expires:
                yield key

    def __len__(self) -> int:
        now = self.__timer()
        # Single dict lookup via .get() instead of ``key in`` + ``[key]``.
        links_get = self.__links.get
        return sum(
            1
            for key in super().__iter__()
            if (link := links_get(key)) is not None and now < link.expires
        )

    @property
    def currsize(self) -> int | float:
        """Current size, excluding expired items."""
        self.expire()
        return super().currsize

    def expire(self, time: float | None = None) -> list[tuple[Any, Any]]:
        """Remove expired items and return them as a list of ``(key, value)`` pairs."""
        if time is None:
            time = self.__timer()
        root = self.__root
        expired: list[tuple[Any, Any]] = []
        link = root.next
        while link is not root and link.expires <= time:
            nxt = link.next
            key = link.key
            try:
                value = Cache.__getitem__(self, key)
                expired.append((key, value))
                del self[key]
            except KeyError:
                pass
            link = nxt
        return expired

    def clear(self) -> None:
        LRUCache.clear(self)
        self.__links.clear()
        self.__root.prev = self.__root
        self.__root.next = self.__root


# ── Decorators ─────────────────────────────────────────────────────────────

_MISS = object()  # sentinel for cache lookup miss


class _Stats:
    """Mutable hit/miss counters shared by cache wrapper closures."""

    __slots__ = ("hits", "misses")

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0


def _get_cache_maxsize(
    cache: collections.abc.MutableMapping,
) -> int | float | None:
    """Return maxsize if *cache* is a ``Cache`` instance, else ``None``."""
    return cache.maxsize if isinstance(cache, Cache) else None


def _get_cache_currsize(cache: collections.abc.MutableMapping) -> int | float:
    """Return current size of *cache*."""
    return cache.currsize if isinstance(cache, Cache) else len(cache)


def _attach_cache_info(
    wrapper: Callable,
    cache: collections.abc.MutableMapping,
    lock: Any | None,
    stats: _Stats,
    maxsize: int | float | None,
) -> None:
    """Attach ``cache_info()`` and ``cache_clear()`` methods to *wrapper*."""

    def cache_info() -> CacheInfo:
        return CacheInfo(stats.hits, stats.misses, maxsize, _get_cache_currsize(cache))

    def cache_clear() -> None:
        with lock if lock else _no_lock():
            cache.clear()
        stats.hits = stats.misses = 0

    wrapper.cache_info = cache_info  # type: ignore[attr-defined]
    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]


def _attach_noop_info(
    wrapper: Callable,
    stats: _Stats,
) -> None:
    """Attach ``cache_info()`` and ``cache_clear()`` for a no-op wrapper."""

    def cache_info() -> CacheInfo:
        return CacheInfo(stats.hits, stats.misses, None, 0)

    def cache_clear() -> None:
        stats.hits = stats.misses = 0

    wrapper.cache_info = cache_info  # type: ignore[attr-defined]
    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]


def _make_sync_wrapper(
    fn: Callable,
    cache: collections.abc.MutableMapping | None,
    key: Callable,
    lock: Any | None,
    info: bool,
) -> Callable:
    """Build a synchronous caching wrapper."""
    stats = _Stats()

    if cache is None:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stats.misses += 1
            return fn(*args, **kwargs)

        if info:
            _attach_noop_info(wrapper, stats)
        return wrapper

    # Pre-cache method references to avoid attribute lookup per call.
    cache_getitem = cache.__getitem__
    cache_setitem = cache.__setitem__
    cache_setdefault = cache.setdefault

    if lock is not None:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            k = key(*args, **kwargs)
            with lock:
                try:
                    result = cache_getitem(k)
                except KeyError:
                    result = _MISS
            if result is not _MISS:
                stats.hits += 1
                return result
            stats.misses += 1
            v = fn(*args, **kwargs)
            # Inline thread-safe store (was _cache_store_default).
            with lock:
                try:
                    return cache_setdefault(k, v)
                except ValueError:
                    return v

    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            k = key(*args, **kwargs)
            try:
                result = cache_getitem(k)
            except KeyError:
                stats.misses += 1
                v = fn(*args, **kwargs)
                try:
                    cache_setitem(k, v)
                except ValueError:
                    pass
                return v
            stats.hits += 1
            return result

    if info:
        _attach_cache_info(wrapper, cache, lock, stats, _get_cache_maxsize(cache))

    return wrapper


def _make_async_wrapper(
    fn: Callable,
    cache: collections.abc.MutableMapping | None,
    key: Callable,
    lock: Any | None,
    info: bool,
) -> Callable:
    """Build an asynchronous caching wrapper."""
    stats = _Stats()

    if cache is None:

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            stats.misses += 1
            return await fn(*args, **kwargs)

        if info:
            _attach_noop_info(wrapper, stats)
        return wrapper

    # Pre-cache method references to avoid attribute lookup per call.
    cache_getitem = cache.__getitem__
    cache_setitem = cache.__setitem__
    cache_setdefault = cache.setdefault

    if lock is not None:

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            k = key(*args, **kwargs)
            async with lock:
                try:
                    result = cache_getitem(k)
                except KeyError:
                    result = _MISS
            if result is not _MISS:
                stats.hits += 1
                return result
            stats.misses += 1
            v = await fn(*args, **kwargs)
            # Inline thread-safe store (was _cache_store_default).
            async with lock:
                try:
                    return cache_setdefault(k, v)
                except ValueError:
                    return v

    else:

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            k = key(*args, **kwargs)
            try:
                result = cache_getitem(k)
            except KeyError:
                stats.misses += 1
                v = await fn(*args, **kwargs)
                try:
                    cache_setitem(k, v)
                except ValueError:
                    pass
                return v
            stats.hits += 1
            return result

    if info:
        _attach_cache_info(wrapper, cache, lock, stats, _get_cache_maxsize(cache))

    return wrapper


class _no_lock:
    """Dummy context manager when no lock is provided."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *exc: Any) -> None:
        pass


def cached(
    cache: Cache | collections.abc.MutableMapping | None,
    *,
    key: Callable[..., Any] = hashkey,
    lock: Any | None = None,
    info: bool = False,
) -> Callable:
    """Decorator to wrap a function with a memoizing callable.

    Args:
        cache: A cache instance (any MutableMapping) or ``None`` to disable.
        key: Key function (default ``hashkey``).
        lock: A lock object (``threading.Lock`` for sync, ``asyncio.Lock``
            for async).  The lock is released while the wrapped function
            executes.
        info: If ``True``, attach ``cache_info()`` and ``cache_clear()``
            methods to the wrapper.
    """

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            wrapper = _make_async_wrapper(fn, cache, key, lock, info)
        else:
            wrapper = _make_sync_wrapper(fn, cache, key, lock, info)
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_key = key  # type: ignore[attr-defined]
        wrapper.cache_lock = lock  # type: ignore[attr-defined]
        return functools.update_wrapper(wrapper, fn)

    return decorator


# ── Convenience Decorators ─────────────────────────────────────────────────


def lru_cache(
    fn: Callable[..., Any] | None = None,
    *,
    maxsize: int = 128,
    key: Callable[..., Any] = hashkey,
    lock: Any | None = None,
    info: bool = True,
) -> Callable[..., Any]:
    """LRU cache decorator.  Auto-detects sync/async.

    Unlike ``functools.lru_cache``, supports TTL (via ``ttl_cache``),
    async functions, and explicit lock objects.

    Args:
        maxsize: Maximum number of cached results.
        key: Key function (default ``hashkey``).
        lock: Optional lock for thread/async safety.
        info: Attach ``cache_info()``/``cache_clear()`` (default ``True``).
    """

    def decorator(fn: Callable) -> Callable:
        c = LRUCache(maxsize)
        return cached(c, key=key, lock=lock, info=info)(fn)

    if fn is not None:
        return decorator(fn)
    return decorator


def fifo_cache(
    fn: Callable[..., Any] | None = None,
    *,
    maxsize: int = 128,
    key: Callable[..., Any] = hashkey,
    lock: Any | None = None,
    info: bool = True,
) -> Callable[..., Any]:
    """FIFO cache decorator.  Auto-detects sync/async.

    Evicts the oldest inserted item regardless of access pattern.
    """

    def decorator(fn: Callable) -> Callable:
        c = FIFOCache(maxsize)
        return cached(c, key=key, lock=lock, info=info)(fn)

    if fn is not None:
        return decorator(fn)
    return decorator


def lfu_cache(
    fn: Callable[..., Any] | None = None,
    *,
    maxsize: int = 128,
    key: Callable[..., Any] = hashkey,
    lock: Any | None = None,
    info: bool = True,
) -> Callable[..., Any]:
    """LFU cache decorator.  Auto-detects sync/async.

    Evicts the least frequently accessed item.
    """

    def decorator(fn: Callable) -> Callable:
        c = LFUCache(maxsize)
        return cached(c, key=key, lock=lock, info=info)(fn)

    if fn is not None:
        return decorator(fn)
    return decorator


def ttl_cache(
    fn: Callable[..., Any] | None = None,
    *,
    maxsize: int = 128,
    ttl: float = 600,
    timer: Callable[[], float] = time.monotonic,
    key: Callable[..., Any] = hashkey,
    lock: Any | None = None,
    info: bool = True,
) -> Callable[..., Any]:
    """TTL cache decorator.  Auto-detects sync/async.

    Items expire after *ttl* seconds (default 600 = 10 minutes).
    Eviction follows LRU order when the cache is full.
    """

    def decorator(fn: Callable) -> Callable:
        c = TTLCache(maxsize, ttl, timer=timer)
        return cached(c, key=key, lock=lock, info=info)(fn)

    if fn is not None:
        return decorator(fn)
    return decorator


# ── Public API ─────────────────────────────────────────────────────────────

__all__ = [
    # Cache classes
    "Cache",
    "FIFOCache",
    "LFUCache",
    "LRUCache",
    "TTLCache",
    # Decorators
    "cached",
    "fifo_cache",
    "lfu_cache",
    "lru_cache",
    "ttl_cache",
    # Key functions
    "hashkey",
    "methodkey",
    "typedkey",
    # Stats
    "CacheInfo",
]
