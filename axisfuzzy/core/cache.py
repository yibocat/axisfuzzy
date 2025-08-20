#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Lightweight Least Recently Used (LRU) cache used by core components.

This module provides a minimal LRU cache implementation backed by
collections.OrderedDict. It is intended for small, fast caches such as
operation-result caching inside FuzznumStrategy instances.
"""

from collections import OrderedDict
from typing import Any, Optional

from axisfuzzy.config import get_config


class LruCache:
    """
    Simple Least Recently Used (LRU) cache.

    The cache stores mapping from keys to values and evicts the least
    recently used item when capacity is exceeded.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of items to hold in the cache. If ``None``, the value
        from ``axisfuzzy.config.get_config().CACHE_SIZE`` is used.

    Attributes
    ----------
    maxsize : int
        Configured maximum number of entries.
    _cache : collections.OrderedDict
        Internal mapping preserving access/insertion order.

    Notes
    -----
    - Keys must be hashable.
    - ``get`` returns ``None`` for missing keys (no exception raised).

    Examples
    --------
    >>> from axisfuzzy.core.cache import LruCache
    >>> c = LruCache(maxsize=2)
    >>> c.put('a', 1)
    >>> c.put('b', 2)
    >>> c.get('a')
    1
    >>> c.put('c', 3)  # least recently used ('b') evicted if 'a' was accessed
    >>> 'b' in c
    False
    """

    def __init__(self, maxsize: Optional[int] = None):
        """
        Initialize an LRU cache.

        Parameters
        ----------
        maxsize : int or None, optional
            Maximum number of cached items. When ``None``, value is read from
            configuration (``get_config().CACHE_SIZE``).
        """
        self._cache = OrderedDict()
        # Set the maximum size of the cache. If maxsize is not provided,
        # it defaults to the CACHE_SIZE defined in the FuzzLab configuration.
        self.maxsize = get_config().CACHE_SIZE if maxsize is None else maxsize

    def get(self, key: str) -> Any:
        """
        Retrieve a value and mark the key as recently used.

        Parameters
        ----------
        key : str
            Key to lookup in the cache.

        Returns
        -------
        Any or None
            The stored value if the key exists, otherwise ``None``.

        Examples
        --------
        >>> c = LruCache(maxsize=2)
        >>> c.put('x', 10)
        >>> c.get('x')
        10
        >>> c.get('missing') is None
        True
        """
        if key not in self._cache:
            return None
        # Move the accessed item to the end to mark it as recently used.
        # This is the core LRU mechanism: recently accessed items are at the "hot" end.
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """
        Insert or update a key-value pair in the cache.

        If the key already exists, it is updated and moved to the most-recently-used
        position. If adding a new item causes the cache to exceed ``maxsize``,
        the least recently used item is evicted.

        Parameters
        ----------
        key : str
            Cache key to insert or update.
        value : Any
            Value to associate with the key.

        Raises
        ------
        ValueError
            If ``maxsize`` is configured as a non-positive integer (depends on config).

        Examples
        --------
        >>> c = LruCache(maxsize=1)
        >>> c.put('a', 1)
        >>> c.put('a', 2)  # update existing key
        >>> c.get('a')
        2
        """
        if key in self._cache:
            # If the key already exists, move it to the end to update its recency.
            self._cache.move_to_end(key)
        # Add or update the key-value pair. If the key was new, it's added at the end.
        self._cache[key] = value
        # Check if the current size of the cache exceeds the maximum allowed size.
        if len(self._cache) > self.maxsize:
            # If it does, remove the least recently used item.
            # `popitem(last=False)` removes and returns the first (oldest/least recently used) item.
            self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        """
        Check whether a key exists in the cache.

        Enables ``in`` operator usage.

        Parameters
        ----------
        key : str
            Key to check for existence.

        Returns
        -------
        bool
            True if the key is in the cache, False otherwise.

        Examples
        --------
        >>> c = LruCache(maxsize=1)
        >>> c.put('k', 1)
        >>> 'k' in c
        True
        """
        return key in self._cache
