#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 02:27
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module provides a simple Least Recently Used (LRU) cache implementation.

The `LruCache` class allows storing a fixed number of key-value pairs,
automatically evicting the least recently accessed items when the cache
reaches its maximum size. This is useful for caching frequently accessed
data to improve performance.
"""
from collections import OrderedDict
from typing import Any, Optional

from fuzzlab.config import get_config


class LruCache:
    """A simple Least Recently Used (LRU) cache implementation.

    This cache stores key-value pairs and automatically evicts the least
    recently used items when the cache size exceeds its `maxsize`.
    It uses `collections.OrderedDict` internally to maintain insertion
    order and facilitate efficient LRU eviction.

    Attributes:
        _cache (OrderedDict): The internal dictionary storing cache items.
                              Keys are cache keys, values are cached data.
                              The order reflects access/insertion recency.
        maxsize (int): The maximum number of items the cache can hold.
                       If not specified during initialization, it defaults
                       to `get_config().CACHE_SIZE`.
    """

    def __init__(self, maxsize: Optional[int] = None):
        """Initializes the LRU cache.

        Args:
            maxsize (Optional[int]): The maximum number of items the cache can hold.
                                     If None, the size is taken from `fuzzlab.config.get_config().CACHE_SIZE`.
        """
        self._cache = OrderedDict()
        # Set the maximum size of the cache. If maxsize is not provided,
        # it defaults to the CACHE_SIZE defined in the FuzzLab configuration.
        self.maxsize = get_config().CACHE_SIZE if maxsize is None else maxsize

    def get(self, key: str) -> Any:
        """Retrieves an item from the cache and marks it as recently used.

        If the key is found in the cache, the corresponding item is moved to
        the end of the internal `OrderedDict` to signify its recent access.

        Args:
            key (str): The key of the item to retrieve.

        Returns:
            Any: The value associated with the key if found, otherwise None.
        """
        if key not in self._cache:
            return None
        # Move the accessed item to the end to mark it as recently used.
        # This is the core LRU mechanism: recently accessed items are at the "hot" end.
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """Inserts or updates an item in the cache.

        If the key already exists, its value is updated, and the item is
        moved to the end (most recently used position). If the key is new,
        it's added. If adding a new item causes the cache to exceed its
        `maxsize`, the least recently used item (at the beginning of the
        `OrderedDict`) is removed.

        Args:
            key (str): The key of the item to put into the cache.
            value (Any): The value to associate with the key.
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
        """Checks if a key is present in the cache.

        This allows using the `in` operator with `LruCache` instances.
        e.g., `if key in my_cache: ...`

        Args:
            key (str): The key to check for existence.

        Returns:
            bool: True if the key is in the cache, False otherwise.
        """
        return key in self._cache
