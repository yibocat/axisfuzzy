#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 02:27
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from collections import OrderedDict
from typing import Any, Optional

from fuzzlab.config import get_config


class LruCache:

    def __init__(self, maxsize: Optional[int] = None):
        self._cache = OrderedDict()
        self.maxsize = get_config().CACHE_SIZE if maxsize is None else maxsize

    def get(self, key: str) -> Any:
        """Get an item from the cache, mark it as recently used."""
        if key not in self._cache:
            return None
        # Move the accessed item to the end to mark it as recently used
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """Put an item into the cache."""
        if key in self._cache:
            # Move the existing key to the end
            self._cache.move_to_end(key)
        self._cache[key] = value
        # Check if the cache exceeds its size limit
        if len(self._cache) > self.maxsize:
            # Pop the least recently used item (from the front)
            self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._cache
