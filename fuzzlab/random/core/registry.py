#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 20:53
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import threading
from typing import Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RandomRegistry:
    """
    Registry for managing random generation functions for different mtypes.

    This class implements a thread-safe singleton registry that maps
    fuzzy number types (mtypes) to their corresponding random generation functions.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._registry = {}
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._fuzznum_generators: Dict[str, Callable] = {}
        self._initialized = True

    def register_fuzznum_generator(self, mtype: str, generator_func: Callable) -> None:
        """
        Register a random generation function for a specific mtype.

        Args:
            mtype: The fuzzy number type (e.g., 'qrofn').
            generator_func: Function that generates random Fuzznum instances.
                           Should accept (rng: np.random.Generator, **kwargs) as parameters.
        """
        with self._lock:
            if not callable(generator_func):
                raise ValueError(f"Generator function must be callable, got {type(generator_func)}")

            self._fuzznum_generators[mtype] = generator_func
            logger.debug(f"Registered random generator for mtype: {mtype}")

    def unregister_fuzznum_generator(self, mtype: str) -> bool:
        """
        Unregister a random generation function for a specific mtype.

        Args:
            mtype: The fuzzy number type to unregister.

        Returns:
            True if the generator was found and removed, False otherwise.
        """
        with self._lock:
            if mtype in self._fuzznum_generators:
                del self._fuzznum_generators[mtype]
                logger.debug(f"Unregistered random generator for mtype: {mtype}")
                return True
            return False

    def get_fuzznum_generator(self, mtype: str) -> Optional[Callable]:
        """
        Get the random generation function for a specific mtype.

        Args:
            mtype: The fuzzy number type.

        Returns:
            The generator function, or None if not found.
        """
        with self._lock:
            return self._fuzznum_generators.get(mtype)

    def list_registered_mtypes(self) -> list:
        """
        Get a list of all registered mtypes.

        Returns:
            List of registered mtype strings.
        """
        with self._lock:
            return list(self._fuzznum_generators.keys())

    def is_registered(self, mtype: str) -> bool:
        """
        Check if a random generator is registered for the given mtype.

        Args:
            mtype: The fuzzy number type to check.

        Returns:
            True if registered, False otherwise.
        """
        with self._lock:
            return mtype in self._fuzznum_generators

    def clear(self) -> None:
        """Clear all registered generators."""
        with self._lock:
            self._fuzznum_generators.clear()
            logger.debug("Cleared all registered random generators")


# Global registry instance
_registry_instance: Optional[RandomRegistry] = None
_registry_lock = threading.RLock()


def get_random_registry() -> RandomRegistry:
    """
    Get the global RandomRegistry instance.

    Returns:
        The singleton RandomRegistry instance.
    """
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = RandomRegistry()
    return _registry_instance


def register_random_generator(mtype: str, generator_func: Callable) -> None:
    """
    Convenience function to register a random generator.

    Args:
        mtype: The fuzzy number type.
        generator_func: The generator function.
    """
    get_random_registry().register_fuzznum_generator(mtype, generator_func)
