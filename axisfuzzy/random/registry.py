#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Random generator registry for managing mtype-specific random generators.

This module provides a centralized registry for managing random number generators
for different fuzzy number types (mtypes). It follows FuzzLab's extensible
architecture based on mtype specialization.
"""
import threading
from typing import Dict, List, Optional
from .base import BaseRandomGenerator


class RandomGeneratorRegistry:
    """
    Thread-safe registry for random generators.

    This registry manages the mapping between fuzzy number types (mtypes)
    and their corresponding random generator instances. It ensures thread-safe
    access and follows the singleton pattern for global consistency.
    """

    _instance: Optional['RandomGeneratorRegistry'] = None
    _lock = threading.RLock()

    def __new__(cls) -> 'RandomGeneratorRegistry':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry if not already initialized."""
        if hasattr(self, '_initialized'):
            return

        self._generators: Dict[str, 'BaseRandomGenerator'] = {}
        self._registry_lock = threading.RLock()
        self._initialized = True

    def register(self, mtype: str, generator: BaseRandomGenerator) -> None:
        """
        Register a random generator for a specific mtype.

        Args:
            mtype: The fuzzy number type identifier.
            generator: The generator instance to register.

        Raises:
            TypeError: If generator is not a BaseRandomGenerator instance.
            ValueError: If mtype is empty or already registered.
        """
        if not isinstance(generator, BaseRandomGenerator):
            raise TypeError(f"Generator must be an instance of BaseRandomGenerator, "
                            f"got {type(generator)}")

        if not mtype or not mtype.strip():
            raise ValueError("mtype cannot be empty")

        with self._registry_lock:
            if mtype in self._generators:
                raise ValueError(f"Random generator for mtype '{mtype}' is already registered")

            # Validate that generator's mtype matches registration mtype
            if hasattr(generator, 'mtype') and generator.mtype != mtype:
                raise ValueError(f"Generator mtype '{generator.mtype}' does not match "
                                 f"registration mtype '{mtype}'")

            self._generators[mtype] = generator

    def unregister(self, mtype: str) -> bool:
        """
        Unregister a random generator for a specific mtype.

        Args:
            mtype: The fuzzy number type identifier.

        Returns:
            True if the generator was successfully unregistered, False otherwise.
        """
        if not mtype:
            return False

        with self._registry_lock:
            if mtype in self._generators:
                del self._generators[mtype]
                return True
            return False

    def get_generator(self, mtype: str) -> Optional['BaseRandomGenerator']:
        """
        Get the random generator for a specific mtype.

        Args:
            mtype: The fuzzy number type identifier.

        Returns:
            The registered generator instance, or None if not found.
        """
        if not mtype:
            return None

        with self._registry_lock:
            return self._generators.get(mtype)

    def is_registered(self, mtype: str) -> bool:
        """
        Check if a generator is registered for the given mtype.

        Args:
            mtype: The fuzzy number type identifier.

        Returns:
            True if registered, False otherwise.
        """
        if not mtype:
            return False

        with self._registry_lock:
            return mtype in self._generators

    def list_mtypes(self) -> List[str]:
        """
        Get a list of all registered mtypes.

        Returns:
            A sorted list of registered mtype identifiers.
        """
        with self._registry_lock:
            return sorted(self._generators.keys())

    def clear(self) -> None:
        """Clear all registered generators."""
        with self._registry_lock:
            self._generators.clear()

    def __len__(self) -> int:
        """Get the number of registered generators."""
        with self._registry_lock:
            return len(self._generators)

    def __contains__(self, mtype: str) -> bool:
        """Check if mtype is registered using 'in' operator."""
        return self.is_registered(mtype)

    def __repr__(self) -> str:
        """String representation of the registry."""
        with self._registry_lock:
            mtypes = list(self._generators.keys())
            return f"RandomGeneratorRegistry(mtypes={mtypes})"


# Global registry instance
_global_registry: Optional[RandomGeneratorRegistry] = None
_global_lock = threading.RLock()


def get_registry_random() -> RandomGeneratorRegistry:
    """
    Get the global random generator registry instance.

    Returns:
        The singleton registry instance.
    """
    global _global_registry

    if _global_registry is None:
        with _global_lock:
            if _global_registry is None:
                _global_registry = RandomGeneratorRegistry()

    return _global_registry


def register_random(cls: type[BaseRandomGenerator]) -> type[BaseRandomGenerator]:
    """
    A class decorator to register a random generator.

    This decorator automatically instantiates the generator class and registers
    it with the global registry based on the class's `mtype` attribute.

    Usage:
        @register_random
        class MyRandomGenerator(BaseRandomGenerator):
            mtype = 'my_type'
            ...
    """
    if not hasattr(cls, 'mtype') or not cls.mtype:
        raise TypeError(f"Class {cls.__name__} must have a non-empty 'mtype' attribute to be registered.")

    # Instantiate the generator and register it
    generator_instance = cls()
    registry = get_registry_random()
    registry.register(cls.mtype, generator_instance)

    return cls


def unregister_random(mtype: str) -> bool:
    """
    Unregister a random generator for a specific mtype.

    Args:
        mtype: The fuzzy number type identifier.

    Returns:
        True if successfully unregistered, False otherwise.
    """
    registry = get_registry_random()
    return registry.unregister(mtype)


def get_random_generator(mtype: str) -> Optional['BaseRandomGenerator']:
    """
    Get the registered random generator for a specific mtype.

    Args:
        mtype: The fuzzy number type identifier.

    Returns:
        The generator instance, or None if not registered.
    """
    registry = get_registry_random()
    return registry.get_generator(mtype)


def list_registered_random() -> List[str]:
    """
    Get a list of all registered mtypes.

    Returns:
        A sorted list of registered mtype identifiers.
    """
    registry = get_registry_random()
    return registry.list_mtypes()


def is_registered_random(mtype: str) -> bool:
    """
    Check if a random generator is registered for the given mtype.

    Args:
        mtype: The fuzzy number type identifier.

    Returns:
        True if registered, False otherwise.
    """
    registry = get_registry_random()
    return registry.is_registered(mtype)


registry = get_registry_random
register_random = register_random
unregister_random = unregister_random
get_generator = get_random_generator
list_registered = list_registered_random
is_registered = is_registered_random
