#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Random generator registry for managing mtype-specific random generators.

This module provides a centralized registry for managing random number generators
for different fuzzy number types (mtypes). It follows AxisFuzzy's extensible
architecture based on mtype specialization and enables automatic registration
and discovery of random generators.

The registry system serves as the foundation for AxisFuzzy's plugin-style
random generation architecture, where each fuzzy number type can provide
its own specialized random generator while maintaining a unified interface.

Architecture
------------
The registry follows these key design principles:

- **Singleton Pattern**: Global registry ensures consistent state across the library
- **Thread Safety**: All operations are protected by locks for concurrent access
- **Automatic Registration**: Generators can self-register using the `@register_random` decorator
- **Type Safety**: Validates generator types and mtype consistency during registration

The registry maintains a mapping from mtype strings to generator instances,
enabling dynamic dispatch based on the requested fuzzy number type.

Classes
-------
RandomGeneratorRegistry
    Thread-safe singleton registry for managing generator instances.

Functions
---------
register_random : decorator
    Class decorator for automatic generator registration.
get_random_generator : function
    Retrieve a registered generator by mtype.
list_registered_random : function
    List all registered mtypes.
is_registered_random : function
    Check if an mtype has a registered generator.

See Also
--------
axisfuzzy.random.base : Abstract base classes for generators
axisfuzzy.random.api : High-level API for random generation
axisfuzzy.fuzztype.qrofn.random : Example generator implementation

Examples
--------
Registering a generator using the decorator:

.. code-block:: python

    from axisfuzzy.random.base import ParameterizedRandomGenerator
    from axisfuzzy.random.registry import register_random

    @register_random
    class MyRandomGenerator(ParameterizedRandomGenerator):
        mtype = "mytype"

        def get_default_parameters(self):
            return {'param1': 1.0}

        def validate_parameters(self, **params):
            pass

        def fuzznum(self, rng, **params):
            # Implementation
            pass

        def fuzzarray(self, rng, shape, **params):
            # Implementation
            pass

Using the registry programmatically:

.. code-block:: python

    from axisfuzzy.random.registry import (
        get_registry_random, get_random_generator, list_registered_random
    )

    # Check available generators
    print("Available generators:", list_registered_random())

    # Get a specific generator
    generator = get_random_generator('qrofn')
    if generator is not None:
        # Use the generator...
        pass

    # Manual registration (not recommended)
    registry = get_registry_random()
    registry.register('custom_type', my_generator_instance)

Notes
-----
- Generators are typically registered automatically using the `@register_random` decorator
- Manual registration is available but not recommended for standard use cases
- The registry is thread-safe and can be accessed concurrently
- Generator instances are cached, so registration should occur at module import time
"""

import threading
from typing import Dict, List, Optional
from .base import BaseRandomGenerator


class RandomGeneratorRegistry:
    """
    Thread-safe registry for random generators.

    This registry manages the mapping between fuzzy number types (mtypes)
    and their corresponding random generator instances. It ensures thread-safe
    access and follows the singleton pattern for global consistency across
    the entire AxisFuzzy library.

    The registry serves as the central dispatch point for the random generation
    system, enabling automatic discovery and instantiation of appropriate
    generators based on the requested mtype.

    Attributes
    ----------
    _instance : RandomGeneratorRegistry or None
        Singleton instance reference.
    _lock : threading.RLock
        Class-level lock for singleton creation.

    Notes
    -----
    This class implements the singleton pattern to ensure a single global
    registry instance. All methods are thread-safe and can be called
    concurrently from multiple threads.

    The registry maintains generator instances (not classes), so generators
    should be stateless or properly handle concurrent access.

    Examples
    --------
    Basic registry usage:

    .. code-block:: python

        # Get the global registry instance
        registry = get_registry_random()

        # Check what generators are available
        print("Registered mtypes:", registry.list_mtypes())

        # Get a specific generator
        gen = registry.get_generator('qrofn')

        # Check if a type is registered
        if registry.is_registered('custom_type'):
            print("Custom type is available")

    Manual registration (advanced usage):

    .. code-block:: python

        from axisfuzzy.random.base import BaseRandomGenerator

        class CustomGenerator(BaseRandomGenerator):
            mtype = "custom"
            # ... implementation

        registry = get_registry_random()
        registry.register('custom', CustomGenerator())
    """

    _instance: Optional['RandomGeneratorRegistry'] = None
    _lock = threading.RLock()

    def __new__(cls) -> 'RandomGeneratorRegistry':
        """
        Create or return the singleton registry instance.

        Ensures that only one registry instance exists throughout the
        application lifetime, providing global consistency for generator
        management.

        Returns
        -------
        RandomGeneratorRegistry
            The singleton registry instance.

        Notes
        -----
        This method is thread-safe and uses double-checked locking
        to minimize synchronization overhead after initialization.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize the registry if not already initialized.

        This method is idempotent - multiple calls have no additional
        effect after the first initialization. This is necessary because
        `__init__` is called every time the singleton is accessed.
        """
        if hasattr(self, '_initialized'):
            return

        self._generators: Dict[str, 'BaseRandomGenerator'] = {}
        self._registry_lock = threading.RLock()
        self._initialized = True

    def register(self, mtype: str, generator: BaseRandomGenerator) -> None:
        """
        Register a random generator for a specific mtype.

        Associates a generator instance with a fuzzy number type identifier,
        making it available for random generation requests. The generator
        must implement the BaseRandomGenerator interface.

        Parameters
        ----------
        mtype : str
            The fuzzy number type identifier (e.g., 'qrofn', 'ivfn').
            Must be a non-empty string and unique within the registry.
        generator : BaseRandomGenerator
            The generator instance to register. Must implement the
            BaseRandomGenerator interface with all required methods.

        Raises
        ------
        TypeError
            If generator is not a BaseRandomGenerator instance.
        ValueError
            If mtype is empty, already registered, or if the generator's
            mtype attribute doesn't match the registration mtype.

        Notes
        -----
        This method validates that the generator's `mtype` attribute
        matches the registration `mtype` parameter to prevent
        configuration errors.

        Registration is thread-safe and atomic - either the generator
        is fully registered or not at all.

        Examples
        --------
        .. code-block:: python

            # Manual registration (typically not needed)
            generator = MyCustomGenerator()
            registry = get_registry_random()
            registry.register('mytype', generator)

            # Verify registration
            assert registry.is_registered('mytype')
            assert registry.get_generator('mytype') is generator
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

        Removes the generator associated with the given mtype from the
        registry. After unregistration, requests for this mtype will
        return None.

        Parameters
        ----------
        mtype : str
            The fuzzy number type identifier to unregister.

        Returns
        -------
        bool
            True if the generator was successfully unregistered,
            False if the mtype was not registered.

        Notes
        -----
        This method is thread-safe and idempotent - multiple calls
        with the same mtype have no additional effect.

        Unregistration is typically not needed in normal usage,
        as generators are usually registered once at module import
        time and persist for the application lifetime.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_random()

            # Unregister a type
            success = registry.unregister('mytype')
            if success:
                print("Successfully unregistered mytype")
            else:
                print("mytype was not registered")

            # Verify unregistration
            assert not registry.is_registered('mytype')
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

        Retrieves the generator instance associated with the given
        fuzzy number type. This is the primary method used by the
        random generation API to obtain the appropriate generator.

        Parameters
        ----------
        mtype : str
            The fuzzy number type identifier.

        Returns
        -------
        BaseRandomGenerator or None
            The registered generator instance, or None if no generator
            is registered for the given mtype.

        Notes
        -----
        This method is thread-safe and can be called concurrently.
        The returned generator instance should be stateless or
        handle concurrent access appropriately.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_random()

            # Get a generator
            qrofn_gen = registry.get_generator('qrofn')
            if qrofn_gen is not None:
                # Use the generator
                fuzznum = qrofn_gen.fuzznum(rng, q=2, md_low=0.1)
            else:
                print("No generator available for qrofn")
        """
        if not mtype:
            return None

        with self._registry_lock:
            return self._generators.get(mtype)

    def is_registered(self, mtype: str) -> bool:
        """
        Check if a generator is registered for the given mtype.

        Provides a quick way to test generator availability without
        retrieving the actual generator instance.

        Parameters
        ----------
        mtype : str
            The fuzzy number type identifier.

        Returns
        -------
        bool
            True if a generator is registered for the mtype,
            False otherwise.

        Notes
        -----
        This method is thread-safe and more efficient than calling
        `get_generator()` when you only need to check availability.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_random()

            # Check availability before use
            if registry.is_registered('qrofn'):
                generator = registry.get_generator('qrofn')
                # Use generator...
            else:
                raise ValueError("QROFN generator not available")
        """
        if not mtype:
            return False

        with self._registry_lock:
            return mtype in self._generators

    def list_mtypes(self) -> List[str]:
        """
        Get a list of all registered mtypes.

        Returns all currently registered fuzzy number type identifiers,
        useful for introspection and validation.

        Returns
        -------
        list of str
            A sorted list of registered mtype identifiers.
            The list is a copy, so modifications don't affect the registry.

        Notes
        -----
        The returned list is sorted for consistent ordering across
        calls and is safe to modify without affecting the registry.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_random()

            # List all available types
            available_types = registry.list_mtypes()
            print("Available random generators:", available_types)

            # Check if specific types are available
            required_types = ['qrofn', 'ivfn']
            missing = [t for t in required_types if t not in available_types]
            if missing:
                print(f"Missing generators: {missing}")
        """
        with self._registry_lock:
            return sorted(self._generators.keys())

    def clear(self) -> None:
        """
        Clear all registered generators.

        Removes all generators from the registry. This method is primarily
        useful for testing or when completely reinitializing the system.

        Notes
        -----
        This operation is thread-safe but should be used with caution
        as it affects the global state of the random generation system.

        After clearing, all random generation requests will fail until
        generators are re-registered.

        Examples
        --------
        .. code-block:: python

            # Typically only used in testing
            registry = get_registry_random()
            registry.clear()

            assert len(registry) == 0
            assert registry.list_mtypes() == []
        """
        with self._registry_lock:
            self._generators.clear()

    def __len__(self) -> int:
        """
        Get the number of registered generators.

        Returns
        -------
        int
            The number of currently registered generators.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_random()
            print(f"Registry contains {len(registry)} generators")
        """
        with self._registry_lock:
            return len(self._generators)

    def __contains__(self, mtype: str) -> bool:
        """
        Check if mtype is registered using 'in' operator.

        Enables Pythonic membership testing with the 'in' operator.

        Parameters
        ----------
        mtype : str
            The fuzzy number type identifier.

        Returns
        -------
        bool
            True if the mtype is registered, False otherwise.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_random()

            if 'qrofn' in registry:
                print("QROFN generator is available")
        """
        return self.is_registered(mtype)

    def __repr__(self) -> str:
        """
        String representation of the registry.

        Returns
        -------
        str
            A readable representation showing registered mtypes.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_random()
            print(registry)
            # Output: RandomGeneratorRegistry(mtypes=['qrofn', 'ivfn'])
        """
        with self._registry_lock:
            mtypes = list(self._generators.keys())
            return f"RandomGeneratorRegistry(mtypes={mtypes})"


# Global registry instance
_global_registry: Optional[RandomGeneratorRegistry] = None
_global_lock = threading.RLock()


def get_registry_random() -> RandomGeneratorRegistry:
    """
    Get the global random generator registry instance.

    This function provides access to the singleton registry instance
    used throughout AxisFuzzy for managing random generators. It ensures
    that all parts of the library use the same registry instance.

    Returns
    -------
    RandomGeneratorRegistry
        The singleton registry instance.

    Notes
    -----
    This function is thread-safe and uses lazy initialization - the
    registry is created on first access. The same instance is returned
    on all subsequent calls.

    Examples
    --------
    .. code-block:: python

        # Get the global registry
        registry = get_registry_random()

        # Use registry methods
        generators = registry.list_mtypes()
        print(f"Available generators: {generators}")

        # Check for specific generator
        if 'qrofn' in registry:
            qrofn_gen = registry.get_generator('qrofn')
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
    it with the global registry based on the class's `mtype` attribute. This is
    the recommended way to register generators as it ensures they are available
    as soon as the module is imported.

    Parameters
    ----------
    cls : type[BaseRandomGenerator]
        The generator class to register. Must be a subclass of BaseRandomGenerator
        and have a non-empty `mtype` class attribute.

    Returns
    -------
    type[BaseRandomGenerator]
        The original class (unmodified) for continued use.

    Raises
    ------
    TypeError
        If the class doesn't have an `mtype` attribute or if the mtype is empty.

    Notes
    -----
    The decorator instantiates the class with no arguments, so generator
    classes should not require constructor parameters. All configuration
    should be handled through the parameter system.

    The registration happens at decoration time (typically module import),
    making the generator immediately available for use.

    Examples
    --------
    Basic generator registration:

    .. code-block:: python

        from axisfuzzy.random.base import ParameterizedRandomGenerator
        from axisfuzzy.random.registry import register_random

        @register_random
        class QROFNRandomGenerator(ParameterizedRandomGenerator):
            mtype = "qrofn"

            def get_default_parameters(self):
                return {
                    'md_dist': 'uniform',
                    'md_low': 0.0,
                    'md_high': 1.0,
                    'nu_mode': 'orthopair'
                }

            # ... other required methods

    After decoration, the generator is immediately available:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Generator is now available for use
        num = fr.rand('qrofn', q=2)
        arr = fr.rand('qrofn', shape=(100,), q=3)
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

    Removes the generator associated with the given mtype from the
    global registry. This is typically not needed in normal usage.

    Parameters
    ----------
    mtype : str
        The fuzzy number type identifier to unregister.

    Returns
    -------
    bool
        True if successfully unregistered, False if the mtype
        was not registered.

    Examples
    --------
    .. code-block:: python

        # Unregister a generator
        success = unregister_random('qrofn')
        if success:
            print("QROFN generator unregistered")
    """
    registry = get_registry_random()
    return registry.unregister(mtype)


def get_random_generator(mtype: str) -> Optional['BaseRandomGenerator']:
    """
    Get the registered random generator for a specific mtype.

    This is the primary function used by the random generation API
    to obtain the appropriate generator for a given fuzzy number type.

    Parameters
    ----------
    mtype : str
        The fuzzy number type identifier.

    Returns
    -------
    BaseRandomGenerator or None
        The generator instance, or None if not registered.

    Examples
    --------
    .. code-block:: python

        # Get a generator directly
        generator = get_random_generator('qrofn')
        if generator is not None:
            # Use the generator for custom generation
            rng = np.random.default_rng(42)
            custom_num = generator.fuzznum(rng, q=3, md_low=0.2)
        else:
            print("QROFN generator not available")
    """
    registry = get_registry_random()
    return registry.get_generator(mtype)


def list_registered_random() -> List[str]:
    """
    Get a list of all registered mtypes.

    Returns all currently available fuzzy number types that have
    registered random generators.

    Returns
    -------
    list of str
        A sorted list of registered mtype identifiers.

    Examples
    --------
    .. code-block:: python

        # Check what generators are available
        available = list_registered_random()
        print("Available random generators:", available)

        # Use in validation
        def validate_mtype(mtype):
            if mtype not in available:
                raise ValueError(f"No generator for mtype '{mtype}'. "
                               f"Available: {available}")
    """
    registry = get_registry_random()
    return registry.list_mtypes()


def is_registered_random(mtype: str) -> bool:
    """
    Check if a random generator is registered for the given mtype.

    Provides a quick way to test generator availability without
    retrieving the actual generator instance.

    Parameters
    ----------
    mtype : str
        The fuzzy number type identifier.

    Returns
    -------
    bool
        True if registered, False otherwise.

    Examples
    --------
    .. code-block:: python

        # Check before using
        if is_registered_random('qrofn'):
            num = fr.rand('qrofn', q=2)
        else:
            raise ValueError("QROFN random generation not available")

        # Use in conditional logic
        supported_types = [t for t in ['qrofn', 'ivfn', 'pfn']
                          if is_registered_random(t)]
    """
    registry = get_registry_random()
    return registry.is_registered(mtype)


# Convenience aliases for backward compatibility and ease of use
registry = get_registry_random
register_random = register_random
unregister_random = unregister_random
get_generator = get_random_generator
list_registered = list_registered_random
is_registered = is_registered_random
