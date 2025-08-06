#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/6 16:32
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module defines the `ExtendFunctionRegistry` class, a singleton registry
for managing and retrieving external functions that can be dynamically
injected into FuzzLab's core classes (Fuzznum, Fuzzarray) and the top-level
`fuzzlab` namespace.

It supports registration of functions based on fuzzy number types (mtype)
and specifies how these functions should be injected (as instance methods,
top-level functions, or both).
"""
from typing import Callable, Dict, Any, Literal, Optional, cast


class ExtendFunctionRegistry:
    """
    A singleton registry for managing external functions in FuzzLab.

    This registry allows developers to register different implementations of
    a function (e.g., 'distance', 'random') based on the `mtype` of the
    fuzzy number. It also controls how these functions are exposed to the user
    (as instance methods on Fuzznum/Fuzzarray objects, or as top-level
    functions in the `fuzzlab` module).

    Attributes:
        _instance (ExtendFunctionRegistry): The singleton instance of the registry.
        _functions (Dict[str, Dict[str, Callable]]): Stores specific function
            implementations. Keys are function names, values are dictionaries
            mapping `mtype` strings to their respective callable implementations.
            Example: {'distance': {'qrofn': qrofn_distance_func, 'ivfn': ivfn_distance_func}}
        _defaults (Dict[str, Callable]): Stores default implementations for
            functions. Keys are function names, values are the default callable
            implementations used when no `mtype`-specific implementation is found.
            Example: {'distance': default_distance_func}
        _metadata (Dict[str, Dict[str, Any]]): Stores additional metadata for
            each registered function, such as `target_classes`.
            Example: {'distance': {'target_classes': ['Fuzznum', 'Fuzzarray']}}
        _injection_types (Dict[str, Literal['instance_method', 'top_level_function', 'both']]):
            Specifies how each function should be injected. Keys are function names,
            values indicate the injection type.
            Example: {'distance': 'both', 'normalize': 'instance_method'}
    """
    # The core data structure of the registry will be a nested dictionary
    #   used to store all implementations.：
    # { 'function_name': { 'mtype': implementation_callable } }
    _instance = None

    _functions: Dict[str, Dict[str, Callable]]
    _defaults: Dict[str, Callable]
    _metadata: Dict[str, Dict[str, Any]]
    _injection_types: Dict[str, Literal['instance_method', 'top_level_function', 'both']]

    def __new__(cls):
        """
        Ensures that only one instance of ExtendFunctionRegistry is created (singleton pattern).

        Returns:
            ExtendFunctionRegistry: The single instance of the registry.
        """
        if cls._instance is None:
            # Create the new instance of the class
            instance = super(ExtendFunctionRegistry, cls).__new__(cls)

            # Initialize the internal dictionaries on the new instance
            instance._functions = {}
            instance._defaults = {}
            instance._metadata = {}
            instance._injection_types = {}

            # Store the newly created instance as the singleton
            cls._instance = instance
        return cls._instance

    def register(self,
                 name: str,
                 mtype: Optional[str] = None,
                 is_default: bool = False,
                 injection_type: Literal['instance_method', 'top_level_function', 'both'] = 'both',
                 **meta) -> Callable:
        """
        A decorator to register an external function with the registry.

        This method allows developers to register specific implementations of a
        function for different fuzzy number types (`mtype`) or a default
        implementation. It also defines how the function will be exposed
        (as an instance method, a top-level function, or both).

        Args:
            name (str): The unified name of the function (e.g., 'distance', 'random').
            mtype (Optional[str]): The specific fuzzy number type (e.g., 'qrofn', 'ivfn')
                this implementation supports. Must be provided if `is_default` is False.
            is_default (bool): If True, this implementation will be used as the
                fallback when no `mtype`-specific implementation is found for `name`.
            injection_type (Literal['instance_method', 'top_level_function', 'both']):
                Specifies how the function should be injected:
                - 'instance_method': Injected as a method on Fuzznum/Fuzzarray instances.
                - 'top_level_function': Injected as a function in the `fuzzlab` module.
                - 'both': Injected as both an instance method and a top-level function.
            **meta: Additional metadata for the function, such as `target_classes`
                (e.g., `target_classes=['Fuzznum', 'Fuzzarray']`) which indicates
                which core classes this function applies to when injected as an
                instance method.

        Returns:
            Callable: The decorated function itself, allowing for function chaining.

        Raises:
            ValueError: If `injection_type` is invalid or if `mtype` is None
                for a non-default implementation.

        Examples:
            >>> # Registering a qrofn-specific distance function
            >>> @registry.register('distance', mtype='qrofn', target_classes=['Fuzznum'])
            ... def qrofn_distance(fuzz1, fuzz2):
            ...     # ... qrofn specific distance logic ...
            ...     return 0.5
            >>>
            >>> # Registering a default random number generator
            >>> @registry.register('random', is_default=True, injection_type='top_level_function')
            ... def default_random_fuzznum(mtype, **kwargs):
            ...     # ... default random generation logic ...
            ...     return Fuzznum(...)
        """
        # Validate the provided injection_type
        if injection_type not in ['instance_method', 'top_level_function', 'both']:
            raise ValueError(f"Invalid injection_type: {injection_type}. "
                             f"Must be 'instance_method', 'top_level_function', or 'both'.")

        def decorator(func: Callable) -> Callable:
            # If it's a default implementation, store it in the _defaults dictionary
            if is_default:
                self._defaults[name] = func
            else:
                # For mtype-specific implementations, ensure mtype is provided
                if name not in self._functions:
                    self._functions[name] = {}
                if mtype is None:
                    raise ValueError("A specific mtype must be provided if not a default implementation.")
                # Store the function under its name and mtype
                self._functions[name][mtype] = func

            # Store or update metadata for the function
            if name not in self._metadata:
                self._metadata[name] = meta
            else:
                self._metadata[name].update(meta)

            # Store the specified injection type for the function
            self._injection_types[name] = injection_type

            return func

        return decorator

    def get_implementation(self, name: str, mtype: str) -> Optional[Callable]:
        """
        Retrieves the specific implementation of a function for a given mtype.

        If an mtype-specific implementation is not found, it falls back to the
        default implementation if one is registered.

        Args:
            name (str): The name of the function (e.g., 'distance').
            mtype (str): The fuzzy number type (e.g., 'qrofn', 'ivfn').

        Returns:
            Optional[Callable]: The callable implementation if found, otherwise None.
        """
        # First, try to find a mtype-specific implementation
        if name in self._functions and mtype in self._functions[name]:
            return self._functions[name][mtype]
        # If not found, return the default implementation if it exists
        return self._defaults.get(name)

    def get_injection_type(self, name: str) -> Literal['instance_method', 'top_level_function', 'both']:
        """
        Retrieves the specified injection type for a given function name.

        Args:
            name (str): The name of the function.

        Returns:
            Literal['instance_method', 'top_level_function', 'both']: The injection type.
                Defaults to 'both' if not explicitly set.
        """
        # Use typing.cast to explicitly tell the type checker that 'both' is a valid Literal value
        return self._injection_types.get(name, cast(Literal['instance_method', 'top_level_function', 'both'], 'both'))

    def get_injection(self) -> Dict[str, Literal['instance_method', 'top_level_function', 'both']]:
        """
        Retrieves a dictionary mapping function names to their registered injection types.

        This method is primarily used by the injection mechanism to determine
        how each registered function should be exposed.

        Returns:
            Dict[str, Literal['instance_method', 'top_level_function', 'both']]:
                A dictionary where keys are function names and values are their
                corresponding injection types.
        """
        return self._injection_types

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves a dictionary containing all registered function metadata.

        This method is primarily used by the injection mechanism to access
        additional information about registered functions, such as `target_classes`.

        Returns:
            Dict[str, Any]: A dictionary where keys are function names and values
                are dictionaries containing their associated metadata.
        """
        return self._metadata


# Global singleton instance of the ExtendFunctionRegistry.
# This instance is used throughout the FuzzLab library to register and retrieve
# external functions.
_registry = ExtendFunctionRegistry()


def get_extend_registry() -> ExtendFunctionRegistry:
    """
    Provides access to the global singleton instance of ExtendFunctionRegistry.

    Returns:
        ExtendFunctionRegistry: The singleton registry instance.
    """
    return _registry
