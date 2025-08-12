#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 14:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
Extension Registry for FuzzLab.

This module defines the core registry system for managing and retrieving
external functions (extensions) based on their `mtype` (fuzzy number type).
It supports specialized implementations for specific `mtype`s, fallback
to default implementations, and priority-based selection.

The registry is thread-safe and serves as the central repository for all
extension metadata and callable functions, which are registered via the
`@extension` decorator defined in `fuzzlab.extension.decorator.py`.
These registered functions are later injected into `Fuzznum` and `Fuzzarray`
classes or the `fuzzlab` top-level namespace by `fuzzlab.extension.injector.py`.
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Tuple, Callable, Any, Union
import threading
import datetime


@dataclass
class FunctionMetadata:
    """
    Dataclass to store metadata for a registered extension function.

    Attributes:
        name: The name of the extension function (e.g., 'distance', '_random').
        mtype: The specific fuzzy number type (e.g., 'qrofn', 'ivfn') this
            implementation is for. `None` indicates a general or default
            implementation.
        target_classes: A list of class names (e.g., ['Fuzznum', 'Fuzzarray'])
            that this extension is intended to be injected into as an instance method.
        injection_type: Specifies how the function should be injected:
            'instance_method': As a method of `target_classes`.
            'top_level_function': As a function in the `fuzzlab` module namespace.
            'both': Both as an instance method and a top-level function.
        is_default: A boolean indicating if this is a default implementation
            for the given `name`. Default implementations are used when no
            `mtype`-specific implementation is found.
        priority: An integer representing the priority of this implementation.
            Higher values indicate higher priority. Used for resolving conflicts
            when multiple implementations exist (though current logic primarily
            uses it for preventing lower priority re-registrations).
        description: An optional string providing a brief description of the function.
    """
    name: str
    mtype: Optional[str]
    target_classes: List[str]
    injection_type: Literal['instance_method', 'top_level_function', 'both']
    is_default: bool = False
    priority: int = 0
    description: str = ""


class ExtensionRegistry:
    """
    Central registry for managing FuzzLab's external extension functions.

    This class provides mechanisms to:
        1. Register specialized function implementations based on `mtype`.
        2. Register and manage default function implementations.
        3. Retrieve the most appropriate function implementation given a function name
           and an `mtype`, with fallback to default if no specialized implementation exists.
        4. Ensure thread-safe operations for registration and retrieval.

    It works in conjunction with `fuzzlab.extension.decorator.py` for registration
    and `fuzzlab.extension.dispatcher.py` and `fuzzlab.extension.injector.py`
    for function dispatching and injection.
    """

    def __init__(self):
        """
        Initializes the ExtensionRegistry.
        """
        self._lock = threading.RLock()

        # Stores specialized implementations: {function_name: {mtype: (implementation_func, metadata)}}
        self._functions: Dict[str, Dict[str, Tuple[Callable, FunctionMetadata]]] = {}

        # Stores default implementations: {function_name: (default_implementation_func, metadata)}
        self._defaults: Dict[str, Tuple[Callable, FunctionMetadata]] = {}

        # Stores a history of all registration attempts.
        self._registration_history: List[Dict[str, Any]] = []

    def register(self,
                 name: str,
                 mtype: Optional[str] = None,
                 target_classes: Union[str, List[str]] = None,
                 injection_type: Literal['instance_method', 'top_level_function', 'both'] = 'both',
                 is_default: bool = False,
                 priority: int = 0,
                 **kwargs) -> Callable:
        """
        Decorator factory to register an external function.

        This method is typically called by the `@extension` decorator
        (from `fuzzlab.extension.decorator.py`) to register a function
        with the registry. It returns a decorator that, when applied to a function,
        stores that function and its metadata.

        Args:
            name: The name of the extension function (e.g., 'distance', '_random').
            mtype: The specific fuzzy number type (e.g., 'qrofn', 'ivfn') this
                implementation is for. `None` indicates a general or default
                implementation.
            target_classes: A string or list of strings representing the names of
                classes (e.g., 'Fuzznum', 'Fuzzarray') that this extension is
                intended to be injected into as an instance method. If `None`,
                defaults to `['Fuzznum', 'Fuzzarray']`.
            injection_type: Specifies how the function should be injected:
                'instance_method', 'top_level_function', or 'both'.
            is_default: A boolean indicating if this is a default implementation
                for the given `name`.
            priority: An integer representing the priority of this implementation.
                Higher values indicate higher priority. If an implementation with
                equal or higher priority already exists, a `ValueError` is raised.
            **kwargs: Additional keyword arguments to be stored in the metadata.

        Returns:
            A decorator function that takes the actual implementation function
            as an argument and registers it.

        Raises:
            ValueError: If a default implementation with higher or equal priority
                already exists for the given `name`, or if a specialized
                implementation for the given `name` and `mtype` with higher
                or equal priority already exists.

        Examples:
            Registering a specialized 'distance' function for 'qrofn' mtype:
            ```python
            # In fuzzlab/fuzzy/qrofs/_func.py
            from fuzzlab.extension import extension
            from fuzzlab.core import Fuzznum

            @extension('distance', mtype='qrofn', target_classes=['Fuzznum'])
            def qrofn_distance(fuzz1: Fuzznum, fuzz2: Fuzznum) -> float:
                # ... qrofn specific distance calculation ...
                return 0.0
            ```

            Registering a default 'distance' function:
            ```python
            # In a general extension module
            from fuzzlab.extension import extension
            from fuzzlab.core import Fuzznum

            @extension('distance', is_default=True, target_classes=['Fuzznum'])
            def default_distance(fuzz1: Fuzznum, fuzz2: Fuzznum) -> float:
                # ... general distance calculation ...
                return 0.0
            ```
        """
        # Normalize target_classes to always be a list of strings.
        if isinstance(target_classes, str):
            target_classes = [target_classes]
        elif target_classes is None:
            target_classes = ['Fuzznum', 'Fuzzarray']

        def decorator(func: Callable) -> Callable:
            # Create metadata object for the function being registered.
            metadata = FunctionMetadata(
                name=name,
                mtype=mtype,
                target_classes=target_classes,
                injection_type=injection_type,
                is_default=is_default,
                priority=priority,
                **kwargs
            )

            # Ensure thread-safe registration.
            with self._lock:
                if is_default:
                    # Handle registration for default implementations.
                    if name in self._defaults:
                        existing_priority = self._defaults[name][1].priority
                        # Prevent re-registration with lower or equal priority.
                        if priority <= existing_priority:
                            raise ValueError(f"Default implementation for '{name}' already exists with higher "
                                             f"or equal priority ({existing_priority}). "
                                             f"Cannot register new with priority {priority}.")
                    # Store the function and its metadata.
                    self._defaults[name] = (func, metadata)
                else:
                    # Handle registration for specialized (mtype-specific) implementations.
                    if name not in self._functions:
                        # Initialize dictionary for this function name if it doesn't exist.
                        self._functions[name] = {}

                    if mtype in self._functions[name]:
                        existing_priority = self._functions[name][mtype][1].priority
                        # Prevent re-registration with lower or equal priority for the same mtype.
                        if priority <= existing_priority:
                            raise ValueError(f"Implementation for '{name}' with mtype '{mtype}' already exists with higher "
                                             f"or equal priority ({existing_priority}). "
                                             f"Cannot register new with priority {priority}.")

                    # Store the function and its metadata for the specific mtype.
                    self._functions[name][mtype] = (func, metadata)

                # Record the registration event for debugging/auditing.
                self._registration_history.append({
                    'name': name,
                    'mtype': mtype,
                    'is_default': is_default,
                    'priority': priority,
                    'timestamp': self._get_timestamp()
                })

            # Return the original function, as decorators typically do.
            return func

        return decorator

    def get_function(self, name: str, mtype: str) -> Optional[Callable]:
        """
        Retrieves the appropriate function implementation for a given name and mtype.

        This method first attempts to find a specialized implementation for the
        given `mtype`. If no such specialized implementation exists, it falls back
        to the default implementation for the given `name`.

        Args:
            name: The name of the extension function (e.g., 'distance').
            mtype: The `mtype` of the fuzzy number for which the function is needed.
                   If `None`, only the default implementation will be considered.

        Returns:
            The callable function implementation if found, otherwise `None`.

        Examples:
            ```python
            registry = get_extension_registry()
            # Assuming 'qrofn_distance' was registered for mtype 'qrofn'
            qrofn_impl = registry.get_function('distance', 'qrofn')
            # Assuming 'default_distance' was registered as default
            default_impl = registry.get_function('distance', 'some_other_mtype')
            ```
        """
        with self._lock:
            # First, try to find a specialized implementation if mtype is provided.
            if mtype and name in self._functions and mtype in self._functions[name]:
                return self._functions[name][mtype][0]

            # If no specialized implementation is found (or mtype was not provided),
            # fall back to the default.
            if name in self._defaults:
                return self._defaults[name][0]

            return None

    def get_metadata(self, name: str, mtype: Optional[str] = None) -> Optional[FunctionMetadata]:
        """
        Retrieves the metadata for a registered function.

        Args:
            name: The name of the extension function.
            mtype: The specific `mtype` for which to retrieve metadata. If `None`,
                it attempts to retrieve metadata for the default implementation.

        Returns:
            The `FunctionMetadata` object if found, otherwise `None`.

        Examples:
            ```python
            registry = get_extension_registry()
            qrofn_meta = registry.get_metadata('distance', 'qrofn')
            default_meta = registry.get_metadata('distance', None)
            ```
        """
        with self._lock:
            # If mtype is provided, try to get specialized metadata.
            if mtype and name in self._functions and mtype in self._functions[name]:
                return self._functions[name][mtype][1]  # Return the metadata object.

            # Otherwise, or if specialized not found, try to get default metadata.
            if name in self._defaults:
                return self._defaults[name][1]  # Return the metadata object.

            return None  # No metadata found.

    def list_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        Lists all registered functions along with their implementation details.

        This method provides a structured overview of all specialized and
        default implementations currently registered in the system.

        Returns:
            A dictionary where keys are function names, and values are dictionaries
            containing 'implementations' (for mtype-specific versions) and 'default'
            (for the default version, if any).

            Example structure:
            ```json
            {
                "distance": {
                    "implementations": {
                        "qrofn": {
                            "priority": 0,
                            "target_classes": ["Fuzznum", "Fuzzarray"],
                            "injection_type": "both"
                        }
                    },
                    "default": {
                        "priority": 0,
                        "target_classes": ["Fuzznum"],
                        "injection_type": "instance_method"
                    }
                },
                "_random": {
                    "implementations": {
                        "qrofn": {
                            "priority": 0,
                            "target_classes": ["Fuzznum"],
                            "injection_type": "top_level_function"
                        }
                    },
                    "default": null
                }
            }
            ```
        """
        with self._lock:
            result = {}

            # Populate with specialized implementations.
            for func_name, implementations in self._functions.items():
                if func_name not in result:
                    result[func_name] = {'implementations': {}, 'default': None}

                for mtype, (func, metadata) in implementations.items():
                    result[func_name]['implementations'][mtype] = {
                        'priority': metadata.priority,
                        'target_classes': metadata.target_classes,
                        'injection_type': metadata.injection_type
                    }

            # Add default implementations.
            for func_name, (func, metadata) in self._defaults.items():
                if func_name not in result:
                    result[func_name] = {'implementations': {}, 'default': None}

                result[func_name]['default'] = {
                    'priority': metadata.priority,
                    'target_classes': metadata.target_classes,
                    'injection_type': metadata.injection_type
                }

            return result

    @staticmethod
    def _get_timestamp():
        """
        Helper static method to get the current timestamp in ISO format.

        Returns:
            A string representing the current timestamp.
        """
        return datetime.datetime.now().isoformat()


# Global singleton instance of ExtensionRegistry.
_extension_registry = None
# Lock to ensure thread-safe initialization of the singleton.
_extension_registry_lock = threading.RLock()


def get_extension_registry() -> ExtensionRegistry:
    """
    Retrieves the global singleton instance of `ExtensionRegistry`.

    This function ensures that only one instance of the registry exists
    across the application, providing a central point for managing extensions.

    Returns:
        The singleton `ExtensionRegistry` instance.

    Examples:
        ```python
        registry = get_extension_registry()
        # Use the registry to register or retrieve functions
        ```
    """
    global _extension_registry
    # Double-checked locking for thread-safe singleton initialization.
    if _extension_registry is None:
        with _extension_registry_lock:
            if _extension_registry is None:
                _extension_registry = ExtensionRegistry()
    return _extension_registry
