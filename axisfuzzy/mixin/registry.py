#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
This module provides a flexible function registry system for FuzzLab's mixin functionalities.

It allows registering functions and dynamically injecting them as instance methods
into specified classes (like `Fuzznum` and `Fuzzarray`) and/or as top-level functions
into a module's namespace. This system is designed to centralize the management
of common operations and utility functions that can be applied across different
FuzzLab data structures, promoting code reusability and a consistent API.

Example:
    To register a function `my_func` that should be an instance method of `Fuzznum`
    and also a top-level function in the `fuzzlab` module:

    >>> from axisfuzzy.mixin.registry import get_registry_mixin
    >>> registry = get_registry_mixin()

    >>> @registry.register_mixin(name='my_func', target_classes=['Fuzznum'], injection_type='both')
    ... def _my_func_impl(self, arg1, arg2):
    ...     # Implementation details
    ...     return f"Called my_func on {self.__class__.__name__} with {arg1}, {arg2}"

    After the registry's `build_and_inject` method is called during FuzzLab's initialization:

    >>> # from axisfuzzy import Fuzznum, my_func
    >>> # fn = Fuzznum(...)
    >>> # fn.my_func(1, 2) # Calls the instance method
    'Called my_func on Fuzznum with 1, 2'
    >>> # my_func(fn, 3, 4) # Calls the top-level function
    'Called my_func on Fuzznum with 3, 4'
"""

import functools
from typing import Dict, Callable, Any, List, Optional, Literal


class MixinFunctionRegistry:
    """
    A registry for managing and injecting functions into classes and modules.

    This class serves as a central hub for registering functions that are intended
    to be dynamically added as methods to FuzzLab's core classes (e.g., `Fuzznum`,
    `Fuzzarray`) or exposed as top-level functions within the `axisfuzzy` package.
    It provides a unified decorator (`register`) to define how each function
    should be injected (as an instance method, a top-level function, or both).

    Attributes:
        _functions (Dict[str, Callable]): Stores the actual callable objects (functions)
            registered, mapped by their unique `name`.
        _metadata (Dict[str, Dict[str, Any]]): Stores metadata for each registered function,
            including its `target_classes` (for instance method injection) and
            `injection_type` (how it should be injected).
    """

    def __init__(self):
        # Stores the core implementation of a function, mapped by its name.
        self._functions: Dict[str, Callable] = {}
        # Stores metadata for each function, such as injection type and target classes.
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self,
                 name: str,
                 target_classes: Optional[List[str]] = None,
                 injection_type: Literal['instance_function', 'top_level_function', 'both'] = 'both') -> Callable:
        """
        Decorator to register a function for dynamic injection.

        This decorator is the primary interface for developers to register
        new mixin functionalities. It allows specifying the function's
        intended name, the classes it should be injected into as a method,
        and whether it should also be available as a top-level function.

        Args:
            name (str): The unique name under which the function will be registered
                and subsequently injected. This name will be used as the method name
                or the top-level function name.
            target_classes (Optional[List[str]]): A list of string names of the classes
                (e.g., `["Fuzznum", "Fuzzarray"]`) to which this function will be
                injected as an instance method. This argument is required if
                `injection_type` is `'instance_function'` or `'both'`.
            injection_type (Literal): Specifies how the function should be injected.
                Defaults to `'both'`.
                - `'instance_function'`: The function will only be injected as an
                  instance method into the `target_classes`.
                - `'top_level_function'`: The function will only be injected as a
                  top-level function into the `axisfuzzy` module's namespace.
                - `'both'`: The function will be injected as both an instance method
                  and a top-level function.

        Returns:
            Callable: A decorator that takes the actual function implementation
                and registers it with the registry.

        Raises:
            ValueError:
                - If `injection_type` is invalid (not one of the allowed literals).
                - If `target_classes` are not provided when `injection_type` is
                  `'instance_function'` or `'both'`.
                - If a function with the same `name` is already registered, ensuring
                  unique function names within the registry.

        Examples:
            Registering a function as an instance method for `Fuzznum` and `Fuzzarray`:

            >>> @registry.register_mixin(name='is_normalized', target_classes=['Fuzznum', 'Fuzzarray'], injection_type='instance_function')
            ... def _is_normalized_impl(self):
            ...     # Check if fuzzy number/array is normalized
            ...     return True

            Registering a function as a top-level function only:

            >>> @registry.register_mixin(name='create_identity', injection_type='top_level_function')
            ... def _create_identity_impl(mtype: str):
            ...     # Create an identity fuzzy number of a given mtype
            ...     return Fuzznum(mtype=mtype, md=1.0, nmd=0.0)

            Registering a function as both an instance method and a top-level function:

            >>> @registry.register_mixin(name='to_array', target_classes=['Fuzznum'], injection_type='both')
            ... def _to_array_impl(self):
            ...     # Convert a Fuzznum to a Fuzzarray
            ...     return Fuzzarray([self])
        """
        # Validate the provided injection_type to ensure it's one of the allowed literals.
        if injection_type not in ['instance_function', 'top_level_function', 'both']:
            raise ValueError(f"Invalid injection_type: {injection_type}. "
                             f"Must be 'instance_function', 'top_level_function', or 'both'.")

        # Validate that target_classes are provided if instance method injection is requested.
        if injection_type in ['instance_function', 'both'] and not target_classes:
            raise ValueError(f"target_classes must be provided for injection_type '{injection_type}'.")

        def decorator(func: Callable) -> Callable:
            # Check for duplicate function names to prevent accidental overwrites.
            if name in self._functions:
                raise ValueError(f"Function '{name}' is already registered.")

            # Store the actual function implementation.
            self._functions[name] = func
            # Store the metadata associated with this function, including its injection preferences.
            self._metadata[name] = {
                'target_classes': target_classes or [],  # Ensure target_classes is always a list
                'injection_type': injection_type
            }

            return func

        return decorator

    def build_and_inject(self,
                         class_map: Dict[str, type],
                         module_namespace: Dict[str, Any]):
        """
        Injects registered functions into target classes and the module namespace.

        This method iterates through all functions registered with the registry
        and performs the actual injection based on their specified `injection_type`.
        It dynamically adds functions as methods to the classes provided in `class_map`
        and/or as callable objects to the `module_namespace`.

        Args:
            class_map (Dict[str, type]): A mapping from class names (strings) to their
                corresponding class objects (e.g., `{"Fuzznum": FuzznumClass, "Fuzzarray": FuzzarrayClass}`).
                This map is used to find the actual class objects for instance method injection.
            module_namespace (Dict[str, Any]): The target namespace (typically `globals()`
                of the module where top-level functions should reside, e.g., `fuzzlab/__init__.py`).
                Top-level functions will be added as key-value pairs to this dictionary.

        Example:
            Assuming `Fuzznum` and `Fuzzarray` classes are defined, and `fuzzlab` is the target module:

            >>> from axisfuzzy.core.fuzznums import Fuzznum
            >>> from axisfuzzy.core.fuzzarray import Fuzzarray
            >>> class_map = {"Fuzznum": Fuzznum, "Fuzzarray": Fuzzarray}
            >>> module_namespace = {} # Simulate a module's globals()

            >>> registry = get_registry_mixin()
            >>> # (Register functions using @registry.register as shown in register method's example)

            >>> registry.build_and_inject(class_map, module_namespace)

            Now, `Fuzznum` instances will have the registered methods, and `module_namespace`
            will contain the registered top-level functions.
        """
        # Iterate over each registered function and its associated metadata.
        for name, func in self._functions.items():
            meta = self._metadata[name]
            injection_type = meta['injection_type']
            target_classes = meta['target_classes']

            # Handle injection as an instance method.
            if injection_type in ['instance_function', 'both']:
                for class_name in target_classes:
                    # Check if the target class exists in the provided map.
                    if class_name in class_map:
                        target_class = class_map[class_name]
                        # Dynamically add the function as a method to the class.
                        setattr(target_class, name, func)

            # # Handle injection as a top-level function.
            if injection_type in ['top_level_function', 'both']:
                if injection_type == 'both':
                    # If the function is injected as 'both', create a wrapper for the top-level
                    # function. This wrapper will delegate the call to the instance method
                    # that was (or will be) injected into the object's class.
                    @functools.wraps(func)
                    def top_level_wrapper(obj: Any, *args, current_name=name, **kwargs):
                        # Check if the object has the method and it's callable.
                        if hasattr(obj, current_name) and callable(getattr(obj, current_name)):
                            # Call the instance method on the provided object.
                            return getattr(obj, current_name)(*args, **kwargs)
                        else:
                            # Raise an error if the method is not supported for the given object type.
                            raise TypeError(f"'{current_name}' is not supported for type '{type(obj).__name__}'")
                    # Inject the wrapper into the module's namespace.
                    module_namespace[name] = top_level_wrapper
                else:
                    # If the function is 'top_level_function' only, inject the function directly
                    # into the module's namespace without any wrapping.
                    module_namespace[name] = func

    def get_top_level_function_names(self) -> List[str]:
        """
        Returns a list of names of all functions registered to be injected as top-level functions.

        This method is useful for populating the `__all__` list of a package,
        allowing users to import these functions directly (e.g., `from axisfuzzy import sum`).

        Returns:
            List[str]: A list of string names for functions that are either
                `'top_level_function'` or `'both'`.

        Example:
            >>> registry = get_registry_mixin()
            >>> # (Register some functions)
            >>> registry.register(name='my_func_tl', injection_type='top_level_function')(lambda x: x)
            >>> registry.register(name='my_func_both', target_classes=['Fuzznum'], injection_type='both')(lambda self: self)
            >>> registry.get_top_level_function_names()
            ['my_func_tl', 'my_func_both']
        """
        names = []
        # Iterate through the metadata of all registered functions.
        for name, meta in self._metadata.items():
            # Check if the function's injection type includes 'top_level_function'.
            if meta['injection_type'] in ['top_level_function', 'both']:
                names.append(name)
        return names


# Create a global instance of the registry.
# This ensures that all parts of the application use the same registry instance
# to register and manage mixin functions, adhering to the Singleton pattern.
_registry = MixinFunctionRegistry()


def get_registry_mixin():
    """
    Returns the global `MixinFunctionRegistry` instance.

    This function provides a centralized access point to the singleton registry.
    Any part of the FuzzLab library that needs to register or access mixin
    functions should use this function to retrieve the registry instance.

    Returns:
        MixinFunctionRegistry: The single, global instance of the registry.

    Example:
        >>> registry = get_registry_mixin()
        >>> # Now 'registry' can be used to register functions or trigger injection.
    """
    return _registry


def register_mixin(name: str,
                   target_classes: Optional[List[str]] = None,
                   injection_type: Literal['instance_function', 'top_level_function', 'both'] = 'both') -> Callable:
    """
    Top-level decorator to register a function for dynamic injection.

    This is a convenience wrapper around `get_registry_mixin().register`.

    Args:
        name (str): The unique name under which the function will be registered.
        target_classes (Optional[List[str]]): A list of class names for injection.
        injection_type (Literal): Specifies how the function should be injected.

    Returns:
        Callable: A decorator for the function implementation.
    """
    return get_registry_mixin().register(name, target_classes, injection_type)
