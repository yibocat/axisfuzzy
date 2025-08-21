#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Mixin Function Registry for AxisFuzzy core class extensions.

This module provides the infrastructure for registering and dynamically injecting
mtype-agnostic structural operations into :class:`axisfuzzy.core.fuzznums.Fuzznum`
and :class:`axisfuzzy.core.fuzzarray.Fuzzarray` classes, as well as the top-level
:mod:`axisfuzzy` namespace.

The registry system enables a clean separation between core class definitions and
extended functionality, allowing for modular development while maintaining a
cohesive user interface.

Architecture Overview
---------------------
The mixin system follows a three-phase lifecycle:

1. **Registration Phase**: Functions are registered with metadata specifying
   their injection targets and exposure types using the :func:`register_mixin`
   decorator.

2. **Storage Phase**: The :class:`MixinFunctionRegistry` stores registered
   functions and their associated metadata in internal dictionaries.

3. **Injection Phase**: During library initialization, :meth:`MixinFunctionRegistry.build_and_inject`
   dynamically attaches registered functions to target classes and the module namespace.

Key Differences from Extension System
-------------------------------------
The mixin system differs from :mod:`axisfuzzy.extension` in several fundamental ways:

- **Scope**: Mixin functions are mtype-agnostic and focus on structural operations
  (reshape, transpose, concatenate) that work uniformly across all fuzzy number types.
- **Dispatch**: No runtime mtype-based dispatch is needed; all functions work the
  same way regardless of the underlying fuzzy number type.
- **Use Cases**: Primarily for NumPy-like array manipulation and container operations.

Injection Types
---------------
Functions can be exposed in three different ways:

- ``'instance_function'``: Injected as bound methods on target classes
  (e.g., ``my_fuzzarray.reshape(2, 3)``).
- ``'top_level_function'``: Injected into the module namespace
  (e.g., ``axisfuzzy.reshape(my_fuzzarray, 2, 3)``).
- ``'both'``: Exposed as both instance methods and top-level functions.

Thread Safety
--------------
The registry is not inherently thread-safe during the registration and injection
phases. However, since these operations typically occur during module initialization
(import time), this is generally not a concern in practice.

See Also
--------
axisfuzzy.mixin.factory : Implementation layer for mixin operations.
axisfuzzy.mixin.register : Registration declarations for standard operations.
axisfuzzy.extension : mtype-sensitive extension system for specialized operations.

Examples
--------
Registering a simple mixin function:

.. code-block:: python

    from axisfuzzy.mixin.registry import register_mixin

    @register_mixin(name='is_empty', target_classes=['Fuzzarray'],
                    injection_type='instance_function')
    def _is_empty_impl(self):
        return self.size == 0

    # After library initialization:
    # arr = fuzzarray([...])
    # arr.is_empty()  # True/False

Registering a function as both instance method and top-level function:

.. code-block:: python

    @register_mixin(name='normalize_shape', target_classes=['Fuzzarray'],
                    injection_type='both')
    def _normalize_shape_impl(obj):
        return obj.reshape(-1)

    # After library initialization:
    # arr.normalize_shape()  # instance method
    # axisfuzzy.normalize_shape(arr)  # top-level function

Manual injection during initialization:

.. code-block:: python

    from axisfuzzy.mixin.registry import get_registry_mixin
    from axisfuzzy.core import Fuzznum, Fuzzarray

    registry = get_registry_mixin()
    class_map = {'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}
    module_globals = globals()

    registry.build_and_inject(class_map, module_globals)
"""

import functools
from typing import Dict, Callable, Any, List, Optional, Literal


class MixinFunctionRegistry:
    """
    Central registry for mtype-agnostic functions extending AxisFuzzy core classes.

    This registry manages the registration, storage, and injection of functions
    that provide NumPy-like structural operations for :class:`Fuzznum` and
    :class:`Fuzzarray` objects. Unlike the extension system, mixin functions
    work uniformly across all fuzzy number types without requiring dispatch logic.

    The registry supports three injection modes: instance methods, top-level functions,
    or both. It ensures that the extended functionality integrates seamlessly with
    the existing class interfaces.

    Attributes
    ----------
    _functions : dict of str to callable
        Maps function names to their implementation callables.
    _metadata : dict of str to dict
        Maps function names to their registration metadata, including
        target classes and injection preferences.

    Notes
    -----
    - The registry is designed as a singleton accessed via :func:`get_registry_mixin`.
    - Registration typically occurs at module import time via decorators.
    - Injection happens once during library initialization.

    See Also
    --------
    register_mixin : Convenience decorator for function registration.
    get_registry_mixin : Access to the global registry singleton.
    """

    def __init__(self):
        """
        Initialize an empty mixin function registry.

        Creates internal storage for functions and their associated metadata.
        The registry starts empty and is populated through decorator-based
        registration during module imports.

        Examples
        --------
        .. code-block:: python

            # Typically not called directly; use get_registry_mixin() instead
            registry = MixinFunctionRegistry()
        """
        # Stores the core implementation of a function, mapped by its name.
        self._functions: Dict[str, Callable] = {}
        # Stores metadata for each function, such as injection type and target classes.
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self,
                 name: str,
                 target_classes: Optional[List[str]] = None,
                 injection_type: Literal['instance_function', 'top_level_function', 'both'] = 'both') -> Callable:
        """
        Decorator factory to register a function for dynamic injection.

        This method provides the core registration mechanism for mixin functions.
        It validates registration parameters, stores the function and its metadata,
        and returns a decorator that can be applied to the implementation function.

        Parameters
        ----------
        name : str
            Unique identifier for the function within the registry. This name
            will be used as the method/function name after injection.
        target_classes : list of str, optional
            Class names where the function should be injected as an instance method.
            Required when ``injection_type`` is 'instance_function' or 'both'.
            Common values are ['Fuzznum'], ['Fuzzarray'], or ['Fuzznum', 'Fuzzarray'].
        injection_type : {'instance_function', 'top_level_function', 'both'}, default 'both'
            Specifies how the function should be exposed:

            - 'instance_function': Only as a bound method on target classes
            - 'top_level_function': Only in the module namespace
            - 'both': As both instance methods and top-level functions

        Returns
        -------
        callable
            A decorator function that accepts the implementation and registers it.

        Raises
        ------
        ValueError
            - If ``injection_type`` is not one of the allowed values
            - If ``target_classes`` is None when instance injection is requested
            - If a function with the same ``name`` is already registered

        Examples
        --------
        Register an instance-only method:

        .. code-block:: python

            @registry.register('is_normalized',
                              target_classes=['Fuzznum', 'Fuzzarray'],
                              injection_type='instance_function')
            def _is_normalized_impl(self):
                # Implementation logic
                return True

        Register a top-level-only function:

        .. code-block:: python

            @registry.register('create_identity', injection_type='top_level_function')
            def _create_identity_impl(mtype='qrofn'):
                # Implementation logic
                return Fuzznum(mtype).create(md=1.0, nmd=0.0)

        Register both instance method and top-level function:

        .. code-block:: python

            @registry.register('to_list', target_classes=['Fuzzarray'], injection_type='both')
            def _to_list_impl(self):
                # Works as both arr.to_list() and axisfuzzy.to_list(arr)
                return list(self)
        """
        # Validate the provided injection_type to ensure it's one of the allowed literals.
        if injection_type not in ['instance_function', 'top_level_function', 'both']:
            raise ValueError(f"Invalid injection_type: {injection_type}. "
                             f"Must be 'instance_function', 'top_level_function', or 'both'.")

        # Validate that target_classes are provided if instance method injection is requested.
        if injection_type in ['instance_function', 'both'] and not target_classes:
            raise ValueError(f"target_classes must be provided for injection_type '{injection_type}'.")

        def decorator(func: Callable) -> Callable:
            """
            Inner decorator that performs the actual registration.

            Parameters
            ----------
            func : callable
                The implementation function to register.

            Returns
            -------
            callable
                The original function (unmodified).
            """
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
        Inject all registered functions into target classes and module namespace.

        This method performs the final injection phase, iterating through all
        registered functions and attaching them to their specified targets based
        on the injection metadata. It handles both instance method injection
        (via :func:`setattr` on classes) and top-level function injection
        (via dictionary assignment on the module namespace).

        Parameters
        ----------
        class_map : dict of str to type
            Maps class names to actual class objects for instance method injection.
            Typically constructed as ``{'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}``.
        module_namespace : dict of str to any
            Target module's namespace (usually ``globals()`` of the target module)
            where top-level functions should be injected. Functions are added as
            key-value pairs to this dictionary.

        Notes
        -----
        - Instance methods are injected using ``setattr(class, name, function)``
        - Top-level functions use different strategies for 'both' vs 'top_level_function' only:

          - 'both': Creates a wrapper that delegates to the instance method
          - 'top_level_function': Injects the original function directly

        - Injection is idempotent but not thread-safe
        - Missing classes in ``class_map`` are silently ignored

        Examples
        --------
        Typical usage during library initialization:

        .. code-block:: python

            from axisfuzzy.core import Fuzznum, Fuzzarray
            from axisfuzzy.mixin.registry import get_registry_mixin

            # Prepare injection targets
            class_map = {
                'Fuzznum': Fuzznum,
                'Fuzzarray': Fuzzarray
            }
            module_globals = globals()

            # Perform injection
            registry = get_registry_mixin()
            registry.build_and_inject(class_map, module_globals)

            # Now functions are available:
            # arr = Fuzzarray(...)
            # arr.reshape(2, 2)  # instance method
            # reshape(arr, 2, 2)  # top-level function

        Custom class mapping:

        .. code-block:: python

            # Only inject into Fuzzarray
            class_map = {'Fuzzarray': Fuzzarray}
            registry.build_and_inject(class_map, {})
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
        Get names of all functions registered for top-level injection.

        This method scans the registry metadata and returns a list of function
        names that should be exposed as top-level functions. It's useful for
        populating ``__all__`` lists and documentation generation.

        Returns
        -------
        list of str
            Sorted list of function names that have ``injection_type`` of
            'top_level_function' or 'both'.

        Examples
        --------
        .. code-block:: python

            registry = get_registry_mixin()

            # Register some functions
            @registry.register('func1', injection_type='instance_function',
                              target_classes=['Fuzznum'])
            def f1(self): pass

            @registry.register('func2', injection_type='top_level_function')
            def f2(): pass

            @registry.register('func3', injection_type='both',
                              target_classes=['Fuzzarray'])
            def f3(self): pass

            names = registry.get_top_level_function_names()
            print(names)  # ['func2', 'func3']

        Use in module ``__all__`` definition:

        .. code-block:: python

            from axisfuzzy.mixin.registry import get_registry_mixin

            # Get all mixin top-level functions
            _mixin_functions = get_registry_mixin().get_top_level_function_names()

            # Combine with other exports
            __all__ = ['Fuzznum', 'Fuzzarray'] + _mixin_functions
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
    Access the global singleton :class:`MixinFunctionRegistry` instance.

    This function provides the standard entry point to the mixin registry system.
    It returns the same registry instance across all calls, ensuring consistent
    registration and injection behavior throughout the library.

    Returns
    -------
    MixinFunctionRegistry
        The global singleton registry instance.

    Notes
    -----
    The registry is created once when this module is first imported and reused
    for all subsequent calls. This singleton pattern ensures that all registered
    functions are stored in the same location and available for injection.

    Examples
    --------
    Basic registry access:

    .. code-block:: python

        from axisfuzzy.mixin.registry import get_registry_mixin

        registry = get_registry_mixin()
        # Use registry.register(...) to register functions

    Use in registration modules:

    .. code-block:: python

        # In axisfuzzy/mixin/register.py or similar
        from axisfuzzy.mixin.registry import get_registry_mixin

        registry = get_registry_mixin()

        @registry.register('my_function', target_classes=['Fuzzarray'])
        def _my_function_impl(self):
            return self.copy()

    Use in library initialization:

    .. code-block:: python

        # In axisfuzzy/__init__.py
        from axisfuzzy.mixin.registry import get_registry_mixin
        from axisfuzzy.core import Fuzznum, Fuzzarray

        # Inject all registered mixin functions
        registry = get_registry_mixin()
        class_map = {'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}
        registry.build_and_inject(class_map, globals())
    """
    return _registry


def register_mixin(name: str,
                   target_classes: Optional[List[str]] = None,
                   injection_type: Literal['instance_function', 'top_level_function', 'both'] = 'both') -> Callable:
    """
    Convenience decorator for registering mixin functions.

    This function provides a streamlined interface to the global mixin registry,
    eliminating the need to explicitly access the registry instance. It's the
    recommended way to register mixin functions in most scenarios.

    Parameters
    ----------
    name : str
        Unique function name for registry and injection.
    target_classes : list of str, optional
        Class names for instance method injection. Required for 'instance_function'
        and 'both' injection types.
    injection_type : {'instance_function', 'top_level_function', 'both'}, default 'both'
        How the function should be exposed after injection.

    Returns
    -------
    callable
        Decorator function that registers the implementation.

    Raises
    ------
    ValueError
        If parameters are invalid or if the function name is already registered.

    Examples
    --------
    Register an instance method:

    .. code-block:: python

        from axisfuzzy.mixin.registry import register_mixin

        @register_mixin('is_square', target_classes=['Fuzzarray'],
                        injection_type='instance_function')
        def _is_square_impl(self):
            return len(set(self.shape)) <= 1

    Register a top-level function:

    .. code-block:: python

        @register_mixin('zeros_like', injection_type='top_level_function')
        def _zeros_like_impl(template):
            # Create zero array with same shape and mtype as template
            return Fuzzarray(shape=template.shape, mtype=template.mtype)

    Register both instance method and top-level function:

    .. code-block:: python

        @register_mixin('flatten', target_classes=['Fuzzarray'], injection_type='both')
        def _flatten_impl(self):
            # Available as both arr.flatten() and axisfuzzy.flatten(arr)
            return self.reshape(-1)

    See Also
    --------
    MixinFunctionRegistry.register : The underlying registration method.
    get_registry_mixin : Access to the global registry instance.
    """
    return get_registry_mixin().register(name, target_classes, injection_type)
