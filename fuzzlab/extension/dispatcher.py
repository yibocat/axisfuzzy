#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 15:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
Extension Dispatcher for FuzzLab.

This module defines the `ExtensionDispatcher` class, which is responsible for
creating dynamic "proxy" functions. These proxy functions, when called,
intelligently dispatch the call to the correct underlying extension
implementation based on the `mtype` of the `Fuzznum` or `Fuzzarray` object
involved in the operation.

The dispatcher works in conjunction with `fuzzlab.extension.registry.py`
(to look up implementations) and `fuzzlab.extension.injector.py` (to inject
these proxy functions into classes or the top-level namespace).
"""
from typing import Callable, Union

from .registry import get_extension_registry
from ..core import Fuzznum, Fuzzarray


class ExtensionDispatcher:
    """
    Manages the creation of dispatching functions for FuzzLab extensions.

    This class generates callable "proxies" that, when invoked, determine
    the appropriate extension function to execute based on the `mtype`
    of the fuzzy number object. It ensures that the correct specialized
    or default implementation is called at runtime.
    """

    def __init__(self):
        """
        Initializes the ExtensionDispatcher.

        It obtains a reference to the global `ExtensionRegistry` to
        perform function lookups.
        """
        self.registry = get_extension_registry()

    def create_instance_method(self, func_name: str) -> Callable:
        """
        Creates a dispatching function suitable for an instance method.

        This function returns a closure that, when bound as a method to
        a `Fuzznum` or `Fuzzarray` instance, will extract the instance's
        `mtype` and use it to find and call the correct registered
        extension implementation.

        Args:
            func_name: The name of the extension function to dispatch (e.g., 'distance').

        Returns:
            A callable function (`method_dispatcher`) that acts as a proxy
            for the actual extension implementation.

        Raises:
            AttributeError: If the object on which the method is called lacks a 'mtype' attribute.
            NotImplementedError: If no implementation for the given `func_name` and `mtype`
                                 (or a suitable default) is found in the registry.

        Examples:
            ```python
            # This method is typically called by ExtensionInjector
            # to create methods like Fuzznum.distance.
            dispatcher = get_extension_dispatcher()
            distance_method = dispatcher.create_instance_method('distance')

            # Later, this 'distance_method' would be set as Fuzznum.distance
            # fuzznum_instance.distance(other_fuzznum) would then call method_dispatcher
            ```
        """
        def method_dispatcher(obj, *args, **kwargs):
            """
            The actual dispatching logic for instance methods.

            This function is dynamically attached to Fuzznum/Fuzzarray instances.
            It extracts the mtype from `obj` and uses it to find the correct
            extension implementation.
            """
            mtype = getattr(obj, 'mtype', None)
            if mtype is None:
                raise AttributeError(f"Object {type(obj).__name__} has no 'mtype' attribute")

            # Retrieve the appropriate implementation from the registry.
            implementation = self.registry.get_function(func_name, mtype)
            if implementation is None:
                # Provide detailed error message for better debugging.
                available_mtypes = list(self.registry._functions.get(func_name, {}).keys())
                has_default = func_name in self.registry._defaults

                error_msg = f"Function '{func_name}' not implemented for mtype '{mtype}'"
                if available_mtypes:
                    error_msg += f". Available for: {available_mtypes}"
                if has_default:
                    # This indicates that a default was registered but get_function returned None,
                    # which should ideally not happen if the default is correctly registered.
                    error_msg += ". Default implementation available but failed to load or was not applicable."

                raise NotImplementedError(error_msg)

            # Call the found implementation with the original arguments.
            return implementation(obj, *args, **kwargs)

        # Set __name__ and __doc__ for better introspection and debugging.
        method_dispatcher.__name__ = func_name
        method_dispatcher.__doc__ = f"Dispatched method for {func_name}"
        return method_dispatcher

    def create_top_level_function(self, func_name: str) -> Callable:
        """
        Creates a dispatching function suitable for a top-level module function.

        This function returns a closure that, when called as a global function
        (e.g., `fuzzlab.distance(...)`), will expect a `Fuzznum` or `Fuzzarray`
        object as its first argument. It then extracts the `mtype` from this
        object to find and call the correct registered extension implementation.

        Args:
            func_name: The name of the extension function to dispatch (e.g., 'distance').

        Returns:
            A callable function (`function_dispatcher`) that acts as a proxy
            for the actual extension implementation.

        Raises:
            TypeError: If the first argument is not a `Fuzznum` or `Fuzzarray` instance.
            AttributeError: If the first argument lacks a 'mtype' attribute.
            NotImplementedError: If no implementation for the given `func_name` and `mtype`
                                 (or a suitable default) is found in the registry.

        Examples:
            ```python
            # This method is typically called by ExtensionInjector
            # to create top-level functions like fuzzlab.distance.
            dispatcher = get_extension_dispatcher()
            distance_func = dispatcher.create_top_level_function('distance')

            # Later, this 'distance_func' would be set as fuzzlab.distance
            # fuzzlab.distance(fuzznum_instance, other_fuzznum) would then call function_dispatcher
            ```
        """
        def function_dispatcher(obj: Union[Fuzznum, Fuzzarray], *args, **kwargs):
            """
            The actual dispatching logic for top-level functions.

            This function is dynamically attached to the fuzzlab module namespace.
            It expects the first argument to be a Fuzznum/Fuzzarray instance
            to extract its mtype and find the correct extension implementation.
            """
            # Ensure the first argument is a Fuzznum or Fuzzarray instance.
            if not isinstance(obj, (Fuzznum, Fuzzarray)):
                raise TypeError(f"First argument must be Fuzznum or Fuzzarray, got {type(obj).__name__}")

            # Extract the mtype from the first argument.
            mtype = getattr(obj, 'mtype', None)
            if mtype is None:
                raise AttributeError(f"Object {type(obj).__name__} has no 'mtype' attribute")

            # Retrieve the appropriate implementation from the registry.
            implementation = self.registry.get_function(func_name, mtype)
            if implementation is None:
                # Similar error handling as in create_instance_method could be added here
                # for more detailed debugging information.
                raise NotImplementedError(f"Function '{func_name}' not implemented for mtype '{mtype}'")

            # Call the found implementation with the original arguments.
            return implementation(obj, *args, **kwargs)

        # Set __name__ and __doc__ for better introspection and debugging.
        function_dispatcher.__name__ = func_name
        function_dispatcher.__doc__ = f"Dispatched top-level function for {func_name}"
        return function_dispatcher


# Global singleton instance of ExtensionDispatcher.
_dispatcher = ExtensionDispatcher()


def get_extension_dispatcher() -> ExtensionDispatcher:
    """
    Retrieves the global singleton instance of `ExtensionDispatcher`.

    This function ensures that only one instance of the dispatcher exists
    across the application, providing a central point for creating
    dispatching proxies.

    Returns:
        The singleton `ExtensionDispatcher` instance.

    Examples:
        ```python
        dispatcher = get_extension_dispatcher()
        # Use the dispatcher to create proxy functions/methods
        ```
    """
    return _dispatcher
