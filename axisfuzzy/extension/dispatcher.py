#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Extension Dispatcher for AxisFuzzy.

This module defines the machinery for building dynamic "proxy" callables that
resolve and invoke the correct extension implementation at runtime based on the
mtype of involved fuzzy objects.

It works in concert with:

- axisfuzzy.extension.registry: Stores registered implementations and metadata.
- axisfuzzy.extension.injector: Attaches dispatcher-built proxies to classes or
  the top-level namespace.

Design
------
The dispatcher is stateless and thread-safe. It builds three kinds of proxies:

- Instance method proxies (callable as ``obj.fn(...)``).
- Instance property proxies (accessed as ``obj.prop``).
- Top-level function proxies (callable as ``axisfuzzy.fn(obj, ...)``).

Each proxy performs a registry lookup by ``(name, mtype)``, falling back to a
default implementation when a specialized one is not available.

Notes
-----
- Top-level proxy resolves ``mtype`` from one of:
  1) explicit keyword argument ``mtype=...``,
  2) the first positional argument if it is a Fuzznum/Fuzzarray instance,
  3) the library default (from config) otherwise.
- Instance proxies read the mtype from the bound object.
- Error messages include available specialized mtypes and whether a default
  implementation exists, aiding debugging.

Examples
--------
Create and use a dispatched instance method:

.. code-block:: python

    from axisfuzzy.extension.dispatcher import get_extension_dispatcher
    dispatcher = get_extension_dispatcher()
    dist_method = dispatcher.create_instance_method('distance')

    # Typically attached by the injector:
    # Fuzznum.distance = dist_method
    # x.distance(y) -> dispatches to ('distance', x.mtype)

Create and use a top-level function:

.. code-block:: python

    from axisfuzzy.extension.dispatcher import get_extension_dispatcher
    dispatcher = get_extension_dispatcher()
    dist_func = dispatcher.create_top_level_function('distance')

    # Typically attached to `axisfuzzy` module:
    # axisfuzzy.distance(x, y) -> dispatches using x.mtype, unless mtype='...' is passed.

Create and use a dispatched property:

.. code-block:: python

    score_prop = dispatcher.create_instance_property('score')
    # Typically attached by injector:
    # Fuzznum.score = score_prop
    # x.score -> dispatches to ('score', x.mtype)
"""

from typing import Callable

from .registry import get_registry_extension
from ..config import get_config
from ..core import Fuzznum, get_registry_fuzztype, Fuzzarray


class ExtensionDispatcher:
    """
    Factory for creating dispatching proxies for AxisFuzzy extensions.

    The dispatcher generates closures and property descriptors that will resolve
    the correct extension implementation (specialized by ``mtype``) at call time.

    Notes
    -----
    The dispatcher itself does not store function implementations; it exclusively
    queries the global :class:`ExtensionRegistry` via :func:`get_registry_extension`.
    """

    def __init__(self):
        """
        Initialize the dispatcher and bind to the global registry.

        The dispatcher remains stateless and obtains the registry handle once.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.extension.dispatcher import ExtensionDispatcher
            disp = ExtensionDispatcher()
        """
        self.registry = get_registry_extension()

    def create_instance_method(self, func_name: str) -> Callable:
        """
        Create an instance method proxy for an extension.

        The returned function is intended to be bound as an instance method on
        ``Fuzznum``/``Fuzzarray``. When invoked, it reads ``obj.mtype`` and
        resolves the implementation using the global extension registry.

        Parameters
        ----------
        func_name : str
            Logical extension name (e.g., 'distance', 'score').

        Returns
        -------
        Callable
            A callable suitable for binding as an instance method.

        Raises
        ------
        AttributeError
            If the bound object has no ``mtype`` attribute.
        NotImplementedError
            If the registry does not have a specialized or default implementation.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.extension import get_extension_dispatcher
            dispatcher = get_extension_dispatcher()
            Fuzznum.distance = dispatcher.create_instance_method('distance')

            d = x.distance(y)
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
        Create a top-level function proxy for an extension.

        The returned function expects a ``Fuzznum``/``Fuzzarray`` instance as the
        first positional argument (or an explicit ``mtype=...`` in kwargs).
        It then resolves and invokes the implementation.

        Parameters
        ----------
        func_name : str
            Logical extension name (e.g., 'distance', 'read_csv').

        Returns
        -------
        Callable
            A callable suitable for injection into the top-level module namespace.

        Raises
        ------
        ValueError
            If an explicit ``mtype`` is invalid (not registered).
        NotImplementedError
            If the registry does not have a specialized or default implementation.

        Notes
        -----
        ``mtype`` resolution order:
        1) ``kwargs['mtype']`` if present (and removed before the final call),
        2) ``args[0].mtype`` if the first argument is a ``Fuzznum``/``Fuzzarray``,
        3) ``config.DEFAULT_MTYPE`` otherwise.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.extension import get_extension_dispatcher
            dispatcher = get_extension_dispatcher()
            distance = dispatcher.create_top_level_function('distance')

            # axisfuzzy.distance(x, y) -> dispatches using x.mtype
            # axisfuzzy.distance(x, y, mtype='qrofn') -> forces 'qrofn'
        """
        def function_dispatcher(*args, **kwargs):
            """
            Internal dispatcher for top-level functions.

            It resolves ``mtype`` and invokes the registered implementation.
            """
            fuzznum_registry = get_registry_fuzztype()
            config = get_config()

            mtype = kwargs.pop('mtype', None)

            # 2. If not in kwargs, try to infer from the first argument.
            if mtype is None and args and isinstance(args[0], (Fuzznum, Fuzzarray)):
                mtype = getattr(args[0], 'mtype', None)

            if mtype is not None and mtype not in fuzznum_registry.get_registered_mtypes():
                raise ValueError(f"Invalid fuzzy number type '{mtype}', could not be found in the registry. "
                                 f"Available fuzzy number types: "
                                 f"{list(fuzznum_registry.get_registered_mtypes().keys())}.")

            if mtype is None:
                mtype = config.DEFAULT_MTYPE

            # 3. Retrieve the implementation. mtype can be None here, in which case
            #    get_function will look for a default implementation.
            implementation = self.registry.get_function(func_name, mtype)

            if implementation is None:
                # 4. Provide a detailed error message for better debugging.
                error_msg = f"Function '{func_name}' could not be dispatched. "
                available_mtypes = list(self.registry._functions.get(func_name, {}).keys())
                has_default = func_name in self.registry._defaults

                if not available_mtypes and not has_default:
                    error_msg += f"Function '{func_name}' is not registered at all. "
                elif mtype:
                    error_msg += f"Function '{func_name}' not implemented for mtype '{mtype}'. "
                else:
                    error_msg += f"Function '{func_name}' requires an explicit 'mtype' argument or a default implementation. "

                if available_mtypes:
                    error_msg += f" Available specialized mtypes: '{available_mtypes}'."
                if has_default:
                    error_msg += " A default implementation exists."

                raise NotImplementedError(error_msg)

            # Call the found implementation with the original arguments.
            return implementation(*args, **kwargs)

            # Set __name__ and __doc__ for better introspection and debugging.
        function_dispatcher.__name__ = func_name
        function_dispatcher.__doc__ = (f"Dispatched top-level function for '{func_name}'. "
                                       f"'mtype' is resolved from kwargs or the first argument.")
        return function_dispatcher

    def create_instance_property(self, func_name: str) -> property:
        """
        Create an instance property proxy for an extension.

        The returned property, when accessed, reads ``obj.mtype`` and resolves
        a getter implementation from the global registry.

        Parameters
        ----------
        func_name : str
            Logical extension name for a read-only property (e.g., 'score').

        Returns
        -------
        property
            A read-only property whose getter dispatches to the registered implementation.

        Raises
        ------
        AttributeError
            If the bound object has no ``mtype`` attribute.
        NotImplementedError
            If the registry does not have a specialized or default implementation.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.extension import get_extension_dispatcher
            dispatcher = get_extension_dispatcher()
            Fuzznum.score = dispatcher.create_instance_property('score')

            s = x.score
        """
        def property_getter(obj):
            """
            Internal getter for instance properties.

            It extracts ``obj.mtype`` and invokes the implementation.
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

                error_msg = f"Property '{func_name}' not implemented for mtype '{mtype}'"
                if available_mtypes:
                    error_msg += f". Available for: {available_mtypes}"
                if has_default:
                    error_msg += ". Default implementation available but failed to load or was not applicable."

                raise NotImplementedError(error_msg)

            # Call the found implementation with the object as the only argument.
            return implementation(obj)

        # Create and return a property object, setting the docstring directly.
        # The name of the property is set when it's attached to the class.
        prop = property(fget=property_getter, doc=f"Dispatched property for {func_name}")
        return prop


# Global singleton instance of ExtensionDispatcher.
_dispatcher = ExtensionDispatcher()


def get_extension_dispatcher() -> ExtensionDispatcher:
    """
    Get the global singleton :class:`ExtensionDispatcher`.

    Returns
    -------
    ExtensionDispatcher
        The global dispatcher instance.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.extension.dispatcher import get_extension_dispatcher
        dispatcher = get_extension_dispatcher()
    """
    return _dispatcher
