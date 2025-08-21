#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Extension Registry for AxisFuzzy.

This module implements the extension registry that records and resolves
"extensions" (external functions) by fuzzy-number type (``mtype``). It is the
book-keeping backbone of the AxisFuzzy extension system and works together with:

- axisfuzzy.extension.__init__ (activation entrypoint, ``apply_extensions``)
- axisfuzzy.extension.decorator (declarative registration via ``@extension``, ``@batch_extension``)
- axisfuzzy.extension.dispatcher (runtime dispatch proxies: instance methods, properties, top-level)
- axisfuzzy.extension.injector (binds proxies to Fuzznum/Fuzzarray classes and module namespace)

For a high-level overview of the extension architecture (Registration → Dispatch → Injection),

Notes
-----
- The registry is thread-safe and supports both specialized (mtype-specific) and
  default implementations for the same function name.
- Implementations can specify how they should be exposed via ``injection_type``:
  'instance_method', 'instance_property', 'top_level_function', or 'both'.
- Priority values are used to prevent overwriting an existing implementation with an
  equal or lower priority.

Examples
--------
Register a specialized instance method for 'qrofn':

.. code-block:: python

    from axisfuzzy.extension import extension
    from axisfuzzy.core import Fuzznum

    @extension(name='distance', mtype='qrofn', target_classes=['Fuzznum'])
    def qrofn_distance(x: Fuzznum, y: Fuzznum, p: int = 2) -> float:
        q = x.q
        return (((abs(x.md**q - y.md**q))**p + (abs(x.nmd**q - y.nmd**q))**p) / 2) ** (1/p)

Register a dispatched read-only property:

.. code-block:: python

    @extension(name='score', mtype='qrofn',
               target_classes=['Fuzznum', 'Fuzzarray'],
               injection_type='instance_property')
    def qrofn_score(obj):
        return obj.md ** obj.q - obj.nmd ** obj.q

Register a default fallback exposed also as a top-level function:

.. code-block:: python

    @extension(name='normalize', is_default=True, target_classes=['Fuzznum'],
               injection_type='both')
    def default_normalize(x):
        # generic fallback
        return x
"""

import threading
import datetime

from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Tuple, Callable, Any, Union


@dataclass
class FunctionMetadata:
    """
    Metadata describing a registered extension function.

    Attributes
    ----------
    name : str
        Logical extension name (e.g., 'distance', 'score', '_random').
    mtype : str or None
        Target fuzzy-number type for the specialized implementation
        (e.g., 'qrofn'). ``None`` indicates a default (fallback) implementation.
    target_classes : list of str
        Class names to inject into when exposed as instance members,
        typically a subset of ['Fuzznum', 'Fuzzarray'].
    injection_type : {'instance_method', 'instance_property', 'top_level_function', 'both'}
        Exposure mode:
        - 'instance_method': dispatched bound method on target classes
        - 'instance_property': dispatched read-only property on target classes
        - 'top_level_function': function injected into the top-level module namespace
        - 'both': both instance method and top-level function
    is_default : bool, optional
        Whether this is a default (fallback) implementation for ``name``.
        Defaults to False.
    priority : int, optional
        Registration priority. Higher values take precedence when preventing
        re-registration with lower or equal priority. Defaults to 0.
    description : str, optional
        Short human-readable description for documentation. Defaults to "".

    Notes
    -----
    This metadata is consumed by the injector and dispatcher to determine
    how and where an extension should be exposed after registration.
    """
    name: str
    mtype: Optional[str]
    target_classes: List[str]
    injection_type: Literal[
        'instance_method',
        'instance_property',
        'top_level_function',
        'both']
    is_default: bool = False
    priority: int = 0
    description: str = ""


class ExtensionRegistry:
    """
    Thread-safe registry for AxisFuzzy extension functions.

    The registry stores multiple implementations per logical extension name:
    - at most one default (fallback) implementation
    - zero or more specialized implementations keyed by ``mtype``

    It provides:
    - A decorator factory (:meth:`register`) to register implementations
    - Lookup by (name, mtype) with default fallback (:meth:`get_function`)
    - Introspection helpers for documentation and injection

    Notes
    -----
    The registry does not perform injection itself. Injection happens later
    via :mod:`axisfuzzy.extension.injector`, which reads the metadata here and
    attaches dispatcher proxies created by :mod:`axisfuzzy.extension.dispatcher`.

    See Also
    --------
    axisfuzzy.extension.decorator : User-facing decorators that call this registry.
    axisfuzzy.extension.injector : Attaches proxies to classes or module.
    axisfuzzy.extension.dispatcher : Builds dispatched proxies used at runtime.
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
                 injection_type: Literal[
                     'instance_method',
                     'instance_property',
                     'top_level_function',
                     'both'] = 'both',
                 is_default: bool = False,
                 priority: int = 0,
                 **kwargs) -> Callable:
        """
        Decorator factory to register an extension function.

        This method is used by :func:`axisfuzzy.extension.decorator.extension`
        (or :func:`axisfuzzy.extension.decorator.batch_extension`) to declare a
        function as a dispatched extension. It records the function and its
        :class:`FunctionMetadata` in a thread-safe manner.

        Parameters
        ----------
        name : str
            Extension name under which the function is registered.
        mtype : str or None, optional
            Specialized fuzzy-number type. If ``None``, registers as default
            implementation for ``name``.
        target_classes : str or list of str, optional
            Injection targets. If ``None``, defaults to ['Fuzznum', 'Fuzzarray'].
        injection_type : Literal['instance_method', 'instance_property', 'top_level_function', 'both'], optional
            Exposure mode (method/property/top-level/both). Default is 'both'.
        is_default : bool, optional
            Register as default implementation. Default is False.
        priority : int, optional
            Priority for conflict prevention. Existing entries with higher or
            equal priority block re-registration. Default is 0.
        **kwargs
            Additional metadata stored into :class:`FunctionMetadata`.

        Returns
        -------
        callable
            A decorator that takes the implementation function and registers it.

        Raises
        ------
        ValueError
            If attempting to re-register a default or specialized implementation
            when an existing one with higher or equal priority is already present.

        Examples
        --------
        Specialized method:

        .. code-block:: python

            from axisfuzzy.extension import extension
            from axisfuzzy.core import Fuzznum

            @extension(name='distance', mtype='qrofn', target_classes=['Fuzznum'])
            def qrofn_distance(x: Fuzznum, y: Fuzznum) -> float:
                q = x.q
                return ((abs(x.md**q - y.md**q)**2 + abs(x.nmd**q - y.nmd**q)**2)/2) ** 0.5

        Default top-level + instance:

        .. code-block:: python

            @extension(name='normalize', is_default=True,
                       target_classes=['Fuzznum'], injection_type='both')
            def normalize_default(x): return x

        Dispatched read-only property:

        .. code-block:: python

            @extension(name='score', mtype='qrofn',
                       target_classes=['Fuzznum','Fuzzarray'],
                       injection_type='instance_property')
            def qrofn_score(obj): return obj.md**obj.q - obj.nmd**obj.q
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
        Retrieve a function implementation for ``(name, mtype)`` with fallback.

        The lookup algorithm is:
        1) Try specialized implementation registered for this ``mtype``.
        2) If not found, return the default implementation for ``name`` (if any).
        3) Otherwise return ``None``.

        Parameters
        ----------
        name : str
            Extension name to look up.
        mtype : str
            Fuzzy-number type for which to retrieve the specialized implementation.

        Returns
        -------
        callable or None
            The resolved implementation function or ``None`` if not found.

        Examples
        --------
        .. code-block:: python

            reg = get_registry_extension()
            fn = reg.get_function('distance', 'qrofn')  # specialized
            if fn is None:
                raise RuntimeError('No distance registered')
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

    def get_top_level_function_names(self) -> List[str]:
        """
        List function names that should be injected as top-level functions.

        This scans both specialized and default registrations and collects
        any name whose ``injection_type`` is 'top_level_function' or 'both'.

        Returns
        -------
        list of str
            Sorted unique function names requiring top-level injection.

        Examples
        --------
        .. code-block:: python

            reg = get_registry_extension()
            for fn_name in reg.get_top_level_function_names():
                print('top-level:', fn_name)
        """
        names = set()
        for func_name, implementations in self._functions.items():
            for mtype, (func, metadata) in implementations.items():
                if metadata.injection_type in ('top_level_function', 'both'):
                    names.add(func_name)
                    # 只要找到一个，就可以确定该函数名需要顶层注入，跳到下一个函数名
                    break
        # 检查所有默认实现
        for func_name, (func, metadata) in self._defaults.items():
            if metadata.injection_type in ('top_level_function', 'both'):
                names.add(func_name)

        return sorted(list(names))

    def get_metadata(self, name: str, mtype: Optional[str] = None) -> Optional[FunctionMetadata]:
        """
        Retrieve metadata for a registered function.

        Parameters
        ----------
        name : str
            Extension name to inspect.
        mtype : str or None, optional
            If provided, returns specialized metadata for that ``mtype``.
            Otherwise, returns default metadata when available.

        Returns
        -------
        FunctionMetadata or None
            Stored metadata object or ``None`` if not present.

        Examples
        --------
        .. code-block:: python

            reg = get_registry_extension()
            meta = reg.get_metadata('distance', 'qrofn')
            if meta:
                print(meta.injection_type, meta.target_classes)
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
        List all registered names with their implementation summaries.

        The result is a structured summary that groups specialized and default
        registrations per logical name. This is primarily intended for
        documentation, debugging, and injection planning.

        Returns
        -------
        dict
            A dictionary with the following structure:

            .. code-block:: xml

                {
                    "distance": {
                        "implementations": {
                            "qrofn": {
                                "priority": 0,
                                "target_classes": ["Fuzznum", "Fuzzarray"],
                                "injection_type": "both",
                            }
                        },
                        "default": {
                            "priority": 0,
                            "target_classes": ["Fuzznum"],
                            "injection_type": "instance_method",
                        },
                    },
                    "_random": {
                        "implementations": {
                            "qrofn": {
                                "priority": 0,
                                "target_classes": ["Fuzznum"],
                                "injection_type": "top_level_function",
                            }
                        },
                        "default": None,
                    },
                }

        Examples
        --------
        .. code-block:: python

            reg = get_registry_extension()
            summary = reg.list_functions()
            for name, info in summary.items():
                print(name, '=>', info)
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
        Current timestamp helper (ISO 8601).

        Returns
        -------
        str
            ISO formatted timestamp.

        Examples
        --------
        .. code-block:: python

            ts = ExtensionRegistry._get_timestamp()
        """
        return datetime.datetime.now().isoformat()


# Global singleton instance of ExtensionRegistry.
_extension_registry = None
# Lock to ensure thread-safe initialization of the singleton.
_extension_registry_lock = threading.RLock()


def get_registry_extension() -> ExtensionRegistry:
    """
    Get the global singleton :class:`ExtensionRegistry`.

    Implements double-checked locking to initialize the singleton in a
    thread-safe manner on first use.

    Returns
    -------
    ExtensionRegistry
        The global registry instance.

    Examples
    --------
    .. code-block:: python

        reg = get_registry_extension()
        # Use reg.register(...) via decorators in axisfuzzy.extension.decorator
        # or call reg.get_function(...) during dispatch.
    """
    global _extension_registry
    # Double-checked locking for thread-safe singleton initialization.
    if _extension_registry is None:
        with _extension_registry_lock:
            if _extension_registry is None:
                _extension_registry = ExtensionRegistry()
    return _extension_registry
