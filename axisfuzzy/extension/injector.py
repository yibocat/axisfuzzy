#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Extension Injector for AxisFuzzy.

This module is responsible for "activating" registered extensions: it reads
the metadata from the ExtensionRegistry and creates appropriate dispatching
proxies using the ExtensionDispatcher, then attaches them to:

- Target classes as instance methods or properties (e.g., Fuzznum, Fuzzarray)
- The top-level module namespace as functions (e.g., axisfuzzy.distance)

Architecture
------------
- Registration: ``@extension`` and ``@batch_extension`` capture functions and
  metadata into the global registry.
- Dispatch: ``ExtensionDispatcher`` builds proxies that pick the right function
  at call-time based on ``mtype``.
- Injection: ``ExtensionInjector`` uses registry summaries to decide where
  and how to attach proxies. Typically triggered once during library init
  (e.g., by :func:`axisfuzzy.extension.apply_extensions`).

Notes
-----
- Injection avoids overwriting: attributes are only added if not already present
  on target classes or the module namespace.
- A single proxy per logical name is created and shared across targets.

Examples
--------
Inject all registered extensions during initialization:

.. code-block:: python

    from axisfuzzy.extension import get_extension_injector
    from axisfuzzy.core import Fuzznum, Fuzzarray
    import axisfuzzy

    injector = get_extension_injector()
    class_map = {'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}
    injector.inject_all(class_map, axisfuzzy.__dict__)
"""

from typing import Dict, Type, Any

from .registry import get_registry_extension
from .dispatcher import get_extension_dispatcher


class ExtensionInjector:
    """
    Injector that binds dispatched proxies to classes and module namespace.

    The injector composes the registry (for metadata) and the dispatcher
    (for proxy creation) to make extensions available to users.

    See Also
    --------
    axisfuzzy.extension.registry : Stores registered functions and metadata.
    axisfuzzy.extension.dispatcher : Builds call-time dispatch proxies.
    """

    def __init__(self):
        """
        Initialize the injector and bind to the global registry and dispatcher.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.extension import ExtensionInjector
            inj = ExtensionInjector()
        """
        self.registry = get_registry_extension()
        self.dispatcher = get_extension_dispatcher()

    def inject_all(self, class_map: Dict[str, Type], module_namespace: Dict[str, Any]):
        """
        Inject all registered extension functions based on registry metadata.

        This scans the registry summary (``list_functions``) to:
        - Determine union of target classes declaring interest for a function.
        - Determine union of injection types ('instance_method', 'instance_property',
          'top_level_function', 'both').
        - Create at most one proxy per function name and attach to requested targets.

        Parameters
        ----------
        class_map : dict
            Maps class names to class objects (e.g., {'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}).
        module_namespace : dict
            Target module ``__dict__`` to inject top-level functions.

        Notes
        -----
        - Safe injection: existing attributes are not overwritten.
        - Idempotent in practice: repeated calls do not duplicate attributes,
          as existence checks prevent redefinition.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.core import Fuzznum, Fuzzarray
            import axisfuzzy
            from axisfuzzy.extension import get_extension_injector

            inj = get_extension_injector()
            inj.inject_all({'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}, axisfuzzy.__dict__)
        """
        functions = self.registry.list_functions()

        for func_name, func_info in functions.items():
            self._inject_function(func_name, func_info, class_map, module_namespace)

    def _inject_function(self,
                         func_name: str,
                         func_info: Dict[str, Any],
                         class_map: Dict[str, Type],
                         module_namespace: Dict[str, Any]):
        """
        Inject a single logical function according to aggregated metadata.

        The injector collects all declared target classes and injection types
        across specialized and default registrations, then attaches a single
        proxy per exposure kind.

        Parameters
        ----------
        func_name : str
            Logical extension name.
        func_info : dict
            Metadata summary as returned by :meth:`ExtensionRegistry.list_functions`.
        class_map : dict
            Mapping from class names to class objects.
        module_namespace : dict
            Target module namespace for top-level functions.

        Notes
        -----
        - Instance method/property: bound to each declared class if not already present.
        - Top-level functions: set on the module namespace if not already present.

        Examples
        --------
        .. code-block:: python

            inj._inject_function('distance', functions['distance'], class_map, axisfuzzy.__dict__)
        """
        # Collect all unique target classes and injection types declared for this function.
        target_classes = set()
        injection_types = set()

        # Gather information from specialized implementations.
        for mtype_info in func_info['implementations'].values():
            target_classes.update(mtype_info['target_classes'])
            injection_types.add(mtype_info['injection_type'])

        # Gather information from the default implementation, if it exists.
        if func_info['default']:
            target_classes.update(func_info['default']['target_classes'])
            injection_types.add(func_info['default']['injection_type'])

        # Inject as an instance method if required by any registration.
        if any(it in ['instance_method', 'both'] for it in injection_types):
            # Create a single instance method dispatcher for this function name.
            method_dispatcher = self.dispatcher.create_instance_method(func_name)

            # Attach the dispatcher to all specified target classes.
            for class_name in target_classes:
                if class_name in class_map:
                    cls = class_map[class_name]
                    # Only inject if the attribute does not already exist on the class.
                    # This prevents overwriting existing methods or properties.
                    if not hasattr(cls, func_name):
                        setattr(cls, func_name, method_dispatcher)

        # Inject as an instance property if required by any registration.
        if 'instance_property' in injection_types:
            property_dispatcher = self.dispatcher.create_instance_property(func_name)
            for class_name in target_classes:
                if class_name in class_map:
                    cls = class_map[class_name]
                    if not hasattr(cls, func_name):
                        setattr(cls, func_name, property_dispatcher)

        # Inject as a top-level function if required by any registration.
        if any(it in ['top_level_function', 'both'] for it in injection_types):
            # Create a single top-level function dispatcher for this function name.
            function_dispatcher = self.dispatcher.create_top_level_function(func_name)

            # Only inject if the attribute does not already exist in the module namespace.
            # This prevents overwriting existing functions or variables.
            if func_name not in module_namespace:
                module_namespace[func_name] = function_dispatcher


# Global singleton instance of ExtensionInjector.
_injector = ExtensionInjector()


def get_extension_injector() -> ExtensionInjector:
    """
    Get the global singleton :class:`ExtensionInjector`.

    Returns
    -------
    ExtensionInjector
        The global injector instance.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.extension import get_extension_injector
        injector = get_extension_injector()
    """
    return _injector
