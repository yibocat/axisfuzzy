#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 15:39
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
Extension Injector for FuzzLab.

This module defines the `ExtensionInjector` class, which is responsible for
dynamically injecting registered extension functions into `Fuzznum`, `Fuzzarray`
classes, and the `fuzzlab` top-level namespace.

It acts as the "activator" of the extension system, taking the metadata from
`fuzzlab.extension.registry.py` and the proxy functions from
`fuzzlab.extension.dispatcher.py` to make extensions callable as methods
or functions.
"""

from typing import Dict, Type, Any

from .registry import get_extension_registry
from .dispatcher import get_extension_dispatcher


class ExtensionInjector:
    """
    Manages the dynamic injection of FuzzLab extension functions.

    The injector reads registered function metadata from the `ExtensionRegistry`
    and uses the `ExtensionDispatcher` to create callable proxies. These proxies
    are then attached to the specified target classes (like `Fuzznum` and `Fuzzarray`)
    or the `fuzzlab` module's global namespace, making the extensions
    seamlessly available to users.
    """

    def __init__(self):
        """
        Initializes the ExtensionInjector.

        It obtains references to the global `ExtensionRegistry` and
        `ExtensionDispatcher` instances.
        """
        self.registry = get_extension_registry()
        self.dispatcher = get_extension_dispatcher()

    def inject_all(self, class_map: Dict[str, Type], module_namespace: Dict[str, Any]):
        """
        Injects all registered extension functions into their respective targets.

        This is the main entry point for the injection process, typically called
        during FuzzLab's initialization (e.g., by `apply_extensions()`).
        It iterates through all functions known to the `ExtensionRegistry`
        and delegates their injection to `_inject_function`.

        Args:
            class_map: A dictionary mapping class names (strings) to their
                actual class objects (e.g., `{'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}`).
                This is used to resolve target classes for method injection.
            module_namespace: The dictionary representing the target module's
                namespace (e.g., `fuzzlab.__dict__`) where top-level functions
                should be injected.

        Examples:
            ```python
            # This is typically called internally by fuzzlab.extension.apply_extensions()
            from fuzzlab.core import Fuzznum, Fuzzarray
            import fuzzlab

            injector = get_extension_injector()
            class_map = {'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}
            module_namespace = fuzzlab.__dict__
            injector.inject_all(class_map, module_namespace)
            ```
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
        Injects a single extension function based on its metadata.

        This method determines where and how a specific function should be
        injected (as an instance method, a top-level function, or both)
        by analyzing its registered metadata. It then uses the `ExtensionDispatcher`
        to create the necessary proxy functions and attaches them to the
        appropriate classes or module namespace.

        Args:
            func_name: The name of the function to inject.
            func_info: A dictionary containing all metadata for the function,
                as returned by `ExtensionRegistry.list_functions()`. This includes
                information about specialized and default implementations.
            class_map: The mapping of class names to class objects.
            module_namespace: The target module's namespace dictionary.
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
    Retrieves the global singleton instance of `ExtensionInjector`.

    This function ensures that only one instance of the injector exists
    across the application, providing a central point for managing
    the injection of extensions.

    Returns:
        The singleton `ExtensionInjector` instance.

    Examples:
        ```python
        injector = get_extension_injector()
        # Use the injector to trigger the injection process
        ```
    """
    return _injector
