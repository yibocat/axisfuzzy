#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 14:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
FuzzLab Extension System Initialization.

This package provides the core infrastructure for FuzzLab's extensible
architecture, allowing developers to register and inject specialized
functions for different fuzzy number types (`mtype`).

It includes:
- `registry.py`: Manages the registration of extension functions and their metadata.
- `decorator.py`: Provides `@extension` and `@batch_extension` for declarative registration.
- `dispatcher.py`: Creates dynamic proxy functions for runtime dispatching based on `mtype`.
- `injector.py`: Handles the dynamic injection of registered functions into classes and modules.
- `utils.py`: Offers utility functions like `call_extension` for internal calls between extensions.

The `apply_extensions()` function is the entry point to activate the entire
extension system, typically called once during FuzzLab's library loading.
"""

from .registry import get_extension_registry
from .dispatcher import get_extension_dispatcher
from .injector import get_extension_injector
from .decorator import extension, batch_extension
from .utils import call_extension


def apply_extensions():
    """
    Applies all registered extension functions to their respective targets.

    This function is responsible for activating the FuzzLab extension system.
    It gathers the necessary class mappings and module namespaces, then
    delegates to the `ExtensionInjector` to perform the dynamic injection
    of all functions registered via the `@extension` decorator.

    This process makes the specialized and default extension functions
    available as methods on `Fuzznum` and `Fuzzarray` instances, and as
    top-level functions within the `fuzzlab` module.

    This function is typically called once during the initial import
    of the `fuzzlab` library.

    Examples:
        ```python
        # This function is called automatically when 'import fuzzlab' is executed.
        # Developers usually don't need to call it manually.
        # Its effect is to make methods like fuzznum_instance.distance()
        # and functions like fuzzlab.distance() available.
        ```
    """
    # Prepare the class map for Fuzznum and Fuzzarray, which are target classes for injection.
    from ..core import Fuzznum, Fuzzarray
    class_map = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray,
    }

    # Get the namespace of the 'fuzzlab' module for injecting top-level functions.
    import fuzzlab
    # Explicitly convert to dict to satisfy type checkers, as __dict__ can be a property.
    module_namespace = fuzzlab.__dict__

    # Get the singleton injector instance and trigger the injection process.
    apply_injector = get_extension_injector()
    apply_injector.inject_all(class_map, module_namespace)


__all__ = [
    'get_extension_registry',
    'get_extension_dispatcher',
    'get_extension_injector',
    'extension',
    'batch_extension',
    'apply_extensions',
    'call_extension',
]