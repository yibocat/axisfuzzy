#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

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

The `apply_extensions()` function is the entry point to activate the entire
extension system, typically called once during FuzzLab's library loading.
"""

import sys
import warnings
from typing import Dict, Any, TYPE_CHECKING

from .registry import get_registry_extension
from .dispatcher import get_extension_dispatcher
from .injector import get_extension_injector
from .decorator import extension, batch_extension

if TYPE_CHECKING:
    from ..core import Fuzznum, Fuzzarray

__all__ = [
    'get_registry_extension',
    'get_extension_dispatcher',
    'get_extension_injector',
    'extension',
    'batch_extension',
    'apply_extensions',
]

# Flag: Records whether injection has already been applied to avoid duplicate injection
_applied = False


def apply_extensions() -> bool:
    """
    Applies all registered extension functions to their respective targets.

    This function is the master switch for the AxisFuzzy extension system.
    It dynamically discovers core classes (`Fuzznum`, `Fuzzarray`) and the
    top-level `axisfuzzy` module, then delegates to the `ExtensionInjector`
    to perform the injection of all registered functions.

    This process makes specialized and default extension functions available
    as methods on `Fuzznum`/`Fuzzarray` instances and as top-level functions
    in the `axisfuzzy` namespace.

    This function is designed to be idempotent and safe to call multiple times.
    It is typically called once during the initial import of the `axisfuzzy` library.

    Returns
    -------
    bool
        True if extensions were applied successfully or were already applied.
        False if an error occurred during injection.

    Notes
    -----
    This function uses runtime imports and dynamic module lookups to break
    potential circular dependencies between the core library and the
    extension system, which is crucial for robust initialization.
    """
    global _applied
    if _applied:
        return True

    try:
        # 1. Dynamically and lazily importing core classes is the key to breaking circular imports.
        from ..core import Fuzznum, Fuzzarray
        class_map: Dict[str, Any] = {
            'Fuzznum': Fuzznum,
            'Fuzzarray': Fuzzarray,
        }

        # 2. Dynamically find the namespace of the axisfuzzy module
        # This is more robust than relying on external input
        target_module_globals = sys.modules.get('axisfuzzy')
        if target_module_globals is None:
            warnings.warn(
                "The 'axisfuzzy' module was not found in sys.modules. "
                "Top-level extensions will not be injected.",
                ImportWarning
            )
            # Even if the top-level module cannot be found, we can still attempt to inject into the class.
            # So we provide an empty dictionary.
            namespace_dict = {}
        else:
            namespace_dict = target_module_globals.__dict__

        # 3. Execute injection
        injector = get_extension_injector()
        injector.inject_all(class_map, namespace_dict)

        _applied = True
        return True

    except ImportError as e:
        # If core classes cannot be imported, it indicates an issue with the initialization order.
        warnings.warn(
            f"Could not apply extensions because core components could not be imported: {e}. "
            "This might happen during documentation generation or if the library is not fully initialized.",
            ImportWarning
        )
        return False
    except Exception as e:
        # Capture other potential injection errors
        warnings.warn(f"An unexpected error occurred while applying extensions: {e}", RuntimeWarning)
        return False

# def apply_extensions(target_module_globals: Dict[str, Any] | None = None) -> bool:
#     """
#     Applies all registered extension functions to their respective targets.

#     This function is responsible for activating the FuzzLab extension system.
#     It gathers the necessary class mappings and module namespaces, then
#     delegates to the `ExtensionInjector` to perform the dynamic injection
#     of all functions registered via the `@extension` decorator.

#     This process makes the specialized and default extension functions
#     available as methods on `Fuzznum` and `Fuzzarray` instances, and as
#     top-level functions within the `axisfuzzy` module.

#     This function is typically called once during the initial import
#     of the `axisfuzzy` library.

#     Examples:
#         ```python
#         # This function is called automatically when 'import axisfuzzy' is executed.
#         # Developers usually don't need to call it manually.
#         # Its effect is to make methods like fuzznum_instance.distance()
#         # and functions like axisfuzzy.distance() available.
#         ```
#     """
#     global _applied
#     if _applied:
#         return True

#     # Prepare the class map for Fuzznum and Fuzzarray, which are target classes for injection.
#     from ..core import Fuzznum, Fuzzarray
#     class_map = {
#         'Fuzznum': Fuzznum,
#         'Fuzzarray': Fuzzarray,
#     }

#     injectors = get_extension_injector()
#     try:
#         injectors.inject_all(class_map, target_module_globals)    # type: ignore
#         _applied = True
#         return True
#     except Exception as e:
#         # 如果有日志, 可以记录日志
#         # logger.error(f"Failed to apply extensions: {e}")
#         return False
