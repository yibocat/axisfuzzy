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

The `apply_extensions()` function is the entry point to activate the entire
extension system, typically called once during FuzzLab's library loading.
"""
import warnings
from typing import Dict, Any

from .registry import get_registry_extension
from .dispatcher import get_extension_dispatcher
from .injector import get_extension_injector
from .decorator import extension, batch_extension

__all__ = [
    'get_registry_extension',
    'get_extension_dispatcher',
    'get_extension_injector',
    'extension',
    'batch_extension',
    'apply_extensions',
]

# 标志：记录是否已经应用过注入，避免重复注入
_applied = False


def apply_extensions(target_module_globals: Dict[str, Any] | None = None) -> bool:
    """
    Applies all registered extension functions to their respective targets.

    This function is responsible for activating the FuzzLab extension system.
    It gathers the necessary class mappings and module namespaces, then
    delegates to the `ExtensionInjector` to perform the dynamic injection
    of all functions registered via the `@extension` decorator.

    This process makes the specialized and default extension functions
    available as methods on `Fuzznum` and `Fuzzarray` instances, and as
    top-level functions within the `axisfuzzy` module.

    This function is typically called once during the initial import
    of the `axisfuzzy` library.

    Examples:
        ```python
        # This function is called automatically when 'import axisfuzzy' is executed.
        # Developers usually don't need to call it manually.
        # Its effect is to make methods like fuzznum_instance.distance()
        # and functions like axisfuzzy.distance() available.
        ```
    """
    global _applied
    if _applied:
        return True

    # Prepare the class map for Fuzznum and Fuzzarray, which are target classes for injection.
    from ..core import Fuzznum, Fuzzarray
    class_map = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray,
    }

    # Get the namespace of the 'axisfuzzy' module for injecting top-level functions.
    import axisfuzzy

    injectors = get_extension_injector()
    try:
        injectors.inject_all(class_map, target_module_globals)    # type: ignore
        _applied = True
        return True
    except Exception as e:
        # 如果有日志, 可以记录日志
        # logger.error(f"Failed to apply extensions: {e}")
        return False
