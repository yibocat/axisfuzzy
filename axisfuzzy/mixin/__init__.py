#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
AxisFuzzy Mixin System Initialization Module.

This module serves as the entry point for the AxisFuzzy mixin system, which provides
mtype-agnostic structural operations for :class:`Fuzznum` and :class:`Fuzzarray`
classes. It handles the registration import, dynamic injection, and initialization
of all mixin functions.

The mixin system enables NumPy-like operations such as ``reshape``, ``flatten``,
``transpose``, and ``concatenate`` that work uniformly across all fuzzy number types
without requiring mtype-specific dispatch logic.

Key Components
--------------
- **Registration Import**: Automatically imports :mod:`axisfuzzy.mixin.register` to
  trigger all ``@register_mixin`` decorators during module loading.
- **Dynamic Injection**: Provides :func:`apply_mixins` to inject registered functions
  into target classes and the module namespace.
- **Initialization Control**: Maintains global state to ensure injection happens only once.

Architecture
------------
The mixin system follows a three-phase lifecycle:

1. **Registration Phase**: Functions are registered via ``@register_mixin`` decorators
   when :mod:`axisfuzzy.mixin.register` is imported.

2. **Storage Phase**: The :class:`MixinFunctionRegistry` stores all registered
   functions and their metadata.

3. **Injection Phase**: :func:`apply_mixins` dynamically attaches registered functions
   to :class:`Fuzznum`/:class:`Fuzzarray` classes and the top-level namespace.

Usage
-----
The mixin system is typically activated during library initialization:

.. code-block:: python

    from axisfuzzy.mixin import apply_mixins

    # Inject all registered mixin functions
    apply_mixins()

    # Now functions are available:
    # arr.reshape(2, 3)  # instance method
    # axisfuzzy.reshape(arr, 2, 3)  # top-level function

Notes
-----
- Injection is idempotent: multiple calls to :func:`apply_mixins` are safe.
- Functions are injected into the main :mod:`axisfuzzy` namespace by default.
- Registration happens automatically when this module is imported.
- Failed injections generate warnings but don't raise exceptions.

See Also
--------
axisfuzzy.mixin.registry : Core registry and injection infrastructure.
axisfuzzy.mixin.register : Registration declarations for standard operations.
axisfuzzy.mixin.factory : Implementation layer for mixin operations.
"""

import warnings
from typing import Dict, Any


from .registry import get_registry_mixin
from ..core import Fuzznum, Fuzzarray

# Critical fix: Import the register module to trigger execution of all @register decorators.
# This import itself doesn't use any variables, but its side effect is to populate the mixin registry.
from . import register

_applied = False


def _apply_functions(target_module_globals: Dict[str, Any] | None = None) -> bool:
    """
    Dynamically inject registered functions into target module namespace and Fuzznum/Fuzzarray classes.

    This function performs the final injection phase of the mixin system, attaching
    all registered functions to their specified targets. It handles both instance
    method injection (on classes) and top-level function injection (in module namespace).

    Parameters
    ----------
    target_module_globals : dict or None, optional
        Target module's global namespace for top-level function injection.
        If None, defaults to :mod:`axisfuzzy` package globals.

    Returns
    -------
    bool
        True if injection succeeds or has already been applied; False if injection fails.

    Notes
    -----
    - Injection is idempotent: subsequent calls return True immediately.
    - Failures are handled gracefully with warnings rather than exceptions.
    - Falls back to local module globals if main package import fails.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.mixin import _apply_functions

        # Inject into axisfuzzy package (default)
        success = _apply_functions()

        # Inject into custom namespace
        my_globals = {}
        success = _apply_functions(my_globals)
    """
    global _applied
    if _applied:
        return True

    # prepare class map
    class_map = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray
    }

    # determine where to inject top-level functions: prefer axisfuzzy package
    if target_module_globals is None:
        try:
            import axisfuzzy
            target_module_globals = axisfuzzy.__dict__
        except Exception as e:
            # Log or handle the exception as needed
            # _logger.exception("Failed to import axisfuzzy for mixin top-level injection: %s", e)

            warnings.warn(f"Failed to import axisfuzzy for mixin top-level injection: {e}")
            # fallback to local mixin module globals to avoid complete failure
            target_module_globals = globals()

    try:
        get_registry_mixin().build_and_inject(class_map, target_module_globals)
        _applied = True
        return True
    except Exception as e:
        # Log or handle the exception as needed
        # _logger.exception("Failed to apply mixin functions: %s", e)

        warnings.warn(f"Failed to injection mixin functions: {e}")
        return False


# Automatic injection (preserved for backward compatibility), with idempotent and exception protection
# _apply_functions()

apply_mixins = _apply_functions

__all__ = ['get_registry_mixin', 'apply_mixins'] + get_registry_mixin().get_top_level_function_names()
