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
from typing import Dict, Any, TYPE_CHECKING, Optional, Union, List, Literal

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
    'external_extension',
]

# Flag: Records whether injection has already been applied to avoid duplicate injection
_applied = False


def apply_extensions(force_reapply: bool = False) -> bool:
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

    Parameters
    ----------
    force_reapply : bool, default=False
        If True, forces re-injection of all extensions even if they were
        previously applied. This is useful when new extensions are registered
        after the initial library import (e.g., external extensions).

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
    
    Examples
    --------
    External extension registration and application:
    
    .. code-block:: python
    
        import axisfuzzy as af
        from axisfuzzy.extension import extension, apply_extensions
        
        # Define external extension
        @extension(name='custom_metric', mtype='qrofn')
        def my_metric(self):
            return self.md ** 2 + self.nmd ** 2
        
        # Apply the new extension
        apply_extensions(force_reapply=True)
        
        # Now it's available
        fuzz = af.Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.3)
        result = fuzz.custom_metric()  # Works!
    """
    global _applied
    if _applied and not force_reapply:
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


def external_extension(name: str,
                       mtype: Optional[str] = None,
                       target_classes: Union[str, List[str]] = None,
                       injection_type: Literal[
                                    'instance_method',
                                    'instance_property',
                                    'top_level_function',
                                    'both'] = 'both',
                       is_default: bool = False,
                       priority: int = 0,
                       auto_apply: bool = True,
                       **kwargs):
    """
    Convenient decorator for registering external extensions with automatic application.
    
    This function is a wrapper around the @extension decorator that automatically
    applies the extension after registration, making it immediately available for use.
    This is especially useful for external libraries and user-defined extensions
    that need to be dynamically added after AxisFuzzy has been imported.
    
    Parameters
    ----------
    name : str
        Extension name (e.g. 'distance').
    mtype : str or None, optional
        Target fuzzy-number type (e.g. 'qrofn'). If None the registration is
        considered a general/default implementation.
    target_classes : str or list of str or None, optional
        Class name or list of class names to inject into (e.g. 'Fuzznum' or
        ['Fuzznum', 'Fuzzarray']). If None, library conventions are used.
    injection_type : str, optional
        Injection mode. Default is 'both'.
    is_default : bool, optional
        Whether this registration is a fallback when no mtype-specific
        implementation exists. Default is False.
    priority : int, optional
        Resolution priority when multiple candidates match. Higher values
        take precedence. Default is 0.
    auto_apply : bool, optional
        Whether to automatically apply extensions after registration.
        Default is True.
    **kwargs
        Additional metadata forwarded to the registry.
        
    Returns
    -------
    callable
        A decorator that accepts the implementation function and registers it.
        
    Examples
    --------
    Simple external extension:
    
    >>> import axisfuzzy as af
    >>> from axisfuzzy.extension import external_extension
    >>> 
    >>> @external_extension('custom_score', mtype='qrofn')
    ... def my_score_function(self):
    ...     return self.md ** 2 - self.nmd ** 2
    >>> 
    >>> # Immediately available without manual apply_extensions() call
    >>> fuzz = af.Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.3)
    >>> score = fuzz.custom_score()  # Works immediately!
        
    Notes
    -----
    This function is recommended for external extensions as it handles the
    complexity of re-injection automatically. For internal extensions (within
    the AxisFuzzy library), continue using the standard @extension decorator.
    """

    def decorator(func):
        # First register using the standard extension decorator
        extension_decorator = extension(
            name=name,
            mtype=mtype,
            target_classes=target_classes,
            injection_type=injection_type,
            is_default=is_default,
            priority=priority,
            **kwargs
        )

        # Apply the extension decorator
        registered_func = extension_decorator(func)

        # Automatically apply extensions if requested
        if auto_apply:
            success = apply_extensions(force_reapply=True)
            if not success:
                warnings.warn(
                    f"Failed to automatically apply extension '{name}' for mtype '{mtype}'. "
                    "You may need to call apply_extensions(force_reapply=True) manually.",
                    RuntimeWarning
                )

        return registered_func

    return decorator
