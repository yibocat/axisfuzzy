#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Decorators for registering axisfuzzy extension functions.

Provides the `extension` and `batch_extension` decorators which register
callables into the ExtensionRegistry. Registered callables are later injected
into target classes (e.g. ``Fuzznum``, ``Fuzzarray``) or the top-level axisfuzzy
namespace by the injector at library initialization.

Examples
--------
Simple specialized extension registration:

.. code-block:: python

    from axisfuzzy.extension import extension
    from axisfuzzy.core import Fuzznum

    @extension('distance', mtype='qrofn', target_classes=['Fuzznum'])
    def qrofn_distance(f1: Fuzznum, f2: Fuzznum, p: int = 2) -> float:
        # qrofn-specific implementation
        ...
"""

from typing import Optional, Union, List, Literal

from .registry import get_registry_extension


def extension(name: str,
              mtype: Optional[str] = None,
              target_classes: Union[str, List[str]] = None,
              injection_type: Literal[
                  'instance_method',
                  'top_level_function',
                  'instance_property',
                  'both'] = 'both',
              is_default: bool = False,
              priority: int = 0,
              **kwargs):
    """
    Register an extension function in the ExtensionRegistry.

    This decorator simplifies the process of adding new functionalities
    to FuzzLab's ``Fuzznum`` and ``Fuzzarray`` objects, or as top-level functions.
    It acts as a wrapper around ``ExtensionRegistry.register()``, providing
    a declarative way to define extension properties.

    Parameters
    ----------
    name : str
        Extension name (e.g. ``distance``).
    mtype : str or None, optional
        Target fuzzy-number type (e.g. ``qrofn``). If None the registration is
        considered a general/default implementation.
    target_classes : str or list of str or None, optional
        Class name or list of class names to inject into (e.g. ``Fuzznum`` or
        ``["Fuzznum", "Fuzzarray"]``). If None, library conventions are used.
    injection_type : {``instance_method``, ``top_level_function``, ``instance_property``, ``both``}, optional
        Injection mode. Default is ``both``.
    is_default : bool, optional
        Whether this registration is a fallback when no mtype-specific
        implementation exists. Default is ``False``.
    priority : int, optional
        Resolution priority when multiple candidates match. Higher values
        take precedence. Default is 0.
    **kwargs
        Additional metadata forwarded to the registry.

    Returns
    -------
    callable
        A decorator that accepts the implementation function and registers it.

    Examples
    --------
    Specialized distance for qrofn:

    .. code-block:: python

        from axisfuzzy.extension import extension
        from axisfuzzy.core import Fuzznum

        @extension('distance', mtype='qrofn', target_classes=['Fuzznum'])
        def qrofn_distance(f1: Fuzznum, f2: Fuzznum, p: int = 2) -> float:
            # qrofn-specific implementation
            ...

    Default distance fallback:

    .. code-block:: python

        from axisfuzzy.extension import extension
        from axisfuzzy.core import Fuzznum

        @extension('distance', is_default=True, target_classes=['Fuzznum'])
        def default_distance(f1: Fuzznum, f2: Fuzznum) -> float:
            # generic fallback implementation
            ...
    """
    registry = get_registry_extension()
    return registry.register(
        name=name,
        mtype=mtype,
        target_classes=target_classes,
        injection_type=injection_type,
        is_default=is_default,
        priority=priority,
        **kwargs
    )


def batch_extension(registrations: List[dict]):
    """
    Decorator to register a single function with multiple extension configurations.

    This is useful when a single function needs to serve different roles
    or be registered under various conditions (e.g., as a specialized
    implementation for one `mtype` and a default for another, or with
    different injection types).

    Parameters
    ----------
    registrations : list of dict
        Each dict supplies keyword arguments for the `extension` decorator,
        e.g. ``{'name': 'normalize', 'mtype': 'qrofn', 'target_classes': ['Fuzznum']}``.

    Returns
    -------
    callable
        A decorator which registers the decorated function multiple times.

    Examples
    --------
    Registering a function that serves as a specialized 'normalize' for 'qrofn'
    and a default 'normalize' for other types:

    .. code-block:: python

        # In axisfuzzy/fuzzy/qrofs/_func.py
        from axisfuzzy.extension import batch_extension
        from axisfuzzy.core import Fuzznum

        @batch_extension([
            {'name': 'normalize', 'mtype': 'qrofn', 'target_classes': ['Fuzznum']},
            {'name': 'normalize', 'is_default': True, 'target_classes': ['Fuzznum']}
        ])
        def qrofn_normalize(fuzz: Fuzznum) -> Fuzznum:
            # QROFN specific normalization logic
            # ...
            return fuzz
    """
    def decorator(func):
        registry = get_registry_extension()
        for reg in registrations:
            registry.register(**reg)(func)
        return func
    return decorator
