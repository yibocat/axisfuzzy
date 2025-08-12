#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 15:35
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
Decorators for registering FuzzLab extension functions.

This module provides convenient decorators (`@extension` and `@batch_extension`)
to simplify the process of registering custom functions with the FuzzLab
extension registry. These decorators act as the primary interface for developers
to integrate their specialized or general fuzzy number operations into the
FuzzLab framework.

Functions registered via these decorators are later injected into `Fuzznum`
and `Fuzzarray` classes or the `fuzzlab` top-level namespace by the
`fuzzlab.extension.injector.py` module, based on the provided metadata.
"""

from typing import Optional, Union, List, Literal

from .registry import get_extension_registry


def extension(name: str,
              mtype: Optional[str] = None,
              target_classes: Union[str, List[str]] = None,
              injection_type: Literal['instance_method', 'top_level_function', 'both'] = 'both',
              is_default: bool = False,
              priority: int = 0,
              **kwargs):
    """
    Decorator to register a single FuzzLab extension function.

    This decorator simplifies the process of adding new functionalities
    to FuzzLab's `Fuzznum` and `Fuzzarray` objects, or as top-level functions.
    It acts as a wrapper around `ExtensionRegistry.register()`, providing
    a declarative way to define extension properties.

    Args:
        name: The name of the extension function (e.g., 'distance', '_random').
        mtype: The specific fuzzy number type (e.g., 'qrofn', 'ivfn') this
            implementation is for. If `None`, this function is considered
            a general or default implementation.
        target_classes: A string or list of strings representing the names of
            classes (e.g., 'Fuzznum', 'Fuzzarray') that this extension is
            intended to be injected into as an instance method. If `None`,
            it defaults to `['Fuzznum', 'Fuzzarray']`.
        injection_type: Specifies how the function should be injected:
            'instance_method': As a method of `target_classes`.
            'top_level_function': As a function in the `fuzzlab` module namespace.
            'both': Both as an instance method and a top-level function.
        is_default: A boolean indicating if this is a default implementation
            for the given `name`. Default implementations are used when no
            `mtype`-specific implementation is found.
        priority: An integer representing the priority of this implementation.
            Higher values indicate higher priority. Used to resolve conflicts
            during registration (e.g., preventing lower priority re-registrations).
        **kwargs: Additional keyword arguments to be stored as metadata
            alongside the function in the registry.

    Returns:
        A decorator function that takes the actual implementation function
        as an argument and registers it with the `ExtensionRegistry`.

    Examples:
        Registering a specialized 'distance' function for 'qrofn' mtype:
        ```python
        # In fuzzlab/fuzzy/qrofs/_func.py
        from fuzzlab.extension import extension
        from fuzzlab.core import Fuzznum

        @extension('distance', mtype='qrofn', target_classes=['Fuzznum'])
        def qrofn_distance(fuzz1: Fuzznum, fuzz2: Fuzznum, p: int = 2) -> float:
            # QROFN specific distance calculation
            q = fuzz1.q
            md_diff = abs(fuzz1.md ** q - fuzz2.md ** q) ** p
            nmd_diff = abs(fuzz1.nmd ** q - fuzz2.nmd ** q) ** p
            return ((md_diff + nmd_diff) / 2) ** (1 / p)
        ```

        Registering a general 'distance' function as a default implementation:
        ```python
        # In a general extension module (e.g., fuzzlab/extension/general_funcs.py)
        from fuzzlab.extension import extension
        from fuzzlab.core import Fuzznum

        @extension('distance', is_default=True, target_classes=['Fuzznum'])
        def default_distance(fuzz1: Fuzznum, fuzz2: Fuzznum) -> float:
            # Generic distance calculation for Fuzznum objects
            # This will be used if no mtype-specific 'distance' is found.
            return (fuzz1.md - fuzz2.md)**2 + (fuzz1.nmd - fuzz2.nmd)**2
        ```

        Registering a constructor-like function as a top-level function:
        ```python
        # In fuzzlab/fuzzy/qrofs/_func.py
        from fuzzlab.extension import extension
        from fuzzlab.core import Fuzznum
        import _random

        @extension('random_qrofn', mtype='qrofn', injection_type='top_level_function')
        def create_random_qrofn(mu: float = 0.5, nu: float = 0.5, q: int = 2) -> Fuzznum:
            # Creates a _random QROFN instance
            md = _random.uniform(0, mu)
            nmd = _random.uniform(0, nu)
            return Fuzznum(md=md, nmd=nmd, q=q, mtype='qrofn')
        ```
    """
    registry = get_extension_registry()
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

    Args:
        registrations: A list of dictionaries, where each dictionary represents
            a set of arguments for the `extension` decorator (e.g., `{'name': 'func_name', 'mtype': 'type1'}`).
            Each dictionary will be passed as `**kwargs` to `ExtensionRegistry.register()`.

    Returns:
        A decorator function that takes the actual implementation function
        as an argument and registers it multiple times with the `ExtensionRegistry`
        according to the provided configurations.

    Examples:
        Registering a function that serves as a specialized 'normalize' for 'qrofn'
        and a default 'normalize' for other types:
        ```python
        # In fuzzlab/fuzzy/qrofs/_func.py
        from fuzzlab.extension import batch_extension
        from fuzzlab.core import Fuzznum

        @batch_extension([
            {'name': 'normalize', 'mtype': 'qrofn', 'target_classes': ['Fuzznum']},
            {'name': 'normalize', 'is_default': True, 'target_classes': ['Fuzznum']}
        ])
        def qrofn_normalize(fuzz: Fuzznum) -> Fuzznum:
            # QROFN specific normalization logic
            # ...
            return fuzz
        ```

        Registering a function as both an instance method and a top-level function
        for a specific mtype:
        ```python
        # In fuzzlab/fuzzy/ivfns/_func.py
        from fuzzlab.extension import batch_extension
        from fuzzlab.core import Fuzznum

        @batch_extension([
            {'name': 'ivfn_transform', 'mtype': 'ivfn', 'injection_type': 'instance_method', 'target_classes': ['Fuzznum']},
            {'name': 'ivfn_transform', 'mtype': 'ivfn', 'injection_type': 'top_level_function'}
        ])
        def ivfn_transform_func(fuzz: Fuzznum, factor: float) -> Fuzznum:
            # IVFN specific transformation
            # ...
            return fuzz
        ```
    """
    def decorator(func):
        registry = get_extension_registry()
        for reg in registrations:
            registry.register(**reg)(func)
        return func
    return decorator
