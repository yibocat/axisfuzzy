#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 19:31
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
Utility functions for FuzzLab's extension system.

This module provides helper functions that facilitate the interaction
with the extension registry, particularly for internal calls between
extension functions. The primary utility is `call_extension`, which
allows one extension to invoke another by name and mtype, bypassing
the injection mechanism.
"""
from typing import Union

from .registry import get_extension_registry
from ..core import Fuzznum, Fuzzarray


def call_extension(func_name: str,
                   obj: Union[Fuzznum, Fuzzarray, None],
                   *args,
                   **kwargs):
    """
    Directly calls an extension function by looking it up in the registry.

    This utility function is crucial for scenarios where one extension
    needs to call another, especially before the full injection process
    (via `fuzzlab.extension.injector.py`) has completed. It bypasses
    the dynamically injected methods/functions and directly retrieves
    the underlying implementation from the `ExtensionRegistry`.

    Args:
        func_name: The name of the extension function to call (e.g., 'distance', 'add').
        obj: The `Fuzznum` or `Fuzzarray` instance on which the extension
            function is intended to operate. Its `mtype` attribute is used
            to find the correct specialized or default implementation.
        *args: Positional arguments to be passed to the target extension function.
        **kwargs: Keyword arguments to be passed to the target extension function.

    Returns:
        The result of the called extension function.

    Raises:
        AttributeError: If the provided `obj` does not have an 'mtype' attribute.
        NotImplementedError: If no implementation for the given `func_name` and
                             `mtype` (or a suitable default) is found in the registry.

    Examples:
        Calling a 'multiply' extension from within a 'power' extension:
        ```python
        # In fuzzlab/fuzzy/qrofs/_extend.py (example)
        from fuzzlab.extension import extension
        from fuzzlab.extension.utils import call_extension
        from fuzzlab.core import Fuzznum

        @extension(name='multiply', mtype='qrofn', target_classes=['Fuzznum'])
        def qrofn_multiply(fuzz1: Fuzznum, fuzz2: Fuzznum) -> Fuzznum:
            # ... QROFN specific multiplication logic ...
            return Fuzznum(md=fuzz1.md * fuzz2.md, nmd=fuzz1.nmd + fuzz2.nmd - fuzz1.nmd * fuzz2.nmd, q=fuzz1.q, mtype='qrofn')

        @extension(name='power', mtype='qrofn', target_classes=['Fuzznum'])
        def qrofn_power(fuzz: Fuzznum, exponent: int) -> Fuzznum:
            if exponent == 0:
                return Fuzznum(md=1.0, nmd=0.0, q=fuzz.q, mtype='qrofn') # Identity for multiplication
            result = fuzz
            for _ in range(1, exponent):
                # Use call_extension to safely invoke 'multiply' for the same mtype
                result = call_extension('multiply', result, fuzz)
            return result
        ```
    """
    registry = get_extension_registry()

    if obj is not None:
        mtype = getattr(obj, 'mtype', None)
        if mtype is None:
            raise AttributeError(f"Object '{type(obj).__name__}' has no 'mtype' attribute")
    else:
        mtype = kwargs.pop('mtype', None)
        if mtype is None:
            raise AttributeError("No 'mtype' provided and 'obj' is None. Cannot determine mtype for extension call.")

    implementation = registry.get_function(func_name, mtype)

    if implementation is None:
        raise NotImplementedError(f"Extension function '{func_name}' not implemented for mtype '{mtype}'")

    if obj is not None:
        return implementation(obj, *args, **kwargs)
    else:
        return implementation(*args, **kwargs)
