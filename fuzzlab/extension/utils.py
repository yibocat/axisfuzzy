#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 19:31
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union

from .registry import get_extension_registry
from ..core import Fuzznum, Fuzzarray


def call_extension(func_name: str,
                   obj: Union[Fuzznum, Fuzzarray],
                   *args,
                   **kwargs):
    """
    Directly calls an extension function by looking it up in the registry.

    This is useful when one extension needs to call another before the injection
    process is complete.

    Args:
        func_name: The name of the extension function to call (e.g., 'distance').
        obj: The Fuzznum or Fuzzarray instance on which to operate.
        *args: Positional arguments for the extension function.
        **kwargs: Keyword arguments for the extension function.

    Returns:
        The result of the extension function call.
    """
    registry = get_extension_registry()
    mtype = getattr(obj, 'mtype', None)
    if mtype is None:
        raise AttributeError(f"Object '{type(obj).__name__}' has no 'mtype' attribute")

    implementation = registry.get_function(func_name, mtype)

    if implementation is None:
        raise NotImplementedError(f"Extension function '{func_name}' not implemented for mtype '{mtype}'")

    return implementation(obj, *args, **kwargs)
