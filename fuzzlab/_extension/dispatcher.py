#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/6 19:03
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module provides factory functions for creating dispatcher wrappers.

These dispatchers act as intermediaries, dynamically routing function calls
to the correct `mtype`-specific implementation registered in the
`ExtendFunctionRegistry`. They are designed to be injected as instance methods
or top-level functions, providing a unified API for users while handling
the underlying type-based dispatch logic.
"""
from typing import Callable, Union, Any, Optional

from .registry import get_extend_registry
from ..config import get_config
from ..core import Fuzzarray, Fuzznum


def create_method_dispatcher(name: str) -> Callable:
    """
    Creates a dispatcher function intended to be injected as an instance method.

    This dispatcher will be bound to a `Fuzznum` or `Fuzzarray` instance.
    When called, it retrieves the `mtype` from the instance (`self`) and
    uses the `ExtendFunctionRegistry` to find and execute the appropriate
    `mtype`-specific implementation.

    Args:
        name (str): The name of the function to dispatch (e.g., 'distance', 'normalize').

    Returns:
        Callable: A dispatcher function that takes `self` (the instance)
            and any additional arguments, then dispatches to the correct
            implementation.

    Raises:
        NotImplementedError: If no implementation (neither mtype-specific nor default)
            is found for the given `name` and `mtype`.

    Examples:
        >>> # This dispatcher would be injected as Fuzznum.distance
        >>> # When called as `my_fuzznum.distance(other_fuzznum)`:
        >>> # 1. It gets `my_fuzznum.mtype`.
        >>> # 2. It queries the registry for 'distance' with that mtype.
        >>> # 3. It calls the found implementation, passing `my_fuzznum` as the first argument.
    """
    def dispatcher(self: Union[Fuzznum, Fuzzarray], *args, **kwargs) -> Any:
        """
        The actual dispatcher function that gets injected as an instance method.
        """
        registry = get_extend_registry()
        # Retrieve the mtype from the instance itself
        mtype = self.mtype
        # Get the appropriate implementation from the registry
        implementation = registry.get_implementation(name, mtype)
        if implementation:
            # Call the found implementation, passing the instance (self) as the first argument
            return implementation(self, *args, **kwargs)
        # If no implementation is found, raise an error
        raise NotImplementedError(f"Function '{name}' is not implemented for mtype '{mtype}'.")
    return dispatcher


def create_top_level_dispatcher(name: str) -> Callable:
    """
    Creates a dispatcher function intended to be injected as a top-level function
    in the `fuzzlab` module (e.g., `fuzzlab.distance()`, `fuzzlab.random()`).

    This dispatcher determines the `mtype` from the first argument (if it's a
    Fuzznum/Fuzzarray instance) or from a keyword argument (`mtype`). It then
    uses the `ExtendFunctionRegistry` to find and execute the appropriate
    `mtype`-specific implementation.

    Args:
        name (str): The name of the function to dispatch (e.g., 'random', 'distance').

    Returns:
        Callable: A dispatcher function that takes an object (or mtype) and
            additional arguments, then dispatches to the correct implementation.

    Raises:
        TypeError: If the `mtype` cannot be determined from the arguments.
        NotImplementedError: If no implementation (neither mtype-specific nor default)
            is found for the given `name` and determined `mtype`.

    Examples:
        >>> # This dispatcher would be injected as fuzzlab.random
        >>> # When called as `fuzzlab.random(mtype='qrofn', q=2)`:
        >>> # 1. It gets 'qrofn' from kwargs.
        >>> # 2. It queries the registry for 'random' with 'qrofn'.
        >>> # 3. It calls the found implementation, passing 'qrofn' and other kwargs.
        >>>
        >>> # When called as `fuzzlab.distance(fuzz1, fuzz2)` where fuzz1 is a Fuzznum:
        >>> # 1. It gets `fuzz1.mtype`.
        >>> # 2. It queries the registry for 'distance' with that mtype.
        >>> # 3. It calls the found implementation, passing fuzz1, fuzz2 and other args.
    """
    def dispatcher(*args, **kwargs) -> Any:
        """
        The actual dispatcher function that gets injected as a top-level function.
        """
        registry = get_extend_registry()
        config = get_config()
        mtype = None

        mtype_from_obj_str = False

        if 'mtype' in kwargs:
            mtype = kwargs['mtype']
        elif args and isinstance(args[0], (Fuzznum, Fuzzarray)):
            mtype = args[0].mtype
        elif args and isinstance(args[0], str):
            mtype = args[0]
            mtype_from_obj_str = True

        if mtype is None:
            mtype = config.DEFAULT_MTYPE

        implementation = registry.get_implementation(name, mtype)
        if implementation:
            if mtype_from_obj_str:
                return implementation(*args[1:], **kwargs)

            return implementation(*args, **kwargs)
        raise NotImplementedError(f"Function '{name}' is not implemented for mtype '{mtype}'.")
    return dispatcher
