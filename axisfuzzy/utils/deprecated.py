#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


import warnings


def deprecated(func):
    """
    Decorator to mark a function or method as deprecated.

    When applied, this decorator will emit a warning each time the decorated
    function is called, indicating that the API is deprecated and may be removed
    in future versions. This is useful for informing users about features that
    are no longer recommended for use.

    Args:
        func (callable): The function or method to be marked as deprecated.

    Returns:
        callable: The wrapped function that issues a warning on each call.
    """
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"'{func.__name__}' is deprecated and may be removed in future versions.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)

    return wrapper
