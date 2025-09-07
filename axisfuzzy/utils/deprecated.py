#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


import warnings


def deprecated(func=None, *, message=None):
    """
    Decorator to mark a function or method as deprecated.

    When applied, this decorator will emit a warning each time the decorated
    function is called, indicating that the API is deprecated and may be removed
    in future versions. This is useful for informing users about features that
    are no longer recommended for use.

    Parameters
    ----------
    func : callable, optional
        The function or method to be marked as deprecated.
    message : str, optional
        Custom message to append to the deprecation warning.
        If provided, the warning will be:
        "'{func.__name__}' will be deprecated and may be removed in future versions. {message}"

    Returns
    -------
    callable
        The wrapped function that issues a warning on each call.

    Examples
    --------
    .. code-block:: python

        @deprecated
        def old_function():
            pass

        @deprecated(message="Please use new_function instead.")
        def another_old_function():
            pass
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            base_message = f"'{f.__name__}' will be deprecated and may be removed in future versions."
            if message:
                full_message = f"{base_message} {message}"
            else:
                full_message = base_message
            
            warnings.warn(
                full_message,
                DeprecationWarning,
                stacklevel=2
            )
            return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        # Called with arguments: @deprecated(message="...")
        return decorator
    else:
        # Called without arguments: @deprecated
        return decorator(func)
