#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import functools
import types
import warnings


# def experimental(func):
#     """
#     Decorator to mark a function or method as experimental.
#
#     When applied, this decorator will emit a warning each time the decorated
#     function is called, indicating that the API is experimental and may change
#     in future versions. This is useful for highlighting features that are under
#     development, not fully tested, or subject to change, so that users are
#     aware of potential instability.
#
#     Args:
#         func (callable): The function or method to be marked as experimental.
#
#     Returns:
#         callable: The wrapped function that issues a warning on each call.
#
#     Example:
#         >>> from axisfuzzy.utils.experimental import experimental
#         >>>
#         >>> @experimental
#         ... def my_experimental_feature(x):
#         ...     return x * 2
#         >>>
#         >>> my_experimental_feature(3)
#         ... # UserWarning: my_experimental_feature is experimental and may change in future versions.
#         6
#     """
#     def wrapper(*args, **kwargs):
#         warnings.warn(
#             f"{func.__name__} is experimental and may change in future versions.",
#             UserWarning,
#             stacklevel=2
#         )
#         return func(*args, **kwargs)
#     return wrapper

def experimental(obj):
    if isinstance(obj, type):
        orig_init = obj.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(f"Warning: {obj.__name__} (class) "
                          f"is experimental and may change in future versions.")
            orig_init(self, *args, **kwargs)

        obj.__init__ = new_init
        return obj
    elif isinstance(obj, (types.FunctionType, types.MethodType)):
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(f"Warning: {obj.__name__} "
                          f"is experimental and may change in future versions.")
            return obj(*args, **kwargs)

        return wrapper
    else:
        return obj
