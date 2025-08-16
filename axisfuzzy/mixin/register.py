#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 20:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
Registration of structural operations for the new SoA-based Fuzzarray.
"""
from .registry import register_mixin
from .factory import (
    _reshape_factory, _flatten_factory, _squeeze_factory, _copy_factory,
    _ravel_factory, _transpose_factory, _broadcast_to_factory, _item_factory,
    _concat_factory, _stack_factory, _append_factory, _pop_factory, _any_factory, _all_factory
)


@register_mixin(name='reshape', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _reshape_impl(self, *shape: int):
    """Gives a new shape to an array without changing its data."""
    return _reshape_factory(self, *shape)


@register_mixin(name='flatten', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _flatten_impl(self):
    """Return a copy of the array collapsed into one dimension."""
    return _flatten_factory(self)


@register_mixin(name='squeeze', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _squeeze_impl(self, axis=None):
    """Remove single-dimensional entries from the shape of an array."""
    return _squeeze_factory(self, axis)


@register_mixin(name='copy', injection_type='top_level_function')
def _copy_impl(obj):
    """Returns a deep copy of the fuzzy object."""
    return _copy_factory(obj)


@register_mixin(name='ravel', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _ravel_impl(self):
    """Return a contiguous flattened array."""
    return _ravel_factory(self)


@register_mixin(name='transpose', injection_type='top_level_function')
def _transpose_impl(obj, *axes):
    """Returns a view of the fuzzy object with axes transposed."""
    return _transpose_factory(obj, *axes)


@register_mixin(name='T', target_classes=["Fuzzarray", "Fuzznum"], injection_type='instance_function')
@property
def _T_impl(self):
    """Returns a view of the fuzzy array with axes transposed."""
    return _transpose_factory(self)


@register_mixin(name='broadcast_to', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _broadcast_to_impl(self, *shape):
    """Broadcasts the fuzzy object to a new shape."""
    return _broadcast_to_factory(self, *shape)


@register_mixin(name='item', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _item_impl(self, *args):
    """Returns the scalar item of the fuzzy object."""
    return _item_factory(self, *args)


@register_mixin(name='any', target_classes=["Fuzzarray"], injection_type='both')
def _any_impl(self):
    """Returns True if any element of the Fuzzarray is True."""
    return _any_factory(self)


@register_mixin(name='all', target_classes=["Fuzzarray"], injection_type='both')
def _all_impl(self):
    """Returns True if all elements of the Fuzzarray are True."""
    return _all_factory(self)


@register_mixin(name='concat', target_classes=['Fuzzarray'], injection_type='both')
def _concat_impl(self, *others, axis: int = 0):
    """Join a sequence of Fuzzarrays along an existing axis."""
    return _concat_factory(self, *others, axis=axis)


@register_mixin(name='stack', target_classes=['Fuzzarray'], injection_type='both')
def _stack_impl(self, *others, axis: int = 0):
    """Stack Fuzzarrays along a new axis."""
    return _stack_factory(self, *others, axis=axis)


@register_mixin(name='append', target_classes=["Fuzznum", "Fuzzarray"], injection_type='both')
def _append_impl(self, item, axis=None, inplace=False):
    """Append elements to the fuzzy object."""
    return _append_factory(self, item, axis, inplace)


@register_mixin(name='pop', target_classes=["Fuzznum", "Fuzzarray"], injection_type='both')
def _pop_impl(self, index=-1, inplace=False):
    """Remove and return an element from a one-dimensional array."""
    return _pop_factory(self, index, inplace)
