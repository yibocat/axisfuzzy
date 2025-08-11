#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 17:31
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union, Tuple, Optional, List

import numpy as np

from ..core.t_fuzzarray import Fuzzarray
from ..core.fuzznums import Fuzznum


# reshape, flatten, squeeze, copy, ravel, transpose, broadcast_to, item,
# concat, stack, append, pop
# ========================= Core Structural Operations =========================

def _reshape_factory(obj: Union[Fuzznum, Fuzzarray], *shape: int) -> Fuzzarray:
    """
    Gives a new shape to an array without changing its data.
    Works with the SoA backend infrastructure.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    if isinstance(obj, Fuzznum):
        # Create a scalar Fuzzarray first, then reshape
        arr = Fuzzarray(data=obj, shape=())
        return _reshape_factory(arr, *shape)

    # Get component arrays from backend
    components = obj.backend.get_component_arrays()

    # Reshape each component array
    try:
        reshaped_components = [comp.reshape(shape) for comp in components]
    except ValueError as e:
        raise ValueError(f"Cannot reshape array of size {obj.size} into shape {shape}") from e

    # Create new backend from reshaped components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*reshaped_components, **obj._mtype_kwargs)

    return Fuzzarray(backend=new_backend)


def _flatten_factory(obj: Union[Fuzznum, Fuzzarray]) -> Fuzzarray:
    """
    Return a copy of the array collapsed into one dimension.
    """
    if isinstance(obj, Fuzznum):
        return Fuzzarray(data=obj, shape=(1,))

    # Get component arrays and flatten them
    components = obj.backend.get_component_arrays()
    flattened_components = [comp.flatten() for comp in components]

    # Create new backend from flattened components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*flattened_components, **obj._mtype_kwargs)

    return Fuzzarray(backend=new_backend)


def _squeeze_factory(obj: Union[Fuzznum, Fuzzarray],
                     axis: Union[int, Tuple[int, ...], None] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    Remove single-dimensional entries from the shape of an array.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()

    # Get component arrays
    components = obj.backend.get_component_arrays()
    squeezed_components = [comp.squeeze(axis=axis) for comp in components]

    # Check if result is scalar
    if squeezed_components[0].ndim == 0:
        # Return a Fuzznum for scalar result
        # We need to extract the scalar values and create a Fuzznum
        # This requires knowledge of the mtype, so we'll use the backend
        flattened_obj = _flatten_factory(obj)
        return flattened_obj[0]  # Get the first (and only) element
    else:
        # Create new backend from squeezed components
        backend_cls = obj.backend.__class__
        new_backend = backend_cls.from_arrays(*squeezed_components, **obj._mtype_kwargs)
        return Fuzzarray(backend=new_backend)


# TODO: 该方法主要用于实现顶层函数,因为其 copy 实例方法以实现
def _copy_factory(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Returns a deep copy of the fuzzy object.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()
    elif isinstance(obj, Fuzzarray):
        return obj.copy()
    else:
        raise TypeError(f"Unsupported type for copy: {type(obj)}")


def _ravel_factory(obj: Union[Fuzznum, Fuzzarray]) -> Fuzzarray:
    """
    Return a contiguous flattened array.
    """
    if isinstance(obj, Fuzznum):
        return Fuzzarray(data=obj, shape=(1,))

    # Ravel is essentially reshape(-1)
    return _reshape_factory(obj, -1)


def _transpose_factory(obj: Union[Fuzzarray, Fuzznum], *axes) -> Union[Fuzzarray, Fuzznum]:
    """
    Returns a view of the fuzzy object with axes transposed.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()

    if not isinstance(obj, Fuzzarray):
        raise TypeError(f"Unsupported type for transpose: {type(obj)}")

    # Handle different ways axes can be passed
    if len(axes) == 0:
        axes = None
    elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = axes[0]

    # Get component arrays and transpose them
    components = obj.backend.get_component_arrays()
    transposed_components = [comp.transpose(axes) for comp in components]

    # Create new backend from transposed components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*transposed_components, **obj._mtype_kwargs)

    return Fuzzarray(backend=new_backend)


def _broadcast_to_factory(obj: Union[Fuzzarray, Fuzznum], *shape: int) -> Fuzzarray:
    """
    Broadcasts the fuzzy object to a new shape.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target_shape = tuple(shape[0])
    else:
        target_shape = tuple(shape)

    if isinstance(obj, Fuzznum):
        # Create a scalar array first
        arr = Fuzzarray(data=obj, shape=())
        return _broadcast_to_factory(arr, target_shape)

    # Get component arrays and broadcast them
    components = obj.backend.get_component_arrays()

    try:
        broadcasted_components = [np.broadcast_to(comp, target_shape) for comp in components]
    except ValueError as e:
        raise ValueError(f"Cannot broadcast object with shape {obj.shape} to shape {target_shape}: {e}")

    # Create new backend from broadcasted components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*broadcasted_components, **obj._mtype_kwargs)
    return Fuzzarray(backend=new_backend)


def _item_factory(obj: Union[Fuzzarray, Fuzznum], *args) -> Fuzznum:
    """
    Returns the scalar item of the fuzzy object.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()

    if isinstance(obj, Fuzzarray):
        if obj.size == 1 and not args:
            # Use backend to get the single element
            return obj.backend.get_fuzznum_view(0 if obj.ndim == 1 else ())
        elif args:
            # Get element at specified index
            return obj.backend.get_fuzznum_view(args[0])
        else:
            raise ValueError("can only convert an array of size 1 to a Python scalar")

    raise TypeError(f"Unsupported type for item() method: {type(obj)}")


# ========================= Container Operations =========================

def _concat_factory(obj: Fuzzarray, *others: Fuzzarray, axis: int = 0) -> Fuzzarray:
    """
    Join a sequence of Fuzzarrays along an existing axis.
    """
    if not isinstance(obj, Fuzzarray):
        raise TypeError("concat: first argument must be Fuzzarray")

    all_arrays = [obj] + list(others)

    # Filter out empty arrays
    effective_arrays = [arr for arr in all_arrays if arr.size > 0]
    if not effective_arrays:
        # Return empty array with same mtype
        return Fuzzarray(data=None, shape=(0,), mtype=obj.mtype, **obj._mtype_kwargs)

    # Check compatibility
    ref_array = effective_arrays[0]
    for arr in effective_arrays:
        if not isinstance(arr, Fuzzarray):
            raise TypeError(f"concat: all arguments must be Fuzzarray, got {type(arr)}")
        if arr.mtype != ref_array.mtype or arr._mtype_kwargs != ref_array._mtype_kwargs:
            raise ValueError("concat: all Fuzzarrays must have the same mtype and parameters")

    # Get component arrays from all arrays
    component_lists = [[] for _ in ref_array.backend.get_component_arrays()]
    for arr in effective_arrays:
        components = arr.backend.get_component_arrays()
        for i, comp in enumerate(components):
            component_lists[i].append(comp)

    # Concatenate each component list
    new_components = [np.concatenate(comp_list, axis=axis) for comp_list in component_lists]

    # Create new backend
    backend_cls = ref_array.backend.__class__
    new_backend = backend_cls.from_arrays(*new_components, **ref_array._mtype_kwargs)

    return Fuzzarray(backend=new_backend)


def _stack_factory(obj: Fuzzarray, *others: Fuzzarray, axis: int = 0) -> Fuzzarray:
    """
    Stack Fuzzarrays along a new axis.
    """
    if not isinstance(obj, Fuzzarray):
        raise TypeError("stack: first argument must be Fuzzarray")

    all_arrays = [obj] + list(others)

    # Check compatibility
    ref_mtype = obj.mtype
    ref_kwargs = obj._mtype_kwargs
    ref_shape = obj.shape

    for arr in all_arrays:
        if not isinstance(arr, Fuzzarray):
            raise TypeError(f"stack: all arguments must be Fuzzarray, got {type(arr)}")
        if arr.mtype != ref_mtype or arr._mtype_kwargs != ref_kwargs:
            raise ValueError("stack: all Fuzzarrays must have the same mtype and parameters")
        if arr.shape != ref_shape:
            raise ValueError(f"stack: all Fuzzarrays must have the same shape, expected {ref_shape}, got {arr.shape}")

    # Get component arrays from all arrays
    component_lists = [[] for _ in obj.backend.get_component_arrays()]
    for arr in all_arrays:
        components = arr.backend.get_component_arrays()
        for i, comp in enumerate(components):
            component_lists[i].append(comp)

    # Stack each component list
    new_components = [np.stack(comp_list, axis=axis) for comp_list in component_lists]

    # Create new backend
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*new_components, **obj._mtype_kwargs)

    return Fuzzarray(backend=new_backend)


def _append_factory(obj: Union[Fuzznum, Fuzzarray],
                    item: Union[Fuzznum, Fuzzarray, List[Fuzznum]],
                    axis: Optional[int] = None,
                    inplace: bool = False) -> Union[Fuzzarray, None]:
    """
    Append elements to the fuzzy object.
    """
    # Handle Fuzznum case
    if isinstance(obj, Fuzznum):
        elements = [obj]
        if isinstance(item, Fuzznum):
            elements.append(item)
        elif isinstance(item, Fuzzarray):
            # Convert Fuzzarray to list of Fuzznums
            flat_arr = _flatten_factory(item)
            for i in range(flat_arr.size):
                elements.append(flat_arr[i])
        elif isinstance(item, list):
            elements.extend(item)
        else:
            raise TypeError(f"append: unsupported item type {type(item)}")

        # Check compatibility
        mtype = obj.mtype
        mtype_kwargs = getattr(obj, '_mtype_kwargs', {})
        for fn in elements:
            if not isinstance(fn, Fuzznum):
                raise TypeError(f"append: all elements must be Fuzznum, got {type(fn)}")
            if fn.mtype != mtype:
                raise ValueError("append: all Fuzznums must have the same mtype")

        return Fuzzarray(data=elements, mtype=mtype, **mtype_kwargs)

    # Handle Fuzzarray case
    if inplace and axis is not None:
        raise ValueError("append: inplace=True is not supported with axis specified")

    # Convert item to appropriate format for concatenation
    if isinstance(item, Fuzznum):
        if item.mtype != obj.mtype:
            raise ValueError("append: mtype mismatch")
        item_array = Fuzzarray(data=item, shape=(1,))
    elif isinstance(item, Fuzzarray):
        if item.mtype != obj.mtype:
            raise ValueError("append: mtype mismatch")
        item_array = item
    elif isinstance(item, list):
        item_array = Fuzzarray(data=item, mtype=obj.mtype, **obj._mtype_kwargs)
    else:
        raise TypeError(f"append: unsupported item type {type(item)}")

    # Perform concatenation
    if axis is None:
        # Flatten both arrays before concatenating
        obj_flat = _flatten_factory(obj)
        item_flat = _flatten_factory(item_array)
        result = _concat_factory(obj_flat, item_flat, axis=0)
    else:
        result = _concat_factory(obj, item_array, axis=axis)

    if inplace:
        # Replace obj's backend with result's backend
        obj._backend = result._backend
        return None
    else:
        return result


def _pop_factory(obj: Union[Fuzznum, Fuzzarray],
                 index: int = -1,
                 inplace: bool = False) -> Union[Fuzznum, Tuple[Fuzznum, Fuzzarray], None]:
    """
    Remove and return an element from a one-dimensional array.
    """
    if isinstance(obj, Fuzznum):
        raise TypeError("pop: Fuzznum object does not support pop operation")

    if obj.ndim != 1:
        raise ValueError("pop: only one-dimensional Fuzzarray is supported")

    if obj.size == 0:
        raise IndexError("pop from empty Fuzzarray")

    # Get the element to pop
    popped_item = obj[index]

    # Create new array without the popped element
    indices = list(range(obj.size))
    if index < 0:
        index = obj.size + index
    indices.pop(index)

    if len(indices) == 0:
        # Result is empty array
        new_array = Fuzzarray(data=None, shape=(0,), mtype=obj.mtype, **obj._mtype_kwargs)
    else:
        # Create new array with remaining elements
        components = obj.backend.get_component_arrays()
        new_components = [comp[indices] for comp in components]

        backend_cls = obj.backend.__class__
        new_backend = backend_cls.from_arrays(*new_components, **obj._mtype_kwargs)
        new_array = Fuzzarray(backend=new_backend)

    if inplace:
        obj._backend = new_array._backend
        return popped_item
    else:
        return popped_item, new_array
