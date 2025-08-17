#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import copy
from typing import Union, Tuple, List

import numpy as np

from .registry import get_registry_mixin
from ..core import Fuzznum, Fuzzarray

mixin = get_registry_mixin()


@mixin.register(name='reshape', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _reshape_impl(self: Union['Fuzzarray', 'Fuzznum'],
                  *shape: int) -> 'Fuzzarray':
    """
    Gives a new shape to an array without changing its data.

    This method follows the behavior of `numpy.ndarray.reshape`.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to reshape.
        *shape (int or tuple of ints): The new shape should be compatible with the
                                       original shape. If an integer, then the
                                       result will be a 1-D array of that length.
                                       One shape dimension can be -1, in which case
                                       the value is inferred from the length of
                                       the array and remaining dimensions.

    Returns:
        Fuzzarray: A new Fuzzarray with the specified shape. It will be a view
                   of the original array if possible, otherwise a copy.

    Raises:
        ValueError: If the new shape is not compatible with the original shape.
    """
    # Handle the case where shape is passed as a single tuple, e.g., reshape((2, 3))
    # which is a common numpy pattern.
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    # If self is a Fuzznum, treat it as a 0-dim array for reshaping.
    if isinstance(self, Fuzznum):
        # np.array(self) creates a 0-dim array containing the Fuzznum object.
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    # Use numpy's reshape and wrap the result in a new Fuzzarray.
    # numpy.reshape returns a view if possible, otherwise a copy.
    reshaped_data = data_array.reshape(shape)
    return Fuzzarray(reshaped_data, copy=False)


@mixin.register(name='flatten', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _flatten_impl(self: Union['Fuzzarray', 'Fuzznum']) -> 'Fuzzarray':
    """
    Return a copy of the array collapsed into one dimension.

    It will return a copy of the flattened one-dimensional `Fuzzarray`.
    For a `Fuzznum`, it will return a one-dimensional `Fuzzarray` containing that element.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to flatten.

    Returns:
        Fuzzarray: A new 1-D Fuzzarray containing a copy of the elements.
    """
    # If self is a Fuzznum, treat it as a 0-dim array.
    if isinstance(self, Fuzznum):
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    # numpy.flatten always returns a new 1-D copy of the array.
    flattened_data = data_array.flatten()
    # Wrap the flattened copy in a new Fuzzarray.
    return Fuzzarray(flattened_data, copy=False)


@mixin.register(name='squeeze', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _squeeze_impl(self: Union['Fuzzarray', 'Fuzznum'],
                  axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Remove single-dimensional entries from the shape of an array.

    Entries that remove one-dimensional elements from the array's shape.
    If the result is a 0-dimensional array, it will return a Fuzznum scalar;
    otherwise, it will return a new Fuzzarray.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to squeeze.
        axis (None or int or tuple of ints, optional): Selects a subset of the
            single-dimensional entries to remove. If an axis is selected with
            a shape entry greater than one, a ValueError is raised.

    Returns:
        Union['Fuzznum', 'Fuzzarray']: The squeezed fuzzy object. If the array
                                       becomes a 0-d array, a Fuzznum is returned.
                                       Otherwise, a Fuzzarray is returned (as a view).
    """
    # If self is a Fuzznum, it's already a scalar (squeezed). Return a copy.
    if isinstance(self, Fuzznum):
        return self.copy()

    # For a Fuzzarray, use the underlying numpy array's squeeze method.
    # This returns a view of the original data.
    squeezed_data = self._data.squeeze(axis=axis)

    # If the result is a 0-d array, it contains a single Fuzznum.
    # Extract and return the Fuzznum object itself.
    if squeezed_data.ndim == 0:
        return squeezed_data.item()
    else:
        # Otherwise, return a new Fuzzarray wrapping the squeezed data view.
        return Fuzzarray(squeezed_data, copy=False)


@mixin.register(name='copy', injection_type='top_level_function')
def _copy_top_level_impl(obj: Union['Fuzzarray', 'Fuzznum']) -> Union['Fuzzarray', 'Fuzznum']:
    """
    Returns a deep copy of the fuzzy object.

    This function creates a new object that is a complete and independent copy
    of the original. Any changes to the new object will not affect the
    original object.

    Args:
        obj (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to copy.

    Returns:
        Union['Fuzzarray', 'Fuzznum']: A new object that is a deep copy of the original.

    Raises:
        TypeError: If the input object is not a Fuzzarray or Fuzznum.
    """
    if isinstance(obj, (Fuzzarray, Fuzznum)):
        # Delegate the actual copying to the object's own copy method.
        # This assumes Fuzzarray and Fuzznum both have a .copy() method
        # that performs a deep copy suitable for their internal structure.
        return obj.copy()
    else:
        raise TypeError(f"Unsupported type for copy: {type(obj)}. Expected Fuzzarray or Fuzznum.")


@mixin.register(name='ravel', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _ravel_impl(self: Union['Fuzzarray', 'Fuzznum']) -> 'Fuzzarray':
    """
    Return a contiguous flattened array.

    A 1-D array, containing the elements of the input, is returned.
    A view is returned if possible; otherwise, a copy is made. This is
    the primary difference from `flatten`, which always returns a copy.

    For a `Fuzznum`, this returns a 1-D `Fuzzarray` of size 1 containing
    the number.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to ravel.

    Returns:
        Fuzzarray: A new 1-D Fuzzarray. It will be a view of the original
                   array if possible, otherwise a copy.
    """
    # If self is a Fuzznum, treat it as a 0-dim array.
    if isinstance(self, Fuzznum):
        # np.array(self) creates a 0-dim array containing the Fuzznum object.
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    # Use numpy.ravel on the data array. This returns a view if possible.
    raveled_data = np.ravel(data_array)

    # Wrap the result in a new Fuzzarray, preserving the view semantics
    # by setting copy=False.
    return Fuzzarray(raveled_data, copy=False)


@mixin.register(name='transpose', injection_type='top_level_function')
def _transpose_impl(obj: Union['Fuzzarray', 'Fuzznum']) -> Union['Fuzzarray', 'Fuzznum']:
    """
    Returns a view of the fuzzy object with axes transposed.

    For a Fuzzarray, this is equivalent to `obj.T`.
    For a Fuzznum, it returns the Fuzznum itself.

    Args:
        obj (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to transpose.

    Returns:
        Union['Fuzzarray', 'Fuzznum']: The transposed fuzzy object.
    """
    if isinstance(obj, Fuzznum):
        return copy.deepcopy(obj)
    elif isinstance(obj, Fuzzarray):
        return Fuzzarray(obj.data.T)
    else:
        raise TypeError(f"Unsupported type for transpose: {type(obj)}")


@mixin.register(name='broadcast_to', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _broadcast_to_impl(self: Union['Fuzzarray', 'Fuzznum'],
                       *shape: int) -> 'Fuzzarray':
    """
    Broadcasts the fuzzy object to a new shape.

    This method mimics the behavior of `numpy.broadcast_to`.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to broadcast.
        *shape (int or tuple of ints): The shape to which the object should be broadcast.
                                       If a single tuple is passed, it should be unpacked.

    Returns:
        Fuzzarray: A new Fuzzarray that is a broadcasted view of the original object.

    Raises:
        ValueError: If the object cannot be broadcast to the specified shape.
    """
    # Handle the case where shape is passed as a single tuple, e.g., broadcast_to((2, 3))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target_shape = tuple(shape[0])
    else:
        target_shape = tuple(shape)

    # If self is a Fuzznum, treat it as a 0-dim array for broadcasting.
    if isinstance(self, Fuzznum):
        # np.array(self) creates a 0-dim array containing the Fuzznum object.
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    try:
        # Use numpy's broadcast_to and wrap the result in a new Fuzzarray.
        # np.broadcast_to returns a read-only view.
        broadcasted_data = np.broadcast_to(data_array, target_shape)
        return Fuzzarray(broadcasted_data, copy=False)
    except ValueError as e:
        raise ValueError(f"Cannot broadcast object with shape {self.shape} to shape {target_shape}: {e}")


@mixin.register(name='item', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _item_impl(self: Union['Fuzzarray', 'Fuzznum']) -> 'Fuzznum':
    """
    Returns the scalar item of the fuzzy object.

    If the object is a Fuzznum, it returns a copy of itself.
    If the object is a Fuzzarray, it must contain exactly one element,
    which is then returned as a Fuzznum.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.

    Returns:
        Fuzznum: The single Fuzznum element.

    Raises:
        ValueError: If the Fuzzarray contains more than one element.
    """
    if isinstance(self, Fuzznum):
        # Fuzznum is already a scalar, return a copy of itself
        return self.copy()
    elif isinstance(self, Fuzzarray):
        if self.size == 1:
            # If Fuzzarray has only one element, return that Fuzznum
            # self._data is a numpy array, so self._data.item() will extract the single Fuzznum object
            return self._data.item()
        else:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
    else:
        # This case should ideally not be reached due to type hints,
        # but added for robustness.
        raise TypeError(f"Unsupported type for item() method: {type(self)}")


@mixin.register(name='sort', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _sort_impl(self: Union['Fuzzarray', 'Fuzznum'],
               axis: int = -1) -> Union['Fuzzarray', 'Fuzznum']:
    """
    Return a sorted copy of a fuzzy array.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to sort.
        axis (int, optional): Axis along which to sort. If None, the array is
                              flattened before sorting. The default is -1,
                              which sorts along the last axis.

    Returns:
        Fuzzarray: A sorted copy of the array.
    """
    if isinstance(self, Fuzznum):
        return self.copy()

    sorted_data = np.sort(self._data, axis=axis)
    return Fuzzarray(sorted_data, copy=False)


@mixin.register(name='argsort', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _argsort_impl(self: Union['Fuzzarray', 'Fuzznum'],
                  axis: int = -1) -> np.ndarray:
    """
    Returns the indices that would sort a fuzzy array.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.
        axis (int, optional): Axis along which to sort. Default is -1.

    Returns:
        np.ndarray: An array of indices that sort the array.
    """
    if isinstance(self, Fuzznum):
        return np.array(0)  # Index of a scalar is 0

    return np.argsort(self._data, axis=axis)


@mixin.register(name='argmax', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _argmax_impl(self: Union['Fuzzarray', 'Fuzznum'],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[np.integer, int, np.ndarray]:
    """
    Return indices of the maximum values along an axis.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes along which to operate.
                                                       By default, flattened input is used.

    Returns:
        np.ndarray: Array of indices into the array.
    """
    if isinstance(self, Fuzznum):
        return np.array(0).item()

    result = np.argmax(self._data, axis=axis)
    if isinstance(result, np.ndarray):
        return result.tolist()
    else:
        return result.item()


@mixin.register(name='argmin', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _argmin_impl(self: Union['Fuzzarray', 'Fuzznum'],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[np.integer, int, np.ndarray]:
    """
    Return indices of the minimum values along an axis.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes along which to operate.
                                                       By default, flattened input is used.

    Returns:
        np.ndarray: Array of indices into the array.
    """
    if isinstance(self, Fuzznum):
        return np.array(0).item()

    result = np.argmin(self._data, axis=axis)
    if isinstance(result, np.ndarray):
        return result.tolist()
    else:
        return result.item()


@mixin.register(name='concat', target_classes=['Fuzzarray'], injection_type='both')
def _concat_impl(self: 'Fuzzarray',
                 *others: 'Fuzzarray',
                 axis: int = 0) -> 'Fuzzarray':
    """
    将一个或多个 Fuzzarray 沿指定轴拼接到 self 之后。

    此方法将 `self` 作为第一个数组，后跟 `others` 中的所有数组，
    然后将它们全部拼接成一个新的 Fuzzarray。

    规则:
      - 所有参与数组的 mtype 和 q 必须与 self 一致。
      - 除拼接轴（axis）外，所有维度必须匹配。

    Args:
        self (Fuzzarray): 第一个数组。
        *others (Fuzzarray): 要拼接到后面的一个或多个 Fuzzarray。
        axis (int): 拼接所沿的轴，默认为 0。

    Returns:
        Fuzzarray: 一个包含所有拼接后元素的新 Fuzzarray。
    """
    if not isinstance(self, Fuzzarray):
        raise TypeError("concat: self 必须是 Fuzzarray")

    # 将 self 和 others 组合成一个待处理的数组列表
    all_arrays = [self] + list(others)

    # 过滤掉空数组，但至少保留 self（如果 self 非空）
    effective_arrays = [arr for arr in all_arrays if arr.size > 0]
    if not effective_arrays:
        return Fuzzarray(np.array([], dtype=object), mtype=self.mtype)  # 如果全部为空，返回空数组

    # 使用第一个有效数组作为引用进行检查
    ref_array = effective_arrays[0]
    base_mtype = ref_array.mtype
    base_q = getattr(ref_array, 'q', None)

    # 检查类型、mtype、q 和形状兼容性
    for arr in effective_arrays:
        if not isinstance(arr, Fuzzarray):
            raise TypeError(f"concat: 只支持 Fuzzarray，但收到了 {type(arr)}")
        if arr.mtype != base_mtype or getattr(arr, 'q', None) != base_q:
            raise ValueError("concat: 所有 Fuzzarray 的 mtype 和 q 必须一致。")
        if arr.ndim != ref_array.ndim:
            raise ValueError("concat: 所有 Fuzzarray 的维度必须一致。")
        for i in range(arr.ndim):
            if i != axis and arr.shape[i] != ref_array.shape[i]:
                raise ValueError(
                    f"concat: 形状在非拼接轴上不兼容。期望在轴 {i} 上大小为 {ref_array.shape[i]}，但收到了 {arr.shape[i]}")

    # 提取数据并执行拼接
    data_to_concat = [arr.data for arr in effective_arrays]
    new_data = np.concatenate(data_to_concat, axis=axis)

    return Fuzzarray(new_data, mtype=base_mtype, copy=False)


@mixin.register(name='stack', target_classes=['Fuzzarray'], injection_type='both')
def _stack_impl(self: 'Fuzzarray',
                *others: 'Fuzzarray',
                axis: int = 0) -> 'Fuzzarray':
    """
    将 self 和一个或多个其他 Fuzzarray 沿新轴堆叠。

    此方法将 `self` 作为第一个数组，后跟 `others` 中的所有数组，
    然后将它们全部堆叠成一个新的、更高维度的 Fuzzarray。

    规则:
      - 所有参与数组的 mtype、q 和 shape 必须完全一致。

    Args:
        self (Fuzzarray): 第一个数组。
        *others (Fuzzarray): 要一起堆叠的一个或多个 Fuzzarray。
        axis (int): 新轴插入的位置，默认为 0。

    Returns:
        Fuzzarray: 一个新的、维度增加的 Fuzzarray。
    """
    if not isinstance(self, Fuzzarray):
        raise TypeError("stack: self 必须是 Fuzzarray")

    all_arrays = [self] + list(others)

    # 检查类型、mtype、q 和形状一致性
    base_mtype = self.mtype
    base_q = getattr(self, 'q', None)
    ref_shape = self.shape

    for arr in all_arrays:
        if not isinstance(arr, Fuzzarray):
            raise TypeError(f"stack: 只支持 Fuzzarray，但收到了 {type(arr)}")
        if arr.mtype != base_mtype or getattr(arr, 'q', None) != base_q:
            raise ValueError("stack: 所有 Fuzzarray 的 mtype 和 q 必须一致。")
        if arr.shape != ref_shape:
            raise ValueError(f"stack: 所有 Fuzzarray 的形状必须一致。期望 {ref_shape}，但收到了 {arr.shape}")

    # 提取数据并执行堆叠
    data_to_stack = [arr.data for arr in all_arrays]
    new_data = np.stack(data_to_stack, axis=axis)

    return Fuzzarray(new_data, mtype=base_mtype, copy=False)


@mixin.register(name='append', target_classes=["Fuzznum", "Fuzzarray"], injection_type='both')
def _append_impl(self: Union['Fuzznum', 'Fuzzarray'],
                 item: Union['Fuzznum', 'Fuzzarray', List['Fuzznum']],
                 axis: int = None,
                 inplace: bool = False) -> Union['Fuzzarray', None]:
    """
    向对象追加元素。

    - 对于 Fuzznum:
      总是返回一个新的 Fuzzarray，包含原 Fuzznum 和新元素。
      忽略 `axis` 和 `inplace` 参数。

    - 对于 Fuzzarray:
      沿指定轴追加元素。
      如果 `axis` 为 None，则在追加前将数组扁平化。
      如果 `inplace=True`，则原地修改并返回 None；否则返回新的 Fuzzarray。

    Args:
        self: Fuzznum 或 Fuzzarray 对象。
        item: 要追加的 Fuzznum、Fuzzarray 或 Fuzznum 列表。
        axis (int, optional): 追加操作的轴。如果为 None，则数组被展平。
        inplace (bool): 如果为 True，则原地修改数组。仅对 Fuzzarray 有效。

    Returns:
        Fuzzarray 或 None: 根据 inplace 的值返回新数组或 None。

    Raises:
        ValueError: 如果 mtype、q 或形状不兼容。
        TypeError: 如果 item 类型不支持。
    """
    # --- Fuzznum 的 append 逻辑 ---
    if isinstance(self, Fuzznum):
        elements = [self]
        if isinstance(item, Fuzznum):
            elements.append(item)
        elif isinstance(item, Fuzzarray):
            elements.extend(list(item.flatten().data))
        elif isinstance(item, list):
            elements.extend(item)
        else:
            raise TypeError(f"append: 不支持的 item 类型 {type(item)}")

        # 检查 mtype 和 q 是否一致
        mtype = self.mtype
        q = getattr(self, 'q', None)
        for fn in elements:
            if not isinstance(fn, Fuzznum):
                raise TypeError(f"append: item 列表必须只包含 Fuzznum，但发现了 {type(fn)}")
            if fn.mtype != mtype or getattr(fn, 'q', None) != q:
                raise ValueError("append: 所有 Fuzznum 的 mtype 和 q 必须一致。")

        return Fuzzarray(elements, mtype=mtype)

    # --- Fuzzarray 的 append 逻辑 ---
    if inplace and axis is not None:
        raise ValueError("append: inplace=True is not supported with axis specified.")

    # 统一将 item 转换为 ndarray
    if isinstance(item, Fuzznum):
        if self.mtype != item.mtype or self.q != item.q:
            raise ValueError("append: Fuzzarray 和 Fuzznum 的 'mtype' 和 'q' 必须一致。")
        item_data = np.array([item], dtype=object)
    elif isinstance(item, Fuzzarray):
        if self.mtype != item.mtype or self.q != item.q:
            raise ValueError("append: 两个 Fuzzarray 的 'mtype' 和 'q' 必须一致。")
        item_data = item.data
    elif isinstance(item, list):
        # 可以在这里添加对列表中 Fuzznum 的 mtype 和 q 的检查
        item_data = np.array(item, dtype=object)
    else:
        raise TypeError(f"append: 不支持的 item 类型 '{type(item)}' for Fuzzarray。")

    source_data = self.data
    if axis is None:
        source_data = source_data.ravel()
        item_data = item_data.ravel()

    new_data = np.append(source_data, item_data, axis=axis if axis is not None else 0)

    if inplace:
        self._data = new_data
        # 如果原数组是多维且 axis=None，需要更新 shape
        if axis is None and self.ndim > 1:
            self._data = self._data.reshape(-1)  # 确保是扁平化的
        return None
    else:
        return Fuzzarray(new_data, mtype=self.mtype, copy=False)


@mixin.register(name='pop', target_classes=["Fuzznum", "Fuzzarray"], injection_type='both')
def _pop_impl(self: Union['Fuzznum', 'Fuzzarray'],
              index: int = -1,
              inplace: bool = False) -> Union['Fuzznum', tuple['Fuzznum', 'Fuzzarray'], None]:
    """
    从一维数组中移除并返回一个元素。

    - 对于 Fuzznum:
      操作无意义，总是抛出 TypeError。

    - 对于 Fuzzarray:
      必须是一维数组。
      如果 `inplace=True`，原地修改并返回被弹出的 Fuzznum。
      如果 `inplace=False`，返回一个元组 (popped_element, new_array)。

    Args:
        self: Fuzznum 或 Fuzzarray 对象。
        index (int): 要移除的元素的索引。
        inplace (bool): 如果为 True，则原地修改数组。

    Returns:
        被弹出的 Fuzznum 或 (Fuzznum, Fuzzarray) 元组。

    Raises:
        TypeError: 如果对 Fuzznum 调用 pop。
        ValueError: 如果 Fuzzarray 不是一维的。
        IndexError: 如果索引越界。
    """
    if isinstance(self, Fuzznum):
        raise TypeError("pop: Fuzznum object does not support pop operation.")

    if self.ndim != 1:
        raise ValueError("pop: Only one-dimensional Fuzzarray is supported.")

    if self.size == 0:
        raise IndexError("pop from empty Fuzzarray")

    # 取出元素并确保类型
    popped_item = self.data[index]
    if isinstance(popped_item, np.ndarray):
        # 极端情况下（如切片），取第一个
        popped_item = popped_item.item()
    # 类型断言
    assert isinstance(popped_item, Fuzznum), "Fuzzarray.data 元素应为 Fuzznum"

    new_data = np.delete(self.data, index)

    if inplace:
        self._data = new_data
        return popped_item
    else:
        new_array = Fuzzarray(new_data, mtype=self.mtype, copy=False)
        return popped_item, new_array
