#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/31 01:51
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import numpy as np
import threading
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Dict
from contextlib import contextmanager
import collections  # 用于 OrderedDict

from fuzzlab.config import get_config
from fuzzlab.core.executor import Executor
from fuzzlab.core.fuzznums import Fuzznum


class Fuzzarray:
    """
    高维模糊数数组，直接封装 numpy.ndarray，其元素为 Fuzznum 对象。

    提供向量化计算能力，支持：
    - 类似 numpy 的数组操作接口 (shape, ndim, size, 索引/切片)
    - 广播运算 (与 Fuzzarray 或单个 Fuzznum 进行运算)
    - 与现有 Fuzznum 框架 (Executor) 完全集成
    - 确保数组内所有 Fuzznum 具有相同的 mtype
    """

    def __init__(self,
                 data: Union[np.ndarray, List, Tuple, Fuzznum],
                 mtype: Optional[str] = None,
                 shape: Optional[Tuple[int, ...]] = None,
                 copy: bool = True):
        """
        初始化 Fuzzarray。

        Args:
            data: 输入数据。
                - numpy.ndarray: 包含 Fuzznum 对象的数组。
                - List/Tuple: 嵌套列表/元组，元素为 Fuzznum。
                - Fuzznum: 单个模糊数，会根据 shape 广播。
            mtype: 强制指定的模糊数类型。如果提供，将验证所有 Fuzznum 的 mtype 是否一致。
            shape: 目标形状。主要用于从单个 Fuzznum 创建数组，或对输入数据进行 reshape。
            copy: 是否复制输入数据中的 Fuzznum 对象。如果为 True，则会进行深拷贝。
        """
        self._config = get_config()
        self._lock = threading.RLock()  # 用于保护内部状态和统计数据
        self._executor = Executor()  # 每个 Fuzzarray 实例拥有自己的 Executor

        self._data: np.ndarray = self._process_input_data(data, shape, copy)
        self._mtype: str = self._validate_and_set_mtype(mtype)

        # 缓存相关 (Fuzzarray 级别的操作结果缓存)
        self._cache_enabled = getattr(self._config, 'ENABLE_FUZZARRAY_CACHE', True)
        self._operation_cache: collections.OrderedDict = collections.OrderedDict()
        self._max_cache_size = getattr(self._config, 'FUZZARRAY_CACHE_SIZE', 256)

        # 性能统计
        self._stats = {
            'operations_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_cache_requests': 0
        }

    def _process_input_data(self,
                            data: Union[np.ndarray, List, Tuple, Fuzznum],
                            shape: Optional[Tuple[int, ...]],
                            copy: bool) -> np.ndarray:
        """
        处理不同类型的输入数据，将其转换为内部的 numpy.ndarray。
        """
        if isinstance(data, Fuzznum):
            if shape is None:
                shape = (1,)  # 默认单个 Fuzznum 视为 1 维数组
            # 使用 np.full 创建指定形状的数组，所有元素都是同一个 Fuzznum 的引用
            arr = np.full(shape, data, dtype=object)
            if copy:
                # 如果需要深拷贝，则复制每个 Fuzznum 对象
                arr = self._deep_copy_fuzznums(arr)
            return arr
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=object)
            if copy:
                arr = self._deep_copy_fuzznums(arr)
            if shape is not None:
                arr = arr.reshape(shape)
            return arr
        elif isinstance(data, np.ndarray):
            if data.dtype != object:
                raise TypeError("Input numpy.ndarray must have dtype=object for Fuzznum elements.")
            arr = data.copy() if copy else data
            if shape is not None:
                arr = arr.reshape(shape)
            return arr
        else:
            raise TypeError(f"Unsupported data type for Fuzzarray: {type(data)}")

    def _deep_copy_fuzznums(self, arr: np.ndarray) -> np.ndarray:
        """
        深度复制数组中的所有 Fuzznum 对象。
        """
        result = np.empty_like(arr, dtype=object)
        for index, item in np.ndenumerate(arr):
            if isinstance(item, Fuzznum):
                result[index] = item.copy()  # 调用 Fuzznum 的 copy 方法
            else:
                # 如果数组中包含非 Fuzznum 对象，直接赋值
                result[index] = item
        return result

    def _validate_and_set_mtype(self, expected_mtype: Optional[str]) -> str:
        """
        验证所有元素都是 Fuzznum 且 mtype 一致，并返回确定的 mtype。
        """
        if self._data.size == 0:
            # 对于空数组，mtype 可以由外部指定或使用默认值
            return expected_mtype or self._config.DEFAULT_MTYPE

        # 检查第一个非 None Fuzznum 元素以确定 mtype
        first_fuzznum = None
        for item in self._data.flat:
            if isinstance(item, Fuzznum):
                first_fuzznum = item
                break

        if first_fuzznum is None:
            raise ValueError("Fuzzarray must contain at least one Fuzznum object.")

        detected_mtype = first_fuzznum.mtype

        # 如果指定了 expected_mtype，检查是否一致
        if expected_mtype and detected_mtype != expected_mtype:
            raise ValueError(f"mtype mismatch: expected '{expected_mtype}', "
                             f"but found '{detected_mtype}' in initial Fuzznum.")

        # 验证所有元素类型一致
        for index, item in np.ndenumerate(self._data):
            if not isinstance(item, Fuzznum):
                raise TypeError(f"All elements in Fuzzarray must be Fuzznum objects, "
                                f"found {type(item)} at index {index}.")
            if item.mtype != detected_mtype:
                raise ValueError(f"All Fuzznums in Fuzzarray must have the same mtype. "
                                 f"Expected '{detected_mtype}', found '{item.mtype}' at index {index}.")
        return detected_mtype

    # ======================== 类似 numpy 的属性接口 ========================

    @property
    def shape(self) -> Tuple[int, ...]:
        """数组的形状。"""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """数组的维度数。"""
        return self._data.ndim

    @property
    def size(self) -> int:
        """数组中元素的总数。"""
        return self._data.size

    @property
    def mtype(self) -> str:
        """数组中所有 Fuzznum 元素的模糊数类型。"""
        return self._mtype

    @property
    def dtype(self) -> str:
        """返回数据类型，对于 Fuzzarray 而言，即为 mtype。"""
        return self._mtype

    def __len__(self) -> int:
        """返回数组第一维的长度。"""
        return len(self._data)

    def __iter__(self) -> Iterator[Union[Fuzznum, 'Fuzzarray']]:
        """
        迭代器，按第一维返回子 Fuzzarray 或单个 Fuzznum。
        """
        for item in self._data:
            if isinstance(item, np.ndarray):
                yield Fuzzarray(item, mtype=self.mtype, copy=False)
            else:
                yield item

    # ======================== 索引和切片操作 ========================

    def __getitem__(self, key) -> Union[Fuzznum, 'Fuzzarray']:
        """
        通过索引或切片访问 Fuzzarray 元素。

        Args:
            key: 索引或切片键，可以是整数、切片对象、元组等。

        Returns:
            Union[Fuzznum, Fuzzarray]: 如果结果是单个元素，返回 Fuzznum；
                                       如果是子数组，返回一个新的 Fuzzarray 实例。
        """
        result = self._data[key]
        if isinstance(result, np.ndarray):
            # 返回新的 Fuzzarray 实例，共享底层数据（copy=False）
            return Fuzzarray(result, mtype=self.mtype, copy=False)
        else:
            # 返回单个 Fuzznum
            return result

    def __setitem__(self, key, value: Union[Fuzznum, 'Fuzzarray']) -> None:
        """
        通过索引或切片设置 Fuzzarray 元素。

        Args:
            key: 索引或切片键。
            value: 要设置的值，可以是单个 Fuzznum 或另一个 Fuzzarray。

        Raises:
            TypeError: 如果赋值类型不兼容。
            ValueError: 如果赋值的 Fuzznum mtype 不一致。
        """
        if isinstance(value, Fuzznum):
            if value.mtype != self._mtype:
                raise ValueError(f"mtype mismatch: expected '{self._mtype}', "
                                 f"got '{value.mtype}' for assigned Fuzznum.")
            self._data[key] = value
        elif isinstance(value, Fuzzarray):
            if value.mtype != self._mtype:
                raise ValueError(f"mtype mismatch: expected '{self._mtype}', "
                                 f"got '{value.mtype}' for assigned Fuzzarray.")
            # 直接将另一个 Fuzzarray 的内部 NumPy 数组赋值
            self._data[key] = value._data
        else:
            raise TypeError(f"Can only assign Fuzznum or Fuzzarray, got {type(value)}.")

    # ======================== 形状操作方法 ========================

    def reshape(self, *args, **kwargs) -> 'Fuzzarray':
        """
        重塑数组的形状。

        Args:
            *args: 新形状的参数。
            **kwargs: 额外的关键字参数，传递给 numpy.ndarray.reshape。

        Returns:
            Fuzzarray: 具有新形状的 Fuzzarray 实例，共享底层数据。
        """
        new_data = self._data.reshape(*args, **kwargs)
        return Fuzzarray(new_data, mtype=self.mtype, copy=False)

    def flatten(self) -> 'Fuzzarray':
        """
        将数组展平为一维数组。

        Returns:
            Fuzzarray: 展平后的 Fuzzarray 实例，共享底层数据。
        """
        return Fuzzarray(self._data.flatten(), mtype=self.mtype, copy=False)

    def transpose(self, *axes) -> 'Fuzzarray':
        """
        转置数组的维度。

        Args:
            *axes: 维度顺序的元组。

        Returns:
            Fuzzarray: 转置后的 Fuzzarray 实例，共享底层数据。
        """
        if not axes:
            new_data = self._data.T
        else:
            new_data = self._data.transpose(*axes)
        return Fuzzarray(new_data, mtype=self.mtype, copy=False)

    @property
    def T(self) -> 'Fuzzarray':
        """
        数组的转置视图。

        Returns:
            Fuzzarray: 转置后的 Fuzzarray 实例，共享底层数据。
        """
        return self.transpose()

    def copy(self) -> 'Fuzzarray':
        """
        创建 Fuzzarray 及其所有 Fuzznum 元素的深拷贝。

        Returns:
            Fuzzarray: Fuzzarray 实例的深拷贝。
        """
        # 调用 _process_input_data 并强制深拷贝
        return Fuzzarray(self._data, mtype=self.mtype, copy=True)

    # ======================== 向量化运算的核心实现 ========================

    def _execute_vectorized_op(self,
                               other: Optional[Union['Fuzzarray', Fuzznum]],
                               op_method_name: str,
                               is_unary: bool = False,
                               **kwargs) -> Union['Fuzzarray', np.ndarray]:
        """
        内部方法：执行向量化运算的通用逻辑。

        Args:
            other: 另一个操作数，可以是 Fuzzarray 或 Fuzznum。
            op_method_name: Executor 中对应操作的方法名（例如 'addition', 'power'）。
            is_unary: 是否为一元运算。
            **kwargs: 传递给 Executor 方法的额外参数。

        Returns:
            Union[Fuzzarray, np.ndarray]: 运算结果。对于比较运算返回 np.ndarray (bool)，
                                          其他运算返回 Fuzzarray。
        """
        with self._lock:
            self._stats['total_cache_requests'] += 1
            cache_key = self._generate_cache_key(op_method_name, other, is_unary, kwargs)
            if self._cache_enabled and cache_key in self._operation_cache:
                self._stats['cache_hits'] += 1
                # 移动到最后表示最近使用 (LRU 策略)
                self._operation_cache.move_to_end(cache_key)
                cached_result = self._operation_cache[cache_key]
                # 返回缓存结果的深拷贝，防止外部修改影响缓存
                if isinstance(cached_result, Fuzzarray):
                    return cached_result.copy()
                return cached_result.copy() if isinstance(cached_result, np.ndarray) else cached_result

            self._stats['cache_misses'] += 1
            self._stats['operations_count'] += 1

        executor_method = getattr(self._executor, op_method_name)
        result_is_bool = op_method_name in ['greater_than', 'less_than', 'equal',
                                            'greater_equal', 'less_equal', 'not_equal']

        if is_unary:
            # 一元运算：对 Fuzzarray 中的每个 Fuzznum 执行操作
            result_data = np.empty_like(self._data, dtype=object if not result_is_bool else bool)
            for index, fuzznum in np.ndenumerate(self._data):
                result_data[index] = executor_method(fuzznum, other, **kwargs)  # other 在一元运算中是 operand
        else:
            # 二元运算：处理 Fuzzarray 与 Fuzzarray 或 Fuzzarray 与 Fuzznum
            if isinstance(other, Fuzznum):
                # Fuzzarray 与单个 Fuzznum 的广播运算
                result_data = np.empty_like(self._data, dtype=object if not result_is_bool else bool)
                for index, self_fuzznum in np.ndenumerate(self._data):
                    result_data[index] = executor_method(self_fuzznum, other, **kwargs)
            elif isinstance(other, Fuzzarray):
                # Fuzzarray 与 Fuzzarray 的逐元素运算 (支持广播)
                if self.mtype != other.mtype:
                    raise ValueError(f"mtype mismatch for binary operation: "
                                     f"'{self.mtype}' vs '{other.mtype}'.")

                # 获取广播后的形状
                try:
                    result_shape = np.broadcast_shapes(self.shape, other.shape)
                except ValueError as e:
                    raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape} for operation "
                                     f"'{op_method_name}'.") from e

                # 广播数组到相同形状
                self_broadcasted = np.broadcast_to(self._data, result_shape)
                other_broadcasted = np.broadcast_to(other._data, result_shape)

                result_data = np.empty(result_shape, dtype=object if not result_is_bool else bool)

                # 逐元素执行操作
                for index, (fuzz1, fuzz2) in np.ndenumerate(self_broadcasted):
                    result_data[index] = executor_method(fuzz1, fuzz2, **kwargs)
            else:
                raise TypeError(f"Unsupported operand type for '{op_method_name}': {type(other)}.")

        final_result = Fuzzarray(result_data, mtype=self.mtype, copy=False) if not result_is_bool else result_data

        with self._lock:
            # 缓存结果
            self._operation_cache[cache_key] = final_result
            if len(self._operation_cache) > self._max_cache_size:
                self._operation_cache.popitem(last=False)  # 移除最旧的 (LRU)

        return final_result

    def _generate_cache_key(self,
                            op_method_name: str,
                            other: Optional[Union['Fuzzarray', Fuzznum]],
                            is_unary: bool,
                            kwargs: Dict[str, Any]) -> str:
        """
        生成操作的缓存键。
        """
        import hashlib

        # Fuzzarray 的标识符：mtype, shape, 以及所有 Fuzznum 实例的唯一标识（基于其属性值）
        self_id_parts = [self.mtype, str(self.shape)]
        for fuzznum in self._data.flat:
            # 使用 Fuzznum 的 to_dict() 来获取其状态，确保内容一致性
            self_id_parts.append(str(fuzznum.to_dict()))
        self_id = hashlib.md5("_".join(self_id_parts).encode()).hexdigest()

        key_parts = [op_method_name, self_id]

        if not is_unary:
            if isinstance(other, Fuzznum):
                key_parts.append(str(other.to_dict()))
            elif isinstance(other, Fuzzarray):
                other_id_parts = [other.mtype, str(other.shape)]
                for fuzznum in other._data.flat:
                    other_id_parts.append(str(fuzznum.to_dict()))
                other_id = hashlib.md5("_".join(other_id_parts).encode()).hexdigest()
                key_parts.append(other_id)
            else:
                key_parts.append(str(other))  # Fallback for unexpected types
        else:  # Unary operation, 'other' is the operand
            key_parts.append(str(other))

        if kwargs:
            key_parts.append(str(sorted(kwargs.items())))

        return hashlib.md5("_".join(key_parts).encode()).hexdigest()

    # ======================== 具体运算方法 (运算符重载) ========================

    def __add__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的加法运算。"""
        return self._execute_vectorized_op(other, 'addition')

    def __sub__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的减法运算。"""
        return self._execute_vectorized_op(other, 'subtract')

    def __mul__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的乘法运算。"""
        return self._execute_vectorized_op(other, 'multiply')

    def __truediv__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的除法运算。"""
        return self._execute_vectorized_op(other, 'divide')

    def __pow__(self, operand: Union[int, float]) -> 'Fuzzarray':
        """Fuzzarray 的幂运算。"""
        return self._execute_vectorized_op(operand, 'power', is_unary=True)

    # ======================== 具体运算方法 (命名方法) ========================

    def times(self, operand: Union[int, float], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的倍数运算。

        Args:
            operand: 乘数（标量）。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op(operand, 'times', is_unary=True, **kwargs)

    # ... 其他一元运算 (exponential, logarithmic) 可以类似实现 ...

    # ======================== 比较运算 (返回 np.ndarray[bool]) ========================

    def __gt__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的大于比较。"""
        return self._execute_vectorized_op(other, 'greater_than')

    def __lt__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的小于比较。"""
        return self._execute_vectorized_op(other, 'less_than')

    def __eq__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的等于比较。"""
        return self._execute_vectorized_op(other, 'equal')

    def __ge__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的大于等于比较。"""
        return self._execute_vectorized_op(other, 'greater_equal')

    def __le__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的小于等于比较。"""
        return self._execute_vectorized_op(other, 'less_equal')

    def __ne__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的不等于比较。"""
        return self._execute_vectorized_op(other, 'not_equal')

    # ======================== 聚合运算方法 ========================

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Fuzznum, 'Fuzzarray']:
        """
        计算 Fuzzarray 沿指定轴的和。

        Args:
            axis: 求和的轴。None 表示对所有元素求和。

        Returns:
            Union[Fuzznum, Fuzzarray]: 如果 axis 为 None，返回单个 Fuzznum；
                                       否则返回一个降维的 Fuzzarray。
        """
        if self.size == 0:
            raise ValueError("Sum of empty Fuzzarray is not defined.")

        if axis is None:
            # 对所有元素求和
            result = None
            for fuzznum in self._data.flat:
                if result is None:
                    result = fuzznum.copy()
                else:
                    result = self._executor.addition(result, fuzznum)
            return result
        else:
            # 按指定轴求和
            # NumPy 的 sum 默认不支持自定义对象，需要手动实现
            # 这是一个复杂的操作，涉及到 Fuzznum 的加法和形状管理
            # 简化版：仅支持单轴求和
            if not isinstance(axis, int):
                raise NotImplementedError("Only single axis sum is supported for now.")
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"Axis {axis} is out of bounds for array with {self.ndim} dimensions.")

            # 计算结果数组的形状
            result_shape = list(self.shape)
            result_shape.pop(axis)
            if not result_shape:  # 如果求和后变成标量
                result_shape = (1,)

            # 创建结果数组
            sum_result_data = np.empty(tuple(result_shape), dtype=object)

            # 遍历除 axis 之外的所有维度
            # 使用 np.apply_along_axis 简化逻辑
            def _sum_fuzznums_along_axis(arr_1d: np.ndarray) -> Fuzznum:
                current_sum = None
                for fnum in arr_1d:
                    if current_sum is None:
                        current_sum = fnum.copy()
                    else:
                        current_sum = self._executor.addition(current_sum, fnum)
                return current_sum

            # np.apply_along_axis 会将指定轴上的元素作为 1D 数组传递给函数
            sum_result_data = np.apply_along_axis(_sum_fuzznums_along_axis, axis, self._data)

            if sum_result_data.shape == (1,):  # 如果结果是标量
                return sum_result_data.item()
            return Fuzzarray(sum_result_data, mtype=self.mtype, copy=False)

    # ... 其他聚合运算 (mean, min, max) 可以类似实现 ...

    # ======================== 工厂方法 ========================

    @classmethod
    def zeros(cls, shape: Tuple[int, ...], mtype: str, **fuzznum_kwargs) -> 'Fuzzarray':
        """
        创建零模糊数数组。

        Args:
            shape: 数组的形状。
            mtype: 模糊数类型。
            **fuzznum_kwargs: 传递给 Fuzznum 构造函数的额外参数。
                              通常用于定义零模糊数的具体属性（如 md=0, nmd=1）。

        Returns:
            Fuzzarray: 包含零模糊数的 Fuzzarray 实例。
        """
        # 零模糊数通常定义为 md=0, nmd=1
        default_zero_kwargs = {'md': 0.0, 'nmd': 1.0}
        # 合并用户提供的 kwargs，用户提供的值优先
        merged_kwargs = {**default_zero_kwargs, **fuzznum_kwargs}
        zero_fuzznum = Fuzznum(mtype=mtype, **merged_kwargs)
        # 使用 np.full 创建数组，并深拷贝每个 Fuzznum 实例
        return cls(zero_fuzznum, shape=shape, mtype=mtype, copy=True)

    @classmethod
    def ones(cls, shape: Tuple[int, ...], mtype: str, **fuzznum_kwargs) -> 'Fuzzarray':
        """
        创建单位模糊数数组。

        Args:
            shape: 数组的形状。
            mtype: 模糊数类型。
            **fuzznum_kwargs: 传递给 Fuzznum 构造函数的额外参数。
                              通常用于定义单位模糊数的具体属性（如 md=1, nmd=0）。

        Returns:
            Fuzzarray: 包含单位模糊数的 Fuzzarray 实例。
        """
        # 单位模糊数通常定义为 md=1, nmd=0
        default_one_kwargs = {'md': 1.0, 'nmd': 0.0}
        merged_kwargs = {**default_one_kwargs, **fuzznum_kwargs}
        one_fuzznum = Fuzznum(mtype=mtype, **merged_kwargs)
        return cls(one_fuzznum, shape=shape, mtype=mtype, copy=True)

    @classmethod
    def empty(cls, shape: Tuple[int, ...], mtype: str) -> 'Fuzzarray':
        """
        创建空的 Fuzzarray（元素未初始化）。

        Args:
            shape: 数组的形状。
            mtype: 模糊数类型。

        Returns:
            Fuzzarray: 包含未初始化元素的 Fuzzarray 实例。
        """
        data = np.empty(shape, dtype=object)
        # 注意：这里 mtype 只是一个占位符，因为元素尚未初始化
        # 实际使用时，如果元素被赋值，会进行 mtype 检查
        return cls(data, mtype=mtype, copy=False)

    # ======================== 缓存和性能管理 ========================

    def enable_cache(self) -> None:
        """启用 Fuzzarray 级别的操作结果缓存。"""
        with self._lock:
            self._cache_enabled = True

    def disable_cache(self) -> None:
        """禁用 Fuzzarray 级别的操作结果缓存并清空现有缓存。"""
        with self._lock:
            self._cache_enabled = False
            self._operation_cache.clear()

    def get_stats(self) -> dict:
        """
        获取 Fuzzarray 实例的性能统计信息。

        Returns:
            dict: 包含操作计数、缓存命中率等信息的字典。
        """
        with self._lock:
            total_requests = self._stats['total_cache_requests']
            cache_hits = self._stats['cache_hits']
            hit_ratio = cache_hits / total_requests if total_requests > 0 else 0.0
            return {
                'operations_count': self._stats['operations_count'],
                'cache_hits': cache_hits,
                'cache_misses': self._stats['cache_misses'],
                'total_cache_requests': total_requests,
                'cache_hit_ratio': hit_ratio,
                'current_cache_size': len(self._operation_cache)
            }

    def reset_stats(self) -> None:
        """重置 Fuzzarray 实例的所有性能统计数据。"""
        with self._lock:
            self._stats = {
                'operations_count': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'total_cache_requests': 0
            }
            self._operation_cache.clear()  # 清空缓存

    # ======================== 字符串表示 ========================

    def __repr__(self) -> str:
        """
        对象的“官方”字符串表示。

        Returns:
            str: 包含 Fuzzarray 形状、mtype 和部分内容的字符串。
        """
        # 为了避免打印过大的数组，只显示部分元素
        if self.size <= 10:  # 小数组直接显示所有元素
            elements_str = np.array2string(self._data, formatter={'all': lambda x: str(x)})
        else:  # 大数组显示省略号
            # 尝试获取前几个和后几个元素
            flat_data = self._data.flat
            first_elements = [str(next(flat_data)) for _ in range(3)]
            last_elements = []
            if self.size > 3:
                # 倒序遍历以获取最后几个元素，但要确保不重复
                temp_list = list(self._data.flat)
                last_elements = [str(x) for x in temp_list[-3:]]

            elements_str = f"[{', '.join(first_elements)}, ..., {', '.join(last_elements)}]"

        return (f"Fuzzarray(\n{elements_str},\n"
                f"       shape={self.shape}, mtype='{self.mtype}')")

    def __str__(self) -> str:
        """
        用户友好的字符串表示。

        Returns:
            str: 简洁的 Fuzzarray 字符串表示。
        """
        # 默认与 __repr__ 相同，可以根据需要自定义更简洁的输出
        return self.__repr__()


# ======================== 便捷的创建函数 ========================

def fuzzarray(data: Union[np.ndarray, List, Tuple, Fuzznum],
              mtype: Optional[str] = None,
              shape: Optional[Tuple[int, ...]] = None,
              copy: bool = True) -> Fuzzarray:
    """
    创建 Fuzzarray 的便捷函数，类似 np.array()。

    Args:
        data: 输入数据。
        mtype: 模糊数类型。
        shape: 目标形状。
        copy: 是否复制数据。

    Returns:
        Fuzzarray: 创建的模糊数数组。
    """
    return Fuzzarray(data, mtype=mtype, shape=shape, copy=copy)


def fuzzeros(shape: Tuple[int, ...], mtype: str, **fuzznum_kwargs) -> Fuzzarray:
    """
    创建零模糊数数组，类似 np.zeros()。

    Args:
        shape: 数组的形状。
        mtype: 模糊数类型。
        **fuzznum_kwargs: 传递给 Fuzznum 构造函数的额外参数。

    Returns:
        Fuzzarray: 包含零模糊数的 Fuzzarray 实例。
    """
    return Fuzzarray.zeros(shape, mtype, **fuzznum_kwargs)


def fuzzones(shape: Tuple[int, ...], mtype: str, **fuzznum_kwargs) -> Fuzzarray:
    """
    创建单位模糊数数组，类似 np.ones()。

    Args:
        shape: 数组的形状。
        mtype: 模糊数类型。
        **fuzznum_kwargs: 传递给 Fuzznum 构造函数的额外参数。

    Returns:
        Fuzzarray: 包含单位模糊数的 Fuzzarray 实例。
    """
    return Fuzzarray.ones(shape, mtype, **fuzznum_kwargs)


def fuzzempty(shape: Tuple[int, ...], mtype: str) -> Fuzzarray:
    """
    创建空模糊数数组，类似 np.empty()。

    Args:
        shape: 数组的形状。
        mtype: 模糊数类型。

    Returns:
        Fuzzarray: 包含未初始化元素的 Fuzzarray 实例。
    """
    return Fuzzarray.empty(shape, mtype)
