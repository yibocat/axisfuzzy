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
from fuzzlab.core._executor import Executor
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

        Args:
            data: 原始输入数据。
            shape: 期望的数组形状。
            copy: 是否深拷贝 Fuzznum 元素。

        Returns:
            np.ndarray: 包含 Fuzznum 对象的 NumPy 数组。

        Raises:
            TypeError: 如果数据类型不支持。
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

    def _validate_and_set_mtype(self, expected_mtype: Optional[str]) -> str:
        """
        验证所有元素都是 Fuzznum 且 mtype 一致，并返回确定的 mtype。

        Args:
            expected_mtype: 期望的模糊数类型。

        Returns:
            str: 数组中 Fuzznum 元素的统一 mtype。

        Raises:
            ValueError: 如果 mtype 不一致或数组中没有 Fuzznum。
            TypeError: 如果数组中包含非 Fuzznum 元素。
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

    def _deep_copy_fuzznums(self, arr: np.ndarray) -> np.ndarray:
        """
        深度复制数组中的所有 Fuzznum 对象。

        Args:
            arr: 包含 Fuzznum 对象的 NumPy 数组。

        Returns:
            np.ndarray: 包含深拷贝 Fuzznum 对象的 NumPy 数组。
        """
        result = np.empty_like(arr, dtype=object)
        for index, item in np.ndenumerate(arr):
            if isinstance(item, Fuzznum):
                result[index] = item.copy()  # 调用 Fuzznum 的 copy 方法
            else:
                # 如果数组中包含非 Fuzznum 对象，直接赋值
                result[index] = item
        return result

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
                               op_method_name: str,
                               fuzznum_or_scalar_operand: Optional[Union['Fuzzarray', Fuzznum, float, int]] = None,
                               **kwargs) -> Union['Fuzzarray', np.ndarray]:
        """
        内部方法：执行向量化运算的通用逻辑。

        Args:
            op_method_name: Executor 中对应操作的方法名（例如 'addition', 'power'）。
            fuzznum_or_scalar_operand: 另一个操作数。
                - 对于二元运算 (e.g., add, mul, gt): 可以是 Fuzzarray, Fuzznum。
                - 对于一元运算 (e.g., power, times, exp, log): 可以是 float, int。
                - 对于纯一元运算 (e.g., complement): 应该为 None。
            **kwargs: 传递给 Executor 方法的额外参数。

        Returns:
            Union[Fuzzarray, np.ndarray]: 运算结果。对于比较运算返回 np.ndarray (bool)，
                                          其他运算返回 Fuzzarray。
        """
        with self._lock:
            self._stats['total_cache_requests'] += 1
            cache_key = self._generate_cache_key(op_method_name, fuzznum_or_scalar_operand, kwargs)
            if self._cache_enabled and cache_key in self._operation_cache:
                self._stats['cache_hits'] += 1
                # 移动到最后表示最近使用 (LRU 策略)
                self._operation_cache.move_to_end(cache_key)
                cached_result = self._operation_cache[cache_key]
                # 返回缓存结果的深拷贝，防止外部修改影响缓存
                if isinstance(cached_result, Fuzzarray):
                    return cached_result.copy()
                # For np.ndarray (like boolean results), returning a copy is also safer
                return cached_result.copy() if isinstance(cached_result, np.ndarray) else cached_result

            self._stats['cache_misses'] += 1
            self._stats['operations_count'] += 1

        executor_method = getattr(self._executor, op_method_name)

        # Determine if the operation is a comparison (returns bool)
        result_is_bool = op_method_name in ['greater_than', 'less_than', 'equal',
                                            'greater_equal', 'less_equal', 'not_equal']

        # Determine the result dtype based on whether it's a boolean result
        result_dtype = object if not result_is_bool else bool

        # Handle different types of operations based on fuzznum_or_scalar_operand
        if fuzznum_or_scalar_operand is None:  # Pure unary op (e.g., complement)
            result_data = np.empty_like(self._data, dtype=result_dtype)
            for index, fuzznum in np.ndenumerate(self._data):
                result_data[index] = executor_method(fuzznum, **kwargs)
        elif isinstance(fuzznum_or_scalar_operand,
                        (float, int)):  # Unary op with scalar operand (e.g., power, times, exp, log)
            result_data = np.empty_like(self._data, dtype=result_dtype)
            for index, fuzznum in np.ndenumerate(self._data):
                result_data[index] = executor_method(fuzznum, fuzznum_or_scalar_operand, **kwargs)
        elif isinstance(fuzznum_or_scalar_operand, Fuzznum):  # Binary op with scalar Fuzznum operand
            # Fuzzarray vs single Fuzznum (broadcasting)
            result_data = np.empty_like(self._data, dtype=result_dtype)
            for index, self_fuzznum in np.ndenumerate(self._data):
                result_data[index] = executor_method(self_fuzznum, fuzznum_or_scalar_operand, **kwargs)
        elif isinstance(fuzznum_or_scalar_operand, Fuzzarray):  # Binary op with Fuzzarray operand
            # Fuzzarray vs Fuzzarray (element-wise with broadcasting)
            other_fuzzarray = fuzznum_or_scalar_operand  # Rename for clarity

            if self.mtype != other_fuzzarray.mtype:
                raise ValueError(f"mtype mismatch for binary operation: "
                                 f"'{self.mtype}' vs '{other_fuzzarray.mtype}'.")

            try:
                result_shape = np.broadcast_shapes(self.shape, other_fuzzarray.shape)
            except ValueError as e:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other_fuzzarray.shape} for operation "
                                 f"'{op_method_name}'.") from e

            self_broadcasted = np.broadcast_to(self._data, result_shape)
            other_broadcasted = np.broadcast_to(other_fuzzarray._data, result_shape)

            result_data = np.empty(result_shape, dtype=result_dtype)

            for index, (fuzz1, fuzz2) in np.ndenumerate(self_broadcasted):
                result_data[index] = executor_method(fuzz1, fuzz2, **kwargs)
        else:
            raise TypeError(f"Unsupported operand type for '{op_method_name}': {type(fuzznum_or_scalar_operand)}.")

        # Wrap result_data in Fuzzarray if it's not a boolean array
        final_result = Fuzzarray(result_data, mtype=self.mtype, copy=False) if not result_is_bool else result_data

        with self._lock:
            # Cache the result
            self._operation_cache[cache_key] = final_result
            if len(self._operation_cache) > self._max_cache_size:
                self._operation_cache.popitem(last=False)  # Remove the least recently used item

        return final_result

    def _generate_cache_key(self,
                            op_method_name: str,
                            fuzznum_or_scalar_operand: Optional[Union['Fuzzarray', Fuzznum, float, int]],
                            kwargs: Dict[str, Any]) -> str:
        """
        生成操作的缓存键。

        Args:
            op_method_name: 操作方法名。
            fuzznum_or_scalar_operand: 另一个操作数。
            kwargs: 额外参数。

        Returns:
            str: 唯一的缓存键。
        """
        import hashlib

        # Fuzzarray 的标识符：mtype, shape, 以及所有 Fuzznum 实例的唯一标识（基于其属性值）
        # 使用 Fuzznum 的 to_dict() 来获取其状态，确保内容一致性
        self_id_parts = [self.mtype, str(self.shape)]
        for fuzznum in self._data.flat:
            self_id_parts.append(str(fuzznum.to_dict()))
        self_id = hashlib.md5("_".join(self_id_parts).encode()).hexdigest()

        key_parts = [op_method_name, self_id]

        # Add other operand to key
        if fuzznum_or_scalar_operand is None:
            key_parts.append("None")
        elif isinstance(fuzznum_or_scalar_operand, (float, int)):
            key_parts.append(str(fuzznum_or_scalar_operand))
        elif isinstance(fuzznum_or_scalar_operand, Fuzznum):
            key_parts.append(str(fuzznum_or_scalar_operand.to_dict()))
        elif isinstance(fuzznum_or_scalar_operand, Fuzzarray):
            other_id_parts = [fuzznum_or_scalar_operand.mtype, str(fuzznum_or_scalar_operand.shape)]
            for fuzznum in fuzznum_or_scalar_operand._data.flat:
                other_id_parts.append(str(fuzznum.to_dict()))
            other_id = hashlib.md5("_".join(other_id_parts).encode()).hexdigest()
            key_parts.append(other_id)
        else:
            key_parts.append(str(fuzznum_or_scalar_operand))  # Fallback for unexpected types

        # Add kwargs to key
        if kwargs:
            key_parts.append(str(sorted(kwargs.items())))

        return hashlib.md5("_".join(key_parts).encode()).hexdigest()

    # ======================== 具体运算方法 (运算符重载) ========================

    def __add__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的加法运算。"""
        return self._execute_vectorized_op('addition', other)

    def __sub__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的减法运算。"""
        return self._execute_vectorized_op('subtract', other)

    def __mul__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的乘法运算。"""
        return self._execute_vectorized_op('multiply', other)

    def __truediv__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        """Fuzzarray 和 Fuzzarray/Fuzznum 的除法运算。"""
        return self._execute_vectorized_op('divide', other)

    def __pow__(self, operand: Union[int, float]) -> 'Fuzzarray':
        """Fuzzarray 的幂运算。"""
        return self._execute_vectorized_op('power', operand)

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
        return self._execute_vectorized_op(operand, 'times', **kwargs)

    def exp(self, operand: Union[int, float], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的指数运算 (e.g., base^Fuzznum)。

        Args:
            operand: 指数运算的底数。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('exponential', operand, **kwargs)

    def log(self, operand: Union[int, float], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的对数运算 (e.g., log_base(Fuzznum))。

        Args:
            operand: 对数运算的底数。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('logarithmic', operand, **kwargs)

    # ======================== 比较运算 (返回 np.ndarray[bool]) ========================

    def __gt__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的大于比较。"""
        return self._execute_vectorized_op('greater_than', other)

    def __lt__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的小于比较。"""
        return self._execute_vectorized_op('less_than', other)

    def __eq__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的等于比较。"""
        return self._execute_vectorized_op('equal', other)

    def __ge__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的大于等于比较。"""
        return self._execute_vectorized_op('greater_equal', other)

    def __le__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的小于等于比较。"""
        return self._execute_vectorized_op('less_equal', other)

    def __ne__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        """Fuzzarray 的不等于比较。"""
        return self._execute_vectorized_op('not_equal', other)

    # ======================== 逻辑运算 ========================

    def intersection(self, other: Union['Fuzzarray', Fuzznum], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的逻辑交 (AND) 运算。

        Args:
            other: 另一个 Fuzzarray 或 Fuzznum。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('intersection', other, **kwargs)

    def union(self, other: Union['Fuzzarray', Fuzznum], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的逻辑并 (OR) 运算。

        Args:
            other: 另一个 Fuzzarray 或 Fuzznum。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('union', other, **kwargs)

    def complement(self, **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的逻辑补 (NOT) 运算。

        Args:
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('complement', None, **kwargs)  # 'None' indicates no explicit second operand

    def implication(self, other: Union['Fuzzarray', Fuzznum], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的逻辑蕴含 (IMPLIES) 运算。

        Args:
            other: 另一个 Fuzzarray 或 Fuzznum。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('implication', other, **kwargs)

    def equivalence(self, other: Union['Fuzzarray', Fuzznum], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的逻辑等价 (EQUIVALENCE) 运算。

        Args:
            other: 另一个 Fuzzarray 或 Fuzznum。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('equivalence', other, **kwargs)

    def difference(self, other: Union['Fuzzarray', Fuzznum], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的逻辑差 (DIFFERENCE) 运算。

        Args:
            other: 另一个 Fuzzarray 或 Fuzznum。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('difference', other, **kwargs)

    def symmetric_difference(self, other: Union['Fuzzarray', Fuzznum], **kwargs) -> 'Fuzzarray':
        """
        Fuzzarray 的逻辑对称差 (SYMMETRIC DIFFERENCE) 运算。

        Args:
            other: 另一个 Fuzzarray 或 Fuzznum。
            **kwargs: 传递给 Executor 的额外参数。

        Returns:
            Fuzzarray: 运算结果。
        """
        return self._execute_vectorized_op('symmetric_difference', other, **kwargs)

    # ======================== 聚合运算 ========================

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
            if isinstance(axis, tuple):
                # 递归处理多轴求和
                temp_arr = self
                for ax in sorted(axis, reverse=True):  # 从高维到低维依次求和
                    temp_arr = temp_arr.sum(axis=ax)
                return temp_arr

            if not isinstance(axis, int):
                raise TypeError(f"Axis must be an integer or a tuple of integers, got {type(axis)}.")
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"Axis {axis} is out of bounds for array with {self.ndim} dimensions.")

            # 计算结果数组的形状
            result_shape = list(self.shape)
            result_shape.pop(axis)
            if not result_shape:  # 如果求和后变成标量
                return_scalar = True
                result_shape = (1,)
            else:
                return_scalar = False

            # 创建结果数组
            sum_result_data = np.empty(tuple(result_shape), dtype=object)

            # 定义一个辅助函数，用于 np.apply_along_axis
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

            if return_scalar:  # 如果结果是标量
                return sum_result_data.item()
            return Fuzzarray(sum_result_data, mtype=self.mtype, copy=False)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Fuzznum, 'Fuzzarray']:
        """
        计算 Fuzzarray 沿指定轴的平均值。

        Args:
            axis: 计算平均值的轴。None 表示对所有元素求平均。

        Returns:
            Union[Fuzznum, Fuzzarray]: 如果 axis 为 None，返回单个 Fuzznum；
                                       否则返回一个降维的 Fuzzarray。
        """
        if self.size == 0:
            raise ValueError("Mean of empty Fuzzarray is not defined.")

        total_sum = self.sum(axis=axis)

        if axis is None:
            count = self.size
        elif isinstance(axis, int):
            count = self.shape[axis]
        elif isinstance(axis, tuple):
            count = np.prod([self.shape[ax] for ax in axis])
        else:
            raise TypeError(f"Axis must be an integer or a tuple of integers, got {type(axis)}.")

        if count == 0:
            raise ValueError("Cannot compute mean of an empty slice.")

        # 假设 Fuzznum 可以被标量除，或者 Executor.times 可以处理 1/count
        if isinstance(total_sum, Fuzznum):  # Sum result is a scalar Fuzznum
            return self._executor.times(total_sum, 1.0 / count)
        else:  # Sum result is a Fuzzarray
            # Apply times operation to each element of the sum_result Fuzzarray
            return total_sum.times(1.0 / count)

    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Fuzznum, 'Fuzzarray']:
        """
        计算 Fuzzarray 沿指定轴的最小值。
        比较基于 Fuzznum 的默认比较逻辑 (Executor.greater_than/less_than)。

        Args:
            axis: 查找最小值的轴。None 表示对所有元素查找。

        Returns:
            Union[Fuzznum, Fuzzarray]: 如果 axis 为 None，返回单个 Fuzznum；
                                       否则返回一个降维的 Fuzzarray。
        """
        if self.size == 0:
            raise ValueError("Min of empty Fuzzarray is not defined.")

        if axis is None:
            # 查找所有元素的最小值
            min_fuzznum = None
            for fuzznum in self._data.flat:
                if min_fuzznum is None:
                    min_fuzznum = fuzznum.copy()
                else:
                    # 使用 Executor 的比较方法
                    if self._executor.less_than(fuzznum, min_fuzznum):
                        min_fuzznum = fuzznum.copy()
            return min_fuzznum
        else:
            if isinstance(axis, tuple):
                temp_arr = self
                for ax in sorted(axis, reverse=True):
                    temp_arr = temp_arr.min(axis=ax)
                return temp_arr

            if not isinstance(axis, int):
                raise TypeError(f"Axis must be an integer or a tuple of integers, got {type(axis)}.")
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"Axis {axis} is out of bounds for array with {self.ndim} dimensions.")

            result_shape = list(self.shape)
            result_shape.pop(axis)
            if not result_shape:
                return_scalar = True
                result_shape = (1,)
            else:
                return_scalar = False

            min_result_data = np.empty(tuple(result_shape), dtype=object)

            def _min_fuzznums_along_axis(arr_1d: np.ndarray) -> Fuzznum:
                current_min = None
                for fnum in arr_1d:
                    if current_min is None:
                        current_min = fnum.copy()
                    else:
                        if self._executor.less_than(fnum, current_min):
                            current_min = fnum.copy()
                return current_min

            min_result_data = np.apply_along_axis(_min_fuzznums_along_axis, axis, self._data)

            if return_scalar:
                return min_result_data.item()
            return Fuzzarray(min_result_data, mtype=self.mtype, copy=False)

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Fuzznum, 'Fuzzarray']:
        """
        计算 Fuzzarray 沿指定轴的最大值。
        比较基于 Fuzznum 的默认比较逻辑 (Executor.greater_than/less_than)。

        Args:
            axis: 查找最大值的轴。None 表示对所有元素查找。

        Returns:
            Union[Fuzznum, Fuzzarray]: 如果 axis 为 None，返回单个 Fuzznum；
                                       否则返回一个降维的 Fuzzarray。
        """
        if self.size == 0:
            raise ValueError("Max of empty Fuzzarray is not defined.")

        if axis is None:
            max_fuzznum = None
            for fuzznum in self._data.flat:
                if max_fuzznum is None:
                    max_fuzznum = fuzznum.copy()
                else:
                    if self._executor.greater_than(fuzznum, max_fuzznum):
                        max_fuzznum = fuzznum.copy()
            return max_fuzznum
        else:
            if isinstance(axis, tuple):
                temp_arr = self
                for ax in sorted(axis, reverse=True):
                    temp_arr = temp_arr.max(axis=ax)
                return temp_arr

            if not isinstance(axis, int):
                raise TypeError(f"Axis must be an integer or a tuple of integers, got {type(axis)}.")
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"Axis {axis} is out of bounds for array with {self.ndim} dimensions.")

            result_shape = list(self.shape)
            result_shape.pop(axis)
            if not result_shape:
                return_scalar = True
                result_shape = (1,)
            else:
                return_scalar = False

            max_result_data = np.empty(tuple(result_shape), dtype=object)

            def _max_fuzznums_along_axis(arr_1d: np.ndarray) -> Fuzznum:
                current_max = None
                for fnum in arr_1d:
                    if current_max is None:
                        current_max = fnum.copy()
                    else:
                        if self._executor.greater_than(fnum, current_max):
                            current_max = fnum.copy()
                return current_max

            max_result_data = np.apply_along_axis(_max_fuzznums_along_axis, axis, self._data)

            if return_scalar:
                return max_result_data.item()
            return Fuzzarray(max_result_data, mtype=self.mtype, copy=False)

    # --- 数组操作 ---
    def tolist(self) -> List[Any]:
        """
        将 Fuzzarray 转换为嵌套的 Python 列表。

        Returns:
            List[Any]: 包含 Fuzznum 对象的嵌套列表。
        """
        return self._data.tolist()

    @classmethod
    def concatenate(cls, arrays: List['Fuzzarray'], axis: int = 0) -> 'Fuzzarray':
        """
        沿指定轴连接 Fuzzarray 序列。

        Args:
            arrays: 要连接的 Fuzzarray 实例列表。
            axis: 沿其连接数组的轴。

        Returns:
            Fuzzarray: 连接后的新 Fuzzarray。

        Raises:
            ValueError: 如果输入数组为空，或 mtype 不一致，或形状不兼容。
            TypeError: 如果输入数组中包含非 Fuzzarray 实例。
        """
        if not arrays:
            raise ValueError("Cannot concatenate empty list of Fuzzarrays.")

        first_mtype = arrays[0].mtype
        for arr in arrays:
            if not isinstance(arr, Fuzzarray):
                raise TypeError(f"All elements in 'arrays' must be Fuzzarray instances, found {type(arr)}.")
            if arr.mtype != first_mtype:
                raise ValueError("All Fuzzarrays to concatenate must have the same mtype.")

        # 提取底层的 numpy 数组
        np_arrays = [arr._data for arr in arrays]

        try:
            concatenated_data = np.concatenate(np_arrays, axis=axis)
        except ValueError as e:
            raise ValueError(f"Shape mismatch for concatenation along axis {axis}: {e}")

        return cls(concatenated_data, mtype=first_mtype, copy=False)

    @classmethod
    def stack(cls, arrays: List['Fuzzarray'], axis: int = 0) -> 'Fuzzarray':
        """
        沿新轴堆叠 Fuzzarray 序列。

        Args:
            arrays: 要堆叠的 Fuzzarray 实例列表。
            axis: 插入新轴的索引。

        Returns:
            Fuzzarray: 堆叠后的新 Fuzzarray。

        Raises:
            ValueError: 如果输入数组为空，或 mtype 不一致，或形状不兼容。
            TypeError: 如果输入数组中包含非 Fuzzarray 实例。
        """
        if not arrays:
            raise ValueError("Cannot stack empty list of Fuzzarrays.")

        first_mtype = arrays[0].mtype
        for arr in arrays:
            if not isinstance(arr, Fuzzarray):
                raise TypeError(f"All elements in 'arrays' must be Fuzzarray instances, found {type(arr)}.")
            if arr.mtype != first_mtype:
                raise ValueError("All Fuzzarrays to stack must have the same mtype.")
            if arr.shape != arrays[0].shape:
                raise ValueError("All Fuzzarrays to stack must have the same shape.")

        np_arrays = [arr._data for arr in arrays]

        try:
            stacked_data = np.stack(np_arrays, axis=axis)
        except ValueError as e:
            raise ValueError(f"Shape mismatch for stacking along axis {axis}: {e}")

        return cls(stacked_data, mtype=first_mtype, copy=False)

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
        # 使用 numpy.array2string 来格式化内部数据，它会自动处理大数组的省略号
        array_str = np.array2string(self._data, formatter={'all': lambda x: str(x)})
        # 移除 numpy.array2string 可能添加的 'array(' 和 ', dtype=object)'
        array_str = array_str.replace("array(", "").replace(", dtype=object)", "")

        return (f"Fuzzarray({array_str},\n"
                f"          shape={self.shape}, mtype='{self.mtype}')")

    def __str__(self) -> str:
        """
        用户友好的字符串表示。

        Returns:
            str: 简洁的 Fuzzarray 字符串表示。
        """
        # 对于 Fuzzarray，__str__ 可以与 __repr__ 相同以提供详细视图
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
