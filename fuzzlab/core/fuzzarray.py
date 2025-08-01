#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/31 13:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Optional, Iterator, Union, List, Any, Dict

import numpy as np

from fuzzlab.config import get_config
from fuzzlab.core.fuzznums import Fuzznum
from fuzzlab.core.operations import get_operation_registry, OperationMixin
from fuzzlab.core.triangular import OperationTNorm


class Fuzzarray:

    def __init__(self,
                 data: np.ndarray | list | tuple | Fuzznum,
                 mtype: Optional[str] = None,
                 shape: Optional[tuple[int, ...]] = None,
                 copy: bool = True):

        self._data: np.ndarray = self._process_input_data(data, shape, copy)
        self._mtype: str = self._validate_and_set_mtype(mtype)
        self._q: int = self._validate_and_set_q()

        # Operation Registry and Cache
        self._operation_registry = get_operation_registry()
        # self._op_cache: Dict[str, Optional[OperationMixin]] = {}

        self._delegated_methods: set[str] = set()
        self._delegated_attributes: set[str] = set()
        self._initialize_delegation()

    def _initialize_delegation(self) -> None:
        """
        By using a prototype Fuzznum instance to detect and register 
            proxyable methods and properties.

        This enables Fuzzarray to act as a facade for its 
            internal Fuzznum elements.
        """
        if self.size == 0:
            return

        # Create a temporary "prototype" Fuzznum instance 
        #   to discover available members
        try:
            prototype_fuzznum = Fuzznum(mtype=self.mtype, qrung=self.q)
        except (ValueError, TypeError):
            return

        # 从原型中获取策略和模板绑定的方法和属性
        strategy_methods = object.__getattribute__(prototype_fuzznum, '_bound_strategy_methods').keys()
        strategy_attrs = object.__getattribute__(prototype_fuzznum, '_bound_strategy_attributes')
        template_methods = object.__getattribute__(prototype_fuzznum, '_bound_template_methods').keys()
        template_attrs = object.__getattribute__(prototype_fuzznum, '_bound_template_attributes')

        self._delegated_methods.update(strategy_methods)
        self._delegated_methods.update(template_methods)
        self._delegated_attributes.update(strategy_attrs)
        self._delegated_attributes.update(template_attrs)

    def _process_input_data(self,
                            data: np.ndarray | list | tuple | Fuzznum,
                            shape: Optional[tuple[int, ...]],
                            copy: bool):

        if isinstance(data, Fuzznum):
            if shape is None:
                shape = ()
            arr = np.full(shape, data, dtype=object)
            if copy:
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

        if self._data.size == 0:
            return expected_mtype or get_config().DEFAULT_MTYPE

        # 检查第一个非 None Fuzznum 元素以确定 mtype
        first_fuzznum = None
        for item in self._data.flat:
            if isinstance(item, Fuzznum):
                first_fuzznum = item
                break

        if first_fuzznum is None:
            raise ValueError("Fuzzarray must contain at least one Fuzznum object.")

        detected_mtype = first_fuzznum.mtype

        if expected_mtype and detected_mtype != expected_mtype:
            raise ValueError(f"mtype mismatch: expected '{expected_mtype}', "
                             f"but found '{detected_mtype}' in initial Fuzznum.")

        for index, item in np.ndenumerate(self._data):
            if not isinstance(item, Fuzznum):
                raise TypeError(f"All elements in Fuzzarray must be Fuzznum objects, "
                                f"found {type(item)} at index {index}.")
            if item.mtype != detected_mtype:
                raise ValueError(f"All Fuzznums in Fuzzarray must have the same mtype. "
                                 f"Expected '{detected_mtype}', found '{item.mtype}' at index {index}.")
        return detected_mtype

    def _validate_and_set_q(self) -> int:
        if self._data.size == 0:
            return 1

        first_q = None
        for item in self._data.flat:
            if isinstance(item, Fuzznum):
                first_q = item.q
                break

        if first_q is None:
            raise ValueError("Fuzzarray must contain at least one Fuzznum object.")

        for index, item in np.ndenumerate(self._data):
            if not isinstance(item, Fuzznum):
                raise TypeError(f"All elements in Fuzzarray must be Fuzznum objects, "
                                f"found {type(item)} at index {index}.")
            if item.q != first_q:
                raise ValueError(f"All Fuzznums in Fuzzarray must have the same qrung. "
                                 f"Expected '{first_q}', found '{item.q}' at index {index}.")

        return first_q

    @staticmethod
    def _deep_copy_fuzznums(arr: np.ndarray) -> np.ndarray:
        result = np.empty_like(arr, dtype=object)
        for index, item in np.ndenumerate(arr):
            if isinstance(item, Fuzznum):
                result[index] = item.copy()
            else:
                result[index] = item
        return result

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array"""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array"""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Total number of elements in the array"""
        return self._data.size

    @property
    def mtype(self) -> str:
        """The fuzzy number type of all Fuzznum elements in the array"""
        return self._mtype

    @property
    def q(self) -> int:
        """The qrung of all Fuzznum elements in the array"""
        return self._q

    @property
    def data(self) -> np.ndarray:
        """The Fuzznum elements in the array"""
        return self._data

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically proxy Fuzznum's methods and properties to achieve vectorized operations.
        """
        if name in self._delegated_attributes:
            getter = np.vectorize(lambda fuzznum: getattr(fuzznum, name) if isinstance(fuzznum, Fuzznum) else None)
            return getter(self._data)

        if name in self._delegated_methods:
            # Return a wrapper function that executes vectorized operations when called.
            def vectorized_method(*args, **kwargs):
                # Define the action to be performed on each element
                op = lambda fuzznum: getattr(fuzznum, name)(*args, **kwargs) \
                    if isinstance(fuzznum, Fuzznum) else None

                # otypes=[object] ensures that the elements of the returned array are 
                #   Python objects, preventing numpy from attempting to convert them.
                vectorized_op = np.vectorize(op, otypes=[object])
                results = vectorized_op(self._data)

                if self.size > 0:
                    first_result = results.flat[0]
                    if isinstance(first_result, Fuzznum):
                        return Fuzzarray(results, mtype=first_result.mtype, copy=False)
                return results

            return vectorized_method

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'.")

    def __len__(self) -> int:
        """Returns the length of the first dimension of the array"""
        return len(self._data)

    def __iter__(self) -> Iterator[Union[Fuzznum, 'Fuzzarray']]:
        """
        Iterator that returns sub Fuzzarrays or individual
            Fuzznums along the first dimension.
        """
        for item in self._data:
            if isinstance(item, np.ndarray):
                yield Fuzzarray(item, mtype=self.mtype, copy=False)
            else:
                yield item

    # ======================== Indexing and slicing operations ========================

    def __getitem__(self, key) -> Union[Fuzznum, 'Fuzzarray']:
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return Fuzzarray(result, mtype=self._mtype, copy=False)
        else:
            return result

    def __setitem__(self, key, value: Union[Fuzznum, 'Fuzzarray']) -> None:
        if isinstance(value, Fuzznum):
            if value.mtype != self._mtype:
                raise ValueError(f"mtype mismatch: expected '{self._mtype}', "
                                 f"got '{value.mtype}' for assigned Fuzznum.")
            self._data[key] = value
        elif isinstance(value, Fuzzarray):
            if value._mtype != self._mtype:
                raise ValueError(f"mtype mismatch: expected '{self._mtype}', "
                                 f"got '{value._mtype}' for assigned Fuzzarray.")
            self._data[key] = value._data
        else:
            raise TypeError(f"Can only assign Fuzznum or Fuzzarray, got {type(value)}.")

    # ======================== Shape Operation Method ========================

    def reshape(self, *shape) -> 'Fuzzarray':

        new_data = self._data.reshape(shape)
        return Fuzzarray(new_data, mtype=self.mtype, copy=False)

    def flatten(self) -> 'Fuzzarray':
        return Fuzzarray(self._data.flatten(), mtype=self.mtype, copy=False)

    def transpose(self, *axes) -> 'Fuzzarray':
        if not axes:
            new_data = self._data.T
        else:
            new_data = self._data.transpose(*axes)
        return Fuzzarray(new_data, mtype=self.mtype, copy=False)

    @property
    def T(self) -> 'Fuzzarray':
        return self.transpose()

    def copy(self) -> 'Fuzzarray':
        return Fuzzarray(self._data, mtype=self.mtype, copy=True)

    # ======================== Core implementation of vectorized operations ========================

    def _execute_vectorized_op(self,
                               op_name: str,
                               other: Optional[Union['Fuzzarray', Fuzznum, np.ndarray, float, int]] = None,
                               **kwargs) -> Union['Fuzzarray', np.ndarray]:
        """
        General logic for executing vectorized operations.

        Find and invoke the corresponding operation implementation through OperationRegistry.
        """
        if self.size == 0:
            return self.copy()

        prototype_fuzznum = self.flatten()[0]
        strategy = prototype_fuzznum.get_strategy_instance()
        operation = strategy.get_operation_instance(op_name)

        if operation is None:
            raise NotImplementedError(
                f"Operation '{op_name}' is not supported for mtype '{self.mtype}'."
            )

        tnorm = OperationTNorm(
            norm_type=kwargs.get('tnorm_type'),
            q=self._q,
            **kwargs.get('tnorm_params', {}))

        # --- Dispatcher Logic ---
        # Check if the 'execute_fuzzarray' method on the concrete operation class
        # is different from the abstract base class's placeholder.
        has_specialized_implementation = (
                hasattr(operation, 'execute_fuzzarray') and
                getattr(operation, 'execute_fuzzarray').__func__ is not OperationMixin.execute_fuzzarray
        )

        # Check whether OperationMixin implements an efficient Fuzzarray version.
        if has_specialized_implementation:
            # Path 1: Specialized high-performance vectorization (no cache)
            return operation.execute_fuzzarray(self, other, tnorm, **kwargs)
        else:
            # Path 2: Default element-wise operation (with cache)
            return self._fallback_vectorized_op(operation, other, tnorm, **kwargs)

    def _fallback_vectorized_op(self,
                                operation: OperationMixin,
                                other: Optional[Union['Fuzzarray', Fuzznum, np.ndarray, float, int]],
                                tnorm: OperationTNorm,
                                **kwargs) -> Union['Fuzzarray', np.ndarray]:
        """
        Fallback vectorized implementation when OperationMixin does not implement execute_fuzzarray.
        """
        op_name = operation.get_operation_name()

        # Define operation function
        # op_func = None
        if op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
            op_func = lambda f1, f2: operation.execute_comparison(
                f1.get_strategy_instance(), f2.get_strategy_instance(), tnorm, **kwargs)
            vectorized_op = np.vectorize(op_func, otypes=[bool])

        elif isinstance(other, (Fuzzarray, Fuzznum)):
            op_func = lambda f1, f2: operation.execute_binary(
                f1.get_strategy_instance(), f2.get_strategy_instance(), tnorm, **kwargs)
            vectorized_op = np.vectorize(op_func, otypes=[object])

        elif isinstance(other, (int, float, np.ndarray)):
            op_func = lambda f1, op: operation.execute_unary_with_operand(
                f1.get_strategy_instance(), op, tnorm, **kwargs)
            vectorized_op = np.vectorize(op_func, otypes=[object])

        elif other is None:  # Pure unary operations
            op_func = lambda f1: operation.execute_unary(f1.get_strategy_instance(), tnorm, **kwargs)
            vectorized_op = np.vectorize(op_func, otypes=[object])
        else:
            raise TypeError(f"Unsupported operand type for fallback operation '{op_name}': {type(other)}")

        # Prepare operand
        if isinstance(other, Fuzzarray):
            result_data = vectorized_op(self._data, other._data)
        elif isinstance(other, Fuzznum):
            # Broadcast single Fuzznum
            other_arr = np.full(self.shape, other, dtype=object)
            result_data = vectorized_op(self._data, other_arr)
        else:  # int, float, np.ndarray, or None
            result_data = vectorized_op(self._data, other)

        # Packaging result
        if isinstance(result_data.flat[0], dict):  # If the returned value is a dictionary, convert it to Fuzznum.
            converter = np.vectorize(lambda d: Fuzznum(mtype=self.mtype, qrung=self.q).from_dict(d), otypes=[object])
            result_data = converter(result_data)
            return Fuzzarray(result_data, mtype=self.mtype, copy=False)
        elif isinstance(result_data.flat[0], bool):
            return result_data
        else:  # Already a Fuzznum object
            return Fuzzarray(result_data, mtype=self.mtype, copy=False)

    # ======================== Specific calculation method (operator overloading) ========================

    def __add__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        return self._execute_vectorized_op('add', other)

    def __sub__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        return self._execute_vectorized_op('sub', other)

    def __mul__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        return self._execute_vectorized_op('mul', other)

    def __truediv__(self, other: Union['Fuzzarray', Fuzznum]) -> 'Fuzzarray':
        return self._execute_vectorized_op('div', other)

    def __pow__(self, operand: Union[int, float, np.ndarray]) -> 'Fuzzarray':
        return self._execute_vectorized_op('pow', operand)

    # ======================== 比较运算 (返回 np.ndarray[bool]) ========================

    def __gt__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        return self._execute_vectorized_op('gt', other)

    def __lt__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        return self._execute_vectorized_op('lt', other)

    def __ge__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        return self._execute_vectorized_op('ge', other)

    def __le__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        return self._execute_vectorized_op('le', other)

    def __eq__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        return self._execute_vectorized_op('eq', other)

    def __ne__(self, other: Union['Fuzzarray', Fuzznum]) -> np.ndarray:
        return self._execute_vectorized_op('ne', other)

    # ======================== 命名运算方法 ========================

    def times(self, operand: Union[int, float, np.ndarray], **kwargs) -> 'Fuzzarray':
        return self._execute_vectorized_op('tim', operand, **kwargs)

    def intersection(self, other: 'Fuzzarray', **kwargs) -> 'Fuzzarray':
        return self._execute_vectorized_op('intersection', other, **kwargs)

    def union(self, other: 'Fuzzarray', **kwargs) -> 'Fuzzarray':
        return self._execute_vectorized_op('union', other, **kwargs)

    def complement(self, **kwargs) -> 'Fuzzarray':
        return self._execute_vectorized_op('complement', None, **kwargs)

    def tolist(self) -> List[Any]:
        return self._data.tolist()

    @classmethod
    def concatenate(cls, arrays: List['Fuzzarray'], axis: int = 0) -> 'Fuzzarray':
        """
        Concatenate Fuzzarray sequences along a specified axis.

        Args:
            arrays: List of Fuzzarray instances to connect to.
            axis: Along the axis of its connected array.

        Returns:
            Fuzzarray: The new Fuzzarray after connection.

        Raises:
            ValueError: If the input array is empty, or the mtype or qrung is
                inconsistent, or the shapes are incompatible.
            TypeError: If the input array contains non-Fuzzarray instances.
        """
        if not arrays:
            raise ValueError("Cannot concatenate empty list of Fuzzarrays.")

        first_mtype = arrays[0].mtype
        first_q = arrays[0].q
        for arr in arrays:
            if not isinstance(arr, Fuzzarray):
                raise TypeError(f"All elements in 'arrays' must be Fuzzarray instances, found {type(arr)}.")
            if arr.mtype != first_mtype or arr.q != first_q:
                raise ValueError("All Fuzzarrays to concatenate must have the same mtype and qrung.")

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
        Stack Fuzzarray sequences along a new axis.

        Args:
            arrays: List of Fuzzarray instances to be stacked.
            axis: Index for inserting a new axis.

        Returns:
            Fuzzarray: The new Fuzzarray after stacking.

        Raises:
            ValueError: If the input array is empty, or the mtype and qrung is inconsistent,
                or the shapes are incompatible.
            TypeError: If the input array contains non-Fuzzarray instances.
        """
        if not arrays:
            raise ValueError("Cannot stack empty list of Fuzzarrays.")

        first_mtype = arrays[0].mtype
        first_q = arrays[0].q
        for arr in arrays:
            if not isinstance(arr, Fuzzarray):
                raise TypeError(f"All elements in 'arrays' must be Fuzzarray instances, found {type(arr)}.")
            if arr.mtype != first_mtype or arr.q != first_q:
                raise ValueError("All Fuzzarrays to stack must have the same mtype and qrung.")
            if arr.shape != arrays[0].shape:
                raise ValueError("All Fuzzarrays to stack must have the same shape.")

        np_arrays = [arr._data for arr in arrays]

        try:
            stacked_data = np.stack(np_arrays, axis=axis)
        except ValueError as e:
            raise ValueError(f"Shape mismatch for stacking along axis {axis}: {e}")

        return cls(stacked_data, mtype=first_mtype, copy=False)

    # ======================== String representation ========================

    def __repr__(self):
        # Use numpy.array2string to format the internal data,
        #   which automatically handles the ellipsis for large arrays.
        array_str = np.array2string(self._data, formatter={'all': lambda x: str(x)})

        # Remove 'array(' and ', dtype=object)' that may be added by numpy.array2string
        array_str = array_str.replace("array(", "").replace(", dtype=object)", "")

        return f"Fuzzarray(\n{array_str}, shape={self.shape}, mtype='{self.mtype}')"

    def __str__(self) -> str:
        array_str = np.array2string(self._data, formatter={'all': lambda x: str(x)})
        array_str = array_str.replace("array(", "").replace(", dtype=object)", "")
        return f"{array_str}"


def fuzzarray(data: np.ndarray | list | tuple | Fuzznum,
              mtype: Optional[str] = None,
              shape: Optional[tuple[int, ...]] = None,
              copy: bool = True) -> Fuzzarray:
    """
    Convenient function to create Fuzzarray

    Args:
        data: Input data.
        mtype: Fuzzy number type.
        shape: Target shape.
        copy: Whether to copy data.

    Returns:
        Fuzzarray: Created array of fuzzy numbers.
    """
    return Fuzzarray(data, mtype=mtype, shape=shape, copy=copy)
