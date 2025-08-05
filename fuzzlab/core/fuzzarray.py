#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/31 13:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module defines the `Fuzzarray` class, a specialized n-dimensional array
designed to hold and operate on `Fuzznum` objects.

`Fuzzarray` provides a NumPy-like interface for handling collections of fuzzy
numbers, enabling vectorized operations and maintaining consistency of fuzzy
number types (`mtype`) and q-rungs (`q`) across all elements. It delegates
operations to the underlying `Fuzznum` objects and supports broadcasting
and common array manipulations.
"""
from typing import Optional, Iterator, Union, List, Any

import numpy as np

from fuzzlab.config import get_config
from fuzzlab.core.fuzznums import Fuzznum
from fuzzlab.core.ops import get_operation_registry, OperationMixin
from fuzzlab.core.triangular import OperationTNorm


class Fuzzarray:
    """
    A specialized n-dimensional array for `Fuzznum` objects, providing NumPy-like
    functionality for vectorized fuzzy number operations.

    All `Fuzznum` elements within a `Fuzzarray` must share the same `mtype`
    (membership type) and `q` (q-rung value). `Fuzzarray` supports various
    initialization methods, indexing, slicing, shape manipulation, and
    overloads standard arithmetic and comparison operators for element-wise
    fuzzy number computations.

    Attributes:
        _data (np.ndarray): The internal NumPy array storing `Fuzznum` objects.
                            Its `dtype` is always `object`.
        _mtype (str): The common membership type of all `Fuzznum` elements in the array.
        _q (int): The common q-rung value of all `Fuzznum` elements in the array.
        _operation_registry: Reference to the global operation registry.
        _delegated_methods (set[str]): A set of method names from `Fuzznum` that
                                       are dynamically proxied for vectorized execution.
        _delegated_attributes (set[str]): A set of attribute names from `Fuzznum` that
                                          are dynamically proxied for vectorized access.
    """

    __array_priority__ = 1.0
    # This attribute tells NumPy that Fuzzarray objects should have higher
    # priority in mixed-type operations, ensuring that Fuzzarray's
    # operator overloads are preferred over NumPy's default behavior
    # when a Fuzzarray is involved in an operation with a NumPy array.

    def __init__(self,
                 data: np.ndarray | list | tuple | Fuzznum,
                 mtype: Optional[str] = None,
                 shape: Optional[tuple[int, ...]] = None,
                 copy: bool = True):
        """
        Initializes a Fuzzarray instance.

        Args:
            data (np.ndarray | list | tuple | Fuzznum): The input data to populate the Fuzzarray.
                                                        Can be a NumPy array of Fuzznum objects,
                                                        a list/tuple of Fuzznum objects, or a single Fuzznum.
            mtype (Optional[str]): The expected membership type for all Fuzznum elements.
                                   If provided, it must match the mtype of the Fuzznum(s) in `data`.
                                   If None, the mtype is inferred from the first Fuzznum in `data`.
            shape (Optional[tuple[int, ...]]): The desired shape of the Fuzzarray.
                                               If `data` is a single Fuzznum, this defines the
                                               shape of the array filled with that Fuzznum.
                                               If `data` is an array-like, it will be reshaped.
            copy (bool): If True, a deep copy of the input Fuzznum objects is made.
                         If False, references to the original Fuzznum objects are used.
                         Defaults to True.

        Raises:
            TypeError: If `data` is of an unsupported type or contains non-Fuzznum elements.
            ValueError: If `mtype` or `q` values are inconsistent among elements,
                        or if `data` is empty and `mtype` cannot be inferred.
        """
        # Process the input data to create the internal NumPy array of Fuzznum objects.
        self._data: np.ndarray = self._process_input_data(data, shape, copy)
        # Validate and set the common membership type for all Fuzznum elements.
        self._mtype: str = self._validate_and_set_mtype(mtype)
        # Validate and set the common q-rung value for all Fuzznum elements.
        self._q: int = self._validate_and_set_q()

        # Get the global operation registry for dispatching operations.
        self._operation_registry = get_operation_registry()
        # self._op_cache: Dict[str, Optional[OperationMixin]] = {} # Currently unused, operations are dispatched directly.

        # Sets to store names of methods and attributes that can be delegated
        # from Fuzznum objects for vectorized operations.
        self._delegated_methods: set[str] = set()
        self._delegated_attributes: set[str] = set()
        # Initialize delegation by discovering proxyable members from a prototype Fuzznum.
        self._initialize_delegation()

    def _initialize_delegation(self) -> None:
        """
        Initializes the sets of delegated methods and attributes by using a
        prototype `Fuzznum` instance to detect and register proxyable methods
        and properties.

        This enables `Fuzzarray` to act as a facade for its internal `Fuzznum`
        elements, allowing calls like `fuzzarray_instance.some_fuzznum_method()`
        to be vectorized across all elements.
        """
        # If the Fuzzarray is empty, there are no Fuzznum elements to delegate from.
        if self.size == 0:
            return

        # Create a temporary "prototype" Fuzznum instance to discover available members.
        # This prototype is created using the Fuzzarray's determined mtype and q.
        try:
            prototype_fuzznum = Fuzznum(mtype=self.mtype, qrung=self.q)
        except (ValueError, TypeError):
            # If a prototype cannot be created (e.g., invalid mtype/q), delegation is skipped.
            return

        # Retrieve the names of methods and attributes that are bound to the
        # strategy and template instances of the prototype Fuzznum.
        # These are the members that Fuzzarray can delegate calls to.
        strategy_methods = object.__getattribute__(prototype_fuzznum, '_bound_strategy_methods').keys()
        strategy_attrs = object.__getattribute__(prototype_fuzznum, '_bound_strategy_attributes')
        template_methods = object.__getattribute__(prototype_fuzznum, '_bound_template_methods').keys()
        template_attrs = object.__getattribute__(prototype_fuzznum, '_bound_template_attributes')

        # Update the internal sets of delegated methods and attributes.
        self._delegated_methods.update(strategy_methods)
        self._delegated_methods.update(template_methods)
        self._delegated_attributes.update(strategy_attrs)
        self._delegated_attributes.update(template_attrs)

    def _process_input_data(self,
                            data: np.ndarray | list | tuple | Fuzznum,
                            shape: Optional[tuple[int, ...]],
                            copy: bool) -> np.ndarray:
        """
        Processes the input data to create the internal NumPy array of Fuzznum objects.

        Handles various input types: single Fuzznum, list/tuple of Fuzznums,
        or an existing NumPy array of Fuzznums. Ensures the array has `dtype=object`.

        Args:
            data (np.ndarray | list | tuple | Fuzznum): The raw input data.
            shape (Optional[tuple[int, ...]]): The desired shape for the resulting array.
            copy (bool): Whether to deep copy Fuzznum elements.

        Returns:
            np.ndarray: A NumPy array with `dtype=object` containing Fuzznum instances.

        Raises:
            TypeError: If the input data type is unsupported or if a NumPy array
                       does not have `dtype=object`.
        """
        if isinstance(data, Fuzznum):
            # If input is a single Fuzznum, create a NumPy array filled with it.
            if shape is None:
                shape = ()  # Default to a scalar array if no shape is given.
            arr = np.full(shape, data, dtype=object)
            if copy:
                # If copy is True, deep copy the single Fuzznum into each position.
                arr = self._deep_copy_fuzznums(arr)
            return arr
        elif isinstance(data, (list, tuple)):
            # If input is a list or tuple, convert it to a NumPy array.
            arr = np.array(data, dtype=object)
            if copy:
                # If copy is True, deep copy each Fuzznum in the array.
                arr = self._deep_copy_fuzznums(arr)
            if shape is not None:
                # Reshape the array if a target shape is provided.
                arr = arr.reshape(shape)
            return arr
        elif isinstance(data, np.ndarray):
            # If input is already a NumPy array.
            if data.dtype != object:
                # Ensure it's an object array to hold Fuzznum instances.
                raise TypeError("Input numpy.ndarray must have dtype=object for Fuzznum elements.")
            arr = data.copy() if copy else data  # Copy if requested, otherwise use reference.
            if shape is not None:
                # Reshape the array if a target shape is provided.
                arr = arr.reshape(shape)
            return arr
        else:
            # Raise an error for unsupported input data types.
            raise TypeError(f"Unsupported data type for Fuzzarray: {type(data)}")

    def _validate_and_set_mtype(self, expected_mtype: Optional[str]) -> str:
        """
        Validates that all Fuzznum elements in the array have a consistent `mtype`
        and sets the `_mtype` attribute of the Fuzzarray.

        Args:
            expected_mtype (Optional[str]): An optional `mtype` to check against.

        Returns:
            str: The validated common `mtype` of the Fuzzarray.

        Raises:
            ValueError: If `mtype` values are inconsistent or cannot be inferred.
            TypeError: If non-Fuzznum objects are found in the array.
        """
        # If the array is empty, return the expected mtype or the default mtype from config.
        if self._data.size == 0:
            return expected_mtype or get_config().DEFAULT_MTYPE

        # Find the mtype from the first Fuzznum element.
        first_fuzznum = None
        for item in self._data.flat:
            if isinstance(item, Fuzznum):
                first_fuzznum = item
                break

        if first_fuzznum is None:
            # If no Fuzznum is found (e.g., array contains only None or other non-Fuzznum objects).
            raise ValueError("Fuzzarray must contain at least one Fuzznum object.")

        detected_mtype = first_fuzznum.mtype

        # If an expected mtype was provided, ensure it matches the detected mtype.
        if expected_mtype and detected_mtype != expected_mtype:
            raise ValueError(f"mtype mismatch: expected '{expected_mtype}', "
                             f"but found '{detected_mtype}' in initial Fuzznum.")

        # Iterate through all elements to ensure consistency.
        for index, item in np.ndenumerate(self._data):
            if not isinstance(item, Fuzznum):
                raise TypeError(f"All elements in Fuzzarray must be Fuzznum objects, "
                                f"found {type(item)} at index {index}.")
            if item.mtype != detected_mtype:
                raise ValueError(f"All Fuzznums in Fuzzarray must have the same mtype. "
                                 f"Expected '{detected_mtype}', found '{item.mtype}' at index {index}.")
        return detected_mtype

    def _validate_and_set_q(self) -> int:
        """
        Validates that all Fuzznum elements in the array have a consistent `q` (q-rung)
        and sets the `_q` attribute of the Fuzzarray.

        Returns:
            int: The validated common `q` value of the Fuzzarray.

        Raises:
            ValueError: If `q` values are inconsistent or cannot be inferred.
            TypeError: If non-Fuzznum objects are found in the array.
        """
        # If the array is empty, return a default q-rung (e.g., 1).
        if self._data.size == 0:
            return 1

        # Find the q-rung from the first Fuzznum element.
        first_q = None
        for item in self._data.flat:
            if isinstance(item, Fuzznum):
                first_q = item.q
                break

        if first_q is None:
            # If no Fuzznum is found.
            raise ValueError("Fuzzarray must contain at least one Fuzznum object.")

        # Iterate through all elements to ensure consistency.
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
        """
        Performs a deep copy of Fuzznum objects within a NumPy array.

        This is necessary because `np.array(..., dtype=object)` or `np.full(..., dtype=object)`
        only creates shallow copies of Python objects. For mutable `Fuzznum` objects,
        a deep copy ensures independent instances.

        Args:
            arr (np.ndarray): The input NumPy array containing Fuzznum objects.

        Returns:
            np.ndarray: A new NumPy array with deep copies of Fuzznum objects.
        """
        result = np.empty_like(arr, dtype=object)
        for index, item in np.ndenumerate(arr):
            if isinstance(item, Fuzznum):
                result[index] = item.copy()  # Assuming Fuzznum has a .copy() method.
            else:
                result[index] = item  # Copy other types directly (e.g., None, scalars).
        return result

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the Fuzzarray.

        This is a read-only property that delegates to the shape of the
        underlying NumPy array.

        Returns:
            tuple[int, ...]: A tuple representing the dimensions of the array.
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the Fuzzarray.

        This is a read-only property that delegates to the ndim of the
        underlying NumPy array.

        Returns:
            int: The number of array dimensions.
        """
        return self._data.ndim

    @property
    def size(self) -> int:
        """
        Returns the total number of elements in the Fuzzarray.

        This is a read-only property that delegates to the size of the
        underlying NumPy array.

        Returns:
            int: The total number of elements.
        """
        return self._data.size

    @property
    def mtype(self) -> str:
        """
        Returns the common fuzzy number type (`mtype`) of all Fuzznum elements in the array.

        Returns:
            str: The membership type string.
        """
        return self._mtype

    @property
    def q(self) -> int:
        """
        Returns the common q-rung value (`q`) of all Fuzznum elements in the array.

        Returns:
            int: The q-rung value.
        """
        return self._q

    @property
    def data(self) -> np.ndarray:
        """
        Returns the underlying NumPy array containing the Fuzznum elements.

        Direct access to `_data` is provided for advanced use cases, but
        modifications to this array should be done carefully to maintain
        Fuzzarray's consistency guarantees (e.g., mtype, q).

        Returns:
            np.ndarray: The internal NumPy array.
        """
        return self._data

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically proxies `Fuzznum`'s methods and properties to achieve
        vectorized operations.

        If an attribute `name` is not found directly on the `Fuzzarray` instance,
        this method checks if it's a delegated attribute or method from `Fuzznum`.
        If it is, it returns a vectorized version of that attribute/method.

        Args:
            name (str): The name of the attribute or method being accessed.

        Returns:
            Any: The vectorized attribute value (a NumPy array) or a callable
                 that performs the vectorized method call.

        Raises:
            AttributeError: If the attribute `name` is not found on `Fuzzarray`
                            and is not a delegated `Fuzznum` member.
        """
        if name in self._delegated_attributes:
            # If the name is a delegated attribute (e.g., `fuzznum.md`),
            # create a vectorized getter that applies `getattr` to each Fuzznum element.
            getter = np.vectorize(lambda fuzznum: getattr(fuzznum, name) if isinstance(fuzznum, Fuzznum) else None)
            return getter(self._data)

        if name in self._delegated_methods:
            # If the name is a delegated method (e.g., `fuzznum.complement()`),
            # return a wrapper function that executes the method in a vectorized manner when called.
            def vectorized_method(*args, **kwargs):
                # Define the action to be performed on each element:
                # call the method `name` on the Fuzznum with the given args/kwargs.
                op = lambda fuzznum: getattr(fuzznum, name)(*args, **kwargs) \
                    if isinstance(fuzznum, Fuzznum) else None

                # Create a NumPy vectorized function. `otypes=[object]` is crucial
                # to ensure that the results (which are Fuzznum objects or other Python objects)
                # are kept as Python objects in the resulting array, preventing NumPy
                # from trying to convert them to a primitive dtype.
                vectorized_op = np.vectorize(op, otypes=[object])
                results = vectorized_op(self._data)

                # If the Fuzzarray is not empty and the first result is a Fuzznum,
                # wrap the results in a new Fuzzarray. Otherwise, return the raw NumPy array.
                if self.size > 0:
                    first_result = results.flat[0]
                    if isinstance(first_result, Fuzznum):
                        # Create a new Fuzzarray from the results, preserving mtype.
                        return Fuzzarray(results, mtype=first_result.mtype, copy=False)
                return results

            return vectorized_method

        # If the attribute is neither directly on Fuzzarray nor a delegated Fuzznum member, raise AttributeError.
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'.")

    def __len__(self) -> int:
        """
        Returns the length of the first dimension of the array.

        This allows `len(fuzzarray_instance)` to work like `len(numpy_array)`.

        Returns:
            int: The size of the first dimension.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[Union[Fuzznum, 'Fuzzarray']]:
        """
        Provides an iterator that yields sub-Fuzzarrays or individual Fuzznums
        along the first dimension.

        This allows iteration over the Fuzzarray, similar to iterating over a NumPy array.

        Yields:
            Union[Fuzznum, 'Fuzzarray']: A Fuzznum object if the item is a scalar,
                                         or a Fuzzarray if the item is a sub-array.
        """
        for item in self._data:
            if isinstance(item, np.ndarray):
                # If the item is a NumPy array (i.e., a sub-array), wrap it in a new Fuzzarray.
                yield Fuzzarray(item, mtype=self.mtype, copy=False)
            else:
                # Otherwise, yield the Fuzznum object directly.
                yield item

    # ======================== Indexing and slicing operations ========================

    def __getitem__(self, key) -> Union[Fuzznum, 'Fuzzarray']:
        """
        Implements indexing and slicing for Fuzzarray.

        Allows accessing individual Fuzznum elements or sub-arrays using NumPy-like
        indexing (e.g., `fuzzarray[0]`, `fuzzarray[1, 2]`, `fuzzarray[:, 0]`).

        Args:
            key: The index or slice object.

        Returns:
            Union[Fuzznum, 'Fuzzarray']: A single Fuzznum if a scalar element is accessed,
                                         or a new Fuzzarray if a slice or sub-array is accessed.
        """
        result = self._data[key]
        if isinstance(result, np.ndarray):
            # If the result of indexing is a NumPy array, wrap it in a new Fuzzarray.
            return Fuzzarray(result, mtype=self._mtype, copy=False)
        else:
            # Otherwise, it's a single Fuzznum element.
            return result

    def __setitem__(self, key, value: Union[Fuzznum, 'Fuzzarray']) -> None:
        """
        Implements item assignment for Fuzzarray.

        Allows setting individual Fuzznum elements or sub-arrays. Ensures that
        assigned Fuzznum(s) have a matching `mtype`.

        Args:
            key: The index or slice object.
            value (Union[Fuzznum, 'Fuzzarray']): The Fuzznum or Fuzzarray to assign.

        Raises:
            ValueError: If the assigned Fuzznum or Fuzzarray has a different `mtype`.
            TypeError: If the assigned value is not a Fuzznum or Fuzzarray.
        """
        if isinstance(value, Fuzznum):
            # If assigning a single Fuzznum, check its mtype.
            if value.mtype != self._mtype:
                raise ValueError(f"mtype mismatch: expected '{self._mtype}', "
                                 f"got '{value.mtype}' for assigned Fuzznum.")
            self._data[key] = value
        elif isinstance(value, Fuzzarray):
            # If assigning a Fuzzarray, check its mtype.
            if value._mtype != self._mtype:
                raise ValueError(f"mtype mismatch: expected '{self._mtype}', "
                                 f"got '{value._mtype}' for assigned Fuzzarray.")
            self._data[key] = value._data  # Assign the underlying NumPy array.
        else:
            # Raise an error for unsupported assignment types.
            raise TypeError(f"Can only assign Fuzznum or Fuzzarray, got {type(value)}.")

    # ======================== Shape Operation Method ========================

    def reshape(self, *shape) -> 'Fuzzarray':
        """
        Returns a new Fuzzarray with a different shape without changing its data.

        Args:
            *shape: The new shape, as a tuple or individual integers.

        Returns:
            Fuzzarray: A new Fuzzarray with the specified shape.
        """
        new_data = self._data.reshape(shape)
        return Fuzzarray(new_data, mtype=self.mtype, copy=False)

    def flatten(self) -> 'Fuzzarray':
        """
        Returns a copy of the array collapsed into one dimension.

        Returns:
            Fuzzarray: A new 1-D Fuzzarray.
        """
        return Fuzzarray(self._data.flatten(), mtype=self.mtype, copy=False)

    def transpose(self, *axes) -> 'Fuzzarray':
        """
        Returns a new Fuzzarray with axes transposed.

        Args:
            *axes: Optional. A tuple or list of integers, specifying the new order of axes.
                   If not provided, the array is simply reversed.

        Returns:
            Fuzzarray: A new Fuzzarray with transposed axes.
        """
        if not axes:
            new_data = self._data.T  # NumPy's transpose property.
        else:
            new_data = self._data.transpose(*axes)
        return Fuzzarray(new_data, mtype=self.mtype, copy=False)

    @property
    def T(self) -> 'Fuzzarray':
        """
        Returns the transposed Fuzzarray (shortcut for `transpose()`).

        Returns:
            Fuzzarray: The transposed Fuzzarray.
        """
        return self.transpose()

    def copy(self) -> 'Fuzzarray':
        """
        Returns a deep copy of the Fuzzarray.

        Returns:
            Fuzzarray: A new Fuzzarray instance with independent copies of all Fuzznum elements.
        """
        return Fuzzarray(self._data, mtype=self.mtype, copy=True)

    # ======================== Core implementation of vectorized operations ========================

    def execute_vectorized_op(self,
                              op_name: str,
                              other: Optional[
                                   Union['Fuzzarray', Fuzznum, np.ndarray, float, int]]
                              = None) -> Union['Fuzzarray', np.ndarray]:
        """
        Executes a specified operation in a vectorized manner across the Fuzzarray.

        This method serves as the central dispatcher for all element-wise operations.
        It attempts to use a specialized, high-performance vectorized implementation
        if available in the `OperationMixin` subclass. Otherwise, it falls back to
        a generic element-wise application using `np.vectorize`.

        Args:
            op_name (str): The name of the operation to perform (e.g., 'add', 'complement').
            other (Optional[Union[Fuzzarray, Fuzznum, np.ndarray, float, int]]): The second operand
                                                                                 for binary operations.
                                                                                 Can be another Fuzzarray,
                                                                                 a single Fuzznum, a NumPy array,
                                                                                 or a scalar (int/float).
                                                                                 None for unary operations.

        Returns:
            Union[Fuzzarray, np.ndarray]: A new Fuzzarray containing the results of the operation,
                                          or a NumPy array (e.g., for comparison operations returning booleans).

        Raises:
            NotImplementedError: If the operation is not supported for the Fuzzarray's `mtype`.
            TypeError: If operand types are incompatible.
        """
        # If the Fuzzarray is empty, return a copy of itself (or an empty array for comparisons).
        if self.size == 0:
            # For comparison operations, an empty array of booleans is appropriate.
            if op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
                return np.array([], dtype=bool)
            return self.copy()

        # Retrieve the specific operation handler from the global registry based on op_name and mtype.
        operation = self._operation_registry.get_operation(op_name, self.mtype)

        if operation is None:
            raise NotImplementedError(
                f"Operation '{op_name}' is not supported for mtype '{self.mtype}'."
            )

        # Dynamically create a t-norm object using Fuzzarray's own q value and default config.
        norm_type, norm_params = self._operation_registry.get_default_t_norm_config()
        tnorm = OperationTNorm(norm_type=norm_type, q=self.q, **norm_params)

        # --- Dispatcher Logic ---
        # Check if the concrete `OperationMixin` subclass has overridden the
        # `execute_fuzzarray_op` method with a specialized implementation.
        # This is done by comparing the method's function object to the one
        # from the base `OperationMixin` class.
        has_specialized_implementation = (
                hasattr(operation, 'execute_fuzzarray_op') and
                getattr(operation, 'execute_fuzzarray_op').__func__ is not OperationMixin.execute_fuzzarray_op
        )

        # If a specialized, high-performance vectorized implementation exists:
        if has_specialized_implementation:
            # Path 1: Use the specialized implementation. This path typically
            # does not involve `np.vectorize` and might be optimized for speed.
            # Caching is usually handled within the specialized implementation if needed.
            return operation.execute_fuzzarray_op(self, other, tnorm)
        else:
            # Path 2: Fallback to a generic element-wise operation.
            # This uses `np.vectorize` to apply the Fuzznum-level operation
            # to each element of the Fuzzarray. Caching is handled at the Fuzznum level.
            return self._fallback_vectorized_op(operation, other, tnorm)

    def _fallback_vectorized_op(self,
                                operation: OperationMixin,
                                other: Optional[
                                    Union['Fuzzarray', Fuzznum, np.ndarray, float, int]],
                                tnorm: OperationTNorm) -> Union['Fuzzarray', np.ndarray]:
        """
        Provides a fallback vectorized implementation when the `OperationMixin`
        subclass does not implement a specialized `execute_fuzzarray_op`.

        This method uses `np.vectorize` to apply the corresponding Fuzznum-level
        operation (binary, unary with operand, pure unary, or comparison) to
        each element of the Fuzzarray.

        Args:
            operation (OperationMixin): The concrete operation handler.
            other (Optional[Union[Fuzzarray, Fuzznum, np.ndarray, float, int]]): The second operand.
            tnorm (OperationTNorm): The t-norm object to use for the operation.

        Returns:
            Union[Fuzzarray, np.ndarray]: A new Fuzzarray or a NumPy array of results.

        Raises:
            TypeError: If the operand type is unsupported for the given operation.
        """
        op_name = operation.get_operation_name()

        # Define the element-wise operation function (`op_func`) and the `np.vectorize` wrapper.
        # The `otypes=[object]` or `otypes=[bool]` is crucial to ensure NumPy handles
        # the return types correctly without trying to convert Fuzznum objects to primitive types.
        if op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
            # Comparison operations return boolean values.
            op_func = lambda f1, f2: operation.execute_comparison_op(
                f1.get_strategy_instance(), f2.get_strategy_instance(), tnorm)
            vectorized_op = np.vectorize(op_func, otypes=[bool])

        elif isinstance(other, (Fuzzarray, Fuzznum)):
            # Binary operations between two Fuzznums (or Fuzzarray and Fuzznum).
            op_func = lambda f1, f2: operation.execute_binary_op(
                f1.get_strategy_instance(), f2.get_strategy_instance(), tnorm)
            vectorized_op = np.vectorize(op_func, otypes=[object])

        elif isinstance(other, (int, float, np.ndarray)):
            # Unary operations with a scalar or NumPy array operand.
            op_func = lambda f1, op: operation.execute_unary_op_operand(
                f1.get_strategy_instance(), op, tnorm)
            vectorized_op = np.vectorize(op_func, otypes=[object])

        elif other is None:  # Pure unary operations (e.g., complement)
            op_func = lambda f1: operation.execute_unary_op_pure(f1.get_strategy_instance(), tnorm)
            vectorized_op = np.vectorize(op_func, otypes=[object])
        else:
            raise TypeError(f"Unsupported operand type for fallback operation '{op_name}': {type(other)}")

        # Prepare the `other` operand for vectorized application.
        if isinstance(other, Fuzzarray):
            other_data = other._data  # Use the underlying NumPy array.
        elif isinstance(other, Fuzznum):
            # If `other` is a single Fuzznum, broadcast it to match the shape of `self`.
            other_data = np.full(self.shape, other, dtype=object)
        else:   # int, float, np.ndarray, or None
            other_data = other  # Use directly.

        # Execute the vectorized operation.
        if other_data is not None:
            result_data = vectorized_op(self._data, other_data)
        else:
            result_data = vectorized_op(self._data)

        # Packaging the result.
        if result_data.size == 0:
            # If the result array is empty, return an empty array of appropriate type.
            return np.array([]) if op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne'] else self.copy()

        if op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
            # If it's a comparison operation, the result is already a boolean NumPy array.
            return result_data

        # For other operations, the result elements might be dictionaries (from strategy execution)
        # or already Fuzznum objects.
        first_result = result_data.flat[0]

        if isinstance(first_result, dict):
            # If the returned value from the Fuzznum operation is a dictionary (e.g., from `execute_operation`),
            # convert each dictionary back into a Fuzznum object.
            convert_func = np.vectorize(
                lambda d: Fuzznum(mtype=self.mtype, qrung=self.q).create(**d),
                otypes=[object])
            result_data = convert_func(result_data)
            return Fuzzarray(result_data, mtype=self.mtype, copy=False)
        else:
            # If the result elements are already Fuzznum objects (unlikely for `execute_operation`
            # but possible for other Fuzznum methods), just wrap the array in a new Fuzzarray.
            return Fuzzarray(result_data, mtype=self.mtype, copy=False)

    # ======================== Specific calculation method (operator overloading) ========================
    # These methods overload standard Python operators to enable intuitive
    # arithmetic and comparison operations directly on Fuzzarray objects.
    # They delegate the actual operation logic to the `dispatcher.operate` function,
    # which handles type dispatching and calls `execute_vectorized_op` internally.

    def __add__(self, other: Any) -> 'Fuzzarray':
        """Implements the addition operator (+)."""
        from .dispatcher import operate
        return operate('add', self, other)

    def __radd__(self, other: Any) -> 'Fuzzarray':
        """Implements the reverse addition operator (+)."""
        from .dispatcher import operate
        return operate('add', self, other)

    def __sub__(self, other: Any) -> 'Fuzzarray':
        """Implements the subtraction operator (-)."""
        from .dispatcher import operate
        return operate('sub', self, other)

    def __mul__(self, other: Any) -> 'Fuzzarray':
        """Implements the multiplication operator (*)."""
        from .dispatcher import operate
        return operate('mul', self, other)

    def __rmul__(self, other: Any) -> 'Fuzzarray':
        """Implements the reverse multiplication operator (*)."""
        from .dispatcher import operate
        return operate('mul', self, other)

    def __truediv__(self, other: Any) -> 'Fuzzarray':
        """Implements the true division operator (/)."""
        from .dispatcher import operate
        return operate('div', self, other)

    def __pow__(self, power: Any, modulo: Optional[Any] = None) -> 'Fuzzarray':
        """Implements the power operator (**)."""
        from .dispatcher import operate
        return operate('pow', self, power)

    def __gt__(self, other: Any) -> np.ndarray:
        """Implements the greater than operator (>). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('gt', self, other)

    def __lt__(self, other: Any) -> np.ndarray:
        """Implements the less than operator (<). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('lt', self, other)

    def __ge__(self, other: Any) -> np.ndarray:
        """Implements the greater than or equal to operator (>=). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('ge', self, other)

    def __le__(self, other: Any) -> np.ndarray:
        """Implements the less than or equal to operator (<=). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('le', self, other)

    def __eq__(self, other: Any) -> np.ndarray:
        """Implements the equality operator (==). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('eq', self, other)

    def __ne__(self, other: Any) -> np.ndarray:
        """Implements the inequality operator (!=). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('ne', self, other)

    def __and__(self, other):
        """Overloads the and operator (&).
            intersection operation.
        """
        from .dispatcher import operate
        return operate('intersection', self, other)

    def __or__(self, other):
        from .dispatcher import operate
        return operate('union', self, other)

    def __invert__(self, other=None):
        """Overloads the invert operator (~).
            Complement operation.
        """
        from .dispatcher import operate
        return operate('complement', self, other)

    def __lshift__(self, other):
        """Overloads the left shift operator (<<).
            Denotes the left implication operation: self <- other
        """
        from .dispatcher import operate
        return operate('implication', other, self)

    def __rshift__(self, other):
        """Overloads the shift operator (>>).
            Denotes the right implication operation: self -> other
        """
        from .dispatcher import operate
        return operate('implication', self, other)

    def __xor__(self, other):
        """Overloads the xor operator (^).
            Denotes the symmetric difference operation.
        """
        from .dispatcher import operate
        return operate('symdiff', self, other)

    def equivalent(self, other):
        """
        Calculate the equivalence level between two fuzzy numbers

        Corresponding to the "if and only if" operation in classical logic,
        it represents the degree to which two fuzzy propositions are
        equivalent to each other.
        """
        from .dispatcher import operate
        return operate('equivalence', self, other)

    # ======================== Array Operations ==========================
    # These methods provide common array manipulation functionalities,
    # similar to those found in NumPy.

    def tolist(self) -> List[Any]:
        """
        Converts the Fuzzarray to a (possibly nested) list of Fuzznum objects.

        Returns:
            List[Any]: A list representation of the Fuzzarray.
        """
        return self._data.tolist()

    @classmethod
    def concatenate(cls, arrays: List['Fuzzarray'], axis: int = 0) -> 'Fuzzarray':
        """
        Concatenates a sequence of Fuzzarray instances along a specified axis.

        All Fuzzarrays in the sequence must have the same `mtype` and `q` value.
        Their shapes must be compatible for concatenation along the given axis.

        Args:
            arrays (List[Fuzzarray]): A list of Fuzzarray instances to concatenate.
            axis (int): The axis along which to concatenate the arrays. Defaults to 0.

        Returns:
            Fuzzarray: A new Fuzzarray resulting from the concatenation.

        Raises:
            ValueError: If the input list is empty, or if `mtype`, `q`, or shapes
                        are inconsistent among the input Fuzzarrays.
            TypeError: If the input list contains non-Fuzzarray instances.
        """
        if not arrays:
            raise ValueError("Cannot concatenate empty list of Fuzzarrays.")

        # Validate mtype and q consistency across all input Fuzzarrays.
        first_mtype = arrays[0].mtype
        first_q = arrays[0].q
        for arr in arrays:
            if not isinstance(arr, Fuzzarray):
                raise TypeError(f"All elements in 'arrays' must be Fuzzarray instances, found {type(arr)}.")
            if arr.mtype != first_mtype or arr.q != first_q:
                raise ValueError("All Fuzzarrays to concatenate must have the same mtype and qrung.")

        # Extract the underlying NumPy arrays for concatenation.
        np_arrays = [arr._data for arr in arrays]

        try:
            # Perform the NumPy concatenation.
            concatenated_data = np.concatenate(np_arrays, axis=axis)
        except ValueError as e:
            # Re-raise ValueError with a more informative message if shapes are incompatible.
            raise ValueError(f"Shape mismatch for concatenation along axis {axis}: {e}")

        # Create a new Fuzzarray from the concatenated data, preserving mtype and q.
        return cls(concatenated_data, mtype=first_mtype, copy=False)

    @classmethod
    def stack(cls, arrays: List['Fuzzarray'], axis: int = 0) -> 'Fuzzarray':
        """
        Stacks a sequence of Fuzzarray instances along a new axis.

        All Fuzzarrays in the sequence must have the same `mtype`, `q`, and identical shapes.

        Args:
            arrays (List[Fuzzarray]): A list of Fuzzarray instances to be stacked.
            axis (int): The index for inserting the new axis. Defaults to 0.

        Returns:
            Fuzzarray: A new Fuzzarray resulting from the stacking operation.

        Raises:
            ValueError: If the input list is empty, or if `mtype`, `q`, or shapes
                        are inconsistent among the input Fuzzarrays.
            TypeError: If the input list contains non-Fuzzarray instances.
        """
        if not arrays:
            raise ValueError("Cannot stack empty list of Fuzzarrays.")

        # Validate mtype, q, and shape consistency across all input Fuzzarrays.
        first_mtype = arrays[0].mtype
        first_q = arrays[0].q
        for arr in arrays:
            if not isinstance(arr, Fuzzarray):
                raise TypeError(f"All elements in 'arrays' must be Fuzzarray instances, found {type(arr)}.")
            if arr.mtype != first_mtype or arr.q != first_q:
                raise ValueError("All Fuzzarrays to stack must have the same mtype and qrung.")
            if arr.shape != arrays[0].shape:
                raise ValueError("All Fuzzarrays to stack must have the same shape.")

        # Extract the underlying NumPy arrays for stacking.
        np_arrays = [arr._data for arr in arrays]

        try:
            # Perform the NumPy stacking.
            stacked_data = np.stack(np_arrays, axis=axis)
        except ValueError as e:
            # Re-raise ValueError with a more informative message if stacking fails.
            raise ValueError(f"Shape mismatch for stacking along axis {axis}: {e}")

        # Create a new Fuzzarray from the stacked data, preserving mtype and q.
        return cls(stacked_data, mtype=first_mtype, copy=False)

    # ======================== String representation ========================

    def __repr__(self) -> str:
        """
        Returns a developer-friendly string representation of the Fuzzarray.

        This representation includes the array's data, q-rung, and mtype,
        making it useful for debugging and introspection.

        Returns:
            str: The string representation.
        """
        # Format the underlying NumPy array's string representation to be indented.
        report = str(self._data).replace('\n', '\n' + ' ' * 10)
        return f'Fuzzarray({report}, q={self._q}, mtype={self.mtype})'

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of the Fuzzarray.

        This typically just shows the underlying data, similar to how NumPy arrays are printed.

        Returns:
            str: The string representation of the array's data.
        """
        return f"{self._data}"


def fuzzarray(data: np.ndarray | list | tuple | Fuzznum,
              mtype: Optional[str] = None,
              shape: Optional[tuple[int, ...]] = None,
              copy: bool = True) -> Fuzzarray:
    """
    A convenient factory function to create a `Fuzzarray` instance.

    This function acts as a wrapper around the `Fuzzarray` constructor,
    providing a more idiomatic way to instantiate `Fuzzarray` objects.

    Args:
        data (np.ndarray | list | tuple | Fuzznum): The input data to populate the Fuzzarray.
                                                        Can be a NumPy array of Fuzznum objects,
                                                        a list/tuple of Fuzznum objects, or a single Fuzznum.
        mtype (Optional[str]): The expected membership type for all Fuzznum elements.
                                   If provided, it must match the mtype of the Fuzznum(s) in `data`.
                                   If None, the mtype is inferred from the first Fuzznum in `data`.
        shape (Optional[tuple[int, ...]]): The desired shape of the Fuzzarray.
                                               If `data` is a single Fuzznum, this defines the
                                               shape of the array filled with that Fuzznum.
                                               If `data` is an array-like, it will be reshaped.
        copy (bool): If True, a deep copy of the input Fuzznum objects is made.
                     If False, references to the original Fuzznum objects are used.
                     Defaults to True.

    Returns:
        Fuzzarray: The newly created Fuzzarray instance.
    """
    return Fuzzarray(data, mtype=mtype, shape=shape, copy=copy)
