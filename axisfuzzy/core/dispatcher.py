#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Central operation dispatcher for AxisFuzzy.

This module provides a central, intelligent dispatcher for all mathematical and
logical operations involving AxisFuzzy data types. It acts as the primary
routing mechanism that enables seamless interaction between `Fuzznum`, `Fuzzarray`,
and standard Python/NumPy types.

Overview
--------
The dispatcher is the core component that powers Python's operator overloading
(e.g., `+`, `*`, `>`) for fuzzy objects. Its main responsibility is to inspect
the types of the operands involved in an operation and route the request to the
most efficient implementation path available in the framework.

Dispatch Logic:

- **`Fuzznum` vs `Fuzznum`**: Operations are delegated directly to the `execute_operation`
  method of the underlying `FuzznumStrategy`, performing element-wise computation.
- **`Fuzzarray` vs `Fuzzarray`**: Operations are dispatched to the `Fuzzarray`'s
  `execute_vectorized_op` method, which leverages the high-performance SoA backend
  for efficient, vectorized calculations.
- **Mixed `Fuzzarray` and `Fuzznum`**: The `Fuzznum` is automatically broadcast into a
  `Fuzzarray` of a compatible shape, and the operation is then handled as a
  `Fuzzarray`-`Fuzzarray` operation.
- **Fuzzy vs Scalar/`ndarray`**: Similar to mixed fuzzy types, scalars or NumPy arrays
  are handled by broadcasting them to operate against the `Fuzzarray`'s backend,
  ensuring maximum performance.
- **Reverse Operations**: It correctly handles commutative operations where the fuzzy
  object is the right-hand operand (e.g., `2 * my_fuzznum`).
"""

from typing import Any, Optional

import numpy as np


def operate(op_name: str, operand1: Any, operand2: Optional[Any]) -> Any:
    """
    Perform a named operation between two operands using intelligent, type-based dispatch.

    This function is the single entry point for all binary and unary operations
    invoked on `Fuzznum` and `Fuzzarray` objects. It determines the most
    efficient execution path by analyzing the types of the operands.

    Parameters
    ----------
    op_name : str
        The name of the operation to perform (e.g., 'add', 'mul', 'gt', 'complement').
        This name corresponds to an operation registered in the `OperationScheduler`.
    operand1 : object
        The first (left-hand) operand. Supported types include `Fuzznum`, `Fuzzarray`,
        `int`, `float`, and `numpy.ndarray`.
    operand2 : object or None
        The second (right-hand) operand. Supported types mirror `operand1`. For pure
        unary operations like 'complement', this should be `None`.

    Returns
    -------
    Any
        The result of the dispatched operation. The return type is dynamic and
        depends on the operation and operand types (e.g., `Fuzznum`, `Fuzzarray`, `bool`).

    Raises
    ------
    TypeError
        If the combination of operand types is not supported for the given operation.

    Notes
    -----
    - **Lazy Imports**: `Fuzznum` and `Fuzzarray` are imported inside the function
      to prevent circular dependencies that can occur during module initialization.
    - **Performance**: The dispatcher prioritizes vectorized `Fuzzarray` operations.
      When an operation involves a `Fuzzarray`, it will always attempt to use the
      backend-accelerated path, broadcasting other operands if necessary.
    - **Broadcasting**: When a `Fuzznum` operates with a `Fuzzarray` or `ndarray`,
      it is implicitly converted into a `Fuzzarray` of the correct shape before
      the operation proceeds.
    - **Operation Aliases**: For convenience, some common operation names are mapped
      to their internal strategy equivalents. For example, `mul` and `div` with a
      scalar are mapped to the `tim` (times) operation.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.core.fuzznums import fuzznum
        from axisfuzzy.core.fuzzarray import fuzzarray

        # Assume qrofn is the default mtype
        a = fuzznum(md=0.5, nmd=0.3)
        b = fuzznum(md=0.6, nmd=0.2)
        arr1 = fuzzarray([a, b])
        arr2 = fuzzarray([b, a])

        # Fuzznum + Fuzznum -> returns Fuzznum
        result_fn = operate('add', a, b)

        # Fuzznum * scalar -> returns Fuzznum
        result_fn_scalar = operate('mul', a, 2.0)

        # Fuzzarray + Fuzzarray -> returns Fuzzarray
        result_arr = operate('add', arr1, arr2)

        # Fuzzarray + Fuzznum (broadcasting) -> returns Fuzzarray
        result_arr_fn = operate('add', arr1, a)

        # scalar + Fuzzarray (reverse operation) -> returns Fuzzarray
        # Note: 'mul' is commutative and supported for reverse ops
        result_rev_arr = operate('mul', 2.0, arr1)

        # Fuzznum complement (unary op) -> returns Fuzznum
        result_unary = operate('complement', a, None)
    """
    # Dynamically import the required classes to avoid circular imports.
    # These imports are placed here to prevent circular dependencies at module load time.
    from .fuzznums import Fuzznum
    from .fuzzarray import Fuzzarray

    # --- Type Dispatch Logic ---
    # This is a simplified dispatch table, which can be optimized using more complex
    #   design patterns (such as multiple dispatch libraries).
    type1 = type(operand1)
    type2 = type(operand2)

    # Rule 1: Fuzznum <op> Fuzznum
    # Handles operations between two Fuzznum instances.
    if isinstance(operand1, Fuzznum) and isinstance(operand2, Fuzznum):
        result_dict = operand1.get_strategy_instance().execute_operation(
            op_name, operand2.get_strategy_instance())
        if op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
            # Comparison operations return boolean values.
            return result_dict.get('value', False)
        return operand1.create(**result_dict)

    # Rule 2: Fuzznum <op> Fuzzarray
    # Handles operations where a Fuzznum interacts with a Fuzzarray.
    if isinstance(operand1, Fuzznum) and isinstance(operand2, Fuzzarray):
        # Broadcast Fuzznum to match the shape of operand2.
        # The operation becomes Fuzzarray <op> Fuzzarray.
        # We can now directly use the Fuzzarray constructor for this.
        broadcasted_fuzzarray = Fuzzarray(data=operand1,
                                          mtype=operand2.mtype,
                                          shape=operand2.shape,
                                          q=operand1.q)
        return operate(op_name, broadcasted_fuzzarray, operand2)

    # Rule 3: Fuzznum <op> Scalar (int, float)
    # Handles operations between a Fuzznum and a standard scalar (int or float).
    if isinstance(operand1, Fuzznum) and isinstance(operand2, (int, float, np.integer, np.floating)):
        # Special handling for 'mul' and 'div' to map them to 'tim' (times) operation.
        if op_name == 'mul':
            op_name = 'tim'
        if op_name == 'div':
            op_name = 'tim'
            operand2 = 1 / operand2  # Division is treated as multiplication by reciprocal.
        result_dict = operand1.get_strategy_instance().execute_operation(op_name, operand2)
        return operand1.create(**result_dict)

    # Rule 4: Fuzznum <op> ndarray (Broadcasting Fuzznum is required)
    # Handles operations where a Fuzznum interacts with a NumPy array.
    if isinstance(operand1, Fuzznum) and isinstance(operand2, np.ndarray):
        # Broadcast Fuzznum into Fuzzarray to match the shape of the ndarray.
        # The rule has become Fuzzarray <op> ndarray, then recursively call operate.
        if op_name == 'mul':
            op_name = 'tim'
        if op_name == 'div':
            op_name = 'tim'
            operand2 = 1 / operand2
        broadcasted_fuzzarray = Fuzzarray(data=operand1,
                                          mtype=operand1.mtype,
                                          shape=operand2.shape,
                                          q=operand1.q)
        return operate(op_name, broadcasted_fuzzarray, operand2)

    # Rule 5: Fuzzarray <op> Fuzzarray / Fuzznum
    # Handles operations where a Fuzzarray interacts with another Fuzzarray or a Fuzznum.
    if isinstance(operand1, Fuzzarray) and isinstance(operand2, (Fuzznum, Fuzzarray)):
        # Directly use Fuzzarray's execute_vectorized_op method, as it's designed for this.
        # Fuzzarray's execute_vectorized_op is already a good dispatcher.
        return operand1.execute_vectorized_op(op_name, operand2)

    # Rule 6: Fuzzarray <op> Scalar / ndarray
    # Handles operations where a Fuzzarray interacts with a scalar or a NumPy array.
    if isinstance(operand1, Fuzzarray) and isinstance(operand2, (int, float, np.integer, np.floating, np.ndarray)):
        # Special handling for 'mul' and 'div' to map them to 'tim' (times) operation.
        if op_name == 'mul':
            op_name = 'tim'
        if op_name == 'div':
            op_name = 'tim'
            operand2 = 1 / operand2
        return operand1.execute_vectorized_op(op_name, operand2)

    # --- Reverse operation processing ---
    # These rules handle cases where the FuzzLab type is the second operand.

    # Rule 7: Scalar <op> Fuzznum / Fuzzarray
    # Handles operations where a scalar is the first operand and a Fuzznum/Fuzzarray is the second.
    if isinstance(operand1, (int, float, np.integer, np.floating)) and isinstance(operand2, (Fuzznum, Fuzzarray)):
        # Swap operands and recursively call operate for commutative operations.
        # Note: This only works for commutative operations (add, mul).
        if op_name in ['add', 'mul']:
            return operate(op_name, operand2, operand1)
        # For non-commutative operations (e.g., subtraction, division), special handling is required.
        # This part would need to be extended if non-commutative reverse operations are needed.

    # Rule 8: ndarray <op> Fuzznum
    # Handles operations where a NumPy array is the first operand and a Fuzzarray/Fuzznum is the second.
    if isinstance(operand1, np.ndarray) and isinstance(operand2, (Fuzzarray, Fuzznum)):
        # Swap operands and recursively call operate for commutative operations.
        if op_name in ['add', 'mul']:
            return operate(op_name, operand2, operand1)

    # Rule 9: Fuzznum / Fuzzarray
    # Pure unary operation, referring to the complement operation
    if isinstance(operand1, (Fuzznum, Fuzzarray)) and operand2 is None:
        if op_name in ['complement']:
            if isinstance(operand1, Fuzznum):
                result_dict = operand1.get_strategy_instance().execute_operation(
                    op_name, operand2)
                return operand1.create(**result_dict)
            else:
                return operand1.execute_vectorized_op(op_name, operand2)

    # If no rule matches, raise a TypeError indicating unsupported operand types.
    raise TypeError(f"Unsupported operand types for operation '{op_name}': "
                    f"{type1.__name__} and {type2.__name__}")
