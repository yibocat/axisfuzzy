#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/4 17:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module provides a central dispatcher for performing operations between
various FuzzLab data types (Fuzznum, Fuzzarray) and standard Python/NumPy types.

It handles type-based dispatching to ensure that operations like addition,
multiplication, and comparisons are correctly applied across different
combinations of fuzzy numbers, fuzzy arrays, scalars, and NumPy arrays.
"""
from typing import Any, Optional

import numpy as np


def operate(op_name: str, operand1: Any, operand2: Optional[Any]) -> Any:
    """Performs an operation between two operands based on their types.

    This function acts as a dispatcher, handling various combinations of Fuzznum,
    Fuzzarray, scalar (int, float), and numpy.ndarray types. It dynamically
    routes the operation to the appropriate handler based on the types of
    `operand1` and `operand2`.

    Args:
        op_name (str): The name of the operation to perform (e.g., 'add', 'mul',
                       'gt', 'tim').
        operand1 (Any): The first operand. Can be a Fuzznum, Fuzzarray, int,
                        float, or numpy.ndarray.
        operand2 (Any): The second operand. Can be a Fuzznum, Fuzzarray, int,
                        float, or numpy.ndarray.

    Returns:
        Any: The result of the operation. The type of the result depends on the
             operation and the types of the operands (e.g., Fuzznum, Fuzzarray,
             or bool for comparison operations).

    Raises:
        TypeError: If the combination of operand types is not supported for the
                   given operation.

    Examples:
        >>> from fuzzlab.core.fuzznums import Fuzznum
        >>> from fuzzlab.core.fuzzarray import Fuzzarray, fuzzarray
        >>> # Assuming Fuzznum and Fuzzarray are properly initialized
        >>> # Fuzznum + Fuzznum
        >>> # result = operate('add', Fuzznum(1), Fuzznum(2))
        >>> # Fuzznum * scalar
        >>> # result = operate('mul', Fuzznum(5), 2)
        >>> # Fuzzarray + Fuzzarray
        >>> # arr1 = fuzzarray([1, 2])
        >>> # arr2 = fuzzarray([3, 4])
        >>> # result = operate('add', arr1, arr2)
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
