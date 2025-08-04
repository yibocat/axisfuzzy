#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/4 17:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Any

import numpy as np


def operate(op_name: str, operand1: Any, operand2: Any) -> Any:
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
    """
    # Dynamically import the required classes to avoid circular imports.
    from .fuzznums import Fuzznum
    from .fuzzarray import Fuzzarray, fuzzarray

    # --- Type Dispatch Logic ---
    # This is a simplified dispatch table, which can be optimized using more complex
    #   design patterns (such as multiple dispatch libraries).
    type1 = type(operand1)
    type2 = type(operand2)

    # Rule 1: Fuzznum <op> Fuzznum
    if isinstance(operand1, Fuzznum) and isinstance(operand2, Fuzznum):
        result_dict = operand1.get_strategy_instance().execute_operation(op_name, operand2.get_strategy_instance())
        if op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
            # Comparison operations return boolean values.
            return result_dict.get('value', False)
        return operand1.create(**result_dict)

    # Rule 2: Fuzznum <op> Fuzzarray
    if isinstance(operand1, Fuzznum) and isinstance(operand2, Fuzzarray):
        # Broadcast Fuzznum into Fuzzarray
        # The rule became Fuzzarray <op> Fuzzarray
        broadcasted_fuzzarray = fuzzarray(operand1, shape=operand2.shape)
        return operate(op_name, broadcasted_fuzzarray, operand2)

    # Rule 3: Fuzznum <op> Scalar (int, float)
    if isinstance(operand1, Fuzznum) and isinstance(operand2, (int, float)):
        if op_name == 'mul':
            op_name = 'tim'
        if op_name == 'div':
            op_name = 'tim'
            operand2 = 1 / operand2
        result_dict = operand1.get_strategy_instance().execute_operation(op_name, operand2)
        return operand1.create(**result_dict)

    # Rule 4: Fuzznum <op> ndarray (Broadcasting Fuzznum is required)
    if isinstance(operand1, Fuzznum) and isinstance(operand2, np.ndarray):
        # Broadcast Fuzznum into Fuzzarray
        # The rule has become Fuzzarray <op> ndarray
        if op_name == 'mul':
            op_name = 'tim'
        if op_name == 'div':
            op_name = 'tim'
            operand2 = 1 / operand2
        broadcasted_fuzzarray = fuzzarray(operand1, shape=operand2.shape)
        return operate(op_name, broadcasted_fuzzarray, operand2)

    # Rule 5: Fuzzarray <op> Fuzzarray / Fuzznum
    if isinstance(operand1, Fuzzarray) and isinstance(operand2, (Fuzznum, Fuzzarray)):
        # Directly use Fuzzarray's execute_vectorized_op method
        # Fuzzarray's execute_vectorized_op is already a good dispatcher.
        return operand1.execute_vectorized_op(op_name, operand2)

    # Rule 6: Fuzzarray <op> Scalar / ndarray
    if isinstance(operand1, Fuzzarray) and isinstance(operand2, (int, float, np.ndarray)):
        if op_name == 'mul':
            op_name = 'tim'
        if op_name == 'div':
            op_name = 'tim'
            operand2 = 1 / operand2
        return operand1.execute_vectorized_op(op_name, operand2)

    # --- Reverse operation processing ---
    # Rule 7: Scalar <op> Fuzznum / Fuzzarray
    if isinstance(operand1, (int, float)) and isinstance(operand2, (Fuzznum, Fuzzarray)):
        # Swap operands, call operate(Fuzznum, scalar)
        # Note: This only works for commutative operations (add, mul)
        if op_name in ['add', 'mul']:
            return operate(op_name, operand2, operand1)
        # For non-commutative operations, special handling is required.
        # ...

    # Rule 8: ndarray <op> Fuzznum
    if isinstance(operand1, np.ndarray) and isinstance(operand2, (Fuzzarray, Fuzznum)):
        # Swap operands, call operate(Fuzznum, ndarray)
        if op_name in ['add', 'mul']:
            return operate(op_name, operand2, operand1)

    raise TypeError(f"Unsupported operand types for operation '{op_name}': "
                    f"{type1.__name__} and {type2.__name__}")
