#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 21:56
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


from typing import Callable
import numpy as np


def _pairwise_combinations(
        a: np.ndarray,
        b: np.ndarray,
        func: Callable) -> np.ndarray:
    """Generate all pairwise combinations between two 1D arrays with a custom binary operation.

    This function applies a given binary function to each combination of
    elements from arrays `a` and `b`, and flattens the results into a 1D array.

    Args:
        a (np.ndarray): First 1D input array.
        b (np.ndarray): Second 1D input array.
        func (Callable): A binary function that takes two NumPy arrays `x` and `y`
            of the same shape and returns an array of results.

    Returns:
        np.ndarray: A 1D array containing results of applying `func` to every pair (ai, bj).

    Raises:
        ValueError: If inputs are not 1D NumPy arrays.

    Examples:
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([10, 20])
        >>> pairwise_combinations(a, b, lambda x, y: x + y)
        array([11, 21, 12, 22, 13, 23])

        >>> pairwise_combinations(a, b, lambda x, y: x * y)
        array([10, 20, 20, 40, 30, 60])

    Notes:
        - Internally uses `np.meshgrid` for broadcasting all combinations.
        - The resulting 2D matrix is flattened into one dimension.
        - Order follows row-major (C-order) flattening, i.e. combinations are grouped by `a` first.

    See Also:
        np.add.outer, np.multiply.outer, np.fromfunction
    """
    if a is None or b is None:
        raise ValueError("Inputs must not be None.")
    if a.ndim != 1 or b.ndim != 1:
        # This can happen if one of the elements in the object array is None
        raise ValueError("Inputs must be 1D NumPy arrays.")

    A, B = np.meshgrid(a, b, indexing="ij")
    return func(A, B).ravel()
