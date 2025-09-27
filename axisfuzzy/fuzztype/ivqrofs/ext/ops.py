#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
IVQROFN Mathematical Operations Extension Methods.

This module implements high-performance mathematical and aggregation operations
for Interval-Valued Q-Rung Orthopair Fuzzy Numbers, including sum, mean, max, min,
and statistical functions with proper interval arithmetic.
"""

from typing import Union, Tuple

import numpy as np

from ....core import OperationTNorm, get_registry_operation, Fuzznum, Fuzzarray


def _ivqrofn_sum(arr: Union[Fuzznum, Fuzzarray],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance sum for IVQROFN using interval-aware t-norm/t-conorm reduction.
    
    For intervals, the sum operation applies t-conorm to membership intervals
    and t-norm to non-membership intervals using vectorized operations.
    
    Parameters:
        arr: IVQROFN object to sum
        axis: Axis or axes along which to compute the sum
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Summed IVQROFN result
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError('Cannot sum empty array')

    if arr.size == 1:
        # Single element case - return directly
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(())

    # Get t-norm operations for proper fuzzy aggregation
    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)

    # Get component interval arrays
    mds, nmds = arr.backend.get_component_arrays()

    if axis is None:
        # For complete reduction, we need to reduce over all logical dimensions
        # but preserve the interval structure (last dimension)
        logical_shape = arr.shape
        if len(logical_shape) == 0:
            # Single Fuzznum case
            return arr
        
        # Flatten logical dimensions but keep interval structure
        flat_mds = mds.reshape(-1, 2)  # Shape: (total_elements, 2)
        flat_nmds = nmds.reshape(-1, 2)  # Shape: (total_elements, 2)
        
        # Reduce over the first dimension (elements), keeping intervals
        md_sum = tnorm.t_conorm_reduce(flat_mds, axis=0)  # Result: (2,)
        nmd_sum = tnorm.t_norm_reduce(flat_nmds, axis=0)   # Result: (2,)
        
        return Fuzznum('ivqrofn', q=arr.q).create(md=md_sum, nmd=nmd_sum)
    else:
        # Apply reduction along specified axis, preserving interval structure
        md_sum = tnorm.t_conorm_reduce(mds, axis=axis)
        nmd_sum = tnorm.t_norm_reduce(nmds, axis=axis)
        
        backend_cls = arr.backend.__class__
        new_backend = backend_cls.from_arrays(md_sum, nmd_sum, q=arr.q)
        return Fuzzarray(backend=new_backend)


def _ivqrofn_mean(arr: Union[Fuzznum, Fuzzarray],
                  axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance mean for IVQROFN using interval-aware averaging.
    
    The mean operation computes interval-wise averages while maintaining
    the mathematical properties of interval-valued fuzzy numbers.
    
    Parameters:
        arr: IVQROFN object to average
        axis: Axis or axes along which to compute the mean
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Mean IVQROFN result
    """
    if arr.size == 0:
        # Empty array - return zero interval
        return Fuzznum('ivqrofn', q=arr.q).create(md=[0.0, 0.0], nmd=[0.0, 0.0])
    
    if arr.size == 1:
        # Single element case
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(())

    # Calculate count for averaging
    if axis is None:
        n = arr.size
    else:
        n = np.prod([arr.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])

    # Use sum and divide by count
    total = _ivqrofn_sum(arr, axis=axis)
    return total / n


def _ivqrofn_max(arr: Union[Fuzznum, Fuzzarray],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance max for IVQROFN based on comprehensive interval score comparison.
    
    For intervals, comparison considers the entire interval information:
    score = (md_lower^q + md_upper^q)/2 - (nmd_lower^q + nmd_upper^q)/2
    
    Parameters:
        arr: IVQROFN object to find maximum
        axis: Axis or axes along which to compute the maximum
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Maximum IVQROFN element(s)
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("max() arg is an empty sequence")

    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(())

    # Get component arrays for comprehensive score calculation
    mds, nmds = arr.backend.get_component_arrays()
    
    # Calculate interval scores considering entire interval information
    md_avg_scores = (mds[..., 0] ** arr.q + mds[..., 1] ** arr.q) / 2
    nmd_avg_scores = (nmds[..., 0] ** arr.q + nmds[..., 1] ** arr.q) / 2
    scores = md_avg_scores - nmd_avg_scores
    
    indices = np.argmax(scores, axis=axis)

    if axis is None:
        # Return single maximum element
        from ....mixin.factory import _flatten_factory
        return _flatten_factory(arr)[indices]
    else:
        # Multi-dimensional case - simplified for single axis
        if isinstance(axis, tuple):
            raise NotImplementedError("max with tuple axis is not yet supported.")

        # Create index grid for selection
        grid = np.indices(indices.shape)
        idx_list = list(grid)
        idx_list.insert(axis, indices)

        return arr[tuple(idx_list)]


def _ivqrofn_min(arr: Union[Fuzznum, Fuzzarray],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance min for IVQROFN based on comprehensive interval score comparison.
    
    For intervals, comparison considers the entire interval information:
    score = (md_lower^q + md_upper^q)/2 - (nmd_lower^q + nmd_upper^q)/2
    
    Parameters:
        arr: IVQROFN object to find minimum
        axis: Axis or axes along which to compute the minimum
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Minimum IVQROFN element(s)
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("min() arg is an empty sequence")

    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(())

    # Get component arrays for comprehensive score calculation
    mds, nmds = arr.backend.get_component_arrays()
    
    # Calculate interval scores considering entire interval information
    md_avg_scores = (mds[..., 0] ** arr.q + mds[..., 1] ** arr.q) / 2
    nmd_avg_scores = (nmds[..., 0] ** arr.q + nmds[..., 1] ** arr.q) / 2
    scores = md_avg_scores - nmd_avg_scores
    
    indices = np.argmin(scores, axis=axis)

    if axis is None:
        from ....mixin.factory import _flatten_factory
        return _flatten_factory(arr)[indices]
    else:
        if isinstance(axis, tuple):
            raise NotImplementedError("min with tuple axis is not yet supported.")

        grid = np.indices(indices.shape)
        idx_list = list(grid)
        idx_list.insert(axis, indices)

        return arr[tuple(idx_list)]


def _ivqrofn_prod(arr: Union[Fuzznum, Fuzzarray],
                  axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance product for IVQROFN using interval-aware t-norm/t-conorm.
    
    For intervals, product uses t-norm for membership and t-conorm for non-membership
    with full vectorized interval operations.
    
    Parameters:
        arr: IVQROFN object to compute product
        axis: Axis or axes along which to compute the product
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Product IVQROFN result
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError('Cannot compute product of empty array')

    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(())

    # Get t-norm operations
    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)

    # Get component arrays
    mds, nmds = arr.backend.get_component_arrays()

    if axis is None:
        # For complete reduction, preserve interval structure
        logical_shape = arr.shape
        if len(logical_shape) == 0:
            return arr
        
        # Flatten logical dimensions but keep interval structure
        flat_mds = mds.reshape(-1, 2)  # Shape: (total_elements, 2)
        flat_nmds = nmds.reshape(-1, 2)  # Shape: (total_elements, 2)
        
        # Reduce over the first dimension, keeping intervals
        md_prod = tnorm.t_norm_reduce(flat_mds, axis=0)   # Result: (2,)
        nmd_prod = tnorm.t_conorm_reduce(flat_nmds, axis=0)  # Result: (2,)
        
        return Fuzznum('ivqrofn', q=arr.q).create(md=md_prod, nmd=nmd_prod)
    else:
        # Apply reduction along specified axis, preserving interval structure
        md_prod = tnorm.t_norm_reduce(mds, axis=axis)
        nmd_prod = tnorm.t_conorm_reduce(nmds, axis=axis)
        
        backend_cls = arr.backend.__class__
        new_backend = backend_cls.from_arrays(md_prod, nmd_prod, q=arr.q)
        return Fuzzarray(backend=new_backend)


def _ivqrofn_var(arr: Union[Fuzznum, Fuzzarray],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance variance for IVQROFN using interval arithmetic.
    
    Computes variance: Var(X) = E[(X - E[X])²] for interval-valued data.
    
    Parameters:
        arr: IVQROFN object to compute variance
        axis: Axis or axes along which to compute the variance
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Variance IVQROFN result
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("var() arg is an empty sequence")

    if arr.size == 1:
        # Single element variance is zero
        return Fuzznum('ivqrofn', q=arr.q).create(md=[0.0, 0.0], nmd=[1.0, 1.0])

    # Calculate mean first
    mean_val = _ivqrofn_mean(arr, axis=axis)

    # Calculate squared differences
    diff = arr - mean_val
    squared_diff = diff * diff

    # Return mean of squared differences
    return _ivqrofn_mean(squared_diff, axis=axis)


def _ivqrofn_std(arr: Union[Fuzznum, Fuzzarray],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance standard deviation for IVQROFN.
    
    Computes standard deviation as the square root of variance for intervals.
    
    Parameters:
        arr: IVQROFN object to compute standard deviation
        axis: Axis or axes along which to compute the standard deviation
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Standard deviation IVQROFN result
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("std() arg is an empty sequence")

    if arr.size == 1:
        return Fuzznum('ivqrofn', q=arr.q).create(md=[0.0, 0.0], nmd=[1.0, 1.0])

    # Calculate variance and take square root
    variance = _ivqrofn_var(arr, axis=axis)
    return variance ** 0.5


def _ivqrofn_score(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance score calculation for IVQROFN intervals.
    
    Score function for intervals: score = (md_lower^q + md_upper^q)/2 - (nmd_lower^q + nmd_upper^q)/2
    This considers the entire interval information, not just upper bounds.
    
    Parameters:
        arr: IVQROFN object to compute score
        
    Returns:
        Union[float, np.ndarray]: Score value(s)
    """
    if isinstance(arr, Fuzznum):
        # Single Fuzznum score using entire interval
        md_avg_score = (arr.md[0] ** arr.q + arr.md[1] ** arr.q) / 2
        nmd_avg_score = (arr.nmd[0] ** arr.q + arr.nmd[1] ** arr.q) / 2
        return md_avg_score - nmd_avg_score

    # Fuzzarray vectorized computation
    mds, nmds = arr.backend.get_component_arrays()
    
    # Calculate average scores for intervals
    md_avg_scores = (mds[..., 0] ** arr.q + mds[..., 1] ** arr.q) / 2
    nmd_avg_scores = (nmds[..., 0] ** arr.q + nmds[..., 1] ** arr.q) / 2
    
    scores = md_avg_scores - nmd_avg_scores
    return scores


def _ivqrofn_acc(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance accuracy calculation for IVQROFN intervals.
    
    Accuracy function for intervals: acc = (md_lower^q + md_upper^q)/2 + (nmd_lower^q + nmd_upper^q)/2
    This considers the entire interval information for a complete assessment.
    
    Parameters:
        arr: IVQROFN object to compute accuracy
        
    Returns:
        Union[float, np.ndarray]: Accuracy value(s)
    """
    if isinstance(arr, Fuzznum):
        # Single Fuzznum accuracy using entire interval
        md_avg_score = (arr.md[0] ** arr.q + arr.md[1] ** arr.q) / 2
        nmd_avg_score = (arr.nmd[0] ** arr.q + arr.nmd[1] ** arr.q) / 2
        return md_avg_score + nmd_avg_score

    # Fuzzarray vectorized computation
    mds, nmds = arr.backend.get_component_arrays()
    
    # Calculate average scores for intervals
    md_avg_scores = (mds[..., 0] ** arr.q + mds[..., 1] ** arr.q) / 2
    nmd_avg_scores = (nmds[..., 0] ** arr.q + nmds[..., 1] ** arr.q) / 2
    
    accuracy = md_avg_scores + nmd_avg_scores
    return accuracy


def _ivqrofn_ind(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance indeterminacy calculation for IVQROFN intervals.
    
    Indeterminacy function for intervals: ind = 1 - acc
    Where acc considers the entire interval information.
    
    Parameters:
        arr: IVQROFN object to compute indeterminacy
        
    Returns:
        Union[float, np.ndarray]: Indeterminacy value(s)
    """
    if isinstance(arr, Fuzznum):
        # Single Fuzznum indeterminacy using entire interval
        md_avg_score = (arr.md[0] ** arr.q + arr.md[1] ** arr.q) / 2
        nmd_avg_score = (arr.nmd[0] ** arr.q + arr.nmd[1] ** arr.q) / 2
        acc = md_avg_score + nmd_avg_score
        return 1.0 - acc

    # Fuzzarray vectorized computation
    mds, nmds = arr.backend.get_component_arrays()
    
    # Calculate average scores for intervals
    md_avg_scores = (mds[..., 0] ** arr.q + mds[..., 1] ** arr.q) / 2
    nmd_avg_scores = (nmds[..., 0] ** arr.q + nmds[..., 1] ** arr.q) / 2
    
    accuracy = md_avg_scores + nmd_avg_scores
    indeterminacy = 1.0 - accuracy
    return indeterminacy