#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) Mathematical Operations Extension Methods.

This module implements mathematical operations and aggregation functions for
classical fuzzy sets (FS), based on Zadeh's fuzzy set theory. All operations
are optimized for high performance using backend component arrays and
vectorized computations.

Mathematical Foundation:
    Classical fuzzy sets have only membership degrees μ ∈ [0, 1].
    Operations follow standard fuzzy set theory:
    - Aggregation: Uses t-norm/t-conorm operations  
    - Statistical measures: Adapted for fuzzy values
    - Score functions: Simplified compared to QROFN (no q parameter)
"""

from typing import Union, Tuple

import numpy as np

from ....core import OperationTNorm, get_registry_operation, Fuzznum, Fuzzarray


def _fs_sum(arr: Union[Fuzznum, Fuzzarray],
            axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance sum for FS using t-conorm reduction.
    
    For classical fuzzy sets, summation is implemented using t-conorm
    (fuzzy OR) operation, representing the union of all fuzzy sets.
    
    Mathematical formulation:
        Sum(A₁, A₂, ..., Aₙ) = S(μ₁, μ₂, ..., μₙ)
        where S is a t-conorm operator
        
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        axis (Union[int, Tuple[int, ...]]): Reduction axis/axes
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Sum result
        
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
            return arr.backend.get_fuzznum_view(0)

    # Get t-conorm operation for fuzzy union
    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, **params)

    # Get membership degrees from backend
    mds, = arr.backend.get_component_arrays()

    # Apply t-conorm reduction (fuzzy union)
    md_sum = tnorm.t_conorm_reduce(mds, axis=axis)

    # Return appropriate type based on axis parameter
    if axis is None:
        return Fuzznum('fs').create(md=md_sum)
    else:
        backend_cls = arr.backend.__class__
        new_backend = backend_cls.from_arrays(mds=md_sum)
        return Fuzzarray(backend=new_backend)


def _fs_mean(arr: Union[Fuzznum, Fuzzarray],
             axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance mean for FS using fuzzy aggregation.
    
    The mean is computed by taking the fuzzy sum and dividing by the count.
    For FS, this provides a balanced aggregation that respects fuzzy logic principles.
    
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        axis (Union[int, Tuple[int, ...]]): Reduction axis/axes
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Mean result
    """
    if arr.size == 0:
        return Fuzznum('fs').create(md=0.0)
    
    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    # Calculate element count for averaging
    if axis is None:
        n = arr.size
    else:
        n = np.prod([arr.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])

    # Compute fuzzy sum and divide by count
    total = _fs_sum(arr, axis=axis)
    
    # For FS, division by scalar is straightforward
    if isinstance(total, Fuzznum):
        return Fuzznum('fs').create(md=total.md / n)
    else:
        mds, = total.backend.get_component_arrays()
        backend_cls = total.backend.__class__
        new_backend = backend_cls.from_arrays(mds=mds / n)
        return Fuzzarray(backend=new_backend)


def _fs_max(arr: Union[Fuzznum, Fuzzarray],
            axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance max for FS based on membership degree.
    
    For classical fuzzy sets, the maximum is determined by the highest
    membership degree, which directly represents the strongest membership.
    
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        axis (Union[int, Tuple[int, ...]]): Reduction axis/axes
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Maximum element(s)
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("max() arg is an empty sequence")

    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    # Get membership degrees for comparison
    mds, = arr.backend.get_component_arrays()
    indices = np.argmax(mds, axis=axis)

    if axis is None:
        # Flatten and select maximum element
        from ....mixin.factory import _flatten_factory
        return _flatten_factory(arr)[indices]
    else:
        # Handle axis-specific maximum
        if isinstance(axis, tuple):
            raise NotImplementedError("max with tuple axis is not yet supported.")

        # Create index grid for selection
        grid = np.indices(indices.shape)
        idx_list = list(grid)
        idx_list.insert(axis, indices)

        return arr[tuple(idx_list)]


def _fs_min(arr: Union[Fuzznum, Fuzzarray],
            axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance min for FS based on membership degree.
    
    For classical fuzzy sets, the minimum is determined by the lowest
    membership degree, representing the weakest membership.
    
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        axis (Union[int, Tuple[int, ...]]): Reduction axis/axes
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Minimum element(s)
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("min() arg is an empty sequence")

    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    # Get membership degrees for comparison
    mds, = arr.backend.get_component_arrays()
    indices = np.argmin(mds, axis=axis)

    if axis is None:
        # Flatten and select minimum element
        from ....mixin.factory import _flatten_factory
        return _flatten_factory(arr)[indices]
    else:
        # Handle axis-specific minimum
        if isinstance(axis, tuple):
            raise NotImplementedError("min with tuple axis is not yet supported.")

        # Create index grid for selection
        grid = np.indices(indices.shape)
        idx_list = list(grid)
        idx_list.insert(axis, indices)

        return arr[tuple(idx_list)]


def _fs_prod(arr: Union[Fuzznum, Fuzzarray],
             axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance product for FS using t-norm reduction.
    
    For classical fuzzy sets, the product is implemented using t-norm
    (fuzzy AND) operation, representing the intersection of all fuzzy sets.
    
    Mathematical formulation:
        Prod(A₁, A₂, ..., Aₙ) = T(μ₁, μ₂, ..., μₙ)
        where T is a t-norm operator
        
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        axis (Union[int, Tuple[int, ...]]): Reduction axis/axes
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Product result
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError('Cannot compute product of empty array')

    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    # Get t-norm operation for fuzzy intersection
    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, **params)

    # Get membership degrees from backend
    mds, = arr.backend.get_component_arrays()

    # Apply t-norm reduction (fuzzy intersection)
    md_prod = tnorm.t_norm_reduce(mds, axis=axis)

    # Return appropriate type based on axis parameter
    if axis is None:
        return Fuzznum('fs').create(md=md_prod)
    else:
        backend_cls = arr.backend.__class__
        new_backend = backend_cls.from_arrays(mds=md_prod)
        return Fuzzarray(backend=new_backend)


def _fs_var(arr: Union[Fuzznum, Fuzzarray],
            axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance variance for FS using fuzzy arithmetic.
    
    Computes fuzzy variance as: Var(X) = E[(X - E[X])²]
    where operations are performed using fuzzy arithmetic.
    
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        axis (Union[int, Tuple[int, ...]]): Reduction axis/axes
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Variance result
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("var() arg is an empty sequence")

    if arr.size == 1:
        # Single element has zero variance
        return Fuzznum('fs').create(md=0.0)

    # Compute mean
    mean_val = _fs_mean(arr, axis=axis)

    # Compute deviations from mean
    diff = arr - mean_val

    # Square the deviations (using fuzzy multiplication)
    squared_diff = diff * diff

    # Return mean of squared deviations
    return _fs_mean(squared_diff, axis=axis)


def _fs_std(arr: Union[Fuzznum, Fuzzarray],
            axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance standard deviation for FS.
    
    Computes fuzzy standard deviation as the square root of variance.
    
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        axis (Union[int, Tuple[int, ...]]): Reduction axis/axes
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Standard deviation result
        
    Raises:
        ValueError: If array is empty
    """
    if arr.size == 0:
        raise ValueError("std() arg is an empty sequence")

    if arr.size == 1:
        # Single element has zero standard deviation
        return Fuzznum('fs').create(md=0.0)

    # Compute variance and take square root
    variance = _fs_var(arr, axis=axis)
    return variance ** 0.5


def _fs_score(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance score calculation for FS.
    
    For classical fuzzy sets, the score is simply the membership degree,
    as it directly represents the strength of membership in the set.
    
    Mathematical formulation:
        Score(A) = μ_A
        
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        
    Returns:
        Union[float, np.ndarray]: Score value(s)
    """
    if isinstance(arr, Fuzznum):
        # Single Fuzznum score
        return arr.md

    # Fuzzarray vectorized computation
    mds, = arr.backend.get_component_arrays()
    return mds.copy()  # Return copy to avoid backend modification


def _fs_acc(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance accuracy calculation for FS.
    
    For classical fuzzy sets, accuracy represents the degree of certainty.
    Since FS only has membership degrees, accuracy equals the membership degree.
    
    Mathematical formulation:
        Accuracy(A) = μ_A
        
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        
    Returns:
        Union[float, np.ndarray]: Accuracy value(s)
    """
    if isinstance(arr, Fuzznum):
        # Single Fuzznum accuracy
        return arr.md

    # Fuzzarray vectorized computation
    mds, = arr.backend.get_component_arrays()
    return mds.copy()  # Return copy to avoid backend modification


def _fs_ind(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance indeterminacy calculation for FS.
    
    For classical fuzzy sets, indeterminacy represents the degree of uncertainty.
    It is computed as the complement of the membership degree.
    
    Mathematical formulation:
        Indeterminacy(A) = 1 - μ_A
        
    Parameters:
        arr (Union[Fuzznum, Fuzzarray]): Input FS object(s)
        
    Returns:
        Union[float, np.ndarray]: Indeterminacy value(s)
    """
    if isinstance(arr, Fuzznum):
        # Single Fuzznum indeterminacy
        return 1.0 - arr.md

    # Fuzzarray vectorized computation
    mds, = arr.backend.get_component_arrays()
    return 1.0 - mds