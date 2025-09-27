#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) Measurement Extension Methods.

This module implements distance and similarity measures for classical fuzzy sets (FS),
providing efficient calculations based on Zadeh's fuzzy set theory. All measurements
are optimized for both scalar and vectorized operations.

Mathematical Foundation:
    For classical fuzzy sets A and B with membership functions μ_A and μ_B,
    various distance metrics can be applied. The most common include:
    - Hamming distance: |μ_A - μ_B|
    - Euclidean distance: (|μ_A - μ_B|^p)^(1/p)
    - Minkowski distance: Generalization with parameter p
"""

from typing import Union

import numpy as np

from ....core import Fuzznum, Fuzzarray


def _fs_distance(
        fuzz_1: Union[Fuzznum, Fuzzarray],
        fuzz_2: Union[Fuzznum, Fuzzarray],
        p_l: int = 2) -> Union[np.ndarray, float]:
    """
    Calculate distance between FS objects using generalized distance formula.
    
    For classical fuzzy sets, the generalized distance is computed as:
    d(A, B) = (1/n * Σ|μ_A(x_i) - μ_B(x_i)|^p_l)^(1/p_l)
    
    This maintains consistency with QROFN and QROHFN distance formulations
    while adapting to the simpler FS structure (only membership degrees).
    
    Parameters:
        fuzz_1 (Union[Fuzznum, Fuzzarray]): First FS object
        fuzz_2 (Union[Fuzznum, Fuzzarray]): Second FS object  
        p_l (int): Minkowski distance parameter (p=1: Manhattan, p=2: Euclidean)
        
    Returns:
        Union[np.ndarray, float]: Distance value(s)
        
    Raises:
        ValueError: If mtype mismatch between objects
        TypeError: If invalid input types
        
    Examples:
        >>> fs1 = af.fuzzynum(md=0.8, mtype='fs')
        >>> fs2 = af.fuzzynum(md=0.6, mtype='fs')
        >>> distance = _fs_distance(fs1, fs2, p_l=2)  # Generalized Euclidean distance
        >>> print(distance)  # 0.2
    """
    # Validate input types and consistency
    if fuzz_1.mtype != fuzz_2.mtype:
        raise ValueError("Both objects must have the same mtype for distance calculation. "
                         f"fuzz_1.mtype: {fuzz_1.mtype}, fuzz_2.mtype: {fuzz_2.mtype}")
    
    if fuzz_1.mtype != 'fs':
        raise ValueError(f"Expected FS objects, got mtype '{fuzz_1.mtype}'")
    
    if p_l <= 0:
        raise ValueError(f"Distance parameter p_l must be positive, got {p_l}")

    # Handle different input combinations for optimal performance
    if isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzzarray):
        # Both are Fuzzarrays - vectorized generalized distance computation
        mds1, = fuzz_1.backend.get_component_arrays()
        mds2, = fuzz_2.backend.get_component_arrays()
        
        # Generalized distance: (1/n * Σ|μ_A - μ_B|^p)^(1/p)
        # For arrays, compute element-wise then aggregate
        diff_powers = np.abs(mds1 - mds2) ** p_l
        
        # Apply mean aggregation and final power
        distance = (np.mean(diff_powers, axis=None if mds1.ndim == 1 else tuple(range(1, mds1.ndim)))) ** (1 / p_l)
            
        return distance

    elif isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzznum):
        # Fuzzarray vs Fuzznum - broadcast Fuzznum with generalized distance
        mds1, = fuzz_1.backend.get_component_arrays()
        md2 = fuzz_2.md
        
        # Generalized distance calculation with broadcasting
        diff_powers = np.abs(mds1 - md2) ** p_l
        
        # Apply mean aggregation and final power
        distance = (np.mean(diff_powers, axis=None if mds1.ndim == 1 else tuple(range(1, mds1.ndim)))) ** (1 / p_l)
            
        return distance
        
    elif isinstance(fuzz_1, Fuzznum) and isinstance(fuzz_2, Fuzzarray):
        # Fuzznum vs Fuzzarray - swap and recurse for consistency
        return _fs_distance(fuzz_2, fuzz_1, p_l)

    else:
        # Both are Fuzznums - scalar generalized distance computation
        # For single elements, generalized distance reduces to simple difference
        distance = (np.abs(fuzz_1.md - fuzz_2.md) ** p_l) ** (1 / p_l)
            
        # Ensure scalar return for scalar inputs
        return float(distance) if isinstance(distance, (np.floating, np.integer)) else distance