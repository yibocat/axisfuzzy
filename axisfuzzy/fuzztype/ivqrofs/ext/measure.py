#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
IVQROFN Measurement Extension Methods.

This module implements distance calculation and similarity measurement functions
for Interval-Valued Q-Rung Orthopair Fuzzy Numbers with high-performance
vectorized operations.
"""

from typing import Union

import numpy as np

from ....core import Fuzznum, Fuzzarray


def _ivqrofn_distance(
        fuzz_1: Union[Fuzznum, Fuzzarray],
        fuzz_2: Union[Fuzznum, Fuzzarray],
        p_l: int = 2,
        indeterminacy: bool = True) -> Union[np.ndarray, float]:
    """
    Calculate distance between two IVQROFN objects.
    
    For IVQROFNs, the distance formula considers both lower and upper bounds
    of membership and non-membership intervals, with optional indeterminacy.
    
    Distance formula:
    d(A,B) = [(1/4) * Σ(|md_A^i - md_B^i|^p + |nmd_A^i - nmd_B^i|^p)]^(1/p)
    where i ∈ {lower, upper} and indeterminacy is optionally included.
    
    Parameters:
        fuzz_1: First IVQROFN object
        fuzz_2: Second IVQROFN object
        p_l: Minkowski distance parameter (default: 2 for Euclidean)
        indeterminacy: Whether to include indeterminacy in calculation
        
    Returns:
        Union[np.ndarray, float]: Distance value(s)
        
    Raises:
        ValueError: If q-rung or mtype parameters don't match
    """
    # Validate compatibility
    if fuzz_1.q != fuzz_2.q:
        raise ValueError("Both IVQROFN objects must have the same q value for distance calculation. "
                         f"fuzz_1.q: {fuzz_1.q}, fuzz_2.q: {fuzz_2.q}")
    if fuzz_1.mtype != fuzz_2.mtype:
        raise ValueError("Both objects must have the same mtype for distance calculation. "
                         f"fuzz_1.mtype: {fuzz_1.mtype}, fuzz_2.mtype: {fuzz_2.mtype}")
    
    q = fuzz_1.q

    # Handle different input combinations with optimized vectorized computation
    if isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzzarray):
        # Both are Fuzzarrays - full vectorized computation
        mds1, nmds1 = fuzz_1.backend.get_component_arrays()
        mds2, nmds2 = fuzz_2.backend.get_component_arrays()

        # Calculate interval-based distance components
        # Membership distance: sum over lower and upper bounds
        md_dist = (np.abs(mds1[..., 0] ** q - mds2[..., 0] ** q) ** p_l +
                   np.abs(mds1[..., 1] ** q - mds2[..., 1] ** q) ** p_l)
        
        # Non-membership distance: sum over lower and upper bounds  
        nmd_dist = (np.abs(nmds1[..., 0] ** q - nmds2[..., 0] ** q) ** p_l +
                    np.abs(nmds1[..., 1] ** q - nmds2[..., 1] ** q) ** p_l)

        if indeterminacy:
            # Calculate indeterminacy for each interval (using upper bounds)
            pi1 = (1 - mds1[..., 1] ** q - nmds1[..., 1] ** q) ** (1 / q)
            pi2 = (1 - mds2[..., 1] ** q - nmds2[..., 1] ** q) ** (1 / q)
            pi_dist = np.abs(pi1 ** q - pi2 ** q) ** p_l
            
            # Total distance with indeterminacy
            distance = (0.25 * (md_dist + nmd_dist + pi_dist)) ** (1 / p_l)
        else:
            # Distance without indeterminacy
            distance = (0.25 * (md_dist + nmd_dist)) ** (1 / p_l)

        return distance

    elif isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzznum):
        # Fuzzarray vs Fuzznum - broadcast Fuzznum across array
        mds1, nmds1 = fuzz_1.backend.get_component_arrays()
        md2, nmd2 = fuzz_2.md, fuzz_2.nmd

        # Vectorized computation with broadcasting
        md_dist = (np.abs(mds1[..., 0] ** q - md2[0] ** q) ** p_l +
                   np.abs(mds1[..., 1] ** q - md2[1] ** q) ** p_l)
        
        nmd_dist = (np.abs(nmds1[..., 0] ** q - nmd2[0] ** q) ** p_l +
                    np.abs(nmds1[..., 1] ** q - nmd2[1] ** q) ** p_l)

        if indeterminacy:
            pi1 = (1 - mds1[..., 1] ** q - nmds1[..., 1] ** q) ** (1 / q)
            pi2 = (1 - md2[1] ** q - nmd2[1] ** q) ** (1 / q)
            pi_dist = np.abs(pi1 ** q - pi2 ** q) ** p_l
            
            distance = (0.25 * (md_dist + nmd_dist + pi_dist)) ** (1 / p_l)
        else:
            distance = (0.25 * (md_dist + nmd_dist)) ** (1 / p_l)

        return distance

    elif isinstance(fuzz_1, Fuzznum) and isinstance(fuzz_2, Fuzzarray):
        # Fuzznum vs Fuzzarray - swap and recurse for consistency
        return _ivqrofn_distance(fuzz_2, fuzz_1, p_l, indeterminacy)

    else:
        # Both are Fuzznums - scalar computation
        md1, nmd1 = fuzz_1.md, fuzz_1.nmd
        md2, nmd2 = fuzz_2.md, fuzz_2.nmd

        # Scalar interval distance calculation
        md_dist = (abs(md1[0] ** q - md2[0] ** q) ** p_l +
                   abs(md1[1] ** q - md2[1] ** q) ** p_l)
        
        nmd_dist = (abs(nmd1[0] ** q - nmd2[0] ** q) ** p_l +
                    abs(nmd1[1] ** q - nmd2[1] ** q) ** p_l)

        if indeterminacy:
            pi1 = (1 - md1[1] ** q - nmd1[1] ** q) ** (1 / q)
            pi2 = (1 - md2[1] ** q - nmd2[1] ** q) ** (1 / q)
            pi_dist = abs(pi1 ** q - pi2 ** q) ** p_l
            
            distance = (0.25 * (md_dist + nmd_dist + pi_dist)) ** (1 / p_l)
        else:
            distance = (0.25 * (md_dist + nmd_dist)) ** (1 / p_l)

        # Return scalar for Fuzznum-Fuzznum computation
        return distance.item() if isinstance(distance, np.floating) else distance