#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Union

import numpy as np

from ....core import Fuzznum, Fuzzarray


def _qrofn_distance(
        fuzz_1: Union[Fuzznum, Fuzzarray],
        fuzz_2: Union[Fuzznum, Fuzzarray],
        p_l: int = 2,
        indeterminacy=True) -> Union[np.ndarray, float]:
    """High-performance distance calculation."""

    if fuzz_1.q != fuzz_2.q:
        raise ValueError("Both Fuzzarrays must have the same q value for distance calculation. "
                         f"fuzz_1.q: {fuzz_1.q}, fuzz_2.q: {fuzz_2.q}")
    if fuzz_1.mtype != fuzz_2.mtype:
        raise ValueError("Both Fuzzarrays must have the same mtype for distance calculation. "
                         f"fuzz_1.mtype: {fuzz_1.mtype}, fuzz_2.mtype: {fuzz_2.mtype}")
    q = fuzz_1.q
    # Handle different input combinations
    if isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzzarray):
        # Both are Fuzzarrays - vectorized computation

        mds1, nmds1 = fuzz_1.backend.get_component_arrays()
        mds2, nmds2 = fuzz_2.backend.get_component_arrays()

        # Vectorized indeterminacy calculation
        pi1 = (1 - mds1 ** q - nmds1 ** q) ** (1 / q)
        pi2 = (1 - mds2 ** q - nmds2 ** q) ** (1 / q)
        pi = np.abs(pi1 ** q - pi2 ** q) ** p_l

        if indeterminacy:
            distance = (0.5 * (np.abs(mds1 ** q - mds2 ** q) ** p_l +
                               np.abs(nmds1 ** q - nmds2 ** q) ** p_l + pi)) ** (1 / p_l)
        else:
            distance = (0.5 * (np.abs(mds1 ** q - mds2 ** q) ** p_l +
                               np.abs(nmds1 ** q - nmds2 ** q) ** p_l)) ** (1 / p_l)
        return distance

    elif isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzznum):
        # Fuzzarray vs Fuzznum - broadcast Fuzznum
        mds1, nmds1 = fuzz_1.backend.get_component_arrays()
        md2, nmd2 = fuzz_2.md, fuzz_2.nmd

        pi1 = (1 - mds1 ** q - nmds1 ** q) ** (1 / q)
        pi2 = (1 - md2 ** q - nmd2 ** q) ** (1 / q)
        pi = np.abs(pi1 ** q - pi2 ** q) ** p_l

        if indeterminacy:
            distance = (0.5 * (np.abs(mds1 ** q - md2 ** q) ** p_l +
                               np.abs(nmds1 ** q - nmd2 ** q) ** p_l + pi)) ** (1 / p_l)
        else:
            distance = (0.5 * (np.abs(mds1 ** q - md2 ** q) ** p_l +
                               np.abs(nmds1 ** q - nmd2 ** q) ** p_l)) ** (1 / p_l)
        return distance
    elif isinstance(fuzz_1, Fuzznum) and isinstance(fuzz_2, Fuzzarray):
        # Fuzznum vs Fuzzarray - swap and recurse
        return _qrofn_distance(fuzz_2, fuzz_1, p_l, indeterminacy)

    else:
        # Both are Fuzznums - fallback to original implementation
        pi1 = fuzz_1.ind
        pi2 = fuzz_2.ind
        pi = np.abs(pi1 ** q - pi2 ** q) ** p_l

        if indeterminacy:
            distance = (0.5 * (np.abs(fuzz_1.md ** q - fuzz_2.md ** q) ** p_l +
                               np.abs(fuzz_1.nmd ** q - fuzz_2.nmd ** q) ** p_l + pi)) ** (1 / p_l)
        else:
            distance = (0.5 * (np.abs(fuzz_1.md ** q - fuzz_2.md ** q) ** p_l +
                               np.abs(fuzz_1.nmd ** q - fuzz_2.nmd ** q) ** p_l)) ** (1 / p_l)

        return distance.item() if isinstance(distance, np.floating) else distance
