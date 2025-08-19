#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 20:37
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


import numpy as np
from typing import Tuple
from axisfuzzy.core import Fuzznum


def _normalize_qrohfn(f1: Fuzznum, f2: Fuzznum, tao: float = 0.5) -> Tuple[Fuzznum, Fuzznum]:
    """
    Normalize two QROHFN-type Fuzznum objects so that their md/nmd sets
    have equal lengths (padding rule determined by risk factor tao).

    Args:
        f1 (Fuzznum): First QROHFN fuzzy number.
        f2 (Fuzznum): Second QROHFN fuzzy number.
        tao (float, optional): Risk factor τ ∈ [0,1].
            - 0.0 → pessimistic (pad with min)
            - 1.0 → optimistic (pad with max)
            - (0,1) → linear combination of min/max

    Returns:
        Tuple[Fuzznum, Fuzznum]: (normalized f1, normalized f2) with equal set lengths.
    """
    if f1.mtype != "qrohfn" or f2.mtype != "qrohfn":
        raise TypeError("_normalize_qrohfn only supports QROHFN fuzzy numbers")

    def pad_set(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensures arr1 and arr2 have the same length by padding the shorter one."""
        n1, n2 = len(arr1), len(arr2)
        if n1 == n2:
            return arr1.copy(), arr2.copy()

        def get_fill_value(arr: np.ndarray) -> float:
            if arr.size == 0:
                return 0.0  # edge case: empty hesitant set
            if tao == 0.0:
                return float(np.min(arr))
            elif tao == 1.0:
                return float(np.max(arr))
            else:
                return float(tao * np.max(arr) + (1 - tao) * np.min(arr))

        if n1 < n2:
            fill_value = get_fill_value(arr1 if arr1.size > 0 else arr2)
            arr1 = np.concatenate([arr1, np.full(n2 - n1, fill_value)])
        else:
            fill_value = get_fill_value(arr2 if arr2.size > 0 else arr1)
            arr2 = np.concatenate([arr2, np.full(n1 - n2, fill_value)])

        return arr1, arr2

    # Normalize membership degrees
    md1 = np.asarray(f1.md, dtype=np.float64).flatten()
    md2 = np.asarray(f2.md, dtype=np.float64).flatten()
    md1_new, md2_new = pad_set(md1, md2)

    # Normalize non-membership degrees
    nmd1 = np.asarray(f1.nmd, dtype=np.float64).flatten()
    nmd2 = np.asarray(f2.nmd, dtype=np.float64).flatten()
    nmd1_new, nmd2_new = pad_set(nmd1, nmd2)

    # Construct new Fuzznums
    nf1 = Fuzznum(mtype=f1.mtype, q=f1.q).create(md=md1_new, nmd=nmd1_new)
    nf2 = Fuzznum(mtype=f2.mtype, q=f2.q).create(md=md2_new, nmd=nmd2_new)
    return nf1, nf2


if __name__ == "__main__":
    # Example usage
    f1 = Fuzznum(mtype="qrohfn", q=3).create(md=[0.1, 0.2], nmd=[0.3, 0.4])
    f2 = Fuzznum(mtype="qrohfn", q=3).create(md=[0.5], nmd=[0.6, 0.7])
    normalized_f1, normalized_f2 = _normalize_qrohfn(f1, f2, tao=0.5)
    print(format(normalized_f1, 'r'))
    print(format(normalized_f2, 'r'))
