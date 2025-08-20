#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 20:24
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


import numpy as np
from typing import Union
from axisfuzzy.core import Fuzznum, Fuzzarray
from .normalize import _normalize_qrohfn


def _qrohfn_distance(
    fuzz_1: Union[Fuzznum, Fuzzarray],
    fuzz_2: Union[Fuzznum, Fuzzarray],
    gamma: int = 2,
    tao: float = 0.5,
    indeterminacy: bool = True
) -> Union[np.ndarray, float]:
    """
    Compute distance between two QROHFN-type fuzzy numbers or fuzzarrays.

    Args:
        fuzz_1: QROHFN Fuzznum or Fuzzarray
        fuzz_2: QROHFN Fuzznum or Fuzzarray
        gamma: Exponent γ (default 2)
        tao: Risk factor τ ∈ [0,1] for normalization
        indeterminacy: Whether to include indeterminacy π term

    Returns:
        float if inputs are Fuzznum, else np.ndarray of same shape if inputs are Fuzzarray.
    """

    # verify type
    if isinstance(fuzz_1, Fuzznum) and isinstance(fuzz_2, Fuzznum):
        return _qrohfn_distance_fuzznum_pair(fuzz_1, fuzz_2, gamma, tao, indeterminacy)

    elif isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzzarray):
        if fuzz_1.shape != fuzz_2.shape:
            raise ValueError("Fuzzarray shapes must match for distance calculation.")
        if fuzz_1.q != fuzz_2.q:
            raise ValueError("q mismatch: fuzz_1.q={}, fuzz_2.q={}".format(fuzz_1.q, fuzz_2.q))

        # 高性能: 遍历索引, 但每对 f1,f2 内部完全用 NumPy 向量运算
        out = np.empty(fuzz_1.shape, dtype=np.float64)
        for idx in np.ndindex(fuzz_1.shape):
            f1 = fuzz_1.backend.get_fuzznum_view(idx)
            f2 = fuzz_2.backend.get_fuzznum_view(idx)
            out[idx] = _qrohfn_distance_fuzznum_pair(f1, f2, gamma, tao, indeterminacy)
        return out

    elif isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzznum):
        out = np.empty(fuzz_1.shape, dtype=np.float64)
        for idx in np.ndindex(fuzz_1.shape):
            f1 = fuzz_1.backend.get_fuzznum_view(idx)
            out[idx] = _qrohfn_distance_fuzznum_pair(f1, fuzz_2, gamma, tao, indeterminacy)
        return out

    elif isinstance(fuzz_1, Fuzznum) and isinstance(fuzz_2, Fuzzarray):
        return _qrohfn_distance(fuzz_2, fuzz_1, gamma, tao, indeterminacy)

    else:
        raise TypeError("Unsupported input types for _qrohfn_distance.")


def _qrohfn_distance_fuzznum_pair(f1: Fuzznum, f2: Fuzznum,
                                  gamma: int, tao: float, indeterminacy: bool) -> float:
    """单对 QROHFN Fuzznum 的距离计算 (内部用 NumPy 高效计算)."""
    if f1.mtype != "qrohfn" or f2.mtype != "qrohfn":
        raise TypeError("_qrohfn_distance only supports QROHFN fuzznums.")
    if f1.q != f2.q:
        raise ValueError("Both fuzznums must have the same q.")

    q = f1.q

    # === 1. pairwise normalization ===
    f1n, f2n = _normalize_qrohfn(f1, f2, tao=tao)

    md1 = np.sort(np.asarray(f1n.md, dtype=np.float64))
    md2 = np.sort(np.asarray(f2n.md, dtype=np.float64))

    nmd1 = np.sort(np.asarray(f1n.nmd, dtype=np.float64))
    nmd2 = np.sort(np.asarray(f2n.nmd, dtype=np.float64))

    # === 2. 批量差值计算 ===
    md_diff = np.mean(np.abs(md1**q - md2**q) ** gamma)
    nmd_diff = np.mean(np.abs(nmd1**q - nmd2**q) ** gamma)

    # === 3. indeterminacy ===
    if indeterminacy:
        pi1 = (1 - np.max(md1) ** q - np.max(nmd1) ** q) ** (1.0 / q)
        pi2 = (1 - np.max(md2) ** q - np.max(nmd2) ** q) ** (1.0 / q)
        pi_diff = np.abs(pi1**q - pi2**q) ** gamma
    else:
        pi_diff = 0.0

    # === 4. combine ===
    return float((0.5 * (md_diff + nmd_diff + pi_diff)) ** (1.0 / gamma))
