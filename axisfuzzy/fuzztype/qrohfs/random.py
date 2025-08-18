#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 20:19
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Any, Dict, Tuple, Optional
import numpy as np

from ...config import get_config
from ...core import Fuzznum, Fuzzarray
from ...random import register_random
from ...random.base import ParameterizedRandomGenerator

from .backend import QROHFNBackend


@register_random
class QROHFNRandomGenerator(ParameterizedRandomGenerator):
    """
    High-performance random generator for q-rung orthopair hesitant fuzzy numbers (QROHFN).

    Key ideas:
    - Vectorized sampling by grouping elements with the same (md_count / nmd_count).
    - Constraint handling:
        * orthopair: nmd upper-bounded by (1 - max(md)^q)^(1/q) per element.
        * independent: sample freely then clamp per-element by the same upper bound if violated.
    - Object arrays (ragged rows) are assembled with a single O(N) assignment loop, avoiding per-element sampling loops.



    ---

    ### 1. **值分布相关参数（value distributions）**

    - **`md_dist`**
      - 说明：`md`（membership degree，隶属度）集合的分布类型。
      - 可选值：`'uniform'`、`'beta'`、`'normal'` 等。

    - **`md_low` / `md_high`**
      - 说明：`md` 取值的下界和上界（闭区间），即每个隶属度的范围。

    - **`nu_mode`**
      - 说明：非隶属度（`nmd`）的生成模式。
      - 可选值：`'orthopair'`（正交对约束）或 `'independent'`（独立采样）。

    - **`nu_dist`**
      - 说明：`nmd`（non-membership degree，非隶属度）集合的分布类型。
      - 可选值同上。

    - **`nu_low` / `nu_high`**
      - 说明：`nmd` 取值的下界和上界。

    - **`a` / `b`**
      - 说明：Beta 分布的形状参数（如果选用 beta 分布）。

    - **`loc` / `scale`**
      - 说明：正态分布的均值和标准差（如果选用 normal 分布）。

    ---

    ### 2. **可选分布参数分离（Optional split params）**

    - **`md_a` / `md_b` / `md_loc` / `md_scale`**
      - 说明：专门为 `md` 设置的分布参数。如果为 `None`，则回退用上面共享的 `a`、`b`、`loc`、`scale`。

    - **`nu_a` / `nu_b` / `nu_loc` / `nu_scale`**
      - 说明：专门为 `nmd` 设置的分布参数。同理。

    ---

    ### 3. **犹豫集长度控制（Hesitant set length controls）**

    - **`md_count_dist`**
      - 说明：`md` 集合长度的分布类型。
      - 可选值：`'uniform_int'`（均匀整数）、`'poisson'`（泊松分布）、`'fixed'`（固定长度）。

    - **`md_count_min` / `md_count_max`**
      - 说明：`md` 集合长度的最小值和最大值（用于 uniform_int 或 poisson 截断）。

    - **`md_count`**
      - 说明：如果 `md_count_dist='fixed'`，则所有 `md` 集合长度都为该值。

    - **`md_count_lam`**
      - 说明：如果 `md_count_dist='poisson'`，则泊松分布的 λ 参数。

    - **`nmd_count_dist` / `nmd_count_min` / `nmd_count_max` / `nmd_count` / `nmd_count_lam`**
      - 说明：同上，只不过是针对 `nmd` 集合。

    ---

    ### 4. **后处理选项（Post-process）**

    - **`sort_sets`**
      - 说明：是否对每个犹豫集（`md` 或 `nmd`）内部排序。

    - **`unique_sets`**
      - 说明：是否对每个犹豫集去重（即集合内元素唯一）。
    """

    mtype = "qrohfn"

    def get_default_parameters(self) -> Dict[str, Any]:
        # Shared defaults (same as qrofn) + QROHFN-specific count controls
        return {
            # value distributions (shared defaults; can be overridden per side)
            'md_dist': 'uniform',
            'md_low': 0.0,
            'md_high': 1.0,
            'nu_mode': 'orthopair',  # 'orthopair' or 'independent'
            'nu_dist': 'uniform',
            'nu_low': 0.0,
            'nu_high': 1.0,
            # Beta shared params
            'a': 2.0,
            'b': 2.0,
            # Normal shared params
            'loc': 0.5,
            'scale': 0.15,

            # Optional split params (fallback to shared if None)
            'md_a': None, 'md_b': None, 'md_loc': None, 'md_scale': None,
            'nu_a': None, 'nu_b': None, 'nu_loc': None, 'nu_scale': None,

            # Hesitant set length controls
            'md_count_dist': 'uniform_int',   # 'uniform_int' | 'poisson' | 'fixed'
            'md_count_min': 1,
            'md_count_max': 4,
            'md_count': None,                 # used when dist='fixed'
            'md_count_lam': None,             # used when dist='poisson'

            'nmd_count_dist': 'uniform_int',
            'nmd_count_min': 1,
            'nmd_count_max': 4,
            'nmd_count': None,
            'nmd_count_lam': None,

            # Post-process
            'sort_sets': True,   # sort each hesitant set ascending (fast, vectorizable)
            'unique_sets': False # deduplicate each set (slower; requires per-row unique)
        }

    def validate_parameters(self, q: int, **p) -> None:
        if not isinstance(q, int) or q <= 0:
            raise ValueError(f"q must be a positive integer, but got {q}")

        # range checks
        if p.get('md_low', 0.0) > p.get('md_high', 1.0):
            raise ValueError("md_low cannot be greater than md_high")
        if p.get('nu_low', 0.0) > p.get('nu_high', 1.0):
            raise ValueError("nu_low cannot be greater than nu_high")

        if 'nu_mode' in p and p['nu_mode'] not in ('orthopair', 'independent'):
            raise ValueError("nu_mode must be 'orthopair' or 'independent'")

        # count ranges
        for side in ('md', 'nmd'):
            cmin = p.get(f'{side}_count_min', 1)
            cmax = p.get(f'{side}_count_max', 1)
            if not (isinstance(cmin, int) and isinstance(cmax, int)):
                raise ValueError(f'{side}_count_min/max must be integers')
            if cmin <= 0 or cmax <= 0:
                raise ValueError(f'{side}_count_min/max must be positive')
            if cmin > cmax:
                raise ValueError(f'{side}_count_min cannot be greater than {side}_count_max')

            dist = p.get(f'{side}_count_dist', 'uniform_int')
            if dist not in ('uniform_int', 'poisson', 'fixed'):
                raise ValueError(f"{side}_count_dist must be 'uniform_int'|'poisson'|'fixed'")

            if dist == 'fixed':
                cfixed = p.get(f'{side}_count')
                if not (isinstance(cfixed, int) and cmin <= cfixed <= cmax):
                    raise ValueError(f"{side}_count must be an int in [{cmin}, {cmax}] when dist='fixed'")
            if dist == 'poisson':
                lam = p.get(f'{side}_count_lam')
                if lam is None or lam <= 0:
                    raise ValueError(f"{side}_count_lam must be a positive number when dist='poisson'")

    # ---------- helpers ----------

    @staticmethod
    def _dist_params(side_prefix: str, p: Dict[str, Any]) -> Dict[str, Any]:
        # Build distribution parameter dict for 'md' or 'nu' side, with fallback to shared
        return {
            'a': p.get(f'{side_prefix}_a', p['a']),
            'b': p.get(f'{side_prefix}_b', p['b']),
            'loc': p.get(f'{side_prefix}_loc', p['loc']),
            'scale': p.get(f'{side_prefix}_scale', p['scale']),
        }

    @staticmethod
    def _sample_counts(rng: np.random.Generator, size: int, *, dist: str,
                       cmin: int, cmax: int, cfixed: Optional[int], lam: Optional[float]) -> np.ndarray:
        if dist == 'fixed':
            return np.full(size, int(cfixed), dtype=np.int32)
        elif dist == 'uniform_int':
            # inclusive high: add 1
            return rng.integers(cmin, cmax + 1, size=size, dtype=np.int32)
        else:  # 'poisson'
            raw = rng.poisson(lam, size=size)
            # clip to [cmin, cmax], and ensure >=1
            raw = np.clip(raw, cmin, cmax).astype(np.int32)
            return raw

    def _build_object_array_from_rows(self, rows: np.ndarray) -> np.ndarray:
        """
        rows: list-like of 1D ndarrays (shape: (N,), dtype=object-compatible)
        returns: object dtype 1D array of length len(rows)
        """
        out = np.empty(len(rows), dtype=object)
        # Assign by row (fast; O(N) lightweight loop)
        for i, r in enumerate(rows):
            out[i] = r
        return out

    # ---------- single fuzznum ----------

    def fuzznum(self,
                rng: np.random.Generator,
                q: Optional[int] = None,
                **kwargs) -> Fuzznum:

        p = self._merge_parameters(**kwargs)
        q = q if q is not None else get_config().DEFAULT_Q
        self.validate_parameters(q=q, **p)

        # sample counts
        md_count = self._sample_counts(
            rng, 1,
            dist=p['md_count_dist'], cmin=p['md_count_min'], cmax=p['md_count_max'],
            cfixed=p['md_count'], lam=p['md_count_lam']
        )[0]
        nmd_count = self._sample_counts(
            rng, 1,
            dist=p['nmd_count_dist'], cmin=p['nmd_count_min'], cmax=p['nmd_count_max'],
            cfixed=p['nmd_count'], lam=p['nmd_count_lam']
        )[0]

        # sample md values
        md_params = self._dist_params('md', p)
        md_vals = self._sample_from_distribution(
            rng, size=md_count, dist=p['md_dist'], low=p['md_low'], high=p['md_high'], **md_params
        )
        if p['sort_sets']:
            md_vals = np.sort(md_vals)
        if p['unique_sets']:
            md_vals = np.unique(md_vals)

        md_max = float(md_vals.max()) if md_vals.size else 0.0
        allow_nmd_max = float((1.0 - md_max ** q) ** (1.0 / q))

        nu_params = self._dist_params('nu', p)
        if p['nu_mode'] == 'orthopair':
            # dynamic upper bound; if below nu_low, force to the upper bound
            eff_high = min(p['nu_high'], allow_nmd_max)
            if eff_high <= 0.0 or eff_high <= p['nu_low']:
                nmd_vals = np.full(nmd_count, eff_high, dtype=np.float64)
            else:
                base = self._sample_from_distribution(
                    rng, size=nmd_count, dist=p['nu_dist'], low=0.0, high=1.0, **nu_params
                )
                nmd_vals = p['nu_low'] + base * (eff_high - p['nu_low'])
                nmd_vals = np.clip(nmd_vals, 0.0, eff_high)
        else:
            nmd_vals = self._sample_from_distribution(
                rng, size=nmd_count, dist=p['nu_dist'], low=p['nu_low'], high=p['nu_high'], **nu_params
            )
            # clamp if violates
            if nmd_vals.size:
                nmd_max = float(nmd_vals.max())
                if md_max ** q + nmd_max ** q > 1.0:
                    nmd_vals = np.minimum(nmd_vals, allow_nmd_max)

        if p['sort_sets']:
            nmd_vals = np.sort(nmd_vals)
        if p['unique_sets']:
            nmd_vals = np.unique(nmd_vals)

        return Fuzznum(mtype='qrohfn', q=q).create(md=md_vals, nmd=nmd_vals)

    # ---------- batch fuzzarray ----------

    def fuzzarray(self,
                  rng: np.random.Generator,
                  shape: Tuple[int, ...],
                  q: Optional[int] = None,
                  **params) -> Fuzzarray:

        p = self._merge_parameters(**params)
        q = q if q is not None else get_config().DEFAULT_Q
        self.validate_parameters(q=q, **p)

        size = int(np.prod(shape))

        # 1) sample counts for each element (vectorized)
        md_counts = self._sample_counts(
            rng, size,
            dist=p['md_count_dist'], cmin=p['md_count_min'], cmax=p['md_count_max'],
            cfixed=p['md_count'], lam=p['md_count_lam']
        )
        nmd_counts = self._sample_counts(
            rng, size,
            dist=p['nmd_count_dist'], cmin=p['nmd_count_min'], cmax=p['nmd_count_max'],
            cfixed=p['nmd_count'], lam=p['nmd_count_lam']
        )

        # 2) sample md for each unique md_count (vectorized by group)
        md_params = self._dist_params('md', p)
        md_rows = [None] * size  # type: ignore[var-annotated]
        md_maxs = np.zeros(size, dtype=np.float64)

        unique_md_counts = np.unique(md_counts)
        for c in unique_md_counts:
            idx = np.where(md_counts == c)[0]
            if c == 0 or idx.size == 0:
                continue
            # matrix of shape (len(idx), c)
            mat = self._sample_from_distribution(
                rng, size=(idx.size * c), dist=p['md_dist'], low=p['md_low'], high=p['md_high'], **md_params
            ).reshape(idx.size, c)
            if p['sort_sets']:
                mat.sort(axis=1)
            # fill rows and max
            md_maxs[idx] = mat.max(axis=1) if c > 0 else 0.0
            # assign object rows
            for j, row in enumerate(mat):
                md_rows[idx[j]] = row if not p['unique_sets'] else np.unique(row)

        md_obj = self._build_object_array_from_rows(np.array(md_rows, dtype=object))

        # 3) sample nmd by groups; handle constraint per element
        nu_params = self._dist_params('nu', p)
        nmd_rows = [None] * size  # type: ignore[var-annotated]

        if p['nu_mode'] == 'orthopair':
            # element-wise upper bound
            allow_max = np.power(1.0 - np.power(md_maxs, q), 1.0 / q)
            allow_max = np.clip(allow_max, 0.0, 1.0)

            unique_nmd_counts = np.unique(nmd_counts)
            for c in unique_nmd_counts:
                idx = np.where(nmd_counts == c)[0]
                if c == 0 or idx.size == 0:
                    continue

                # per-row effective high: min(nu_high, allow_max[row])
                eff_high = np.minimum(p['nu_high'], allow_max[idx])
                # handle rows where eff_high <= nu_low: directly fill with eff_high
                mask_hard = eff_high <= p['nu_low']
                hard_rows = idx[mask_hard]
                for k in hard_rows:
                    nmd_rows[k] = np.full(c, eff_high[mask_hard][np.where(hard_rows == k)[0][0]], dtype=np.float64)

                soft_rows = idx[~mask_hard]
                if soft_rows.size > 0:
                    # sample in [0,1], then scale row-wise to [nu_low, eff_high[row]]
                    mat = self._sample_from_distribution(
                        rng, size=(soft_rows.size * c), dist=p['nu_dist'], low=0.0, high=1.0, **nu_params
                    ).reshape(soft_rows.size, c)
                    row_high = eff_high[~mask_hard]  # shape (soft_rows.size,)
                    mat = p['nu_low'] + mat * (row_high[:, None] - p['nu_low'])
                    # numeric safety
                    mat = np.clip(mat, 0.0, row_high[:, None])

                    if p['sort_sets']:
                        mat.sort(axis=1)
                    # assign rows
                    for j, row in enumerate(mat):
                        nmd_rows[soft_rows[j]] = row if not p['unique_sets'] else np.unique(row)

        else:  # 'independent'
            unique_nmd_counts = np.unique(nmd_counts)
            for c in unique_nmd_counts:
                idx = np.where(nmd_counts == c)[0]
                if c == 0 or idx.size == 0:
                    continue
                # sample free
                mat = self._sample_from_distribution(
                    rng, size=(idx.size * c), dist=p['nu_dist'], low=p['nu_low'], high=p['nu_high'], **nu_params
                ).reshape(idx.size, c)
                # clamp per-row if violates
                # compute nmd_maxs for rows
                nmd_maxs = mat.max(axis=1) if c > 0 else np.zeros(idx.size)
                md_maxs_sel = md_maxs[idx]
                violates = (np.power(md_maxs_sel, q) + np.power(nmd_maxs, q)) > 1.0
                if np.any(violates):
                    allow = np.power(1.0 - np.power(md_maxs_sel[violates], q), 1.0 / q)
                    mat_v = mat[violates]
                    # clip all entries in violating rows to per-row 'allow'
                    mat[violates] = np.minimum(mat_v, allow[:, None])

                if p['sort_sets']:
                    mat.sort(axis=1)
                # assign rows
                for j, row in enumerate(mat):
                    nmd_rows[idx[j]] = row if not p['unique_sets'] else np.unique(row)

        nmd_obj = self._build_object_array_from_rows(np.array(nmd_rows, dtype=object))

        # 4) reshape to shape and build backend
        mds = md_obj.reshape(shape)
        nmds = nmd_obj.reshape(shape)
        backend = QROHFNBackend.from_arrays(mds=mds, nmds=nmds, q=q)
        return Fuzzarray(backend=backend)
