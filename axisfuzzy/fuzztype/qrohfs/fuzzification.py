#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 14:06
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import numpy as np
from typing import Optional, Dict, List, Any, Union

from ...config import get_config
from ...core import Fuzznum, Fuzzarray, get_registry_fuzztype
from ...fuzzifier import FuzzificationStrategy, register_fuzzifier


@register_fuzzifier(is_default=True)
class QROHFNFuzzificationStrategy(FuzzificationStrategy):
    """
    QROHFN 模糊化策略（hesitant fuzzy numbers）
    - 标量 → 返回单个 Fuzznum，内部存储 list/mds
    - 数组 → 返回 Fuzzarray，元素是 Fuzznum（每个含有 list/mds）
    - 多个 mf_params 永远融合为单个 hesitant
    """
    mtype = 'qrohfn'
    method = 'default'

    def __init__(self,
                 q: Optional[int] = None,
                 pi: Optional[float] = None,
                 nmd_generation_mode: str = "pi_based"):
        q = q if q is not None else get_config().DEFAULT_Q
        super().__init__(q=q)
        self.pi = pi if pi is not None else 0.1
        self.nmd_generation_mode = nmd_generation_mode

        if not (0 <= self.pi <= 1):
            raise ValueError("pi must be in [0,1]")
        if self.nmd_generation_mode not in {"pi_based", "proportional", "uniform"}:
            raise ValueError("Invalid nmd_generation_mode")

    def fuzzify(self,
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:

        x = np.asarray(x, dtype=float)
        shape = x.shape if x.shape else ()

        # 用 object 数组保存 hesitant 数据
        mds_obj = np.empty(shape, dtype=object)
        nmds_obj = np.empty(shape, dtype=object)

        for idx in np.ndindex(shape if shape else (1,)):
            xi = x[idx] if shape else x.item()
            mds = [float(np.clip(mf_cls(**p).compute(xi), 0, 1)) for p in mf_params_list]
            nmds = self._compute_nmds(np.asarray(mds))
            if shape:
                mds_obj[idx] = np.asarray(mds, dtype=float)
                nmds_obj[idx] = np.asarray(nmds, dtype=float)
            else:
                mds_obj = np.asarray(mds, dtype=object)
                nmds_obj = np.asarray(nmds, dtype=object)

        backend_cls = get_registry_fuzztype().get_backend(self.mtype)
        backend = backend_cls.from_arrays(mds=mds_obj, nmds=nmds_obj, q=self.q)
        arr = Fuzzarray(backend=backend, mtype=self.mtype, q=self.q)

        # 标量情况 → 返回 Fuzznum
        if x.ndim == 0 or x.size == 1:
            return arr[()]  # 返回单元素
        return arr

    def _compute_nmds(self, mds: np.ndarray) -> np.ndarray:
        # TODO: 找一种生成非隶属度的方法
        if self.nmd_generation_mode == "pi_based":
            nmds = []
            for md in mds:
                available = max(1 - md**self.q - self.pi**self.q, 0)
                nmd_i = min(self.pi, available**(1/self.q))
                nmds.append(nmd_i)
            return np.array(nmds)

        elif self.nmd_generation_mode == "proportional":
            nmds = []
            for md in mds:
                max_possible = (1 - md**self.q)**(1/self.q)
                scale = min(self.pi, max_possible) / (1 - md) if md < 1 else 0.0
                nmd_i = (1 - md) * scale
                nmds.append(min(nmd_i, max_possible))
            return np.array(nmds)

        elif self.nmd_generation_mode == "uniform":
            nmds = []
            for md in mds:
                max_possible = (1 - md**self.q)**(1/self.q)
                target = min(self.pi, max_possible)
                # 添加扰动避免全相等
                jitter = target * (0.9 + 0.2 * np.random.rand())
                nmds.append(min(jitter, max_possible))
            return np.array(nmds)

        else:
            raise ValueError(f"Unknown nmd_generation_mode {self.nmd_generation_mode}")

