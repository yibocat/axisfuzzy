#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 14:05
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import numpy as np
from typing import Optional, Dict, List, Union

from ...core import Fuzznum, Fuzzarray, get_registry_fuzztype
from ...fuzzifier import FuzzificationStrategy, register_fuzzifier


@register_fuzzifier(is_default=True)
class QROFNFuzzificationStrategy(FuzzificationStrategy):
    """
    QROFN 模糊化策略:
    - mf_params 只有一组参数时 → 返回 Fuzznum
    - mf_params 多组参数时 → 返回 Fuzzarray（包含多个 Fuzznum）
    """

    mtype = "qrofn"
    method = "default"

    def __init__(self, q: Optional[int] = None, pi: Optional[float] = None):
        super().__init__(q=q)
        self.pi = pi if pi is not None else 0.1
        if not (0 <= self.pi <= 1):
            raise ValueError("pi must be in [0,1]")

    def fuzzify(self,
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Fuzzarray:

        x = np.asarray(x, dtype=float)
        results = []

        for params in mf_params_list:
            mf = mf_cls(**params)

            # 矢量计算
            mds = np.clip(mf.compute(x), 0, 1)
            nmds = np.maximum(1 - mds**self.q - self.pi**self.q, 0.0) ** (1/self.q)

            backend_cls = get_registry_fuzztype().get_backend(self.mtype)
            backend = backend_cls.from_arrays(mds=mds, nmds=nmds, q=self.q)
            results.append(Fuzzarray(backend=backend, mtype=self.mtype, q=self.q))

        # 单参数 → 返回一组 Fuzzarray
        if len(results) == 1:
            return results[0]

        else:
            from ...mixin.factory import _stack_factory
            return _stack_factory(results[0], *results[1:], axis=0)

