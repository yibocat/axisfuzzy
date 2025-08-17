#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Any, Optional, Union

import numpy as np

from ...config import get_config
from ...core import Fuzznum, Fuzzarray, get_registry_fuzztype
from ...fuzzify import FuzzificationStrategy, register_fuzzify
from ...membership import MembershipFunction


@register_fuzzify('qrofn', 'default')
class QROFNFuzzificationStrategy(FuzzificationStrategy):

    def __init__(self,
                 q: Optional[int] = None,
                 pi: Optional[float] = None):
        q = q if q is not None else get_config().DEFAULT_Q
        super().__init__(int(q), pi=pi)
        self.mtype = "qrofn"
        self.method = 'default'

    def fuzzify_scalar(self,
                       x: Optional[float],
                       mf: Optional[MembershipFunction] = None) -> 'Fuzznum':

        pi = self.kwargs.get('pi')
        if pi is None:
            raise ValueError("Parameter 'pi'(hesitation factor) is required for QROFN fuzzification.")

        q = self.q
        # 计算隶属度
        md = float(np.clip(mf.compute(x), 0.0, 1.0))
        # 基于犹豫因子计算非隶属度
        nmd = float(self._compute_nmd_from_hesitation(md, pi, q))

        return Fuzznum(mtype=self.mtype, q=q).create(md=float(md), nmd=float(nmd))

    def fuzzify_array(self,
                      x: Optional[np.ndarray],
                      mf: Optional[MembershipFunction] = None) -> 'Fuzzarray':

        pi = self.kwargs.get('pi')
        if pi is None:
            raise ValueError("Parameter 'pi'(hesitation factor) is required for QROFN fuzzification.")

        q = self.q
        # 批量计算隶属度并裁剪
        mds = np.clip(mf.compute(x), 0.0, 1.0)
        # 基于犹豫因子批量计算非隶属度
        nmds = self._compute_nmd_from_hesitation(mds, pi, q)

        # 直接创建FuzzarrayBackend（与文档一致的参数名）
        registry = get_registry_fuzztype()
        backend_cls = registry.get_backend(self.mtype)
        backend = backend_cls.from_arrays(mds=mds, nmds=nmds, q=q)

        return Fuzzarray(backend=backend, mtype=self.mtype, q=q)

    @staticmethod
    def _compute_nmd_from_hesitation(md: Union[float, np.ndarray],
                                     pi: Union[float, np.ndarray],
                                     q: int) -> Union[float, np.ndarray]:
        """
        基于隶属度和犹豫度计算非隶属度

        公式: nmd = (1 - md^q - pi)^(1/q)
        其中: pi = 1 - md^q - nmd^q (犹豫度/不确定度)
        """
        md = np.asarray(md, dtype=float)
        pi = np.asarray(pi, dtype=float)

        md = np.clip(md, 0.0, 1.0)
        pi = np.clip(pi, 0.0, 1.0)

        remaining_space = 1.0 - md ** q - pi ** q
        remaining_space = np.maximum(remaining_space, 0.0)
        nmd = remaining_space ** (1.0 / q)

        return nmd
