#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/14 19:56
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Any, Optional, Union

import numpy as np

from ...core import Fuzznum, Fuzzarray, get_fuzznum_registry
from ...fuzzify import FuzzificationStrategy, register_fuzzification_strategy
from ...membership import MembershipFunction


@register_fuzzification_strategy('qrofn', 'default')
class QROFNFuzzificationStrategy(FuzzificationStrategy):

    def __init__(self, q: int = 1, pi: Optional[float] = None):
        super().__init__(q, pi=pi)
        self.mtype = "qrofn"
        self.method = 'default'
        self.q = q

    def fuzzify_scalar(self,
                       x: Optional[float],
                       mf: Optional[MembershipFunction] = None) -> 'Fuzznum':

        if not self.kwargs.get('pi'):
            raise ValueError("Parameter 'pi'(hesitation factor) is required for QROFN fuzzification.")

        q = self.q
        pi = self.kwargs.get('pi')

        # 计算隶属度
        md = mf.compute(x)

        # 基于犹豫因子计算非隶属度
        nmd = self._compute_nmd_from_hesitation(md, pi, q)

        return Fuzznum(mtype=self.mtype, q=q).create(md=float(md), nmd=float(nmd))

    def fuzzify_array(self,
                      x: Optional[np.ndarray],
                      mf: Optional[MembershipFunction] = None) -> 'Fuzzarray':

        if not self.kwargs.get('pi'):
            raise ValueError("Parameter 'pi'(hesitation factor) is required for QROFN fuzzification.")

        q = self.q
        pi = self.kwargs.get('pi')

        # 批量计算隶属度
        md = mf.compute(x)

        # 基于犹豫因子批量计算非隶属度
        nmd = self._compute_nmd_from_hesitation(md, pi, q)

        # 直接创建FuzzarrayBackend
        registry = get_fuzznum_registry()
        backend_cls = registry.get_backend(self.mtype)
        backend = backend_cls.from_arrays(mds=md, nmds=nmd, q=q)

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
        md = np.asarray(md)
        pi = np.asarray(pi)

        # 确保值在合理范围内
        md = np.clip(md, 0, 1)
        pi = np.clip(pi, 0, 1)

        # 计算可用于非隶属度的剩余空间
        remaining_space = 1.0 - md ** q - pi ** q

        # 确保剩余空间非负
        remaining_space = np.maximum(remaining_space, 0.0)

        # 计算非隶属度
        nmd = remaining_space ** (1/q)

        return nmd
