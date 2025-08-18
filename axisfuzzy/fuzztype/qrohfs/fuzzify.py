#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 21:25
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzification strategy for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

This module implements a flexible fuzzification strategy that can generate
hesitant fuzzy sets by applying membership functions with multiple parameter
configurations, supporting both scalar and array inputs efficiently.
"""
import inspect
from typing import Any, Optional, Union, Dict, List

import numpy as np

from ...config import get_config
from ...core import Fuzznum, Fuzzarray, get_registry_fuzztype
from ...fuzzify import FuzzificationStrategy, register_fuzzify
from ...membership import MembershipFunction

from .backend import QROHFNBackend


@register_fuzzify('qrohfn', 'default', is_default=True)
class QROHFNFuzzificationStrategy(FuzzificationStrategy):
    """
    Default fuzzification strategy for Q-Rung Orthopair Hesitant Fuzzy Numbers.

    Key Features:
    - Multi-parameter hesitant membership generation using hybrid parameter mode
    - Automatic non-membership degree computation based on hesitation factor
    - High-performance batch processing for arrays
    - Orthopair constraint satisfaction: max(md)^q + max(nmd)^q ≤ 1
    """

    # 向 Fuzzifier 声明本策略将直接处理隶属函数参数
    HANDLES_MF_PARAMS_DIRECTLY = True

    def __init__(self,
                 q: Optional[int] = None,
                 pi: Optional[float] = None,
                 nmd_generation_mode: str = 'pi_based',
                 **kwargs: Any):
        # ...existing code before storing mf params...
        q = q if q is not None else get_config().DEFAULT_Q
        super().__init__(q=q, pi=pi, nmd_generation_mode=nmd_generation_mode, **kwargs)

        self.mtype = "qrohfn"
        self.method = 'default'

        self.pi = pi if pi is not None else 0.1
        self.nmd_generation_mode = nmd_generation_mode

        # 提取并标准化 mf 参数规范：
        # 1) 支持 mf_params=... 或 params=...
        # 2) 否则把剩余 kwargs 当成 mf 参数（兼容直接扁平传参）
        mf_spec = kwargs.pop('mf_params', None)
        if mf_spec is None:
            mf_spec = kwargs.pop('params', None)
        self.mf_params = mf_spec if mf_spec is not None else kwargs

        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """Validate initialization parameters."""
        if not 0.0 <= self.pi <= 1.0:
            raise ValueError(f"pi must be in [0, 1], got {self.pi}")

        if self.nmd_generation_mode not in ['pi_based', 'proportional', 'uniform']:
            raise ValueError(f"Invalid nmd_generation_mode: {self.nmd_generation_mode}")

    def _parse_mf_params(self, mf_params: Union[Dict, List[Dict]]) -> List[Dict[str, float]]:
        """
        将 mf 参数规范解析为“参数字典列表”。
        允许：
        - 混合模式：{'a': [..], 'b': 0.5} -> [{'a': .., 'b': 0.5}, ...]
        - 列表模式：[{'a': .., 'b': ..}, ...]
        - 兼容外层包裹 {'params': {...}} 或 [{'params': {...}}, ...]
        """
        if not mf_params:
            return [{}]

        # 兼容外层 'params' 包裹
        if isinstance(mf_params, dict) and 'params' in mf_params and set(mf_params.keys()) == {'params'}:
            mf_params = mf_params['params']
        if isinstance(mf_params, list) and mf_params and isinstance(mf_params[0], dict) and 'params' in mf_params[0]:
            mf_params = [d.get('params', {}) for d in mf_params]

        # 列表模式
        if isinstance(mf_params, list):
            if not all(isinstance(d, dict) for d in mf_params):
                raise ValueError("When mf_params is a list, all elements must be dictionaries")
            return mf_params

        # 混合模式
        if not isinstance(mf_params, dict):
            raise ValueError("mf_params must be either a dict (hybrid mode) or list of dicts")

        list_params: Dict[str, np.ndarray] = {}
        scalar_params: Dict[str, Any] = {}
        hesitant_count: Optional[int] = None

        for key, value in mf_params.items():
            if isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value)
                if hesitant_count is None:
                    hesitant_count = len(arr)
                elif len(arr) != hesitant_count:
                    raise ValueError(f"All parameter lists must have same length: expected {hesitant_count}, "
                                     f"got {len(arr)} for '{key}'")
                list_params[key] = arr
            else:
                scalar_params[key] = value

        if not list_params:
            return [scalar_params]

        out: List[Dict[str, Any]] = []
        for i in range(int(hesitant_count or 0)):
            d = dict(scalar_params)
            for k, arr in list_params.items():
                v = arr[i]
                # 标量化 numpy 类型
                d[k] = float(v) if np.isscalar(v) else v
            out.append(d)
        return out

    def _filter_params_for_mf(self, mf_class: type, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤掉隶属函数构造器不接受的参数键（如 'params', 'mf_params', 以及策略自有键）。
        如构造器含 **kwargs，则直接返回原参数。
        """
        sig = inspect.signature(mf_class.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return params
        valid = {k for k in sig.parameters.keys() if k != 'self'}
        # 常见额外参数名一并忽略
        blacklist = {'params', 'mf_params', 'pi', 'q', 'nmd_generation_mode'}
        return {k: v for k, v in params.items() if k in valid and k not in blacklist}

    def _compute_mds(self, x: Union[float, np.ndarray], mf_input: Union[type, MembershipFunction],
                     param_list: List[Dict[str, float]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compute membership degrees using multiple parameter sets.

        Args:
            x: Input value(s)
            mf_input: Membership function class or an already configured instance
            param_list: List of parameter dictionaries

        Returns:
            For scalar x: 1D array of membership degrees
            For array x: List of 1D arrays, one per input element
        """
        is_scalar = np.isscalar(x)
        mf_class = mf_input if inspect.isclass(mf_input) else mf_input.__class__

        if is_scalar:
            mds: List[float] = []
            for params in param_list:
                clean = self._filter_params_for_mf(mf_class, params)
                mf_instance = mf_class(**clean)
                md = float(np.clip(mf_instance.compute(x), 0.0, 1.0))
                mds.append(md)
            return np.array(mds, dtype=np.float64)
        else:
            x_array = np.asarray(x)
            rows: List[np.ndarray] = []
            for params in param_list:
                clean = self._filter_params_for_mf(mf_class, params)
                mf_instance = mf_class(**clean)
                md_array = np.clip(mf_instance.compute(x_array), 0.0, 1.0)
                rows.append(md_array)
            mat = np.asarray(rows)  # shape: (n_sets, n_elems)
            return [mat[:, i] for i in range(mat.shape[1])]

    def _compute_nmds(self, mds: Union[np.ndarray, List[np.ndarray]],
                      q: int) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compute non-membership degrees based on membership degrees and strategy.

        Args:
            mds: Membership degrees (scalar case: 1D array, array case: list of 1D arrays)
            q: Q-rung parameter

        Returns:
            Non-membership degrees in same format as mds
        """
        if isinstance(mds, np.ndarray):
            # Scalar case: single element with multiple mds
            return self._compute_nmds_single(mds, q)
        else:
            # Array case: multiple elements, each with multiple mds
            return [self._compute_nmds_single(md_array, q) for md_array in mds]

    def _compute_nmds_single(self, mds: np.ndarray, q: int) -> np.ndarray:
        """
        Compute non-membership degrees for a single hesitant set.

        Args:
            mds: 1D array of membership degrees for one element
            q: Q-rung parameter

        Returns:
            1D array of non-membership degrees
        """
        if len(mds) == 0:
            return np.array([], dtype=np.float64)

        max_md = float(np.max(mds))

        if self.nmd_generation_mode == 'pi_based':
            # Based on hesitation factor: max(nmd) derived from pi
            if max_md ** q + self.pi ** q > 1.0:
                # Constraint violation: reduce pi proportionally
                available_space = 1.0 - max_md ** q
                if available_space <= 0:
                    return np.zeros_like(mds, dtype=np.float64)
                max_nmd = (available_space * (self.pi ** q)) ** (1.0 / q)
            else:
                max_nmd = self.pi

            # Generate nmd values: all equal to max_nmd
            return np.full_like(mds, max_nmd, dtype=np.float64)

        elif self.nmd_generation_mode == 'proportional':
            # Proportional to (1-md): nmd_i = (1-md_i) * scale
            max_possible_nmd = (1.0 - max_md ** q) ** (1.0 / q)
            scale = min(self.pi, max_possible_nmd) / (1.0 - max_md) if max_md < 1.0 else 0.0
            nmds = (1.0 - mds) * scale
            return np.clip(nmds, 0.0, max_possible_nmd)

        elif self.nmd_generation_mode == 'uniform':
            # Uniform distribution within constraint
            max_possible_nmd = (1.0 - max_md ** q) ** (1.0 / q)
            target_nmd = min(self.pi, max_possible_nmd)
            # Add small random variations around target (deterministic for now)
            base_nmds = np.full_like(mds, target_nmd, dtype=np.float64)
            return base_nmds

        else:
            raise ValueError(f"Unknown nmd_generation_mode: {self.nmd_generation_mode}")

    def fuzzify_scalar(self,
                       x: Optional[float],
                       mf: Optional[Union[type, MembershipFunction]] = None) -> Fuzznum:
        """
        Fuzzify a single scalar value into a QROHFN.

        Args:
            x: Input crisp value
            mf: Membership function class or instance

        Returns:
            QROHFN Fuzznum with hesitant membership and non-membership degrees
        """
        if x is None or mf is None:
            raise ValueError("Both x and mf must be provided for QROHFN fuzzification")

        # 解析 + 过滤
        param_list = self._parse_mf_params(self.mf_params)
        mds = self._compute_mds(x, mf, param_list)
        nmds = self._compute_nmds(mds, self.q)
        return Fuzznum(mtype='qrohfn', q=self.q).create(md=mds, nmd=nmds)

    def fuzzify_array(self,
                      x: Optional[np.ndarray],
                      mf: Optional[Union[type, MembershipFunction]] = None) -> Fuzzarray:
        """
        Fuzzify an array of values into a QROHFN Fuzzarray (high-performance batch version).

        Args:
            x: Input crisp value array
            mf: Membership function class or instance

        Returns:
            QROHFN Fuzzarray with hesitant sets for each element
        """
        if x is None or mf is None:
            raise ValueError("Both x and mf must be provided for QROHFN fuzzification")

        x_array = np.asarray(x)
        flat = x_array.ravel()

        param_list = self._parse_mf_params(self.mf_params)
        element_mds = self._compute_mds(flat, mf, param_list)
        element_nmds = self._compute_nmds(element_mds, self.q)

        mds_obj = np.empty(flat.size, dtype=object)
        nmds_obj = np.empty(flat.size, dtype=object)
        for i, (mds, nmds) in enumerate(zip(element_mds, element_nmds)):
            mds_obj[i] = np.asarray(mds, dtype=np.float64)
            nmds_obj[i] = np.asarray(nmds, dtype=np.float64)

        backend = QROHFNBackend.from_arrays(mds=mds_obj.reshape(x_array.shape),
                                            nmds=nmds_obj.reshape(x_array.shape),
                                            q=self.q)
        return Fuzzarray(backend=backend)
