#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Any, Tuple, cast, Optional, Callable

import numpy as np

from ...core import Fuzznum, FuzzarrayBackend, register_backend
from ...config import get_config

from .qrofn import QROFNStrategy


@register_backend
class QROFNBackend(FuzzarrayBackend):
    """
    SoA (Struct of Arrays) backend for q-rung orthopair fuzzy numbers.

    This backend stores membership and non-membership degrees as separate
    NumPy arrays, enabling high-performance vectorized operations.
    """

    mtype = 'qrofn'

    def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
        """
        Initializes the QROFNBackend.

        Args:
            shape: The shape of the array.
            q: The q-rung parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(shape, q, **kwargs)

    @property
    def cmpnum(self) -> int:
        return 2

    @property
    def cmpnames(self) -> Tuple[str, ...]:
        return 'md', 'nmd'

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    def _initialize_arrays(self):
        """Initialize mds and nmds arrays for QROFN data."""
        # 初始化两个核心数组：membership degrees 和 non-membership degrees
        self.mds = np.zeros(self.shape, dtype=np.float64)
        self.nmds = np.zeros(self.shape, dtype=np.float64)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """Create a Fuzznum object from the data at the given index."""
        # 从 NumPy 数组中提取标量值
        md_value = float(self.mds[index])
        nmd_value = float(self.nmds[index])

        # 创建 Fuzznum 对象，传入正确的参数
        return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """Set data at the given index from a Fuzznum object."""
        if fuzznum.mtype != self.mtype:
            raise ValueError(f"Mtype mismatch: expected {self.mtype}, got {fuzznum.mtype}")

        if fuzznum.q != self.q:
            raise ValueError(f"Q parameter mismatch: expected {self.q}, got {fuzznum.q}")

        # 提取 Fuzznum 的数据并设置到对应的数组位置
        self.mds[index] = fuzznum.md
        self.nmds[index] = fuzznum.nmd

    def copy(self) -> 'QROFNBackend':
        """Create a deep copy of the backend."""
        new_backend = QROFNBackend(self.shape, self.q, **self.kwargs)
        new_backend.mds = self.mds.copy()
        new_backend.nmds = self.nmds.copy()
        return cast('QROFNBackend', new_backend)

    def slice_view(self, key) -> 'QROFNBackend':
        """Create a view of the backend with the given slice."""
        new_shape = self.mds[key].shape
        new_backend = QROFNBackend(new_shape, self.q, **self.kwargs)
        new_backend.mds = self.mds[key]
        new_backend.nmds = self.nmds[key]
        return cast('QROFNBackend', new_backend)

    # ================== 智能显示系统实现 ==================

    def _get_element_formatter(self, format_spec: str) -> Callable:
        """获取元素格式化函数"""
        precision = get_config().DEFAULT_PRECISION

        if format_spec in ('p', 'j', 'r'):
            # 特殊格式：使用 Strategy 进行格式化
            strategy_formatter = QROFNStrategy(q=self.q)
            return lambda md, nmd: strategy_formatter.format_from_components(md, nmd, format_spec)
        else:
            # 默认格式：使用高效的字符串操作
            return self._create_default_formatter(precision)

    def _create_default_formatter(self, precision: int) -> Callable:
        """创建默认格式化函数"""
        def format_pair(md: float, nmd: float) -> str:
            # 高效的数值格式化
            fmt = f"%.{precision}f"
            md_str = (fmt % md).rstrip('0').rstrip('.')
            nmd_str = (fmt % nmd).rstrip('0').rstrip('.')

            # 处理全零情况
            md_str = md_str if md_str else '0'
            nmd_str = nmd_str if nmd_str else '0'

            return f"<{md_str},{nmd_str}>"

        return format_pair

    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """格式化单个元素"""
        md_value = float(self.mds[index])
        nmd_value = float(self.nmds[index])
        return formatter(md_value, nmd_value)

    def _format_all_elements(self, format_spec: str) -> np.ndarray:
        """
        完整格式化所有元素（针对 QROFN 的优化实现）

        对于小数据集，使用向量化的 NumPy 字符串操作来提升性能
        """
        precision = get_config().DEFAULT_PRECISION

        if format_spec in ('p', 'j', 'r'):
            # 特殊格式：回退到基类的逐元素处理
            return super()._format_all_elements(format_spec)

        # 默认格式：使用向量化字符串操作
        fmt = f"%.{precision}f"
        md_strs = np.char.mod(fmt, np.round(self.mds, precision))
        nmd_strs = np.char.mod(fmt, np.round(self.nmds, precision))

        def _trim(arr: np.ndarray) -> np.ndarray:
            # 去掉尾部 0
            trimmed = np.char.rstrip(np.char.rstrip(arr, '0'), '.')
            # 若全部被去掉（例如 "0.0000"）则恢复为 "0"
            return np.where(trimmed == '', '0', trimmed)

        md_trimmed = _trim(md_strs)
        nmd_trimmed = _trim(nmd_strs)

        combined = np.char.add(
            np.char.add(
                np.char.add("<", md_trimmed),
                np.char.add(",", nmd_trimmed)
            ),
            ">"
        )
        return np.array(combined, dtype=object)

    @staticmethod
    def _validate_fuzzy_constraints_static(mds: np.ndarray, nmds: np.ndarray, q: int) -> None:
        """
        Static method for validating QROFN fuzzy constraints without creating backend instance.
        
        Parameters
        ----------
        mds : np.ndarray
            Membership degrees array.
        nmds : np.ndarray
            Non-membership degrees array.
        q : int
            The q-rung parameter for constraint validation.
            
        Raises
        ------
        ValueError
            If any elements violate the QROFN constraint md^q + nmd^q <= 1.
        """
        # Vectorized constraint check: md^q + nmd^q <= 1 + epsilon
        from ...config import get_config
        epsilon = get_config().DEFAULT_EPSILON
        
        # Use numpy broadcasting for efficient computation
        sum_of_powers = np.power(mds, q) + np.power(nmds, q)
        violations = sum_of_powers > (1.0 + epsilon)
        
        if np.any(violations):
            # Find first violation for detailed error message
            violation_indices = np.where(violations)
            first_idx = tuple(idx[0] for idx in violation_indices)
            
            md_val = mds[first_idx]
            nmd_val = nmds[first_idx]
            sum_val = sum_of_powers[first_idx]
            
            raise ValueError(
                f"QROFN constraint violation at index {first_idx}: "
                f"md^q ({md_val}^{q}) + nmd^q ({nmd_val}^{q}) = {sum_val:.4f} > 1.0. "
                f"(q: {q}, md: {md_val}, nmd: {nmd_val})"
            )

    @classmethod
    def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int, **kwargs) -> 'QROFNBackend':
        """
        Create a QROFNBackend from existing arrays.

        Args:
            mds: Membership degrees array
            nmds: Non-membership degrees array
            q: q-rung parameter
            **kwargs: Type-specific parameters

        Returns:
            New QROFNBackend instance
        """
        if mds.shape != nmds.shape:
            raise ValueError(f"Shape mismatch: mds {mds.shape} vs nmds {nmds.shape}")

        # Direct constraint validation without creating temporary backend
        cls._validate_fuzzy_constraints_static(mds, nmds, q=q)
        
        backend = cls(mds.shape, q, **kwargs)
        backend.mds = mds.copy()
        backend.nmds = nmds.copy()
        return backend

    def fill_from_values(self, md_value: float, nmd_value: float):
        """
        Fill all elements with the given md and nmd values.

        Args:
            md_value: Membership degree value
            nmd_value: Non-membership degree value
        """
        self.mds.fill(md_value)
        self.nmds.fill(nmd_value)

    def get_component_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the underlying component arrays.

        Returns:
            Tuple of (mds, nmds) arrays
        """
        return self.mds, self.nmds
