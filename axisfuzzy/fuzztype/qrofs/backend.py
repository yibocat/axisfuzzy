#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Any, Tuple, cast, Optional

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

    mtype = "qrofn"

    def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
        """
        Initializes the QROFNBackend.

        Args:
            shape: The shape of the array.
            q: The q-rung parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(shape, q, **kwargs)

    def _initialize_arrays(self):
        """Initialize mds and nmds arrays for QROFN data."""

        # 初始化两个核心数组：membership degrees 和 non-membership degrees
        self.mds = np.zeros(self.shape, dtype=np.float64)
        self.nmds = np.zeros(self.shape, dtype=np.float64)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Create a Fuzznum object from the data at the given index.
        """
        # 从 NumPy 数组中提取标量值
        md_value = float(self.mds[index])
        nmd_value = float(self.nmds[index])

        # 创建 Fuzznum 对象，传入正确的参数
        return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """
        Set data at the given index from a Fuzznum object.
        """
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

    def format_elements(self, format_spec: str = "") -> np.ndarray:
        """
        高性能批量格式化：
        默认 / 'c' 使用向量化字符串操作 (np.char.*)，避免 np.vectorize。
        特殊格式 'p','j','r'（少用）回退逐元素。
        """
        precision = get_config().DEFAULT_PRECISION
        if format_spec in ('p', 'j', 'r'):
            out = np.empty(self.shape, dtype=object)
            # 创建一个策略实例用于格式化

            # strategy_cls = get_fuzztype_strategy(self.mtype)
            # strategy_formatter = strategy_cls(q=self.q)

            strategy_formatter = QROFNStrategy(q=self.q)
            it = np.nditer(self.mds, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                md = float(self.mds[idx])
                nmd = float(self.nmds[idx])
                out[idx] = strategy_formatter.format_from_components(md, nmd, format_spec)  # type: ignore
                it.iternext()
            return out

        # 批量数值格式化
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
