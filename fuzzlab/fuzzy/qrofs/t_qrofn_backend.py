#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/10 11:34
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Any, Tuple

import numpy as np

from ...core import Fuzznum
from ...core.t_backend import FuzzarrayBackend


class QROFNBackend(FuzzarrayBackend):
    """
    SoA (Struct of Arrays) backend for q-rung orthopair fuzzy numbers.

    This backend stores membership and non-membership degrees as separate
    NumPy arrays, enabling high-performance vectorized operations.
    """

    mtype = "qrofn"

    def _initialize_arrays(self):
        """Initialize mds and nmds arrays for QROFN data."""
        # 获取 q 参数，默认值为 1
        self.q = self.mtype_kwargs.get('q', 1)

        # 初始化两个核心数组：membership degrees 和 non-membership degrees
        self.mds = np.zeros(self.shape, dtype=np.float64)
        self.nmds = np.zeros(self.shape, dtype=np.float64)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Create a Fuzznum object from the data at the given index.
        """
        from ...core.fuzznums import Fuzznum

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
            raise ValueError(f"Mtype mismatch: expected '{self.mtype}', got '{fuzznum.mtype}'")

        if fuzznum.q != self.q:
            raise ValueError(f"Q parameter mismatch: expected q={self.q}, got q={fuzznum.q}")

        # 提取 Fuzznum 的数据并设置到对应的数组位置
        self.mds[index] = fuzznum.md
        self.nmds[index] = fuzznum.nmd

    def copy(self) -> 'QROFNBackend':
        """
        Create a deep copy of this backend.
        """
        new_backend = QROFNBackend(shape=self.shape, **self.mtype_kwargs)
        new_backend.mds = self.mds.copy()
        new_backend.nmds = self.nmds.copy()
        return new_backend

    def slice_view(self, key) -> 'QROFNBackend':
        """
        Create a new backend representing a slice of this one.
        """
        # 应用切片操作到数组
        sliced_mds = self.mds[key]
        sliced_nmds = self.nmds[key]

        # 创建新的后端
        new_backend = QROFNBackend(shape=sliced_mds.shape, **self.mtype_kwargs)
        new_backend.mds = sliced_mds
        new_backend.nmds = sliced_nmds

        return new_backend

    @classmethod
    def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, **mtype_kwargs) -> 'QROFNBackend':
        """
        Factory method to create a backend directly from NumPy arrays.
        This is the high-performance path for batch operations.

        Args:
            mds: Membership degrees array
            nmds: Non-membership degrees array
            **mtype_kwargs: Mtype-specific parameters

        Returns:
            A new QROFNBackend instance
        """
        if mds.shape != nmds.shape:
            raise ValueError("mds and nmds arrays must have the same shape")

        backend = cls(shape=mds.shape, **mtype_kwargs)
        backend.mds = mds.copy() if isinstance(mds, np.ndarray) else np.array(mds)
        backend.nmds = nmds.copy() if isinstance(nmds, np.ndarray) else np.array(nmds)

        return backend

    def fill_from_values(self, md_value: float, nmd_value: float):
        """
        Fill the entire backend with constant values.
        Useful for creating uniform arrays.
        """
        self.mds.fill(md_value)
        self.nmds.fill(nmd_value)

    def get_component_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get direct access to the underlying arrays.
        WARNING: Modifying these arrays directly can break consistency.
        Use only for high-performance read operations.
        """
        return self.mds, self.nmds
