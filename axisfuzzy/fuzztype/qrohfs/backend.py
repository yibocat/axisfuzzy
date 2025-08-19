#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Any, cast, Tuple, Callable

import numpy as np

from ...core import Fuzznum, FuzzarrayBackend, register_backend
from ...config import get_config
from .qrohfn import QROHFNStrategy


@register_backend
class QROHFNBackend(FuzzarrayBackend):
    """
    SoA (Struct of Arrays) backend for q-rung orthopair hesitant fuzzy numbers.

    This backend stores membership and non-membership degrees as separate
    NumPy arrays, enabling high-performance vectorized operations.
    """

    mtype = "qrohfn"

    def __init__(self, shape: tuple[int, ...], q: int | None = None, **kwargs):
        """
        Initializes the QROHFNBackend.

        Args:
            shape: The shape of the array.
            q: The q-rung parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(shape, q, **kwargs)

    def _initialize_arrays(self):
        """Initialize mds and nmds arrays for QROHFN data."""

        # Initialize two core arrays: membership degrees and non-membership degrees
        self.mds = np.empty(self.shape, dtype=object)
        self.nmds = np.empty(self.shape, dtype=object)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Create a Fuzznum object from the data at the given index.
        """
        md_value = self.mds[index]
        nmd_value = self.nmds[index]

        return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """
        Set data at the given index from a Fuzznum object.
        """
        if fuzznum.mtype != self.mtype:
            raise ValueError(f"Mtype mismatch: expected {self.mtype}, got {fuzznum.mtype}")

        if fuzznum.q != self.q:
            raise ValueError(f"Q parameter mismatch: expected {self.q}, got {fuzznum.q}")

        # The strategy ensures md and nmd are ndarrays
        self.mds[index] = fuzznum.md
        self.nmds[index] = fuzznum.nmd

    def copy(self) -> 'QROHFNBackend':
        """Create a deep copy of the backend."""
        new_backend = QROHFNBackend(self.shape, self.q, **self.kwargs)
        # For object arrays, a simple copy is shallow. We need to copy each element.
        new_backend.mds = np.array([arr.copy() if arr is not None else None for arr in self.mds.flatten()],
                                   dtype=object).reshape(self.shape)
        new_backend.nmds = np.array([arr.copy() if arr is not None else None for arr in self.nmds.flatten()],
                                    dtype=object).reshape(self.shape)
        return cast('QROHFNBackend', new_backend)

    def slice_view(self, key) -> 'QROHFNBackend':
        """Create a view of the backend with the given slice."""
        new_shape = self.mds[key].shape
        new_backend = QROHFNBackend(new_shape, self.q, **self.kwargs)
        new_backend.mds = self.mds[key]
        new_backend.nmds = self.nmds[key]
        return cast('QROHFNBackend', new_backend)

    @classmethod
    def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int, **kwargs) -> 'QROHFNBackend':
        """
        Construct a backend directly from component object arrays.
        """
        if mds.shape != nmds.shape:
            raise ValueError("Shape mismatch between mds and nmds arrays.")
        if mds.dtype != object or nmds.dtype != object:
            raise TypeError(f"Input arrays for QROHFNBackend must have dtype=object. "
                            f"Got {mds.dtype} and {nmds.dtype}.")

        backend = cls(shape=mds.shape, q=q, **kwargs)
        backend.mds = mds
        backend.nmds = nmds
        return backend

    def fill_from_values(self, md_value: Any, nmd_value: Any):
        """
        Fill all elements with the given md and nmd hesitant sets.

        Args:
            md_value: Membership hesitant set (list or ndarray).
            nmd_value: Non-membership hesitant set (list or ndarray).
        """
        # Ensure values are ndarrays for consistency
        md_arr = np.asarray(md_value, dtype=np.float64)
        nmd_arr = np.asarray(nmd_value, dtype=np.float64)

        # Fill the object arrays. This requires iterating.
        for i in np.ndindex(self.shape):
            self.mds[i] = md_arr.copy()
            self.nmds[i] = nmd_arr.copy()

    def get_component_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the underlying component arrays.

        Returns:
            A tuple of (mds, nmds) object arrays.
        """
        return self.mds, self.nmds

    # ================== 智能显示系统实现 ==================

    def _get_element_formatter(self, format_spec: str) -> Callable:
        """获取元素格式化函数"""
        precision = get_config().DEFAULT_PRECISION

        if format_spec == "" or format_spec == "u":
            return self._create_default_formatter(precision)
        elif format_spec == "r":
            return self._create_raw_formatter(precision)
        elif format_spec == "p":
            return self._create_python_formatter(precision)
        elif format_spec == "j":
            return self._create_json_formatter(precision)
        else:
            return self._create_default_formatter(precision)

    def _create_default_formatter(self, precision: int) -> Callable:
        """创建默认格式化函数"""
        def format_hesitant_set(hesitant_set):
            if hesitant_set is None or len(hesitant_set) == 0:
                return "[]"

            arr = np.unique(np.round(np.asarray(hesitant_set, dtype=np.float64), precision))

            if len(arr) == 1:
                val_str = f"{arr[0]:.{precision}f}".rstrip('0').rstrip('.')
                return f"[{val_str if val_str else '0'}]"
            else:
                formatted_vals = [
                    (val_str := f"{val:.{precision}f}".rstrip('0').rstrip('.')) or '0'
                    for val in arr
                ]
                return f"[{', '.join(formatted_vals)}]"

        def format_pair(md_set, nmd_set):
            md_str = format_hesitant_set(md_set)
            nmd_str = format_hesitant_set(nmd_set)
            return f"<{md_str},{nmd_str}>"

        return format_pair

    def _create_raw_formatter(self, precision: int) -> Callable:
        """创建原始格式化函数（保留重复值）"""
        def format_hesitant_set_raw(hesitant_set):
            if hesitant_set is None or len(hesitant_set) == 0:
                return "[]"

            arr = np.round(np.asarray(hesitant_set, dtype=np.float64), precision)

            if len(arr) == 1:
                val_str = f"{arr[0]:.{precision}f}".rstrip('0').rstrip('.')
                return f"[{val_str if val_str else '0'}]"
            else:
                formatted_vals = [
                    (val_str := f"{val:.{precision}f}".rstrip('0').rstrip('.')) or '0'
                    for val in arr
                ]
                return f"[{', '.join(formatted_vals)}]"

        def format_pair_raw(md_set, nmd_set):
            md_str = format_hesitant_set_raw(md_set)
            nmd_str = format_hesitant_set_raw(nmd_set)
            return f"<{md_str},{nmd_str}>"

        return format_pair_raw

    def _create_python_formatter(self, precision: int) -> Callable:
        """创建Python格式化函数"""
        def format_python_pair(md_set, nmd_set):
            def process_set(hesitant_set):
                if hesitant_set is None or len(hesitant_set) == 0:
                    return []
                arr = np.unique(np.round(np.asarray(hesitant_set, dtype=np.float64), precision))
                return arr.tolist()

            md_list = process_set(md_set)
            nmd_list = process_set(nmd_set)
            return f"({md_list}, {nmd_list})"

        return format_python_pair

    def _create_json_formatter(self, precision: int) -> Callable:
        """创建JSON格式化函数"""
        import json

        def format_json_pair(md_set, nmd_set):
            def process_set(hesitant_set):
                if hesitant_set is None or len(hesitant_set) == 0:
                    return []
                arr = np.unique(np.round(np.asarray(hesitant_set, dtype=np.float64), precision))
                return arr.tolist()

            md_list = process_set(md_set)
            nmd_list = process_set(nmd_set)
            return json.dumps({
                'mtype': self.mtype,
                'md': md_list,
                'nmd': nmd_list,
                'q': self.q
            })

        return format_json_pair

    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """格式化单个元素"""
        md_val = self.mds[index]
        nmd_val = self.nmds[index]
        return formatter(md_val, nmd_val)
