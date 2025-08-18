#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Any, cast, Tuple

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
        new_backend.mds = np.array([arr.copy() if arr is not None else None for arr in self.mds.flatten()], dtype=object).reshape(self.shape)
        new_backend.nmds = np.array([arr.copy() if arr is not None else None for arr in self.nmds.flatten()], dtype=object).reshape(self.shape)
        return cast('QROHFNBackend', new_backend)

    def slice_view(self, key) -> 'QROHFNBackend':
        """Create a view of the backend with the given slice."""
        new_shape = self.mds[key].shape
        new_backend = QROHFNBackend(new_shape, self.q, **self.kwargs)
        new_backend.mds = self.mds[key]
        new_backend.nmds = self.nmds[key]
        return cast('QROHFNBackend', new_backend)

    def format_elements(self, format_spec: str = "") -> np.ndarray:
        """
        Format all elements into a NumPy array of strings.
        This operation is element-wise due to the nature of hesitant sets.
        """
        out = np.empty(self.shape, dtype=object)
        strategy_formatter = QROHFNStrategy(q=self.q)

        it = np.nditer(out, flags=['multi_index', 'refs_ok'], op_flags=['writeonly'])
        while not it.finished:
            idx = it.multi_index
            md_val = self.mds[idx]
            nmd_val = self.nmds[idx]
            out[idx] = strategy_formatter.format_from_components(md_val, nmd_val, format_spec)
            it.iternext()
        return out

    @classmethod
    def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int, **kwargs) -> 'QROHFNBackend':
        """
        Construct a backend directly from component object arrays.
        """
        if mds.shape != nmds.shape:
            raise ValueError("Shape mismatch between mds and nmds arrays.")
        if mds.dtype != object or nmds.dtype != object:
            raise TypeError("Input arrays for QROHFNBackend must have dtype=object.")

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
