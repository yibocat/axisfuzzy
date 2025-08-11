#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/10 11:29
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

# from .fuzznums import Fuzznum


class FuzzarrayBackend(ABC):
    """
    Abstract base class for Fuzzarray data backends (SoA implementation).

    Each subclass is specific to a mtype and is responsible for managing
    the underlying NumPy arrays that store the fuzzy number components.
    This enables high-performance vectorized operations while maintaining
    the flexibility of FuzzLab's mtype-based architecture.
    """

    # 每个具体的后端都必须定义其对应的 mtype
    mtype: str = "unknown"

    def __init__(self, shape: Tuple[int, ...], **mtype_kwargs):
        """
        Initialize the backend with given shape and mtype-specific parameters.

        Args:
            shape: Shape of the fuzzy array
            **mtype_kwargs: Mtype-specific parameters (e.g., q for qrofn)
        """
        self.shape = shape
        self.size = int(np.prod(shape))
        self.mtype_kwargs = mtype_kwargs

        # 子类需要在这里初始化具体的 NumPy 数组
        self._initialize_arrays()

    @abstractmethod
    def _initialize_arrays(self):
        """Initialize the NumPy arrays that store the fuzzy number components."""
        pass

    @abstractmethod
    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Creates a lightweight Fuzznum object for the element at the given index.
        This should be a relatively fast operation for single-element access.

        Args:
            index: Index to access (can be int, tuple, or slice)

        Returns:
            A Fuzznum object representing the data at the given index
        """
        pass

    @abstractmethod
    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """
        Sets the data at the given index using a Fuzznum object.

        Args:
            index: Index to set
            fuzznum: Fuzznum object containing the data to set
        """
        pass

    @abstractmethod
    def copy(self) -> 'FuzzarrayBackend':
        """
        Returns a deep copy of the backend and all its data arrays.

        Returns:
            A new backend instance with copied data
        """
        pass

    @abstractmethod
    def slice_view(self, key) -> 'FuzzarrayBackend':
        """
        Returns a new backend that represents a slice/view of this backend.
        This should be a fast operation that shares data when possible.

        Args:
            key: Slice key (slice object, tuple of slices, etc.)

        Returns:
            A new backend representing the sliced data
        """
        pass

    @staticmethod
    def from_arrays(*components, **mtype_kwargs) -> 'FuzzarrayBackend':
        pass

    def fill_from_values(self, *values: float):
        """
        Fill all elements with the given md and nmd values.

        Args:
            values: Values to fill (should match the number of components for the mtype)
        """
        pass

    def get_component_arrays(self) -> tuple:
        """
        Get the underlying component arrays.

        Returns:
            Tuple of arrays
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, mtype='{self.mtype}')"
