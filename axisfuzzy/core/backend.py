#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/10 11:29
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
axisfuzzy.core.backend
======================

Abstract backend interface for Fuzzarray (Struct-of-Arrays).

This module defines the `FuzzarrayBackend` abstract base class which is the
primary contract between high-level `Fuzzarray` containers and concrete,
mtype-specific, NumPy-backed implementations.
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

import numpy as np

from ..config import get_config


class FuzzarrayBackend(ABC):
    """
    Abstract base class for Fuzzarray data backends (SoA implementation).

    Each concrete backend manages the NumPy arrays that store the fuzzy-number
    components for a specific `mtype`. Subclasses must implement array
    initialization, element view construction, slicing/copy semantics and
    formatting.

    Parameters
    ----------
    shape : tuple of int
        Shape of the logical Fuzzarray.
    q : int, optional
        q-rung parameter for q-rung based mtypes. If ``None``, the default
        from configuration is used.
    **kwargs
        Additional backend-specific parameters.

    Attributes
    ----------
    shape : tuple of int
        The logical shape of the array.
    size : int
        Total number of elements (product of shape).
    q : int
        Effective q-rung used by this backend instance.
    kwargs : dict
        Backend-specific extra parameters.
    mtype : str
        Class attribute that concrete subclasses should override to indicate
        the supported mtype string.

    Notes
    -----
    - Implementations are expected to prefer views (shared memory) for slicing
      whenever possible to avoid expensive copies.
    - The backend API focuses on array-level operations; high-level semantics
      (e.g., operator dispatch) are handled by other core components.
    """

    mtype: str = "unknown"

    def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
        self.shape = shape
        self.size = int(np.prod(shape))
        self.q = q if q is not None else get_config().DEFAULT_Q

        self.kwargs = kwargs

        # 子类需要在这里初始化具体的 NumPy 数组
        self._initialize_arrays()

    @abstractmethod
    def _initialize_arrays(self):
        """
        Initialize the NumPy arrays that store the fuzzy-number components.

        Implementations should create and store the component arrays (for
        example: membership degrees, non-membership degrees, etc.) as
        attributes of the backend instance.

        Notes
        -----
        - This method is called during construction; subclasses should not
          assume the existence of other backend methods being callable yet.
        """
        pass

    @abstractmethod
    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Create a lightweight Fuzznum view for the element at `index`.

        The view should be a thin wrapper (no deep copy) that provides
        per-element access and can be used by high-level operations that
        expect a Fuzznum-like object.

        Parameters
        ----------
        index : int or tuple or slice
            Index of the element to view. Accepts the same indexing semantics
            as NumPy arrays for single-element access.

        Returns
        -------
        Fuzznum
            A Fuzznum-like object representing the element at `index`.

        Raises
        ------
        IndexError
            If `index` is out of bounds.

        Notes
        -----
        - Implementations may return a proxy object that reads/writes directly
          into the backend arrays to avoid copying.
        """
        pass

    @abstractmethod
    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """
        Set data at `index` using values from a Fuzznum object.

        Parameters
        ----------
        index : int or tuple or slice
            Target location to set.
        fuzznum : Fuzznum
            Source Fuzznum providing component values.

        Raises
        ------
        IndexError
            If `index` is out of bounds.
        TypeError
            If `fuzznum` is not compatible with this backend's mtype.
        """
        pass

    @abstractmethod
    def copy(self) -> 'FuzzarrayBackend':
        """
        Create a deep copy of this backend and its underlying arrays.

        Returns
        -------
        FuzzarrayBackend
            A new backend instance with duplicated (deep-copied) arrays.

        Notes
        -----
        - This operation must not share memory with the original instance.
        """
        pass

    @abstractmethod
    def slice_view(self, key) -> 'FuzzarrayBackend':
        """
        Return a backend representing a slice/view of this backend.

        Parameters
        ----------
        key : slice, tuple of slices or other valid indexing key
            Indexing key describing the requested view.

        Returns
        -------
        FuzzarrayBackend
            A new backend instance representing the requested slice. The
            returned backend should share data with the original when possible.

        Notes
        -----
        - Preferred behavior is to avoid copying and to provide a view with
          consistent semantics for read/write operations.
        """
        pass

    @abstractmethod
    def format_elements(self, format_spec: str = "") -> np.ndarray:
        """
        Produce a NumPy array of formatted strings for each element.

        The backend must implement formatting logic appropriate for the mtype,
        combining component arrays into readable string representations.

        Parameters
        ----------
        format_spec : str, optional
            Format specification passed from higher-level formatting calls.

        Returns
        -------
        numpy.ndarray
            1-D array of dtype 'object' containing formatted strings for every
            element in the backend (size == self.size).
        """
        pass

    @staticmethod
    def from_arrays(*components, **kwargs) -> 'FuzzarrayBackend':
        """
        Construct a backend instance from raw component arrays.

        Parameters
        ----------
        *components : array_like
            Component arrays that represent the SoA layout for a specific mtype.
        **kwargs
            Backend-specific keyword arguments (e.g., shape, q, dtype).

        Returns
        -------
        FuzzarrayBackend
            A new backend instance initialised from the given arrays.

        Notes
        -----
        - This factory is a convenience hook and should be implemented by
          concrete subclasses to validate shapes and dtypes.
        """
        pass

    def fill_from_values(self, *values: float):
        """
        Fill every element in the backend with the provided component values.

        Parameters
        ----------
        *values : float
            Values to broadcast to every element for each component.

        Notes
        -----
        - Concrete backends should validate the number of provided values
          against the expected number of components for the mtype.
        """
        pass

    def get_component_arrays(self) -> tuple:
        """
        Return the underlying component arrays used by the backend.

        Returns
        -------
        tuple
            Tuple containing the NumPy arrays that make up the SoA backend.

        Notes
        -----
        - The order and meaning of returned arrays is backend-specific and
          should be documented by each concrete implementation.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, mtype='{self.mtype}')"
