#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Abstract backend interface for Fuzzarray (Struct-of-Arrays).

This module defines the :class:`~.base.FuzzarrayBackend` abstract base class which is the
primary contract between high-level :class:`~/base.Fuzzarray` containers and concrete,
mtype-specific, NumPy-backed implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Callable, TYPE_CHECKING

import numpy as np

from ..config import get_config

if TYPE_CHECKING:
    from .fuzznums import Fuzznum


class FuzzarrayBackend(ABC):
    """
    Abstract base class for SoA (Struct-of-Arrays) fuzzy-number backends.

    The backend owns the NumPy arrays that store components for each element
    (e.g. membership / non-membership arrays depending on `mtype`) and exposes
    array-level operations used by higher-level containers.

    Parameters
    ----------
    shape : tuple of int
        Logical shape of the array.
    q : int, optional
        Effective q-rung for q-rung fuzzy types. If None, uses
        :func:`axisfuzzy.config.get_config().DEFAULT_Q`.
    **kwargs
        Backend-specific options.

    Attributes
    ----------
    shape : tuple of int
        Logical shape of the array.
    size : int
        Total number of elements (product of ``shape``).
    q : int
        Effective q-rung used by this backend instance.
    kwargs : dict
        Backend-specific extra parameters.
    mtype : str
        Backend-reported mtype; default read from configuration.

    Notes
    -----
    - Concrete subclasses must implement array initialization, element views,
      slicing/copy semantics and formatting helpers.
    - Implementations should prefer views (shared memory) for slicing when
      possible to avoid unnecessary copies.

    Examples
    --------
    Create a concrete backend subclass and instantiate it:

    .. code-block:: python

        class MyBackend(FuzzarrayBackend):
            def _initialize_arrays(self):
                self._a = np.zeros(self.shape, dtype=float)
                self._b = np.ones(self.shape, dtype=float)
            # implement other abstract methods...

        be = MyBackend((2, 3), q=2)
        print(be.shape, be.size)
    """

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
        Initialize underlying NumPy arrays used by the backend.

        Notes
        -----
        - Must assign all arrays used by other methods (e.g., membership /
          non-membership arrays) as instance attributes.
        - Called from the base class constructor.

        Examples
        --------
        .. code-block:: python

            def _initialize_arrays(self):
                # For a hypothetical 2-component mtype:
                self._comp1 = np.zeros(self.shape, dtype=float)
                self._comp2 = np.zeros(self.shape, dtype=float)
        """
        pass

    @abstractmethod
    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Return a lightweight Fuzznum view for the element at ``index``.

        The view should be a thin proxy that reads/writes directly into the
        backend arrays without performing a deep copy.

        Parameters
        ----------
        index : int or tuple or slice
            Indexing key following NumPy semantics for single-element access.

        Returns
        -------
        Fuzznum
            A Fuzznum-like proxy object.

        Notes
        -----
        - Implementations may return a proxy class that reflects changes back
          into the underlying arrays.

        Examples
        --------
        .. code-block:: python

            view = backend.get_fuzznum_view((0, 1))
            view.membership = 0.7  # updates backend arrays in-place
        """
        pass

    @abstractmethod
    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """
        Assign values at ``index`` using a Fuzznum-like object.

        Parameters
        ----------
        index : int or tuple or slice
            Target location to set.
        fuzznum : Fuzznum
            Source providing component values.

        Raises
        ------
        IndexError
            If ``index`` is out of bounds.
        TypeError
            If ``fuzznum`` is incompatible with this backend's ``mtype``.

        Examples
        --------
        .. code-block:: python

            backend.set_fuzznum_data(0, some_fuzznum)
        """
        pass

    @abstractmethod
    def copy(self) -> 'FuzzarrayBackend':
        """
        Produce a deep copy of this backend and its arrays.

        Returns
        -------
        FuzzarrayBackend
            New backend instance with duplicated arrays (no shared memory).

        Notes
        -----
        - The returned instance must not share memory with the original.

        Examples
        --------
        .. code-block:: python

            new_backend = backend.copy()
            assert new_backend is not backend
        """
        pass

    @abstractmethod
    def slice_view(self, key) -> 'FuzzarrayBackend':
        """
        Return a backend representing a view/slice of this backend.

        Parameters
        ----------
        key : slice, tuple of slices or other valid indexing key
            Indexing key describing the requested view.

        Returns
        -------
        FuzzarrayBackend
            Backend representing the requested slice. Should share memory
            with the original wherever feasible.

        Notes
        -----
        - Prefer returning a view that supports read/write semantics without
          unnecessary copies.

        Examples
        --------
        .. code-block:: python

            view = backend.slice_view(np.s_[0:2, :])
        """
        pass

    @staticmethod
    def from_arrays(*components, **kwargs) -> 'FuzzarrayBackend':
        """
        Factory to construct a backend from raw component arrays.

        Parameters
        ----------
        *components : array_like
            Component arrays representing the SoA layout for a specific ``mtype``.
        **kwargs
            Backend-specific keyword arguments (e.g., shape, q, dtype).

        Returns
        -------
        FuzzarrayBackend
            New backend instance initialised from the component arrays.

        Notes
        -----
        - Concrete subclasses should validate shapes and dtypes and may
          accept already-viewed arrays to avoid copies.

        Examples
        --------
        .. code-block:: python

            be = ConcreteBackend.from_arrays(m_arr, n_arr)
        """
        pass

    def fill_from_values(self, *values: float):
        """
        Broadcast provided component values to every element.

        Parameters
        ----------
        *values : float
            Values to broadcast to each component across the whole backend.

        Notes
        -----
        - Subclasses should validate the number of values against the
          expected number of components for the backend's ``mtype``.
        """
        pass

    def get_component_arrays(self) -> tuple:
        """
        Return the underlying component arrays.

        In fact, it is the attribute component of a specific mtype.

        Returns
        -------
        tuple
            Tuple of NumPy arrays that form the SoA backend.

        Notes
        -----
        - The order and meaning of the returned arrays is backend-specific.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, mtype='{self.mtype}')"

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the logical Fuzzarray.

        Returns
        -------
        int
            The number of dimensions (len(shape)).
        """
        return len(self.shape)

    # ================== Metadata for Validation and Introspection ==================
    @property
    @abstractmethod
    def cmpnum(self) -> int:
        """
        Return the number of component arrays expected by this backend.

        Returns
        -------
        int
            The number of component arrays (e.g., 2 for q-ROFNs).
        """
        pass

    @property
    @abstractmethod
    def cmpnames(self) -> Tuple[str, ...]:
        """
        Return the names of the component arrays.

        Returns
        -------
        Tuple[str, ...]
            A tuple of component names, e.g., ('md', 'nmd').
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Return the expected numpy dtype for component arrays.

        Returns
        -------
        np.dtype
            The expected data type, e.g., np.float64.
        """
        pass

    # ================== Smart Display General Implementation ==================

    def format_elements(self, format_spec: str = "") -> np.ndarray:
        """
        Smart, truncated formatting for element display.

        For small arrays the entire content is formatted. For larger arrays a
        head/tail strategy is used and middle elements are replaced by ellipses.

        Parameters
        ----------
        format_spec : str, optional
            Format specification passed from higher-level formatting calls.

        Returns
        -------
        numpy.ndarray
            Array of formatted strings with the same logical shape.

        Examples
        --------
        .. code-block:: python

            formatted = backend.format_elements('.3f')
        """
        if self.size == 0:
            return np.array([], dtype=object).reshape(self.shape)

        config = get_config()
        if self.size <= config.DISPLAY_THRESHOLD_SMALL:
            return self._format_all_elements(format_spec)
        else:
            return self._format_partial_elements(format_spec)

    def _format_all_elements(self, format_spec: str) -> np.ndarray:
        """Fully format all elements (for small datasets)"""
        # For small datasets, subclasses can optionally override this method for optimization
        # Default implementation: element-wise formatting
        result = np.empty(self.shape, dtype=object)
        formatter = self._get_element_formatter(format_spec)

        it = np.nditer(result, flags=['multi_index', 'refs_ok'], op_flags=['writeonly'])
        while not it.finished:
            idx = it.multi_index
            result[idx] = self._format_single_element(idx, formatter, format_spec)
            it.iternext()

        return result

    def _format_partial_elements(self, format_spec: str) -> np.ndarray:
        """
        Partially format elements for large arrays.

        Only selected indices are formatted; other entries are set to '...'.
        """
        # Calculate display parameters
        display_params = self._calculate_display_parameters()
        # Create result array, default filled with ellipses
        result = np.full(self.shape, '...', dtype=object)
        # Get indices to format
        indices_to_format = self._get_display_indices(display_params)
        # Only format the elements that need to be displayed
        formatter = self._get_element_formatter(format_spec)

        for idx in indices_to_format:
            result[idx] = self._format_single_element(idx, formatter, format_spec)

        return result

    def _calculate_display_parameters(self) -> dict:
        """
        Compute display parameters based on size and dimension.

        Returns
        -------
        dict
            Display parameters including 'edge_items', 'threshold', 'ndim',
            'shape' and 'size'.
        """
        config = get_config()

        if self.size < config.DISPLAY_THRESHOLD_MEDIUM:
            edge_items = config.DISPLAY_EDGE_ITEMS_MEDIUM
            threshold = config.DISPLAY_THRESHOLD_MEDIUM
        elif self.size < config.DISPLAY_THRESHOLD_LARGE:
            edge_items = config.DISPLAY_EDGE_ITEMS_LARGE
            threshold = config.DISPLAY_THRESHOLD_LARGE
        else:
            edge_items = config.DISPLAY_EDGE_ITEMS_HUGE
            threshold = config.DISPLAY_THRESHOLD_HUGE

        return {
            'edge_items': edge_items,
            'threshold': threshold,
            'ndim': self.ndim,
            'shape': self.shape,
            'size': self.size
        }

    def _get_display_indices(self, params: dict) -> list:
        """
        Determine which indices should be formatted for display.

        Parameters
        ----------
        params : dict
            Display parameters returned by :meth:`_calculate_display_parameters`.

        Returns
        -------
        list
            List of indices (tuples for ndim>1 or ints for 1D) to format.
        """
        indices = []
        edge_items = params['edge_items']
        shape = params['shape']

        if self.ndim == 1:
            if shape[0] <= 2 * edge_items + 1:
                indices = list(range(shape[0]))
            else:
                indices.extend(list(range(edge_items)))
                indices.extend(list(range(shape[0] - edge_items, shape[0])))

        elif self.ndim == 2:
            # 2D arrays: display corner regions
            rows, cols = shape
            # Determine which rows and columns to display
            if rows <= 2 * edge_items + 1:
                row_indices = list(range(rows))
            else:
                row_indices = (list(range(edge_items)) +
                               list(range(rows - edge_items, rows)))

            if cols <= 2 * edge_items + 1:
                col_indices = list(range(cols))
            else:
                col_indices = (list(range(edge_items)) +
                               list(range(cols - edge_items, cols)))
            # Generate all required (row, col) combinations
            for r in row_indices:
                for c in col_indices:
                    indices.append((r, c))

        else:
            # High-dimensional array: Recursive processing (simplified implementation)
            indices = self._get_high_dim_indices(shape, edge_items)

        return indices

    def _get_high_dim_indices(self, shape: tuple, edge_items: int) -> list:
        """Generate indices to display for high-dimensional arrays."""
        indices = []

        def generate_edge_indices(current_shape, current_idx=None):
            if current_idx is None:
                current_idx = []
            if not current_shape:
                indices.append(tuple(current_idx))
                return

            dim_size = current_shape[0]
            remaining_shape = current_shape[1:]

            if dim_size <= 2 * edge_items + 1:
                for i in range(dim_size):
                    generate_edge_indices(remaining_shape, current_idx + [i])
            else:
                for i in range(edge_items):
                    generate_edge_indices(remaining_shape, current_idx + [i])
                for i in range(dim_size - edge_items, dim_size):
                    generate_edge_indices(remaining_shape, current_idx + [i])

        generate_edge_indices(shape)
        return indices

    @abstractmethod
    def _get_element_formatter(self, format_spec: str) -> Callable:
        """
        Return a callable that formats a single element.

        The returned callable is used by formatting helpers and should accept
        the minimal information required to produce a string for a single
        element at a given index.

        Parameters
        ----------
        format_spec : str
            Format specification forwarded from higher layer.

        Returns
        -------
        Callable
            A callable used to format an element.

        Examples
        --------
        .. code-block:: python

            def formatter(idx, *args, **kwargs):
                return "<fuzznum>"
            return formatter
        """
        pass

    @abstractmethod
    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """
        Format a single element located at ``index``.

        Parameters
        ----------
        index : int or tuple
            Index of the element.
        formatter : Callable
            Callable returned by :meth:`_get_element_formatter`.
        format_spec : str
            Format specification.

        Returns
        -------
        str
            Formatted string for the element.

        Notes
        -----
        - Implementations decide how to extract component values and present
          them (e.g. as "(m,n)" or other textual form).

        Examples
        --------
        .. code-block:: python

            return formatter(index, self._comp1[index], self._comp2[index])
        """
        pass
