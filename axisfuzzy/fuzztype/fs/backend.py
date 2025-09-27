#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) Backend Implementation.

This module implements the high-performance backend for classical fuzzy sets,
providing efficient Struct-of-Arrays (SoA) storage and vectorized operations
for collections of fuzzy numbers with only membership degrees.

The FSBackend class provides:
- Single membership degree array storage (md ∈ [0, 1])
- Optimal NumPy integration and vectorized operations
- Efficient memory layout for large-scale fuzzy computations
- Simple constraint validation (only [0, 1] range checking)

Mathematical Foundation:
    A fuzzy set A in universe X is characterized by a membership function
    μ_A: X → [0, 1], where μ_A(x) represents the degree of membership of
    element x in fuzzy set A. This is the most fundamental fuzzy type.

Examples:
    .. code-block:: python

        from axisfuzzy import Fuzzarray
        import numpy as np
        
        # Create FS array from membership degrees
        md_array = np.array([0.1, 0.5, 0.8, 0.9])
        fs_backend = FSBackend.from_arrays(md_array)
        fs_array = Fuzzarray(backend=fs_backend)
        
        # High-performance operations
        print(fs_array.shape)  # (4,)
        print(fs_array[0])     # <0.1>
"""

from typing import Any, Tuple, cast, Optional, Callable

import numpy as np

from axisfuzzy.core import Fuzznum, FuzzarrayBackend, register_backend
from axisfuzzy.config import get_config

from .fn import FSStrategy


@register_backend
class FSBackend(FuzzarrayBackend):
    """
    SoA (Struct of Arrays) backend for Fuzzy Sets (FS).

    This backend stores membership degrees as a single NumPy array, enabling
    high-performance vectorized operations on classical fuzzy sets. FS is the
    most fundamental fuzzy type with only membership degrees and no complex
    mathematical constraints beyond the [0, 1] range.

    Attributes:
        mtype (str): Membership type identifier, set to 'fs'
        mds (np.ndarray): Membership degrees array with values in [0, 1]

    Mathematical Properties:
        - Single component: membership degree md ∈ [0, 1]
        - No cross-component constraints (unlike QROFN)
        - Represents classical Zadeh fuzzy sets
        - Optimal for basic fuzzy logic operations

    Performance Characteristics:
        - Minimal memory footprint (single array storage)
        - Maximum NumPy vectorization compatibility
        - Efficient for large-scale fuzzy computations
        - Simple validation (only range checking)

    Examples:
        .. code-block:: python

            # Create backend with shape (2, 3)
            backend = FSBackend((2, 3))
            
            # Fill with membership degrees
            backend.fill_from_values(0.7)
            
            # Access elements
            fuzznum = backend.get_fuzznum_view((0, 1))
            print(fuzznum)  # <0.7>
    """

    # Type identifier for registration system
    mtype = 'fs'

    def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
        """
        Initialize the FSBackend.

        Parameters:
            shape (Tuple[int, ...]): The logical shape of the fuzzy array
            q (Optional[int]): q-rung parameter (inherited for compatibility,
                             not used in basic FS but maintained for framework consistency)
            **kwargs: Additional backend-specific keyword arguments

        Notes:
            The q parameter is inherited from the base FuzzarrayBackend class
            for compatibility with the framework, but is not used in basic
            fuzzy sets. It's maintained to ensure consistent interface across
            all fuzzy types.
        """
        super().__init__(shape, q, **kwargs)

    # ================== Component Metadata Properties ==================

    @property
    def cmpnum(self) -> int:
        """
        Return the number of component arrays for FS.

        Returns:
            int: Always 1 for FS (only membership degree)
        """
        return 1

    @property
    def cmpnames(self) -> Tuple[str, ...]:
        """
        Return the names of the component arrays.

        Returns:
            Tuple[str, ...]: Component names ('md',) for membership degree
        """
        return ('md',)

    @property
    def dtype(self) -> np.dtype:
        """
        Return the expected numpy dtype for component arrays.

        Returns:
            np.dtype: np.float64 for optimal precision and performance
        """
        return np.dtype(np.float64)

    # ================== Core Array Operations ==================

    def _initialize_arrays(self):
        """
        Initialize the membership degrees array for FS data.

        This method creates a single NumPy array to store membership degrees
        for all elements in the fuzzy array. The array is initialized with
        zeros and uses float64 dtype for optimal precision.

        Notes:
            Called automatically during backend construction. The array is
            initialized with zeros, which represent valid FS values (md=0
            indicates no membership).
        """
        # Initialize single array for membership degrees
        self.mds = np.zeros(self.shape, dtype=np.float64)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Create a Fuzznum object from the data at the given index.

        This method extracts the membership degree at the specified index
        and creates a corresponding Fuzznum object. The returned object
        provides a view into the backend data.

        Parameters:
            index (Any): Index following NumPy indexing semantics

        Returns:
            Fuzznum: A Fuzznum object representing the FS at the given index

        Examples:
            .. code-block:: python

                backend = FSBackend((3,))
                backend.mds[1] = 0.8
                fuzznum = backend.get_fuzznum_view(1)
                print(fuzznum.md)  # 0.8
        """
        # Extract scalar value from NumPy array
        md_value = float(self.mds[index])

        # Create Fuzznum object with correct parameters
        return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value)

    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """
        Set data at the given index from a Fuzznum object.

        This method validates the input Fuzznum and extracts its membership
        degree to store in the backend array at the specified index.

        Parameters:
            index (Any): Target location following NumPy indexing semantics
            fuzznum (Fuzznum): Source Fuzznum object providing the membership degree

        Raises:
            ValueError: If fuzznum has incompatible mtype
            ValueError: If fuzznum has incompatible q parameter

        Examples:
            .. code-block:: python

                backend = FSBackend((2,))
                fs_num = Fuzznum(mtype='fs').create(md=0.7)
                backend.set_fuzznum_data(0, fs_num)
                print(backend.mds[0])  # 0.7
        """
        # Validate mtype compatibility
        if fuzznum.mtype != self.mtype:
            raise ValueError(f"Mtype mismatch: expected {self.mtype}, got {fuzznum.mtype}")

        # Validate q parameter compatibility
        if fuzznum.q != self.q:
            raise ValueError(f"Q parameter mismatch: expected {self.q}, got {fuzznum.q}")

        # Extract membership degree and store in array
        self.mds[index] = fuzznum.md

    # ================== Memory Management Operations ==================

    def copy(self) -> 'FSBackend':
        """
        Create a deep copy of the backend.

        This method produces a new FSBackend instance with duplicated arrays,
        ensuring no shared memory between the original and copy.

        Returns:
            FSBackend: New backend instance with copied data

        Examples:
            .. code-block:: python

                original = FSBackend((2, 2))
                original.fill_from_values(0.5)
                
                copy = original.copy()
                copy.mds[0, 0] = 0.9
                
                print(original.mds[0, 0])  # 0.5 (unchanged)
                print(copy.mds[0, 0])      # 0.9
        """
        new_backend = FSBackend(self.shape, self.q, **self.kwargs)
        new_backend.mds = self.mds.copy()
        return cast('FSBackend', new_backend)

    def slice_view(self, key) -> 'FSBackend':
        """
        Create a view of the backend with the given slice.

        This method returns a new FSBackend that represents a view/slice of
        the original backend. The returned backend shares memory with the
        original for efficient operations.

        Parameters:
            key: Slice key following NumPy indexing semantics

        Returns:
            FSBackend: Backend representing the requested slice

        Examples:
            .. code-block:: python

                backend = FSBackend((4, 4))
                backend.fill_from_values(0.8)
                
                # Create a 2x2 view of the top-left corner
                view = backend.slice_view(np.s_[0:2, 0:2])
                print(view.shape)  # (2, 2)
                
                # Modifications to view affect original
                view.mds[0, 0] = 0.1
                print(backend.mds[0, 0])  # 0.1
        """
        new_shape = self.mds[key].shape
        new_backend = FSBackend(new_shape, self.q, **self.kwargs)
        new_backend.mds = self.mds[key]
        return cast('FSBackend', new_backend)

    # ================== Factory Methods ==================

    @classmethod
    def from_arrays(cls, mds: np.ndarray, q: Optional[int] = None, **kwargs) -> 'FSBackend':
        """
        Create an FSBackend from an existing membership degrees array.

        This factory method provides an efficient way to construct an FSBackend
        from pre-computed NumPy arrays, with automatic constraint validation.

        Parameters:
            mds (np.ndarray): Membership degrees array with values in [0, 1]
            q (Optional[int]): q-rung parameter (for framework compatibility)
            **kwargs: Additional backend-specific parameters

        Returns:
            FSBackend: New FSBackend instance initialized from the array

        Raises:
            ValueError: If membership degrees violate [0, 1] constraints

        Examples:
            .. code-block:: python

                import numpy as np
                
                # Create from existing array
                md_array = np.array([[0.1, 0.5], [0.8, 0.9]])
                backend = FSBackend.from_arrays(md_array)
                
                print(backend.shape)  # (2, 2)
                print(backend.mds[1, 1])  # 0.9
        """
        # Validate FS constraints: membership degrees must be in [0, 1]
        cls._validate_fuzzy_constraints_static(mds)
        
        # Create backend and assign array
        backend = cls(mds.shape, q, **kwargs)
        backend.mds = mds.copy()
        return backend

    @staticmethod
    def _validate_fuzzy_constraints_static(mds: np.ndarray) -> None:
        """
        Static method for validating FS fuzzy constraints with enhanced error messages.
        
        For FS, the only constraint is that membership degrees must be in [0, 1].
        This provides detailed error messages that explain FS mathematical foundations.
        
        Parameters:
            mds (np.ndarray): Membership degrees array to validate
            
        Raises:
            ValueError: If any membership degrees are outside [0, 1] range with detailed explanation
        """
        # Vectorized constraint check: 0 <= md <= 1
        violations_low = mds < 0.0
        violations_high = mds > 1.0
        
        if np.any(violations_low) or np.any(violations_high):
            # Find first violation for detailed error message
            if np.any(violations_low):
                violation_indices = np.where(violations_low)
                first_idx = tuple(idx[0] for idx in violation_indices)
                md_val = mds[first_idx]
                raise ValueError(
                    f"FS constraint violation at index {first_idx}: "
                    f"membership degree {md_val} < 0.0. "
                    f"Classical Fuzzy Sets (FS) require membership degrees μ ∈ [0, 1], "
                    f"where 0 indicates no membership and 1 indicates full membership. "
                    f"This is the fundamental constraint of Zadeh's fuzzy set theory."
                )
            else:
                violation_indices = np.where(violations_high)
                first_idx = tuple(idx[0] for idx in violation_indices)
                md_val = mds[first_idx]
                raise ValueError(
                    f"FS constraint violation at index {first_idx}: "
                    f"membership degree {md_val} > 1.0. "
                    f"Classical Fuzzy Sets (FS) require membership degrees μ ∈ [0, 1], "
                    f"where 0 indicates no membership and 1 indicates full membership. "
                    f"This is the fundamental constraint of Zadeh's fuzzy set theory."
                )

    def fill_from_values(self, md_value: float):
        """
        Fill all elements with the given membership degree value.

        This method provides an efficient way to initialize all elements
        in the fuzzy array with the same membership degree.

        Parameters:
            md_value (float): Membership degree value in [0, 1] to fill

        Raises:
            ValueError: If md_value is outside [0, 1] range

        Examples:
            .. code-block:: python

                backend = FSBackend((3, 3))
                backend.fill_from_values(0.7)
                
                print(np.all(backend.mds == 0.7))  # True
        """
        # Validate input value with enhanced error message
        if not (0.0 <= md_value <= 1.0):
            raise ValueError(
                f"FS membership degree constraint violation: {md_value} ∉ [0, 1]. "
                f"Classical Fuzzy Sets (FS) require membership degrees in range [0, 1], "
                f"where 0 indicates no membership and 1 indicates full membership in the fuzzy set. "
                f"This is the fundamental constraint of Zadeh's fuzzy set theory."
            )
        
        # Fill array efficiently
        self.mds.fill(md_value)

    def get_component_arrays(self) -> Tuple[np.ndarray]:
        """
        Get the underlying component arrays.

        For FS, this returns a single-element tuple containing the membership
        degrees array.

        Returns:
            Tuple[np.ndarray]: Tuple containing the membership degrees array

        Examples:
            .. code-block:: python

                backend = FSBackend((2, 2))
                (mds,) = backend.get_component_arrays()
                print(mds.shape)  # (2, 2)
        """
        return (self.mds,)

    # ================== Smart Display System Implementation ==================

    def _get_element_formatter(self, format_spec: str) -> Callable:
        """
        Get element formatting function for FS display.

        This method returns a callable that formats individual FS elements
        according to the specified format. For FS, formatting is simpler
        than QROFN since there's only one component.

        Parameters:
            format_spec (str): Format specification string
                - '' (default): Standard representation <md>
                - 'p': Parentheses format (md)
                - 'j': JSON format {"mtype": "fs", "md": md}
                - 'r': Raw format md

        Returns:
            Callable: Function that formats a single membership degree

        Examples:
            .. code-block:: python

                backend = FSBackend((2,))
                formatter = backend._get_element_formatter('')
                result = formatter(0.75)  # "<0.75>"
        """
        precision = get_config().DEFAULT_PRECISION

        if format_spec in ('p', 'j', 'r'):
            # Special formats: use Strategy for consistent formatting
            strategy_formatter = FSStrategy(q=self.q)
            return lambda md: strategy_formatter.format_from_components(md, format_spec)
        else:
            # Default format: use high-performance string operations
            return self._create_default_formatter(precision)

    def _create_default_formatter(self, precision: int) -> Callable:
        """
        Create default formatting function for FS elements.

        This method creates an optimized formatter for the default FS
        representation format <md>.

        Parameters:
            precision (int): Number of decimal places for formatting

        Returns:
            Callable: Optimized formatting function

        Examples:
            .. code-block:: python

                formatter = backend._create_default_formatter(3)
                result = formatter(0.75)  # "<0.75>"
        """
        def format_md(md: float) -> str:
            # Efficient numerical formatting
            fmt = f"%.{precision}f"
            md_str = (fmt % md).rstrip('0').rstrip('.')

            # Handle all-zero case
            md_str = md_str if md_str else '0'

            return f"<{md_str}>"

        return format_md

    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """
        Format a single FS element at the specified index.

        This method extracts the membership degree at the given index
        and applies the formatter to produce a string representation.

        Parameters:
            index (Any): Index of the element to format
            formatter (Callable): Formatting function from _get_element_formatter
            format_spec (str): Format specification string

        Returns:
            str: Formatted string representation of the FS element

        Examples:
            .. code-block:: python

                backend = FSBackend((2,))
                backend.mds[0] = 0.8
                formatter = backend._get_element_formatter('')
                result = backend._format_single_element(0, formatter, '')
                print(result)  # "<0.8>"
        """
        md_value = float(self.mds[index])
        return formatter(md_value)

    def _format_all_elements(self, format_spec: str) -> np.ndarray:
        """
        Optimized formatting for all FS elements.

        This method provides vectorized formatting for small FS arrays,
        using NumPy string operations for optimal performance.

        Parameters:
            format_spec (str): Format specification string

        Returns:
            np.ndarray: Array of formatted strings with same logical shape

        Examples:
            .. code-block:: python

                backend = FSBackend((2, 2))
                backend.fill_from_values(0.5)
                formatted = backend._format_all_elements('')
                print(formatted[0, 0])  # "<0.5>"
        """
        precision = get_config().DEFAULT_PRECISION

        if format_spec in ('p', 'j', 'r'):
            # Special formats: fallback to element-wise processing
            return super()._format_all_elements(format_spec)

        # Default format: use vectorized string operations for performance
        fmt = f"%.{precision}f"
        md_strs = np.char.mod(fmt, np.round(self.mds, precision))

        def _trim(arr: np.ndarray) -> np.ndarray:
            # Remove trailing zeros and decimal points
            trimmed = np.char.rstrip(np.char.rstrip(arr, '0'), '.')
            # Handle case where everything is stripped (e.g., "0.0000")
            return np.where(trimmed == '', '0', trimmed)

        md_trimmed = _trim(md_strs)

        # Combine with angle brackets for FS format
        combined = np.char.add(
            np.char.add("<", md_trimmed),
            ">"
        )
        return np.array(combined, dtype=object)
