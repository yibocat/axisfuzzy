#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFN) Backend Implementation.

This module implements the high-performance backend for interval-valued q-rung
orthopair fuzzy numbers, providing efficient Struct-of-Arrays (SoA) storage and
vectorized operations for collections of interval-valued fuzzy numbers.
"""

from typing import Any, Tuple, cast, Optional, Callable

import numpy as np

from axisfuzzy.core import Fuzznum, FuzzarrayBackend, register_backend
from axisfuzzy.config import get_config

from .ivqrofn import IVQROFNStrategy


@register_backend
class IVQROFNBackend(FuzzarrayBackend):
    """
    SoA (Struct of Arrays) backend for Interval-Valued Q-Rung Orthopair Fuzzy Numbers.

    This backend stores membership and non-membership intervals as separate
    NumPy arrays with shape (..., 2), enabling high-performance vectorized
    operations on interval-valued fuzzy numbers.
    """

    mtype = 'ivqrofn'

    def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
        """Initialize the IVQROFNBackend."""
        super().__init__(shape, q, **kwargs)

    @property
    def cmpnum(self) -> int:
        return 2

    @property
    def cmpnames(self) -> Tuple[str, ...]:
        return ('md', 'nmd')

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    def _initialize_arrays(self):
        """Initialize the membership and non-membership interval arrays."""
        # Initialize arrays with extra dimension for interval storage [lower, upper]
        interval_shape = self.shape + (2,)
        self.mds = np.zeros(interval_shape, dtype=np.float64)
        self.nmds = np.zeros(interval_shape, dtype=np.float64)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """Create a Fuzznum object from the interval data at the given index."""
        md_interval = self.mds[index]
        nmd_interval = self.nmds[index]
        
        # Ensure we have proper interval shape (2,) for IVQROFN
        if md_interval.ndim == 0:
            # This shouldn't happen for IVQROFN, but handle it just in case
            raise ValueError(f"Invalid interval data: got scalar instead of interval")
        elif md_interval.shape == ():
            # Handle () shape (scalar array)
            raise ValueError(f"Invalid interval data: got empty shape instead of interval")
        elif md_interval.shape == (1, 2):
            # Handle (1, 2) shape - flatten to (2,)
            md_interval = md_interval.flatten()
            nmd_interval = nmd_interval.flatten()
        elif md_interval.shape != (2,):
            raise ValueError(f"Invalid interval shape: expected (2,), got {md_interval.shape}")
            
        return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_interval.copy(), nmd=nmd_interval.copy())

    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """Set interval data at the given index from a Fuzznum object."""
        if fuzznum.mtype != self.mtype:
            raise ValueError(f"Mtype mismatch: expected {self.mtype}, got {fuzznum.mtype}")

        if fuzznum.q != self.q:
            raise ValueError(f"Q parameter mismatch: expected {self.q}, got {fuzznum.q}")

        if fuzznum.md.shape != (2,) or fuzznum.nmd.shape != (2,):
            raise ValueError(f"Invalid interval shape: md {fuzznum.md.shape}, nmd {fuzznum.nmd.shape}")

        self.mds[index] = fuzznum.md
        self.nmds[index] = fuzznum.nmd

    def copy(self) -> 'IVQROFNBackend':
        """Create a deep copy of the backend."""
        new_backend = IVQROFNBackend(self.shape, self.q, **self.kwargs)
        new_backend.mds = self.mds.copy()
        new_backend.nmds = self.nmds.copy()
        return cast('IVQROFNBackend', new_backend)

    def slice_view(self, key) -> 'IVQROFNBackend':
        """Create a view of the backend with the given slice."""
        sliced_mds = self.mds[key]
        sliced_nmds = self.nmds[key]
        
        new_shape = sliced_mds.shape[:-1]  # Remove interval dimension
        new_backend = IVQROFNBackend(new_shape, self.q, **self.kwargs)
        new_backend.mds = sliced_mds
        new_backend.nmds = sliced_nmds
        return cast('IVQROFNBackend', new_backend)

    @staticmethod
    def _validate_fuzzy_constraints_static(mds: np.ndarray, nmds: np.ndarray, q: int) -> None:
        """
        Static method for validating IVQROFN fuzzy constraints.
        
        For IVQROFN, the constraints are:
        1. All values must be in [0, 1]
        2. Each interval must be ordered: lower ≤ upper  
        3. Q-rung constraint: md_upper^q + nmd_upper^q ≤ 1
        """
        if mds.shape != nmds.shape:
            raise ValueError(f"Shape mismatch: mds {mds.shape} vs nmds {nmds.shape}")
        
        if mds.shape[-1] != 2 or nmds.shape[-1] != 2:
            raise ValueError(f"Invalid interval shape: expected (..., 2), got mds {mds.shape}, nmds {nmds.shape}")
        
        from axisfuzzy.config import get_config
        epsilon = get_config().DEFAULT_EPSILON
        
        # Range validation
        if not (np.all(mds >= 0.0) and np.all(mds <= 1.0)):
            violation_indices = np.where((mds < 0.0) | (mds > 1.0))
            first_idx = tuple(idx[0] for idx in violation_indices)
            raise ValueError(f"IVQROFN range violation at index {first_idx}")
        
        if not (np.all(nmds >= 0.0) and np.all(nmds <= 1.0)):
            violation_indices = np.where((nmds < 0.0) | (nmds > 1.0))
            first_idx = tuple(idx[0] for idx in violation_indices)
            raise ValueError(f"IVQROFN range violation at index {first_idx}")
        
        # Interval ordering validation
        md_violations = mds[..., 0] > mds[..., 1] + epsilon
        if np.any(md_violations):
            violation_indices = np.where(md_violations)
            first_idx = tuple(idx[0] for idx in violation_indices)
            raise ValueError(f"IVQROFN interval ordering violation at index {first_idx}")
        
        nmd_violations = nmds[..., 0] > nmds[..., 1] + epsilon
        if np.any(nmd_violations):
            violation_indices = np.where(nmd_violations)
            first_idx = tuple(idx[0] for idx in violation_indices)
            raise ValueError(f"IVQROFN interval ordering violation at index {first_idx}")
        
        # Q-rung constraint validation
        md_upper = mds[..., 1]
        nmd_upper = nmds[..., 1]
        
        sum_of_powers = np.power(md_upper, q) + np.power(nmd_upper, q)
        violations = sum_of_powers > (1.0 + epsilon)
        
        if np.any(violations):
            violation_indices = np.where(violations)
            first_idx = tuple(idx[0] for idx in violation_indices)
            
            md_val = md_upper[first_idx]
            nmd_val = nmd_upper[first_idx]
            sum_val = sum_of_powers[first_idx]
            
            raise ValueError(
                f"IVQROFN constraint violation at index {first_idx}: "
                f"md_upper^q ({md_val}^{q}) + nmd_upper^q ({nmd_val}^{q}) = {sum_val:.4f} > 1.0"
            )

    @classmethod
    def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int, **kwargs) -> 'IVQROFNBackend':
        """
        Create an IVQROFNBackend from existing interval arrays.
        
        This method handles both properly shaped interval arrays and flattened arrays
        that may result from append operations or other array manipulations.
        
        Args:
            mds: Membership degree intervals array
            nmds: Non-membership degree intervals array  
            q: Q-rung parameter
            **kwargs: Additional backend arguments
            
        Returns:
            IVQROFNBackend: A new backend instance
            
        Raises:
            ValueError: If arrays cannot be reshaped to valid interval format
                       or if fuzzy constraints are violated
        """
        # Convert input arrays to numpy arrays
        mds = np.asarray(mds, dtype=np.float64)
        nmds = np.asarray(nmds, dtype=np.float64)
        
        # Handle flattened arrays from append operations
        # IVQROFN intervals should have shape (..., 2), but append may flatten them
        if mds.shape != nmds.shape:
            raise ValueError(f"Shape mismatch: mds {mds.shape} vs nmds {nmds.shape}")
        
        # Smart reshaping for flattened interval data
        if mds.ndim == 1:
            if mds.size == 2:
                # Single interval case: reshape to (1, 2) for backend consistency
                mds = mds.reshape(1, 2)
                nmds = nmds.reshape(1, 2)
            elif mds.size % 2 == 0:
                # Multiple intervals: reshape flattened data back to interval format
                # Example: [a, b, c, d] -> [[a, b], [c, d]] for 2 intervals
                num_intervals = mds.size // 2
                mds = mds.reshape(num_intervals, 2)
                nmds = nmds.reshape(num_intervals, 2)
            else:
                raise ValueError(f"Cannot reshape 1D arrays to interval format: size {mds.size} is not even")
        elif mds.ndim > 1 and mds.shape[-1] != 2:
            # Handle multi-dimensional arrays that are not interval-shaped
            if mds.size % 2 == 0:
                # Try to reshape to interval format
                total_intervals = mds.size // 2
                mds = mds.reshape(total_intervals, 2)
                nmds = nmds.reshape(total_intervals, 2)
            else:
                raise ValueError(f"Cannot reshape arrays to interval format: mds {mds.shape}, nmds {nmds.shape}")
        
        # Now validate the properly shaped arrays
        cls._validate_fuzzy_constraints_static(mds, nmds, q=q)
        
        logical_shape = mds.shape[:-1]
        backend = cls(logical_shape, q, **kwargs)
        backend.mds = mds.copy()
        backend.nmds = nmds.copy()
        return backend

    def fill_from_values(self, md_interval: Any, nmd_interval: Any):
        """Fill all elements with the given intervals."""
        md_arr = np.asarray(md_interval, dtype=np.float64)
        nmd_arr = np.asarray(nmd_interval, dtype=np.float64)
        
        if md_arr.size != 2 or nmd_arr.size != 2:
            raise ValueError("Intervals must contain exactly 2 elements")
        
        md_arr = md_arr.flatten()
        nmd_arr = nmd_arr.flatten()
        
        # Validate
        test_mds = np.broadcast_to(md_arr, (1, 2))
        test_nmds = np.broadcast_to(nmd_arr, (1, 2))
        self._validate_fuzzy_constraints_static(test_mds, test_nmds, self.q)
        
        self.mds[...] = md_arr
        self.nmds[...] = nmd_arr

    def get_component_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the underlying component arrays."""
        return self.mds, self.nmds

    # ================== Smart Display System Implementation ==================

    def _get_element_formatter(self, format_spec: str) -> Callable:
        """Get element formatting function."""
        precision = get_config().DEFAULT_PRECISION

        if format_spec in ('p', 'j', 'r'):
            strategy_formatter = IVQROFNStrategy(q=self.q)
            return lambda md, nmd: strategy_formatter.format_from_components(md, nmd, format_spec)
        else:
            return self._create_default_formatter(precision)

    def _create_default_formatter(self, precision: int) -> Callable:
        """Create default formatting function."""
        def format_interval_pair(md_interval: np.ndarray, nmd_interval: np.ndarray) -> str:
            def format_interval(interval):
                fmt = f"%.{precision}f"
                lower_str = (fmt % interval[0]).rstrip('0').rstrip('.')
                upper_str = (fmt % interval[1]).rstrip('0').rstrip('.')
                lower_str = lower_str if lower_str else '0'
                upper_str = upper_str if upper_str else '0'
                return f"[{lower_str},{upper_str}]"
            
            md_str = format_interval(md_interval)
            nmd_str = format_interval(nmd_interval)
            return f"<{md_str},{nmd_str}>"

        return format_interval_pair

    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """Format a single element."""
        md_interval = self.mds[index]
        nmd_interval = self.nmds[index]
        return formatter(md_interval, nmd_interval)

    def _format_all_elements(self, format_spec: str) -> np.ndarray:
        """Format all elements."""
        precision = get_config().DEFAULT_PRECISION

        if format_spec in ('p', 'j', 'r'):
            return super()._format_all_elements(format_spec)

        # Default format with vectorized operations
        fmt = f"%.{precision}f"
        
        # Format lower and upper bounds
        md_lower_strs = np.char.mod(fmt, np.round(self.mds[..., 0], precision))
        md_upper_strs = np.char.mod(fmt, np.round(self.mds[..., 1], precision))
        nmd_lower_strs = np.char.mod(fmt, np.round(self.nmds[..., 0], precision))
        nmd_upper_strs = np.char.mod(fmt, np.round(self.nmds[..., 1], precision))

        def _trim(arr: np.ndarray) -> np.ndarray:
            trimmed = np.char.rstrip(np.char.rstrip(arr, '0'), '.')
            return np.where(trimmed == '', '0', trimmed)

        md_lower_trimmed = _trim(md_lower_strs)
        md_upper_trimmed = _trim(md_upper_strs)
        nmd_lower_trimmed = _trim(nmd_lower_strs)
        nmd_upper_trimmed = _trim(nmd_upper_strs)

        # Combine into interval format
        md_intervals = np.char.add(
            np.char.add(
                np.char.add("[", md_lower_trimmed),
                np.char.add(",", md_upper_trimmed)
            ),
            "]"
        )
        
        nmd_intervals = np.char.add(
            np.char.add(
                np.char.add("[", nmd_lower_trimmed),
                np.char.add(",", nmd_upper_trimmed)
            ),
            "]"
        )

        # Final format: <[a,b],[c,d]>
        combined = np.char.add(
            np.char.add(
                np.char.add("<", md_intervals),
                np.char.add(",", nmd_intervals)
            ),
            ">"
        )
        
        return np.array(combined, dtype=object)