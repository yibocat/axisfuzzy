#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFN) Strategy Implementation.

This module implements interval-valued q-rung orthopair fuzzy numbers, where both
membership and non-membership degrees are represented as intervals [a, b] with a ≤ b.
This extends QROFN by allowing uncertainty in the degree values themselves while
maintaining the q-rung orthopair constraint on the upper bounds.

The IVQROFNStrategy class provides:
- Interval-valued membership and non-membership degrees (md ∈ [a1, b1], nmd ∈ [a2, b2])
- Constraint validation: max(md)^q + max(nmd)^q ≤ 1
- High-performance NumPy array operations (unlike QROHFN's variable-length sets)
- Automatic data transformation from lists/tuples to NumPy arrays

Mathematical Foundation:
    An interval-valued q-rung orthopair fuzzy number is characterized by:
    - Membership interval: μ_A(x) = [μ_A^L(x), μ_A^U(x)] where 0 ≤ μ_A^L(x) ≤ μ_A^U(x) ≤ 1
    - Non-membership interval: ν_A(x) = [ν_A^L(x), ν_A^U(x)] where 0 ≤ ν_A^L(x) ≤ ν_A^U(x) ≤ 1
    - Constraint: (μ_A^U(x))^q + (ν_A^U(x))^q ≤ 1, where q ≥ 1

Examples:
    .. code-block:: python

        from axisfuzzy import Fuzznum
        
        # Create IVQROFN with membership interval [0.6, 0.8] and non-membership interval [0.1, 0.2]
        ivqrofn = Fuzznum(mtype='ivqrofn', q=2).create(md=[0.6, 0.8], nmd=[0.1, 0.2])
        print(ivqrofn)  # Output: <[0.6,0.8],[0.1,0.2]>
        
        # Automatic constraint validation
        ivqrofn.md = [0.9, 0.95]  # Valid: 0.95^2 + 0.2^2 = 0.9425 ≤ 1
        ivqrofn.nmd = [0.3, 0.4]  # Raises ValueError: 0.95^2 + 0.4^2 = 1.0625 > 1
"""

from typing import Optional, Any, Union, List

import numpy as np

from axisfuzzy.config import get_config
from axisfuzzy.core import FuzznumStrategy, register_strategy


@register_strategy
class IVQROFNStrategy(FuzznumStrategy):
    """
    Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFN) Strategy Implementation.
    
    This strategy implements interval-valued q-rung orthopair fuzzy numbers where
    both membership and non-membership degrees are represented as intervals [a, b].
    Unlike QROHFN which uses variable-length hesitant sets, IVQROFN uses fixed-length
    intervals enabling high-performance vectorized operations.
    
    Attributes:
        mtype (str): Membership type identifier, set to 'ivqrofn'
        md (Optional[np.ndarray]): Membership degree interval [a, b] where a ≤ b
        nmd (Optional[np.ndarray]): Non-membership degree interval [c, d] where c ≤ d
    
    Mathematical Properties:
        - Membership interval: md = [md_L, md_U] where 0 ≤ md_L ≤ md_U ≤ 1
        - Non-membership interval: nmd = [nmd_L, nmd_U] where 0 ≤ nmd_L ≤ nmd_U ≤ 1
        - Constraint: md_U^q + nmd_U^q ≤ 1 (q-rung constraint on upper bounds)
        - Fixed array length: exactly 2 elements per interval
    
    Performance Characteristics:
        - High-performance NumPy vectorization (unlike QROHFN)
        - Fixed memory footprint (2 × float64 per component)
        - Efficient constraint validation using array operations
        - Seamless integration with mathematical operations
    
    Examples:
        .. code-block:: python
        
            # Create strategy instance
            strategy = IVQROFNStrategy(q=2)
            strategy.md = [0.6, 0.8]    # Valid interval
            strategy.nmd = [0.1, 0.2]   # Valid interval
            
            # Constraint validation
            strategy.md = [0.8, 0.6]    # Raises ValueError: interval not ordered
            strategy.nmd = [0.3, 0.4]   # Raises ValueError: constraint violation
    """
    
    # Type identifier for registration system
    mtype = 'ivqrofn'
    
    # Core fuzzy components with type annotations
    md: Optional[np.ndarray] = None
    nmd: Optional[np.ndarray] = None

    def __init__(self, q: Optional[int] = None):
        """
        Initialize IVQROFN strategy with validation framework.
        
        Parameters:
            q (Optional[int]): q-rung parameter for constraint validation (q ≥ 1)
        
        Notes:
            The initialization sets up a three-stage validation pipeline:
            1. Transformers: Convert input data to NumPy arrays
            2. Validators: Check interval validity and range constraints
            3. Change callbacks: Enforce q-rung orthopair constraints
        """
        super().__init__(q=q)
        
        # Data transformer: Convert lists/tuples to NumPy arrays
        def _to_interval_array(x):
            """Transform input to 2-element NumPy array representing an interval."""
            if x is None:
                return None
            
            # Convert to NumPy array
            arr = np.asarray(x, dtype=np.float64)
            
            # Ensure exactly 2 elements for interval representation
            if arr.size != 2:
                raise ValueError(f"Interval must contain exactly 2 elements, got {arr.size}")
            
            # Ensure 1-dimensional array
            arr = arr.flatten()
            
            return arr
        
        # Register transformers for automatic data conversion
        self.add_attribute_transformer('md', _to_interval_array)
        self.add_attribute_transformer('nmd', _to_interval_array)
        
        # Interval and range validator
        def _interval_validator(x):
            """Validate interval properties: ordering and range constraints."""
            if x is None:
                return True
            
            # First convert to array if it's not already (for initial validation)
            try:
                if not isinstance(x, np.ndarray):
                    arr = np.asarray(x, dtype=np.float64)
                    if arr.size != 2:
                        return False
                    arr = arr.flatten()
                else:
                    arr = x
                    if arr.size != 2:
                        return False
            except (ValueError, TypeError):
                return False
            
            # Check range constraints: all values in [0, 1]
            if not (np.all(arr >= 0.0) and np.all(arr <= 1.0)):
                return False
            
            # Check interval ordering: a ≤ b
            if arr[0] > arr[1]:
                return False
            
            return True
        
        # Register attribute validators
        self.add_attribute_validator('md', _interval_validator)
        self.add_attribute_validator('nmd', _interval_validator)
        
        # Register change callbacks for constraint validation
        self.add_change_callback('md', self._on_interval_change)
        self.add_change_callback('nmd', self._on_interval_change)
        self.add_change_callback('q', self._on_q_change)

    def _fuzz_constraint(self):
        """
        Validate interval-valued q-rung orthopair constraints.
        
        For IVQROFN, the constraint is applied to the upper bounds of the intervals:
        max(md)^q + max(nmd)^q ≤ 1
        
        This ensures that the most optimistic scenario (maximum membership and
        non-membership) still satisfies the fundamental q-rung orthopair constraint.
        
        Raises:
            ValueError: If the constraint is violated
        """
        if self.md is not None and self.nmd is not None and self.q is not None:
            # Get upper bounds of intervals
            md_upper = self.md[1]  # Upper bound of membership interval
            nmd_upper = self.nmd[1]  # Upper bound of non-membership interval
            
            # Apply q-rung constraint to upper bounds
            sum_of_powers = md_upper ** self.q + nmd_upper ** self.q
            epsilon = get_config().DEFAULT_EPSILON
            
            if sum_of_powers > 1.0 + epsilon:
                raise ValueError(
                    f"IVQROFN constraint violation: "
                    f"max(md)^q ({md_upper}^{self.q}) + max(nmd)^q ({nmd_upper}^{self.q}) = {sum_of_powers:.4f} > 1.0. "
                    f"(q: {self.q}, md: {self.md}, nmd: {self.nmd})"
                )

    def _on_interval_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """
        Callback triggered when membership or non-membership interval changes.
        
        This method is automatically called whenever md or nmd is modified,
        ensuring that the q-rung orthopair constraint is maintained.
        
        Parameters:
            attr_name (str): Name of the changed attribute ('md' or 'nmd')
            old_value (Any): Previous value of the attribute
            new_value (Any): New value being assigned
        """
        if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
            self._fuzz_constraint()

    def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """
        Callback triggered when q parameter changes.
        
        This method ensures that existing interval values remain valid
        under the new q-rung constraint.
        
        Parameters:
            attr_name (str): Name of the changed attribute ('q')
            old_value (Any): Previous q value
            new_value (Any): New q value
        """
        if self.md is not None and self.nmd is not None and new_value is not None:
            self._fuzz_constraint()

    def _validate(self) -> None:
        """
        Perform comprehensive validation of the interval-valued fuzzy number.
        
        This method ensures mathematical consistency by checking:
        1. Base class validation (inherited constraints)
        2. Interval-specific constraints (ordering and q-rung constraint)
        
        Raises:
            ValueError: If any validation fails
        """
        super()._validate()
        self._fuzz_constraint()

    def format_from_components(self, md: np.ndarray, nmd: np.ndarray, format_spec: str = "") -> str:
        """
        Format IVQROFN from interval components.
        
        This method provides flexible string representation of the interval-valued
        fuzzy number based on the membership and non-membership intervals.
        
        Parameters:
            md (np.ndarray): Membership degree interval [a, b]
            nmd (np.ndarray): Non-membership degree interval [c, d]
            format_spec (str): Format specification string
                - '' (default): Standard representation <[a,b],[c,d]>
                - 'p': Parentheses format ([a,b], [c,d])
                - 'j': JSON format {"mtype": "ivqrofn", "md": [a,b], "nmd": [c,d], "q": q}
        
        Returns:
            str: Formatted string representation of the IVQROFN
        
        Examples:
            .. code-block:: python
            
                strategy = IVQROFNStrategy(q=2)
                md_interval = np.array([0.6, 0.8])
                nmd_interval = np.array([0.1, 0.2])
                
                print(strategy.format_from_components(md_interval, nmd_interval))      # <[0.6,0.8],[0.1,0.2]>
                print(strategy.format_from_components(md_interval, nmd_interval, 'p')) # ([0.6,0.8], [0.1,0.2])
        """
        if md is None and nmd is None:
            return "<>"
        
        precision = get_config().DEFAULT_PRECISION
        
        def _format_interval(interval: np.ndarray) -> str:
            """Format a 2-element interval array."""
            if interval is None or interval.size != 2:
                return "[]"
            
            def strip_trailing_zeros(x: float) -> str:
                """Remove trailing zeros from float representation."""
                s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
                return s if s else "0"
            
            lower_str = strip_trailing_zeros(interval[0])
            upper_str = strip_trailing_zeros(interval[1])
            return f"[{lower_str},{upper_str}]"
        
        md_str = _format_interval(md)
        nmd_str = _format_interval(nmd)
        
        # Parentheses format
        if format_spec == 'p':
            return f"({md_str}, {nmd_str})"
        
        # JSON format
        if format_spec == 'j':
            import json
            md_list = md.tolist() if md is not None else None
            nmd_list = nmd.tolist() if nmd is not None else None
            return json.dumps({
                'mtype': self.mtype,
                'md': md_list,
                'nmd': nmd_list,
                'q': self.q
            })
        
        # Default format: <[a,b],[c,d]>
        return f"<{md_str},{nmd_str}>"

    def report(self) -> str:
        """
        Generate a report string for the current IVQROFN.
        
        Returns:
            str: String representation of the IVQROFN in standard format
        
        Examples:
            .. code-block:: python
            
                strategy = IVQROFNStrategy(q=2)
                strategy.md = [0.6, 0.8]
                strategy.nmd = [0.1, 0.2]
                print(strategy.report())  # <[0.6,0.8],[0.1,0.2]>
        """
        return self.format_from_components(self.md, self.nmd)

    def str(self) -> str:
        """
        Generate string representation of the IVQROFN.
        
        Returns:
            str: String representation of the IVQROFN
        
        Notes:
            This method provides the default string conversion for the IVQROFN,
            used by Python's str() function and string formatting operations.
        """
        return self.format_from_components(self.md, self.nmd)

    def __format__(self, format_spec: str) -> str:
        """
        Provide custom formatting support for the IVQROFN.
        
        This method enables the use of Python's format() function and f-string
        formatting with custom format specifications.
        
        Parameters:
            format_spec (str): Format specification string
        
        Returns:
            str: Formatted string representation
        
        Examples:
            .. code-block:: python
            
                strategy = IVQROFNStrategy(q=2)
                strategy.md = [0.6, 0.8]
                strategy.nmd = [0.1, 0.2]
                
                print(f"{strategy}")      # <[0.6,0.8],[0.1,0.2]>
                print(f"{strategy:p}")    # ([0.6,0.8], [0.1,0.2])
                print(f"{strategy:j}")    # {"mtype": "ivqrofn", "md": [0.6, 0.8], "nmd": [0.1, 0.2], "q": 2}
        """
        # Handle custom format specifications
        if format_spec and format_spec not in ['r', 'p', 'j']:
            # For non-custom format specs, apply to the string representation
            return format(self.str(), format_spec)
        
        # Use custom formatting for recognized specifications
        return self.format_from_components(self.md, self.nmd, format_spec)