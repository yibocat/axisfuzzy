#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) Strategy Implementation.

This module implements the most basic and efficient fuzzy number type - Fuzzy Sets.
FS represents classical fuzzy sets with only membership degrees, making it the
foundation for fuzzy logic applications and the most commonly used fuzzy type.

The FSStrategy class provides:
- Simple membership degree representation (md ∈ [0, 1])
- Efficient validation and constraint checking
- Clean string formatting and representation
- Foundation for more complex fuzzy types

Mathematical Foundation:
    A fuzzy set A in universe X is characterized by a membership function
    μ_A: X → [0, 1], where μ_A(x) represents the degree of membership of
    element x in fuzzy set A.

Examples:
    .. code-block:: python

        from axisfuzzy import Fuzznum
        
        # Create a basic fuzzy set with membership degree 0.7
        fs = Fuzznum(mtype='fs').create(md=0.7)
        print(fs)  # Output: <0.7>
        
        # Validate membership degree constraints
        fs.md = 1.2  # Raises ValueError: membership degree must be in [0, 1]
"""

from typing import Optional, Any

import numpy as np

from axisfuzzy.config import get_config
from axisfuzzy.core import FuzznumStrategy, register_strategy


@register_strategy
class FSStrategy(FuzznumStrategy):
    """
    Fuzzy Sets (FS) Strategy Implementation.
    
    This strategy implements the most fundamental fuzzy number type - classical
    fuzzy sets with only membership degrees. FS provides the foundation for
    fuzzy logic applications and serves as the base for more complex fuzzy types.
    
    Attributes:
        mtype (str): Membership type identifier, set to 'fs'
        md (Optional[float]): Membership degree in range [0, 1]
    
    Mathematical Properties:
        - Membership degree: md ∈ [0, 1]
        - No additional constraints (unlike QROFN or other complex types)
        - Represents classical Zadeh fuzzy sets
    
    Notes:
        - This is the most efficient fuzzy type implementation
        - Suitable for basic fuzzy logic operations and applications
        - Forms the foundation for understanding more complex fuzzy types
        - Widely used in practical fuzzy systems
    
    Examples:
        .. code-block:: python
        
            # Create FS through strategy
            strategy = FSStrategy()
            strategy.md = 0.8
            print(strategy.report())  # Output: <0.8>
            
            # Validation in action
            strategy.md = -0.1  # Raises ValueError
    """
    
    # Type identifier for registration system
    mtype = 'fs'
    
    # Core fuzzy component with type annotation
    md: Optional[float] = None

    def __init__(self, q: Optional[int] = None):
        """
        Initialize FS strategy with validation framework.
        
        Parameters:
            q (Optional[int]): q-rung parameter (inherited from base class,
                             not used in basic FS but maintained for compatibility)
        
        Notes:
            The q parameter is inherited from the base FuzznumStrategy class
            for compatibility with the framework, but is not used in basic
            fuzzy sets. It's maintained to ensure consistent interface across
            all fuzzy types.
        """
        super().__init__(q=q)
        
        # Register membership degree validator
        # Ensures md is None or a numeric value in [0, 1]
        self.add_attribute_validator(
            'md', 
            lambda x: x is None or (
                isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1
            )
        )
        
        # Register change callback for membership degree
        # Triggers validation when md is modified
        self.add_change_callback('md', self._on_membership_change)

    def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """
        Callback triggered when membership degree changes.
        
        This method is called automatically whenever the membership degree
        is modified, allowing for reactive validation and constraint checking.
        
        Parameters:
            attr_name (str): Name of the changed attribute ('md')
            old_value (Any): Previous value of the attribute
            new_value (Any): New value being assigned
        
        Notes:
            For basic FS, this callback primarily serves as a hook for
            future extensions and maintains consistency with the framework's
            reactive programming model.
        """
        # For basic FS, no additional constraints beyond range validation
        # This callback is maintained for framework consistency and future extensions
        pass

    def _validate(self) -> None:
        """
        Perform comprehensive validation of the fuzzy set.
        
        This method is called to ensure the fuzzy set maintains mathematical
        validity. For basic FS, it primarily validates the membership degree
        constraints.
        
        Raises:
            ValueError: If membership degree violates constraints
        
        Notes:
            This method calls the parent validation and can be extended
            for additional constraint checking in derived classes.
        """
        super()._validate()
        # For basic FS, all validation is handled by attribute validators
        # No additional cross-attribute constraints needed

    def format_from_components(self, md: float, format_spec: str = "") -> str:
        """
        Format fuzzy set from membership degree component.
        
        This method provides flexible string representation of the fuzzy set
        based on the membership degree value and format specification.
        
        Parameters:
            md (float): Membership degree value to format
            format_spec (str): Format specification string
                - '' (default): Standard representation <md>
                - 'p': Parentheses format (md)
                - 'j': JSON format {"mtype": "fs", "md": md}
        
        Returns:
            str: Formatted string representation of the fuzzy set
        
        Examples:
            .. code-block:: python
            
                strategy = FSStrategy()
                strategy.md = 0.75
                
                print(strategy.format_from_components(0.75))      # <0.75>
                print(strategy.format_from_components(0.75, 'p')) # (0.75)
                print(strategy.format_from_components(0.75, 'j')) # {"mtype": "fs", "md": 0.75}
        """
        if md is None:
            return "<>"
        
        precision = get_config().DEFAULT_PRECISION
        
        # Parentheses format
        if format_spec == 'p':
            return f"({md})"
        
        # JSON format
        if format_spec == 'j':
            import json
            return json.dumps({'mtype': self.mtype, 'md': md})
        
        # Default format: <md>
        def strip_trailing_zeros(x: float) -> str:
            """Remove trailing zeros from float representation."""
            s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
            return s if s else "0"
        
        md_str = strip_trailing_zeros(md)
        return f"<{md_str}>"

    def report(self) -> str:
        """
        Generate a report string for the current fuzzy set.
        
        Returns:
            str: String representation of the fuzzy set in standard format
        
        Examples:
            .. code-block:: python
            
                strategy = FSStrategy()
                strategy.md = 0.8
                print(strategy.report())  # <0.8>
        """
        return self.format_from_components(self.md)

    def str(self) -> str:
        """
        Generate string representation of the fuzzy set.
        
        Returns:
            str: String representation of the fuzzy set
        
        Notes:
            This method provides the default string conversion for the fuzzy set,
            used by Python's str() function and string formatting operations.
        """
        return self.format_from_components(self.md)

    def __format__(self, format_spec: str) -> str:
        """
        Provide custom formatting support for the fuzzy set.
        
        This method enables the use of Python's format() function and f-string
        formatting with custom format specifications.
        
        Parameters:
            format_spec (str): Format specification string
        
        Returns:
            str: Formatted string representation
        
        Examples:
            .. code-block:: python
            
                strategy = FSStrategy()
                strategy.md = 0.75
                
                print(f"{strategy}")      # <0.75>
                print(f"{strategy:p}")    # (0.75)
                print(f"{strategy:j}")    # {"mtype": "fs", "md": 0.75}
                print(f"{strategy:.2f}")  # <0.75> (with 2 decimal places)
        """
        # Handle custom format specifications
        if format_spec and format_spec not in ['r', 'p', 'j']:
            # For non-custom format specs, apply to the string representation
            return format(self.str(), format_spec)
        
        # Use custom formatting for recognized specifications
        return self.format_from_components(self.md, format_spec)
