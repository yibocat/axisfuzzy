#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) String Conversion Extension Methods.

This module implements string parsing functionality for classical fuzzy sets (FS),
enabling conversion from string representations to Fuzznum objects. The parsing
follows the standard FS format: <md> where md is the membership degree.

Mathematical Foundation:
    Classical fuzzy sets are represented as <md> where:
    - md ∈ [0, 1] is the membership degree
    - No additional constraints (unlike QROFN with q-rung constraints)
"""

import re

from ....core import Fuzznum

# Regular expression pattern for matching FS string format
# Matches numbers including integers, decimals, and scientific notation
_NUMBER = r'-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

# Pattern for FS format: <md> with optional whitespace
_FS_PATTERN = re.compile(
    rf'^\s*<\s*({_NUMBER})\s*>\s*$'
)


def _fs_from_str(fuzznum_str: str) -> Fuzznum:
    """
    Convert string representation to FS Fuzznum.
    
    Parses strings in the format '<md>' where md is a membership degree
    in the range [0, 1]. The function handles various numeric formats
    including integers, decimals, and scientific notation.
    
    Parameters:
        fuzznum_str (str): String representation of FS in format '<md>'
        
    Returns:
        Fuzznum: FS Fuzznum object created from parsed string
        
    Raises:
        TypeError: If input is not a string
        ValueError: If string format is invalid or values are out of bounds
        
    Examples:
        >>> _fs_from_str('<0.8>')  # Basic format
        <0.8>
        >>> _fs_from_str('< 0.75 >')  # With whitespace
        <0.75>
        >>> _fs_from_str('<1.0e-1>')  # Scientific notation
        <0.1>
        
    Notes:
        - Membership degrees must be in [0, 1]
        - Whitespace around values is automatically stripped
        - Scientific notation is supported for numeric values
    """
    # Input validation
    if not isinstance(fuzznum_str, str):
        raise TypeError(f"fuzznum_str must be a str. got '{type(fuzznum_str).__name__}'")

    # Attempt to match the FS pattern
    match = _FS_PATTERN.match(fuzznum_str)
    if not match:
        raise ValueError(f"Format error: "
                         f"Cannot parse FS string: {fuzznum_str!r}. "
                         f"Expected format: '<md>' where md is in [0,1]")

    # Extract and convert membership degree
    try:
        md = float(match.group(1))
    except ValueError as e:
        raise ValueError(f"Invalid numeric value in FS string: {fuzznum_str!r}. {e}")

    # Validate FS constraints: membership degree must be in [0, 1]
    if not (0.0 <= md <= 1.0):
        raise ValueError(f"Value out of bounds: md={md} must be within [0,1]")

    # Create and return FS Fuzznum
    return Fuzznum('fs').create(md=md)