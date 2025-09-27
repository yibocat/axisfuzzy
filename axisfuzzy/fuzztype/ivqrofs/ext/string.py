#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
IVQROFN String Conversion Extension Methods.

This module implements string parsing and conversion functions for 
Interval-Valued Q-Rung Orthopair Fuzzy Numbers, supporting the standard
IVQROFN string format: <[md_lower,md_upper],[nmd_lower,nmd_upper]>
"""

import re
from typing import List

from ....core import Fuzznum

# Regular expression patterns for IVQROFN string parsing
# Matches floating point numbers including scientific notation
_NUMBER = r'-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

# Main pattern for IVQROFN format: <[a,b],[c,d]>
# Supports whitespace and various bracket configurations
_IVQROFN_PATTERN = re.compile(
    rf'^\s*<\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*,\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*>\s*$'
)

# Alternative pattern for parentheses format: ([a,b], [c,d])
_IVQROFN_PAREN_PATTERN = re.compile(
    rf'^\s*\(\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*,\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*\)\s*$'
)


def _ivqrofn_from_str(fuzznum_str: str, q: int = 1) -> Fuzznum:
    """
    Convert string representation to IVQROFN Fuzznum.
    
    Supports multiple string formats:
    - Standard format: <[md_lower,md_upper],[nmd_lower,nmd_upper]>
    - Parentheses format: ([md_lower,md_upper], [nmd_lower,nmd_upper])
    
    Parameters:
        fuzznum_str: String representation of IVQROFN
        q: Q-rung parameter for constraint validation
        
    Returns:
        Fuzznum: IVQROFN object created from string
        
    Raises:
        TypeError: If input is not a string
        ValueError: If q is invalid, string format is incorrect, 
                   values are out of bounds, or constraints are violated
        
    Examples:
        >>> _ivqrofn_from_str("<[0.6,0.8],[0.1,0.2]>", q=2)
        <[0.6,0.8],[0.1,0.2]>
        
        >>> _ivqrofn_from_str("([0.5, 0.7], [0.2, 0.3])", q=3)
        <[0.5,0.7],[0.2,0.3]>
    """
    # Input validation
    if not isinstance(fuzznum_str, str):
        raise TypeError(f"fuzznum_str must be a str. got '{type(fuzznum_str).__name__}'")
    if not isinstance(q, int) or q < 1:
        raise ValueError(f"'q' must be a positive integer. got '{q!r}'")

    # Try standard format first
    match = _IVQROFN_PATTERN.match(fuzznum_str)
    if not match:
        # Try parentheses format
        match = _IVQROFN_PAREN_PATTERN.match(fuzznum_str)
    
    if not match:
        raise ValueError(
            f"Format error: Cannot parse string: {fuzznum_str!r}. "
            f"Expected format: <[md_lower,md_upper],[nmd_lower,nmd_upper]> "
            f"or ([md_lower,md_upper], [nmd_lower,nmd_upper])"
        )

    # Extract and convert numeric values
    try:
        md_lower = float(match.group(1))
        md_upper = float(match.group(2))
        nmd_lower = float(match.group(3))
        nmd_upper = float(match.group(4))
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid numeric values in string: {fuzznum_str!r}") from e

    # Validate interval bounds
    if not (0.0 <= md_lower <= 1.0 and 0.0 <= md_upper <= 1.0):
        raise ValueError(
            f"Membership interval values out of bounds: [{md_lower}, {md_upper}] "
            f"must be within [0,1]"
        )
    
    if not (0.0 <= nmd_lower <= 1.0 and 0.0 <= nmd_upper <= 1.0):
        raise ValueError(
            f"Non-membership interval values out of bounds: [{nmd_lower}, {nmd_upper}] "
            f"must be within [0,1]"
        )

    # Validate interval ordering
    if md_lower > md_upper:
        raise ValueError(
            f"Invalid membership interval ordering: {md_lower} > {md_upper}. "
            f"Lower bound must be ≤ upper bound."
        )
    
    if nmd_lower > nmd_upper:
        raise ValueError(
            f"Invalid non-membership interval ordering: {nmd_lower} > {nmd_upper}. "
            f"Lower bound must be ≤ upper bound."
        )

    # Validate q-rung constraint using upper bounds
    constraint_value = md_upper ** q + nmd_upper ** q
    if constraint_value > 1.0 + 1e-12:  # Small tolerance for floating point errors
        raise ValueError(
            f"Violation of q-rung constraint: "
            f"md_upper^q + nmd_upper^q = {md_upper}^{q} + {nmd_upper}^{q} = {constraint_value:.6f} > 1 "
            f"(q={q})"
        )

    # Create and return IVQROFN
    md_interval = [md_lower, md_upper]
    nmd_interval = [nmd_lower, nmd_upper]
    
    return Fuzznum('ivqrofn', q=q).create(md=md_interval, nmd=nmd_interval)


def _parse_interval_sequence(interval_str: str) -> List[List[float]]:
    """
    Parse a sequence of intervals from string format.
    
    Helper function for parsing multiple intervals in batch operations.
    Supports formats like: "[0.1,0.2] [0.3,0.4] [0.5,0.6]"
    
    Parameters:
        interval_str: String containing space-separated intervals
        
    Returns:
        List[List[float]]: List of interval pairs
        
    Raises:
        ValueError: If interval format is invalid
    """
    # Pattern for individual interval: [a,b]
    interval_pattern = re.compile(rf'\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]')
    
    matches = interval_pattern.findall(interval_str)
    if not matches:
        raise ValueError(f"No valid intervals found in: {interval_str}")
    
    intervals = []
    for match in matches:
        try:
            lower = float(match[0])
            upper = float(match[1])
            if lower > upper:
                raise ValueError(f"Invalid interval ordering: {lower} > {upper}")
            intervals.append([lower, upper])
        except ValueError as e:
            raise ValueError(f"Invalid interval values: {match}") from e
    
    return intervals


def _validate_ivqrofn_string_format(fuzznum_str: str) -> bool:
    """
    Validate if string matches IVQROFN format without full parsing.
    
    Quick validation function for format checking without creating objects.
    Useful for batch validation or preprocessing.
    
    Parameters:
        fuzznum_str: String to validate
        
    Returns:
        bool: True if string matches IVQROFN format, False otherwise
    """
    if not isinstance(fuzznum_str, str):
        return False
    
    # Check against both supported patterns
    return (_IVQROFN_PATTERN.match(fuzznum_str) is not None or
            _IVQROFN_PAREN_PATTERN.match(fuzznum_str) is not None)


def _extract_ivqrofn_components(fuzznum_str: str) -> dict:
    """
    Extract numeric components from IVQROFN string without validation.
    
    Utility function to extract raw numeric values for analysis or conversion.
    Does not perform constraint validation.
    
    Parameters:
        fuzznum_str: IVQROFN string representation
        
    Returns:
        dict: Dictionary with 'md_lower', 'md_upper', 'nmd_lower', 'nmd_upper' keys
        
    Raises:
        ValueError: If string format is invalid
    """
    match = _IVQROFN_PATTERN.match(fuzznum_str)
    if not match:
        match = _IVQROFN_PAREN_PATTERN.match(fuzznum_str)
    
    if not match:
        raise ValueError(f"Invalid IVQROFN string format: {fuzznum_str}")
    
    return {
        'md_lower': float(match.group(1)),
        'md_upper': float(match.group(2)),
        'nmd_lower': float(match.group(3)),
        'nmd_upper': float(match.group(4))
    }