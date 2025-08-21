#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
AxisFuzzy Membership Function Module.

This module provides the complete membership function system for AxisFuzzy,
including base classes, factory functions, and standard implementations.
Membership functions are the mathematical foundation of fuzzy logic, mapping
crisp input values to membership degrees in the range [0, 1].

The module offers:
- Abstract base class for creating custom membership functions
- Factory system for dynamic function creation by name
- Comprehensive library of standard membership function implementations
- Integration with the broader AxisFuzzy fuzzification system

Key Components
--------------
- :class:`MembershipFunction` : Abstract base class for all membership functions
- :func:`create_mf` : Factory function for creating instances by name
- :func:`get_mf_class` : Function to get class references by name
- Standard implementations : TriangularMF, GaussianMF, SigmoidMF, etc.

Quick Start
-----------
.. code-block:: python

    from axisfuzzy.membership import create_mf, TriangularMF

    # Create via factory
    mf, _ = create_mf('trimf', a=0, b=0.5, c=1)

    # Create directly
    mf = TriangularMF(0, 0.5, 1)

    # Evaluate membership
    result = mf.compute(0.3)  # Returns ~0.6

See Also
--------
axisfuzzy.fuzzify : Fuzzification system using membership functions
axisfuzzy.membership.base : Base class documentation
axisfuzzy.membership.factory : Factory system documentation
axisfuzzy.membership.function : Standard function implementations
"""

from .base import MembershipFunction
from .factory import create_mf, get_mf_class
from . import function

# Facilitates direct import of specific function classes from axisfuzzy.membership for users.
from .function import (
    SigmoidMF,
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    SMF,
    ZMF,
    DoubleGaussianMF,
    GeneralizedBellMF,
    PiMF
)

__all__ = [
    'MembershipFunction',
    'create_mf',
    'get_mf_class',
    'function',
    'SigmoidMF',
    'TriangularMF',
    'TrapezoidalMF',
    'GaussianMF',
    'SMF',
    'ZMF',
    'DoubleGaussianMF',
    'GeneralizedBellMF',
    'PiMF'
]
