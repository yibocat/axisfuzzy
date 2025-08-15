#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 16:52
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

# Re-export the base class for type-hinting and subclassing.
from .base import MembershipFunction as MembershipFunction

# Re-export the factory functions. Their signatures are defined in factory.pyi.
from .factory import create_mf as create_mf
from .factory import get_mf_class as get_mf_class

# Re-export the entire 'function' module, as it's in __all__.
from . import function as function

# Re-export all concrete membership function classes for direct access.
from .function import (
    SigmoidMF as SigmoidMF,
    TriangularMF as TriangularMF,
    TrapezoidalMF as TrapezoidalMF,
    GaussianMF as GaussianMF,
    SMF as SMF,
    ZMF as ZMF,
    DoubleGaussianMF as DoubleGaussianMF,
    GeneralizedBellMF as GeneralizedBellMF,
    PiMF as PiMF
)

# 显式声明包公共 API（便于类型检查器）
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
