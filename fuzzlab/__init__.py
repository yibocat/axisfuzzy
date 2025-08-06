#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 21:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
FuzzLab: A Python library for fuzzy number and fuzzy array computations.
"""

from .core import *
from .mixin import *
from .fuzzy import *
from .config import *
from .extend import apply_extensions

__all__ = []

__all__.extend(core.__all__)
__all__.extend(fuzzy.__all__)
__all__.extend(config.__all__)
__all__.extend(mixin.__all__)
__all__.extend(extend.__all__)

apply_extensions()
