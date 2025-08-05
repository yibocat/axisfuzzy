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
from .modules import *
from .config import *

__all__ = []

__all__.extend(core.__all__)
__all__.extend(modules.__all__)
__all__.extend(config.__all__)
__all__.extend(mixin.__all__)
