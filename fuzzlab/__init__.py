#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 21:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
FuzzLab: A Python library for fuzzy number and fuzzy array computations.
"""

__all__ = []

from .config import *
from .core import *
from .extension import (
    apply_extensions,
    get_extension_registry,
    get_extension_dispatcher,
    get_extension_injector,
    batch_extension
)
from .fuzztype import *
from .mixin import *
from .membership import *

from . import random
from . import fuzzify

__all__.extend(config.__all__)
__all__.extend(core.__all__)
__all__.extend(extension.__all__)
__all__.extend(fuzztype.__all__)
__all__.extend(mixin.__all__)
__all__.extend(membership.__all__)

apply_extensions()
