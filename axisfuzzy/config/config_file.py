#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 21:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    """
    Global configuration dataclass for AxisFuzzy.

    This class centralizes configurable behaviors and defaults used by the
    library (precision, default fuzzy number type, caching settings, etc.).
    Each attribute contains `metadata` used by the manager for categorization
    and validation.

    Attributes
    ----------
    DEFAULT_MTYPE : str
        Default fuzzy number type used when constructing fuzznum objects.
    DEFAULT_PRECISION : int
        Default number of decimal places for display/rounding.
    DEFAULT_EPSILON : float
        Numerical tolerance used for float comparisons.
    CACHE_SIZE : int
        Maximum number of entries in operation caches.
    TNORM_VERIFY : bool
        Whether to run additional verification checks for T-norms (debug).
    """
    # ================== Basic Configuration ===================
    DEFAULT_MTYPE: str = field(
        default='qrofn',
        metadata={
            'category': 'basic',
            'description': 'Default fuzzy number type, affects the type selection '
                           'when Fuzznum is constructed without parameters',
            'validator': lambda x: isinstance(x, str) and len(x) > 0,
            'error_msg': "Must be a non-empty string."
        }
    )

    DEFAULT_Q: int = field(
        default=1,
        metadata={
            'category': 'basic',
            'description': 'Default q-rung value for fuzzy numbers, '
                           'affects the number of fuzzy sets in q-rung orthopair fuzzy numbers',
            'validator': lambda x: isinstance(x, (int, np.integer)) and x > 0,
            'error_msg': "Must be a positive integer.",
            'note': 'For t-norm calculations that do not support q-rung, they are not affected.'
        }
    )

    DEFAULT_PRECISION: int = field(
        default=4,
        metadata={
            'category': 'basic',
            'description': 'Default calculation precision (number of decimal places), '
                           'affects all numeric calculations and display',
            'validator': lambda x: isinstance(x, int) and x >= 0,
            'error_msg': "Must be a non-negative integer."
        }
    )
    #
    DEFAULT_EPSILON: float = field(
        default=1e-12,
        metadata={
            'category': 'basic',
            'description': 'Default numerical tolerance, used for floating-point '
                           'number comparison and zero value judgment',
            'validator': lambda x: isinstance(x, (int, float)) and x > 0,
            'error_msg': "Must be a positive number."
        }
    )

    CACHE_SIZE: int = field(
        default=256,
        metadata={
            'category': 'performance',
            'description': 'Maximum number of entries in the operation cache, '
                           'which controls memory usage',
            'validator': lambda x: isinstance(x, int) and x >= 0,
            'error_msg': "Must be a non-negative integer."
        }
    )

    TNORM_VERIFY: bool = field(
        default=False,
        metadata={
            'category': 'debug',
            'description': 'T-Norm verification switch, used to verify the mathematical '
                           'properties after T-Norm initialization. It is generally set '
                           'to default off (False) to improve the computational efficiency '
                           'of fuzzy numbers.',
            'validator': lambda x: isinstance(x, bool),
            'error_msg': "Must be a boolean value (True/False)."
        }
    )