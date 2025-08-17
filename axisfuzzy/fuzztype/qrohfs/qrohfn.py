#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Optional

import numpy as np

from ...core import FuzznumStrategy


class QROHFNStrategy(FuzznumStrategy):
    mtype = 'qrohfn'
    md: Optional[np.ndarray] = np.array([])
    nmd: Optional[np.ndarray] = np.array([])

    def __init__(self, q: Optional[int] = None):
        super().__init__(q=q)


