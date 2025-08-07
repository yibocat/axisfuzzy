#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/31 23:37
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .qrofn import QROFNStrategy, QROFNTemplate

from .op import register_qrofn_operations

from . import _func

register_qrofn_operations()

__all__ = [
    'QROFNStrategy',
    'QROFNTemplate',
    'register_qrofn_operations'
]
