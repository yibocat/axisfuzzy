#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 21:56
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .generator import QROFNRandomGenerator, qrofn_random_generator
from ....random.core import register_random_generator

__all__ = [
    'QROFNRandomGenerator',
    'qrofn_random_generator',
]

register_random_generator('qrofn', qrofn_random_generator)
