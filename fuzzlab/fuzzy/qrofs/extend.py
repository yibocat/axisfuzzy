#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/6 20:46
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import random

from ...core import Fuzznum
from ...extend import get_extend_registry

registry = get_extend_registry()


# 测试 distance
@registry.register(
    name='distance',
    mtype='qrofn',
    target_classes=['Fuzznum', 'Fuzzarray'],
    injection_type='both'
)
def qrofn_distance(fuzz1: Fuzznum, fuzz2: Fuzznum, p: int = 2) -> float:
    q = fuzz1.q
    md_diff = abs(fuzz1.md ** q - fuzz2.md ** q) ** p
    nmd_diff = abs(fuzz1.nmd ** q - fuzz2.nmd ** q) ** p
    return ((md_diff + nmd_diff) / 2) ** (1 / p)


# @registry.register(
#     name='random',
#     mtype='qrofn',
#     injection_type='top_level_function'
# )
# def create_random_qrofn(q: int = 1, **kwargs) -> Fuzznum:
#     md = random.random()
#     nmd = random.uniform(0, (1 - md ** q) ** (1 / q))
#     return Fuzznum(mtype='qrofn', qrung=q).create(md=md, nmd=nmd)


# TODO: 默认距离,暂且放在这里,后续需要在 `extend` 模块单独完成
# @registry.register(
#     name='distance',
#     is_default=True,
#     target_classes=['Fuzznum', 'Fuzzarray'],
#     injection_type='both'
# )
# def default_distance(fuzz1: Fuzznum, fuzz2: Fuzznum, **kwargs) -> float:
#     """默认距离实现，当没有为特定 mtype 找到实现时调用。"""
#     raise NotImplementedError(f"Distance function is not implemented for mtype '{fuzz1.mtype}'.")