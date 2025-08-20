#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 22:45
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import re
from typing import List

from ....core import Fuzznum
from ....config import get_config

# 匹配数字：整数、小数、科学计数法
_NUMBER = r'-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

# 匹配形如 <[md1, md2, ...], [nmd1, nmd2, ...]> 的字符串
_PATTERN = re.compile(
    rf'^\s*<\s*\[\s*(?P<md>(?:{_NUMBER}(?:\s*,\s*{_NUMBER})*)?)\s*\]\s*,\s*'
    rf'\[\s*(?P<nmd>(?:{_NUMBER}(?:\s*,\s*{_NUMBER})*)?)\s*\]\s*>\s*$'
)


def _parse_number_list(content: str) -> List[float]:
    """解析逗号分隔的数字字符串为浮点数列表"""
    content = content.strip()
    if not content:
        return []

    parts = [p.strip() for p in content.split(',')]
    numbers = []
    for part in parts:
        try:
            numbers.append(float(part))
        except ValueError as e:
            raise ValueError(f"无法解析为数值: {part!r}") from e

    return numbers


def _validate_bounds(md_list: List[float], nmd_list: List[float]) -> None:
    """验证所有值都在[0,1]范围内"""
    for value in md_list:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"md 中存在越界值: {value} 不在 [0,1] 范围内")

    for value in nmd_list:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"nmd 中存在越界值: {value} 不在 [0,1] 范围内")


def _validate_q_constraint(md_list: List[float], nmd_list: List[float], q: int) -> None:
    """验证q-rung约束条件"""
    # 仅在两边都非空时检查约束
    if len(md_list) == 0 or len(nmd_list) == 0:
        return

    epsilon = get_config().DEFAULT_EPSILON
    max_md_power_q = max(md_list) ** q
    max_nmd_power_q = max(nmd_list) ** q
    sum_of_powers = max_md_power_q + max_nmd_power_q

    if sum_of_powers > 1.0 + epsilon:
        raise ValueError(
            f"违反 q-rung 约束: max(md)^q + max(nmd)^q = "
            f"{max_md_power_q:.6f} + {max_nmd_power_q:.6f} = {sum_of_powers:.6f} > 1.0 "
            f"(q={q}, md={md_list}, nmd={nmd_list})"
        )


def _qrohfn_from_str(fuzznum_str: str, q: int = 1) -> Fuzznum:
    """
    将字符串解析为 qrohfn Fuzznum。

    支持格式：
        <[a1, a2, ...], [b1, b2, ...]>

    Args:
        fuzznum_str (str): 要解析的字符串
        q (int): q-rung参数，默认为1

    Returns:
        Fuzznum: 解析后的qrohfn模糊数

    Raises:
        TypeError: 当 fuzznum_str 不是字符串时
        ValueError: 当 q 不是非负整数或字符串格式错误时
        ValueError: 当数值越界或违反约束条件时
    """
    if not isinstance(fuzznum_str, str):
        raise TypeError(f"fuzznum_str 必须为 str，得到: {type(fuzznum_str).__name__}")

    if not isinstance(q, int) or q < 0:
        raise ValueError(f"'q' 必须为非负整数，得到: {q!r}")

    # 匹配角括号格式
    match = _PATTERN.match(fuzznum_str)
    if not match:
        raise ValueError(f"格式错误: 无法解析字符串 {fuzznum_str!r}")

    # 解析数字列表
    md_list = _parse_number_list(match.group('md'))
    nmd_list = _parse_number_list(match.group('nmd'))

    # 验证边界条件
    _validate_bounds(md_list, nmd_list)

    # 验证q-rung约束
    _validate_q_constraint(md_list, nmd_list, q)

    # 创建并返回Fuzznum对象
    return Fuzznum('qrohfn', q=q).create(md=md_list, nmd=nmd_list)


# def _parse_json_like(text: str) -> Tuple[List[float], List[float], int | None] | None:
#     t = text.strip()
#     if not (t.startswith('{') and t.endswith('}')):
#         return None
#     try:
#         data = json.loads(t)
#     except Exception:
#         return None
#
#     if 'md' not in data or 'nmd' not in data:
#         raise ValueError("JSON 缺少必要键: 'md' 与 'nmd'")
#
#     if 'mtype' in data and data['mtype'] not in (None, 'qrohfn'):
#         raise ValueError(f"JSON mtype 不匹配: {data['mtype']!r}")
#
#     md = data['md']
#     nmd = data['nmd']
#     if not isinstance(md, list) or not isinstance(nmd, list):
#         raise TypeError("JSON 中 'md' 与 'nmd' 必须为列表")
#
#     try:
#         md_list = [float(x) for x in md]
#         nmd_list = [float(x) for x in nmd]
#     except Exception as e:
#         raise ValueError("JSON 中 'md' 或 'nmd' 包含非数值元素") from e
#
#     q_in = data.get('q', None)
#     if q_in is not None and not (isinstance(q_in, int) and q_in >= 0):
#         raise ValueError(f"JSON 中的 q 必须为非负整数，得到: {q_in!r}")
#
#     return md_list, nmd_list, q_in