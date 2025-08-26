#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/25 11:58
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
High-level application API for AxisFuzzy analysis workflows.

This package provides simplified, user-friendly interfaces for building
and executing fuzzy data analysis pipelines. It serves as an application
layer over the more flexible but complex core pipeline system.

The main components are:
- Sequential: A linear workflow builder for simple analysis chains
"""

from .sequential import Sequential
from . import layers

__all__ = ["Sequential", "layers"]


# TODO: 三个主要问题:
#  1. 目前仅支持 单输入 -> 单输出,我们还要让其能够'单输入 -> 多输出','多输入 -> 单输出','多输入 -> 多输出'
#  2. 目前仅支持 线性流程, 通过多输出和多初入, 构建非线性计算分析流程
#  3. 需要改进支持管道嵌套功能
#  4. 需要支持更多的模糊数据类型, 关键在于模糊化. 现在有一个 bug 是我们设置模糊化只能为 qrofn. qrohfn 会报错.


