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

from .model import Model

__all__ = ["Model"]

