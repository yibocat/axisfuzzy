#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/25 14:34
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Provides a high-level, user-friendly namespace for analysis components.

This module acts as a facade, providing simple aliases for the core
`AnalysisComponent` classes located in `axisfuzzy.analysis.component`.
This allows users to build `Sequential` models with a clean and intuitive
syntax, similar to popular deep learning frameworks.

The components are not re-implemented here; they are simply aliased for
a better developer experience.
"""

from ..component.basic import (
    ToolNormalization as normalization,
    ToolWeightNormalization as weightnormalization,
    ToolStatistics as statistics,
    ToolFuzzification as fuzzification,
    ToolSimpleAggregation as simpleaggregation
)

# --- Future components can be aliased here ---
# For example, when you implement TOPSIS or AHP components:
#
# from ..component.topsis import ToolTOPSIS as TOPSIS
# from ..component.ahp import ToolAHP as AHP

# You can use __all__ to define the public API of this module
__all__ = [
    'normalization',
    'weightnormalization',
    'statistics',
    'fuzzification',
    'simpleaggregation'
]
