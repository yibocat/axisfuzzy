#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 14:32
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from functools import partial
from .base import register_tool

# Create a specialized decorator for weighting tools.
# It's just a pre-filled version of the general `register_tool`.
register_weighting = partial(register_tool, category='weighting')

# Set a nice docstring for our specialized decorator
register_weighting.__doc__ = """
A decorator to register a weighting tool.

This is a specialized version of `@register_tool` with the `category`
parameter automatically set to 'weighting'.

Parameters
----------
name : str
    The unique name to identify the tool.
inputs : dict[str, str]
    A mapping of parameter names to their required contract names.
outputs : dict[str, str]
    A mapping of output keys to their data contract names.

Examples
--------
.. code-block:: python

    from axisfuzzy.analysis.registry.weighting import register_weighting

    @register_weighting(
        name="entropy",
        inputs={'matrix': 'FuzzyTable'},
        outputs={'weights': 'WeightVector'}
    )
    def entropy_weight_calculator(matrix):
        # ... implementation ...
        pass
"""
