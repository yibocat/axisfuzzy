#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 16:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the @contract decorator for attaching metadata to component methods.
"""
from typing import Dict, Callable


def contract(inputs: Dict[str, str], outputs: Dict[str, str]) -> Callable:
    """
    A decorator to attach data contract metadata to a class method.

    This decorator is the cornerstone of the component-based tool system. It
    allows developers to declare the input and output contracts directly on the
    methods of their analysis component classes. The FuzzyPipeline engine then
    reads this metadata via reflection to perform graph-time and runtime
    validation.

    Parameters
    ----------
    inputs : dict[str, str]
        A mapping of the method's parameter names to their required contract names.
    outputs : dict[str, str]
        A mapping of the method's output names (if returning a dict) or a single
        output name to its resulting contract name.

    Returns
    -------
    Callable
        A decorator that attaches the contract metadata to the decorated method.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.analysis._components.base import AnalysisComponent
        from axisfuzzy.analysis.contracts import contract

        class MyTool(AnalysisComponent):
            @contract(
                inputs={'matrix': 'FuzzyTable', 'weights': 'WeightVector'},
                outputs={'scores': 'ScoreVector'}
            )
            def calculate_scores(self, matrix, weights):
                # ... implementation ...
                return {'scores': some_scores}
    """

    def decorator(method: Callable) -> Callable:
        # We simply attach the metadata directly to the function object.
        # This is a standard and robust way to add metadata in Python.
        setattr(method, '_contract_inputs', inputs)
        setattr(method, '_contract_outputs', outputs)
        # We also add a marker to easily identify methods with contracts.
        setattr(method, '_is_contract_method', True)
        return method

    return decorator
