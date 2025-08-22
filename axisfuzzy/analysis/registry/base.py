#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 14:24
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Core registry for managing and discovering analysis tools.

This module provides the central `ToolRegistry` and the primary decorator,
`@register_tool`, for adding new analysis functions to the system. The
registry stores not only the tool function but also its metadata, including
input/output contracts and category, which is crucial for the pipeline engine.
"""
import inspect
import warnings
from typing import Dict, Any, Callable, List, Optional, Union


class ToolRegistry:
    """
    A singleton registry for all analysis tools available in AxisFuzzy.

    This class provides a central point for registering, retrieving, and
    querying analysis tools. Each registered tool includes its callable
    function along with essential metadata like its input/output data
    contracts and its functional category.

    It is not intended to be instantiated directly. Use the global instance
    `tool_registry` and the `@register_tool` decorator.
    """
    _instance: Optional['ToolRegistry'] = None
    _initialized: bool = False

    def __new__(cls) -> 'ToolRegistry':
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

    def register(self,
                 name: str,
                 *,                         # Forces subsequent arguments to be keyword-only
                 fn: Callable,
                 inputs: Dict[str, str],
                 outputs: Dict[str, str],
                 category: str | None = None):
        """
        Registers a tool with its metadata. Recommended to use @register_tool instead.

        Parameters
        ----------
        name : str
            The unique name of the tool.
        fn : Callable
            The function that implements the tool's logic.
        inputs : dict[str, str]
            A mapping of the tool's input parameter names to their
            required contract names.
        outputs : dict[str, str]
            A mapping of the tool's output names to their resulting
            contract names.
        category : str, optional
            A category for grouping the tool (e.g., 'weighting', 'decision').
        """
        if name in self._tools:
            # ADDED: Add a warning or raise an error for duplicate registration
            # This helps in debugging if two tools accidentally get the same name.
            warnings.warn(f"Overwriting existing tool registration for '{name}'.")

        self._tools[name] = {
            "fn": fn,
            "inputs": inputs,
            "outputs": outputs,
            "category": category
        }

    def get(self, name: str) -> Callable:
        """
        Retrieve the callable function for a registered tool.

        Parameters
        ----------
        name : str
            The name of the tool to retrieve.

        Returns
        -------
        Callable
            The tool's implementation function.

        Raises
        ------
        KeyError
            If no tool with the given name is registered.
        """
        # MODIFIED: More informative error message
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                f"Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]["fn"]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Retrieve the full metadata for a registered tool.

        Parameters
        ----------
        name : str
            The name of the tool.

        Returns
        -------
        dict
            A dictionary containing the tool's function, inputs, outputs,
            and category.

        Raises
        ------
        KeyError
            If no tool with the given name is registered.
        """
        # MODIFIED: More informative error message
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                f"Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List the names of all registered tools, optionally filtered by category.

        Parameters
        ----------
        category : str, optional
            If provided, only tools in this category will be returned.

        Returns
        -------
        list[str]
            A list of tool names.
        """
        if category is None:
            return list(self._tools.keys())
        return [
            name for name, meta in self._tools.items()
            if meta["category"] == category
        ]


# Global instance of the registry
tool_registry = ToolRegistry()


def register_tool(
        func: Optional[Callable] = None, *,   # func can be the decorated function itself OR None
        name: Optional[str] = None,           # Explicit name override
        inputs: Union[str, Dict[str, str]],
        outputs: Union[str, Dict[str, str]],
        category: Optional[str] = 'general') -> Callable:
    """
    A decorator to register an analysis tool function with an intelligent signature.

    This decorator simplifies tool registration by automatically handling
    single/multiple inputs and outputs.

    Parameters
    ----------
    func : Callable, optional
        The function to be registered. Populated automatically by the decorator.
    name : str, optional
        A custom name for the tool. Defaults to the function's name.
    inputs : str or dict[str, str]
        - For a single data input, provide the contract name as a string
          (e.g., `input='CrispTable'`). The decorator will automatically map
          it to the first parameter of the decorated function.
        - For multiple data inputs, provide a dictionary mapping parameter
          names to contract names (e.g., `inputs={'matrix': 'FuzzyTable', 'weights': 'WeightVector'}`).
    outputs : str or dict[str, str]
        - For a single output, provide the contract name as a string
          (e.g., `output='FuzzyTable'`). The tool should return a single value.
          The output name will default to 'result'.
        - For multiple outputs, provide a dictionary mapping output names to
          contract names. The tool must return a dictionary with matching keys.
    category : str, optional
        An optional category for classifying the tool. Defaults to 'general'.

    Returns
    -------
    Callable
        The decorated function itself.

    Examples
    --------
    # Single input/output (simple)
    @register_tool(inputs='CrispTable', outputs='FuzzyTable')
    def fuzzify(data, mtype, mf): # 'data' is automatically mapped to 'CrispTable'
        ...
        return fuzzy_dataframe

    # Multiple inputs/outputs (advanced)
    @register_tool(
        inputs={'matrix': 'FuzzyTable', 'weights': 'WeightVector'},
        outputs={'scores': 'ScoreVector', 'ranking': 'RankingResult'}
    )
    def my_decision_tool(matrix, weights, some_param):
        ...
        return {'scores': ..., 'ranking': ...}
    """
    def decorator(f: Callable) -> Callable:
        final_tool_name = name if name is not None else f.__name__

        # 1. Handle inputs
        final_inputs_dict: Dict[str, str]
        if isinstance(inputs, str):
            # If it is a string (single input),
            # it is automatically mapped to the first parameter of the function.
            sig = inspect.signature(f)
            params = list(sig.parameters.keys())
            if not params:
                raise TypeError(f"Tool function '{f.__name__}' has no parameters "
                                f"to map the single input contract to.")
            first_param_name = params[0]
            final_inputs_dict = {first_param_name: inputs}
        elif isinstance(inputs, dict):
            # If it is a dictionary, use it directly
            final_inputs_dict = inputs
        else:
            raise TypeError("`inputs` must be a string or a dictionary.")

        # 2. Handle outputs
        final_outputs_dict: Dict[str, str]
        if isinstance(outputs, str):
            # If it is a string (single output),
            # the output name is agreed to be 'outputs'
            final_outputs_dict = {"outputs": outputs}
        elif isinstance(outputs, dict):
            # If it is a dictionary, use it directly
            final_outputs_dict = outputs
        else:
            raise TypeError("`outputs` must be a string or a dictionary.")

        tool_registry.register(
            name=final_tool_name,
            fn=f,
            inputs=final_inputs_dict, # <-- 传入处理后的字典
            outputs=final_outputs_dict, # <-- 传入处理后的字典
            category=category
        )
        return f

    if func is None:
        # Decorator was called with arguments, e.g., @register_tool(name="foo", ...)
        return decorator
    else:
        # Decorator was called without arguments, e.g., @register_tool
        return decorator(func)
