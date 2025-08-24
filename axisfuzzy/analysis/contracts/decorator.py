#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 18:12
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the v2 @contract decorator, which infers contracts from type hints.
"""

import inspect
from typing import get_type_hints, Dict, Any, get_origin, get_args, Union

from .base import Contract


def contract(func: callable) -> callable:
    """
    A parameter-less decorator that infers data contracts from type annotations.

    This decorator inspects the decorated function's signature, identifies
    parameters and return values annotated with ``Contract`` objects, and
    attaches the extracted contract metadata to the function for the pipeline
    engine to use.

    Returns
    -------
    Callable
        The original function, now with `_contract_inputs` and
        `_contract_outputs` attributes.

    Raises
    ------
    TypeError
        If type hints cannot be resolved or if the return annotation is
        in an unsupported format.

    Examples
    --------
    .. code-block:: python

        from .contract import ContractCrispTable, ContractWeightVector, ContractScoreVector
        from typing import Dict

        @contract
        def process_data(data: ContractCrispTable, weights: ContractWeightVector) -> ContractScoreVector:
            # ...
            pass

        @contract
        def multi_output_tool(data: ContractCrispTable) -> Dict[str, ContractScoreVector]:
            # ...
            pass
    """
    # --- 1 .Get the type annotation of the function ---
    #    `get_type_hints` will parse annotations in string form
    try:
        # Use inspect to get 'hints', which is robust for forward references
        hints = get_type_hints(func, include_extras=True)
    except (NameError, TypeError) as e:
        raise TypeError(
            f"Cannot resolve type hints for '{func.__qualname__}'. "
            f"Ensure all contract types are correctly imported. Original error: {e}"
        ) from e

    input_contracts: Dict[str, Contract] = {}
    output_contracts: Dict[str, Contract] = {}

    # --- 2. Parse Input Contracts ---
    for param_name, param_type in hints.items():
        if param_name == 'return':
            continue
        # We only care about parameters annotated as Contract objects.
        if isinstance(param_type, Contract):
            input_contracts[param_name] = param_type

    # --- 3. Parse Output Contracts ---
    if 'return' in hints:
        return_type = hints['return']
        if isinstance(return_type, Contract):
            # Single output: by convention, name it 'output'
            output_contracts['output'] = return_type
        elif get_origin(return_type) is dict:
            # Multi-output: e.g., Dict[str, ContractScoreVector]
            # We assume the value type is the contract.
            key_type, value_type = get_args(return_type)
            if key_type is str and isinstance(value_type, Contract):
                # Convention: use the contract name in lowercase as the key
                output_contracts[value_type.name.lower()] = value_type
            elif key_type is str and get_origin(value_type) is Union:
                # Handle Dict[str, Union[ContractA, ContractB]]
                for t in get_args(value_type):
                    if isinstance(t, Contract):
                        output_contracts[t.name.lower()] = t
            else:
                raise TypeError(
                    f"Unsupported multi-output annotation for '{func.__qualname__}': "
                    f"Expected Dict[str, Contract] or Dict[str, Union[Contract,...]], "
                    f"but got {return_type}."
                )

    # 4. Attach metadata for the pipeline engine
    # We store the names, as the pipeline will use Contract.get(name)
    setattr(func, '_contract_inputs', {k: v.name for k, v in input_contracts.items()})
    setattr(func, '_contract_outputs', {k: v.name for k, v in output_contracts.items()})
    setattr(func, '_is_contract_method', True)

    # 可以在这里添加一个包装器，在函数调用时进行即时验证（可选）
    # def wrapper(*args, **kwargs):
    #     # ... pre-execution validation ...
    #     result = func(*args, **kwargs)
    #     # ... post-execution validation ...
    #     return result
    # return wrapper

    return func
