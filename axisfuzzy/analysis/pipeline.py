#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 15:59
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the Fluent API and DAG execution engine for fuzzy analysis workflows.

This module provides the core components for building and running declarative,
graph-based analysis pipelines:
- `FuzzyPipeline`: The main class for constructing a computational graph (DAG).
- `StepOutput`: A symbolic object representing the future output of a pipeline step.
- `tool`: A factory function to wrap registered analysis tools for use within a pipeline.
"""
import uuid
from collections import deque
from typing import List, Dict, Any, Tuple, Union, Optional

from .registry.base import tool_registry
from .contracts import validate as validate_contract


class StepOutput:
    """
    A symbolic representation of a future output from a pipeline step.

    This object acts as a placeholder or "promise" for a value that will be
    computed when the pipeline runs. It doesn't hold any real data itself but
    contains the necessary information to track dependencies within the DAG.

    Instances of this class are what enable the Fluent API, allowing users to
    chain operations together in a readable and type-safe manner.

    Attributes
    ----------
    step_id : str
        The unique identifier of the step that will produce this output.
    output_name : str
        The name of the specific output from the step (for tools with multiple returns).
    pipeline : 'FuzzyPipeline'
        A reference to the parent pipeline instance that owns this step.
    """
    def __init__(self, step_id: str, output_name: str, pipeline: 'FuzzyPipeline'):
        self.step_id = step_id
        self.output_name = output_name
        self.pipeline = pipeline

    def __repr__(self) -> str:
        step_tool = self.pipeline.get_step_info(self.step_id).get('tool', 'input')
        return f"<StepOutput of '{step_tool}' (step: {self.step_id[:8]}, output: {self.output_name})>"

    # 未来可以扩展，例如 .to_dict(), .visualize() 等方法
    def then(self, tool_name: str, **kwargs) -> 'StepOutput':
        """
        A convenient chaining method to apply the next tool.

        The ``then()`` method is syntactic sugar that makes more natural chained expressions like
        ``init_data.then("fuzzify")(...).then("weights")(...)`` possible.

        Syntactic sugar for `tool(tool_name)(pipeline=self.pipeline, ...)`

        Parameters
        ----------
        tool_name : str
            The name of the tool to apply.
        **kwargs
            Input arguments for the tool. The primary input is implicitly this
            StepOutput object.

        Returns
        -------
        StepOutput
            The symbolic output of the newly added step.
        """
        # 假设工具的主输入参数名为 'data' 或 'matrix'，这里需要一个约定
        # 简单起见，我们假设第一个参数是主输入
        tool_meta = tool_registry.get_metadata(tool_name)
        primary_input_name = list(tool_meta['inputs'].keys())[0]

        # 将 self 作为主输入，其他 kwargs 作为辅助输入
        all_inputs = {primary_input_name: self, **kwargs}
        return self.pipeline.tool(tool_name)(**all_inputs)


class FuzzyPipeline:
    """
    A builder for creating and executing a Directed Acyclic Graph (DAG) of
    fuzzy analysis operations.

    This class provides a Fluent API for defining complex, non-linear workflows.
    Users define the pipeline by linking `tool` calls together, which internally
    builds a computational graph. The graph is only executed when the `run`
    method is called.

    Parameters
    ----------
    steps : list[dict]
        A list of dictionaries, where each dictionary configures a single
        step of the pipeline. Each step must define a 'tool' name, and
        can optionally specify 'inputs', 'outputs', and 'params'.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.analysis import FuzzyPipeline

        # 1. Create a pipeline builder
        p = FuzzyPipeline()

        # 2. Define inputs and build the graph using the fluent API
        init_data = p.input("init_data")
        fuzz_table = p.tool("fuzzify_simple")(matrix=init_data)
        weights = p.tool("entropy_weight")(matrix=fuzz_table)

        # 3. The pipeline is now a defined graph, ready for execution
        #    It can be passed to an accessor to be run with real data.
        #    df.fuzzy.run(p)
    """
    def __init__(self):
        # _steps: 这是一个字典，存储了管道中所有步骤（节点）的定义。
        # 键是每个步骤的唯一ID (step_id)，值是一个字典，包含了该步骤的所有元数据和依赖信息。
        # 它是 DAG 在内存中的核心表示。
        # 示例: {'step_id_1': {'id': 'step_id_1', 'tool': 'fuzzify', 'inputs': {...}, ...}}
        self._steps: Dict[str, Dict[str, Any]] = {}

        # _input_nodes: 这是一个字典，存储了管道的输入节点信息。
        # 键是用户定义的输入名称 (e.g., 'init_data')，值是该输入节点在 _steps 中对应的 step_id。
        # 这些是 DAG 的起始点，其真实数据由 run 方法的 initial_data 参数提供。
        # 示例: {'init_data': 'input_id_xyz'}
        self._input_nodes: Dict[str, str] = {}

    def input(self,
              name: Optional[str] = None,
              contract: str = 'any') -> StepOutput:
        """
        Defines an input entry point for the pipeline, with an optional contract.

        This creates a "source" node in the DAG, which will be populated with
        real data when the pipeline's `run` method is called.

        Parameters
        ----------
        name : str, optional
            The name of the input handle. If omitted, a default name will be
            generated. This is recommended for single-input pipelines. (e.g., 'init_data').
        contract : str, default 'any'
            The data contract that the input data is expected to conform to.
            This allows for graph-time validation for the first step.
            If 'any', no contract validation is performed on this input's promise.

        Returns
        -------
        StepOutput
            A symbolic object representing this input.
        """
        if name is None:
            if self._input_nodes:
                raise TypeError(
                    "Input name can only be omitted for the first input of a pipeline. "
                    "Please provide explicit names for subsequent inputs."
                )

            # 为唯一的、匿名的输入生成一个默认名称
            name = "default_input"

        if name not in self._input_nodes:
            step_id = f"input_{name}_{uuid.uuid4().hex[:8]}"
            self._input_nodes[name] = step_id
            self._steps[step_id] = {
                "id": step_id,
                "tool": f"input_{name}",
                "outputs": {"output": contract}
            }

        step_id = self._input_nodes[name]
        return StepOutput(step_id=step_id, output_name="output", pipeline=self)

    def _add_step(self,
                  tool_name: str,
                  **kwargs) -> StepOutput | Dict[str, StepOutput]:
        """
        Internal method to add a new computational step (node) to the graph.

        This method is called by the `tool` factory. It performs graph-time
        validation of contracts.
        """
        step_id = f"{tool_name}_{uuid.uuid4().hex[:8]}"

        # --- Graph-Time Contract Validation ---
        tool_meta = tool_registry.get_metadata(tool_name)
        expected_inputs_contracts = tool_meta.get('inputs', {})

        # --- Separate data input and tool parameters ---
        data_inputs_from_kwargs: Dict[str, StepOutput] = {}     # For StepOutput inputs
        tool_parameters: Dict[str, Any] = {}                    # Parameter of stored literal type

        for arg_name, value in kwargs.items():
            if isinstance(value, StepOutput):
                data_inputs_from_kwargs[arg_name] = value
            else:
                tool_parameters[arg_name] = value

        # --- Graph Construction Time Contract Verification (Data Input Only) ---
        # Verify that all the declared data inputs have been provided.
        if set(expected_inputs_contracts.keys()) != set(data_inputs_from_kwargs.keys()):
            missing_inputs = set(expected_inputs_contracts.keys()) - set(data_inputs_from_kwargs.keys())
            extra_inputs = set(data_inputs_from_kwargs.keys()) - set(expected_inputs_contracts.keys())
            error_msg = f"Tool '{tool_name}' expects data inputs {list(expected_inputs_contracts.keys())}. "
            if missing_inputs:
                error_msg += f"Missing: {list(missing_inputs)}. "
            if extra_inputs:
                error_msg += f"Extra data inputs provided: {list(extra_inputs)}. "
            raise TypeError(error_msg.strip())

        resolved_dependencies = {}
        for arg_name, value_step_output in data_inputs_from_kwargs.items():
            source_step_id = value_step_output.step_id
            source_output_name = value_step_output.output_name

            source_step_meta = self.get_step_info(source_step_id)
            promised_contract = source_step_meta['outputs'].get(source_output_name)

            # 检查承诺的契约是否匹配期望的契约
            expected_contract = expected_inputs_contracts.get(arg_name)

            contracts_match = (
                promised_contract == 'any' or
                expected_contract == promised_contract
            )

            if expected_contract and promised_contract and not contracts_match:
                raise TypeError(
                    f"Contract mismatch for tool '{tool_name}' on input '{arg_name}'. "
                    f"Expected '{expected_contract}', but received a promise for "
                    f"'{promised_contract}' from step '{source_step_meta['tool']}'."
                )

            # Storing the symbol object itself as a dependency
            resolved_dependencies[arg_name] = value_step_output

        # Register the new step in the graph
        self._steps[step_id] = {
            "id": step_id,
            "tool": tool_name,
            "parameters": tool_parameters,                  # Static parameters of storage tools
            "inputs": data_inputs_from_kwargs,              # Store only data input (StepOutput)
            "outputs": tool_meta.get('outputs', {}),        # Output Contract of the Step
            "dependencies": resolved_dependencies           # Actual dependencies used to construct the DAG
        }

        # Create symbolic output(s) for this new step
        outputs_meta = tool_meta.get('outputs', {})
        if len(outputs_meta) == 1:
            output_name = list(outputs_meta.keys())[0]
            return StepOutput(step_id=step_id, output_name=output_name, pipeline=self)
        else:
            return {
                name: StepOutput(step_id=step_id, output_name=name, pipeline=self)
                for name in outputs_meta.keys()
            }

    def get_step_info(self, step_id: str) -> Dict[str, Any]:
        """Retrieves the metadata for a given step ID."""
        if step_id not in self._steps:
            raise ValueError(f"Step with ID {step_id} not found in this pipeline.")
        return self._steps[step_id]

    def tool(self, tool_name: str) -> callable:
        """
        Factory method to create a callable tool wrapper bound to this pipeline.
        This is the primary way to add steps to the pipeline.

        Parameters
        ----------
        tool_name : str
            The name of the tool as registered in the ``tool_registry``.

        Returns
        -------
        callable
            A function that, when called with the tool's arguments, adds a
            new step to the pipeline and returns its symbolic output(s).
        """
        def tool_wrapper(**kwargs):
            return self._add_step(tool_name, **kwargs)
        return tool_wrapper

    def _build_execution_order(self) -> List[str]:
        """
        Performs a topological sort of the graph to get a linear execution order.

        This method builds an adjacency list and an in-degree count for all
        steps, then uses Kahn's algorithm to sort them. It also detects
        cyclical dependencies.

        Returns
        -------
        list[str]
            A list of step IDs in a valid execution order.

        Raises
        ------
        ValueError
            If a cycle is detected in the pipeline graph.
        """
        adj: Dict[str, List[str]] = {step_id: [] for step_id in self._steps}
        in_degree: Dict[str, int] = {step_id: 0 for step_id in self._steps}

        # Build adjacency list and in-degree map
        for step_id, step_info in self._steps.items():
            if 'dependencies' in step_info:
                for dep in step_info['dependencies'].values():
                    source_id = dep.step_id
                    adj[source_id].append(step_id)
                    in_degree[step_id] += 1

        # Kahn's algorithm for topological sorting
        queue = deque([step_id for step_id, degree in in_degree.items() if degree == 0])
        execution_order = []

        while queue:
            current_id = queue.popleft()
            execution_order.append(current_id)

            for neighbor_id in adj[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # Check for cycles
        if len(execution_order) != len(self._steps):
            raise ValueError("A cycle was detected in the pipeline graph. Please check dependencies.")

        # Filter out the input nodes, as they don't need to be "executed"
        task_order = [
            step_id for step_id in execution_order
            if step_id not in self._input_nodes.values()
        ]
        return task_order

    @staticmethod
    def _resolve_step_inputs(step_info: Dict[str, Any],
                             state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves a step's inputs from symbolic form to real data from the state,
        and combines them with static parameters.
        This function also performs runtime contract validation on the resolved data.
        """
        resolved_inputs_and_params = {}
        tool_meta = tool_registry.get_metadata(step_info['tool'])

        # 1. Parsing data input (from upstream StepOutput)
        for arg_name, value_step_output in step_info['inputs'].items():
            source_step_id = value_step_output.step_id
            source_output_name = value_step_output.output_name

            source_result = state[source_step_id]

            if isinstance(source_result, dict):
                real_data = source_result.get(source_output_name)
            else:
                real_data = source_result

            resolved_inputs_and_params[arg_name] = real_data

        # 2. Parsing static parameters (from tool parameters)
        resolved_inputs_and_params.update(step_info['parameters'])

        # --- Runtime Contract Verification (Data Input Only) ---
        for arg_name, data in resolved_inputs_and_params.items():
            expected_contract = tool_meta['inputs'].get(arg_name)   # 只有 data_inputs 才会有 contract
            if expected_contract:                                   # 仅对声明了契约的参数进行验证
                if not validate_contract(expected_contract, data):
                    raise TypeError(
                        f"Runtime contract validation failed for tool '{step_info['tool']}' "
                        f"on input '{arg_name}'. Expected data conforming to '{expected_contract}', "
                        f"but received object of type {type(data).__name__}."
                    )

        return resolved_inputs_and_params

    def run(self,
            initial_data: Union[Dict[str, Any], Any],
            return_intermediate: bool = False
            ) -> Union[Dict[str, Any], Any, Tuple[Union[Dict[str, Any], Any], Dict[str, Any]]]:
        """
        Executes the defined pipeline DAG.

        This method first computes a valid execution order using topological
        sort, then iterates through the steps, resolving inputs, performing
        runtime validation, and executing the tool functions.

        Parameters
        ----------
        initial_data : dict[str, Any] or Any
            - For multi-input pipelines: A dictionary mapping input names
              (defined via `p.input()`) to the actual data objects.
            - For single-input pipelines: The single data object itself.
        return_intermediate : bool, default False
            If True, returns a tuple containing the final results of all
            terminal nodes and a dictionary of all intermediate step results.

        Returns
        -------
        Any or dict[str, Any] or tuple
            - If the pipeline has a single terminal node with a single output,
              returns the computed value directly.
            - Otherwise, returns a dictionary mapping terminal tool names to
              their computed results.
            - If `return_intermediate` is True, the first element of the
              returned tuple follows the logic above.
        """
        state: Dict[str, Any] = {}          # The state dictionary for execution
        num_inputs = len(self._input_nodes)

        if not isinstance(initial_data, dict):
            # The user passed in a single data object that is not a dictionary.
            if num_inputs == 1:
                # If there is only one input node, this is allowed.
                input_name = list(self._input_nodes.keys())[0]
                input_step_id = self._input_nodes[input_name]
                state[input_step_id] = initial_data
            elif num_inputs == 0:
                 raise ValueError("The pipeline has no defined inputs, but data was provided.")
            else:   # num_inputs > 1
                # If there are multiple input nodes, but only a single
                # non-dictionary data is provided, this is ambiguous.
                raise TypeError(
                    f"The pipeline has multiple inputs ({list(self._input_nodes.keys())}), "
                    f"so `initial_data` must be a dictionary mapping input names to data objects."
                )
        else:
            # The user passed in a dictionary, which is a standard case of
            # multiple inputs or a single input (in dictionary form).
            if set(initial_data.keys()) != set(self._input_nodes.keys()):
                raise KeyError(
                    f"The keys in the `initial_data` dictionary {list(initial_data.keys())} "
                    f"do not match the pipeline's defined input names {list(self._input_nodes.keys())}."
                )
            # 1. Populate state with initial data from the dictionary
            for input_name, data in initial_data.items():
                input_step_id = self._input_nodes[input_name]
                state[input_step_id] = data

        execution_order = self._build_execution_order()

        # 2. Execute each step in the topologically sorted order
        for step_id in execution_order:
            step_info = self.get_step_info(step_id)
            tool_name = step_info['tool']
            tool_fn = tool_registry.get(tool_name)
            resolved_inputs = self._resolve_step_inputs(step_info, state)
            # Execute the tool function
            result = tool_fn(**resolved_inputs)
            # Store the result in the state
            state[step_id] = result

        # 3.1. Identify terminal nodes
        all_step_ids = set(self._steps.keys())
        source_step_ids = set(
            dep.step_id for step in self._steps.values()
            if 'dependencies' in step for dep in step['dependencies'].values())
        terminal_step_ids = all_step_ids - source_step_ids

        # 3.2. Collect results from terminal nodes
        terminal_results = {}
        for step_id in terminal_step_ids:
            if step_id in state:
                tool_name = self._steps[step_id]['tool']
                terminal_results[tool_name] = state[step_id]

        # 3.3 Determine and return the final output(s)
        # The final outputs are the results from the terminal (leaf) nodes of the graph.
        final_output: Any
        if len(terminal_results) == 1:
            # If there's only one terminal node, check if its output is a single value
            single_result = list(terminal_results.values())[0]

            # Also check how many outputs this tool *declared*.
            # This handles the case where a tool returns a dict, but we only care if it *declared* one output.
            single_terminal_tool_name = list(terminal_results.keys())[0]
            tool_meta = tool_registry.get_metadata(single_terminal_tool_name)

            if len(tool_meta['outputs']) == 1:
                # Unambiguous single result: return the value directly
                final_output = single_result
            else:
                # Ambiguous: single terminal node, but it has multiple declared outputs.
                # Return the dictionary of results from this single node.
                final_output = terminal_results
        else:
            # Ambiguous: zero or multiple terminal nodes.
            # Return the dictionary of all terminal results.
            final_output = terminal_results

        if return_intermediate:
            return final_output, state
        else:
            return final_output

    @property
    def steps(self) -> List[Dict[str, Any]]:
        """Returns a list of all defined steps in the pipeline."""
        return list(self._steps.values())

    def __repr__(self) -> str:
        num_inputs = len(self._input_nodes)
        num_tasks = len(self._steps) - num_inputs
        return f"<FuzzyPipeline with {num_inputs} inputs and {num_tasks} tasks>"
