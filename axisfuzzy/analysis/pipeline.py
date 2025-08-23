#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 15:59
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the component-based Fluent API and DAG execution engine for fuzzy
analysis workflows.

This module provides the core components for building and running declarative,
graph-based analysis pipelines using Analysis Components.
- ``FuzzyPipeline``: The main class for constructing a computational graph (DAG).
- ``StepOutput``: A symbolic object representing the future output of a pipeline step.
"""
import uuid
from collections import deque
from typing import List, Dict, Any, Tuple, Union, Optional, Callable

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

    # [REMOVED] The 'then' method is removed as it was tied to the old tool() system.
    # Users should now chain steps explicitly via the FuzzyPipeline instance.
    # def then(...) -> ...


class FuzzyPipeline:
    """
    A builder for creating and executing a Directed Acyclic Graph (DAG) of
    fuzzy analysis operations using Analysis Components.

    This class provides a Fluent API for defining complex, non-linear workflows
    by linking component methods together. The graph is only executed when the
    ``run`` method is called.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.analysis import FuzzyPipeline
        from my_app.tools import FuzzifyComponent, WeightComponent

        # 1. Instantiate components
        fuzzifier = FuzzifyComponent()
        weighter = WeightComponent(method='entropy')

        # 2. Create a pipeline builder
        p = FuzzyPipeline()

        # 3. Define inputs and build the graph using the fluent API
        init_data = p.input("init_data")
        fuzz_table = p.add(fuzzifier.run, data=init_data)
        weights = p.add(weighter.calculate, matrix=fuzz_table)

        # 4. The pipeline is now a defined graph, ready to be run.
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
            name = "default_input"
        if name not in self._input_nodes:
            step_id = f"input_{name}_{uuid.uuid4().hex[:8]}"
            self._input_nodes[name] = step_id
            self._steps[step_id] = {
                "id": step_id,
                "display_name": f"input.{name}",
                "callable": None,               # Input nodes have no callable
                "outputs": {"output": contract}
            }
        step_id = self._input_nodes[name]
        return StepOutput(step_id=step_id, output_name="output", pipeline=self)

    def add(self, callable_tool: Callable, **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:
        """
        Adds a new step to the pipeline using a callable component method.

        This is now the ONLY way to add a computational step to the pipeline.
        The provided callable must be a method decorated with @contract.
        """
        if not callable(callable_tool) or not hasattr(callable_tool, '_is_contract_method'):
            raise TypeError(
                "The object passed to `add()` must be a callable method "
                "decorated with @contract."
            )
        return self._add_step(callable_tool=callable_tool, **kwargs)

    def _add_step(self,
                  callable_tool: Callable,
                  **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:
        """
        Internal method to add a new step from a callable.
        """
        # --- 1. Retrieve metadata directly from the callable ---
        display_name = callable_tool.__qualname__
        expected_inputs_contracts = getattr(callable_tool, '_contract_inputs', {})
        outputs_meta = getattr(callable_tool, '_contract_outputs', {})
        step_id = f"{display_name.replace('.', '_')}_{uuid.uuid4().hex[:8]}"

        # --- 2. Separate data inputs and static parameters (no change in logic) ---
        data_inputs_from_kwargs: Dict[str, StepOutput] = {}
        tool_parameters: Dict[str, Any] = {}
        for arg_name, value in kwargs.items():
            if isinstance(value, StepOutput):
                data_inputs_from_kwargs[arg_name] = value
            else:
                tool_parameters[arg_name] = value

        # --- 3. Graph-Time Validation (no change in logic) ---
        if set(expected_inputs_contracts.keys()) != set(data_inputs_from_kwargs.keys()):
            missing = set(expected_inputs_contracts.keys()) - set(data_inputs_from_kwargs.keys())
            extra = set(data_inputs_from_kwargs.keys()) - set(expected_inputs_contracts.keys())
            raise TypeError(f"Input mismatch for '{display_name}'. Missing: {missing}, Extra: {extra}")

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
                    f"Contract mismatch for tool '{display_name}' on input '{arg_name}'. "
                    f"Expected '{expected_contract}', but received a promise for "
                    f"'{promised_contract}' from step '{source_step_meta['tool']}'."
                )

            # Storing the symbol object itself as a dependency
            resolved_dependencies[arg_name] = value_step_output

        # --- 4. Register the new step in the graph ---
        self._steps[step_id] = {
            "id": step_id,
            "display_name": display_name,
            "callable": callable_tool,
            "parameters": tool_parameters,
            "inputs": data_inputs_from_kwargs,
            "input_contracts": expected_inputs_contracts,
            "outputs": outputs_meta,
            "dependencies": data_inputs_from_kwargs.copy()
        }

        # --- 5. Create symbolic output(s) (no change in logic) ---
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
    def _resolve_step_inputs(step_info: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves a step's inputs from symbolic form to real data from the state,
        and combines them with static parameters.
        This function also performs runtime contract validation on the resolved data.
        """
        resolved_inputs_and_params = {}

        # 1. Resolve data inputs from state
        for arg_name, step_output in step_info['inputs'].items():
            source_result = state[step_output.step_id]
            if isinstance(source_result, dict):
                resolved_inputs_and_params[arg_name] = source_result.get(step_output.output_name)
            else:
                resolved_inputs_and_params[arg_name] = source_result

        # 2. Add static parameters
        resolved_inputs_and_params.update(step_info['parameters'])

        # 3. Runtime Contract Validation
        input_contracts = step_info['input_contracts']
        for arg_name, data in resolved_inputs_and_params.items():
            if arg_name in input_contracts:
                expected_contract = input_contracts[arg_name]
                if not validate_contract(expected_contract, data):
                    raise TypeError(
                        f"Runtime contract validation failed for '{step_info['display_name']}' on input '{arg_name}'. "
                        f"Expected '{expected_contract}', but received {type(data).__name__}."
                    )
        return resolved_inputs_and_params

    def run(self,
            initial_data: Union[Dict[str, Any], Any],
            return_intermediate: bool = False
            ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """
        Executes the defined pipeline DAG with the provided initial data.

        This method orchestrates the entire analysis workflow. It first performs a
        topological sort of the graph to determine a valid execution order. Then, it
        iterates through the steps, resolving symbolic inputs to concrete data,
        performing runtime contract validation, and executing the component methods.

        Parameters
        ----------
        initial_data : dict[str, Any] or Any
            The starting data for the pipeline. The format depends on the number
            of inputs defined in the pipeline:

            - If the pipeline has a single input (defined via `p.input()`), you
              can provide the data object directly (e.g., a `pandas.DataFrame`).
            - If the pipeline has multiple inputs, you **must** provide a
              dictionary mapping the input names to their corresponding data
              objects.
        return_intermediate : bool, default False
            If ``True``, the method returns a tuple containing the final results
            and a dictionary of all intermediate step results. This is useful
            for debugging and inspecting the data flow. If ``False``, only the
            final results are returned.

        Returns
        -------
        Any or tuple[Any, dict[str, Any]]
            The final output(s) of the pipeline. The structure of the output
            depends on the number of terminal (leaf) nodes in the graph:

            - If the graph has a single terminal node that declares a single
              output, the computed value is returned directly.
            - If the graph has multiple terminal nodes, or a single terminal
              node that declares multiple outputs, a dictionary mapping the
              terminal nodes' display names to their results is returned.

            If `return_intermediate` is ``True``, the return value will be a
            tuple `(final_output, intermediate_states)`, where `final_output`
            follows the logic above and `intermediate_states` is a dictionary
            mapping every step ID to its computed result.

        Raises
        ------
        ValueError
            - If a cycle is detected in the pipeline graph.
            - If the pipeline has no inputs but data is provided.
        KeyError
            - If the keys in the `initial_data` dictionary do not perfectly
              match the names of the defined pipeline inputs.
        TypeError
            - If the pipeline has multiple inputs but `initial_data` is not a
              dictionary.
            - If, at runtime, the data passed to a component method does not
              conform to its declared input contract.

        See Also
        --------
        add : The method for adding a component step to the pipeline.
        input : The method for defining an entry point for the pipeline.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.analysis import FuzzyPipeline
            from my_app.tools import FuzzifyComponent, WeightComponent
            import pandas as pd

            # --- Setup ---
            fuzzifier = FuzzifyComponent()
            weighter = WeightComponent(method='entropy')
            p = FuzzyPipeline()

            # --- Building the graph ---
            init_data = p.input("decision_matrix")
            fuzz_table = p.add(fuzzifier.run, data=init_data)
            weights = p.add(weighter.calculate, matrix=fuzz_table)

            # --- Running the pipeline ---
            my_data = pd.DataFrame(...)
            # Execute and get the final result directly
            final_weights = p.run({"decision_matrix": my_data})

            # Execute and get intermediate results for debugging
            final_weights, all_steps = p.run(
                {"decision_matrix": my_data},
                return_intermediate=True
            )
            print(f"Final Weights: {final_weights}")
            # `all_steps` contains results from fuzzifier.run and weighter.calculate
            # print(f"Fuzzified Table: {all_steps[fuzz_table.step_id]}")
        """
        state: Dict[str, Any] = {}          # The state dictionary for execution
        num_inputs = len(self._input_nodes)

        # 1. Prepare and validate initial data, populating the initial state.
        if isinstance(initial_data, dict):
            # Standard case: initial_data is a dictionary.
            if set(initial_data.keys()) != set(self._input_nodes.keys()):
                raise KeyError(
                    f"The keys in `initial_data` {list(initial_data.keys())} "
                    f"do not match the pipeline's defined inputs {list(self._input_nodes.keys())}."
                )
            for input_name, data in initial_data.items():
                input_step_id = self._input_nodes[input_name]
                state[input_step_id] = data
        else:
            # Convenience case: initial_data is a single object.
            if num_inputs == 0:
                raise ValueError("Pipeline has no inputs, but data was provided.")
            elif num_inputs > 1:
                raise TypeError(
                    f"Pipeline has multiple inputs ({list(self._input_nodes.keys())}), "
                    f"so `initial_data` must be a dictionary."
                )
            # num_inputs == 1, which is the only valid case here.
            input_name = list(self._input_nodes.keys())[0]
            input_step_id = self._input_nodes[input_name]
            state[input_step_id] = initial_data

        # 2. Determine the execution order via topological sort.
        execution_order = self._build_execution_order()

        # 3. Execute each step in the determined order.
        for step_id in execution_order:
            step_info = self.get_step_info(step_id)
            tool_fn = step_info['callable']

            # Resolve symbolic inputs to concrete data from the state dictionary.
            # This also performs runtime contract validation.
            resolved_inputs = self._resolve_step_inputs(step_info, state)

            # Execute the component method with the resolved inputs.
            result = tool_fn(**resolved_inputs)

            # Store the result of the step in the state dictionary.
            state[step_id] = result

        # 4. Identify terminal nodes to determine the final output.
        all_step_ids = set(self._steps.keys())
        source_step_ids = set(
            dep.step_id for step in self._steps.values()
            if 'dependencies' in step for dep in step['dependencies'].values()
        )
        terminal_step_ids = all_step_ids - source_step_ids

        # 5. Collect results from all terminal nodes.
        terminal_results = {}
        for step_id in terminal_step_ids:
            # An input node can be a terminal node if it's not connected.
            if step_id in state:
                step_info = self.get_step_info(step_id)
                # Use display_name for the key, which is more user-friendly.
                terminal_results[step_info['display_name']] = state[step_id]

        # 6. Format the final output based on the number of terminal results.
        final_output: Any
        if len(terminal_results) == 1:
            # If there's only one terminal node, its result is the final output.
            final_output = list(terminal_results.values())[0]
        else:
            # If there are zero or multiple terminal nodes, return a dictionary
            # of all their results.
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
