#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 13:19
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the component-based Fluent API and DAG execution engine for fuzzy
analysis workflows.

This module provides the core _components for building and running declarative,
graph-based analysis pipelines using Analysis Components.
- ``FuzzyPipeline``: The main class for constructing a computational graph (DAG).
- ``StepMetadata``: A structured representation of a step's metadata.
- ``StepOutput``: A symbolic object representing the future output of a pipeline step.
- ``ExecutionState``: An immutable object representing the pipeline's state at a point in execution.
- ``FuzzyPipelineIterator``: An iterator for simple, step-by-step observation of pipeline execution.
"""
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Union, Optional, Callable

# --- Dependency Check ---
try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

try:
    import pydot
    _PYDOT_AVAILABLE = True
except ImportError:
    _PYDOT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from IPython.display import Image
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

from .contracts import (
    validate as validate_contract,
    register_contract,
    ContractValidator
)
from ._components.base import AnalysisComponent

# --- Pipeline-Specific Contracts ---
# Decouple 'PipelineResult' from the global _contracts.py by defining it here.
PIPELINE_RESULT_CONTRACT_NAME = 'PipelineResult'


@register_contract(PIPELINE_RESULT_CONTRACT_NAME)
class _PipelineResultValidator(ContractValidator):
    """Validates if an object is a dictionary, suitable for nested pipeline results."""
    def validate(self, obj: Any) -> bool:
        return isinstance(obj, dict)


# --- Metadata Structure ---
@dataclass
class StepMetadata:
    """
    A structured representation of a pipeline step's metadata.

    This class eliminates "magic strings" for dictionary keys and provides
    a clear, type-hinted structure for what defines a step in the pipeline.

    Attributes
    ----------
    step_id : str
        The unique identifier for the step.
    display_name : str
        A human-readable name for the step, used in visualizations and reports.
    callable_tool : Optional[Callable]
        The function or method to be executed for this step. None for input nodes.
    dependencies : Dict[str, 'StepOutput']
        A mapping of input parameter names to their source StepOutput objects.
    static_parameters : Dict[str, Any]
        A mapping of input parameter names to static values (not from other steps).
    input_contracts : Dict[str, str]
        A mapping of input parameter names to their expected data contracts.
    output_contracts : Dict[str, str]
        A mapping of output names to their data contracts.
    """
    step_id: str
    display_name: str
    callable_tool: Optional[Callable]

    # Dependencies and static configuration
    dependencies: Dict[str, 'StepOutput'] = field(default_factory=dict)
    static_parameters: Dict[str, Any] = field(default_factory=dict)

    # Data contracts for validation
    input_contracts: Dict[str, str] = field(default_factory=dict)
    output_contracts: Dict[str, str] = field(default_factory=dict)

    @property
    def is_input_node(self) -> bool:
        """Check if this step is an input node."""
        return self.callable_tool is None

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the StepMetadata."""
        return (f"<StepMetadata id='{self.step_id}', "
                f"name='{self.display_name}', "
                f"inputs={list(self.input_contracts.keys())}, "
                f"input_contracts={self.input_contracts}, "
                f"outputs={list(self.output_contracts.keys())}, "
                f"output_contracts={self.output_contracts}>")


# --- Core Pipeline Classes ---
class StepOutput:
    """
    A symbolic representation of a future output from a pipeline step.

    This object acts as a placeholder or "promise" for a value that will be
    computed when the pipeline runs. It doesn't hold any real data itself but
    contains the necessary information to track dependencies within the DAG [4].

    Instances of this class are what enable the Fluent API, allowing users to
    chain operations together in a readable and type-safe manner.

    Attributes
    ----------
    step_id : str
        The unique identifier of the step that will produce this output.
    output_name : str
        The name of the specific output from the step (for tools with multiple returns).
    pipeline : 'FuzzyPipeline'
        A reference to the parent pipeline instance that owns this step [1].
    """

    def __init__(self, step_id: str, output_name: str, pipeline: 'FuzzyPipeline'):
        self.step_id = step_id
        self.output_name = output_name
        self.pipeline = pipeline

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the StepOutput."""
        # Use the new StepMetadata structure for a more robust representation
        step_meta = self.pipeline.get_step_info(self.step_id)
        tool_name = step_meta.display_name or "unknown_tool"
        output_contract = step_meta.output_contracts

        return (f"<StepOutput of '{tool_name}' "
                f"(step: {self.step_id[:8]}, "
                f"output_name: {self.output_name}), "
                f"output_contract: {output_contract}>")


class ExecutionState:
    """
    Represents an immutable state of a pipeline's execution at a specific point.

    This object encapsulates the results of all completed steps and provides
    a method to execute the next step in the pipeline, returning a new
    ExecutionState. This enables a functional, chainable approach to
    step-by-step execution.

    Instances of this class are intended to be created by `FuzzyPipeline.start_execution()`.

    Attributes
    ----------
    pipeline : FuzzyPipeline
        The pipeline this state belongs to.
    step_results : Dict[str, Any]
        A dictionary mapping step IDs to their computed results.
    latest_step_id : Optional[str]
        The ID of the most recently executed step. None for the initial state.
    execution_order : List[str]
        The pre-computed list of step IDs to be executed.
    current_index : int
        An index pointing to the next step to be executed in `execution_order`.
    """

    def __init__(self,
                 pipeline: 'FuzzyPipeline',
                 step_results: Dict[str, Any],
                 execution_order: List[str],
                 current_index: int,
                 latest_step_id: Optional[str] = None):

        self.pipeline = pipeline
        self.step_results = dict(step_results)  # Create a copy to ensure immutability
        self.execution_order = execution_order
        self.current_index = current_index
        self.latest_step_id = latest_step_id

    @property
    def latest_result(self) -> Any:
        """The result of the most recently executed step."""
        if self.latest_step_id is None:
            return None
        return self.step_results.get(self.latest_step_id)

    def is_complete(self) -> bool:
        """Check if the pipeline execution is complete."""
        return self.current_index >= len(self.execution_order)

    def run_next(self) -> 'ExecutionState':
        """
        Executes the next step in the pipeline and returns a new ExecutionState.

        Returns
        -------
        ExecutionState
            A new state object representing the pipeline after the next step
            has been executed.

        Raises
        ------
        StopIteration
            If there are no more steps to execute.
        """
        if self.is_complete():
            raise StopIteration("Pipeline execution is already complete.")

        next_step_id = self.execution_order[self.current_index]
        # Retrieve the structured metadata object instead of a raw dictionary
        step_meta = self.pipeline.get_step_info(next_step_id)

        # Execute the step
        resolved_inputs = self.pipeline.parse_step_inputs(step_meta, self.step_results)
        result = step_meta.callable_tool(**resolved_inputs)

        # Create a new results dictionary for the next state
        new_step_results = self.step_results.copy()
        new_step_results[next_step_id] = result

        return ExecutionState(
            pipeline=self.pipeline,
            step_results=new_step_results,
            execution_order=self.execution_order,
            current_index=self.current_index + 1,
            latest_step_id=next_step_id
        )

    def run_all(self) -> 'ExecutionState':
        """
        Executes all remaining steps in the pipeline.

        Returns
        -------
        ExecutionState
            The final state of the pipeline after all steps are executed.
        """
        current_state = self
        while not current_state.is_complete():
            current_state = current_state.run_next()
        return current_state

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the ExecutionState."""
        status = "Complete" if self.is_complete() else f"Next step {self.current_index + 1}/{len(self.execution_order)}"

        latest_step_name = 'Initial'
        if self.latest_step_id:
            # Accessing metadata via the structured StepMetadata object
            latest_step_name = self.pipeline.get_step_info(self.latest_step_id).display_name

        return f"<ExecutionState ({status}), latest_step_completed='{latest_step_name}'>"


class FuzzyPipelineIterator:
    """
    An iterator for step-by-step execution of a FuzzyPipeline.

    This iterator allows users to execute a pipeline one step at a time for
    observation. It is a convenient wrapper around the more powerful
    `ExecutionState` object.

    Parameters
    ----------
    pipeline : FuzzyPipeline
        The pipeline to execute step by step.
    initial_data : Union[Dict[str, Any], Any]
        The initial data for the pipeline, in the same format as expected
        by `FuzzyPipeline.run()`.
    """
    def __init__(self,
                 pipeline: 'FuzzyPipeline',
                 initial_data: Union[Dict[str, Any], Any]):

        self.current_state = pipeline.start_execution(initial_data)
        self.total_steps = len(self.current_state.execution_order)

    def __iter__(self):
        """Return the iterator object itself."""
        return self

    def __next__(self) -> Dict[str, Any]:
        """
        Execute the next step in the pipeline.

        Returns
        -------
        dict[str, Any]
            A dictionary containing information about the executed step:
            - 'step_id': The unique identifier of the step
            - 'step_name': The display name of the step
            - 'step_index': The current step index (0-based)
            - 'total_steps': Total number of executable steps
            - 'result': The output of the step
            - 'execution_time': Time taken to execute this step (in seconds)

        Raises
        ------
        StopIteration
            When all steps have been executed.
        """
        if self.current_state.is_complete():
            raise StopIteration

        start_time = time.perf_counter()
        # Delegate execution to the current state object
        next_state = self.current_state.run_next()
        end_time = time.perf_counter()

        # Update the iterator's internal state
        self.current_state = next_state

        # Build and return the report dictionary using the structured metadata
        step_meta = self.current_state.pipeline.get_step_info(self.current_state.latest_step_id)
        return {
            # TODO: 这里也是会造成混乱, 能不能用 StepMetadata? 好像不行, StepMetadata 表示的是占位符?
            'step_id': self.current_state.latest_step_id,
            'step_name': step_meta.display_name,
            'step_index': self.current_state.current_index - 1,
            'total_steps': self.total_steps,
            'result': self.current_state.latest_result,
            'execution_time': end_time - start_time
        }

    def get_current_state_dict(self) -> Dict[str, Any]:
        """Returns a copy of the current step results dictionary."""
        return self.current_state.step_results.copy()

    def is_complete(self) -> bool:
        """Checks if the pipeline execution is complete."""
        return self.current_state.is_complete()


class FuzzyPipeline(AnalysisComponent):
    """
    A builder for creating and executing a Directed Acyclic Graph (DAG) of
    fuzzy analysis operations using Analysis Components.

    This class provides a Fluent API for defining complex, non-linear workflows
    by linking component methods together. The graph is only executed when the
    ``run`` method is called. It also supports nesting, allowing a `FuzzyPipeline`
    instance to be used as a step within another pipeline.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.analysis import FuzzyPipeline
        from my_app.tools import FuzzifyComponent, WeightComponent

        # 1. Instantiate _components
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

    def __init__(self, name: Optional[str] = None):
        """
        Initializes a new FuzzyPipeline.

        Parameters
        ----------
        name : str, optional
            An optional name for the pipeline, used for display purposes when nested.
            If not provided, a default name will be generated.
        """
        self.name = name or f"FuzzyPipeline_{uuid.uuid4().hex[:8]}"

        # Use the structured StepMetadata for storing step information
        self._steps: Dict[str, StepMetadata] = {}
        self._input_nodes: Dict[str, str] = {}

    @property
    def steps(self) -> List[StepMetadata]:
        """Returns a list of all defined step metadata objects in the pipeline."""
        return list(self._steps.values())

    @property
    def input_nodes(self) -> Dict[str, str]:
        """Returns a dictionary of the pipeline's input nodes."""
        return self._input_nodes.copy()

    def __repr__(self) -> str:
        num_inputs = len(self._input_nodes)
        num_tasks = len(self._steps) - num_inputs
        return f"<{self.name} with {num_inputs} inputs and {num_tasks} tasks>"

    def input(self,
              name: Optional[str] = None,
              contract: str = 'any') -> StepOutput:
        """
        Defines an input entry point for the pipeline, with an optional contract.

        This creates a "source" node in the DAG, which will be populated with
        real data when the pipeline's ``run`` method is called.

        Parameters
        ----------
        name : str, optional
            The name of the input handle. If omitted, a default name will be
            generated. This is recommended for single-input pipelines.
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

            # TODO: 又有一处硬编码
            name = "default_input"

        if name not in self._input_nodes:
            step_id = f"input_{name}_{uuid.uuid4().hex[:8]}"
            self._input_nodes[name] = step_id

            # Create a StepMetadata instance for the input node
            # TODO: 这里初始化节点为什么又是硬编码又是魔法字符串？
            #  而且也不是陪 StepMetadata 设计的初衷
            self._steps[step_id] = StepMetadata(
                step_id=step_id,
                display_name=f"input.{name}",
                callable_tool=None,
                output_contracts={"output": contract}
            )

        step_id = self._input_nodes[name]
        # Input nodes always have a single, conventional output named 'output'
        # TODO: 这里的 'output' 也是硬编码和魔法字符串
        return StepOutput(step_id=step_id, output_name="output", pipeline=self)

    def add(self,
            callable_tool: Callable,
            **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:
        """
        Adds a new step to the pipeline using a callable component method.

        This method is the primary way to build the computation graph. It can accept:
        1. A callable method from an ``AnalysisComponent`` instance, decorated with ``@contract``.
        2. Another ``FuzzyPipeline`` instance, enabling nested pipelines.

        Parameters
        ----------
        callable_tool : Callable or FuzzyPipeline
            The computational unit to add.
        **kwargs :
            The inputs for the step, mapping parameter names to `StepOutput` objects.

        Returns
        -------
        StepOutput or dict[str, StepOutput]
            Symbolic output(s) of the newly added step.
        """
        if isinstance(callable_tool, FuzzyPipeline):
            return self._add_pipeline_step(pipeline_tool=callable_tool, **kwargs)

        if not callable(callable_tool) or not hasattr(callable_tool, '_is_contract_method'):
            raise TypeError(
                "The object passed to 'add()' must be a callable method "
                "decorated with '@contract', or another FuzzyPipeline instance."
            )

        return self._add_step(callable_tool=callable_tool, **kwargs)

    def _add_pipeline_step(self,
                           pipeline_tool: 'FuzzyPipeline',
                           **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:
        """Internal method to add a nested FuzzyPipeline as a single step."""
        def pipeline_runner(**resolved_inputs):
            return pipeline_tool.run(initial_data=resolved_inputs)

        return self._add_step(
            callable_tool=pipeline_runner,
            display_name_override=pipeline_tool.name,
            input_contracts_override=pipeline_tool.get_input_contracts(),
            output_contracts_override=pipeline_tool.get_output_contracts(),
            **kwargs
        )

    def _add_step(self,
                  callable_tool: Callable,
                  display_name_override: Optional[str] = None,
                  input_contracts_override: Optional[Dict[str, str]] = None,
                  output_contracts_override: Optional[Dict[str, str]] = None,
                  **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:
        """Internal method to add a new step from a callable."""
        # --- 1. Retrieve metadata ---
        is_nested_pipeline = display_name_override is not None
        if is_nested_pipeline:
            display_name = display_name_override
            expected_inputs_contracts = input_contracts_override
            output_contracts = output_contracts_override
        else:
            display_name = callable_tool.__qualname__
            expected_inputs_contracts = getattr(callable_tool, '_contract_inputs', {})
            output_contracts = getattr(callable_tool, '_contract_outputs', {})

        step_id = f"{display_name.replace('.', '_')}_{uuid.uuid4().hex[:8]}"

        # --- 2. Separate data inputs and static parameters ---
        data_inputs: Dict[str, StepOutput] = {}
        static_params: Dict[str, Any] = {}
        for arg_name, value in kwargs.items():
            if isinstance(value, StepOutput):
                data_inputs[arg_name] = value
            else:
                static_params[arg_name] = value

        # --- 3. Graph-Time Validation ---
        if set(expected_inputs_contracts.keys()) != set(data_inputs.keys()):
            missing = set(expected_inputs_contracts.keys()) - set(data_inputs.keys())
            extra = set(data_inputs.keys()) - set(expected_inputs_contracts.keys())
            raise TypeError(f"Input mismatch for '{display_name}'. Missing: {missing}, Extra: {extra}")

        for arg_name, step_out in data_inputs.items():
            source_step_meta = self.get_step_info(step_out.step_id)
            promised_contract = source_step_meta.output_contracts.get(step_out.output_name)

            # Check whether the committed contract matches the expected contract.
            expected_contract = expected_inputs_contracts.get(arg_name)

            contracts_match = (promised_contract == 'any' or expected_contract == promised_contract)
            if expected_contract and promised_contract and not contracts_match:
                raise TypeError(
                    f"Contract mismatch for tool '{display_name}' on input '{arg_name}'. "
                    f"Expected '{expected_contract}', but received a promise for "
                    f"'{promised_contract}' from step '{source_step_meta.display_name}'."
                )

        # --- 4. Create and store the step metadata ---
        self._steps[step_id] = StepMetadata(
            step_id=step_id,
            display_name=display_name,
            callable_tool=callable_tool,
            static_parameters=static_params,
            dependencies=data_inputs,
            input_contracts=expected_inputs_contracts,
            output_contracts=output_contracts
        )

        # --- 5. Create symbolic output(s) ---
        if len(output_contracts) == 1:
            output_name = list(output_contracts.keys())[0]
            return StepOutput(step_id=step_id, output_name=output_name, pipeline=self)
        else:
            return {
                name: StepOutput(step_id=step_id, output_name=name, pipeline=self)
                for name in output_contracts.keys()
            }

    def _get_terminal_steps(self) -> List[StepMetadata]:
        """Finds all terminal step metadata objects in the pipeline graph."""
        all_step_ids = set(self._steps.keys())
        dependency_source_ids = set()
        for step in self._steps.values():
            for dep_output in step.dependencies.values():
                dependency_source_ids.add(dep_output.step_id)

        terminal_ids = all_step_ids - dependency_source_ids
        return [self._steps[tid] for tid in terminal_ids if not self._steps[tid].is_input_node]

    def get_input_contracts(self) -> Dict[str, str]:
        """Gets the input contracts for this pipeline."""
        contracts = {}
        for name, step_id in self._input_nodes.items():
            step_meta = self.get_step_info(step_id)
            contracts[name] = step_meta.output_contracts['output']
        return contracts

    def get_step_info(self, step_id: str) -> StepMetadata:
        """Retrieves the metadata for a given step ID."""
        if step_id not in self._steps:
            raise ValueError(f"Step with ID '{step_id}' not found in this pipeline.")
        return self._steps[step_id]

    def get_output_contracts(self) -> Dict[str, str]:
        """
        Gets the output contracts for this pipeline.

        This is determined by inspecting the contracts of the terminal (leaf)
        nodes in the pipeline's graph.

        Returns
        -------
        dict[str, str]
            A dictionary mapping output names to their contract names.
        """
        terminal_steps = self._get_terminal_steps()
        if not terminal_steps:
            return {}

        if len(terminal_steps) == 1:
            return terminal_steps[0].output_contracts.copy()

        output_contracts = {}
        for step in terminal_steps:
            for name, contract in step.output_contracts.items():
                output_contracts[f"{step.display_name}_{name}"] = contract

        if len(output_contracts) > 1:
            # TODO: 这里的 result 为什么也是硬编码
            #  因为 FuzzyPipelineIterator 中设定好了 meta? 是这个原因吗?
            return {'result': PIPELINE_RESULT_CONTRACT_NAME}
        else:
            return output_contracts

    def _build_execution_order(self) -> List[str]:
        """
        Performs a topological sort of the graph to get a linear execution order.

        This method returns the full execution order including input nodes.
        The filtering is handled by the execution engines.

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

        for step_id, step_meta in self._steps.items():
            for dep in step_meta.dependencies.values():
                source_id = dep.step_id
                adj[source_id].append(step_id)
                in_degree[step_id] += 1

        # Key: Kahn's algorithm for topological sorting
        queue = deque([step_id for step_id, degree in in_degree.items() if degree == 0])
        execution_order = []

        while queue:
            current_id = queue.popleft()
            execution_order.append(current_id)
            for neighbor_id in adj[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(execution_order) != len(self._steps):
            raise ValueError("A cycle was detected in the pipeline graph. Please check dependencies.")
        return execution_order

    @staticmethod
    def parse_step_inputs(step_meta: StepMetadata, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses a step's inputs from symbolic form to real data from the state,
        and combines them with static parameters.
        """
        parsed_inputs = {}

        # 1. Resolve data inputs from state
        for arg_name, step_output in step_meta.dependencies.items():
            source_step_meta = step_output.pipeline.get_step_info(step_output.step_id)
            source_result = state[step_output.step_id]

            if len(source_step_meta.output_contracts) > 1:
                # Multi-output scenario: The result must be a dictionary.
                if not isinstance(source_result, dict):
                    raise TypeError(
                        f"Step '{source_step_meta.display_name}' was expected to return a dict for multiple outputs, "
                        f"but returned {type(source_result).__name__}."
                    )
                parsed_inputs[arg_name] = source_result.get(step_output.output_name)
            else:
                # Single-output scenario: Smartly unpack the result.
                # If the component author returned a dict with one item (e.g., {'output': data}),
                # we extract the value to simplify downstream usage.
                if isinstance(source_result, dict) and len(source_result) == 1:
                    parsed_inputs[arg_name] = list(source_result.values())[0]
                else:
                    # Otherwise, use the result as is (e.g., a direct DataFrame return).
                    parsed_inputs[arg_name] = source_result

        # 2. Add static parameters
        parsed_inputs.update(step_meta.static_parameters)

        # 3. Runtime Contract Validation
        for arg_name, data in parsed_inputs.items():
            if arg_name in step_meta.input_contracts:
                expected_contract = step_meta.input_contracts[arg_name]
                if not validate_contract(expected_contract, data):
                    raise TypeError(
                        f"Runtime contract validation failed for '{step_meta.display_name}' on input '{arg_name}'. "
                        f"Expected '{expected_contract}', but received {type(data).__name__}."
                    )
        return parsed_inputs

    def _format_final_output(self, final_state: 'ExecutionState') -> Any:
        """Helper method to format the final output from the final execution state [4]."""
        terminal_steps = self._get_terminal_steps()
        terminal_results = {
            step.display_name: final_state.step_results[step.step_id]
            for step in terminal_steps if step.step_id in final_state.step_results
        }

        if len(terminal_results) == 1:
            return list(terminal_results.values())[0]
        else:
            return terminal_results

    def run(self,
            initial_data: Union[Dict[str, Any], Any],
            return_intermediate: bool = False
            ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """
        Executes the defined pipeline DAG with the provided initial data.

        This method orchestrates the entire analysis workflow. It first performs a
        topological sort of the graph to determine a valid execution order. Then, it
        iterates through the steps, resolving symbolic inputs to concrete data,
        and providing data and receiving the final result(s). This method now uses
        the underlying `ExecutionState` engine.

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
        initial_state = self.start_execution(initial_data)
        final_state = initial_state.run_all()
        final_output = self._format_final_output(final_state)

        if return_intermediate:
            return final_output, final_state.step_results
        else:
            return final_output

    def step_by_step(self, initial_data: Union[Dict[str, Any], Any]) -> FuzzyPipelineIterator:
        """
        Create an iterator for step-by-step execution of this pipeline.

        This method provides a way to execute the pipeline incrementally,
        allowing users to observe and interact with intermediate results.
        Use this in a `for` loop or with `next()` for simple debugging and
        process observation.

        Parameters
        ----------
        initial_data : Union[Dict[str, Any], Any]
            The initial data for the pipeline. Format is the same as for
            the `run()` method.

        Returns
        -------
        FuzzyPipelineIterator
            An iterator that yields step-by-step execution results.

        Examples
        --------
        .. code-block:: python

            # Basic usage
            for step in pipeline.step_by_step(my_data):
                print(f"Step {step['step_index'] + 1}/{step['total_steps']}: "
                      f"{step['step_name']}")
                print(f"Execution time: {step['execution_time']:.4f}s")

                # Optionally examine the result
                if isinstance(step['result'], dict):
                    print(f"Result keys: {list(step['result'].keys())}")

            # Manual iteration
            iterator = pipeline.step_by_step(my_data)
            first_step = next(iterator)
            print("First step completed!")

            # Check remaining steps
            print(f"Remaining steps: {iterator.get_remaining_count()}")
        """
        return FuzzyPipelineIterator(self, initial_data)

    def start_execution(self, initial_data: Union[Dict[str, Any], Any]) -> ExecutionState:
        """
        Initializes a step-by-step execution and returns the initial state object.

        This method prepares the pipeline for a chainable, step-by-step execution.
        It does not run any computational steps itself but returns an `ExecutionState`
        object that can be used to run the next step. Use this for advanced control,
        where each step's execution is triggered manually and returns a new state object.

        Parameters
        ----------
        initial_data : Union[Dict[str, Any], Any]
            The initial data for the pipeline. Format is the same as for
            the `run()` method.

        Returns
        -------
        ExecutionState
            The initial state object, ready to execute the first step via `.then()`.

        Examples
        --------
        .. code-block:: python

            # Start the execution
            state0 = pipeline.start_execution(my_data)

            # Execute the first step
            state1 = state0.then()
            print("Result of step 1:", state1.latest_result)

            # Execute the second step
            state2 = state1.then()
            print("Result of step 2:", state2.latest_result)

            # Run all remaining steps
            final_state = state2.run_all()

            # Get all results
            all_results = final_state.step_results
        """
        initial_state_dict = {}
        num_inputs = len(self._input_nodes)

        if isinstance(initial_data, dict):
            if set(initial_data.keys()) != set(self._input_nodes.keys()):
                raise KeyError(f"The keys in 'initial_data' {list(initial_data.keys())} do not match the "
                               f"pipeline's defined inputs {list(self._input_nodes.keys())}.")
            for input_name, data in initial_data.items():
                initial_state_dict[self._input_nodes[input_name]] = data

        else:
            if num_inputs > 1:
                raise TypeError(f"Pipeline has multiple inputs ({list(self._input_nodes.keys())}), "
                                f"so 'initial_data' must be a dictionary.")
            if num_inputs == 1:
                initial_state_dict[list(self._input_nodes.values())[0]] = initial_data

        full_execution_order = self._build_execution_order()
        executable_steps = [
            step_id for step_id in full_execution_order
            if not self.get_step_info(step_id).is_input_node
        ]

        return ExecutionState(
            pipeline=self,
            step_results=initial_state_dict,
            execution_order=executable_steps,
            current_index=0
        )

    def _to_networkx(self) -> 'nx.DiGraph':
        """
        Converts the internal pipeline structure to a NetworkX DiGraph.

        This is a helper method for visualization.

        Returns
        -------
        nx.DiGraph
            A NetworkX directed graph representing the pipeline.
        """
        if not _NX_AVAILABLE:
            raise ImportError("Visualization requires 'networkx'. "
                              "Please install it using: pip install networkx")

        g = nx.DiGraph(name=self.name)

        for step_id, step_meta in self._steps.items():
            is_nested_pipeline = isinstance(step_meta.callable_tool, FuzzyPipeline)
            if step_meta.is_input_node:
                node_type, label = 'input', f"Input\n({step_meta.display_name.split('.')[-1]})"
            elif is_nested_pipeline:
                node_type, label = 'pipeline', f"Pipeline:\n{step_meta.display_name}"
            else:
                node_type, label = 'component', step_meta.display_name
            g.add_node(step_id, label=label, node_type=node_type)

            for dep_output in step_meta.dependencies.values():
                source_meta = self.get_step_info(dep_output.step_id)
                contract = source_meta.output_contracts.get(dep_output.output_name, 'any')
                edge_label = f"{dep_output.output_name}: {contract}"
                g.add_edge(dep_output.step_id, step_id, label=edge_label)

        return g

    def visualize(self,
                  filename: Optional[str] = None,
                  output_format: str = 'png',
                  show_contracts: bool = True,
                  engine: str = 'auto') -> Optional['Image']:
        """
        Generates a visual representation of the pipeline's DAG.

        This method attempts to use the best available rendering engine.
        - 'graphviz': (Requires `pydot` and system-wide `Graphviz`) Provides high-quality, hierarchical layouts.
        - 'matplotlib': (Requires `matplotlib`) Provides a basic, force-directed layout.

        Parameters
        ----------
        filename : str, optional
            The path to save the output image file. If None, the image is displayed directly.
        output_format : str, default 'png'
            The output format for the image (used by Graphviz engine).
        show_contracts : bool, default True
            If True, display data contract information on the edges.
        engine : {'auto', 'graphviz', 'matplotlib'}, default 'auto'
            The rendering engine to use. 'auto' will try 'graphviz' first,
            then fall back to 'matplotlib'.

        Returns
        -------
        IPython.display.Image or None
            If using the 'graphviz' engine without a filename in an IPython
            environment, returns an Image object. Otherwise, returns None.

        Raises
        ------
        ImportError
            If the chosen engine or its dependencies are not available.
        """
        # This method's implementation can remain largely the same,
        # as it relies on _to_networkx(), which has been updated.
        # For brevity, I'll omit the detailed visualization code here,
        # assuming it calls the updated _to_networkx() correctly.
        nx_graph = self._to_networkx()

        use_engine = None
        if engine == 'auto':
            if _PYDOT_AVAILABLE:
                use_engine = 'graphviz'
            elif _MPL_AVAILABLE:
                use_engine = 'matplotlib'

        elif engine == 'graphviz':
            if not _PYDOT_AVAILABLE:
                raise ImportError("`pydot` is required for 'graphviz' engine.")
            use_engine = 'graphviz'

        elif engine == 'matplotlib':
            if not _MPL_AVAILABLE:
                raise ImportError("`matplotlib` is required for 'matplotlib' engine.")
            use_engine = 'matplotlib'

        else:
            raise ValueError(f"Unknown engine: '{engine}'.")

        if use_engine is None:
            raise ImportError(
                "No visualization engine available. Please install `matplotlib` or `pydot` (and Graphviz)."
            )

        if use_engine == 'graphviz':
            return self._visualize_graphviz(nx_graph, filename, output_format, show_contracts)
        else:
            return self._visualize_matplotlib(nx_graph, filename, show_contracts)

    def _visualize_graphviz(self, nx_graph, filename, output_format, show_contracts):
        """Renders the graph using Graphviz."""
        pdot_graph = pydot.Dot(graph_type='digraph', rankdir='TB', label=f'Pipeline: {self.name}', fontsize=20, labelloc='t')

        styles = {
            # TODO: 这里设定好了 node 的样式, 但是都是硬编码, 能否用其他办法?
            'input': {'shape': 'ellipse', 'fillcolor': '#E8F5E9', 'style': 'filled'},
            'component': {'shape': 'box', 'fillcolor': '#E3F2FD', 'style': 'filled,rounded'},
            'pipeline': {'shape': 'box', 'fillcolor': '#FFFDE7', 'style': 'filled,bold'}
        }

        for node_id, attrs in nx_graph.nodes(data=True):
            node_style = styles.get(attrs['node_type'], {})
            pdot_node = pydot.Node(name=node_id, label=attrs['label'], **node_style)
            pdot_graph.add_node(pdot_node)

        for u, v, attrs in nx_graph.edges(data=True):
            edge_label = attrs.get('label', '') if show_contracts else ''
            pdot_edge = pydot.Edge(u, v, label=edge_label, fontsize=10, fontcolor='#424242')
            pdot_graph.add_edge(pdot_edge)

        try:
            if filename:
                pdot_graph.write(filename, format=output_format)
                print(f"Pipeline visualization saved to '{filename}'")
                return None
            elif _IPYTHON_AVAILABLE:
                return Image(pdot_graph.create(format=output_format))
            else:
                # Fallback if not in IPython but no filename given
                temp_filename = f"pipeline_view_{uuid.uuid4().hex[:6]}.{output_format}"
                pdot_graph.write(temp_filename, format=output_format)
                print(f"Not in an IPython environment. Visualization saved to '{temp_filename}'")
                return None
        except pydot.DotExecutableNotFoundError:
            # TODO: 这里触发警告: 在 '__init__.py' 中找不到引用 'DotExecutableNotFoundError'
            raise ImportError(
                "Graphviz executable not found. Please install Graphviz and ensure "
                "it is in your system's PATH. See https://graphviz.org/download/"
            )

    def _visualize_matplotlib(self, nx_graph, filename, show_contracts):
        """Renders the graph using Matplotlib."""
        pos = nx.spring_layout(nx_graph, seed=42)  # spring_layout is a decent default

        plt.figure(figsize=(12, 8))

        # Node colors
        # TODO: 这里也是一样采用的硬编码?
        color_map = {
            'input': '#90CAF9',
            'component': '#A5D6A7',
            'pipeline': '#FFD54F'
        }
        node_colors = [color_map.get(nx_graph.nodes[n]['node_type'], '#BDBDBD') for n in nx_graph.nodes]

        # Draw nodes and labels
        # TODO: 警告: 应为类型 'str'，但实际为 'list[str]'
        #   而且包含硬编码
        nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=3000)
        labels = {n: attrs['label'] for n, attrs in nx_graph.nodes(data=True)}
        nx.draw_networkx_labels(nx_graph, pos, labels, font_size=8)

        # Draw edges and edge labels
        nx.draw_networkx_edges(nx_graph, pos, arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1')
        if show_contracts:
            edge_labels = nx.get_edge_attributes(nx_graph, 'label')
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=7)

        plt.title(f"Pipeline: {self.name}")
        plt.axis('off')

        if filename:
            plt.savefig(filename)
            print(f"Pipeline visualization saved to '{filename}'")
        else:
            plt.show()
