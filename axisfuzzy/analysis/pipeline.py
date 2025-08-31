#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 21:05
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Defines the component-based Fluent API and DAG execution engine.

This module provides the core components for building and running declarative,
graph-based analysis pipelines. This version is fully refactored to integrate
with the new, type-annotation-driven contract system (`contracts`).

Core Components:
- ``FuzzyPipeline``: The main class for constructing a computational graph (DAG).
- ``StepMetadata``: A structured representation of a step's metadata.
- ``StepOutput``: A symbolic object representing the future output of a step.
- ``ExecutionState``: An immutable object for step-by-step execution.
- ``FuzzyPipelineIterator``: An iterator for observing pipeline execution.
"""

from __future__ import annotations
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

# --- Local Imports ---
from .component import AnalysisComponent
from .contracts import Contract
from .build_in import ContractAny


@dataclass
class StepMetadata:
    """
    A structured representation of a pipeline step's metadata.

    This class stores all essential information about a single step within the
    pipeline's graph, including its dependencies and data contracts. It now
    stores actual ``Contract`` objects for robust, type-safe validation.

    Attributes
    ----------
    step_id : str
        The unique identifier for the step.
    display_name : str
        A human-readable name for the step, used in visualizations.
    callable_tool : Optional[Callable]
        The function or method to be executed. It is ``None`` for input nodes.
    dependencies : Dict[str, 'StepOutput']
        A mapping of the callable's parameter names to their source ``StepOutput`` objects.
    static_parameters : Dict[str, Any]
        A mapping of parameter names to static values provided at graph-build time.
    input_contracts : Dict[str, Contract]
        A mapping of input parameter names to their expected ``Contract`` objects.
    output_contracts : Dict[str, Contract]
        A mapping of output names to their resulting ``Contract`` objects.
    """
    step_id: str
    display_name: str
    callable_tool: Optional[Callable]
    dependencies: Dict[str, 'StepOutput'] = field(default_factory=dict)
    static_parameters: Dict[str, Any] = field(default_factory=dict)
    input_contracts: Dict[str, Contract] = field(default_factory=dict)
    output_contracts: Dict[str, Contract] = field(default_factory=dict)

    @property
    def is_input_node(self) -> bool:
        """bool: ``True`` if this step is an input node, ``False`` otherwise."""
        return self.callable_tool is None

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the StepMetadata."""
        return (f"<StepMetadata id='{self.step_id[:8]}', "
                f"name='{self.display_name}', "
                f"inputs={list(self.input_contracts.keys())}, "
                f"outputs={list(self.output_contracts.keys())}>")


class StepOutput:
    """
    A symbolic representation of a future output from a pipeline step.

    This object acts as a placeholder or "promise" for a value that will be
    computed when the pipeline runs. It doesn't hold any real data itself but
    contains the necessary information to track dependencies within the DAG.

    Attributes
    ----------
    step_id : str
        The unique identifier of the step that will produce this output.
    output_name : str
        The name of the specific output from the step.
    pipeline : 'FuzzyPipeline'
        A reference to the parent pipeline instance that owns this step.
    """

    def __init__(self, step_id: str, output_name: str, pipeline: 'FuzzyPipeline'):
        self.step_id = step_id
        self.output_name = output_name
        self.pipeline = pipeline

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the StepOutput."""
        step_meta = self.pipeline.get_step_info(self.step_id)
        tool_name = step_meta.display_name
        output_contract = step_meta.output_contracts.get(self.output_name, 'Unknown')
        return (f"<StepOutput of '{tool_name}' "
                f"(step: {self.step_id[:8]}, output: '{self.output_name}') "
                f"promise: {output_contract}>")


class ExecutionState:
    """
    Represents an immutable state of a pipeline's execution at a specific point.

    This object encapsulates the results of all completed steps and provides
    a method to execute the next step in the pipeline, returning a new
    ``ExecutionState``. This enables a functional, chainable approach to
    step-by-step execution.
    """

    # This class is primarily for internal use by the pipeline engine.
    # Its implementation is sound and does not require extensive docstrings for public API.
    def __init__(self,
                 pipeline: 'FuzzyPipeline',
                 step_results: Dict[str, Any],
                 execution_order: List[str],
                 current_index: int,
                 latest_step_id: Optional[str] = None):
        self.pipeline = pipeline
        self.step_results = dict(step_results)
        self.execution_order = execution_order
        self.current_index = current_index
        self.latest_step_id = latest_step_id

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the execution state."""
        pipeline_name = self.pipeline.name
        total_steps = len(self.execution_order)
        current_step_num = self.current_index

        if self.is_complete():
            status = f"completed (step {current_step_num}/{total_steps})"
            next_step_info = "-- "
        else:
            status = f"step {current_step_num}/{total_steps}"
            next_step_id = self.execution_order[self.current_index]
            next_step_info = f"'{self.pipeline.get_step_info(next_step_id).display_name}'"

        last_step_info = f"'{self.pipeline.get_step_info(self.latest_step_id).display_name}'" if self.latest_step_id else "-- "

        return (
            f"<ExecutionState for '{pipeline_name}' "
            f"({status}, next: {next_step_info}, last: {last_step_info})>"
        )

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
        parsed_inputs = self.pipeline.parse_step_inputs(step_meta, self.step_results)
        result = step_meta.callable_tool(**parsed_inputs)

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


class FuzzyPipelineIterator:
    """
    An iterator for step-by-step execution of a FuzzyPipeline.

    This iterator allows users to execute a pipeline one step at a time for
    observation, debugging, or integration with user interfaces.

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
        Execute the next step and return a report dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing information about the executed step:
            - 'step_id': The unique identifier of the step.
            - 'step_name': The display name of the step.
            - 'step_index': The current step index (0-based).
            - 'total_steps': Total number of executable steps.
            - 'result': The output of the step.
            - 'execution_time': Time taken to execute this step (in seconds).

        Raises
        ------
        StopIteration
            When all steps have been executed.
        """
        if self.current_state.is_complete():
            raise StopIteration

        start_time = time.perf_counter()
        next_state = self.current_state.run_next()
        end_time = time.perf_counter()
        self.current_state = next_state

        step_meta = self.current_state.pipeline.get_step_info(self.current_state.latest_step_id)
        return {
            'step_id': self.current_state.latest_step_id,
            'step_name': step_meta.display_name,
            'step_index': self.current_state.current_index,
            'total_steps': self.total_steps,
            'result': self.current_state.latest_result,
            'execution_time(ms)': round((end_time - start_time) * 1e3, 5)
        }
    
    @property
    def result(self) -> Any:
        """The result of the pipeline execution."""
        return self.current_state.latest_result


class FuzzyPipeline(AnalysisComponent):
    """
    A builder for creating and executing a Directed Acyclic Graph (DAG) of
    analysis operations.

    This class provides a Fluent API for defining complex, non-linear workflows
    by linking component methods together. The graph is only executed when the
    ``run`` method is called. It supports robust, type-safe graph construction
    through the `contracts` system.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.analysis.pipeline import FuzzyPipeline
        from axisfuzzy.analysis.components import ToolNormalization, ToolWeightNormalization
        from axisfuzzy.analysis.contracts import CrispTable, WeightVector
        import pandas as pd
        import numpy as np

        # 1. Instantiate components
        normalizer = ToolNormalization(method='min_max')
        weight_normalizer = ToolWeightNormalization()

        # 2. Create a pipeline
        p = FuzzyPipeline(name="Data Preprocessing")

        # 3. Define inputs and build the graph
        raw_data = p.input("raw_data", contract=CrispTable)
        raw_weights = p.input("raw_weights", contract=WeightVector)

        norm_data = p.add(normalizer.run, data=raw_data)
        norm_weights = p.add(weight_normalizer.run, weights=raw_weights)

        # 4. The pipeline is now a defined graph, ready to be run.
        #    It can be executed via its .run() method.
        df = pd.DataFrame(np.random.rand(3, 3))
        weights = np.array([1, 2, 3])
        results = p.run(initial_data={
            "raw_data": df,
            "raw_weights": weights
        })
    """

    # Constants to eliminate magic strings
    DEFAULT_INPUT_NAME = "default_input"
    SINGLE_OUTPUT_NAME = "output"

    def __init__(self, name: Optional[str] = None):
        """
        Initializes a new FuzzyPipeline.

        Parameters
        ----------
        name : str, optional
            An optional name for the pipeline, used for display purposes when nested.
            If not provided, a default name will be generated.
        """
        self.name = name or f"Pipeline_{uuid.uuid4().hex[:8]}"

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

    def __str__(self) -> str:
        return self.__repr__()

    def get_config(self) -> dict:
        return {}

    # --- Fluent API Methods ---

    def input(self,
              name: Optional[str] = None,
              contract: Contract = ContractAny) -> StepOutput:
        """
        Defines an input entry point for the pipeline.

        Parameters
        ----------
        name : str, optional
            The name of the input handle. If omitted for a single-input pipeline,
            it defaults to 'default_input'.
        contract : Contract, default AnyContract
            The data contract the input data is expected to conform to.

        Returns
        -------
        StepOutput
            A symbolic object representing this input.
        """
        if name is None:
            if self._input_nodes:
                raise TypeError("Input name can only be omitted for the first input of a pipeline.")
            name = self.DEFAULT_INPUT_NAME

        if name not in self._input_nodes:
            step_id = f"input_{name}_{uuid.uuid4().hex[:8]}"
            self._input_nodes[name] = step_id
            # Create metadata for the input node with a clear display name and contract.
            self._steps[step_id] = StepMetadata(
                step_id=step_id,
                display_name=f"Input: {name}",
                callable_tool=None,
                output_contracts={self.SINGLE_OUTPUT_NAME: contract}
            )

        step_id = self._input_nodes[name]
        return StepOutput(step_id=step_id, output_name=self.SINGLE_OUTPUT_NAME, pipeline=self)

    def add(self, callable_tool: Callable, **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:
        """
        Adds a new step to the pipeline using a callable component method.

        Parameters
        ----------
        callable_tool : Callable or FuzzyPipeline
            The computational unit to add. Must be a method decorated with
            ``@contract`` or another ``FuzzyPipeline`` instance.
        **kwargs :
            The inputs for the step, mapping parameter names to ``StepOutput``
            objects or providing static values.

        Returns
        -------
        StepOutput or Dict[str, StepOutput]
            Symbolic output(s) of the newly added step.
        """
        if isinstance(callable_tool, FuzzyPipeline):
            return self._add_pipeline_step(pipeline_tool=callable_tool, **kwargs)

        if not callable(callable_tool) or not hasattr(callable_tool, '_is_contract_method'):
            raise TypeError("Object for 'add()' must be a @contract decorated method or a FuzzyPipeline.")

        return self._add_step(callable_tool=callable_tool, **kwargs)

    def _add_pipeline_step(
            self,
            pipeline_tool: 'FuzzyPipeline',
            **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:

        def pipeline_runner(**prased_inputs):
            return pipeline_tool.run(initial_data=prased_inputs)

        return self._add_step(
            callable_tool=pipeline_runner,
            display_name_override=pipeline_tool.name,
            input_contracts_override={k: v.name for k, v in pipeline_tool.get_input_contracts().items()},
            output_contracts_override={k: v.name for k, v in pipeline_tool.get_output_contracts().items()},
            **kwargs
        )

    def _add_step(self,
                  callable_tool: Callable,
                  display_name_override: Optional[str] = None,
                  input_contracts_override: Optional[Dict[str, str]] = None,
                  output_contracts_override: Optional[Dict[str, str]] = None,
                  **kwargs) -> Union[StepOutput, Dict[str, StepOutput]]:
        """Internal method to add a new step from a callable."""
        # --- 1. Retrieve and Convert Metadata ---
        is_nested_pipeline = display_name_override is not None
        if is_nested_pipeline:
            display_name = display_name_override
            input_contracts = {k: Contract.get(v) for k, v in input_contracts_override.items()}
            output_contracts = {k: Contract.get(v) for k, v in output_contracts_override.items()}
        else:
            display_name = callable_tool.__qualname__
            input_contracts = {k: Contract.get(v) for k, v in getattr(callable_tool, '_contract_inputs', {}).items()}
            output_contracts = {k: Contract.get(v) for k, v in getattr(callable_tool, '_contract_outputs', {}).items()}

        step_id = f"{display_name.replace('.', '_')}_{uuid.uuid4().hex[:8]}"

        # --- 2. Separate Data Inputs and Static Parameters ---
        data_inputs: Dict[str, StepOutput] = {k: v for k, v in kwargs.items() if isinstance(v, StepOutput)}
        static_params: Dict[str, Any] = {k: v for k, v in kwargs.items() if not isinstance(v, StepOutput)}

        # --- 3. Robust Graph-Time Validation ---
        if set(input_contracts.keys()) != set(data_inputs.keys()):
            missing = set(input_contracts.keys()) - set(data_inputs.keys())
            extra = set(data_inputs.keys()) - set(input_contracts.keys())
            raise TypeError(f"Input mismatch for '{display_name}'. Missing: {missing}, Extra: {extra}")

        for arg_name, step_out in data_inputs.items():
            source_meta = self.get_step_info(step_out.step_id)
            promised_contract = source_meta.output_contracts.get(step_out.output_name)
            expected_contract = input_contracts.get(arg_name)

            if not promised_contract.is_compatible_with(expected_contract):
                raise TypeError(
                    f"Contract incompatibility for '{display_name}' on input '{arg_name}'. "
                    f"Expected compatible with '{expected_contract.name}', but received a promise for "
                    f"'{promised_contract.name}' from step '{source_meta.display_name}'."
                )

        # --- 4. Create and Store Step Metadata ---
        self._steps[step_id] = StepMetadata(
            step_id=step_id,
            display_name=display_name,
            callable_tool=callable_tool,
            static_parameters=static_params,
            dependencies=data_inputs,
            input_contracts=input_contracts,
            output_contracts=output_contracts
        )

        # --- 5. Create Symbolic Output(s) ---
        if len(output_contracts) == 1:
            output_name = list(output_contracts.keys())[0]
            return StepOutput(step_id=step_id, output_name=output_name, pipeline=self)
        else:
            return {name: StepOutput(step_id=step_id, output_name=name, pipeline=self) for name in output_contracts}

    def get_step_info(self, step_id: str) -> StepMetadata:
        """Retrieves the metadata for a given step ID."""
        if step_id not in self._steps:
            raise ValueError(f"Step with ID '{step_id}' not found in this pipeline.")
        return self._steps[step_id]

    def get_input_contracts(self) -> Dict[str, Contract]:
        """Gets the input contracts for this pipeline."""
        contracts = {}
        for name, step_id in self._input_nodes.items():
            step_meta = self.get_step_info(step_id)
            contracts[name] = step_meta.output_contracts[self.SINGLE_OUTPUT_NAME]
        return contracts

    def get_output_contracts(self) -> Dict[str, Contract]:
        terminal_steps = self._get_terminal_steps()
        if not terminal_steps:
            return {}

        output_contracts = {}
        for step in terminal_steps:
            for name, contract in step.output_contracts.items():
                # For multiple terminal nodes, create unique output names
                output_name = name if len(terminal_steps) == 1 else f"{step.display_name.replace('.', '_')}_{name}"
                output_contracts[output_name] = contract
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
        adj: Dict[str, List[str]] = {sid: [] for sid in self._steps}
        in_degree: Dict[str, int] = {sid: 0 for sid in self._steps}

        for sid, meta in self._steps.items():
            for dep in meta.dependencies.values():
                adj[dep.step_id].append(sid)
                in_degree[sid] += 1

        # Key: Kahn's algorithm for topological sorting
        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._steps):
            raise ValueError("A cycle was detected in the pipeline graph.")
        return order

    @staticmethod
    def parse_step_inputs(step_meta: StepMetadata, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parses inputs and performs runtime validation using Contract objects."""

        parsed_inputs = {}
        # 1. Resolve data inputs from state with smart unpacking
        for arg_name, step_output in step_meta.dependencies.items():
            source_result = state[step_output.step_id]
            if isinstance(source_result, dict) and step_output.output_name in source_result:
                parsed_inputs[arg_name] = source_result[step_output.output_name]
            else:
                # Assumes single direct output if not found in a dict
                parsed_inputs[arg_name] = source_result

        # 2. Add static parameters
        parsed_inputs.update(step_meta.static_parameters)

        # 3. Runtime Contract Validation
        for arg_name, data in parsed_inputs.items():
            if arg_name in step_meta.input_contracts:
                expected_contract = step_meta.input_contracts[arg_name]
                if not expected_contract.validate(data):
                    raise TypeError(
                        f"Runtime contract validation failed for '{step_meta.display_name}' on input '{arg_name}'. "
                        f"Expected '{expected_contract.name}', but received object of type {type(data).__name__}."
                    )
        return parsed_inputs

    def _format_final_output(self, final_state: 'ExecutionState') -> Any:
        """Formats the final output from the final execution state."""
        terminal_steps = self._get_terminal_steps()
        if not terminal_steps:
            return None

        final_results = {}
        for step in terminal_steps:
            result = final_state.step_results.get(step.step_id)
            if len(step.output_contracts) == 1:
                # If single output, use a descriptive name or the result directly
                final_results[step.display_name] = result
            else:
                # If multiple outputs, the result should be a dict
                if isinstance(result, dict):
                    for out_name, out_val in result.items():
                        final_results[f"{step.display_name}_{out_name}"] = out_val

        return list(final_results.values())[0] if len(final_results) == 1 else final_results

    def run(self,
            initial_data: Union[Dict[str, Any], Any],
            return_intermediate: bool = False) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """
        Executes the defined pipeline DAG with the provided initial data.

        Parameters
        ----------
        initial_data : dict[str, Any] or Any
            The starting data for the pipeline. If the pipeline has multiple
            inputs, this must be a dictionary mapping input names to data.
        return_intermediate : bool, default False
            If ``True``, returns a tuple of `(final_output, intermediate_states)`.

        Returns
        -------
        result: Any or tuple[Any, dict[str, Any]]
            The final output(s) of the pipeline. If `return_intermediate` is
            ``True``, a tuple containing the output and a dictionary of all
            intermediate step results is returned.
        """

        initial_state = self.start_execution(initial_data)
        final_state = initial_state.run_all()
        final_output = self._format_final_output(final_state)

        return (final_output, final_state.step_results) if return_intermediate else final_output

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
        """Initializes a step-by-step execution and returns the initial state."""
        initial_state_dict = {}
        if isinstance(initial_data, dict):
            if set(initial_data.keys()) != set(self._input_nodes.keys()):
                raise KeyError(f"Initial data keys {list(initial_data.keys())} do not match pipeline inputs {list(self._input_nodes.keys())}.")
            for name, data in initial_data.items():
                initial_state_dict[self._input_nodes[name]] = data
        else:
            if len(self._input_nodes) > 1:
                raise TypeError("Pipeline has multiple inputs, initial_data must be a dict.")
            if len(self._input_nodes) == 1:
                input_name = list(self._input_nodes.keys())[0]
                initial_state_dict[self._input_nodes[input_name]] = initial_data

        full_order = self._build_execution_order()
        exec_steps = [sid for sid in full_order if not self.get_step_info(sid).is_input_node]
        return ExecutionState(self, initial_state_dict, exec_steps, 0)

    def _get_terminal_steps(self) -> List[StepMetadata]:
        """Finds all terminal (leaf) step metadata objects in the graph."""
        all_ids = set(self._steps.keys())
        source_ids = set(dep.step_id for meta in self._steps.values() for dep in meta.dependencies.values())
        terminal_ids = all_ids - source_ids
        return [self._steps[tid] for tid in terminal_ids if not self._steps[tid].is_input_node]

    # --- Visualization Methods ---

    def _to_networkx(self) -> 'nx.DiGraph':
        """Converts the internal pipeline structure to a NetworkX DiGraph."""
        if not _NX_AVAILABLE:
            raise ImportError("networkx is required for visualization.")
        g = nx.DiGraph(name=self.name)
        for sid, meta in self._steps.items():
            node_type = 'input' if meta.is_input_node else ('pipeline' if isinstance(meta.callable_tool, FuzzyPipeline) else 'component')
            g.add_node(sid, label=meta.display_name, node_type=node_type)
            for dep_out in meta.dependencies.values():
                source_meta = self.get_step_info(dep_out.step_id)
                contract = source_meta.output_contracts.get(dep_out.output_name)
                g.add_edge(dep_out.step_id, sid, label=f"{dep_out.output_name}: {contract.name}")
        return g

    def visualize(self,
                  filename: Optional[str] = None,
                  output_format: str = 'png',
                  show_contracts: bool = True,
                  engine: str = 'auto',
                  styles: Optional[Dict] = None) -> Optional['Image']:
        """
        Generates a visual representation of the pipeline's DAG.

        Parameters
        ----------
        filename : str, optional
            Path to save the output image. If None, displays directly.
        output_format : str, default 'png'
            Output format for the image (e.g., 'png', 'svg').
        show_contracts : bool, default True
            If True, display data contract information on the edges.
        engine : {'auto', 'graphviz', 'matplotlib'}, default 'auto'
            The rendering engine to use.
        styles : dict, optional
            A dictionary to override the default node styles/colors for visualization.
            Keys can be 'input', 'component', 'pipeline'.

        Returns
        -------
        IPython.display.Image or None
            An Image object if in an IPython environment and no filename is given.
        """
        nx_graph = self._to_networkx()
        use_engine = engine
        if use_engine == 'auto':
            use_engine = 'graphviz' if _PYDOT_AVAILABLE else 'matplotlib'

        if use_engine == 'graphviz':
            if not _PYDOT_AVAILABLE: raise ImportError("pydot and graphviz are required for 'graphviz' engine.")
            return self._visualize_graphviz(nx_graph, filename, output_format, show_contracts, styles)
        elif use_engine == 'matplotlib':
            if not _MPL_AVAILABLE: raise ImportError("matplotlib is required for 'matplotlib' engine.")
            return self._visualize_matplotlib(nx_graph, filename, show_contracts, styles)
        else:
            raise ValueError(f"Unknown engine: '{use_engine}'. Please choose 'auto', 'graphviz', or 'matplotlib'.")

    def _visualize_graphviz(self,
                            g: 'nx.DiGraph',
                            filename: str,
                            fmt: str,
                            show_contracts: bool,
                            custom_styles: Optional[Dict]):
        # Default styles, can be overridden by custom_styles
        default_styles = {
            'input': {'shape': 'ellipse', 'fillcolor': '#E8F5E9', 'style': 'filled'},
            'component': {'shape': 'box', 'fillcolor': '#E3F2FD', 'style': 'filled,rounded'},
            'pipeline': {'shape': 'box', 'fillcolor': '#FFFDE7', 'style': 'filled,bold'}
        }
        styles = {**default_styles, **(custom_styles or {})}

        dot = pydot.Dot(graph_type='digraph', rankdir='TB', label=f'Pipeline: {self.name}', fontsize=20, labelloc='t')
        for nid, attrs in g.nodes(data=True):
            node_style = styles.get(attrs['node_type'], {})
            dot.add_node(pydot.Node(name=nid, label=attrs['label'], **node_style))
        for u, v, attrs in g.edges(data=True):
            label = attrs.get('label', '') if show_contracts else ''
            dot.add_edge(pydot.Edge(u, v, label=label, fontsize=10, fontcolor='#424242'))

        try:
            if filename:
                dot.write(filename, format=fmt)
                print(f"Pipeline visualization saved to '{filename}'")
                return None
            elif _IPYTHON_AVAILABLE:
                return Image(dot.create(format=fmt))
            else:
                temp_file = f"pipeline_view_{uuid.uuid4().hex[:6]}.{fmt}"
                dot.write(temp_file, format=fmt)
                print(f"Not in an IPython environment. Visualization saved to '{temp_file}'")
                return None
        except pydot.DotExecutableNotFoundError:
            raise ImportError("Graphviz executable not found. Please install Graphviz and ensure it is in your system's PATH.")

    def _visualize_matplotlib(self,
                              g: 'nx.DiGraph',
                              filename: str,
                              show_contracts: bool,
                              custom_styles: Optional[Dict]):
        """Renders the graph using Matplotlib."""
        # for hardcoded styles is resolved by accepting a `custom_styles` dict.
        default_colors = {'input': '#90CAF9', 'component': '#A5D6A7', 'pipeline': '#FFD54F'}
        color_map = {**default_colors, **(custom_styles or {})}
        node_colors = [color_map.get(g.nodes[n]['node_type'], '#BDBDBD') for n in g.nodes]

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(g, seed=42, k=0.8)

        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=3500, edgecolors='black')
        nx.draw_networkx_labels(g, pos, {n: attrs['label'] for n, attrs in g.nodes(data=True)}, font_size=8)
        nx.draw_networkx_edges(g, pos, arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', node_size=3500)

        if show_contracts:
            edge_labels = nx.get_edge_attributes(g, 'label')
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7, font_color='darkred')

        plt.title(f"Pipeline: {self.name}", size=15)
        plt.axis('off')
        plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches='tight')
            print(f"Pipeline visualization saved to '{filename}'")
        else:
            plt.show()

        plt.close()
        return None
