#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/25 11:59
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
import importlib
import json
import uuid
from pathlib import Path

from typing import List, Optional, Any, Union

from ..component import AnalysisComponent
from ..pipeline import FuzzyPipeline, FuzzyPipelineIterator


class Sequential:
    """
    A simplified, sequential pipeline builder for linear analysis workflows.

    This class provides a more straightforward, list-based way to define a
    pipeline, acting as a high-level wrapper around the more flexible
    FuzzyPipeline. It's designed for users who want to quickly build
    linear analysis flows without dealing with the complexity of
    graph construction.

    Parameters
    ----------
    steps : List[AnalysisComponent]
        A list of analysis component instances to be executed in order.
        Each component must inherit from AnalysisComponent and have a
        @contract decorated run method.
    name : str, optional
        An optional name for the sequential flow. If not provided,
        a default name will be generated.

    Attributes
    ----------
    name : str
        The name of the sequential model.
    steps : List[AnalysisComponent]
        The list of analysis components in execution order.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.analysis.app import Sequential
        from axisfuzzy.analysis.component.basic import (
            ToolNormalization, ToolFuzzification
        )
        from axisfuzzy.fuzzifier import Fuzzifier

        # Create components
        normalizer = ToolNormalization(method='min_max')
        fuzzifier = ToolFuzzification(fuzzifier=Fuzzifier(...))

        # Build sequential model
        model = Sequential([
            normalizer,
            fuzzifier
        ], name="MyAnalysisFlow")

        # Run the model
        result = model.run(initial_data=my_dataframe)

        # Visualize the workflow
        model.visualize()
    """
    def __init__(self,
                 steps: List[AnalysisComponent],
                 name: Optional[str] = None):

        if not steps:
            raise ValueError("Steps list cannot be empty.")

        # Validate that all steps are AnalysisComponent instances
        for i, step in enumerate(steps):
            if not isinstance(step, AnalysisComponent):
                raise TypeError(
                    f"Step {i} must be an instance of AnalysisComponent, "
                    f"got {type(step).__name__}."
                )

            # Check that the step has a contract-decorated run method
            if not hasattr(step.run, '_is_contract_method'):
                raise TypeError(
                    f"Step {i} ({type(step).__name__}) must have a "
                    f"@contract decorated run method."
                )

        self.name = name or f"Sequential_{uuid.uuid4().hex[:8]}"
        self.steps = steps
        self._pipeline = None  # Lazy initialization

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the Sequential model."""
        step_names = [type(step).__name__ for step in self.steps]
        return f"<Sequential '{self.name}' with {len(self.steps)} steps: {step_names}>"

    def __str__(self) -> str:
        """Provides a user-friendly string representation."""
        return self.__repr__()

    @property
    def pipeline(self) -> FuzzyPipeline:
        """
        Returns the underlying FuzzyPipeline instance.

        The pipeline is built lazily on first access.

        Returns
        -------
        FuzzyPipeline
            The underlying pipeline that executes the sequential workflow.
        """
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def _build_pipeline(self) -> FuzzyPipeline:
        """
        Internal method to construct the FuzzyPipeline from the steps list.

        This method creates a linear chain where each step's single output
        is connected to the next step's single input. It now dynamically
        infers the initial input contract from the first step in the sequence.

        Returns
        -------
        FuzzyPipeline
            A configured pipeline ready for execution.

        Raises
        ------
        ValueError
            If any step has multiple inputs or outputs, or if the first
            step has no contract-defined inputs.
        """
        if not self.steps:
            raise ValueError("Cannot build pipeline: steps list is empty.")

        p = FuzzyPipeline(name=f"{self.name}_Pipeline")

        # Enhanced first step analysis
        first_component = self.steps[0]
        first_run_method = first_component.run
        first_input_contracts = getattr(first_run_method, '_contract_inputs', {})

        if not first_input_contracts:
            raise ValueError(
                f"The first component '{first_component.__class__.__name__}' has no "
                f"contract-defined inputs. Components must use @contract decorator."
            )
        if len(first_input_contracts) > 1:
            available_inputs = list(first_input_contracts.keys())
            raise ValueError(
                f"The first component '{first_component.__class__.__name__}' has "
                f"multiple inputs {available_inputs}, which is not supported as the "
                f"starting point of a Sequential model. Use FuzzyPipeline for "
                f"complex multi-input scenarios."
            )

        # Create input with proper contract
        first_input_name = list(first_input_contracts.keys())[0]
        first_input_contract = first_input_contracts[first_input_name]

        from ..contracts import Contract
        inferred_contract = Contract.get(first_input_contract)

        current_output = p.input("initial_data", contract=inferred_contract)

        # Enhanced step-by-step building with compatibility checking
        for i, component in enumerate(self.steps):
            run_method = component.run
            component_name = component.__class__.__name__

            input_contracts = getattr(run_method, '_contract_inputs', {})
            output_contracts = getattr(run_method, '_contract_outputs', {})

            # Validate component has exactly one input (Sequential requirement)
            if len(input_contracts) != 1:
                raise ValueError(
                    f"Sequential step {i} ({component_name}) must have exactly "
                    f"one input, but has {len(input_contracts)}: {list(input_contracts.keys())}. "
                    f"Multi-input components require FuzzyPipeline."
                )

            # Handle multi-output components gracefully
            if len(output_contracts) > 1:
                print(f"⚠️  Warning: Step {i} ({component_name}) has multiple outputs "
                      f"{list(output_contracts.keys())}. Sequential will use the first output.")

            input_param_name = list(input_contracts.keys())[0]

            try:
                current_output = p.add(run_method, **{input_param_name: current_output})
            except Exception as e:
                raise ValueError(
                    f"Failed to connect step {i} ({component_name}): {str(e)}. "
                    f"Check component contracts and compatibility."
                ) from e

        self._pipeline_built_successfully = True
        return p

    def run(self, initial_data: Any, return_intermediate: bool = False) -> Any:
        """
        Executes the sequential analysis workflow.

        This method is a convenient wrapper around the underlying
        `FuzzyPipeline.run()` method.

        Parameters
        ----------
        initial_data : Any
            The initial data to be fed into the first step of the sequence.
            This typically is a pandas DataFrame.
        return_intermediate : bool, default False
            If ``True``, returns a tuple of `(final_output, intermediate_states)`.

        Returns
        -------
        Any or tuple[Any, dict]
            The final output of the last component in the sequence. If
            `return_intermediate` is True, returns a tuple containing the
            final output and a dictionary of all intermediate step results.
        """
        return self.pipeline.run(
            initial_data,
            return_intermediate=return_intermediate)

    def visualize(self, **kwargs):
        """
        Visualizes the underlying pipeline structure.

        This method delegates to the underlying FuzzyPipeline's visualization
        capabilities, allowing users to see the constructed DAG.

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed directly to FuzzyPipeline.visualize().
            See FuzzyPipeline.visualize() documentation for available options.

        Returns
        -------
        IPython.display.Image or None
            Visualization result, depending on the environment and parameters.
        """
        return self.pipeline.visualize(**kwargs)

    def step_by_step(self, initial_data: Any) -> 'FuzzyPipelineIterator':
        """
        Create an iterator for step-by-step execution of this sequential model.

        Parameters
        ----------
        initial_data : Any
            The initial data for the sequential execution.

        Returns
        -------
        FuzzyPipelineIterator
            An iterator that yields step-by-step execution results.
        """
        return self.pipeline.step_by_step(initial_data)

    def summary(self, line_length: int = 70) -> None:
        """
        Prints a summary of the sequential model's architecture.

        This method provides a table-like representation of the model,
        showing each layer, its type, and the input/output contracts
        it adheres to. This is highly useful for debugging and understanding
        the data flow within the model.

        Parameters
        ----------
        line_length : int, default 100
            The total width of the summary lines.

        Examples
        --------
        .. code-block:: python

            model.summary()
            # Expected output:
            # Model: "MyAnalysisFlow_Pipeline"
            # =================================================================
            # Layer (type)          Input Contracts      Output Contracts
            # -----------------------------------------------------------------
            # Normalization         ContractCrispTable   ContractCrispTable
            # Fuzzification         ContractCrispTable   ContractFuzzyTable
            # =================================================================
            # Total layers: 2
            # =================================================================
        """
        print("=" * line_length)
        print(f"Model: \"{self.name}\"")
        print("=" * line_length)

        # Table headers
        print(f"{'Layer (type)':<20} {'Input Contract':<20} {'Output Contract':<20}")
        print("-" * line_length)

        # Iterate through each step and extract contract information
        for i, component in enumerate(self.steps):
            layer_name = component.__class__.__name__
            run_method = component.run

            # Extract contract information
            input_contracts = getattr(run_method, '_contract_inputs', {})
            output_contracts = getattr(run_method, '_contract_outputs', {})

            # Format contract names (remove 'Contract' prefix for readability)
            input_contract_name = 'None'
            if input_contracts:
                first_input = list(input_contracts.values())[0]
                input_contract_name = first_input.replace('Contract', '')

            output_contract_name = 'None'
            if output_contracts:
                first_output = list(output_contracts.values())[0]
                output_contract_name = first_output.replace('Contract', '')

            print(f"{layer_name:<20} {input_contract_name:<20} {output_contract_name:<20}")

        print("=" * line_length)
        print(f"Total layers: {len(self.steps)}")
        print("=" * line_length)

    def save(self, filepath: str):
        """
        Saves the model's architecture and configuration to a file.

        This method serializes the model's configuration into a JSON file,
        allowing it to be reloaded later.

        Parameters
        ----------
        filepath : str
            Path to the file where the model will be saved.
        """
        model_config = {
            'name': self.name,
            'steps': []
        }
        for step in self.steps:
            step_config = {
                'class_path': f"{step.__class__.__module__}.{step.__class__.__name__}",
                'config': step.get_config()
            }
            model_config['steps'].append(step_config)

        with open(filepath, 'w') as f:
            json.dump(model_config, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Sequential':
        """
        Loads a model from a configuration file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the model's configuration file.

        Returns
        -------
        Sequential
            A new instance of the Sequential model.
        """
        # if not filepath.exists():
        #     raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'r') as f:
            model_config = json.load(f)

        model_name = model_config['name']
        loaded_steps = []

        for step_config in model_config['steps']:
            class_path = step_config['class_path']
            config = step_config['config']

            # Dynamically import the class
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)

            # Instantiate the component with its config
            loaded_steps.append(component_class(**config))

        return cls(steps=loaded_steps, name=model_name)
