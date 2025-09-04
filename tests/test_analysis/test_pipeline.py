"""
Tests for the `FuzzyPipeline` in `axisfuzzy.analysis.pipeline`.

This file covers the core functionalities of the pipeline, including:
- Pipeline creation and validation.
- Connecting components in various configurations (1-to-1, 1-to-N, N-to-1).
- Execution flow and data passing.
- Error handling for invalid connections.
"""
import pytest
import pandas as pd

from axisfuzzy.analysis.pipeline import FuzzyPipeline
from axisfuzzy.analysis.component import AnalysisComponent
from axisfuzzy.analysis.contracts import Contract, contract
from axisfuzzy.analysis.build_in import ContractCrispTable, ContractFuzzyTable

# Simple components for testing
class ComponentA(AnalysisComponent):
    def get_config(self):
        return {}
    
    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        return data.copy()

class ComponentB(AnalysisComponent):
    def get_config(self):
        return {}
    
    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        return data.copy()

class ComponentC(AnalysisComponent):
    def get_config(self):
        return {}
    
    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        return data

class MultiInputComponent(AnalysisComponent):
    def get_config(self):
        return {}
    
    @contract
    def run(self, data1: ContractCrispTable, data2: ContractCrispTable) -> ContractCrispTable:
        # Simple concatenation for testing
        return pd.concat([data1, data2], axis=1)

class MultiOutputComponent(AnalysisComponent):
    def get_config(self):
        return {}
    
    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        # For testing, just return the same data
        # The pipeline will handle multiple outputs through different step outputs
        return data


class TestFuzzyPipelineCore:
    """Tests for the core functionalities of FuzzyPipeline."""

    def test_pipeline_creation_empty(self):
        """Test creating an empty pipeline."""
        pipeline = FuzzyPipeline()
        assert len(pipeline.steps) == 0
        assert len(pipeline.input_nodes) == 0

    def test_pipeline_creation_with_name(self):
        """Test creating a pipeline with a custom name."""
        pipeline = FuzzyPipeline(name="TestPipeline")
        assert pipeline.name == "TestPipeline"
        assert "TestPipeline" in repr(pipeline)

    def test_pipeline_representation(self):
        """Test the string representation of the pipeline."""
        pipeline = FuzzyPipeline(name="TestPipeline")
        data_input = pipeline.input("data", ContractCrispTable)
        pipeline.add(ComponentA().run, data=data_input)
        assert "TestPipeline" in repr(pipeline)
        assert "1 inputs" in repr(pipeline)
        assert "1 tasks" in repr(pipeline)

    def test_add_input(self):
        """Test adding an input to the pipeline."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        assert len(pipeline.input_nodes) == 1
        assert "data" in pipeline.input_nodes
        assert data_input.output_name == "output"


class TestPipelineBuildingAndExecution:
    """Tests for pipeline graph building and execution logic."""

    def test_simple_linear_pipeline(self, sample_dataframe):
        """Test a simple 1-to-1 linear pipeline."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        step_a = pipeline.add(ComponentA().run, data=data_input)
        step_b = pipeline.add(ComponentB().run, data=step_a)
        step_c = pipeline.add(ComponentC().run, data=step_b)
        
        # Execute the pipeline
        result = pipeline.run({"data": sample_dataframe})
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_single_input_pipeline(self, sample_dataframe):
        """Test a pipeline with single input (no dict required)."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input(contract=ContractCrispTable)  # Specify correct contract
        step_a = pipeline.add(ComponentA().run, data=data_input)
        
        # Execute with single input
        result = pipeline.run(sample_dataframe)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_many_to_one_join(self, sample_dataframe):
        """Test a pipeline that joins multiple sources into one component."""
        pipeline = FuzzyPipeline()
        data_input1 = pipeline.input("data1", ContractCrispTable)
        data_input2 = pipeline.input("data2", ContractCrispTable)
        
        # Join two separate inputs
        join_result = pipeline.add(MultiInputComponent().run, 
                                 data1=data_input1, 
                                 data2=data_input2)

        result = pipeline.run({"data1": sample_dataframe, "data2": sample_dataframe})
        assert isinstance(result, pd.DataFrame)
        # Result should have double the columns due to concatenation
        assert len(result.columns) == len(sample_dataframe.columns) * 2

    def test_invalid_connection_contract_mismatch(self):
        """Test that connecting incompatible components raises a TypeError."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        
        # Create a component that expects fuzzy data but we're giving it crisp data
        class FuzzyOnlyComponent(AnalysisComponent):
            def get_config(self):
                return {}
            
            @contract
            def run(self, data: ContractFuzzyTable) -> ContractFuzzyTable:
                return data
        
        with pytest.raises(TypeError, match="Contract incompatibility"):
            pipeline.add(FuzzyOnlyComponent().run, data=data_input)

    def test_invalid_input_mismatch(self):
        """Test that providing wrong inputs raises a TypeError."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        
        with pytest.raises(TypeError, match="Input mismatch"):
            # ComponentA expects 'data' parameter but we provide 'wrong_param'
            pipeline.add(ComponentA().run, wrong_param=data_input)

    def test_execution_with_wrong_initial_data(self, sample_dataframe):
        """Test that executing with wrong initial data raises an error."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        pipeline.add(ComponentA().run, data=data_input)
        
        with pytest.raises(KeyError, match="Initial data keys.*do not match pipeline inputs"):
            # Wrong key name
            pipeline.run({"wrong_key": sample_dataframe})


class TestDataPassingAndState:
    """Tests for data passing between components and state management."""

    def test_data_passing_between_components(self, sample_dataframe):
        """Test that data is correctly passed between components."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        step_a = pipeline.add(ComponentA().run, data=data_input)
        step_b = pipeline.add(ComponentB().run, data=step_a)
        
        result = pipeline.run({"data": sample_dataframe})
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_pipeline_state_tracking(self, sample_dataframe):
        """Test that pipeline tracks execution state correctly."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        step_a = pipeline.add(ComponentA().run, data=data_input)
        step_b = pipeline.add(ComponentB().run, data=step_a)
        
        # Execute with return_intermediate=True to get execution state
        result, intermediate_results = pipeline.run({"data": sample_dataframe}, return_intermediate=True)
        
        # Check that intermediate results are tracked
        assert isinstance(intermediate_results, dict)
        assert len(intermediate_results) >= 2  # Should have results for each step
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_step_by_step_execution(self, sample_dataframe):
        """Test step-by-step execution mode."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        step_a = pipeline.add(ComponentA().run, data=data_input)
        step_b = pipeline.add(ComponentB().run, data=step_a)
        
        # Start step-by-step execution
        execution = pipeline.step_by_step({"data": sample_dataframe})
        
        # Execute steps one by one
        step_results = list(execution)
        
        # Should have at least 2 steps
        assert len(step_results) >= 2
        
        # Each step should be a dict with expected keys
        for step_result in step_results:
            assert isinstance(step_result, dict)
            assert 'step_index' in step_result
            assert 'total_steps' in step_result
            assert 'step_name' in step_result
            assert 'result' in step_result
            assert 'execution_time' in step_result


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_component_execution_error(self):
        """Test that component execution errors are properly handled."""
        @contract
        def error_component(data: ContractCrispTable) -> ContractCrispTable:
            raise ValueError("Simulated component error")
        
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        pipeline.add(error_component, data=data_input)
        
        with pytest.raises(ValueError, match="Simulated component error"):
            pipeline.run({"data": pd.DataFrame()})

    def test_invalid_input_data_type(self, sample_dataframe):
        """Test that invalid input data types are rejected."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        pipeline.add(ComponentA().run, data=data_input)
        
        # Try to pass invalid data type (string instead of DataFrame)
        with pytest.raises((TypeError, ValueError)):
            pipeline.run({"data": "invalid_data"})

    def test_missing_required_inputs(self):
        """Test that missing required inputs are detected."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        pipeline.add(ComponentA().run, data=data_input)
        
        # Try to run without providing required input
        with pytest.raises(KeyError, match="Initial data keys.*do not match pipeline inputs"):
            pipeline.run({})  # Missing 'data' input

    def test_empty_pipeline_execution(self):
        """Test that executing an empty pipeline raises an error."""
        pipeline = FuzzyPipeline()
        
        # Empty pipeline should raise an error when trying to run
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            pipeline.run({})


class TestPipelineInformation:
    """Tests for pipeline information and introspection."""

    def test_pipeline_step_count(self, sample_dataframe):
        """Test that pipeline correctly reports step count."""
        pipeline = FuzzyPipeline()
        
        # Empty pipeline should have 0 steps
        assert len(pipeline.steps) == 0
        
        # Add input - this creates 1 step
        data_input = pipeline.input("data", ContractCrispTable)
        assert len(pipeline.steps) == 1
        
        # Add components
        step_a = pipeline.add(ComponentA().run, data=data_input)
        assert len(pipeline.steps) == 2
        
        step_b = pipeline.add(ComponentB().run, data=step_a)
        assert len(pipeline.steps) == 3

    def test_pipeline_input_information(self):
        """Test that pipeline correctly reports input information."""
        pipeline = FuzzyPipeline()
        
        assert len(pipeline.input_nodes) == 0
        
        data_input = pipeline.input("data", ContractCrispTable)
        assert len(pipeline.input_nodes) == 1
        assert "data" in pipeline.input_nodes
        
        other_input = pipeline.input("other", ContractFuzzyTable)
        assert len(pipeline.input_nodes) == 2
        assert "other" in pipeline.input_nodes

    def test_pipeline_execution_history(self, sample_dataframe):
        """Test that pipeline tracks execution history."""
        pipeline = FuzzyPipeline()
        data_input = pipeline.input("data", ContractCrispTable)
        step_a = pipeline.add(ComponentA().run, data=data_input)
        step_b = pipeline.add(ComponentB().run, data=step_a)
        
        # Execute with intermediate results
        result, intermediate_results = pipeline.run({"data": sample_dataframe}, return_intermediate=True)
        
        # Check that execution history is tracked
        assert isinstance(intermediate_results, dict)
        assert len(intermediate_results) >= 2  # Should have results for each step
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_dataframe)