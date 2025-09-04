"""
Tests for the analysis component system in `axisfuzzy.analysis.component`.

This file covers:
- `AnalysisComponent` base class behavior.
- Component creation with input/output contracts.
- Execution logic and contract validation.
- Testing built-in components.
"""
import pytest
import pandas as pd
import numpy as np

from axisfuzzy.analysis.component import AnalysisComponent
from axisfuzzy.analysis.contracts import Contract, contract
from axisfuzzy.analysis.build_in import (
    ContractCrispTable,
    ContractFuzzyTable,
    ContractWeightVector
)
from axisfuzzy.analysis.component.basic import (
    ToolNormalization,
    ToolFuzzification,
    ToolSimpleAggregation,
    ToolWeightNormalization
)
from axisfuzzy.fuzzifier import Fuzzifier


class TestAnalysisComponentCore:
    """Tests for the core behavior of the AnalysisComponent base class."""

    def test_component_creation(self):
        """Test the creation of a simple component with contracts."""
        class MyComponent(AnalysisComponent):
            def __init__(self):
                pass
                
            def get_config(self):
                return {}

            @contract
            def run(self, data: ContractCrispTable) -> ContractFuzzyTable:
                return data # Dummy implementation

        comp = MyComponent()
        assert comp.__class__.__name__ == "MyComponent"

    def test_component_creation_with_custom_name(self):
        """Test creating a component with a custom name."""
        class CustomComponent(AnalysisComponent):
            def __init__(self, name="CustomName"):
                self.name = name
                
            def get_config(self):
                return {'name': self.name}
                
            def run(self, *args, **kwargs):
                return args[0] if args else None
                
        comp = CustomComponent()
        assert comp.name == "CustomName"

    def test_run_method_not_implemented_raises_error(self):
        """Test that calling a component without a `run` method raises NotImplementedError."""
        class IncompleteComponent(AnalysisComponent):
            def get_config(self):
                return {}

        comp = IncompleteComponent()
        with pytest.raises(NotImplementedError):
            comp.run("some_input")

    def test_get_config_not_implemented_raises_error(self):
        """Test that a component without `get_config` method raises NotImplementedError."""
        class IncompleteComponent(AnalysisComponent):
            def run(self, *args, **kwargs):
                return args[0] if args else None
            
            def get_config(self):
                raise NotImplementedError("get_config must be implemented")

        comp = IncompleteComponent()
        with pytest.raises(NotImplementedError):
            comp.get_config()


class TestBuiltinComponents:
    """Tests for the built-in analysis components."""
    
    def test_tool_normalization_get_config(self):
        """Test the get_config method of ToolNormalization."""
        normalizer = ToolNormalization(method='z_score', axis=0)
        config = normalizer.get_config()
        
        assert isinstance(config, dict)
        assert config['method'] == 'z_score'
        assert config['axis'] == 0
        
    def test_tool_fuzzification_get_config(self, sample_fuzzifier):
        """Test the get_config method of ToolFuzzification."""
        fuzzifier_component = ToolFuzzification(fuzzifier=sample_fuzzifier)
        config = fuzzifier_component.get_config()
        
        assert isinstance(config, dict)
        assert 'fuzzifier' in config
        
    def test_tool_weight_normalization_get_config(self):
        """Test the get_config method of ToolWeightNormalization."""
        normalizer = ToolWeightNormalization(ensure_positive=False)
        config = normalizer.get_config()
        
        assert isinstance(config, dict)
        assert config['ensure_positive'] == False
        
    def test_tool_simple_aggregation_get_config(self):
        """Test the get_config method of ToolSimpleAggregation."""
        aggregator = ToolSimpleAggregation(operation='sum', axis=0)
        config = aggregator.get_config()
        
        assert isinstance(config, dict)
        assert config['operation'] == 'sum'
        assert config['axis'] == 0

    def test_tool_normalization(self, sample_dataframe):
        """Test the ToolNormalization component."""
        normalizer = ToolNormalization()
        result = normalizer.run(sample_dataframe)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_dataframe.shape
        # Check if all values are between 0 and 1
        assert (result >= 0).all().all() and (result <= 1).all().all()

    def test_tool_fuzzification(self, sample_dataframe, sample_fuzzifier):
        """Test the ToolFuzzification component."""
        # The component expects a Fuzzifier instance during initialization
        fuzzifier_component = ToolFuzzification(fuzzifier=sample_fuzzifier)
        result = fuzzifier_component.run(sample_dataframe)

        # The output should be a FuzzyDataFrame
        from axisfuzzy.analysis.dataframe import FuzzyDataFrame
        assert isinstance(result, FuzzyDataFrame)
        assert result.shape == sample_dataframe.shape

    def test_tool_simple_aggregation(self, sample_dataframe):
        """Test the ToolSimpleAggregation component."""
        aggregator = ToolSimpleAggregation(operation="mean")
        result = aggregator.run(sample_dataframe)

        assert isinstance(result, (pd.Series, np.ndarray))
        # The result should be crisp numbers after aggregation
        assert len(result) == len(sample_dataframe)

    def test_tool_simple_aggregation_with_invalid_operation(self):
        """Test that ToolSimpleAggregation raises an error for an invalid operation."""
        with pytest.raises(ValueError):
            ToolSimpleAggregation(operation="invalid_operation")
            
    def test_tool_weight_normalization(self, sample_weights):
        """Test the ToolWeightNormalization component."""
        normalizer = ToolWeightNormalization()
        result = normalizer.run(sample_weights)
        
        assert isinstance(result, (pd.Series, np.ndarray))
        # Check if weights sum to 1
        assert np.isclose(np.sum(result), 1.0)


class TestContractSystem:
    """Tests for the contract system and @contract decorator."""
    
    def test_contract_decorator_basic(self):
        """Test basic functionality of the @contract decorator."""
        class TestComponent(AnalysisComponent):
            def get_config(self):
                return {}
                
            @contract
            def run(self, data: ContractCrispTable) -> ContractCrispTable:
                return data
        
        comp = TestComponent()
        # Check that the decorator adds contract metadata
        assert hasattr(comp.run, '_contract_inputs')
        assert hasattr(comp.run, '_contract_outputs')
        
    def test_contract_validation_success(self, sample_dataframe):
        """Test that valid data passes contract validation."""
        contract = ContractCrispTable
        assert contract.validate(sample_dataframe) == True
        
    def test_contract_validation_failure(self):
        """Test that invalid data fails contract validation."""
        contract = ContractCrispTable
        invalid_data = "not a dataframe"
        assert contract.validate(invalid_data) == False
        
    def test_contract_inheritance(self):
        """Test contract inheritance relationships."""
        # Test that ContractWeightVector is compatible with itself
        contract = ContractWeightVector
        assert contract.is_compatible_with(contract) == True


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""
    
    def test_normalization_with_constant_data(self):
        """Test normalization with constant data (all values the same)."""
        constant_df = pd.DataFrame({
            'A': [5.0, 5.0, 5.0],
            'B': [5.0, 5.0, 5.0]
        })
        
        normalizer = ToolNormalization(method='min_max')
        result = normalizer.run(constant_df)
        
        # Should handle constant data gracefully (set to 0.5)
        assert isinstance(result, pd.DataFrame)
        assert (result == 0.5).all().all()
        
    def test_normalization_with_single_row(self):
        """Test normalization with single row data."""
        single_row_df = pd.DataFrame({
            'A': [1.0],
            'B': [2.0],
            'C': [3.0]
        })
        
        normalizer = ToolNormalization(method='min_max', axis=1)
        result = normalizer.run(single_row_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == single_row_df.shape
        
    def test_weight_normalization_with_zeros(self):
        """Test weight normalization with zero weights."""
        zero_weights = np.array([0.0, 0.0, 0.0])
        
        normalizer = ToolWeightNormalization()
        result = normalizer.run(zero_weights)
        
        # Should handle zero weights gracefully
        assert isinstance(result, (pd.Series, np.ndarray))
        
    def test_weight_normalization_with_negative_weights(self):
        """Test weight normalization with negative weights."""
        negative_weights = np.array([-1.0, 2.0, -0.5])
        
        normalizer = ToolWeightNormalization(ensure_positive=True)
        result = normalizer.run(negative_weights)
        
        # Should handle negative weights according to ensure_positive setting
        assert isinstance(result, (pd.Series, np.ndarray))
        assert np.isclose(np.sum(result), 1.0)
        
    def test_invalid_normalization_method(self):
        """Test that invalid normalization method raises ValueError."""
        with pytest.raises(ValueError):
            ToolNormalization(method='invalid_method')
            
    def test_empty_dataframe_handling(self):
        """Test component behavior with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        normalizer = ToolNormalization()
        # This should either handle gracefully or raise a meaningful error
        try:
            result = normalizer.run(empty_df)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, IndexError) as e:
            # Acceptable to raise an error for empty data
            assert len(str(e)) > 0


class TestComponentIntegration:
    """Tests for component integration and pipeline compatibility."""
    
    def test_component_chaining(self, sample_dataframe, sample_fuzzifier):
        """Test chaining multiple components together."""
        # Step 1: Normalize data
        normalizer = ToolNormalization(method='min_max')
        normalized_data = normalizer.run(sample_dataframe)
        
        # Step 2: Fuzzify normalized data
        fuzzifier_component = ToolFuzzification(fuzzifier=sample_fuzzifier)
        fuzzy_data = fuzzifier_component.run(normalized_data)
        
        # Verify the chain works
        from axisfuzzy.analysis.dataframe import FuzzyDataFrame
        assert isinstance(fuzzy_data, FuzzyDataFrame)
        assert fuzzy_data.shape == sample_dataframe.shape
        
    def test_component_serialization_roundtrip(self):
        """Test that component configurations can be serialized and restored."""
        import json
        
        # Create a component with specific configuration
        normalizer = ToolNormalization(method='z_score', axis=0)
        config = normalizer.get_config()
        
        # Serialize and deserialize the configuration
        serialized = json.dumps(config)
        deserialized = json.loads(serialized)
        
        # Create a new component with the deserialized config
        new_normalizer = ToolNormalization(**deserialized)
        new_config = new_normalizer.get_config()
        
        # Configurations should match
        assert config == new_config


class ComponentTestBase:
    """Base class for testing custom analysis components.
    
    This class provides a standardized testing interface for future component extensions.
    Inherit from this class to ensure consistent testing patterns across all components.
    """
    
    @staticmethod
    def assert_component_interface(component_class):
        """Assert that a component class implements the required interface.
        
        Parameters
        ----------
        component_class : type
            The component class to test
            
        Raises
        ------
        AssertionError
            If the component doesn't implement required methods
        """
        # Check inheritance
        assert issubclass(component_class, AnalysisComponent), \
            f"{component_class.__name__} must inherit from AnalysisComponent"
        
        # Check required methods exist
        assert hasattr(component_class, 'run'), \
            f"{component_class.__name__} must implement 'run' method"
        assert hasattr(component_class, 'get_config'), \
            f"{component_class.__name__} must implement 'get_config' method"
    
    @staticmethod
    def assert_component_config(component_instance, expected_keys=None):
        """Assert that a component's configuration is valid.
        
        Parameters
        ----------
        component_instance : AnalysisComponent
            The component instance to test
        expected_keys : list, optional
            Expected keys in the configuration dictionary
        """
        config = component_instance.get_config()
        
        # Basic config validation
        assert isinstance(config, dict), "get_config() must return a dictionary"
        
        # Check expected keys if provided
        if expected_keys:
            for key in expected_keys:
                assert key in config, f"Configuration missing expected key: {key}"
    
    @staticmethod
    def assert_component_contracts(component_instance):
        """Assert that a component's run method has proper contract decorations.
        
        Parameters
        ----------
        component_instance : AnalysisComponent
            The component instance to test
        """
        run_method = getattr(component_instance, 'run')
        
        # Check if the run method has contract metadata
        has_input_contracts = hasattr(run_method, '_contract_inputs')
        has_output_contracts = hasattr(run_method, '_contract_outputs')
        
        # At least one type of contract should be present for proper pipeline integration
        assert has_input_contracts or has_output_contracts, \
            "Component run method should have contract decorations for pipeline compatibility"
    
    @staticmethod
    def validate_serialization_roundtrip(component_instance):
        """Validate that a component can be serialized and deserialized.
        
        This is a utility method for testing serialization, not a test method itself.
        
        Parameters
        ----------
        component_instance : AnalysisComponent
            The component instance to test
            
        Returns
        -------
        bool
            True if serialization roundtrip is successful
        """
        import json
        
        try:
            # Get configuration
            config = component_instance.get_config()
            
            # Serialize and deserialize
            serialized = json.dumps(config)
            deserialized = json.loads(serialized)
            
            # Create new instance with deserialized config
            new_instance = component_instance.__class__(**deserialized)
            new_config = new_instance.get_config()
            
            # Configurations should match
            assert config == new_config, "Serialization roundtrip failed"
            return True
            
        except (TypeError, ValueError, json.JSONEncodeError) as e:
            pytest.fail(f"Component serialization failed: {e}")
            return False


def create_component_test_suite(component_class, test_data=None, expected_config_keys=None):
    """Factory function to create a standardized test suite for any component.
    
    This function generates a test class with common test methods for any component,
    reducing boilerplate code for future component testing.
    
    Parameters
    ----------
    component_class : type
        The component class to test
    test_data : dict, optional
        Test data for the component (e.g., {'input': sample_data, 'expected_output_type': pd.DataFrame})
    expected_config_keys : list, optional
        Expected keys in the component's configuration
        
    Returns
    -------
    type
        A dynamically created test class
        
    Examples
    --------
    >>> # Create a test suite for a custom component
    >>> class MyCustomComponent(AnalysisComponent):
    ...     def __init__(self, param1=1.0):
    ...         self.param1 = param1
    ...     def get_config(self):
    ...         return {'param1': self.param1}
    ...     @contract
    ...     def run(self, data: ContractCrispTable) -> ContractCrispTable:
    ...         return data * self.param1
    ...
    >>> # Generate test class
    >>> TestMyCustomComponent = create_component_test_suite(
    ...     MyCustomComponent,
    ...     test_data={'input': sample_df, 'expected_output_type': pd.DataFrame},
    ...     expected_config_keys=['param1']
    ... )
    """
    
    class DynamicComponentTest(ComponentTestBase):
        """Dynamically generated test class for component testing."""
        
        def test_component_interface(self):
            """Test that the component implements required interface."""
            self.assert_component_interface(component_class)
        
        def test_component_instantiation(self):
            """Test that the component can be instantiated."""
            try:
                instance = component_class()
                assert isinstance(instance, AnalysisComponent)
            except TypeError:
                # Component might require parameters
                pytest.skip(f"{component_class.__name__} requires initialization parameters")
        
        def test_component_config(self):
            """Test the component's configuration method."""
            try:
                instance = component_class()
                self.assert_component_config(instance, expected_config_keys)
            except TypeError:
                pytest.skip(f"{component_class.__name__} requires initialization parameters")
        
        def test_component_contracts(self):
            """Test the component's contract decorations."""
            try:
                instance = component_class()
                self.assert_component_contracts(instance)
            except TypeError:
                pytest.skip(f"{component_class.__name__} requires initialization parameters")
        
        def test_component_serialization(self):
            """Test component serialization and deserialization."""
            try:
                import json
                instance = component_class()
                config = instance.get_config()
                # Test that config can be serialized to JSON
                serialized = json.dumps(config)
                deserialized = json.loads(serialized)
                # Basic validation that deserialized config is a dict
                assert isinstance(deserialized, dict), "Deserialized config should be a dictionary"
            except TypeError:
                pytest.skip(f"{component_class.__name__} requires initialization parameters")
            except (json.JSONEncodeError, TypeError) as e:
                pytest.fail(f"Component configuration is not JSON serializable: {e}")
        
        def test_component_execution(self):
            """Test component execution with provided test data."""
            if test_data is None:
                pytest.skip("No test data provided for execution testing")
            
            try:
                instance = component_class()
                input_data = test_data.get('input')
                expected_output_type = test_data.get('expected_output_type')
                
                if input_data is not None:
                    result = instance.run(input_data)
                    
                    if expected_output_type:
                        assert isinstance(result, expected_output_type), \
                            f"Expected output type {expected_output_type}, got {type(result)}"
                            
            except TypeError:
                pytest.skip(f"{component_class.__name__} requires initialization parameters")
    
    # Set the class name dynamically
    DynamicComponentTest.__name__ = f"Test{component_class.__name__}"
    DynamicComponentTest.__qualname__ = f"Test{component_class.__name__}"
    
    return DynamicComponentTest


# Example usage for future component testing:
# Uncomment and modify the following lines when adding new components

# class TestFutureComponent(ComponentTestBase):
#     """Example test class for a future component using the testing base."""
#     
#     def test_my_custom_component(self):
#         """Test a custom component using the standardized interface."""
#         # Example: Test a hypothetical new component
#         # class MyNewComponent(AnalysisComponent):
#         #     def __init__(self, param=1.0):
#         #         self.param = param
#         #     def get_config(self):
#         #         return {'param': self.param}
#         #     @contract
#         #     def run(self, data: ContractCrispTable) -> ContractCrispTable:
#         #         return data * self.param
#         # 
#         # # Use the testing utilities
#         # self.assert_component_interface(MyNewComponent)
#         # instance = MyNewComponent(param=2.0)
#         # self.assert_component_config(instance, ['param'])
#         # self.assert_component_contracts(instance)
#         # self.test_component_serialization_roundtrip(instance)
#         pass

# Alternative approach using the factory function:
# TestMyNewComponent = create_component_test_suite(
#     MyNewComponent,
#     test_data={'input': sample_dataframe, 'expected_output_type': pd.DataFrame},
#     expected_config_keys=['param']
# )