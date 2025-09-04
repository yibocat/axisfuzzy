"""Comprehensive extension interface example for AxisFuzzy Analysis module.

This file demonstrates how to use both the component testing framework and the
contract testing framework to test new analysis components and contracts. It serves
as a complete guide for extending the AxisFuzzy analysis system with custom
components and contracts.

The file covers:
1. Testing custom contracts using ContractTestBase and create_contract_test_suite
2. Testing custom components using ComponentTestBase and create_component_test_suite
3. Integration testing between custom contracts and components
4. Best practices for extensible testing
"""
import pytest
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Union, List

from axisfuzzy.analysis.component.base import AnalysisComponent
from axisfuzzy.analysis.contracts import contract
from axisfuzzy.analysis.build_in import ContractCrispTable
from axisfuzzy.analysis.contracts.base import Contract

# Import the testing utilities
from tests.test_analysis.test_components import ComponentTestBase, create_component_test_suite
from tests.test_analysis.test_contracts import ContractTestBase, create_contract_test_suite


# ============================================================================
# PART 1: Custom Contract Examples
# ============================================================================

def create_time_series_validator(required_columns: List[str] = None, min_rows: int = 1):
    """Create a validator function for time series data.
    
    Parameters
    ----------
    required_columns : List[str], optional
        List of required column names
    min_rows : int, default=1
        Minimum number of rows required
        
    Returns
    -------
    Callable[[Any], bool]
        Validator function
    """
    required_columns = required_columns or ['timestamp', 'value']
    
    def validator(data: Any) -> bool:
        """Validate that data conforms to time series contract."""
        if not isinstance(data, pd.DataFrame):
            return False
            
        # Check minimum rows
        if len(data) < min_rows:
            return False
            
        # Check required columns
        for col in required_columns:
            if col not in data.columns:
                return False
                
        # Check timestamp column if present
        if 'timestamp' in data.columns:
            try:
                # Convert to datetime and check for any NaT values
                # Suppress datetime format inference warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    datetime_series = pd.to_datetime(data['timestamp'], errors='coerce')
                if datetime_series.isna().any():
                    return False
            except (ValueError, TypeError):
                return False
                
        return True
    
    return validator


# Create example time series contract
ExampleTimeSeriesContract = Contract(
    name="example_time_series",
    validator=create_time_series_validator(['timestamp', 'value'], min_rows=1)
)


def create_numeric_validator(min_value: float = None, max_value: float = None):
    """Create a validator function for numeric data.
    
    Parameters
    ----------
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value
        
    Returns
    -------
    Callable[[Any], bool]
        Validator function
    """
    def validator(data: Any) -> bool:
        """Validate numeric data."""
        if isinstance(data, (int, float)):
            numeric_data = [data]
        elif isinstance(data, (list, tuple)):
            numeric_data = data
        elif isinstance(data, np.ndarray):
            numeric_data = data.flatten()
        elif isinstance(data, pd.DataFrame):
            # Check if all columns are numeric
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) != len(data.columns):
                return False
            numeric_data = data.values.flatten()
        else:
            return False
            
        # Check if all values are numeric
        try:
            numeric_values = [float(x) for x in numeric_data if pd.notna(x)]
        except (ValueError, TypeError):
            return False
            
        # Check value ranges
        if min_value is not None:
            if any(x < min_value for x in numeric_values):
                return False
                
        if max_value is not None:
            if any(x > max_value for x in numeric_values):
                return False
                
        return True
    
    return validator


# Create example numeric contract
ExampleNumericContract = Contract(
    name="example_numeric",
    validator=create_numeric_validator(min_value=0.0, max_value=100.0)
)


# ============================================================================
# PART 2: Contract Testing Examples
# ============================================================================

# Method 1: Manual contract testing using ContractTestBase
class TestExampleTimeSeriesContract(ContractTestBase):
    """Test suite for ExampleTimeSeriesContract using manual approach."""
    
    @pytest.fixture
    def contract_instance(self):
        """Provide a contract instance for testing."""
        import uuid
        return Contract(
            name=f"test_time_series_{uuid.uuid4().hex[:8]}",
            validator=create_time_series_validator(['timestamp', 'value'], min_rows=2)
        )
    
    @pytest.fixture
    def valid_data(self):
        """Provide valid test data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
    
    @pytest.fixture
    def invalid_data(self):
        """Provide invalid test data."""
        return pd.DataFrame({
            'timestamp': ['invalid_date', '2023-01-02'],
            'value': [1.0, 2.0]
        })
    
    def test_contract_validation_valid_data(self, contract_instance, valid_data):
        """Test contract validation with valid data."""
        assert contract_instance.validate(valid_data)
    
    def test_contract_validation_invalid_data(self, contract_instance, invalid_data):
        """Test contract validation with invalid data."""
        assert not contract_instance.validate(invalid_data)
    
    def test_contract_name_and_representation(self, contract_instance):
        """Test contract name and string representation."""
        assert contract_instance.name.startswith("test_time_series_")
        assert "test_time_series" in str(contract_instance)


# Method 2: Using factory function for contract testing (recommended)
TestExampleNumericContractAuto = create_contract_test_suite(
    Contract,
    validator_func=create_numeric_validator(min_value=0.0, max_value=10.0),
    test_data={
        'valid': [pd.DataFrame({'A': [1, 2], 'B': [3, 4]})],
        'invalid': [pd.DataFrame({'A': [-1, 2], 'B': [3, 4]})]  # Contains negative values
    }
)

TestExampleTimeSeriesContractAuto = create_contract_test_suite(
    Contract,
    validator_func=create_time_series_validator(),
    test_data={
        'valid': [pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'value': [1.0, 2.0],
            'category': ['A', 'B']
        })],
        'invalid': [pd.DataFrame({'value': [1.0, 2.0]})]  # Missing timestamp column
    }
)


# ============================================================================
# PART 3: Custom Component Examples
# ============================================================================

# Example 1: Simple scaling component for demonstration
class ExampleScalingComponent(AnalysisComponent):
    """Example component that scales data by a factor."""
    
    def __init__(self, scale_factor=2.0):
        self.scale_factor = scale_factor
    
    def get_config(self):
        return {'scale_factor': self.scale_factor}
    
    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        """Scale the input data by the scale factor."""
        return data * self.scale_factor


# Example 2: Manual testing using ComponentTestBase
class TestExampleScalingComponentManual(ComponentTestBase):
    """Manual test class using the ComponentTestBase utilities."""
    
    def test_scaling_component_interface(self):
        """Test that the scaling component implements required interface."""
        self.assert_component_interface(ExampleScalingComponent)
    
    def test_scaling_component_config(self):
        """Test the scaling component's configuration."""
        component = ExampleScalingComponent(scale_factor=3.0)
        self.assert_component_config(component, expected_keys=['scale_factor'])
        
        config = component.get_config()
        assert config['scale_factor'] == 3.0
    
    def test_scaling_component_contracts(self):
        """Test the scaling component's contract decorations."""
        component = ExampleScalingComponent()
        self.assert_component_contracts(component)
    
    def test_scaling_component_serialization(self):
        """Test scaling component serialization."""
        component = ExampleScalingComponent(scale_factor=1.5)
        
        # Test serialization manually since we don't have a fixture
        import json
        config = component.get_config()
        serialized = json.dumps(config)
        deserialized_config = json.loads(serialized)
        new_instance = ExampleScalingComponent(**deserialized_config)
        assert new_instance.get_config() == config
    
    def test_scaling_component_functionality(self, sample_dataframe):
        """Test the actual scaling functionality."""
        component = ExampleScalingComponent(scale_factor=2.0)
        result = component.run(sample_dataframe)
        
        # Check that data is properly scaled
        expected = sample_dataframe * 2.0
        pd.testing.assert_frame_equal(result, expected)


# ============================================================================
# PART 4: Component Testing Examples
# ============================================================================

# Example 3: Automatic testing using the factory function
# This approach generates a test class automatically with minimal code
TestExampleScalingComponentAuto = create_component_test_suite(
    ExampleScalingComponent,
    test_data={
        'input': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),
        'expected_output_type': pd.DataFrame
    },
    expected_config_keys=['scale_factor']
)


# Example 4: Testing a component that requires complex initialization
class ExampleComplexComponent(AnalysisComponent):
    """Example component with complex initialization requirements."""
    
    def __init__(self, weights, method='weighted_sum', normalize=True):
        if not isinstance(weights, (list, np.ndarray)):
            raise TypeError("weights must be a list or numpy array")
        
        self.weights = np.array(weights)
        self.method = method
        self.normalize = normalize
    
    def get_config(self):
        return {
            'weights': self.weights.tolist(),  # Convert to list for JSON serialization
            'method': self.method,
            'normalize': self.normalize
        }
    
    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        """Apply weighted transformation to the data."""
        if len(self.weights) != data.shape[1]:
            raise ValueError("Number of weights must match number of columns")
        
        result = data * self.weights
        
        if self.normalize:
            row_sums = result.sum(axis=1)
            result = result.div(row_sums, axis=0)
        
        return result


class TestExampleComplexComponent(ComponentTestBase):
    """Test class for the complex component."""
    
    def test_complex_component_with_valid_params(self):
        """Test the complex component with valid parameters."""
        weights = [0.3, 0.7]
        component = ExampleComplexComponent(weights=weights, method='weighted_sum')
        
        # Test interface
        self.assert_component_interface(ExampleComplexComponent)
        
        # Test configuration
        self.assert_component_config(component, expected_keys=['weights', 'method', 'normalize'])
        
        # Test contracts
        self.assert_component_contracts(component)
        
        # Test serialization manually
        import json
        config = component.get_config()
        serialized = json.dumps(config)
        deserialized_config = json.loads(serialized)
        new_instance = ExampleComplexComponent(**deserialized_config)
        assert new_instance.get_config() == config
    
    def test_complex_component_functionality(self):
        """Test the complex component's functionality."""
        test_data = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        
        weights = [0.5, 0.5]
        component = ExampleComplexComponent(weights=weights, normalize=False)
        result = component.run(test_data)
        
        # Check that weights are applied correctly
        expected = test_data * weights
        pd.testing.assert_frame_equal(result, expected)
    
    def test_complex_component_error_handling(self):
        """Test error handling in the complex component."""
        # Test invalid weights type
        with pytest.raises(TypeError):
            ExampleComplexComponent(weights="invalid")
        
        # Test mismatched weights and data dimensions
        component = ExampleComplexComponent(weights=[0.5, 0.5, 0.3])  # 3 weights
        test_data = pd.DataFrame({'A': [1], 'B': [2]})  # 2 columns
        
        with pytest.raises(ValueError):
            component.run(test_data)


# Example 5: How to test components that don't require parameters
class ExampleSimpleComponent(AnalysisComponent):
    """Example component with no initialization parameters."""
    
    def get_config(self):
        return {}
    
    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        """Return the data unchanged (identity transformation)."""
        return data.copy()


# This will work seamlessly with the factory function
TestExampleSimpleComponent = create_component_test_suite(
    ExampleSimpleComponent,
    test_data={
        'input': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
        'expected_output_type': pd.DataFrame
    },
    expected_config_keys=[]  # No expected keys for this simple component
)


# ============================================================================
# PART 5: Integration Testing Examples
# ============================================================================

class TestComponentContractIntegration:
    """Integration tests between custom components and contracts."""
    
    def test_scaling_component_with_numeric_contract(self):
        """Test scaling component with numeric contract validation."""
        # Create component and contract
        component = ExampleScalingComponent(scale_factor=2.0)
        contract = Contract(
            name="integration_numeric",
            validator=create_numeric_validator(min_value=0.0, max_value=20.0)
        )
        
        # Test data
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Validate input
        assert contract.validate(test_data)
        
        # Run component
        result = component.run(test_data)
        
        # Validate output
        assert contract.validate(result)
        
        # Check scaling worked correctly
        expected = test_data * 2.0
        pd.testing.assert_frame_equal(result, expected)
    
    def test_complex_component_with_time_series_contract(self):
        """Test complex component with time series contract."""
        # Create time series data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='D'),
            'value1': [1.0, 2.0, 3.0],
            'value2': [4.0, 5.0, 6.0]
        })
        
        # Create contract for time series validation
        contract = Contract(
            name="integration_time_series",
            validator=create_time_series_validator(['timestamp', 'value1', 'value2'], min_rows=2)
        )
        
        # Validate input
        assert contract.validate(test_data)
        
        # Create component for numeric columns only
        numeric_data = test_data[['value1', 'value2']]
        component = ExampleComplexComponent(weights=[0.6, 0.4], normalize=True)
        
        # Run component
        result = component.run(numeric_data)
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape == numeric_data.shape
    
    def test_component_chain_with_contracts(self):
        """Test chaining multiple components with contract validation."""
        # Initial data
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create contracts
        numeric_contract = Contract(
            name="integration_chain_numeric",
            validator=create_numeric_validator(min_value=0.0, max_value=100.0)
        )
        
        # Create component chain
        scaler = ExampleScalingComponent(scale_factor=2.0)
        identity = ExampleSimpleComponent()
        
        # Validate initial data
        assert numeric_contract.validate(data)
        
        # Apply first component
        scaled_data = scaler.run(data)
        assert numeric_contract.validate(scaled_data)
        
        # Apply second component
        final_data = identity.run(scaled_data)
        assert numeric_contract.validate(final_data)
        
        # Verify final result
        expected = data * 2.0
        pd.testing.assert_frame_equal(final_data, expected)


if __name__ == "__main__":
    # This demonstrates how the testing framework can be used interactively
    print("Component Testing Framework Example")
    print("====================================")
    
    # Test the scaling component
    component = ExampleScalingComponent(scale_factor=2.5)
    test_base = ComponentTestBase()
    
    print(f"Testing {component.__class__.__name__}...")
    
    try:
        test_base.assert_component_interface(ExampleScalingComponent)
        print("✓ Interface check passed")
        
        test_base.assert_component_config(component, ['scale_factor'])
        print("✓ Configuration check passed")
        
        test_base.assert_component_contracts(component)
        print("✓ Contract check passed")
        
        test_base.test_component_serialization_roundtrip(component)
        print("✓ Serialization check passed")
        
        print("\nAll tests passed! The component is ready for integration.")
        
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")