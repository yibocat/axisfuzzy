"""Tests for the core contract system in `axisfuzzy.analysis.contracts`.

This file covers the fundamental functionalities of the contract system, including:
- `Contract` base class behavior and validation.
- Contract registration and retrieval mechanisms.
- Compatibility checks between contracts (inheritance hierarchy).
- Comprehensive validation logic for all built-in contracts.
- @contract decorator functionality with type hint parsing.
- Extensible test framework for future contract additions.
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Union
from unittest.mock import Mock

from axisfuzzy.analysis.contracts.base import Contract
from axisfuzzy.analysis.contracts.decorator import contract
from axisfuzzy.analysis import build_in
from axisfuzzy.core import Fuzznum, Fuzzarray
from axisfuzzy.analysis.dataframe import FuzzyDataFrame


# Test fixtures for various data types
@pytest.fixture
def sample_crisp_dataframe():
    """A clean numeric DataFrame for testing."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0],
        'B': [4.0, 5.0, 6.0],
        'C': [7.0, 8.0, 9.0]
    })


@pytest.fixture
def mixed_dataframe():
    """A DataFrame with mixed types including Fuzzarray."""
    return pd.DataFrame({
        'crisp': [1.0, 2.0, 3.0],
        'fuzzy': [Fuzzarray([1, 2, 3]), Fuzzarray([4, 5, 6]), Fuzzarray([7, 8, 9])]
    })


@pytest.fixture
def sample_weights():
    """Sample weight vector."""
    return np.array([0.3, 0.4, 0.3])


@pytest.fixture
def normalized_weights():
    """Normalized weight vector that sums to 1.0."""
    return np.array([0.2, 0.5, 0.3])


@pytest.fixture
def pairwise_matrix():
    """Square pairwise comparison matrix."""
    return pd.DataFrame([
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 2.0],
        [0.33, 0.5, 1.0]
    ])


# Helper to reset registry for test isolation
@pytest.fixture(autouse=True)
def reset_contract_registry():
    """Ensures the contract registry is clean before each test."""
    original_registry = Contract._registry.copy()
    Contract._registry.clear()
    yield
    Contract._registry.clear()
    Contract._registry.update(original_registry)


class TestContractCore:
    """Tests for the core functionalities of the Contract base class."""

    def test_contract_creation_and_registration(self):
        """Test basic contract creation and automatic registration."""
        def validator(obj):
            return isinstance(obj, str)
        
        test_contract = Contract('TestContract', validator)
        
        assert 'TestContract' in Contract._registry
        assert Contract.get('TestContract') is test_contract
        assert test_contract.name == 'TestContract'
        assert test_contract.validator is validator
        assert test_contract.parent is None

    def test_contract_creation_with_parent(self):
        """Test contract creation with parent relationship."""
        parent_contract = Contract('ParentContract', lambda obj: True)
        child_contract = Contract('ChildContract', lambda obj: isinstance(obj, str), parent=parent_contract)
        
        assert child_contract.parent is parent_contract
        assert parent_contract.parent is None

    def test_get_contract_by_name(self):
        """Test retrieving contracts by name."""
        test_contract = Contract('GetTestContract', lambda obj: True)
        
        retrieved = Contract.get('GetTestContract')
        assert retrieved is test_contract

    def test_get_contract_by_instance(self):
        """Test that passing a Contract instance to get() returns it directly."""
        test_contract = Contract('InstanceTestContract', lambda obj: True)
        
        retrieved = Contract.get(test_contract)
        assert retrieved is test_contract

    def test_get_non_existent_contract_raises_error(self):
        """Test that getting a non-existent contract raises a NameError."""
        with pytest.raises(NameError, match="Contract 'NonExistent' is not registered"):
            Contract.get('NonExistent')

    def test_duplicate_contract_name_raises_error(self):
        """Test that creating a contract with a duplicate name raises a NameError."""
        Contract('DuplicateContract', lambda obj: True)
        
        with pytest.raises(NameError, match="Contract with name 'DuplicateContract' is already registered"):
            Contract('DuplicateContract', lambda obj: False)

    def test_invalid_validator_raises_error(self):
        """Test that providing a non-callable validator raises a TypeError."""
        with pytest.raises(TypeError, match="Validator must be callable"):
            Contract('InvalidValidatorContract', 'not_a_function')

    def test_invalid_parent_raises_error(self):
        """Test that providing a non-Contract parent raises a TypeError."""
        with pytest.raises(TypeError, match="The 'parent' must be another Contract instance"):
            Contract('InvalidParentContract', lambda obj: True, parent='not_a_contract')

    def test_contract_representation(self):
        """Test the string representation of a contract."""
        test_contract = Contract('ReprTestContract', lambda obj: True)
        
        assert repr(test_contract) == "Contract 'ReprTestContract'"
        assert str(test_contract) == "Contract 'ReprTestContract'"

    def test_contract_validation(self):
        """Test the validate method calls the validator function."""
        def custom_validator(obj):
            return isinstance(obj, int) and obj > 0
        
        test_contract = Contract('ValidationTestContract', custom_validator)
        
        assert test_contract.validate(5) == True
        assert test_contract.validate(-1) == False
        assert test_contract.validate('string') == False


class TestContractCompatibility:
    """Tests for the contract compatibility logic."""

    def test_self_compatibility(self):
        """A contract should always be compatible with itself."""
        contract = Contract('SelfCompatContract', lambda obj: True)
        assert contract.is_compatible_with(contract) == True

    def test_parent_child_compatibility(self):
        """A child contract should be compatible with its parent."""
        parent_contract = Contract('ParentCompatContract', lambda obj: True)
        child_contract = Contract('ChildCompatContract', lambda obj: isinstance(obj, str), parent=parent_contract)

        # Child should be compatible with parent
        assert child_contract.is_compatible_with(parent_contract) == True
        # Parent should NOT be compatible with child
        assert parent_contract.is_compatible_with(child_contract) == False

    def test_grandparent_compatibility(self):
        """A grandchild contract should be compatible with its grandparent."""
        grandparent = Contract('GrandparentContract', lambda obj: True)
        parent = Contract('ParentInChainContract', lambda obj: isinstance(obj, (str, int)), parent=grandparent)
        grandchild = Contract('GrandchildContract', lambda obj: isinstance(obj, str), parent=parent)

        # Grandchild should be compatible with both parent and grandparent
        assert grandchild.is_compatible_with(parent) == True
        assert grandchild.is_compatible_with(grandparent) == True
        
        # But not the reverse
        assert grandparent.is_compatible_with(grandchild) == False
        assert parent.is_compatible_with(grandchild) == False

    def test_sibling_incompatibility(self):
        """Contracts with the same parent but different branches are not compatible."""
        parent_contract = Contract('SiblingParentContract', lambda obj: True)
        sibling_a = Contract('SiblingAContract', lambda obj: isinstance(obj, str), parent=parent_contract)
        sibling_b = Contract('SiblingBContract', lambda obj: isinstance(obj, int), parent=parent_contract)

        # Siblings should not be compatible with each other
        assert sibling_a.is_compatible_with(sibling_b) == False
        assert sibling_b.is_compatible_with(sibling_a) == False
        
        # But both should be compatible with parent
        assert sibling_a.is_compatible_with(parent_contract) == True
        assert sibling_b.is_compatible_with(parent_contract) == True

    def test_unrelated_incompatibility(self):
        """Completely unrelated contracts should not be compatible."""
        unrelated_a = Contract('UnrelatedAContract', lambda obj: isinstance(obj, str))
        unrelated_b = Contract('UnrelatedBContract', lambda obj: isinstance(obj, int))

        assert unrelated_a.is_compatible_with(unrelated_b) == False
        assert unrelated_b.is_compatible_with(unrelated_a) == False

    def test_complex_inheritance_chain(self):
        """Test compatibility in a complex inheritance chain."""
        # Create a chain: Root -> Branch1 -> Leaf1
        #                      -> Branch2 -> Leaf2
        root = Contract('RootContract', lambda obj: True)
        branch1 = Contract('Branch1Contract', lambda obj: hasattr(obj, '__len__'), parent=root)
        branch2 = Contract('Branch2Contract', lambda obj: isinstance(obj, (int, float)), parent=root)
        leaf1 = Contract('Leaf1Contract', lambda obj: isinstance(obj, str), parent=branch1)
        leaf2 = Contract('Leaf2Contract', lambda obj: isinstance(obj, int), parent=branch2)

        # Test various compatibility relationships
        assert leaf1.is_compatible_with(branch1) == True
        assert leaf1.is_compatible_with(root) == True
        assert leaf1.is_compatible_with(leaf2) == False
        assert leaf1.is_compatible_with(branch2) == False
        
        assert leaf2.is_compatible_with(branch2) == True
        assert leaf2.is_compatible_with(root) == True
        assert leaf2.is_compatible_with(leaf1) == False
        assert leaf2.is_compatible_with(branch1) == False


class TestBuiltinContracts:
    """Tests for all built-in contract types from build_in.py."""

    def test_contract_any(self):
        """Test ContractAny accepts everything."""
        assert build_in.ContractAny.validate("string") == True
        assert build_in.ContractAny.validate(123) == True
        assert build_in.ContractAny.validate([1, 2, 3]) == True
        assert build_in.ContractAny.validate(None) == True

    def test_contract_crisp_table(self, sample_crisp_dataframe):
        """Test ContractCrispTable validation logic."""
        # Should accept numeric DataFrames
        assert build_in.ContractCrispTable.validate(sample_crisp_dataframe) == True
        assert build_in.ContractCrispTable.validate(pd.DataFrame()) == True  # Empty DataFrame
        
        # Should reject non-DataFrames
        assert build_in.ContractCrispTable.validate([1, 2, 3]) == False
        assert build_in.ContractCrispTable.validate("not a dataframe") == False
        
        # Test with string DataFrame (should fail)
        string_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
        assert build_in.ContractCrispTable.validate(string_df) == False
        
        # Test with mixed DataFrame (should fail due to non-numeric columns)
        mixed_df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, 3.0],
            'string_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        assert build_in.ContractCrispTable.validate(mixed_df) == False
        
        # Test with pure numeric DataFrame
        numeric_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
        assert build_in.ContractCrispTable.validate(numeric_df) == True

    def test_contract_fuzzy_table(self):
        """Test ContractFuzzyTable validation logic."""
        # Create a FuzzyDataFrame for testing
        from axisfuzzy.analysis.dataframe import FuzzyDataFrame
        from axisfuzzy.fuzzifier import Fuzzifier
        
        # Create a simple fuzzifier
        fuzzifier = Fuzzifier('gaussmf', mtype='qrofn', q=2, mf_params={'sigma': 0.1, 'c': 0.5})
        
        # Create a crisp DataFrame first
        crisp_df = pd.DataFrame({'A': [0.1, 0.2, 0.3], 'B': [0.4, 0.5, 0.6]})
        
        # Convert to FuzzyDataFrame
        fuzzy_df = FuzzyDataFrame.from_pandas(crisp_df, fuzzifier)
        
        # Should accept FuzzyDataFrame instances
        assert build_in.ContractFuzzyTable.validate(fuzzy_df) == True
        
        # Should reject regular DataFrames (create a regular DataFrame for testing)
        regular_df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        assert build_in.ContractFuzzyTable.validate(regular_df) == False
        
        # Should reject non-DataFrames
        assert build_in.ContractFuzzyTable.validate([1, 2, 3]) == False

    def test_contract_weight_vector(self, sample_weights):
        """Test ContractWeightVector validation logic."""
        # Should accept numpy arrays and pandas Series
        assert build_in.ContractWeightVector.validate(sample_weights) == True
        assert build_in.ContractWeightVector.validate(pd.Series([0.1, 0.2, 0.3])) == True
        
        # Should reject 2D arrays
        assert build_in.ContractWeightVector.validate(np.array([[1, 2], [3, 4]])) == False
        
        # Should reject non-arrays
        assert build_in.ContractWeightVector.validate([1, 2, 3]) == False
        assert build_in.ContractWeightVector.validate("not an array") == False

    def test_contract_matrix(self, pairwise_matrix):
        """Test ContractMatrix validation logic."""
        # Should accept 2D numpy arrays and DataFrames
        assert build_in.ContractMatrix.validate(pairwise_matrix) == True
        assert build_in.ContractMatrix.validate(np.array([[1, 2], [3, 4]])) == True
        
        # Should reject 1D arrays
        assert build_in.ContractMatrix.validate(np.array([1, 2, 3])) == False
        
        # Should reject non-matrices
        assert build_in.ContractMatrix.validate([1, 2, 3]) == False
        assert build_in.ContractMatrix.validate("not a matrix") == False

    def test_contract_fuzzy_number(self):
        """Test ContractFuzzyNumber validation logic."""
        # Import Fuzznum for testing
        from axisfuzzy.core.fuzznums import Fuzznum
        
        fuzzy_num = Fuzznum(mtype='qrofn', q=2).create(md=0.8, nmd=0.1)
        
        # Should accept Fuzznum instances
        assert build_in.ContractFuzzyNumber.validate(fuzzy_num) == True
        
        # Should reject non-Fuzznum objects
        assert build_in.ContractFuzzyNumber.validate(123) == False
        assert build_in.ContractFuzzyNumber.validate([1, 2, 3]) == False
        assert build_in.ContractFuzzyNumber.validate("not a fuzzy number") == False

    def test_contract_fuzzy_array(self):
        """Test ContractFuzzyArray validation logic."""
        # Import Fuzzarray and Fuzznum for testing
        from axisfuzzy.core.fuzzarray import Fuzzarray
        from axisfuzzy.core.fuzznums import Fuzznum
        
        fuzzy_arr = Fuzzarray([Fuzznum(mtype='qrofn', q=2).create(md=0.8, nmd=0.1), 
                               Fuzznum(mtype='qrofn', q=2).create(md=0.6, nmd=0.3)])
        
        # Should accept Fuzzarray instances
        assert build_in.ContractFuzzyArray.validate(fuzzy_arr) == True
        
        # Should reject non-Fuzzarray objects
        assert build_in.ContractFuzzyArray.validate([1, 2, 3]) == False
        assert build_in.ContractFuzzyArray.validate(np.array([1, 2, 3])) == False
        assert build_in.ContractFuzzyArray.validate("not a fuzzy array") == False

    def test_contract_numeric_value(self):
        """Test ContractNumericValue validation logic."""
        # Should accept int and float
        assert build_in.ContractNumericValue.validate(42) == True
        assert build_in.ContractNumericValue.validate(3.14) == True
        
        # Should reject bool (even though it's technically numeric)
        assert build_in.ContractNumericValue.validate(True) == False
        assert build_in.ContractNumericValue.validate(False) == False
        
        # Should reject non-numeric types
        assert build_in.ContractNumericValue.validate("42") == False
        assert build_in.ContractNumericValue.validate([42]) == False

    def test_contract_string_list(self):
        """Test ContractStringList validation logic."""
        # Should accept lists of strings
        assert build_in.ContractStringList.validate(["a", "b", "c"]) == True
        assert build_in.ContractStringList.validate([]) == True  # Empty list
        
        # Should reject lists with non-strings
        assert build_in.ContractStringList.validate(["a", 1, "c"]) == False
        assert build_in.ContractStringList.validate([1, 2, 3]) == False
        
        # Should reject non-lists
        assert build_in.ContractStringList.validate("not a list") == False
        assert build_in.ContractStringList.validate(("a", "b", "c")) == False


    def test_derived_contracts(self, normalized_weights, pairwise_matrix):
        """Test derived contracts with inheritance relationships."""
        # Test ContractScoreVector (inherits from ContractWeightVector)
        assert build_in.ContractScoreVector.validate(normalized_weights) == True
        assert build_in.ContractScoreVector.validate(pd.Series([0.1, 0.2, 0.3])) == True
        assert build_in.ContractScoreVector.parent is build_in.ContractWeightVector
        
        # Test ContractNormalizedWeights (inherits from ContractWeightVector)
        assert build_in.ContractNormalizedWeights.validate(normalized_weights) == True
        non_normalized = np.array([0.1, 0.2, 0.4])  # Doesn't sum to 1.0
        assert build_in.ContractNormalizedWeights.validate(non_normalized) == False
        assert build_in.ContractNormalizedWeights.parent is build_in.ContractWeightVector
        
        # Test ContractPairwiseMatrix (inherits from ContractMatrix)
        assert build_in.ContractPairwiseMatrix.validate(pairwise_matrix) == True
        non_square = pd.DataFrame([[1, 2, 3], [4, 5, 6]])  # Not square
        assert build_in.ContractPairwiseMatrix.validate(non_square) == False
        assert build_in.ContractPairwiseMatrix.parent is build_in.ContractMatrix
        
        # Test ContractCriteriaList and ContractAlternativeList (inherit from ContractStringList)
        assert build_in.ContractCriteriaList.validate(["criterion1", "criterion2"]) == True
        assert build_in.ContractAlternativeList.validate(["alt1", "alt2"]) == True
        assert build_in.ContractCriteriaList.parent is build_in.ContractStringList
        assert build_in.ContractAlternativeList.parent is build_in.ContractStringList

    def test_specialized_contracts(self):
        """Test specialized contracts for specific use cases."""
        # Test ContractRankingResult
        ranking_series = pd.Series(["A", "B", "C"])
        ranking_list = ["A", "B", "C"]
        assert build_in.ContractRankingResult.validate(ranking_series) == True
        assert build_in.ContractRankingResult.validate(ranking_list) == True
        assert build_in.ContractRankingResult.validate([1, 2, 3]) == True  # Numeric ranking
        assert build_in.ContractRankingResult.validate("not a ranking") == False
        
        # Test ContractThreeWayResult
        three_way_result = {'accept': ["A"], 'reject': ["B"], 'defer': ["C"]}
        assert build_in.ContractThreeWayResult.validate(three_way_result) == True
        incomplete_result = {'accept': ["A"], 'reject': ["B"]}  # Missing 'defer'
        assert build_in.ContractThreeWayResult.validate(incomplete_result) == False
        
        # Test ContractStatisticsDict
        stats_dict = {'mean': 5.0, 'std': 1.2, 'count': 100}
        assert build_in.ContractStatisticsDict.validate(stats_dict) == True
        invalid_stats = {'mean': 5.0, 'description': 'text'}  # Non-numeric value
        assert build_in.ContractStatisticsDict.validate(invalid_stats) == False

    def test_inheritance_compatibility_builtin(self):
        """Test that built-in derived contracts are compatible with their parents."""
        # ContractScoreVector should be compatible with ContractWeightVector
        assert build_in.ContractScoreVector.is_compatible_with(build_in.ContractWeightVector) == True
        assert build_in.ContractWeightVector.is_compatible_with(build_in.ContractScoreVector) == False
        
        # ContractNormalizedWeights should be compatible with ContractWeightVector
        assert build_in.ContractNormalizedWeights.is_compatible_with(build_in.ContractWeightVector) == True
        
        # ContractPairwiseMatrix should be compatible with ContractMatrix
        assert build_in.ContractPairwiseMatrix.is_compatible_with(build_in.ContractMatrix) == True
        
        # ContractCriteriaList should be compatible with ContractStringList
        assert build_in.ContractCriteriaList.is_compatible_with(build_in.ContractStringList) == True


class TestContractDecorator:
    """Tests for the @contract decorator with type hint parsing."""

    def test_decorator_single_input_single_output(self):
        """Test decorator with single input and output contracts."""
        input_contract = Contract('DecoratorInputContract', lambda obj: isinstance(obj, str))
        output_contract = Contract('DecoratorOutputContract', lambda obj: isinstance(obj, int))
        
        @contract
        def my_func(data: input_contract) -> output_contract:
            return len(data)
        
        assert hasattr(my_func, '_contract_inputs')
        assert hasattr(my_func, '_contract_outputs')
        assert hasattr(my_func, '_is_contract_method')
        
        assert my_func._contract_inputs == {'data': 'DecoratorInputContract'}
        assert my_func._contract_outputs == {'output': 'DecoratorOutputContract'}
        assert my_func._is_contract_method == True

    def test_decorator_multiple_inputs(self):
        """Test decorator with multiple input contracts."""
        input_contract1 = Contract('MultiInput1Contract', lambda obj: isinstance(obj, str))
        input_contract2 = Contract('MultiInput2Contract', lambda obj: isinstance(obj, int))
        output_contract = Contract('MultiOutputContract', lambda obj: isinstance(obj, float))
        
        @contract
        def my_func(data1: input_contract1, data2: input_contract2) -> output_contract:
            return float(len(data1) + data2)
        
        assert my_func._contract_inputs == {
            'data1': 'MultiInput1Contract',
            'data2': 'MultiInput2Contract'
        }
        assert my_func._contract_outputs == {'output': 'MultiOutputContract'}

    def test_decorator_dict_output(self):
        """Test decorator with Dict return type annotation."""
        input_contract = Contract('DictInputContract', lambda obj: isinstance(obj, str))
        output_contract = Contract('DictOutputContract', lambda obj: isinstance(obj, int))
        
        @contract
        def my_func(data: input_contract) -> Dict[str, output_contract]:
            return {'result': len(data)}
        
        assert my_func._contract_inputs == {'data': 'DictInputContract'}
        assert my_func._contract_outputs == {'dictoutputcontract': 'DictOutputContract'}

    def test_decorator_union_output(self):
        """Test decorator with Union return type in Dict."""
        input_contract = Contract('UnionInputContract', lambda obj: isinstance(obj, str))
        output_contract1 = Contract('UnionOutput1Contract', lambda obj: isinstance(obj, int))
        output_contract2 = Contract('UnionOutput2Contract', lambda obj: isinstance(obj, float))
        
        @contract
        def my_func(data: input_contract) -> Dict[str, Union[output_contract1, output_contract2]]:
            return {'result1': len(data), 'result2': float(len(data))}
        
        assert my_func._contract_inputs == {'data': 'UnionInputContract'}
        assert my_func._contract_outputs == {
            'unionoutput1contract': 'UnionOutput1Contract',
            'unionoutput2contract': 'UnionOutput2Contract'
        }

    def test_decorator_no_contracts(self):
        """Test decorator on function with no contract annotations."""
        @contract
        def my_func(data: str) -> int:
            return len(data)
        
        assert my_func._contract_inputs == {}
        assert my_func._contract_outputs == {}
        assert my_func._is_contract_method == True

    def test_decorator_type_resolution_error(self):
        """Test decorator with unresolvable type hints."""
        with pytest.raises(TypeError, match="Cannot resolve type hints"):
            @contract
            def my_func(data: 'NonExistentContract') -> int:
                return 42

    def test_decorator_unsupported_return_annotation(self):
        """Test decorator with unsupported return type annotation."""
        input_contract = Contract('UnsupportedInputContract', lambda obj: True)
        
        with pytest.raises(TypeError, match="Unsupported multi-output annotation"):
            @contract
            def my_func(data: input_contract) -> Dict[int, str]:  # Key type is not str
                return {1: 'result'}


class TestContractExtensibility:
    """Tests for contract system extensibility and future expansion."""

    def test_custom_contract_registration(self):
        """Test that custom contracts can be easily registered."""
        def custom_validator(obj):
            return isinstance(obj, dict) and 'required_key' in obj
        
        custom_contract = Contract('CustomTestContract', custom_validator)
        
        # Should be automatically registered
        assert 'CustomTestContract' in Contract._registry
        assert Contract.get('CustomTestContract') is custom_contract
        
        # Should work with validation
        assert custom_contract.validate({'required_key': 'value'}) is True
        assert custom_contract.validate({'other_key': 'value'}) is False

    def test_contract_inheritance_extensibility(self):
        """Test that contract inheritance supports future extensions."""
        # Create a base contract for data structures
        base_data_contract = Contract('BaseDataContract', lambda obj: hasattr(obj, '__len__'))
        
        # Create specialized contracts that inherit from it
        list_contract = Contract('ListDataContract', 
                               lambda obj: isinstance(obj, list), 
                               parent=base_data_contract)
        
        tuple_contract = Contract('TupleDataContract', 
                                lambda obj: isinstance(obj, tuple), 
                                parent=base_data_contract)
        
        # Test inheritance relationships
        assert list_contract.is_compatible_with(base_data_contract) == True
        assert tuple_contract.is_compatible_with(base_data_contract) == True
        assert list_contract.is_compatible_with(tuple_contract) == False
        
        # Test validation
        assert list_contract.validate([1, 2, 3]) == True
        assert list_contract.validate((1, 2, 3)) == False
        assert tuple_contract.validate((1, 2, 3)) == True
        assert tuple_contract.validate([1, 2, 3]) == False

    def test_decorator_with_custom_contracts(self):
        """Test that @contract decorator works with custom contracts."""
        custom_input = Contract('CustomDecoratorInput', lambda obj: isinstance(obj, dict))
        custom_output = Contract('CustomDecoratorOutput', lambda obj: isinstance(obj, list))
        
        @contract
        def process_data(data: custom_input) -> custom_output:
            return list(data.keys())
        
        assert process_data._contract_inputs == {'data': 'CustomDecoratorInput'}
        assert process_data._contract_outputs == {'output': 'CustomDecoratorOutput'}

    def test_future_contract_patterns(self):
        """Test patterns that support future contract system expansion."""
        # Test contract with complex validation logic
        def complex_validator(obj):
            return (isinstance(obj, pd.DataFrame) and 
                   len(obj.columns) >= 3 and 
                   all(pd.api.types.is_numeric_dtype(dtype) for dtype in obj.dtypes))
        
        complex_contract = Contract('ComplexDataContract', complex_validator)
        
        # Should work with complex validation
        valid_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        invalid_df = pd.DataFrame({'A': [1, 2]})  # Only 2 columns
        
        assert complex_contract.validate(valid_df) == True
        assert complex_contract.validate(invalid_df) == False
        
        # Test contract composition (contracts that depend on other contracts)
        base_numeric = Contract('BaseNumericContract', lambda obj: isinstance(obj, (int, float)))
        positive_numeric = Contract('PositiveNumericContract', 
                                  lambda obj: base_numeric.validate(obj) and obj > 0,
                                  parent=base_numeric)
        
        assert positive_numeric.validate(5) == True
        assert positive_numeric.validate(-5) == False
        assert positive_numeric.validate('5') == False
        assert positive_numeric.is_compatible_with(base_numeric) == True


# =============================================================================
# CONTRACT TESTING FRAMEWORK FOR FUTURE EXTENSIONS
# =============================================================================

class ContractTestBase:
    """Base class providing standardized testing methods for contract validation.
    
    This class provides a comprehensive set of testing utilities for validating
    contract behavior, compatibility, and extensibility. It serves as the foundation
    for testing both built-in and custom contracts in the AxisFuzzy analysis system.
    
    Methods
    -------
    assert_contract_interface(contract_class)
        Validates that a contract class implements the required interface.
    assert_contract_validation(contract_instance, valid_data, invalid_data)
        Tests contract validation logic with both valid and invalid inputs.
    assert_contract_compatibility(contract_a, contract_b, expected_result)
        Tests compatibility relationships between contracts.
    assert_contract_registration(contract_instance)
        Verifies that a contract is properly registered in the global registry.
    validate_contract_serialization(contract_instance)
        Tests contract metadata serialization and deserialization.
    
    Examples
    --------
    .. code-block:: python
    
        class TestMyContract(ContractTestBase):
            def test_my_contract_interface(self):
                self.assert_contract_interface(MyContract)
            
            def test_my_contract_validation(self):
                contract = MyContract('TestContract', validator_func)
                self.assert_contract_validation(
                    contract, 
                    valid_data=[1, 2, 3], 
                    invalid_data="invalid"
                )
    """
    
    @staticmethod
    def assert_contract_interface(contract_class):
        """Assert that a contract class implements the required interface.
        
        Parameters
        ----------
        contract_class : type
            The contract class to validate.
            
        Raises
        ------
        AssertionError
            If the contract class doesn't implement required methods or attributes.
        """
        # Check that it's a subclass of Contract
        assert issubclass(contract_class, Contract), f"{contract_class.__name__} must inherit from Contract"
        
        # Check required methods exist
        required_methods = ['validate', 'is_compatible_with', 'get']
        for method in required_methods:
            assert hasattr(contract_class, method), f"Contract must implement {method} method"
        
        # Check required attributes
        required_attrs = ['name', 'validator', 'parent', '_registry']
        instance = None
        try:
            # Try to create a minimal instance for testing
            instance = contract_class('TestContract', lambda x: True)
            for attr in required_attrs:
                assert hasattr(instance, attr), f"Contract must have {attr} attribute"
        except Exception as e:
            # If we can't create an instance, at least check class-level attributes
            assert hasattr(contract_class, '_registry'), "Contract must have _registry class attribute"
        finally:
            # Clean up test instance from registry if created
            if instance and 'TestContract' in Contract._registry:
                del Contract._registry['TestContract']
    
    @staticmethod
    def assert_contract_validation(contract_instance, valid_data, invalid_data):
        """Assert that contract validation works correctly.
        
        Parameters
        ----------
        contract_instance : Contract
            The contract instance to test.
        valid_data : list
            List of data that should pass validation.
        invalid_data : list
            List of data that should fail validation.
            
        Raises
        ------
        AssertionError
            If validation doesn't behave as expected.
        """
        # Ensure valid_data and invalid_data are lists
        if not isinstance(valid_data, list):
            valid_data = [valid_data]
        if not isinstance(invalid_data, list):
            invalid_data = [invalid_data]
        
        # Test valid data
        for data in valid_data:
            result = contract_instance.validate(data)
            assert result == True, f"Contract {contract_instance.name} should accept {type(data).__name__}: {data}"
        
        # Test invalid data
        for data in invalid_data:
            result = contract_instance.validate(data)
            assert result == False, f"Contract {contract_instance.name} should reject {type(data).__name__}: {data}"
    
    @staticmethod
    def assert_contract_compatibility(contract_a, contract_b, expected_result):
        """Assert that contract compatibility works as expected.
        
        Parameters
        ----------
        contract_a : Contract
            First contract in compatibility check.
        contract_b : Contract
            Second contract in compatibility check.
        expected_result : bool
            Expected result of contract_a.is_compatible_with(contract_b).
            
        Raises
        ------
        AssertionError
            If compatibility check doesn't match expected result.
        """
        result = contract_a.is_compatible_with(contract_b)
        assert result == expected_result, (
            f"Expected {contract_a.name}.is_compatible_with({contract_b.name}) "
            f"to be {expected_result}, but got {result}"
        )
    
    @staticmethod
    def assert_contract_registration(contract_instance):
        """Assert that a contract is properly registered.
        
        Parameters
        ----------
        contract_instance : Contract
            The contract instance to check.
            
        Raises
        ------
        AssertionError
            If the contract is not properly registered.
        """
        # Check that contract is in registry
        assert contract_instance.name in Contract._registry, (
            f"Contract {contract_instance.name} is not registered"
        )
        
        # Check that registry returns the same instance
        retrieved = Contract.get(contract_instance.name)
        assert retrieved is contract_instance, (
            f"Registry returned different instance for {contract_instance.name}"
        )
    
    @staticmethod
    def validate_contract_serialization(contract_instance):
        """Validate that contract metadata can be serialized.
        
        Parameters
        ----------
        contract_instance : Contract
            The contract instance to test.
            
        Raises
        ------
        AssertionError
            If contract metadata cannot be serialized properly.
        """
        import json
        
        # Create serializable metadata
        metadata = {
            'name': contract_instance.name,
            'parent_name': contract_instance.parent.name if contract_instance.parent else None,
            'has_validator': callable(contract_instance.validator)
        }
        
        # Test JSON serialization
        try:
            serialized = json.dumps(metadata)
            deserialized = json.loads(serialized)
            
            assert deserialized['name'] == contract_instance.name
            assert deserialized['parent_name'] == (contract_instance.parent.name if contract_instance.parent else None)
            assert deserialized['has_validator'] == True
            
        except (TypeError, ValueError) as e:
            raise AssertionError(f"Contract metadata serialization failed: {e}")


def create_contract_test_suite(contract_class, validator_func, test_data=None, parent_contract=None):
    """Factory function to create a comprehensive test suite for a contract.
    
    This function automatically generates a test class with standardized tests
    for contract interface, validation, compatibility, and registration.
    
    Parameters
    ----------
    contract_class : type
        The contract class to test (should be Contract or a subclass).
    validator_func : callable
        The validator function for the contract.
    test_data : dict, optional
        Dictionary containing 'valid' and 'invalid' data lists for testing.
        If not provided, basic tests will be skipped.
    parent_contract : Contract, optional
        Parent contract for inheritance testing.
        
    Returns
    -------
    type
        A dynamically created test class with comprehensive contract tests.
        
    Examples
    --------
    .. code-block:: python
    
        # Create a test suite for a custom contract
        TestMyContract = create_contract_test_suite(
            Contract,
            validator_func=lambda x: isinstance(x, str),
            test_data={
                'valid': ['hello', 'world'],
                'invalid': [123, [], None]
            }
        )
    """
    
    class GeneratedContractTest(ContractTestBase):
        """Dynamically generated test class for contract validation."""
        
        @classmethod
        def setup_class(cls):
            """Set up test contract instance."""
            cls.contract_name = f"Test{contract_class.__name__}_{id(cls)}"
            cls.contract_instance = contract_class(cls.contract_name, validator_func, parent=parent_contract)
        
        @classmethod
        def teardown_class(cls):
            """Clean up test contract from registry."""
            if hasattr(cls, 'contract_name') and cls.contract_name in Contract._registry:
                del Contract._registry[cls.contract_name]
        
        def test_contract_interface(self):
            """Test that the contract implements required interface."""
            self.assert_contract_interface(contract_class)
        
        def test_contract_instantiation(self):
            """Test that the contract can be instantiated properly."""
            assert self.contract_instance is not None
            assert self.contract_instance.name == self.contract_name
            assert self.contract_instance.validator is validator_func
            assert self.contract_instance.parent is parent_contract
        
        def test_contract_registration(self):
            """Test that the contract is properly registered."""
            self.assert_contract_registration(self.contract_instance)
        
        def test_contract_serialization(self):
            """Test that contract metadata can be serialized."""
            self.validate_contract_serialization(self.contract_instance)
        
        def test_contract_validation_logic(self):
            """Test contract validation with provided test data."""
            if test_data and 'valid' in test_data and 'invalid' in test_data:
                self.assert_contract_validation(
                    self.contract_instance,
                    test_data['valid'],
                    test_data['invalid']
                )
        
        def test_contract_compatibility(self):
            """Test contract compatibility relationships."""
            # Self-compatibility
            self.assert_contract_compatibility(
                self.contract_instance, 
                self.contract_instance, 
                True
            )
            
            # Parent compatibility (if parent exists)
            if parent_contract:
                self.assert_contract_compatibility(
                    self.contract_instance,
                    parent_contract,
                    True
                )
                self.assert_contract_compatibility(
                    parent_contract,
                    self.contract_instance,
                    False
                )
    
    # Set a meaningful name for the generated class
    GeneratedContractTest.__name__ = f"Test{contract_class.__name__}Generated"
    GeneratedContractTest.__qualname__ = GeneratedContractTest.__name__
    
    return GeneratedContractTest


# =============================================================================
# HELPER FUNCTIONS FOR CONTRACT TESTING
# =============================================================================

def create_test_contract(name, validator_func, parent=None):
    """Helper function to create test contracts with automatic cleanup.
    
    Parameters
    ----------
    name : str
        Name for the test contract.
    validator_func : callable
        Validator function for the contract.
    parent : Contract, optional
        Parent contract for inheritance.
        
    Returns
    -------
    Contract
        A test contract instance that will be automatically cleaned up.
    """
    return Contract(name, validator_func, parent=parent)


def assert_contract_hierarchy(contracts, expected_hierarchy):
    """Assert that a list of contracts follows the expected hierarchy.
    
    Parameters
    ----------
    contracts : list of Contract
        List of contract instances to check.
    expected_hierarchy : list of tuples
        List of (child_index, parent_index) tuples defining expected relationships.
        Use None for parent_index to indicate no parent.
        
    Raises
    ------
    AssertionError
        If the hierarchy doesn't match expectations.
        
    Examples
    --------
    .. code-block:: python
    
        contracts = [root, child1, child2, grandchild]
        # root has no parent, child1 and child2 inherit from root, grandchild from child1
        expected = [(0, None), (1, 0), (2, 0), (3, 1)]
        assert_contract_hierarchy(contracts, expected)
    """
    for child_idx, parent_idx in expected_hierarchy:
        child_contract = contracts[child_idx]
        expected_parent = contracts[parent_idx] if parent_idx is not None else None
        
        assert child_contract.parent is expected_parent, (
            f"Contract {child_contract.name} should have parent "
            f"{expected_parent.name if expected_parent else 'None'}, "
            f"but has {child_contract.parent.name if child_contract.parent else 'None'}"
        )


if __name__ == "__main__":
    pytest.main([__file__])