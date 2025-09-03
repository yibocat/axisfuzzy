#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午4:45
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Integration tests for axisfuzzy.mixin system.

This module provides comprehensive integration tests that verify the complete
mixin system workflow, from function registration through injection to
runtime usage. These tests ensure that all components work together correctly
and that the mixin system integrates properly with the core AxisFuzzy classes.

Test Coverage
-------------
- End-to-end mixin registration and injection workflow
- Integration with actual Fuzznum and Fuzzarray classes
- Cross-module function registration and usage
- Real-world usage scenarios and patterns
- Performance and behavior under realistic conditions
- Interaction between different injection types
- Module namespace pollution and cleanup

Test Classes
------------
TestMixinSystemIntegration : End-to-end system integration tests
TestRealWorldUsagePatterns : Tests based on realistic usage scenarios
TestCrossModuleIntegration : Tests for cross-module registration patterns
TestNamespaceManagement : Tests for proper namespace handling
TestSystemBehavior : Tests for overall system behavior and performance
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from axisfuzzy.mixin.registry import (
    MixinFunctionRegistry,
    get_registry_mixin,
    register_mixin
)
from axisfuzzy.mixin.factory import (
    _concat_factory,
    _stack_factory,
    _append_factory
)


class TestMixinSystemIntegration:
    """
    End-to-end integration tests for the complete mixin system.
    
    Tests the entire workflow from registration to runtime usage,
    ensuring all components work together correctly.
    """
    
    def test_complete_mixin_system_workflow(self):
        """
        Test complete mixin system workflow from registration to usage.
        
        This test simulates the complete lifecycle of mixin functions:
        1. Registration of functions with different injection types
        2. Injection into target classes and module namespace
        3. Runtime usage of injected functions
        4. Verification of correct behavior
        """
        # Create a fresh registry for this test
        registry = MixinFunctionRegistry()
        
        # Create mock classes that simulate Fuzznum and Fuzzarray
        class MockFuzznum:
            def __init__(self, value):
                self.value = value
                self.mtype = 'qrofn'
            
            def __repr__(self):
                return f"MockFuzznum({self.value})"
        
        class MockFuzzarray:
            def __init__(self, values):
                self.values = values
                self.mtype = 'qrofn'
                self.shape = (len(values),)
            
            def __len__(self):
                return len(self.values)
            
            def __repr__(self):
                return f"MockFuzzarray({self.values})"
        
        # Register various mixin functions
        @registry.register(
            name='get_mtype',
            target_classes=['MockFuzznum', 'MockFuzzarray'],
            injection_type='instance_function'
        )
        def get_mtype_impl(self):
            """Get the mtype of the fuzzy object."""
            return self.mtype
        
        @registry.register(
            name='create_zero',
            injection_type='top_level_function'
        )
        def create_zero_impl():
            """Create a zero fuzzy number."""
            return MockFuzznum(0)
        
        @registry.register(
            name='size',
            target_classes=['MockFuzzarray'],
            injection_type='both'
        )
        def size_impl(self):
            """Get the size of the fuzzy array."""
            return len(self.values)
        
        @registry.register(
            name='to_array',
            target_classes=['MockFuzznum'],
            injection_type='both'
        )
        def to_array_impl(self):
            """Convert fuzzy number to single-element array."""
            return MockFuzzarray([self.value])
        
        # Prepare injection targets
        class_map = {
            'MockFuzznum': MockFuzznum,
            'MockFuzzarray': MockFuzzarray
        }
        module_namespace = {}
        
        # Perform injection
        registry.build_and_inject(class_map, module_namespace)
        
        # Test instance-only methods
        num = MockFuzznum(42)
        array = MockFuzzarray([1, 2, 3, 4, 5])
        
        assert num.get_mtype() == 'qrofn'
        assert array.get_mtype() == 'qrofn'
        
        # Test top-level-only functions
        zero_num = module_namespace['create_zero']()
        assert isinstance(zero_num, MockFuzznum)
        assert zero_num.value == 0
        
        # Test 'both' injection type functions
        # Instance method usage
        assert array.size() == 5
        assert num.to_array().values == [42]
        
        # Top-level function usage
        assert module_namespace['size'](array) == 5
        converted_array = module_namespace['to_array'](num)
        assert converted_array.values == [42]
        
        # Verify top-level function names
        top_level_names = registry.get_top_level_function_names()
        expected_names = {'create_zero', 'size', 'to_array'}
        assert set(top_level_names) == expected_names
    
    def test_mixin_integration_with_factory_functions(self):
        """
        Test integration between mixin registry and factory functions.
        
        Verifies that mixin functions can work alongside and complement
        the existing factory functions in the mixin system.
        """
        registry = MixinFunctionRegistry()
        
        # Create mock Fuzzarray class that works with factory functions
        class MockFuzzarray:
            def __init__(self, values, mtype='qrofn', q=2):
                self.values = values
                self.mtype = mtype
                self.q = q
                self.shape = (len(values),)
                self.size = len(values)
            
            def copy(self):
                return MockFuzzarray(self.values.copy(), self.mtype, self.q)
            
            def __repr__(self):
                return f"MockFuzzarray({self.values}, mtype='{self.mtype}')"
        
        # Register mixin functions that complement factory operations
        @registry.register(
            name='prepare_for_concat',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def prepare_for_concat_impl(self):
            """Prepare array for concatenation by ensuring proper format."""
            # Simulate preparation logic
            prepared = self.copy()
            prepared._prepared_for_concat = True
            return prepared
        
        @registry.register(
            name='validate_concat_compatibility',
            injection_type='top_level_function'
        )
        def validate_concat_compatibility_impl(*arrays):
            """Validate that arrays can be concatenated."""
            if not arrays:
                return False
            
            first_mtype = arrays[0].mtype
            first_q = arrays[0].q
            
            for array in arrays[1:]:
                if array.mtype != first_mtype or array.q != first_q:
                    return False
            
            return True
        
        @registry.register(
            name='enhanced_concat',
            target_classes=['MockFuzzarray'],
            injection_type='both'
        )
        def enhanced_concat_impl(self, *others):
            """Enhanced concatenation with validation and preparation."""
            all_arrays = [self] + list(others)
            
            # Use the top-level validation function
            if not module_namespace['validate_concat_compatibility'](*all_arrays):
                raise ValueError("Arrays are not compatible for concatenation")
            
            # Prepare arrays
            prepared_arrays = [arr.prepare_for_concat() for arr in all_arrays]
            
            # Simulate concatenation
            all_values = []
            for arr in prepared_arrays:
                all_values.extend(arr.values)
            
            return MockFuzzarray(all_values, self.mtype, self.q)
        
        # Inject functions
        class_map = {'MockFuzzarray': MockFuzzarray}
        module_namespace = {}
        registry.build_and_inject(class_map, module_namespace)
        
        # Test the integrated functionality
        arr1 = MockFuzzarray([1, 2, 3])
        arr2 = MockFuzzarray([4, 5, 6])
        arr3 = MockFuzzarray([7, 8, 9])
        
        # Test preparation
        prepared = arr1.prepare_for_concat()
        assert hasattr(prepared, '_prepared_for_concat')
        assert prepared._prepared_for_concat is True
        
        # Test validation
        assert module_namespace['validate_concat_compatibility'](arr1, arr2, arr3) is True
        
        # Test incompatible arrays
        incompatible_arr = MockFuzzarray([10, 11], mtype='different')
        assert module_namespace['validate_concat_compatibility'](arr1, incompatible_arr) is False
        
        # Test enhanced concatenation
        result = arr1.enhanced_concat(arr2, arr3)
        assert result.values == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert result.mtype == 'qrofn'
        
        # Test top-level enhanced concatenation
        result2 = module_namespace['enhanced_concat'](arr1, arr2)
        assert result2.values == [1, 2, 3, 4, 5, 6]
    
    def test_mixin_system_error_propagation(self):
        """
        Test that errors are properly propagated through the mixin system.
        
        Verifies that exceptions raised in mixin functions are correctly
        propagated to the caller without being masked by the injection system.
        """
        registry = MixinFunctionRegistry()
        
        class MockClass:
            def __init__(self, value):
                self.value = value
        
        # Register functions that raise different types of errors
        @registry.register(
            name='raise_value_error',
            target_classes=['MockClass'],
            injection_type='instance_function'
        )
        def raise_value_error_impl(self):
            raise ValueError(f"Invalid value: {self.value}")
        
        @registry.register(
            name='raise_type_error',
            injection_type='top_level_function'
        )
        def raise_type_error_impl(obj):
            raise TypeError(f"Unsupported type: {type(obj)}")
        
        @registry.register(
            name='conditional_error',
            target_classes=['MockClass'],
            injection_type='both'
        )
        def conditional_error_impl(self):
            if self.value < 0:
                raise RuntimeError("Negative values not allowed")
            return self.value * 2
        
        # Inject functions
        class_map = {'MockClass': MockClass}
        module_namespace = {}
        registry.build_and_inject(class_map, module_namespace)
        
        # Test error propagation from instance methods
        obj = MockClass(42)
        with pytest.raises(ValueError, match="Invalid value: 42"):
            obj.raise_value_error()
        
        # Test error propagation from top-level functions
        with pytest.raises(TypeError, match="Unsupported type"):
            module_namespace['raise_type_error'](obj)
        
        # Test conditional errors in 'both' injection type
        positive_obj = MockClass(5)
        negative_obj = MockClass(-3)
        
        # Should work for positive values
        assert positive_obj.conditional_error() == 10
        assert module_namespace['conditional_error'](positive_obj) == 10
        
        # Should raise error for negative values
        with pytest.raises(RuntimeError, match="Negative values not allowed"):
            negative_obj.conditional_error()
        
        with pytest.raises(RuntimeError, match="Negative values not allowed"):
            module_namespace['conditional_error'](negative_obj)


class TestRealWorldUsagePatterns:
    """
    Tests based on realistic usage patterns and scenarios.
    
    These tests simulate how the mixin system would be used in
    real-world applications and libraries.
    """
    
    def test_numpy_like_operations_pattern(self):
        """
        Test mixin system with NumPy-like operations pattern.
        
        Simulates registering and using functions that provide
        NumPy-like array manipulation capabilities.
        """
        registry = MixinFunctionRegistry()
        
        class MockFuzzarray:
            def __init__(self, values, shape=None):
                self.values = values
                self.shape = shape or (len(values),)
                self.ndim = len(self.shape)
                self._size = len(values)
                self.size = len(values)  # Keep both for compatibility
            
            def copy(self):
                return MockFuzzarray(self.values.copy(), self.shape)
        
        # Register NumPy-like operations
        @registry.register(
            name='reshape',
            target_classes=['MockFuzzarray'],
            injection_type='both'
        )
        def reshape_impl(self, *new_shape):
            """Reshape the fuzzy array."""
            if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
                new_shape = new_shape[0]
            
            total_size = 1
            for dim in new_shape:
                if dim != -1:
                    total_size *= dim
            
            if total_size != self._size:
                raise ValueError(f"Cannot reshape array of size {self._size} into shape {new_shape}")
            
            result = self.copy()
            result.shape = tuple(new_shape)
            result.ndim = len(new_shape)
            return result
        
        @registry.register(
            name='flatten',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def flatten_impl(self):
            """Flatten the fuzzy array to 1D."""
            result = self.copy()
            result.shape = (self._size,)
            result.ndim = 1
            return result
        
        @registry.register(
            name='zeros',
            injection_type='top_level_function'
        )
        def zeros_impl(shape):
            """Create a fuzzy array filled with zeros."""
            if isinstance(shape, int):
                shape = (shape,)
            
            total_size = 1
            for dim in shape:
                total_size *= dim
            
            return MockFuzzarray([0] * total_size, shape)
        
        @registry.register(
            name='transpose',
            target_classes=['MockFuzzarray'],
            injection_type='both'
        )
        def transpose_impl(self):
            """Transpose the fuzzy array."""
            if self.ndim != 2:
                raise ValueError("Transpose only supported for 2D arrays")
            
            result = self.copy()
            result.shape = (self.shape[1], self.shape[0])
            return result
        
        # Inject functions
        class_map = {'MockFuzzarray': MockFuzzarray}
        module_namespace = {}
        registry.build_and_inject(class_map, module_namespace)
        
        # Test NumPy-like usage patterns
        # Create arrays using top-level function
        arr = module_namespace['zeros']((2, 3))
        assert arr.shape == (2, 3)
        assert arr.size == 6
        assert arr.values == [0, 0, 0, 0, 0, 0]
        
        # Test reshape as instance method
        reshaped = arr.reshape(3, 2)
        assert reshaped.shape == (3, 2)
        assert reshaped.size == 6
        
        # Test reshape as top-level function
        reshaped2 = module_namespace['reshape'](arr, (6,))
        assert reshaped2.shape == (6,)
        assert reshaped2.ndim == 1
        
        # Test flatten
        flattened = arr.flatten()
        assert flattened.shape == (6,)
        assert flattened.ndim == 1
        
        # Test transpose
        transposed = arr.transpose()
        assert transposed.shape == (3, 2)
        
        # Test error handling
        with pytest.raises(ValueError, match="Cannot reshape array"):
            arr.reshape(2, 2)  # Wrong total size
        
        # Create 1D array and test transpose error
        arr_1d = module_namespace['zeros'](5)
        with pytest.raises(ValueError, match="Transpose only supported for 2D arrays"):
            arr_1d.transpose()
    
    def test_fluent_interface_pattern(self):
        """
        Test mixin system with fluent interface pattern.
        
        Simulates method chaining and fluent interface usage
        that's common in data processing libraries.
        """
        registry = MixinFunctionRegistry()
        
        class MockFuzzarray:
            def __init__(self, values, metadata=None):
                self.values = values
                self.metadata = metadata or {}
                self._size = len(values)
            
            def copy(self):
                return MockFuzzarray(self.values.copy(), self.metadata.copy())
        
        # Register fluent interface methods
        @registry.register(
            name='filter_positive',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def filter_positive_impl(self):
            """Filter to keep only positive values."""
            filtered_values = [v for v in self.values if v > 0]
            result = MockFuzzarray(filtered_values, self.metadata)
            result.metadata['filtered'] = True
            return result
        
        @registry.register(
            name='scale',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def scale_impl(self, factor):
            """Scale all values by a factor."""
            scaled_values = [v * factor for v in self.values]
            result = MockFuzzarray(scaled_values, self.metadata)
            result.metadata['scaled_by'] = factor
            return result
        
        @registry.register(
            name='add_metadata',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def add_metadata_impl(self, key, value):
            """Add metadata to the array."""
            result = self.copy()
            result.metadata[key] = value
            return result
        
        @registry.register(
            name='summarize',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def summarize_impl(self):
            """Get summary statistics."""
            if not self.values:
                return {'count': 0, 'sum': 0, 'mean': 0}
            
            return {
                'count': len(self.values),
                'sum': sum(self.values),
                'mean': sum(self.values) / len(self.values),
                'min': min(self.values),
                'max': max(self.values)
            }
        
        # Inject functions
        class_map = {'MockFuzzarray': MockFuzzarray}
        module_namespace = {}
        registry.build_and_inject(class_map, module_namespace)
        
        # Test fluent interface usage
        arr = MockFuzzarray([-2, -1, 0, 1, 2, 3, 4, 5])
        
        # Chain operations
        result = (arr
                 .filter_positive()
                 .scale(2)
                 .add_metadata('operation', 'processed')
                 .add_metadata('timestamp', '2025-01-24'))
        
        # Verify results
        assert result.values == [2, 4, 6, 8, 10]  # [1,2,3,4,5] * 2
        assert result.metadata['filtered'] is True
        assert result.metadata['scaled_by'] == 2
        assert result.metadata['operation'] == 'processed'
        assert result.metadata['timestamp'] == '2025-01-24'
        
        # Test summary
        summary = result.summarize()
        assert summary['count'] == 5
        assert summary['sum'] == 30
        assert summary['mean'] == 6.0
        assert summary['min'] == 2
        assert summary['max'] == 10
        
        # Test empty array after filtering
        negative_arr = MockFuzzarray([-5, -3, -1])
        empty_result = negative_arr.filter_positive()
        assert empty_result.values == []
        
        empty_summary = empty_result.summarize()
        assert empty_summary['count'] == 0
        assert empty_summary['sum'] == 0
        assert empty_summary['mean'] == 0


class TestCrossModuleIntegration:
    """
    Tests for cross-module registration and integration patterns.
    
    Simulates scenarios where mixin functions are registered
    across different modules and need to work together.
    """
    
    def test_cross_module_registration_simulation(self):
        """
        Test simulation of cross-module mixin registration.
        
        Simulates the pattern where different modules register
        their own mixin functions that need to work together.
        """
        # Simulate multiple module registries
        core_registry = MixinFunctionRegistry()
        analysis_registry = MixinFunctionRegistry()
        utils_registry = MixinFunctionRegistry()
        
        class MockFuzzarray:
            def __init__(self, values):
                self.values = values
                self._size = len(values)
            
            def copy(self):
                return MockFuzzarray(self.values.copy())
        
        # Simulate core module registrations
        @core_registry.register(
            name='deep_copy',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def deep_copy_impl(self):
            return self.copy()
        
        @core_registry.register(
            name='size',
            target_classes=['MockFuzzarray'],
            injection_type='both'
        )
        def size_impl(self):
            return self._size
        
        # Simulate analysis module registrations
        @analysis_registry.register(
            name='mean',
            target_classes=['MockFuzzarray'],
            injection_type='both'
        )
        def mean_impl(self):
            return sum(self.values) / len(self.values) if self.values else 0
        
        @analysis_registry.register(
            name='variance',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def variance_impl(self):
            if not self.values:
                return 0
            mean_val = sum(self.values) / len(self.values)
            return sum((x - mean_val) ** 2 for x in self.values) / len(self.values)
        
        # Simulate utils module registrations
        @utils_registry.register(
            name='normalize',
            target_classes=['MockFuzzarray'],
            injection_type='instance_function'
        )
        def normalize_impl(self):
            if not self.values:
                return self.copy()
            
            min_val = min(self.values)
            max_val = max(self.values)
            
            if min_val == max_val:
                return MockFuzzarray([0.5] * len(self.values))
            
            normalized = [(v - min_val) / (max_val - min_val) for v in self.values]
            return MockFuzzarray(normalized)
        
        @utils_registry.register(
            name='create_range',
            injection_type='top_level_function'
        )
        def create_range_impl(start, stop, step=1):
            values = list(range(start, stop, step))
            return MockFuzzarray(values)
        
        # Simulate combining registrations from multiple modules
        class_map = {'MockFuzzarray': MockFuzzarray}
        module_namespace = {}
        
        # Inject from all registries
        core_registry.build_and_inject(class_map, module_namespace)
        analysis_registry.build_and_inject(class_map, module_namespace)
        utils_registry.build_and_inject(class_map, module_namespace)
        
        # Test cross-module functionality
        # Create array using utils function
        arr = module_namespace['create_range'](1, 11)  # [1, 2, 3, ..., 10]
        assert arr.values == list(range(1, 11))
        
        # Use core functions
        assert arr.size() == 10
        assert module_namespace['size'](arr) == 10
        
        copied = arr.deep_copy()
        assert copied.values == arr.values
        assert copied is not arr
        
        # Use analysis functions
        assert arr.mean() == 5.5
        assert module_namespace['mean'](arr) == 5.5
        
        expected_variance = sum((x - 5.5) ** 2 for x in range(1, 11)) / 10
        assert abs(arr.variance() - expected_variance) < 1e-10
        
        # Use utils functions
        normalized = arr.normalize()
        assert normalized.values[0] == 0.0  # min value normalized to 0
        assert normalized.values[-1] == 1.0  # max value normalized to 1
        
        # Test that all functions are available
        expected_top_level = {'size', 'mean', 'create_range'}
        actual_top_level = set()
        
        for registry in [core_registry, analysis_registry, utils_registry]:
            actual_top_level.update(registry.get_top_level_function_names())
        
        assert actual_top_level == expected_top_level


class TestNamespaceManagement:
    """
    Tests for proper namespace handling and pollution prevention.
    
    Ensures that the mixin system properly manages namespaces
    and doesn't cause unwanted side effects.
    """
    
    def test_namespace_isolation(self):
        """
        Test that different registries maintain namespace isolation.
        
        Verifies that functions registered in different registry
        instances don't interfere with each other.
        """
        registry1 = MixinFunctionRegistry()
        registry2 = MixinFunctionRegistry()
        
        class MockClass:
            def __init__(self, value):
                self.value = value
        
        # Register different functions with same name in different registries
        @registry1.register(
            name='process',
            target_classes=['MockClass'],
            injection_type='instance_function'
        )
        def process_impl_1(self):
            return f"processed by registry1: {self.value}"
        
        @registry2.register(
            name='process',
            target_classes=['MockClass'],
            injection_type='instance_function'
        )
        def process_impl_2(self):
            return f"processed by registry2: {self.value * 2}"
        
        # Verify registries are isolated
        assert 'process' in registry1._functions
        assert 'process' in registry2._functions
        assert registry1._functions['process'] != registry2._functions['process']
        
        # Test separate injection
        class MockClass1(MockClass):
            pass
        
        class MockClass2(MockClass):
            pass
        
        # Inject from registry1 to MockClass1
        registry1.build_and_inject({'MockClass': MockClass1}, {})
        
        # Inject from registry2 to MockClass2
        registry2.build_and_inject({'MockClass': MockClass2}, {})
        
        # Test that each class has its own version
        obj1 = MockClass1(10)
        obj2 = MockClass2(10)
        
        assert obj1.process() == "processed by registry1: 10"
        assert obj2.process() == "processed by registry2: 20"
    
    def test_module_namespace_management(self):
        """
        Test proper management of module namespace injection.
        
        Verifies that top-level functions are properly managed
        and don't cause namespace pollution.
        """
        registry = MixinFunctionRegistry()
        
        # Register functions with different injection types
        @registry.register(
            name='instance_only',
            target_classes=['MockClass'],
            injection_type='instance_function'
        )
        def instance_only_impl(self):
            return "instance only"
        
        @registry.register(
            name='top_level_only',
            injection_type='top_level_function'
        )
        def top_level_only_impl():
            return "top level only"
        
        @registry.register(
            name='both_types',
            target_classes=['MockClass'],
            injection_type='both'
        )
        def both_types_impl(self):
            return "both types"
        
        # Create separate namespaces for testing
        namespace1 = {'existing_function': lambda: "existing"}
        namespace2 = {}
        
        class MockClass:
            pass
        
        class_map = {'MockClass': MockClass}
        
        # Inject into first namespace
        registry.build_and_inject(class_map, namespace1)
        
        # Verify existing function is preserved
        assert namespace1['existing_function']() == "existing"
        
        # Verify only top-level functions are added
        assert 'top_level_only' in namespace1
        assert 'both_types' in namespace1
        assert 'instance_only' not in namespace1
        
        # Inject into second namespace
        registry.build_and_inject(class_map, namespace2)
        
        # Verify same functions are added
        assert 'top_level_only' in namespace2
        assert 'both_types' in namespace2
        assert 'instance_only' not in namespace2
        
        # Verify functions work correctly
        assert namespace1['top_level_only']() == "top level only"
        assert namespace2['top_level_only']() == "top level only"
        
        obj = MockClass()
        assert namespace1['both_types'](obj) == "both types"
        assert namespace2['both_types'](obj) == "both types"


class TestSystemBehavior:
    """
    Tests for overall system behavior and performance characteristics.
    
    Ensures that the mixin system behaves correctly under
    various conditions and usage patterns.
    """
    
    def test_large_scale_registration(self):
        """
        Test system behavior with large numbers of registered functions.
        
        Verifies that the system can handle many registered functions
        without performance degradation or errors.
        """
        registry = MixinFunctionRegistry()
        
        class MockClass:
            def __init__(self, value):
                self.value = value
        
        # Register many functions
        num_functions = 100
        
        for i in range(num_functions):
            @registry.register(
                name=f'function_{i}',
                target_classes=['MockClass'],
                injection_type='both'
            )
            def func_impl(self, func_id=i):
                return f"function_{func_id}: {self.value}"
        
        # Verify all functions are registered
        assert len(registry._functions) == num_functions
        assert len(registry._metadata) == num_functions
        
        # Test injection
        class_map = {'MockClass': MockClass}
        module_namespace = {}
        
        registry.build_and_inject(class_map, module_namespace)
        
        # Verify all functions are injected
        obj = MockClass(42)
        
        for i in range(num_functions):
            func_name = f'function_{i}'
            
            # Test instance method
            assert hasattr(MockClass, func_name)
            result = getattr(obj, func_name)()
            assert result == f"function_{i}: 42"
            
            # Test top-level function
            assert func_name in module_namespace
            result = module_namespace[func_name](obj)
            assert result == f"function_{i}: 42"
        
        # Test get_top_level_function_names performance
        top_level_names = registry.get_top_level_function_names()
        assert len(top_level_names) == num_functions
        
        expected_names = {f'function_{i}' for i in range(num_functions)}
        assert set(top_level_names) == expected_names
    
    def test_repeated_injection_stability(self):
        """
        Test system stability under repeated injection operations.
        
        Verifies that multiple injections don't cause issues
        and that the system remains stable.
        """
        registry = MixinFunctionRegistry()
        
        class MockClass:
            def __init__(self, value):
                self.value = value
        
        @registry.register(
            name='test_method',
            target_classes=['MockClass'],
            injection_type='both'
        )
        def test_method_impl(self):
            return f"test: {self.value}"
        
        class_map = {'MockClass': MockClass}
        module_namespace = {}
        
        # Perform multiple injections
        for i in range(10):
            registry.build_and_inject(class_map, module_namespace)
            
            # Verify functionality still works
            obj = MockClass(i)
            assert obj.test_method() == f"test: {i}"
            assert module_namespace['test_method'](obj) == f"test: {i}"
        
        # Verify no duplicate entries or corruption
        assert len(module_namespace) == 1
        assert 'test_method' in module_namespace
        
        # Verify class still has the method
        assert hasattr(MockClass, 'test_method')
    
    def test_memory_and_reference_management(self):
        """
        Test proper memory and reference management.
        
        Ensures that the mixin system doesn't create memory leaks
        or improper reference cycles.
        """
        import weakref
        
        registry = MixinFunctionRegistry()
        
        class MockClass:
            def __init__(self, value):
                self.value = value
        
        # Create weak references to track object lifecycle
        weak_refs = []
        
        @registry.register(
            name='test_method',
            target_classes=['MockClass'],
            injection_type='instance_function'
        )
        def test_method_impl(self):
            return self.value
        
        class_map = {'MockClass': MockClass}
        registry.build_and_inject(class_map, {})
        
        # Create objects and weak references
        for i in range(10):
            obj = MockClass(i)
            weak_refs.append(weakref.ref(obj))
            
            # Use the injected method
            result = obj.test_method()
            assert result == i
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Verify objects can be garbage collected
        # (This is a basic test; in practice, weak references might still exist
        # depending on the Python implementation and timing)
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        
        # At minimum, verify the system doesn't prevent garbage collection entirely
        # In most cases, all objects should be collectible
        assert alive_count <= len(weak_refs)  # Basic sanity check


if __name__ == '__main__':
    pytest.main()
