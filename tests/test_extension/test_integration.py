#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Integration tests for the AxisFuzzy extension system.

This module provides comprehensive end-to-end integration tests that verify
the complete extension system workflow, from function registration through
injection to runtime usage. These tests ensure that all components work
together correctly and that the extension system integrates properly with
the core AxisFuzzy classes.

Test Coverage
-------------
- End-to-end extension registration and injection workflow
- Integration with actual Fuzznum and Fuzzarray classes (mocked)
- Cross-component function registration and usage
- Real-world usage scenarios and patterns
- Performance and behavior under realistic conditions
- Interaction between different injection types
- Module namespace management and cleanup
- Error propagation through the complete system
- Concurrent registration and injection scenarios

Test Classes
------------
TestExtensionSystemIntegration : End-to-end system integration tests
TestRealWorldUsagePatterns : Tests based on realistic usage scenarios
TestCrossComponentIntegration : Tests for cross-component registration patterns
TestNamespaceManagement : Tests for proper namespace handling
TestSystemBehavior : Tests for overall system behavior and performance
TestErrorPropagation : Tests for error handling across the complete system
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from axisfuzzy.extension.registry import ExtensionRegistry, get_registry_extension
from axisfuzzy.extension.dispatcher import ExtensionDispatcher
from axisfuzzy.extension.injector import ExtensionInjector, get_extension_injector
from axisfuzzy.extension import extension, apply_extensions


class TestExtensionSystemIntegration:
    """
    End-to-end integration tests for the complete extension system.
    
    Tests the entire workflow from registration to runtime usage,
    ensuring all components work together correctly.
    """
    
    def setup_method(self):
        """Set up test fixtures with fresh components."""
        # Create fresh instances for isolated testing
        self.registry = ExtensionRegistry()
        self.dispatcher = ExtensionDispatcher()
        self.injector = ExtensionInjector()
        
        # Wire components together
        self.dispatcher.registry = self.registry
        self.injector.registry = self.registry
        self.injector.dispatcher = self.dispatcher
        
        # Create mock classes that simulate Fuzznum and Fuzzarray
        class MockFuzznum:
            def __init__(self, mtype='qrofn', **kwargs):
                self.mtype = mtype
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class MockFuzzarray:
            def __init__(self, mtype='qrofn', **kwargs):
                self.mtype = mtype
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.MockFuzznum = MockFuzznum
        self.MockFuzzarray = MockFuzzarray
        
        self.class_map = {
            'Fuzznum': MockFuzznum,
            'Fuzzarray': MockFuzzarray
        }
        self.module_namespace = {}
    
    def test_complete_extension_system_workflow(self):
        """
        Test complete extension system workflow from registration to usage.
        
        This test simulates the complete lifecycle of extension functions:
        1. Registration of functions with different injection types
        2. Injection into target classes and module namespace
        3. Runtime usage of injected functions
        4. Verification of correct behavior
        """
        # Step 1: Register multiple functions with different types
        @self.registry.register(
            'distance', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrofn_distance(self, other):
            return f"qrofn distance between {self.mtype} and {other.mtype}"
        
        @self.registry.register(
            'distance', mtype='qrohfn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrohfn_distance(self, other):
            return f"qrohfn distance between {self.mtype} and {other.mtype}"
        
        @self.registry.register(
            'score', mtype='qrofn',
            target_classes=['Fuzznum', 'Fuzzarray'], injection_type='instance_property'
        )
        def qrofn_score(self):
            return f"qrofn score for {self.mtype}"
        
        @self.registry.register(
            'create_positive', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def qrofn_create_positive(mtype='qrofn'):
            return f"created positive {mtype} number"
        
        @self.registry.register(
            'normalize', mtype=None,  # Default implementation
            target_classes=['Fuzznum'], injection_type='both', is_default=True
        )
        def default_normalize(obj):
            return f"normalized {obj.mtype} object"
        
        # Step 2: Perform injection
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Step 3: Test runtime usage
        
        # Test instance methods with different mtypes
        obj1 = self.MockFuzznum('qrofn')
        obj2 = self.MockFuzznum('qrohfn')
        obj3 = self.MockFuzznum('unknown_type')
        
        # Test specialized implementations
        assert hasattr(self.MockFuzznum, 'distance')
        assert obj1.distance(obj2) == "qrofn distance between qrofn and qrohfn"
        assert obj2.distance(obj1) == "qrohfn distance between qrohfn and qrofn"
        
        # Test fallback to default for unknown type
        assert obj3.normalize() == "normalized unknown_type object"
        
        # Test instance properties
        assert hasattr(self.MockFuzznum, 'score')
        assert obj1.score == "qrofn score for qrofn"
        
        # Test array objects
        arr1 = self.MockFuzzarray('qrofn')
        assert arr1.score == "qrofn score for qrofn"
        
        # Test top-level functions
        assert 'create_positive' in self.module_namespace
        assert self.module_namespace['create_positive']() == "created positive qrofn number"
        
        # Test 'both' injection type
        assert hasattr(self.MockFuzznum, 'normalize')  # Instance method
        assert 'normalize' in self.module_namespace     # Top-level function
        
        # Test both forms work
        assert obj1.normalize() == "normalized qrofn object"
        assert self.module_namespace['normalize'](obj1) == "normalized qrofn object"
    
    def test_extension_system_error_propagation(self):
        """
        Test that errors are properly propagated through the extension system.
        
        Verifies that exceptions raised in extension functions are correctly
        propagated to the caller without being masked by the injection system.
        """
        # Register a function that raises an exception
        @self.registry.register(
            'error_method', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def error_implementation(self):
            raise ValueError("Test error from extension function")
        
        # Inject the function
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test that the error is properly propagated
        obj = self.MockFuzznum('qrofn')
        with pytest.raises(ValueError, match="Test error from extension function"):
            obj.error_method()
    
    def test_extension_system_with_priority_handling(self):
        """
        Test extension system behavior with priority-based registration.
        
        Verifies that higher priority implementations override lower priority ones.
        """
        # Register low priority implementation
        @self.registry.register(
            'priority_test', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method',
            priority=1
        )
        def low_priority_impl(self):
            return "low priority"
        
        # Register high priority implementation
        @self.registry.register(
            'priority_test', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method',
            priority=10
        )
        def high_priority_impl(self):
            return "high priority"
        
        # Inject and test
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        obj = self.MockFuzznum('qrofn')
        assert obj.priority_test() == "high priority"
    
    def test_extension_system_concurrent_registration(self):
        """
        Test extension system behavior under concurrent registration.
        
        Verifies thread safety of the registration and injection process.
        """
        results = []
        errors = []
        
        def register_function(name_suffix, mtype):
            try:
                @self.registry.register(
                    f'concurrent_test_{name_suffix}', mtype=mtype,
                    target_classes=['Fuzznum'], injection_type='instance_method'
                )
                def concurrent_impl(self):
                    return f"result_{name_suffix}_{mtype}"
                
                results.append(f"registered_{name_suffix}_{mtype}")
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads for concurrent registration
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=register_function,
                args=(i, 'qrofn' if i % 2 == 0 else 'qrohfn')
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent registration: {errors}"
        assert len(results) == 10
        
        # Verify injection works after concurrent registration
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test that all functions were registered and work correctly
        obj_qrofn = self.MockFuzznum('qrofn')
        obj_qrohfn = self.MockFuzznum('qrohfn')
        
        for i in range(10):
            method_name = f'concurrent_test_{i}'
            assert hasattr(self.MockFuzznum, method_name)
            
            if i % 2 == 0:  # qrofn
                assert getattr(obj_qrofn, method_name)() == f"result_{i}_qrofn"
            else:  # qrohfn
                assert getattr(obj_qrohfn, method_name)() == f"result_{i}_qrohfn"


class TestRealWorldUsagePatterns:
    """
    Tests based on realistic usage scenarios.
    
    Simulates how the extension system would be used in practice
    with multiple modules and complex registration patterns.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ExtensionRegistry()
        self.dispatcher = ExtensionDispatcher()
        self.injector = ExtensionInjector()
        
        # Wire components
        self.dispatcher.registry = self.registry
        self.injector.registry = self.registry
        self.injector.dispatcher = self.dispatcher
        
        # Mock classes
        class MockFuzznum:
            def __init__(self, mtype='qrofn', md=0.5, nmd=0.3, q=2):
                self.mtype = mtype
                self.md = md
                self.nmd = nmd
                self.q = q
        
        self.MockFuzznum = MockFuzznum
        self.class_map = {'Fuzznum': MockFuzznum}
        self.module_namespace = {}
    
    def test_mathematical_operations_extension_pattern(self):
        """
        Test realistic mathematical operations extension pattern.
        
        Simulates extending the system with mathematical operations
        for different fuzzy number types.
        """
        # Store reference to MockFuzznum for the closure
        MockFuzznum = self.MockFuzznum
        
        # Register mathematical operations for different types
        @self.registry.register(
            'add', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrofn_add(self, other):
            return self.__class__(
                'qrofn',
                md=min(1.0, self.md + other.md),
                nmd=max(0.0, self.nmd + other.nmd - 1.0),
                q=self.q
            )
        
        @self.registry.register(
            'multiply', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrofn_multiply(self, other):
            return self.__class__(
                'qrofn',
                md=self.md * other.md,
                nmd=self.nmd + other.nmd - self.nmd * other.nmd,
                q=self.q
            )
        
        @self.registry.register(
            'distance', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrofn_distance(self, other):
            return ((abs(self.md**self.q - other.md**self.q)**2 + 
                    abs(self.nmd**self.q - other.nmd**self.q)**2) / 2) ** 0.5
        
        # Register utility functions
        @self.registry.register(
            'create_zero', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def qrofn_create_zero():
            return MockFuzznum('qrofn', md=0.0, nmd=1.0, q=2)
        
        @self.registry.register(
            'create_one', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def qrofn_create_one():
            return MockFuzznum('qrofn', md=1.0, nmd=0.0, q=2)
        
        # Inject all functions
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test realistic usage scenario
        a = self.MockFuzznum('qrofn', md=0.7, nmd=0.2, q=2)
        b = self.MockFuzznum('qrofn', md=0.6, nmd=0.3, q=2)
        
        # Test mathematical operations
        c = a.add(b)
        assert c.mtype == 'qrofn'
        assert c.md == min(1.0, 0.7 + 0.6)
        assert c.nmd == max(0.0, 0.2 + 0.3 - 1.0)
        
        d = a.multiply(b)
        assert d.mtype == 'qrofn'
        assert d.md == 0.7 * 0.6
        assert d.nmd == 0.2 + 0.3 - 0.2 * 0.3
        
        # Test distance calculation
        dist = a.distance(b)
        assert isinstance(dist, float)
        assert dist >= 0
        
        # Test utility functions
        zero = self.module_namespace['create_zero']()
        one = self.module_namespace['create_one']()
        
        assert zero.md == 0.0 and zero.nmd == 1.0
        assert one.md == 1.0 and one.nmd == 0.0
    
    def test_analysis_extension_pattern(self):
        """
        Test realistic analysis extension pattern.
        
        Simulates extending the system with analysis functions
        that might be used in data analysis scenarios.
        """
        # Register analysis properties
        @self.registry.register(
            'score', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def qrofn_score(self):
            return self.md**self.q - self.nmd**self.q
        
        @self.registry.register(
            'accuracy', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def qrofn_accuracy(self):
            return self.md - self.nmd
        
        @self.registry.register(
            'indeterminacy', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def qrofn_indeterminacy(self):
            return 1 - self.md - self.nmd
        
        # Register analysis methods
        @self.registry.register(
            'is_positive', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrofn_is_positive(self):
            return self.md > self.nmd
        
        @self.registry.register(
            'is_negative', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrofn_is_negative(self):
            return self.nmd > self.md
        
        @self.registry.register(
            'complement', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def qrofn_complement(self):
            return self.__class__('qrofn', md=self.nmd, nmd=self.md, q=self.q)
        
        # Inject all functions
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test analysis scenario
        positive_num = self.MockFuzznum('qrofn', md=0.8, nmd=0.1, q=2)
        negative_num = self.MockFuzznum('qrofn', md=0.2, nmd=0.7, q=2)
        neutral_num = self.MockFuzznum('qrofn', md=0.4, nmd=0.4, q=2)
        
        # Test properties
        assert positive_num.score > 0
        assert negative_num.score < 0
        assert abs(neutral_num.score) < 0.1  # Close to zero
        
        assert positive_num.accuracy > 0
        assert negative_num.accuracy < 0
        assert neutral_num.accuracy == 0
        
        # Test methods
        assert positive_num.is_positive() is True
        assert positive_num.is_negative() is False
        
        assert negative_num.is_positive() is False
        assert negative_num.is_negative() is True
        
        assert neutral_num.is_positive() is False
        assert neutral_num.is_negative() is False
        
        # Test complement
        comp = positive_num.complement()
        assert comp.md == positive_num.nmd
        assert comp.nmd == positive_num.md


class TestCrossComponentIntegration:
    """
    Tests for cross-component registration and integration patterns.
    
    Simulates scenarios where different components register
    their own extension functions that need to work together.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ExtensionRegistry()
        self.dispatcher = ExtensionDispatcher()
        self.injector = ExtensionInjector()
        
        # Wire components
        self.dispatcher.registry = self.registry
        self.injector.registry = self.registry
        self.injector.dispatcher = self.dispatcher
        
        # Mock classes
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.MockFuzznum = MockFuzznum
        self.class_map = {'Fuzznum': MockFuzznum}
        self.module_namespace = {}
    
    def test_cross_component_registration_simulation(self):
        """
        Test simulation of cross-component extension registration.
        
        Simulates the pattern where different components register
        their own extension functions that need to work together.
        """
        # Simulate core module registrations
        @self.registry.register(
            'core_method', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def core_implementation(self):
            return f"core method for {self.mtype}"
        
        # Simulate analysis module registrations
        @self.registry.register(
            'analyze', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def analysis_implementation(self):
            # This method uses the core method
            core_result = self.core_method()
            return f"analysis of {core_result}"
        
        # Simulate visualization module registrations
        @self.registry.register(
            'plot', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def visualization_implementation(self):
            # This method uses both core and analysis methods
            analysis_result = self.analyze()
            return f"plotting {analysis_result}"
        
        # Simulate utility module registrations
        @self.registry.register(
            'export_data', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def utility_export(obj):
            plot_result = obj.plot()
            return f"exported {plot_result}"
        
        # Inject all functions
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test cross-component integration
        obj = self.MockFuzznum('qrofn')
        
        # Test that all methods are available
        assert hasattr(obj, 'core_method')
        assert hasattr(obj, 'analyze')
        assert hasattr(obj, 'plot')
        assert 'export_data' in self.module_namespace
        
        # Test that methods can call each other
        core_result = obj.core_method()
        assert core_result == "core method for qrofn"
        
        analysis_result = obj.analyze()
        assert analysis_result == "analysis of core method for qrofn"
        
        plot_result = obj.plot()
        assert plot_result == "plotting analysis of core method for qrofn"
        
        export_result = self.module_namespace['export_data'](obj)
        assert export_result == "exported plotting analysis of core method for qrofn"
    
    def test_component_dependency_chain(self):
        """
        Test complex dependency chains between components.
        
        Verifies that components can build upon each other's functionality
        in complex dependency chains.
        """
        # Base component
        @self.registry.register(
            'base_value', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def base_value_impl(self):
            return 42
        
        # Component that depends on base
        @self.registry.register(
            'derived_value', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def derived_value_impl(self):
            return self.base_value * 2
        
        # Component that depends on derived
        @self.registry.register(
            'final_value', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def final_value_impl(self):
            return self.derived_value + 10
        
        # Method that uses the final value
        @self.registry.register(
            'compute_result', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def compute_result_impl(self):
            return f"Final computation: {self.final_value}"
        
        # Inject all functions
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test dependency chain
        obj = self.MockFuzznum('qrofn')
        
        assert obj.base_value == 42
        assert obj.derived_value == 84  # 42 * 2
        assert obj.final_value == 94    # 84 + 10
        assert obj.compute_result() == "Final computation: 94"


class TestNamespaceManagement:
    """
    Tests for proper namespace handling and management.
    
    Verifies that top-level functions are properly managed
    and don't cause namespace pollution or conflicts.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ExtensionRegistry()
        self.dispatcher = ExtensionDispatcher()
        self.injector = ExtensionInjector()
        
        # Wire components
        self.dispatcher.registry = self.registry
        self.injector.registry = self.registry
        self.injector.dispatcher = self.dispatcher
        
        # Mock classes
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.MockFuzznum = MockFuzznum
        self.class_map = {'Fuzznum': MockFuzznum}
        self.module_namespace = {}
    
    def test_module_namespace_management(self):
        """
        Test proper management of module namespace injection.
        
        Verifies that top-level functions are properly managed
        and don't cause namespace pollution.
        """
        # Register functions with different injection types
        @self.registry.register(
            'instance_only', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def instance_only_impl(self):
            return "instance only"
        
        @self.registry.register(
            'top_level_only', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def top_level_only_impl(obj):
            return "top level only"
        
        @self.registry.register(
            'both_types', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='both'
        )
        def both_types_impl(obj):
            return "both types"
        
        # Inject functions
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Verify namespace contents
        expected_top_level = {'top_level_only', 'both_types'}
        actual_top_level = set(self.module_namespace.keys())
        assert actual_top_level == expected_top_level
        
        # Verify instance methods
        obj = self.MockFuzznum('qrofn')
        assert hasattr(obj, 'instance_only')
        assert hasattr(obj, 'both_types')
        assert not hasattr(obj, 'top_level_only')  # Should not be instance method
        
        # Test functionality
        assert obj.instance_only() == "instance only"
        assert obj.both_types() == "both types"
        assert self.module_namespace['top_level_only'](obj) == "top level only"
        assert self.module_namespace['both_types'](obj) == "both types"
    
    def test_namespace_conflict_prevention(self):
        """
        Test prevention of namespace conflicts.
        
        Verifies that the system handles potential naming conflicts
        appropriately without overwriting existing functions.
        """
        # Pre-populate namespace with existing function
        def existing_function():
            return "existing"
        
        self.module_namespace['test_function'] = existing_function
        
        # Try to register a function with the same name
        @self.registry.register(
            'test_function', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def new_function_impl(obj):
            return "new function"
        
        # Inject functions
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Verify that existing function was not overwritten
        assert self.module_namespace['test_function']() == "existing"
    
    def test_namespace_cleanup_behavior(self):
        """
        Test namespace cleanup behavior.
        
        Verifies that repeated injections don't cause accumulation
        of stale references or memory leaks.
        """
        @self.registry.register(
            'cleanup_test', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def cleanup_test_impl(obj):
            return "cleanup test"
        
        # Perform multiple injections
        for i in range(5):
            self.injector.inject_all(self.class_map, self.module_namespace)
            
            # Verify function is still available and working
            assert 'cleanup_test' in self.module_namespace
            obj = self.MockFuzznum('qrofn')
            assert self.module_namespace['cleanup_test'](obj) == "cleanup test"
        
        # Verify no accumulation of duplicate entries
        assert len([k for k in self.module_namespace.keys() if 'cleanup_test' in k]) == 1


class TestSystemBehavior:
    """
    Tests for overall system behavior and performance characteristics.
    
    Verifies that the extension system behaves correctly under
    various conditions and maintains good performance characteristics.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ExtensionRegistry()
        self.dispatcher = ExtensionDispatcher()
        self.injector = ExtensionInjector()
        
        # Wire components
        self.dispatcher.registry = self.registry
        self.injector.registry = self.registry
        self.injector.dispatcher = self.dispatcher
        
        # Mock classes
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.MockFuzznum = MockFuzznum
        self.class_map = {'Fuzznum': MockFuzznum}
        self.module_namespace = {}
    
    def test_system_scalability_with_many_functions(self):
        """
        Test system behavior with a large number of registered functions.
        
        Verifies that the system scales well with many registered functions.
        """
        # Register many functions
        num_functions = 100
        for i in range(num_functions):
            @self.registry.register(
                f'function_{i}', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def func_impl(self, func_id=i):
                return f"function_{func_id}_result"
        
        # Inject all functions
        start_time = time.time()
        self.injector.inject_all(self.class_map, self.module_namespace)
        injection_time = time.time() - start_time
        
        # Verify all functions were injected
        obj = self.MockFuzznum('qrofn')
        for i in range(num_functions):
            assert hasattr(obj, f'function_{i}')
        
        # Test performance of function calls
        start_time = time.time()
        for i in range(min(10, num_functions)):  # Test first 10 functions
            result = getattr(obj, f'function_{i}')()
            assert f'function_{i}_result' in result
        call_time = time.time() - start_time
        
        # Basic performance assertions (these are quite lenient)
        assert injection_time < 5.0  # Should inject 100 functions in under 5 seconds
        assert call_time < 1.0       # Should call 10 functions in under 1 second
    
    def test_system_memory_efficiency(self):
        """
        Test system memory efficiency.
        
        Verifies that the system doesn't create excessive memory overhead.
        """
        import gc
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Register and inject many functions
        num_functions = 50
        for i in range(num_functions):
            @self.registry.register(
                f'memory_test_{i}', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def memory_func_impl(self, func_id=i):
                return f"memory_test_{func_id}"
        
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Check memory state after injection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # The increase should be reasonable (this is a rough heuristic)
        object_increase = final_objects - initial_objects
        # Allow for some overhead, but it shouldn't be excessive
        assert object_increase < num_functions * 10  # Very lenient upper bound
    
    def test_system_injection_idempotency(self):
        """
        Test that multiple injections are idempotent.
        
        Verifies that calling inject_all multiple times
        doesn't cause problems or change behavior.
        """
        @self.registry.register(
            'idempotency_test', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def idempotency_impl(self):
            return "idempotency test result"
        
        # Perform injection multiple times
        for i in range(5):
            self.injector.inject_all(self.class_map, self.module_namespace)
            
            # Verify function is available and works correctly
            obj = self.MockFuzznum('qrofn')
            assert hasattr(obj, 'idempotency_test')
            assert obj.idempotency_test() == "idempotency test result"
        
        # Verify no duplicate methods or other issues
        # This is a basic check - in a real implementation you might
        # want to check for specific signs of duplication
        assert hasattr(self.MockFuzznum, 'idempotency_test')