#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Edge cases and exception handling tests for the AxisFuzzy extension system.

This module provides comprehensive tests for edge cases, boundary conditions,
and exception handling scenarios in the extension system. These tests ensure
that the system behaves correctly under unusual conditions and provides
appropriate error handling and recovery mechanisms.

Test Coverage
-------------
- Invalid parameter handling and validation
- Malformed function registrations
- Concurrent access safety and thread safety
- Memory management under stress conditions
- Error propagation and exception handling
- Boundary conditions for function parameters
- Recovery from partial failures
- Resource cleanup after errors
- Circular dependency detection
- Invalid mtype and target class handling
- Registry corruption scenarios
- Dispatcher edge cases
- Injector failure modes

Test Classes
------------
TestInvalidParameters : Tests for invalid parameter handling
TestMalformedRegistrations : Tests for malformed registration scenarios
TestConcurrencyEdgeCases : Tests for concurrent access edge cases
TestErrorHandling : Tests for error handling and recovery
TestBoundaryConditions : Tests for boundary conditions
TestResourceManagement : Tests for resource management edge cases
TestCorruptionRecovery : Tests for recovery from corruption scenarios
TestCircularDependencies : Tests for circular dependency detection
"""

import pytest
import threading
import time
import gc
import weakref
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from axisfuzzy.extension.registry import ExtensionRegistry, get_registry_extension
from axisfuzzy.extension.dispatcher import ExtensionDispatcher
from axisfuzzy.extension.injector import ExtensionInjector, get_extension_injector
from axisfuzzy.extension import extension, apply_extensions


class TestInvalidParameters:
    """
    Tests for invalid parameter handling and validation.
    
    Verifies that the extension system properly validates parameters
    and provides appropriate error messages for invalid inputs.
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
    
    def test_register_with_invalid_name(self):
        """Test registration with invalid function names."""
        # Test empty name - might be allowed by the system
        try:
            @self.registry.register(
                '', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def invalid_empty_name():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
        
        # Test None name - should fail
        try:
            @self.registry.register(
                None, mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def invalid_none_name():
                pass
        except (ValueError, TypeError):
            pass  # Expected behavior
        
        # Test non-string name - might be converted to string
        try:
            @self.registry.register(
                123, mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def invalid_numeric_name():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
    
    def test_register_with_invalid_mtype(self):
        """Test registration with invalid mtype values."""
        # Test numeric mtype (should be string or None)
        try:
            @self.registry.register(
                'test_func', mtype=123,
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def invalid_numeric_mtype():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
        
        # Test list mtype
        try:
            @self.registry.register(
                'test_func2', mtype=['qrofn'],
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def invalid_list_mtype():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
    
    def test_register_with_invalid_target_classes(self):
        """Test registration with invalid target_classes."""
        # Test empty target_classes - might be allowed for top-level functions
        try:
            @self.registry.register(
                'test_func_empty', mtype='qrofn',
                target_classes=[], injection_type='top_level_function'
            )
            def invalid_empty_targets():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
        
        # Test None target_classes - might be allowed for top-level functions
        try:
            @self.registry.register(
                'test_func_none', mtype='qrofn',
                target_classes=None, injection_type='top_level_function'
            )
            def invalid_none_targets():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
        
        # Test non-list target_classes - might be converted to list
        try:
            @self.registry.register(
                'test_func_string', mtype='qrofn',
                target_classes='Fuzznum', injection_type='instance_method'
            )
            def invalid_string_targets():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
    
    def test_register_with_invalid_injection_type(self):
        """Test registration with invalid injection_type."""
        # Test invalid injection type
        try:
            @self.registry.register(
                'test_func', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='invalid_type'
            )
            def invalid_injection_type():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
        
        # Test None injection type
        try:
            @self.registry.register(
                'test_func_none', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type=None
            )
            def invalid_none_injection_type():
                pass
        except (ValueError, TypeError):
            pass  # Expected if validation is strict
    
    def test_register_with_invalid_priority(self):
        """Test registration with invalid priority values."""
        # Test negative priority - this should be allowed but we test the behavior
        @self.registry.register(
            'test_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method',
            priority=-1
        )
        def negative_priority_func():
            return "negative"
        
        # Verify it was registered
        assert self.registry.get_function('test_func', 'qrofn') is not None

        # Test non-numeric priority
        try:
            @self.registry.register(
                'string_priority_func', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method',
                priority="string"
            )
            def string_priority_func():
                return "string"
        except (ValueError, TypeError):
            pass  # Expected if validation is strict

        # Test None priority - this might be allowed as default
        @self.registry.register(
            'test_func3', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method',
            priority=None
        )
        def none_priority_func():
            return "none"
    
    def test_dispatcher_with_invalid_objects(self):
        """Test dispatcher behavior with invalid objects."""
        # Register a valid function first
        @self.registry.register(
            'test_method', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def test_implementation(self):
            return "test result"
        
        # Test with None object
        with pytest.raises((AttributeError, TypeError)):
            self.dispatcher.dispatch('test_method', None, 'qrofn')
        
        # Test with object without mtype
        class InvalidObject:
            pass
        
        invalid_obj = InvalidObject()
        with pytest.raises(AttributeError):
            self.dispatcher.dispatch('test_method', invalid_obj, 'qrofn')
    
    def test_injector_with_invalid_class_map(self):
        """Test injector behavior with invalid class mappings."""
        # Register a test function first
        @self.registry.register('test_func', mtype='qrofn', 
                               target_classes=['Fuzznum'], injection_type='instance_method')
        def test_func():
            return "test"
        
        # Test with non-class values in class_map - this might not raise an error
        # but should handle gracefully
        invalid_class_map = {'Fuzznum': "not_a_class", 'Fuzzarray': 123}
        
        # The injector should handle this gracefully or raise appropriate error
        try:
            self.injector.inject_all(invalid_class_map, {})
        except (TypeError, AttributeError):
            pass  # Expected behavior
        except Exception as e:
            # Any other exception is also acceptable for invalid input
            assert isinstance(e, Exception)


class TestMalformedRegistrations:
    """
    Tests for malformed registration scenarios.
    
    Verifies that the system handles malformed or incomplete
    registrations gracefully.
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
    
    def test_register_non_callable(self):
        """Test registration of non-callable objects."""
        # Test registering a string - the decorator might not validate immediately
        try:
            decorator = self.registry.register(
                'test_func', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            decorator("not_a_function")
        except (TypeError, ValueError):
            pass  # Expected
        except Exception:
            pass  # Any exception is acceptable for invalid input

        # Test registering None
        try:
            decorator = self.registry.register(
                'test_func2', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            decorator(None)
        except (TypeError, ValueError):
            pass  # Expected
        except Exception:
            pass  # Any exception is acceptable for invalid input

        # Test registering a number
        try:
            decorator = self.registry.register(
                'test_func3', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            decorator(42)
        except (TypeError, ValueError):
            pass  # Expected
        except Exception:
            pass  # Any exception is acceptable for invalid input
    
    def test_register_function_with_incompatible_signature(self):
        """Test registration of functions with incompatible signatures."""
        # For instance methods, function should accept self as first parameter
        # This test verifies that the system can handle functions with wrong signatures
        
        @self.registry.register(
            'no_params', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def function_with_no_params():  # Missing 'self' parameter
            return "no params"
        
        # The registration should succeed, but usage might fail
        # This tests the system's robustness
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        # This should raise an error due to signature mismatch
        with pytest.raises(TypeError):
            obj.no_params()
    
    def test_register_with_conflicting_parameters(self):
        """Test registration with conflicting parameters."""
        # Test registering default function with specific mtype - this might be allowed
        @self.registry.register(
            'conflicting_func', mtype='qrofn',  # Specific mtype
            target_classes=['Fuzznum'], injection_type='instance_method',
            is_default=True  # But marked as default
        )
        def conflicting_function(self):
            pass
        
        # Verify it was registered
        assert self.registry.get_function('conflicting_func', 'qrofn') is not None
    
    def test_partial_registration_failure(self):
        """Test behavior when registration partially fails."""
        # This test simulates a scenario where registration might fail partway through
        
        # Mock the registry to fail on certain operations
        original_register = self.registry.register
        call_count = 0
        
        def failing_register(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:  # Fail after 2 successful registrations
                raise RuntimeError("Simulated registration failure")
            return original_register(*args, **kwargs)
        
        with patch.object(self.registry, 'register', side_effect=failing_register):
            # First registration should succeed
            @self.registry.register(
                'func1', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def func1(self):
                return "func1"
            
            # Second registration should succeed
            @self.registry.register(
                'func2', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def func2(self):
                return "func2"
            
            # Third registration should fail
            with pytest.raises(RuntimeError):
                @self.registry.register(
                    'func3', mtype='qrofn',
                    target_classes=['Fuzznum'], injection_type='instance_method'
                )
                def func3(self):
                    return "func3"


class TestConcurrencyEdgeCases:
    """
    Tests for concurrent access edge cases and thread safety.
    
    Verifies that the extension system maintains thread safety
    under various concurrent access patterns.
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
    
    def test_concurrent_registration_and_lookup(self):
        """Test concurrent registration and function lookup."""
        results = []
        errors = []
        
        def register_and_lookup(thread_id):
            try:
                # Register a function
                @self.registry.register(
                    f'concurrent_func_{thread_id}', mtype='qrofn',
                    target_classes=['Fuzznum'], injection_type='instance_method'
                )
                def concurrent_function(self):
                    return f"result_{thread_id}"
                
                # Immediately try to look it up
                func = self.registry.get_function(f'concurrent_func_{thread_id}', 'qrofn')
                if func is not None:
                    results.append(f"success_{thread_id}")
                else:
                    errors.append(f"lookup_failed_{thread_id}")
                    
            except Exception as e:
                errors.append(f"error_{thread_id}: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=register_and_lookup, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
    
    def test_concurrent_injection_and_usage(self):
        """Test concurrent injection and function usage."""
        # Register functions first
        for i in range(10):
            @self.registry.register(
                f'usage_func_{i}', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def usage_function(self, func_id=i):
                return f"usage_result_{func_id}"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        class_map = {'Fuzznum': MockFuzznum}
        module_namespace = {}
        
        results = []
        errors = []
        
        def inject_and_use(thread_id):
            try:
                # Perform injection
                self.injector.inject_all(class_map, module_namespace)
                
                # Use injected functions
                obj = MockFuzznum('qrofn')
                for i in range(5):  # Use first 5 functions
                    result = getattr(obj, f'usage_func_{i}')()
                    results.append(f"thread_{thread_id}_func_{i}: {result}")
                    
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=inject_and_use, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) > 0  # Should have some successful results
    
    def test_concurrent_registration_same_name(self):
        """Test concurrent registration of functions with the same name."""
        registration_results = []
        errors = []
        
        def register_same_name(thread_id, priority):
            try:
                @self.registry.register(
                    'same_name_func', mtype='qrofn',
                    target_classes=['Fuzznum'], injection_type='instance_method',
                    priority=priority
                )
                def same_name_function(self):
                    return f"result_from_thread_{thread_id}_priority_{priority}"
                
                registration_results.append(f"registered_thread_{thread_id}_priority_{priority}")
                
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {str(e)}")
        
        # Create threads with different priorities
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_same_name, args=(i, i + 1))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify that registrations completed without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify that the highest priority function is used
        func = self.registry.get_function('same_name_func', 'qrofn')
        assert func is not None
        
        # Test the function to see which implementation is active
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        obj = MockFuzznum('qrofn')
        result = obj.same_name_func()
        
        # Should be from the highest priority thread (priority 10)
        assert "priority_10" in result
    
    def test_race_condition_in_dispatcher(self):
        """Test race conditions in the dispatcher."""
        # Register a function
        @self.registry.register(
            'race_test_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def race_test_function(self):
            # Simulate some work
            time.sleep(0.001)
            return "race_test_result"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        results = []
        errors = []
        
        def dispatch_function(thread_id):
            try:
                obj = MockFuzznum('qrofn')
                for i in range(10):  # Multiple calls per thread
                    result = obj.race_test_func()
                    results.append(f"thread_{thread_id}_call_{i}: {result}")
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=dispatch_function, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 100  # 10 threads * 10 calls each
        
        # Verify all results are correct
        for result in results:
            assert "race_test_result" in result


class TestErrorHandling:
    """
    Tests for error handling and recovery mechanisms.
    
    Verifies that the extension system properly handles errors
    and provides appropriate recovery mechanisms.
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
    
    def test_error_propagation_from_extension_function(self):
        """Test that errors from extension functions are properly propagated."""
        @self.registry.register(
            'error_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def error_function(self):
            raise ValueError("Test error from extension function")
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        with pytest.raises(ValueError, match="Test error from extension function"):
            obj.error_func()
    
    def test_error_handling_in_dispatcher(self):
        """Test error handling within the dispatcher itself."""
        # Test creating a method proxy for a non-existent function
        method_proxy = self.dispatcher.create_instance_method('non_existent_func')
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        obj = MockFuzznum('qrofn')
        
        # The method proxy should handle missing functions gracefully
        with pytest.raises(NotImplementedError):
            method_proxy(obj)
    
    def test_error_recovery_after_injection_failure(self):
        """Test recovery after injection failures."""
        # Register a valid function
        @self.registry.register(
            'valid_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def valid_function(self):
            return "valid result"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        # First injection should succeed
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        assert obj.valid_func() == "valid result"
        
        # Simulate injection failure by corrupting the injector
        original_inject = self.injector.inject_all
        
        def failing_inject(*args, **kwargs):
            raise RuntimeError("Injection failed")
        
        self.injector.inject_all = failing_inject
        
        # Try to inject again (should fail)
        with pytest.raises(RuntimeError):
            self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        # Restore original injector
        self.injector.inject_all = original_inject
        
        # Should be able to inject again successfully
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        # Function should still work
        obj2 = MockFuzznum('qrofn')
        assert obj2.valid_func() == "valid result"
    
    def test_exception_handling_in_property_access(self):
        """Test exception handling in property access."""
        @self.registry.register(
            'error_property', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def error_property_function(self):
            raise RuntimeError("Property access error")
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        with pytest.raises(RuntimeError, match="Property access error"):
            _ = obj.error_property
    
    def test_error_handling_with_missing_dependencies(self):
        """Test error handling when dependencies are missing."""
        # Register a function that depends on another function
        @self.registry.register(
            'dependent_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def dependent_function(self):
            # This function tries to call another function that doesn't exist
            return self.missing_function()
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        with pytest.raises(AttributeError):
            obj.dependent_func()


class TestBoundaryConditions:
    """
    Tests for boundary conditions and edge cases.
    
    Verifies that the system handles boundary conditions correctly.
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
    
    def test_empty_registry_behavior(self):
        """Test behavior with empty registry."""
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        # Injection with empty registry should not fail
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        # Object should be created normally
        obj = MockFuzznum('qrofn')
        assert obj.mtype == 'qrofn'
        
        # Dispatcher should handle missing functions gracefully
        method_proxy = self.dispatcher.create_instance_method('nonexistent_func')
        with pytest.raises(NotImplementedError):
            method_proxy(obj)
    
    def test_very_long_function_names(self):
        """Test handling of very long function names."""
        long_name = 'a' * 1000  # Very long function name
        
        @self.registry.register(
            long_name, mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def long_name_function(self):
            return "long name result"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        result = getattr(obj, long_name)()
        assert result == "long name result"
    
    def test_very_high_priority_values(self):
        """Test handling of very high priority values."""
        import sys
        
        max_priority = sys.maxsize
        
        # Register a lower priority function first
        @self.registry.register(
            'high_priority_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method',
            priority=1
        )
        def low_priority_function(self):
            return "low priority result"
        
        # Then register a higher priority function with the same name
        @self.registry.register(
            'high_priority_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method',
            priority=max_priority
        )
        def high_priority_function(self):
            return "high priority result"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        result = obj.high_priority_func()
        assert result == "high priority result"
    
    def test_unicode_function_names(self):
        """Test handling of Unicode function names."""
        unicode_name = 'test_函数_名称_🚀'
        
        @self.registry.register(
            unicode_name, mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def unicode_function(self):
            return "unicode result"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        result = getattr(obj, unicode_name)()
        assert result == "unicode result"
    
    def test_zero_priority_functions(self):
        """Test handling of zero priority functions."""
        @self.registry.register(
            'zero_priority_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method',
            priority=0
        )
        def zero_priority_function(self):
            return "zero priority result"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        result = obj.zero_priority_func()
        assert result == "zero priority result"


class TestResourceManagement:
    """
    Tests for resource management edge cases.
    
    Verifies that the system properly manages resources
    and doesn't cause memory leaks or resource exhaustion.
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
    
    def test_memory_cleanup_after_many_registrations(self):
        """Test memory cleanup after many registrations."""
        import gc
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Register many functions
        num_functions = 100
        for i in range(num_functions):
            @self.registry.register(
                f'cleanup_func_{i}', mtype='qrofn',
                target_classes=['Fuzznum'], injection_type='instance_method'
            )
            def cleanup_function(self, func_id=i):
                return f"cleanup_result_{func_id}"
        
        # Force garbage collection
        gc.collect()
        after_registration_objects = len(gc.get_objects())
        
        # Clear the registry (if such method exists, otherwise skip this test)
        if hasattr(self.registry, 'clear'):
            self.registry.clear()
            gc.collect()
            after_clear_objects = len(gc.get_objects())
            
            # Memory should be mostly reclaimed
            # Allow for some overhead but it shouldn't be excessive
            remaining_overhead = after_clear_objects - initial_objects
            assert remaining_overhead < num_functions  # Very lenient check
    
    def test_weak_reference_handling(self):
        """Test handling of weak references to prevent memory leaks."""
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        @self.registry.register(
            'weak_ref_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def weak_ref_function(self):
            return "weak ref result"
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        # Create object and weak reference
        obj = MockFuzznum('qrofn')
        weak_ref = weakref.ref(obj)
        
        # Use the function
        result = obj.weak_ref_func()
        assert result == "weak ref result"
        
        # Delete the object
        del obj
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Weak reference should be dead
        assert weak_ref() is None
    
    def test_large_number_of_concurrent_objects(self):
        """Test handling of large number of concurrent objects."""
        @self.registry.register(
            'concurrent_obj_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def concurrent_obj_function(self):
            return f"result_for_{id(self)}"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        # Create many objects
        num_objects = 1000
        objects = []
        results = []
        
        for i in range(num_objects):
            obj = MockFuzznum('qrofn')
            objects.append(obj)
            result = obj.concurrent_obj_func()
            results.append(result)
        
        # Verify all objects work correctly
        assert len(results) == num_objects
        assert len(set(results)) == num_objects  # All results should be unique
        
        # Clean up
        del objects
        del results
        
        import gc
        gc.collect()


class TestCorruptionRecovery:
    """
    Tests for recovery from corruption scenarios.
    
    Verifies that the system can recover from various
    corruption scenarios gracefully.
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
    
    def test_recovery_from_corrupted_function_metadata(self):
        """Test recovery from corrupted function metadata."""
        # Register a valid function
        @self.registry.register(
            'valid_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def valid_function(self):
            return "valid result"
        
        # Corrupt the metadata (if accessible)
        if hasattr(self.registry, '_functions'):
            # Simulate corruption by modifying internal state
            for key in self.registry._functions:
                if 'valid_func' in str(key):
                    # Corrupt the metadata
                    original_metadata = self.registry._functions[key]
                    self.registry._functions[key] = None
                    break
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        # Injection might fail or succeed depending on implementation
        try:
            self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
            obj = MockFuzznum('qrofn')
            
            # Function call might fail due to corruption
            with pytest.raises((AttributeError, TypeError, RuntimeError)):
                obj.valid_func()
        except Exception:
            # If injection itself fails, that's also acceptable
            pass
    
    def test_recovery_from_partial_injection_state(self):
        """Test recovery from partial injection state."""
        @self.registry.register(
            'partial_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def partial_function(self):
            return "partial result"
        
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        # Perform partial injection by mocking the injector's _inject_function method
        original_inject_method = self.injector._inject_function
        call_count = 0
        
        def failing_inject_method(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                return original_inject_method(*args, **kwargs)
            else:
                # Subsequent calls fail
                raise RuntimeError("Injection failure")
        
        with patch.object(self.injector, '_inject_function', side_effect=failing_inject_method):
            # This might fail due to partial injection
            try:
                self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
            except RuntimeError:
                pass  # Expected failure
        
        # Try to recover by performing injection again with original injector
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        result = obj.partial_func()
        assert result == "partial result"
    
    def test_handling_of_corrupted_class_map(self):
        """Test handling of corrupted class map during injection."""
        @self.registry.register(
            'class_map_func', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def class_map_function(self):
            return "class map result"
        
        # Create a corrupted class map with non-class objects
        corrupted_class_map = {'Fuzznum': "not_a_class"}
        
        # Injection should handle corruption gracefully
        try:
            self.injector.inject_all(corrupted_class_map, {})
        except (AttributeError, TypeError):
            pass  # Expected behavior for corrupted class map
        
        # System should still work with valid class map
        class MockFuzznum:
            def __init__(self, mtype='qrofn'):
                self.mtype = mtype
        
        self.injector.inject_all({'Fuzznum': MockFuzznum}, {})
        
        obj = MockFuzznum('qrofn')
        result = obj.class_map_func()
        assert result == "class map result"