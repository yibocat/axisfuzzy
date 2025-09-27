#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Tests for ExtensionRegistry core functionality.

This module tests the core functionality of the ExtensionRegistry class,
including function registration, storage, retrieval, and thread safety.
"""

import pytest
import threading
import time
from unittest.mock import Mock

from axisfuzzy.extension.registry import ExtensionRegistry, FunctionMetadata, get_registry_extension


class TestExtensionRegistry:
    """Test suite for ExtensionRegistry core functionality."""

    def setup_method(self):
        """Setup for each test method - create a fresh registry instance."""
        self.registry = ExtensionRegistry()

    def test_registry_initialization(self):
        """Test that registry initializes with empty state."""
        assert self.registry._functions == {}
        assert self.registry._defaults == {}
        assert self.registry._lock is not None

    def test_singleton_registry_access(self):
        """Test that get_registry_extension returns the same instance."""
        reg1 = get_registry_extension()
        reg2 = get_registry_extension()
        assert reg1 is reg2

    def test_register_specialized_function(self):
        """Test registering a specialized function for a specific mtype."""
        def test_func():
            return "specialized"

        # Register the function
        decorator = self.registry.register(
            name='test_method',
            mtype='qrofn',
            target_classes=['Fuzznum'],
            injection_type='instance_method'
        )
        registered_func = decorator(test_func)

        # Verify registration
        assert registered_func is test_func
        assert 'test_method' in self.registry._functions
        assert 'qrofn' in self.registry._functions['test_method']
        
        stored_func, metadata = self.registry._functions['test_method']['qrofn']
        assert stored_func is test_func
        assert metadata.name == 'test_method'
        assert metadata.mtype == 'qrofn'
        assert metadata.target_classes == ['Fuzznum']
        assert metadata.injection_type == 'instance_method'

    def test_register_default_function(self):
        """Test registering a default (fallback) function."""
        def default_func():
            return "default"

        # Register as default
        decorator = self.registry.register(
            name='test_method',
            is_default=True,
            target_classes=['Fuzznum', 'Fuzzarray'],
            injection_type='both'
        )
        registered_func = decorator(default_func)

        # Verify registration
        assert registered_func is default_func
        assert 'test_method' in self.registry._defaults
        
        stored_func, metadata = self.registry._defaults['test_method']
        assert stored_func is default_func
        assert metadata.name == 'test_method'
        assert metadata.mtype is None
        assert metadata.is_default is True
        assert metadata.target_classes == ['Fuzznum', 'Fuzzarray']
        assert metadata.injection_type == 'both'

    def test_register_with_priority(self):
        """Test that priority prevents overwriting higher priority registrations."""
        def high_priority_func():
            return "high"

        def low_priority_func():
            return "low"

        # Register high priority function first
        decorator1 = self.registry.register(
            name='test_method',
            mtype='qrofn',
            priority=10,
            target_classes=['Fuzznum']
        )
        decorator1(high_priority_func)

        # Try to register lower priority function - should raise ValueError
        with pytest.raises(ValueError, match="already exists with higher or equal priority"):
            decorator2 = self.registry.register(
                name='test_method',
                mtype='qrofn',
                priority=5,
                target_classes=['Fuzznum']
            )
            decorator2(low_priority_func)

        # Verify high priority function is still registered
        stored_func, metadata = self.registry._functions['test_method']['qrofn']
        assert stored_func is high_priority_func
        assert metadata.priority == 10

    def test_register_with_equal_priority_fails(self):
        """Test that equal priority prevents overwriting."""
        def first_func():
            return "first"

        def second_func():
            return "second"

        # Register first function
        decorator1 = self.registry.register(
            name='test_method',
            mtype='qrofn',
            priority=5,
            target_classes=['Fuzznum']
        )
        decorator1(first_func)

        # Try to register second function with same priority - should fail
        with pytest.raises(ValueError, match="already exists with higher or equal priority"):
            decorator2 = self.registry.register(
                name='test_method',
                mtype='qrofn',
                priority=5,
                target_classes=['Fuzznum']
            )
            decorator2(second_func)

        # Verify first function is still registered
        stored_func, metadata = self.registry._functions['test_method']['qrofn']
        assert stored_func is first_func

    def test_get_function_specialized(self):
        """Test retrieving a specialized function."""
        def specialized_func():
            return "specialized"

        # Register specialized function
        decorator = self.registry.register(
            name='test_method',
            mtype='qrofn',
            target_classes=['Fuzznum']
        )
        decorator(specialized_func)

        # Retrieve function
        retrieved_func = self.registry.get_function('test_method', 'qrofn')
        assert retrieved_func is specialized_func

    def test_get_function_fallback_to_default(self):
        """Test that get_function falls back to default when specialized not found."""
        def default_func():
            return "default"

        # Register only default function
        decorator = self.registry.register(
            name='test_method',
            is_default=True,
            target_classes=['Fuzznum']
        )
        decorator(default_func)

        # Try to get specialized function - should return default
        retrieved_func = self.registry.get_function('test_method', 'qrofn')
        assert retrieved_func is default_func

    def test_get_function_not_found(self):
        """Test that get_function returns None when function not found."""
        retrieved_func = self.registry.get_function('nonexistent', 'qrofn')
        assert retrieved_func is None

    def test_get_function_prefers_specialized_over_default(self):
        """Test that specialized function is preferred over default."""
        def specialized_func():
            return "specialized"

        def default_func():
            return "default"

        # Register both specialized and default
        decorator1 = self.registry.register(
            name='test_method',
            mtype='qrofn',
            target_classes=['Fuzznum']
        )
        decorator1(specialized_func)

        decorator2 = self.registry.register(
            name='test_method',
            is_default=True,
            target_classes=['Fuzznum']
        )
        decorator2(default_func)

        # Should return specialized function
        retrieved_func = self.registry.get_function('test_method', 'qrofn')
        assert retrieved_func is specialized_func

    def test_get_top_level_function_names(self):
        """Test getting names of functions that should be injected as top-level."""
        def func1():
            pass

        def func2():
            pass

        def func3():
            pass

        # Register functions with different injection types
        self.registry.register(
            name='method1',
            injection_type='top_level_function',
            target_classes=['Fuzznum']
        )(func1)

        self.registry.register(
            name='method2',
            injection_type='both',
            target_classes=['Fuzznum']
        )(func2)

        self.registry.register(
            name='method3',
            injection_type='instance_method',
            target_classes=['Fuzznum']
        )(func3)

        # Get top-level function names
        top_level_names = self.registry.get_top_level_function_names()
        
        # Should include method1 and method2, but not method3
        assert 'method1' in top_level_names
        assert 'method2' in top_level_names
        assert 'method3' not in top_level_names
        assert len(top_level_names) == 2

    def test_get_metadata_specialized(self):
        """Test retrieving metadata for specialized function."""
        def test_func():
            pass

        # Register function
        self.registry.register(
            name='test_method',
            mtype='qrofn',
            target_classes=['Fuzznum'],
            injection_type='instance_property',
            priority=5,
            description='Test function'
        )(test_func)

        # Get metadata
        metadata = self.registry.get_metadata('test_method', 'qrofn')
        assert metadata is not None
        assert metadata.name == 'test_method'
        assert metadata.mtype == 'qrofn'
        assert metadata.target_classes == ['Fuzznum']
        assert metadata.injection_type == 'instance_property'
        assert metadata.priority == 5
        assert metadata.description == 'Test function'

    def test_get_metadata_default(self):
        """Test retrieving metadata for default function."""
        def test_func():
            pass

        # Register default function
        self.registry.register(
            name='test_method',
            is_default=True,
            target_classes=['Fuzzarray'],
            injection_type='both'
        )(test_func)

        # Get metadata without specifying mtype
        metadata = self.registry.get_metadata('test_method')
        assert metadata is not None
        assert metadata.name == 'test_method'
        assert metadata.mtype is None
        assert metadata.is_default is True
        assert metadata.target_classes == ['Fuzzarray']
        assert metadata.injection_type == 'both'

    def test_get_metadata_not_found(self):
        """Test that get_metadata returns None when function not found."""
        metadata = self.registry.get_metadata('nonexistent', 'qrofn')
        assert metadata is None

    def test_list_functions_empty(self):
        """Test list_functions returns empty dict when no functions registered."""
        result = self.registry.list_functions()
        assert result == {}

    def test_list_functions_with_registrations(self):
        """Test list_functions returns correct structure with registrations."""
        def specialized_func():
            pass

        def default_func():
            pass

        # Register specialized function
        self.registry.register(
            name='method1',
            mtype='qrofn',
            target_classes=['Fuzznum'],
            injection_type='instance_method',
            priority=5
        )(specialized_func)

        # Register default function
        self.registry.register(
            name='method1',
            is_default=True,
            target_classes=['Fuzzarray'],
            injection_type='both',
            priority=3
        )(default_func)

        # Register another specialized function
        self.registry.register(
            name='method2',
            mtype='ivfn',
            target_classes=['Fuzznum'],
            injection_type='top_level_function',
            priority=1
        )(specialized_func)

        result = self.registry.list_functions()

        # Verify structure
        assert 'method1' in result
        assert 'method2' in result

        # Check method1 structure
        method1_info = result['method1']
        assert 'implementations' in method1_info
        assert 'default' in method1_info
        
        # Check specialized implementation
        assert 'qrofn' in method1_info['implementations']
        qrofn_impl = method1_info['implementations']['qrofn']
        assert qrofn_impl['priority'] == 5
        assert qrofn_impl['target_classes'] == ['Fuzznum']
        assert qrofn_impl['injection_type'] == 'instance_method'

        # Check default implementation
        default_impl = method1_info['default']
        assert default_impl['priority'] == 3
        assert default_impl['target_classes'] == ['Fuzzarray']
        assert default_impl['injection_type'] == 'both'

        # Check method2 structure
        method2_info = result['method2']
        assert 'ivfn' in method2_info['implementations']
        assert method2_info['default'] is None

    def test_thread_safety_concurrent_registration(self):
        """Test that concurrent registration is thread-safe."""
        results = []
        errors = []

        def register_function(thread_id):
            try:
                def test_func():
                    return f"thread_{thread_id}"

                decorator = self.registry.register(
                    name=f'method_{thread_id}',
                    mtype='qrofn',
                    target_classes=['Fuzznum']
                )
                decorator(test_func)
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_function, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert set(results) == set(range(10))

        # Verify all functions were registered
        for i in range(10):
            func = self.registry.get_function(f'method_{i}', 'qrofn')
            assert func is not None

    def test_thread_safety_concurrent_access(self):
        """Test that concurrent access to registry is thread-safe."""
        # Pre-register some functions
        for i in range(5):
            def test_func():
                return f"func_{i}"

            self.registry.register(
                name=f'method_{i}',
                mtype='qrofn',
                target_classes=['Fuzznum']
            )(test_func)

        results = []
        errors = []

        def access_function(thread_id):
            try:
                # Perform various read operations
                for i in range(5):
                    func = self.registry.get_function(f'method_{i}', 'qrofn')
                    metadata = self.registry.get_metadata(f'method_{i}', 'qrofn')
                    top_level_names = self.registry.get_top_level_function_names()
                    function_list = self.registry.list_functions()
                    
                    assert func is not None
                    assert metadata is not None
                    
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=access_function, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

    def test_function_metadata_dataclass(self):
        """Test FunctionMetadata dataclass functionality."""
        metadata = FunctionMetadata(
            name='test_method',
            mtype='qrofn',
            target_classes=['Fuzznum'],
            injection_type='instance_method',
            is_default=False,
            priority=5,
            description='Test description'
        )

        assert metadata.name == 'test_method'
        assert metadata.mtype == 'qrofn'
        assert metadata.target_classes == ['Fuzznum']
        assert metadata.injection_type == 'instance_method'
        assert metadata.is_default is False
        assert metadata.priority == 5
        assert metadata.description == 'Test description'

    def test_function_metadata_defaults(self):
        """Test FunctionMetadata default values."""
        metadata = FunctionMetadata(
            name='test_method',
            mtype='qrofn',
            target_classes=['Fuzznum'],
            injection_type='instance_method'
        )

        assert metadata.is_default is False
        assert metadata.priority == 0
        assert metadata.description == ""

    def test_timestamp_generation(self):
        """Test that timestamp generation works correctly."""
        timestamp = ExtensionRegistry._get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        
        # Test that timestamps are different when called at different times
        time.sleep(0.001)  # Small delay
        timestamp2 = ExtensionRegistry._get_timestamp()
        assert timestamp != timestamp2