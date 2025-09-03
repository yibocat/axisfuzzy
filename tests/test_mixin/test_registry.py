#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午4:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Test suite for axisfuzzy.mixin.registry module.

This module provides comprehensive tests for the MixinFunctionRegistry class
and related functionality, ensuring proper registration, injection, and
management of mixin functions.

Test Coverage
-------------
- MixinFunctionRegistry class initialization and basic operations
- Function registration with different injection types
- Validation of registration parameters
- Build and injection process for instance methods and top-level functions
- Error handling for invalid registrations and duplicate names
- Registry singleton behavior
- Integration with class and module namespace injection

Test Classes
------------
TestMixinFunctionRegistry : Tests for the core registry class
TestRegistrationDecorator : Tests for the register_mixin decorator
TestInjectionProcess : Tests for the build_and_inject functionality
TestRegistryIntegration : Integration tests with mock classes
TestErrorHandling : Tests for error conditions and edge cases
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from axisfuzzy.mixin.registry import (
    MixinFunctionRegistry,
    get_registry_mixin,
    register_mixin
)


class TestMixinFunctionRegistry:
    """
    Test suite for MixinFunctionRegistry class core functionality.
    
    Tests the basic operations of the registry including initialization,
    function registration, metadata storage, and retrieval operations.
    """
    
    def test_registry_initialization(self):
        """
        Test that MixinFunctionRegistry initializes with empty storage.
        
        Verifies that a new registry instance starts with empty function
        and metadata dictionaries.
        """
        registry = MixinFunctionRegistry()
        
        assert registry._functions == {}
        assert registry._metadata == {}
    
    def test_register_instance_function(self):
        """
        Test registration of instance-only functions.
        
        Verifies that functions can be registered for instance method
        injection with proper metadata storage.
        """
        registry = MixinFunctionRegistry()
        
        @registry.register(
            name='test_method',
            target_classes=['TestClass'],
            injection_type='instance_function'
        )
        def test_func(self):
            return "instance method"
        
        # Verify function is stored
        assert 'test_method' in registry._functions
        assert registry._functions['test_method'] == test_func
        
        # Verify metadata is correct
        metadata = registry._metadata['test_method']
        assert metadata['target_classes'] == ['TestClass']
        assert metadata['injection_type'] == 'instance_function'
    
    def test_register_top_level_function(self):
        """
        Test registration of top-level-only functions.
        
        Verifies that functions can be registered for module namespace
        injection without requiring target classes.
        """
        registry = MixinFunctionRegistry()
        
        @registry.register(
            name='top_level_func',
            injection_type='top_level_function'
        )
        def test_func():
            return "top level function"
        
        # Verify function is stored
        assert 'top_level_func' in registry._functions
        assert registry._functions['top_level_func'] == test_func
        
        # Verify metadata is correct
        metadata = registry._metadata['top_level_func']
        assert metadata['target_classes'] == []  # Should be empty list
        assert metadata['injection_type'] == 'top_level_function'
    
    def test_register_both_injection_types(self):
        """
        Test registration of functions for both instance and top-level injection.
        
        Verifies that functions can be registered to be exposed as both
        instance methods and top-level functions.
        """
        registry = MixinFunctionRegistry()
        
        @registry.register(
            name='dual_func',
            target_classes=['TestClass1', 'TestClass2'],
            injection_type='both'
        )
        def test_func(self):
            return "dual function"
        
        # Verify function is stored
        assert 'dual_func' in registry._functions
        assert registry._functions['dual_func'] == test_func
        
        # Verify metadata is correct
        metadata = registry._metadata['dual_func']
        assert metadata['target_classes'] == ['TestClass1', 'TestClass2']
        assert metadata['injection_type'] == 'both'
    
    def test_register_duplicate_name_error(self):
        """
        Test that registering duplicate function names raises ValueError.
        
        Ensures that the registry prevents accidental overwrites by
        raising an error when the same name is registered twice.
        """
        registry = MixinFunctionRegistry()
        
        # Register first function
        @registry.register(
            name='duplicate_name',
            target_classes=['TestClass'],
            injection_type='instance_function'
        )
        def first_func(self):
            return "first"
        
        # Attempt to register second function with same name
        with pytest.raises(ValueError, match="Function 'duplicate_name' is already registered"):
            @registry.register(
                name='duplicate_name',
                target_classes=['TestClass'],
                injection_type='instance_function'
            )
            def second_func(self):
                return "second"
    
    def test_register_invalid_injection_type(self):
        """
        Test that invalid injection types raise ValueError.
        
        Verifies that only the allowed injection type literals are accepted.
        """
        registry = MixinFunctionRegistry()
        
        with pytest.raises(ValueError, match="Invalid injection_type: invalid_type"):
            @registry.register(
                name='test_func',
                injection_type='invalid_type'  # type: ignore
            )
            def test_func():
                pass
    
    def test_register_missing_target_classes_for_instance(self):
        """
        Test that instance injection requires target_classes.
        
        Verifies that ValueError is raised when target_classes is None
        for instance_function or both injection types.
        """
        registry = MixinFunctionRegistry()
        
        # Test instance_function without target_classes
        with pytest.raises(ValueError, match="target_classes must be provided"):
            @registry.register(
                name='test_func',
                injection_type='instance_function'
            )
            def test_func():
                pass
        
        # Test 'both' without target_classes
        with pytest.raises(ValueError, match="target_classes must be provided"):
            @registry.register(
                name='test_func2',
                injection_type='both'
            )
            def test_func2():
                pass
    
    def test_get_top_level_function_names(self):
        """
        Test retrieval of top-level function names.
        
        Verifies that the registry correctly identifies and returns
        names of functions registered for top-level injection.
        """
        registry = MixinFunctionRegistry()
        
        # Register functions with different injection types
        @registry.register(
            name='instance_only',
            target_classes=['TestClass'],
            injection_type='instance_function'
        )
        def func1(self):
            pass
        
        @registry.register(
            name='top_level_only',
            injection_type='top_level_function'
        )
        def func2():
            pass
        
        @registry.register(
            name='both_types',
            target_classes=['TestClass'],
            injection_type='both'
        )
        def func3(self):
            pass
        
        # Get top-level function names
        top_level_names = registry.get_top_level_function_names()
        
        # Should include 'top_level_only' and 'both_types', but not 'instance_only'
        assert 'top_level_only' in top_level_names
        assert 'both_types' in top_level_names
        assert 'instance_only' not in top_level_names
        assert len(top_level_names) == 2


class TestBuildAndInject:
    """
    Test suite for the build_and_inject functionality.
    
    Tests the injection process that attaches registered functions
    to target classes and module namespaces.
    """
    
    def test_inject_instance_methods(self):
        """
        Test injection of instance methods into target classes.
        
        Verifies that functions registered for instance injection
        are properly attached to the specified classes.
        """
        registry = MixinFunctionRegistry()
        
        # Create mock classes
        class MockClass1:
            pass
        
        class MockClass2:
            pass
        
        # Register function for instance injection
        @registry.register(
            name='test_method',
            target_classes=['MockClass1', 'MockClass2'],
            injection_type='instance_function'
        )
        def test_func(self):
            return f"method called on {type(self).__name__}"
        
        # Prepare class map
        class_map = {
            'MockClass1': MockClass1,
            'MockClass2': MockClass2
        }
        
        # Inject functions
        registry.build_and_inject(class_map, {})
        
        # Verify methods are attached to classes
        assert hasattr(MockClass1, 'test_method')
        assert hasattr(MockClass2, 'test_method')
        
        # Test method functionality
        obj1 = MockClass1()
        obj2 = MockClass2()
        
        assert obj1.test_method() == "method called on MockClass1"
        assert obj2.test_method() == "method called on MockClass2"
    
    def test_inject_top_level_functions(self):
        """
        Test injection of top-level functions into module namespace.
        
        Verifies that functions registered for top-level injection
        are properly added to the module namespace.
        """
        registry = MixinFunctionRegistry()
        
        # Register function for top-level injection
        @registry.register(
            name='top_level_func',
            injection_type='top_level_function'
        )
        def test_func(arg):
            return f"top level called with {arg}"
        
        # Prepare module namespace
        module_namespace = {}
        
        # Inject functions
        registry.build_and_inject({}, module_namespace)
        
        # Verify function is in namespace
        assert 'top_level_func' in module_namespace
        assert callable(module_namespace['top_level_func'])
        
        # Test function functionality
        result = module_namespace['top_level_func']('test_arg')
        assert result == "top level called with test_arg"
    
    def test_inject_both_types(self):
        """
        Test injection of functions as both instance methods and top-level functions.
        
        Verifies that functions registered with 'both' injection type
        are properly exposed in both contexts with correct delegation.
        """
        registry = MixinFunctionRegistry()
        
        # Create mock class
        class MockClass:
            def __init__(self, value):
                self.value = value
        
        # Register function for both injection types
        @registry.register(
            name='dual_func',
            target_classes=['MockClass'],
            injection_type='both'
        )
        def test_func(self):
            return f"dual function on {self.value}"
        
        # Prepare injection targets
        class_map = {'MockClass': MockClass}
        module_namespace = {}
        
        # Inject functions
        registry.build_and_inject(class_map, module_namespace)
        
        # Verify instance method is attached
        assert hasattr(MockClass, 'dual_func')
        
        # Verify top-level function is in namespace
        assert 'dual_func' in module_namespace
        assert callable(module_namespace['dual_func'])
        
        # Test instance method functionality
        obj = MockClass('test_value')
        assert obj.dual_func() == "dual function on test_value"
        
        # Test top-level function functionality (should delegate to instance method)
        result = module_namespace['dual_func'](obj)
        assert result == "dual function on test_value"
    
    def test_inject_missing_class_ignored(self):
        """
        Test that missing classes in class_map are silently ignored.
        
        Verifies that injection continues normally when a registered
        function targets a class that's not in the class_map.
        """
        registry = MixinFunctionRegistry()
        
        # Register function targeting non-existent class
        @registry.register(
            name='test_method',
            target_classes=['NonExistentClass', 'AnotherMissingClass'],
            injection_type='instance_function'
        )
        def test_func(self):
            return "test"
        
        # Inject with empty class map (no errors should occur)
        registry.build_and_inject({}, {})
        
        # Verify function is still registered
        assert 'test_method' in registry._functions
    
    def test_top_level_wrapper_error_handling(self):
        """
        Test error handling in top-level function wrappers.
        
        Verifies that top-level functions created for 'both' injection
        type properly handle objects that don't support the method.
        """
        registry = MixinFunctionRegistry()
        
        # Create mock class
        class MockClass:
            pass
        
        # Register function for both injection types
        @registry.register(
            name='test_method',
            target_classes=['MockClass'],
            injection_type='both'
        )
        def test_func(self):
            return "test"
        
        # Inject functions
        class_map = {'MockClass': MockClass}
        module_namespace = {}
        registry.build_and_inject(class_map, module_namespace)
        
        # Test with object that doesn't have the method
        class UnsupportedClass:
            pass
        
        unsupported_obj = UnsupportedClass()
        
        # Should raise TypeError for unsupported object
        with pytest.raises(TypeError, match="'test_method' is not supported for type 'UnsupportedClass'"):
            module_namespace['test_method'](unsupported_obj)


class TestRegistryIntegration:
    """
    Integration tests for registry functionality.
    
    Tests the complete workflow from registration to injection
    with realistic scenarios.
    """
    
    def test_complete_registration_and_injection_workflow(self):
        """
        Test complete workflow from registration to injection.
        
        Verifies that the entire process works correctly with
        multiple functions and injection types.
        """
        registry = MixinFunctionRegistry()
        
        # Create mock classes
        class Fuzznum:
            def __init__(self, value):
                self.value = value
        
        class Fuzzarray:
            def __init__(self, values):
                self.values = values
        
        # Register multiple functions with different injection types
        @registry.register(
            name='get_value',
            target_classes=['Fuzznum'],
            injection_type='instance_function'
        )
        def get_value_impl(self):
            return self.value
        
        @registry.register(
            name='create_empty',
            injection_type='top_level_function'
        )
        def create_empty_impl():
            return Fuzzarray([])
        
        @registry.register(
            name='size',
            target_classes=['Fuzzarray'],
            injection_type='both'
        )
        def size_impl(self):
            return len(self.values)
        
        # Prepare injection targets
        class_map = {
            'Fuzznum': Fuzznum,
            'Fuzzarray': Fuzzarray
        }
        module_namespace = {}
        
        # Perform injection
        registry.build_and_inject(class_map, module_namespace)
        
        # Test instance-only method
        num = Fuzznum(42)
        assert num.get_value() == 42
        
        # Test top-level-only function
        empty_array = module_namespace['create_empty']()
        assert isinstance(empty_array, Fuzzarray)
        assert empty_array.values == []
        
        # Test both injection types
        array = Fuzzarray([1, 2, 3, 4, 5])
        
        # Instance method
        assert array.size() == 5
        
        # Top-level function
        assert module_namespace['size'](array) == 5
        
        # Verify top-level function names
        top_level_names = registry.get_top_level_function_names()
        assert 'create_empty' in top_level_names
        assert 'size' in top_level_names
        assert 'get_value' not in top_level_names
        assert len(top_level_names) == 2


class TestRegistryDecorator:
    """
    Test suite for the register_mixin convenience decorator.
    
    Tests the standalone decorator function that provides
    access to the global registry.
    """
    
    def test_register_mixin_decorator(self):
        """
        Test the register_mixin convenience decorator.
        
        Verifies that the decorator properly delegates to the
        global registry instance.
        """
        # Get initial registry state
        global_registry = get_registry_mixin()
        initial_function_count = len(global_registry._functions)
        
        # Register function using convenience decorator
        @register_mixin(
            name='test_convenience_func',
            target_classes=['TestClass'],
            injection_type='instance_function'
        )
        def test_func(self):
            return "convenience decorator test"
        
        # Verify function was registered in global registry
        assert 'test_convenience_func' in global_registry._functions
        assert len(global_registry._functions) == initial_function_count + 1
        
        # Verify metadata is correct
        metadata = global_registry._metadata['test_convenience_func']
        assert metadata['target_classes'] == ['TestClass']
        assert metadata['injection_type'] == 'instance_function'
    
    def test_register_mixin_parameter_validation(self):
        """
        Test parameter validation in register_mixin decorator.
        
        Verifies that the decorator properly validates parameters
        and raises appropriate errors.
        """
        # Test invalid injection type
        with pytest.raises(ValueError, match="Invalid injection_type"):
            @register_mixin(
                name='test_func',
                injection_type='invalid_type'  # type: ignore
            )
            def test_func():
                pass
        
        # Test missing target_classes for instance injection
        with pytest.raises(ValueError, match="target_classes must be provided"):
            @register_mixin(
                name='test_func2',
                injection_type='instance_function'
            )
            def test_func2():
                pass


class TestRegistrySingleton:
    """
    Test suite for registry singleton behavior.
    
    Verifies that the global registry maintains singleton
    behavior across multiple access points.
    """
    
    def test_get_registry_mixin_singleton(self):
        """
        Test that get_registry_mixin returns the same instance.
        
        Verifies singleton behavior of the global registry.
        """
        registry1 = get_registry_mixin()
        registry2 = get_registry_mixin()
        
        # Should be the same instance
        assert registry1 is registry2
        
        # Modifications should be visible across references
        @registry1.register(
            name='singleton_test_func',
            injection_type='top_level_function'
        )
        def test_func():
            return "singleton test"
        
        # Should be visible in registry2
        assert 'singleton_test_func' in registry2._functions
        assert registry2._functions['singleton_test_func'] == test_func


class TestErrorHandling:
    """
    Test suite for error handling and edge cases.
    
    Tests various error conditions and boundary cases
    to ensure robust behavior.
    """
    
    def test_empty_registry_operations(self):
        """
        Test operations on empty registry.
        
        Verifies that operations work correctly when no
        functions have been registered.
        """
        registry = MixinFunctionRegistry()
        
        # Empty registry should return empty list
        assert registry.get_top_level_function_names() == []
        
        # Injection should work without errors
        registry.build_and_inject({}, {})
    
    def test_injection_with_empty_targets(self):
        """
        Test injection with empty class_map and module_namespace.
        
        Verifies that injection handles empty targets gracefully.
        """
        registry = MixinFunctionRegistry()
        
        # Register some functions
        @registry.register(
            name='test_func1',
            target_classes=['TestClass'],
            injection_type='instance_function'
        )
        def func1(self):
            pass
        
        @registry.register(
            name='test_func2',
            injection_type='top_level_function'
        )
        def func2():
            pass
        
        # Injection with empty targets should not raise errors
        registry.build_and_inject({}, {})
        
        # Functions should still be registered
        assert 'test_func1' in registry._functions
        assert 'test_func2' in registry._functions
    
    def test_multiple_injections_idempotent(self):
        """
        Test that multiple injections are idempotent.
        
        Verifies that calling build_and_inject multiple times
        doesn't cause issues or duplicate injections.
        """
        registry = MixinFunctionRegistry()
        
        # Create mock class
        class MockClass:
            pass
        
        # Register function
        @registry.register(
            name='test_method',
            target_classes=['MockClass'],
            injection_type='instance_function'
        )
        def test_func(self):
            return "test"
        
        class_map = {'MockClass': MockClass}
        
        # Inject multiple times
        registry.build_and_inject(class_map, {})
        registry.build_and_inject(class_map, {})
        registry.build_and_inject(class_map, {})
        
        # Should still work correctly
        assert hasattr(MockClass, 'test_method')
        obj = MockClass()
        assert obj.test_method() == "test"


if __name__ == '__main__':
    pytest.main()