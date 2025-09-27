"""
Tests for the external_extension decorator and automatic application functionality.

This module tests the @external_extension decorator which provides a convenient
way to register and automatically apply extensions, especially useful for
external libraries and user-defined extensions.
"""

import pytest
import warnings
from unittest.mock import patch, MagicMock

from axisfuzzy.extension import external_extension, apply_extensions
from axisfuzzy.extension.registry import get_registry_extension


class TestExternalExtension:
    """Test the @external_extension decorator functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_external_extension_basic_registration(self):
        """Test basic external extension registration."""
        @external_extension('test_method', mtype='qrofn', auto_apply=False)
        def test_func(self):
            return "test_result"

        # Verify function is registered
        assert 'test_method' in self.registry._functions
        assert 'qrofn' in self.registry._functions['test_method']
        
        # Verify metadata
        metadata = self.registry.get_metadata('test_method', 'qrofn')
        assert metadata.name == 'test_method'
        assert metadata.mtype == 'qrofn'
        assert metadata.target_classes == ['Fuzznum', 'Fuzzarray']  # Default
        assert metadata.injection_type == 'both'  # Default
        assert not metadata.is_default

    def test_external_extension_with_all_parameters(self):
        """Test external extension with all parameters specified."""
        @external_extension(
            name='custom_method',
            mtype='qrohfn',
            target_classes=['Fuzznum'],
            injection_type='instance_method',
            is_default=False,
            priority=5,
            auto_apply=False,
            description='test description'
        )
        def custom_func(self):
            return "custom_result"

        # Verify registration
        metadata = self.registry.get_metadata('custom_method', 'qrohfn')
        assert metadata.name == 'custom_method'
        assert metadata.mtype == 'qrohfn'
        assert metadata.target_classes == ['Fuzznum']
        assert metadata.injection_type == 'instance_method'
        assert not metadata.is_default
        assert metadata.priority == 5
        assert metadata.description == 'test description'

    def test_external_extension_default_registration(self):
        """Test external extension for default (no mtype) registration."""
        @external_extension('default_method', is_default=True, auto_apply=False)
        def default_func(self):
            return "default_result"

        # Verify default registration
        assert 'default_method' in self.registry._defaults
        
        metadata = self.registry.get_metadata('default_method', None)
        assert metadata.name == 'default_method'
        assert metadata.mtype is None
        assert metadata.is_default

    def test_external_extension_preserves_function_attributes(self):
        """Test that external extension preserves original function attributes."""
        def original_func(self, x, y=10):
            """Original function docstring."""
            return x + y

        decorated_func = external_extension(
            'preserve_test', 
            mtype='qrofn', 
            auto_apply=False
        )(original_func)

        # Verify function attributes are preserved
        assert decorated_func.__name__ == 'original_func'
        assert decorated_func.__doc__ == "Original function docstring."
        assert decorated_func.__module__ == original_func.__module__

    @patch('axisfuzzy.extension.apply_extensions')
    def test_external_extension_auto_apply_success(self, mock_apply):
        """Test automatic application when auto_apply=True."""
        mock_apply.return_value = True

        @external_extension('auto_method', mtype='qrofn', auto_apply=True)
        def auto_func(self):
            return "auto_result"

        # Verify apply_extensions was called with force_reapply=True
        mock_apply.assert_called_once_with(force_reapply=True)

    @patch('axisfuzzy.extension.apply_extensions')
    def test_external_extension_auto_apply_failure_warning(self, mock_apply):
        """Test warning when automatic application fails."""
        mock_apply.return_value = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            @external_extension('fail_method', mtype='qrofn', auto_apply=True)
            def fail_func(self):
                return "fail_result"

            # Verify warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "Failed to automatically apply extension 'fail_method'" in str(w[0].message)

    @patch('axisfuzzy.extension.apply_extensions')
    def test_external_extension_auto_apply_disabled(self, mock_apply):
        """Test that auto_apply=False doesn't call apply_extensions."""
        @external_extension('no_auto_method', mtype='qrofn', auto_apply=False)
        def no_auto_func(self):
            return "no_auto_result"

        # Verify apply_extensions was not called
        mock_apply.assert_not_called()

    def test_external_extension_multiple_registrations(self):
        """Test multiple external extensions with different configurations."""
        @external_extension('multi_method', mtype='qrofn', auto_apply=False)
        def qrofn_func(self):
            return "qrofn_result"

        @external_extension('multi_method', mtype='qrohfn', auto_apply=False)
        def qrohfn_func(self):
            return "qrohfn_result"

        # Verify both registrations exist
        assert 'qrofn' in self.registry._functions['multi_method']
        assert 'qrohfn' in self.registry._functions['multi_method']

        # Verify correct functions are registered
        qrofn_func_stored = self.registry.get_function('multi_method', 'qrofn')
        qrohfn_func_stored = self.registry.get_function('multi_method', 'qrohfn')
        
        assert qrofn_func_stored == qrofn_func
        assert qrohfn_func_stored == qrohfn_func

    def test_external_extension_priority_handling(self):
        """Test priority handling in external extensions."""
        # Register higher priority first (higher number = higher priority)
        @external_extension('priority_method', mtype='qrofn', priority=10, auto_apply=False)
        def high_priority_func(self):
            return "high_priority"

        # Attempt to register lower priority - should raise ValueError
        with pytest.raises(ValueError, match="already exists with higher or equal priority"):
            @external_extension('priority_method', mtype='qrofn', priority=5, auto_apply=False)
            def low_priority_func(self):
                return "low_priority"

    def test_external_extension_target_classes_string(self):
        """Test external extension with target_classes as string."""
        @external_extension(
            'string_target_method', 
            mtype='qrofn', 
            target_classes='Fuzznum',
            auto_apply=False
        )
        def string_target_func(self):
            return "string_target_result"

        metadata = self.registry.get_metadata('string_target_method', 'qrofn')
        assert metadata.target_classes == ['Fuzznum']  # Should be converted to list

    def test_external_extension_target_classes_list(self):
        """Test external extension with target_classes as list."""
        @external_extension(
            'list_target_method', 
            mtype='qrofn', 
            target_classes=['Fuzznum', 'Fuzzarray'],
            auto_apply=False
        )
        def list_target_func(self):
            return "list_target_result"

        metadata = self.registry.get_metadata('list_target_method', 'qrofn')
        assert metadata.target_classes == ['Fuzznum', 'Fuzzarray']

    def test_external_extension_injection_types(self):
        """Test different injection types."""
        injection_types = ['instance_method', 'instance_property', 'top_level_function', 'both']
        
        for injection_type in injection_types:
            @external_extension(
                f'{injection_type}_method', 
                mtype='qrofn', 
                injection_type=injection_type,
                auto_apply=False
            )
            def injection_func(self):
                return f"{injection_type}_result"

            metadata = self.registry.get_metadata(f'{injection_type}_method', 'qrofn')
            assert metadata.injection_type == injection_type

    def test_external_extension_description_field(self):
        """Test external extension with description field."""
        @external_extension(
            'description_method', 
            mtype='qrofn', 
            auto_apply=False,
            description='A test method with description'
        )
        def description_func(self):
            return "description_result"

        metadata = self.registry.get_metadata('description_method', 'qrofn')
        assert metadata.description == 'A test method with description'

    def test_external_extension_function_call(self):
        """Test that the decorated function can still be called directly."""
        @external_extension('callable_method', mtype='qrofn', auto_apply=False)
        def callable_func(self, x, y=5):
            return x * y

        # Test direct function call
        class MockSelf:
            pass

        mock_self = MockSelf()
        result = callable_func(mock_self, 3, y=4)
        assert result == 12

    @patch('axisfuzzy.extension.apply_extensions')
    def test_external_extension_integration_with_apply_extensions(self, mock_apply):
        """Test integration between external_extension and apply_extensions."""
        mock_apply.return_value = True

        # Register multiple external extensions
        @external_extension('integration_method1', mtype='qrofn', auto_apply=True)
        def func1(self):
            return "result1"

        @external_extension('integration_method2', mtype='qrohfn', auto_apply=True)
        def func2(self):
            return "result2"

        # Verify apply_extensions was called for each registration
        assert mock_apply.call_count == 2
        
        # Verify all calls used force_reapply=True
        for call in mock_apply.call_args_list:
            assert call[1]['force_reapply'] is True

    def test_external_extension_error_handling_duplicate_priority(self):
        """Test error handling for duplicate registrations with same priority."""
        # Register first function
        @external_extension('duplicate_method', mtype='qrofn', priority=5, auto_apply=False)
        def first_func(self):
            return "first_result"

        # Attempt to register second function with same priority
        with pytest.raises(ValueError, match="already exists with higher or equal priority"):
            @external_extension('duplicate_method', mtype='qrofn', priority=5, auto_apply=False)
            def second_func(self):
                return "second_result"


class TestInternalStateManagement:
    """Test internal state management of the extension system."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_force_reapply_functionality(self):
        """Test force_reapply parameter functionality."""
        from axisfuzzy.extension import apply_extensions
        
        # Register a test function
        @external_extension('force_test_method', mtype='qrofn', auto_apply=False)
        def force_test_func(self):
            return "force_test_result"
        
        # Multiple applications should work
        result1 = apply_extensions()
        assert result1 is True
        
        result2 = apply_extensions()
        assert result2 is True
        
        # Force reapply should work
        result3 = apply_extensions(force_reapply=True)
        assert result3 is True

    def test_registry_state_persistence(self):
        """Test that registry state persists across apply_extensions calls."""
        from axisfuzzy.extension import apply_extensions
        
        # Register multiple functions
        @external_extension('persist_method1', mtype='qrofn', auto_apply=False)
        def persist_func1(self):
            return "persist_result1"
        
        @external_extension('persist_method2', mtype='qrohfn', auto_apply=False)
        def persist_func2(self):
            return "persist_result2"
        
        # Verify functions are registered
        assert 'persist_method1' in self.registry._functions
        assert 'persist_method2' in self.registry._functions
        
        # Apply extensions
        apply_extensions()
        
        # Verify functions are still registered after application
        assert 'persist_method1' in self.registry._functions
        assert 'persist_method2' in self.registry._functions
        
        # Verify we can still retrieve them
        func1 = self.registry.get_function('persist_method1', 'qrofn')
        func2 = self.registry.get_function('persist_method2', 'qrohfn')
        assert func1 == persist_func1
        assert func2 == persist_func2

    def test_external_extension_auto_apply_behavior(self):
        """Test the auto_apply behavior of external_extension."""
        # Test that auto_apply=True triggers apply_extensions
        with patch('axisfuzzy.extension.apply_extensions') as mock_apply:
            mock_apply.return_value = True
            
            @external_extension('auto_apply_test', mtype='qrofn', auto_apply=True)
            def auto_apply_func(self):
                return "auto_apply_result"
            
            # apply_extensions should be called with force_reapply=True
            mock_apply.assert_called_once_with(force_reapply=True)
        
        # Test that auto_apply=False does not trigger apply_extensions
        with patch('axisfuzzy.extension.apply_extensions') as mock_apply:
            mock_apply.return_value = True
            
            @external_extension('no_auto_apply_test', mtype='qrofn', auto_apply=False)
            def no_auto_apply_func(self):
                return "no_auto_apply_result"
            
            # apply_extensions should not be called
            mock_apply.assert_not_called()

    def test_external_extension_warning_on_apply_failure(self):
        """Test that external_extension issues warning when apply_extensions fails."""
        with patch('axisfuzzy.extension.apply_extensions') as mock_apply:
            mock_apply.return_value = False  # Simulate failure
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                @external_extension('warning_test', mtype='qrofn', auto_apply=True)
                def warning_func(self):
                    return "warning_result"
                
                # A warning should be issued
                assert len(w) > 0
                assert any("failed to automatically apply" in str(warning.message).lower() for warning in w)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    @patch('axisfuzzy.extension.apply_extensions')
    def test_multiple_injection_types_combination(self, mock_apply):
        """Test combination of different injection types in one session."""
        mock_apply.return_value = True
        
        # Register functions with different injection types
        @external_extension('combo_method', mtype='qrofn', injection_type='instance_method')
        def combo_instance_method(self):
            return "instance_method_result"
        
        @external_extension('combo_property', mtype='qrofn', injection_type='instance_property')
        def combo_instance_property(self):
            return "instance_property_result"
        
        @external_extension('combo_function', mtype='qrofn', injection_type='top_level_function')
        def combo_top_level_function(x):
            return "top_level_function_result"
        
        @external_extension('combo_both', mtype='qrofn', injection_type='both')
        def combo_both_function(x):
            return "both_injection_result"
        
        # Verify all functions are registered with correct injection types
        method_meta = self.registry.get_metadata('combo_method', 'qrofn')
        property_meta = self.registry.get_metadata('combo_property', 'qrofn')
        function_meta = self.registry.get_metadata('combo_function', 'qrofn')
        both_meta = self.registry.get_metadata('combo_both', 'qrofn')
        
        assert method_meta.injection_type == 'instance_method'
        assert property_meta.injection_type == 'instance_property'
        assert function_meta.injection_type == 'top_level_function'
        assert both_meta.injection_type == 'both'
        
        # Verify apply_extensions was called for each auto_apply=True registration
        assert mock_apply.call_count == 4

    def test_mixed_traditional_and_external_extensions(self):
        """Test mixing traditional @extension with @external_extension."""
        from axisfuzzy.extension import extension
        
        # Register traditional extension (should not auto-apply)
        @extension(name='traditional_method', mtype='qrofn', target_classes=['Fuzznum'])
        def traditional_func(self):
            return "traditional_result"
        
        # Register external extension (should auto-apply by default)
        with patch('axisfuzzy.extension.apply_extensions') as mock_apply:
            mock_apply.return_value = True
            
            @external_extension('external_method', mtype='qrofn')
            def external_func(self):
                return "external_result"
            
            # Only external_extension should trigger apply_extensions
            mock_apply.assert_called_once_with(force_reapply=True)
        
        # Both should be registered in the registry
        assert 'traditional_method' in self.registry._functions
        assert 'external_method' in self.registry._functions
        
        # Verify correct functions are stored
        trad_func = self.registry.get_function('traditional_method', 'qrofn')
        ext_func = self.registry.get_function('external_method', 'qrofn')
        assert trad_func == traditional_func
        assert ext_func == external_func

    @patch('axisfuzzy.extension.apply_extensions')
    def test_auto_apply_disabled_scenario(self, mock_apply):
        """Test scenario where auto_apply is disabled for multiple functions."""
        mock_apply.return_value = True
        
        # Register multiple functions with auto_apply=False
        @external_extension('manual1', mtype='qrofn', auto_apply=False)
        def manual_func1(self):
            return "manual1_result"
        
        @external_extension('manual2', mtype='qrohfn', auto_apply=False)
        def manual_func2(self):
            return "manual2_result"
        
        @external_extension('manual3', mtype='qrofn', auto_apply=False)
        def manual_func3(self):
            return "manual3_result"
        
        # apply_extensions should not be called automatically
        mock_apply.assert_not_called()
        
        # All functions should be registered
        assert 'manual1' in self.registry._functions
        assert 'manual2' in self.registry._functions
        assert 'manual3' in self.registry._functions
        
        # Manual application should work
        from axisfuzzy.extension import apply_extensions
        result = apply_extensions(force_reapply=True)
        mock_apply.assert_called_once_with(force_reapply=True)

    def test_priority_based_registration_scenario(self):
        """Test priority-based registration with conflict detection."""
        # Register high priority implementation first (higher number = higher priority)
        @external_extension('scenario_method', mtype='qrofn', priority=10, auto_apply=False)
        def high_priority_implementation(self):
            return "high_priority"

        # Verify high priority implementation is stored
        assert 'scenario_method' in self.registry._functions
        assert 'qrofn' in self.registry._functions['scenario_method']
        stored_func, metadata = self.registry._functions['scenario_method']['qrofn']
        assert metadata.priority == 10

        # Attempt to register lower priority implementation should fail
        with pytest.raises(ValueError, match="already exists with higher or equal priority"):
            @external_extension('scenario_method', mtype='qrofn', priority=5, auto_apply=False)
            def lower_priority_implementation(self):
                return "lower_priority"

class TestAppliedFlagStateManagement:
    """Test _applied flag state management functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_applied_flag_initial_state(self):
        """Test the initial state of _applied flag."""
        from axisfuzzy.extension import apply_extensions
        
        # Get the _applied flag from the module
        applied_flag = apply_extensions.__globals__.get('_applied', None)
        
        # The flag should exist and be a boolean
        assert applied_flag is not None
        assert isinstance(applied_flag, bool)

    def test_applied_flag_state_changes(self):
        """Test _applied flag state changes during extension application."""
        from axisfuzzy.extension import apply_extensions
        
        # Register a test extension
        @external_extension('flag_test_method', mtype='qrofn', auto_apply=False)
        def flag_test_func(self):
            return "flag_test_result"
        
        # Get initial state
        initial_applied = apply_extensions.__globals__['_applied']
        
        # Apply extensions
        result = apply_extensions()
        assert result is True
        
        # Check state after application
        after_applied = apply_extensions.__globals__['_applied']
        assert after_applied is True

    def test_applied_flag_force_reapply_behavior(self):
        """Test _applied flag behavior with force_reapply parameter."""
        from axisfuzzy.extension import apply_extensions
        
        # Register a test extension
        @external_extension('force_flag_test', mtype='qrofn', auto_apply=False)
        def force_flag_test_func(self):
            return "force_flag_test_result"
        
        # First application
        result1 = apply_extensions()
        assert result1 is True
        assert apply_extensions.__globals__['_applied'] is True
        
        # Second application without force_reapply (should still work)
        result2 = apply_extensions()
        assert result2 is True
        
        # Force reapply
        result3 = apply_extensions(force_reapply=True)
        assert result3 is True
        assert apply_extensions.__globals__['_applied'] is True

    def test_applied_flag_manual_reset(self):
        """Test manual reset of _applied flag."""
        from axisfuzzy.extension import apply_extensions
        
        # Register a test extension
        @external_extension('manual_reset_test', mtype='qrofn', auto_apply=False)
        def manual_reset_func(self):
            return "manual_reset_result"
        
        # Apply extensions
        apply_extensions()
        assert apply_extensions.__globals__['_applied'] is True
        
        # Manually reset the flag
        apply_extensions.__globals__['_applied'] = False
        assert apply_extensions.__globals__['_applied'] is False
        
        # Apply again - should work
        result = apply_extensions()
        assert result is True
        assert apply_extensions.__globals__['_applied'] is True

    def test_applied_flag_with_external_extension_auto_apply(self):
        """Test _applied flag behavior with auto_apply=True."""
        from axisfuzzy.extension import apply_extensions
        
        # Get initial state
        initial_applied = apply_extensions.__globals__['_applied']
        
        # Register with auto_apply=True (mocked to avoid actual application)
        with patch('axisfuzzy.extension.apply_extensions') as mock_apply:
            mock_apply.return_value = True
            
            @external_extension('auto_apply_flag_test', mtype='qrofn', auto_apply=True)
            def auto_apply_flag_func(self):
                return "auto_apply_flag_result"
            
            # Verify apply_extensions was called
            mock_apply.assert_called_once_with(force_reapply=True)


class TestPerformanceAndStability:
    """Test performance and stability of extension system."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_multiple_apply_extensions_performance(self):
        """Test performance of multiple apply_extensions calls."""
        import time
        from axisfuzzy.extension import apply_extensions
        
        # Register a test extension
        @external_extension('perf_test_method', mtype='qrofn', auto_apply=False)
        def perf_test_func(self):
            return "perf_test_result"
        
        # Measure time for multiple applications
        start_time = time.time()
        for _ in range(10):
            apply_extensions(force_reapply=True)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        elapsed_time = end_time - start_time
        assert elapsed_time < 1.0, f"Multiple apply_extensions took too long: {elapsed_time:.4f}s"

    def test_extension_registration_stability(self):
        """Test stability of extension registration under repeated operations."""
        from axisfuzzy.extension import apply_extensions
        
        # Register multiple extensions
        for i in range(5):
            @external_extension(f'stability_test_{i}', mtype='qrofn', auto_apply=False)
            def stability_func(self):
                return f"stability_result_{i}"
        
        # Apply extensions multiple times
        for _ in range(5):
            result = apply_extensions(force_reapply=True)
            assert result is True
        
        # Verify all functions are still registered
        for i in range(5):
            assert f'stability_test_{i}' in self.registry._functions
            assert 'qrofn' in self.registry._functions[f'stability_test_{i}']


class TestRealWorldIntegrationScenarios:
    """Test real-world integration scenarios with multiple extension types."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_mixed_injection_types_scenario(self):
        """Test a realistic scenario with mixed injection types."""
        from axisfuzzy.extension import apply_extensions
        
        # Register different types of extensions for a complete workflow
        @external_extension('calculate_score', mtype='qrofn', 
                          injection_type='instance_method', auto_apply=False)
        def calculate_score_method(self, data):
            return sum(data) / len(data) if data else 0
        
        @external_extension('validate_data', injection_type='top_level_function', auto_apply=False)
        def validate_data_func(data):
            return isinstance(data, list) and len(data) > 0
        
        @external_extension('data_source', mtype='qrofn', 
                          injection_type='instance_property', auto_apply=False)
        def data_source_property(self):
            return "external_data_source"
        
        # Apply all extensions
        result = apply_extensions()
        assert result is True
        
        # Verify all extensions are registered
        assert 'calculate_score' in self.registry._functions
        assert 'validate_data' in self.registry._functions
        assert 'data_source' in self.registry._functions

    def test_multi_mtype_workflow_scenario(self):
        """Test workflow with multiple mtype extensions."""
        from axisfuzzy.extension import apply_extensions
        
        # Register extensions for different mtypes
        @external_extension('process_data', mtype='qrofn', auto_apply=False)
        def process_qrofn_data(self, data):
            return f"qrofn_processed: {data}"
        
        @external_extension('process_data', mtype='triangular', auto_apply=False)
        def process_triangular_data(self, data):
            return f"triangular_processed: {data}"
        
        @external_extension('process_data', auto_apply=False)  # default
        def process_default_data(self, data):
            return f"default_processed: {data}"
        
        # Apply extensions
        result = apply_extensions()
        assert result is True
        
        # Verify all mtypes are registered
        assert 'process_data' in self.registry._functions
        assert 'qrofn' in self.registry._functions['process_data']
        assert 'triangular' in self.registry._functions['process_data']
        # Default function (mtype=None) is stored in _functions, not _defaults
        assert None in self.registry._functions['process_data']

    def test_priority_based_workflow_scenario(self):
        """Test realistic priority-based extension workflow."""
        from axisfuzzy.extension import apply_extensions
        
        # Register extensions with different priorities (higher number = higher priority)
        @external_extension('optimization_method', mtype='qrofn', 
                          priority=5, auto_apply=False)
        def basic_optimization(self):
            return "basic_optimization"
        
        @external_extension('optimization_method', mtype='qrofn', 
                          priority=10, auto_apply=False)
        def advanced_optimization(self):
            return "advanced_optimization"
        
        # Apply extensions
        apply_extensions()
        
        # Verify higher priority function is registered
        func = self.registry.get_function('optimization_method', 'qrofn')
        assert func is not None
        # The function should be the one with higher priority (10)
        assert func.__name__ == 'advanced_optimization'

    def test_error_recovery_scenario(self):
        """Test error recovery in real-world scenarios."""
        from axisfuzzy.extension import apply_extensions
        
        # Register a valid extension with explicit priority
        @external_extension('valid_method', mtype='qrofn', priority=10, auto_apply=False)
        def valid_method(self):
            return "valid_result"
        
        # Try to register with same name, mtype, and priority (should raise error)
        with pytest.raises(ValueError, match="already exists with higher or equal priority"):
            @external_extension('valid_method', mtype='qrofn', 
                              priority=10, auto_apply=False)
            def duplicate_method(self):
                return "duplicate_result"
        
        # Verify original registration is still intact
        apply_extensions()
        func = self.registry.get_function('valid_method', 'qrofn')
        assert func is not None
        assert func.__name__ == 'valid_method'

    def test_complex_extension_combination_scenario(self):
        """Test complex combination of extensions in a realistic workflow."""
        from axisfuzzy.extension import apply_extensions
        
        # Simulate a complete fuzzy logic workflow with extensions
        @external_extension('preprocess', injection_type='top_level_function', auto_apply=False)
        def preprocess_data(data):
            return [x * 2 for x in data] if isinstance(data, list) else data
        
        @external_extension('fuzzify', mtype='triangular', 
                          injection_type='instance_method', auto_apply=False)
        def triangular_fuzzify(self, value):
            return f"triangular_fuzzified({value})"
        
        @external_extension('defuzzify', mtype='triangular', 
                          injection_type='instance_method', auto_apply=False)
        def triangular_defuzzify(self, fuzzy_value):
            return f"triangular_defuzzified({fuzzy_value})"
        
        @external_extension('config', injection_type='instance_property', auto_apply=False)
        def workflow_config(self):
            return {"method": "triangular", "precision": 0.01}
        
        # Apply all extensions
        result = apply_extensions()
        assert result is True
        
        # Verify complete workflow is available
        assert 'preprocess' in self.registry._functions
        assert 'fuzzify' in self.registry._functions
        assert 'defuzzify' in self.registry._functions
        assert 'config' in self.registry._functions
        
        # Verify mtype-specific registrations
        assert 'triangular' in self.registry._functions['fuzzify']
        assert 'triangular' in self.registry._functions['defuzzify']


class TestExtensionVsExternalExtensionComparison:
    """Test comparison between traditional @extension and new @external_extension."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_registration_method_comparison(self):
        """Compare registration methods between @extension and @external_extension."""
        from axisfuzzy.extension import extension, apply_extensions
        
        # Traditional @extension registration
        @extension('traditional_method', mtype='qrofn')
        def traditional_func(self):
            return "traditional_result"
        
        # New @external_extension registration
        @external_extension('external_method', mtype='qrofn', auto_apply=False)
        def external_func(self):
            return "external_result"
        
        # Apply external extensions
        apply_extensions()
        
        # Both should be registered
        assert 'traditional_method' in self.registry._functions
        assert 'external_method' in self.registry._functions
        assert 'qrofn' in self.registry._functions['traditional_method']
        assert 'qrofn' in self.registry._functions['external_method']

    def test_auto_apply_vs_force_reapply_behavior(self):
        """Compare auto_apply behavior vs manual application behavior."""
        from axisfuzzy.extension import extension, apply_extensions
        
        # Traditional @extension (manual application)
        @extension('immediate_method', mtype='qrofn')
        def immediate_func(self):
            return "immediate_result"
        
        # New @external_extension with auto_apply=False (manual application)
        @external_extension('manual_method', mtype='qrofn', auto_apply=False)
        def manual_func(self):
            return "manual_result"
        
        # Traditional extension should be available after apply_extensions
        apply_extensions()
        immediate_func_retrieved = self.registry.get_function('immediate_method', 'qrofn')
        assert immediate_func_retrieved is not None
        
        # External extension should be available after apply_extensions
        manual_func_retrieved = self.registry.get_function('manual_method', 'qrofn')
        assert manual_func_retrieved is not None

    def test_mixed_usage_scenario(self):
        """Test mixed usage of both @extension and @external_extension."""
        from axisfuzzy.extension import extension, apply_extensions
        
        # Mix traditional and external extensions
        @extension('mixed_traditional', mtype='qrofn')
        def mixed_traditional_func(self):
            return "mixed_traditional_result"
        
        @external_extension('mixed_external', mtype='qrofn', auto_apply=False)
        def mixed_external_func(self):
            return "mixed_external_result"
        
        @extension('mixed_traditional_2', mtype='triangular')
        def mixed_traditional_func_2(self):
            return "mixed_traditional_2_result"
        
        @external_extension('mixed_external_2', mtype='triangular', auto_apply=False)
        def mixed_external_func_2(self):
            return "mixed_external_2_result"
        
        # Apply external extensions
        apply_extensions()
        
        # All should be available
        assert self.registry.get_function('mixed_traditional', 'qrofn') is not None
        assert self.registry.get_function('mixed_external', 'qrofn') is not None
        assert self.registry.get_function('mixed_traditional_2', 'triangular') is not None
        assert self.registry.get_function('mixed_external_2', 'triangular') is not None

    def test_priority_handling_comparison(self):
        """Compare priority handling between @extension and @external_extension."""
        from axisfuzzy.extension import extension, apply_extensions
        
        # Traditional @extension with priority
        @extension('priority_traditional', mtype='qrofn', priority=5)
        def priority_traditional_func(self):
            return "priority_traditional_result"
        
        # External @external_extension with higher priority
        @external_extension('priority_external', mtype='qrofn', priority=10, auto_apply=False)
        def priority_external_func(self):
            return "priority_external_result"
        
        # Apply external extensions
        apply_extensions()
        
        # Both should be registered with their respective priorities
        traditional_func = self.registry.get_function('priority_traditional', 'qrofn')
        external_func = self.registry.get_function('priority_external', 'qrofn')
        
        assert traditional_func is not None
        assert external_func is not None
        assert traditional_func.__name__ == 'priority_traditional_func'
        assert external_func.__name__ == 'priority_external_func'

    def test_injection_type_compatibility(self):
        """Test injection type compatibility between both decorators."""
        from axisfuzzy.extension import extension, apply_extensions
        
        # Traditional @extension for instance method
        @extension('compat_instance', mtype='qrofn')
        def compat_instance_func(self):
            return "compat_instance_result"
        
        # External @external_extension for top-level function
        @external_extension('compat_toplevel', injection_type='top_level_function', auto_apply=False)
        def compat_toplevel_func(data):
            return f"compat_toplevel_result: {data}"
        
        # External @external_extension for instance property
        @external_extension('compat_property', mtype='qrofn', 
                          injection_type='instance_property', auto_apply=False)
        def compat_property_func(self):
            return "compat_property_result"
        
        # Apply external extensions
        apply_extensions()
        
        # All should be registered
        assert 'compat_instance' in self.registry._functions
        assert 'compat_toplevel' in self.registry._functions
        assert 'compat_property' in self.registry._functions

    def test_error_handling_comparison(self):
        """Compare error handling between @extension and @external_extension."""
        from axisfuzzy.extension import extension, apply_extensions
        
        # Register with traditional @extension first
        @extension('error_test', mtype='qrofn', priority=10)
        def error_traditional_func(self):
            return "error_traditional_result"
        
        # Try to register with @external_extension with same priority (should fail)
        with pytest.raises(ValueError, match="already exists with higher or equal priority"):
            @external_extension('error_test', mtype='qrofn', priority=10, auto_apply=False)
            def error_external_func(self):
                return "error_external_result"
        
        # Original registration should remain intact
        func = self.registry.get_function('error_test', 'qrofn')
        assert func is not None
        assert func.__name__ == 'error_traditional_func'


class TestComprehensiveExtensionScenarios:
    """Test comprehensive extension scenarios and workflows."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_external_extension_convenience_methods(self):
        """Test the convenience of @external_extension vs traditional @extension."""
        from axisfuzzy.extension import apply_extensions
        
        # Test external_extension with auto_apply=True (convenient)
        with patch('axisfuzzy.extension.apply_extensions') as mock_apply:
            mock_apply.return_value = True
            
            @external_extension('convenient_method', mtype='qrofn', auto_apply=True)
            def convenient_func(self):
                return "convenient_result"
            
            # Should automatically call apply_extensions
            mock_apply.assert_called_once_with(force_reapply=True)
        
        # Test external_extension with auto_apply=False (manual control)
        @external_extension('manual_method', mtype='qrofn', auto_apply=False)
        def manual_func(self):
            return "manual_result"
        
        # Should be registered but not applied
        assert 'manual_method' in self.registry._functions
        
        # Manual application
        result = apply_extensions()
        assert result is True

    def test_top_level_function_registration(self):
        """Test registration of top-level functions."""
        # Register a top-level function (stored in _functions with None as mtype)
        @external_extension('top_level_distance', injection_type='top_level_function', auto_apply=False)
        def external_distance(a, b):
            """Calculate distance between two fuzzy numbers."""
            return abs(a - b)
        
        # Verify registration (top-level functions are stored in _functions with None as mtype)
        assert 'top_level_distance' in self.registry._functions
        assert None in self.registry._functions['top_level_distance']
        
        # Verify function can be retrieved (top-level functions use None as mtype)
        func = self.registry.get_function('top_level_distance', None)
        assert func == external_distance

    def test_instance_property_registration(self):
        """Test registration of instance properties."""
        # Register an instance property
        @external_extension('custom_property', mtype='qrofn', injection_type='instance_property', auto_apply=False)
        def custom_property_func(self):
            """Custom property for fuzzy numbers."""
            return f"Property value for {self}"
        
        # Verify registration
        assert 'custom_property' in self.registry._functions
        
        # Verify property function can be retrieved
        prop_func = self.registry.get_function('custom_property', 'qrofn')
        assert prop_func == custom_property_func

    def test_comprehensive_extension_workflow(self):
        """Test a comprehensive extension workflow with multiple types."""
        from axisfuzzy.extension import apply_extensions
        
        # Register various types of extensions
        @external_extension('workflow_method', mtype='qrofn', injection_type='instance_method', auto_apply=False)
        def workflow_method(self):
            return "workflow_method_result"
        
        @external_extension('workflow_property', mtype='qrofn', injection_type='instance_property', auto_apply=False)
        def workflow_property(self):
            return "workflow_property_result"
        
        @external_extension('workflow_function', injection_type='top_level_function', auto_apply=False)
        def workflow_function(x, y):
            return x + y
        
        # Verify all are registered
        assert 'workflow_method' in self.registry._functions
        assert 'workflow_property' in self.registry._functions
        assert 'workflow_function' in self.registry._functions
        assert None in self.registry._functions['workflow_function']  # top-level function uses None as mtype
        
        # Apply all extensions
        result = apply_extensions()
        assert result is True
        
        # Verify all functions are still accessible after application
        method_func = self.registry.get_function('workflow_method', 'qrofn')
        property_func = self.registry.get_function('workflow_property', 'qrofn')
        function_func = self.registry.get_function('workflow_function', None)
        
        assert method_func == workflow_method
        assert property_func == workflow_property
        assert function_func == workflow_function