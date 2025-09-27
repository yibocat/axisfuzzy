"""
Tests for the extension dispatcher and type-based method dispatch mechanism.

This module tests the ExtensionDispatcher class which creates dynamic proxy
callables that resolve the correct implementation at runtime based on the
mtype of involved fuzzy objects.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from axisfuzzy.extension.dispatcher import ExtensionDispatcher, get_extension_dispatcher
from axisfuzzy.extension.registry import get_registry_extension


class TestExtensionDispatcher:
    """Test the ExtensionDispatcher functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.dispatcher = ExtensionDispatcher()
        self.registry = get_registry_extension()
        # Clear any existing registrations for clean testing
        self.registry._functions.clear()
        self.registry._defaults.clear()

    def test_dispatcher_initialization(self):
        """Test dispatcher initialization and registry binding."""
        dispatcher = ExtensionDispatcher()
        assert dispatcher.registry is not None
        assert dispatcher.registry == get_registry_extension()

    def test_get_extension_dispatcher_singleton(self):
        """Test that get_extension_dispatcher returns a singleton."""
        dispatcher1 = get_extension_dispatcher()
        dispatcher2 = get_extension_dispatcher()
        assert dispatcher1 is dispatcher2
        assert isinstance(dispatcher1, ExtensionDispatcher)

    def test_create_instance_method_basic(self):
        """Test creating a basic instance method proxy."""
        # Register a test function
        def test_func(obj, x, y=10):
            return f"test_result_{obj.mtype}_{x}_{y}"

        self.registry._functions['test_method'] = {
            'qrofn': (test_func, Mock())
        }

        # Create instance method proxy
        method_proxy = self.dispatcher.create_instance_method('test_method')
        
        # Verify proxy attributes
        assert callable(method_proxy)
        assert method_proxy.__name__ == 'test_method'
        assert 'Dispatched method for test_method' in method_proxy.__doc__

        # Test method call with mock object
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'
        
        result = method_proxy(mock_obj, 5, y=20)
        assert result == "test_result_qrofn_5_20"

    def test_create_instance_method_no_mtype_attribute(self):
        """Test instance method proxy with object lacking mtype attribute."""
        method_proxy = self.dispatcher.create_instance_method('test_method')
        
        mock_obj = Mock(spec=[])  # Object without mtype attribute
        
        with pytest.raises(AttributeError, match="has no 'mtype' attribute"):
            method_proxy(mock_obj)

    def test_create_instance_method_not_implemented(self):
        """Test instance method proxy when implementation is not found."""
        method_proxy = self.dispatcher.create_instance_method('nonexistent_method')
        
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'
        
        with pytest.raises(NotImplementedError, match="Function 'nonexistent_method' not implemented for mtype 'qrofn'"):
            method_proxy(mock_obj)

    def test_create_instance_method_with_available_mtypes_error(self):
        """Test instance method error message includes available mtypes."""
        # Register function for different mtype
        def test_func(obj):
            return "result"

        self.registry._functions['test_method'] = {
            'qrohfn': (test_func, Mock()),
            'ifn': (test_func, Mock())
        }

        method_proxy = self.dispatcher.create_instance_method('test_method')
        
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'  # Not registered
        
        with pytest.raises(NotImplementedError) as exc_info:
            method_proxy(mock_obj)
        
        error_msg = str(exc_info.value)
        assert "not implemented for mtype 'qrofn'" in error_msg
        assert "Available for: ['qrohfn', 'ifn']" in error_msg

    def test_create_instance_method_with_default_available(self):
        """Test instance method uses default implementation when specialized not found."""
        # Register default implementation
        def default_func(obj):
            return "default_result"

        self.registry._defaults['test_method'] = (default_func, Mock())

        method_proxy = self.dispatcher.create_instance_method('test_method')
        
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'
        
        # Should use default implementation, not raise error
        result = method_proxy(mock_obj)
        assert result == "default_result"

    def test_create_top_level_function_basic(self):
        """Test creating a basic top-level function proxy."""
        # Register a test function
        def test_func(obj, x, y=10):
            return f"top_level_result_{obj.mtype}_{x}_{y}"

        self.registry._functions['test_function'] = {
            'qrofn': (test_func, Mock())
        }

        # Create top-level function proxy
        func_proxy = self.dispatcher.create_top_level_function('test_function')
        
        # Verify proxy attributes
        assert callable(func_proxy)
        assert func_proxy.__name__ == 'test_function'
        assert 'Dispatched top-level function' in func_proxy.__doc__

    @patch('axisfuzzy.extension.dispatcher.get_registry_fuzztype')
    @patch('axisfuzzy.extension.dispatcher.get_config')
    def test_create_top_level_function_mtype_from_kwargs(self, mock_config, mock_fuzztype_registry):
        """Test top-level function with explicit mtype in kwargs."""
        # Setup mocks
        mock_fuzztype_registry.return_value.get_registered_mtypes.return_value = {'qrofn': Mock()}
        
        # Register a test function
        def test_func(obj, x):
            return f"result_{x}"

        self.registry._functions['test_function'] = {
            'qrofn': (test_func, Mock())
        }

        func_proxy = self.dispatcher.create_top_level_function('test_function')
        
        mock_obj = Mock()
        result = func_proxy(mock_obj, 42, mtype='qrofn')
        assert result == "result_42"

    @patch('axisfuzzy.extension.dispatcher.get_registry_fuzztype')
    @patch('axisfuzzy.extension.dispatcher.get_config')
    def test_create_top_level_function_mtype_from_first_arg(self, mock_config, mock_fuzztype_registry):
        """Test top-level function with mtype inferred from first argument."""
        # Setup mocks
        mock_fuzztype_registry.return_value.get_registered_mtypes.return_value = {'qrofn': Mock()}
        
        # Register a test function
        def test_func(obj, x):
            return f"inferred_result_{obj.mtype}_{x}"

        self.registry._functions['test_function'] = {
            'qrofn': (test_func, Mock())
        }

        func_proxy = self.dispatcher.create_top_level_function('test_function')
        
        # Create mock Fuzznum object
        from axisfuzzy.core import Fuzznum
        mock_obj = Mock(spec=Fuzznum)
        mock_obj.mtype = 'qrofn'
        
        result = func_proxy(mock_obj, 42)
        assert result == "inferred_result_qrofn_42"

    @patch('axisfuzzy.extension.dispatcher.get_registry_fuzztype')
    @patch('axisfuzzy.extension.dispatcher.get_config')
    def test_create_top_level_function_invalid_mtype(self, mock_config, mock_fuzztype_registry):
        """Test top-level function with invalid mtype."""
        # Setup mocks
        mock_fuzztype_registry.return_value.get_registered_mtypes.return_value = {'qrofn': Mock()}
        
        func_proxy = self.dispatcher.create_top_level_function('test_function')
        
        with pytest.raises(ValueError, match="Invalid fuzzy number type 'invalid_type'"):
            func_proxy(Mock(), mtype='invalid_type')

    @patch('axisfuzzy.extension.dispatcher.get_registry_fuzztype')
    @patch('axisfuzzy.extension.dispatcher.get_config')
    def test_create_top_level_function_default_mtype(self, mock_config, mock_fuzztype_registry):
        """Test top-level function falling back to default mtype."""
        # Setup mocks
        mock_config.return_value.DEFAULT_MTYPE = 'qrofn'
        mock_fuzztype_registry.return_value.get_registered_mtypes.return_value = {'qrofn': Mock()}
        
        # Register a test function
        def test_func(x):
            return f"default_result_{x}"

        self.registry._functions['test_function'] = {
            'qrofn': (test_func, Mock())
        }

        func_proxy = self.dispatcher.create_top_level_function('test_function')
        
        result = func_proxy(42)  # No mtype specified, no Fuzznum object
        assert result == "default_result_42"

    def test_create_top_level_function_not_implemented(self):
        """Test top-level function when implementation is not found."""
        func_proxy = self.dispatcher.create_top_level_function('nonexistent_function')
        
        with pytest.raises(NotImplementedError, match="Function 'nonexistent_function' is not registered at all"):
            func_proxy(Mock())

    def test_create_top_level_function_detailed_error_messages(self):
        """Test detailed error messages for top-level functions."""
        # Register function for specific mtype
        def test_func(obj):
            return "result"

        self.registry._functions['test_function'] = {
            'qrohfn': (test_func, Mock())
        }

        func_proxy = self.dispatcher.create_top_level_function('test_function')
        
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'  # Different from registered
        
        with pytest.raises(NotImplementedError) as exc_info:
            func_proxy(mock_obj)
        
        error_msg = str(exc_info.value)
        assert "not implemented for mtype 'qrofn'" in error_msg
        assert "Available specialized mtypes:" in error_msg
        assert "qrohfn" in error_msg

    def test_create_instance_property_basic(self):
        """Test creating a basic instance property proxy."""
        # Register a test function
        def test_property_func(obj):
            return f"property_value_{obj.mtype}"

        self.registry._functions['test_property'] = {
            'qrofn': (test_property_func, Mock())
        }

        # Create instance property proxy
        prop_proxy = self.dispatcher.create_instance_property('test_property')
        
        # Verify it's a property
        assert isinstance(prop_proxy, property)
        assert 'Dispatched property for test_property' in prop_proxy.__doc__

        # Test property access with mock object
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'
        
        result = prop_proxy.fget(mock_obj)
        assert result == "property_value_qrofn"

    def test_create_instance_property_no_mtype_attribute(self):
        """Test instance property proxy with object lacking mtype attribute."""
        prop_proxy = self.dispatcher.create_instance_property('test_property')
        
        mock_obj = Mock(spec=[])  # Object without mtype attribute
        
        with pytest.raises(AttributeError, match="has no 'mtype' attribute"):
            prop_proxy.fget(mock_obj)

    def test_create_instance_property_not_implemented(self):
        """Test instance property proxy when implementation is not found."""
        prop_proxy = self.dispatcher.create_instance_property('nonexistent_property')
        
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'
        
        with pytest.raises(NotImplementedError, match="Property 'nonexistent_property' not implemented for mtype 'qrofn'"):
            prop_proxy.fget(mock_obj)

    def test_create_instance_property_with_available_mtypes_error(self):
        """Test instance property error message includes available mtypes."""
        # Register property for different mtype
        def test_property_func(obj):
            return "value"

        self.registry._functions['test_property'] = {
            'qrohfn': (test_property_func, Mock()),
            'ifn': (test_property_func, Mock())
        }

        prop_proxy = self.dispatcher.create_instance_property('test_property')
        
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'  # Not registered
        
        with pytest.raises(NotImplementedError) as exc_info:
            prop_proxy.fget(mock_obj)
        
        error_msg = str(exc_info.value)
        assert "not implemented for mtype 'qrofn'" in error_msg
        assert "Available for: ['qrohfn', 'ifn']" in error_msg

    def test_create_instance_property_with_default_available(self):
        """Test instance property uses default implementation when specialized not found."""
        # Register default implementation
        def default_property_func(obj):
            return "default_value"

        self.registry._defaults['test_property'] = (default_property_func, Mock())

        prop_proxy = self.dispatcher.create_instance_property('test_property')
        
        mock_obj = Mock()
        mock_obj.mtype = 'qrofn'
        
        # Should use default implementation, not raise error
        result = prop_proxy.fget(mock_obj)
        assert result == "default_value"

    def test_dispatcher_registry_integration(self):
        """Test integration between dispatcher and registry."""
        # Register multiple implementations
        def qrofn_func(obj, x):
            return f"qrofn_{x}"

        def qrohfn_func(obj, x):
            return f"qrohfn_{x}"

        def default_func(obj, x):
            return f"default_{x}"

        self.registry._functions['multi_impl'] = {
            'qrofn': (qrofn_func, Mock()),
            'qrohfn': (qrohfn_func, Mock())
        }
        self.registry._defaults['multi_impl'] = (default_func, Mock())

        # Create method proxy
        method_proxy = self.dispatcher.create_instance_method('multi_impl')

        # Test different mtype dispatching
        mock_obj_qrofn = Mock()
        mock_obj_qrofn.mtype = 'qrofn'
        assert method_proxy(mock_obj_qrofn, 42) == "qrofn_42"

        mock_obj_qrohfn = Mock()
        mock_obj_qrohfn.mtype = 'qrohfn'
        assert method_proxy(mock_obj_qrohfn, 42) == "qrohfn_42"

    def test_dispatcher_thread_safety(self):
        """Test that dispatcher operations are thread-safe."""
        import threading
        import time

        results = []
        errors = []

        def register_and_dispatch(mtype_suffix):
            try:
                # Register function
                def test_func(obj, x):
                    return f"result_{mtype_suffix}_{x}"

                mtype = f"test_type_{mtype_suffix}"
                self.registry._functions[f'thread_test_{mtype_suffix}'] = {
                    mtype: (test_func, Mock())
                }

                # Create and use dispatcher
                method_proxy = self.dispatcher.create_instance_method(f'thread_test_{mtype_suffix}')
                
                mock_obj = Mock()
                mock_obj.mtype = mtype
                
                result = method_proxy(mock_obj, 100)
                results.append(result)
                
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_and_dispatch, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        for i, result in enumerate(results):
            assert f"result_{i}_100" in result