#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Test suite for the ExtensionInjector.

This module tests the dynamic injection mechanism that attaches extension
methods and properties to target classes and top-level functions to module
namespaces.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from axisfuzzy.extension.injector import ExtensionInjector, get_extension_injector
from axisfuzzy.extension.registry import ExtensionRegistry
from axisfuzzy.extension.dispatcher import ExtensionDispatcher


class TestExtensionInjector:
    """Test suite for ExtensionInjector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.injector = ExtensionInjector()
        
        # Create mock classes for injection testing
        class MockFuzznum:
            def __init__(self, mtype='test'):
                self.mtype = mtype
        
        class MockFuzzarray:
            def __init__(self, mtype='test'):
                self.mtype = mtype
        
        self.MockFuzznum = MockFuzznum
        self.MockFuzzarray = MockFuzzarray
        
        # Create class map and module namespace
        self.class_map = {
            'Fuzznum': MockFuzznum,
            'Fuzzarray': MockFuzzarray
        }
        self.module_namespace = {}

    def test_injector_initialization(self):
        """Test ExtensionInjector initialization."""
        injector = ExtensionInjector()
        
        # Verify injector has registry and dispatcher
        assert hasattr(injector, 'registry')
        assert hasattr(injector, 'dispatcher')
        assert isinstance(injector.registry, ExtensionRegistry)
        assert isinstance(injector.dispatcher, ExtensionDispatcher)

    def test_get_extension_injector_singleton(self):
        """Test that get_extension_injector returns singleton instance."""
        injector1 = get_extension_injector()
        injector2 = get_extension_injector()
        
        assert injector1 is injector2
        assert isinstance(injector1, ExtensionInjector)

    def test_inject_instance_method(self):
        """Test injection of instance methods."""
        # Mock registry to return function info for instance method
        mock_functions = {
            'test_method': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'instance_method'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return a callable
            mock_method = Mock(return_value="method_result")
            with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify method was injected into Fuzznum
                assert hasattr(self.MockFuzznum, 'test_method')
                assert self.MockFuzznum.test_method is mock_method
                
                # Verify method was not injected into Fuzzarray
                assert not hasattr(self.MockFuzzarray, 'test_method')
                
                # Verify no top-level function was created
                assert 'test_method' not in self.module_namespace

    def test_inject_instance_property(self):
        """Test injection of instance properties."""
        # Mock registry to return function info for instance property
        mock_functions = {
            'test_property': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum', 'Fuzzarray'],
                        'injection_type': 'instance_property'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return a property
            mock_property = property(lambda self: "property_value")
            with patch.object(self.injector.dispatcher, 'create_instance_property', return_value=mock_property):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify property was injected into both classes
                assert hasattr(self.MockFuzznum, 'test_property')
                assert hasattr(self.MockFuzzarray, 'test_property')
                assert self.MockFuzznum.test_property is mock_property
                assert self.MockFuzzarray.test_property is mock_property
                
                # Verify no top-level function was created
                assert 'test_property' not in self.module_namespace

    def test_inject_top_level_function(self):
        """Test injection of top-level functions."""
        # Mock registry to return function info for top-level function
        mock_functions = {
            'test_function': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'top_level_function'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return a callable
            mock_function = Mock(return_value="function_result")
            with patch.object(self.injector.dispatcher, 'create_top_level_function', return_value=mock_function):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify function was injected into module namespace
                assert 'test_function' in self.module_namespace
                assert self.module_namespace['test_function'] is mock_function
                
                # Verify no instance methods were created
                assert not hasattr(self.MockFuzznum, 'test_function')
                assert not hasattr(self.MockFuzzarray, 'test_function')

    def test_inject_both_type(self):
        """Test injection of 'both' type (instance method + top-level function)."""
        # Mock registry to return function info for 'both' injection type
        mock_functions = {
            'test_both': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'both'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return callables
            mock_method = Mock(return_value="method_result")
            mock_function = Mock(return_value="function_result")
            
            with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method), \
                 patch.object(self.injector.dispatcher, 'create_top_level_function', return_value=mock_function):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify both instance method and top-level function were created
                assert hasattr(self.MockFuzznum, 'test_both')
                assert self.MockFuzznum.test_both is mock_method
                assert 'test_both' in self.module_namespace
                assert self.module_namespace['test_both'] is mock_function

    def test_inject_with_default_implementation(self):
        """Test injection when default implementation is present."""
        # Mock registry to return function info with default implementation
        mock_functions = {
            'test_default': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'instance_method'
                    }
                },
                'default': {
                    'target_classes': ['Fuzznum', 'Fuzzarray'],
                    'injection_type': 'both'
                }
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return callables
            mock_method = Mock(return_value="method_result")
            mock_function = Mock(return_value="function_result")
            
            with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method), \
                 patch.object(self.injector.dispatcher, 'create_top_level_function', return_value=mock_function):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify union of target classes (Fuzznum + Fuzzarray)
                assert hasattr(self.MockFuzznum, 'test_default')
                assert hasattr(self.MockFuzzarray, 'test_default')
                
                # Verify union of injection types (instance_method + both = both)
                assert 'test_default' in self.module_namespace

    def test_inject_multiple_implementations_same_function(self):
        """Test injection when multiple implementations exist for same function."""
        # Mock registry to return function info with multiple implementations
        mock_functions = {
            'multi_impl': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'instance_method'
                    },
                    'ifn': {
                        'target_classes': ['Fuzzarray'],
                        'injection_type': 'top_level_function'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return callables
            mock_method = Mock(return_value="method_result")
            mock_function = Mock(return_value="function_result")
            
            with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method), \
                 patch.object(self.injector.dispatcher, 'create_top_level_function', return_value=mock_function):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify union of target classes
                assert hasattr(self.MockFuzznum, 'multi_impl')
                assert hasattr(self.MockFuzzarray, 'multi_impl')
                
                # Verify union of injection types
                assert 'multi_impl' in self.module_namespace

    def test_inject_avoids_overwriting_existing_attributes(self):
        """Test that injection does not overwrite existing attributes."""
        # Add existing method to class
        def existing_method(self):
            return "existing"
        
        self.MockFuzznum.existing_method = existing_method
        
        # Add existing function to module namespace
        self.module_namespace['existing_function'] = lambda: "existing"
        
        # Mock registry to return function info that would conflict
        mock_functions = {
            'existing_method': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'instance_method'
                    }
                },
                'default': None
            },
            'existing_function': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'top_level_function'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return callables
            mock_method = Mock(return_value="new_method")
            mock_function = Mock(return_value="new_function")
            
            with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method), \
                 patch.object(self.injector.dispatcher, 'create_top_level_function', return_value=mock_function):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify existing attributes were not overwritten
                assert self.MockFuzznum.existing_method is existing_method
                assert self.module_namespace['existing_function']() == "existing"

    def test_inject_with_missing_target_classes(self):
        """Test injection when target classes are missing from class_map."""
        # Mock registry to return function info with missing target class
        mock_functions = {
            'missing_class_method': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['MissingClass', 'Fuzznum'],
                        'injection_type': 'instance_method'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return a callable
            mock_method = Mock(return_value="method_result")
            with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method):
                
                # Perform injection
                self.injector.inject_all(self.class_map, self.module_namespace)
                
                # Verify method was injected only into existing class
                assert hasattr(self.MockFuzznum, 'missing_class_method')
                # MissingClass is not in class_map, so no error should occur

    def test_inject_empty_registry(self):
        """Test injection when registry is empty."""
        with patch.object(self.injector.registry, 'list_functions', return_value={}):
            # Should not raise any errors
            self.injector.inject_all(self.class_map, self.module_namespace)
            
            # Verify no attributes were added
            original_fuzznum_attrs = set(dir(self.MockFuzznum))
            original_fuzzarray_attrs = set(dir(self.MockFuzzarray))
            
            # Create new instances to check
            fuzznum_attrs = set(dir(self.MockFuzznum))
            fuzzarray_attrs = set(dir(self.MockFuzzarray))
            
            # Should be the same (no new attributes)
            assert fuzznum_attrs == original_fuzznum_attrs
            assert fuzzarray_attrs == original_fuzzarray_attrs
            assert len(self.module_namespace) == 0

    def test_inject_function_detailed_behavior(self):
        """Test detailed behavior of _inject_function method."""
        # Test the internal _inject_function method directly
        func_info = {
            'implementations': {
                'qrofn': {
                    'target_classes': ['Fuzznum'],
                    'injection_type': 'instance_method'
                },
                'ifn': {
                    'target_classes': ['Fuzzarray'],
                    'injection_type': 'instance_property'
                }
            },
            'default': {
                'target_classes': ['Fuzznum', 'Fuzzarray'],
                'injection_type': 'top_level_function'
            }
        }
        
        # Mock dispatcher methods
        mock_method = Mock(return_value="method_result")
        mock_property = property(lambda self: "property_value")
        mock_function = Mock(return_value="function_result")
        
        with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method), \
             patch.object(self.injector.dispatcher, 'create_instance_property', return_value=mock_property), \
             patch.object(self.injector.dispatcher, 'create_top_level_function', return_value=mock_function):
            
            # Call _inject_function directly
            self.injector._inject_function('test_func', func_info, self.class_map, self.module_namespace)
            
            # Verify all injection types were applied
            assert hasattr(self.MockFuzznum, 'test_func')  # instance_method
            assert hasattr(self.MockFuzzarray, 'test_func')  # instance_property
            assert 'test_func' in self.module_namespace  # top_level_function

    def test_injection_idempotency(self):
        """Test that repeated injection calls are idempotent."""
        # Mock registry to return function info
        mock_functions = {
            'idempotent_test': {
                'implementations': {
                    'qrofn': {
                        'target_classes': ['Fuzznum'],
                        'injection_type': 'both'
                    }
                },
                'default': None
            }
        }
        
        with patch.object(self.injector.registry, 'list_functions', return_value=mock_functions):
            # Mock dispatcher to return callables
            mock_method = Mock(return_value="method_result")
            mock_function = Mock(return_value="function_result")
            
            with patch.object(self.injector.dispatcher, 'create_instance_method', return_value=mock_method), \
                 patch.object(self.injector.dispatcher, 'create_top_level_function', return_value=mock_function):
                
                # Perform injection twice
                self.injector.inject_all(self.class_map, self.module_namespace)
                first_method = self.MockFuzznum.idempotent_test
                first_function = self.module_namespace['idempotent_test']
                
                self.injector.inject_all(self.class_map, self.module_namespace)
                second_method = self.MockFuzznum.idempotent_test
                second_function = self.module_namespace['idempotent_test']
                
                # Verify same objects are used (no duplication)
                assert first_method is second_method
                assert first_function is second_function


class TestInjectionIntegration:
    """Integration tests for injection with real registry and dispatcher."""

    def setup_method(self):
        """Set up test fixtures with real components."""
        # Create fresh registry and dispatcher for isolated testing
        self.registry = ExtensionRegistry()
        self.dispatcher = ExtensionDispatcher()
        self.injector = ExtensionInjector()
        
        # Override injector's registry and dispatcher for testing
        self.injector.registry = self.registry
        self.injector.dispatcher = self.dispatcher
        
        # Ensure dispatcher uses the same registry
        self.dispatcher.registry = self.registry
        
        # Create mock classes
        class MockFuzznum:
            def __init__(self, mtype='test'):
                self.mtype = mtype
        
        class MockFuzzarray:
            def __init__(self, mtype='test'):
                self.mtype = mtype
        
        self.MockFuzznum = MockFuzznum
        self.MockFuzzarray = MockFuzzarray
        
        self.class_map = {
            'Fuzznum': MockFuzznum,
            'Fuzzarray': MockFuzzarray
        }
        self.module_namespace = {}

    def test_end_to_end_injection_flow(self):
        """Test the complete flow from registration to injection to execution."""
        # Register test implementations using decorator syntax
        @self.registry.register(
            'test_method', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_method'
        )
        def test_method_impl(self):
            return "qrofn method result"
        
        @self.registry.register(
            'test_property', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='instance_property'
        )
        def test_property_impl(self):
            return "qrofn property value"
        
        @self.registry.register(
            'test_function', mtype='qrofn',
            target_classes=['Fuzznum'], injection_type='top_level_function'
        )
        def test_function_impl(obj):
            return "qrofn function result"
        
        # Perform injection
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test injected instance method
        obj = self.MockFuzznum('qrofn')
        assert hasattr(self.MockFuzznum, 'test_method')
        assert obj.test_method() == "qrofn method result"
        
        # Test injected instance property
        assert hasattr(self.MockFuzznum, 'test_property')
        assert obj.test_property == "qrofn property value"
        
        # Test injected top-level function
        assert 'test_function' in self.module_namespace
        assert self.module_namespace['test_function'](obj) == "qrofn function result"

    def test_injection_with_fallback_to_default(self):
        """Test injection behavior when falling back to default implementations."""
        # Register default implementation using decorator syntax
        @self.registry.register(
            'fallback_test', mtype=None,  # None = default
            target_classes=['Fuzznum'], injection_type='instance_method', is_default=True
        )
        def default_impl(self):
            return "default implementation"
        
        # Perform injection
        self.injector.inject_all(self.class_map, self.module_namespace)
        
        # Test with object that has no specific implementation
        obj = self.MockFuzznum('unknown_type')
        assert hasattr(self.MockFuzznum, 'fallback_test')
        assert obj.fallback_test() == "default implementation"