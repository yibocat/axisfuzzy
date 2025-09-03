import pytest
from typing import Dict, List, Any

from axisfuzzy.fuzzifier import (
    FuzzificationStrategy,
    get_registry_fuzzify,
    register_fuzzifier
)


class TestFuzzificationStrategyRegistry:
    
    def test_registry_singleton(self):
        registry1 = get_registry_fuzzify()
        registry2 = get_registry_fuzzify()
        assert registry1 is registry2
    
    def test_initial_registry_state(self, registry_instance):
        available_mtypes = registry_instance.get_available_mtypes()
        assert 'qrofn' in available_mtypes
        assert 'qrohfn' in available_mtypes
        
        qrofn_methods = registry_instance.get_available_methods('qrofn')
        assert 'default' in qrofn_methods
        
        qrohfn_methods = registry_instance.get_available_methods('qrohfn')
        assert 'default' in qrohfn_methods
    
    def test_get_default_method(self, registry_instance):
        default_qrofn = registry_instance.get_default_method('qrofn')
        assert default_qrofn == 'default'
        
        default_qrohfn = registry_instance.get_default_method('qrohfn')
        assert default_qrohfn == 'default'
        
        nonexistent = registry_instance.get_default_method('nonexistent')
        assert nonexistent is None
    
    def test_get_strategy(self, registry_instance):
        qrofn_strategy = registry_instance.get_strategy('qrofn', 'default')
        assert qrofn_strategy is not None
        assert issubclass(qrofn_strategy, FuzzificationStrategy)
        
        qrohfn_strategy = registry_instance.get_strategy('qrohfn', 'default')
        assert qrohfn_strategy is not None
        assert issubclass(qrohfn_strategy, FuzzificationStrategy)
        
        nonexistent_strategy = registry_instance.get_strategy('nonexistent', 'default')
        assert nonexistent_strategy is None
    
    def test_get_strategy_with_default_method(self, registry_instance):
        qrofn_strategy = registry_instance.get_strategy('qrofn')
        assert qrofn_strategy is not None
        
        qrohfn_strategy = registry_instance.get_strategy('qrohfn')
        assert qrohfn_strategy is not None
        
        nonexistent_strategy = registry_instance.get_strategy('nonexistent')
        assert nonexistent_strategy is None
    
    def test_list_strategies(self, registry_instance):
        all_strategies = registry_instance.list_strategies()
        assert len(all_strategies) >= 2
        assert ('qrofn', 'default') in all_strategies
        assert ('qrohfn', 'default') in all_strategies
        
        qrofn_strategies = registry_instance.list_strategies('qrofn')
        assert ('qrofn', 'default') in qrofn_strategies
        
        qrohfn_strategies = registry_instance.list_strategies('qrohfn')
        assert ('qrohfn', 'default') in qrohfn_strategies
    
    def test_get_registry_info(self, registry_instance):
        info = registry_instance.get_registry_info()
        assert 'total_strategies' in info
        assert 'supported_mtypes' in info
        assert 'default_methods' in info
        assert 'strategies' in info
        
        assert info['total_strategies'] >= 2
        assert 'qrofn' in info['supported_mtypes']
        assert 'qrohfn' in info['supported_mtypes']
        assert info['default_methods']['qrofn'] == 'default'
        assert info['default_methods']['qrohfn'] == 'default'


class TestStrategyRegistration:
    
    def test_manual_registration(self):
        registry = get_registry_fuzzify()
        
        class TestStrategy(FuzzificationStrategy):
            mtype = 'test'
            method = 'manual'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        registry.register('test', 'manual', TestStrategy, is_default=True)
        
        assert registry.get_strategy('test', 'manual') is TestStrategy
        assert registry.get_default_method('test') == 'manual'
        assert 'test' in registry.get_available_mtypes()
        assert 'manual' in registry.get_available_methods('test')
    
    def test_registration_validation(self):
        registry = get_registry_fuzzify()
        
        class InvalidStrategy:
            pass
        
        with pytest.raises(TypeError, match="must be a subclass of FuzzificationStrategy"):
            registry.register('invalid', 'test', InvalidStrategy)
    
    def test_duplicate_registration(self):
        registry = get_registry_fuzzify()
        
        class TestStrategy1(FuzzificationStrategy):
            mtype = 'test'
            method = 'duplicate'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        class TestStrategy2(FuzzificationStrategy):
            mtype = 'test'
            method = 'duplicate'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        registry.register('test', 'duplicate', TestStrategy1)
        
        with pytest.raises(ValueError, match="Strategy for .* is already registered"):
            registry.register('test', 'duplicate', TestStrategy2)
    
    def test_decorator_registration(self):
        original_registry = get_registry_fuzzify()
        original_strategies = original_registry.list_strategies()
        
        @register_fuzzifier(is_default=False)
        class DecoratorTestStrategy(FuzzificationStrategy):
            mtype = 'decorator_test'
            method = 'test_method'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        registry = get_registry_fuzzify()
        assert registry.get_strategy('decorator_test', 'test_method') is DecoratorTestStrategy
        assert 'decorator_test' in registry.get_available_mtypes()
        assert 'test_method' in registry.get_available_methods('decorator_test')
    
    def test_default_strategy_override(self):
        registry = get_registry_fuzzify()
        
        class Strategy1(FuzzificationStrategy):
            mtype = 'override_test'
            method = 'method1'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        class Strategy2(FuzzificationStrategy):
            mtype = 'override_test'
            method = 'method2'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        registry.register('override_test', 'method1', Strategy1, is_default=True)
        assert registry.get_default_method('override_test') == 'method1'
        
        registry.register('override_test', 'method2', Strategy2, is_default=True)
        assert registry.get_default_method('override_test') == 'method2'
    
    def test_first_registration_becomes_default(self):
        registry = get_registry_fuzzify()
        
        class Strategy1(FuzzificationStrategy):
            mtype = 'first_default_test'
            method = 'method1'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        class Strategy2(FuzzificationStrategy):
            mtype = 'first_default_test'
            method = 'method2'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        registry.register('first_default_test', 'method1', Strategy1, is_default=False)
        assert registry.get_default_method('first_default_test') == 'method1'
        
        registry.register('first_default_test', 'method2', Strategy2, is_default=False)
        assert registry.get_default_method('first_default_test') == 'method1'


class TestRegistryEdgeCases:

    def test_get_available_methods_nonexistent_mtype(self, registry_instance):
        methods = registry_instance.get_available_methods('nonexistent_mtype')
        assert methods == []
    
    def test_registry_info_format(self, registry_instance):
        info = registry_instance.get_registry_info()
        
        assert isinstance(info['total_strategies'], int)
        assert isinstance(info['supported_mtypes'], list)
        assert isinstance(info['default_methods'], dict)
        assert isinstance(info['strategies'], dict)
        
        for strategy_key, strategy_name in info['strategies'].items():
            assert '.' in strategy_key
            assert isinstance(strategy_name, str)
    
    def test_strategy_class_attributes(self, registry_instance):
        qrofn_strategy_cls = registry_instance.get_strategy('qrofn', 'default')
        assert hasattr(qrofn_strategy_cls, 'mtype')
        assert hasattr(qrofn_strategy_cls, 'method')
        assert qrofn_strategy_cls.mtype == 'qrofn'
        assert qrofn_strategy_cls.method == 'default'
        
        qrohfn_strategy_cls = registry_instance.get_strategy('qrohfn', 'default')
        assert hasattr(qrohfn_strategy_cls, 'mtype')
        assert hasattr(qrohfn_strategy_cls, 'method')
        assert qrohfn_strategy_cls.mtype == 'qrohfn'
        assert qrohfn_strategy_cls.method == 'default'


if __name__ == '__main__':
    pytest.main()
    