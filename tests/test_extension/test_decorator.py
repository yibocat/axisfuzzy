"""
测试 AxisFuzzy 扩展装饰器功能

本模块测试 @extension 和 @batch_extension 装饰器的核心功能，包括：
- 装饰器参数验证
- 函数注册到 ExtensionRegistry
- 函数签名和元数据保持
- 不同注册配置的处理
- 错误处理和边界情况
"""

import pytest
import inspect
from unittest.mock import Mock, patch

from axisfuzzy.extension.decorator import extension, batch_extension
from axisfuzzy.extension.registry import ExtensionRegistry, get_registry_extension


class TestExtensionDecorator:
    """测试 @extension 装饰器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 创建新的注册表实例用于测试
        self.registry = ExtensionRegistry()
    
    def test_extension_basic_registration(self):
        """测试基本的扩展函数注册"""
        # 使用 patch 替换全局注册表
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('test_func', mtype='qrofn', target_classes=['Fuzznum'])
            def test_function(x, y):
                """测试函数"""
                return x + y
        
        # 验证函数已注册
        assert 'test_func' in self.registry._functions
        assert 'qrofn' in self.registry._functions['test_func']
        
        # 验证注册的函数是原函数
        stored_func, metadata = self.registry._functions['test_func']['qrofn']
        assert stored_func is test_function
        
        # 验证元数据
        assert metadata.name == 'test_func'
        assert metadata.mtype == 'qrofn'
        assert metadata.target_classes == ['Fuzznum']
        assert metadata.injection_type == 'both'
        assert metadata.is_default is False
        assert metadata.priority == 0
    
    def test_extension_default_registration(self):
        """测试默认扩展函数注册"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('default_func', is_default=True, target_classes=['Fuzznum'])
            def default_function():
                return "default"
        
        # 验证默认函数已注册
        assert 'default_func' in self.registry._defaults
        stored_func, metadata = self.registry._defaults['default_func']
        assert stored_func is default_function
        assert metadata.is_default is True
    
    def test_extension_with_priority(self):
        """测试带优先级的扩展函数注册"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('priority_func', mtype='qrofn', priority=10, target_classes=['Fuzznum'])
            def high_priority_function():
                return "high"
        
        stored_func, metadata = self.registry._functions['priority_func']['qrofn']
        assert metadata.priority == 10
    
    def test_extension_with_injection_type(self):
        """测试不同注入类型的扩展函数注册"""
        injection_types = ['instance_method', 'top_level_function', 'instance_property', 'both']
        
        for injection_type in injection_types:
            with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
                @extension(f'func_{injection_type}', mtype='qrofn', 
                          injection_type=injection_type, target_classes=['Fuzznum'])
                def test_func():
                    return injection_type
            
            stored_func, metadata = self.registry._functions[f'func_{injection_type}']['qrofn']
            assert metadata.injection_type == injection_type
    
    def test_extension_with_multiple_target_classes(self):
        """测试多个目标类的扩展函数注册"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('multi_target', mtype='qrofn', target_classes=['Fuzznum', 'Fuzzarray'])
            def multi_target_function():
                return "multi"
        
        stored_func, metadata = self.registry._functions['multi_target']['qrofn']
        assert metadata.target_classes == ['Fuzznum', 'Fuzzarray']
    
    def test_extension_with_single_target_class_string(self):
        """测试单个目标类（字符串形式）的扩展函数注册"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('single_target', mtype='qrofn', target_classes='Fuzznum')
            def single_target_function():
                return "single"
        
        stored_func, metadata = self.registry._functions['single_target']['qrofn']
        # 字符串会被转换为列表
        assert metadata.target_classes == ['Fuzznum']
    
    def test_extension_with_kwargs(self):
        """测试带额外关键字参数的扩展函数注册"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('kwargs_func', mtype='qrofn', target_classes=['Fuzznum'],
                      description="Test function with kwargs")
            def kwargs_function():
                return "kwargs"
        
        stored_func, metadata = self.registry._functions['kwargs_func']['qrofn']
        assert metadata.description == "Test function with kwargs"
    
    def test_extension_preserves_function_signature(self):
        """测试装饰器保持函数签名"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('signature_test', mtype='qrofn', target_classes=['Fuzznum'])
            def original_function(a: int, b: str = "default", *args, **kwargs) -> str:
                """Original docstring"""
                return f"{a}_{b}"
        
        # 验证函数名和文档字符串保持不变
        assert original_function.__name__ == 'original_function'
        assert original_function.__doc__ == "Original docstring"
        
        # 验证函数签名保持不变
        sig = inspect.signature(original_function)
        params = list(sig.parameters.keys())
        assert params == ['a', 'b', 'args', 'kwargs']
        assert sig.parameters['a'].annotation == int
        assert sig.parameters['b'].default == "default"
        assert sig.return_annotation == str
    
    def test_extension_function_still_callable(self):
        """测试装饰后的函数仍然可调用"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('callable_test', mtype='qrofn', target_classes=['Fuzznum'])
            def callable_function(x, y):
                return x * y
        
        # 验证函数仍然可以正常调用
        result = callable_function(3, 4)
        assert result == 12
    
    def test_extension_without_mtype(self):
        """测试不指定 mtype 的扩展函数注册"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @extension('no_mtype', target_classes=['Fuzznum'], is_default=True)
            def no_mtype_function():
                return "no_mtype"
        
        # 应该注册为默认函数
        assert 'no_mtype' in self.registry._defaults
        stored_func, metadata = self.registry._defaults['no_mtype']
        assert stored_func is no_mtype_function
        assert metadata.mtype is None
        assert metadata.is_default is True


class TestBatchExtensionDecorator:
    """测试 @batch_extension 装饰器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.registry = ExtensionRegistry()
    
    def test_batch_extension_multiple_registrations(self):
        """测试批量注册多个配置"""
        registrations = [
            {'name': 'batch_func', 'mtype': 'qrofn', 'target_classes': ['Fuzznum']},
            {'name': 'batch_func', 'mtype': 'ivfn', 'target_classes': ['Fuzznum']},
            {'name': 'batch_func', 'is_default': True, 'target_classes': ['Fuzznum']}
        ]
        
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @batch_extension(registrations)
            def batch_function():
                return "batch"
        
        # 验证所有配置都已注册
        assert 'batch_func' in self.registry._functions
        assert 'qrofn' in self.registry._functions['batch_func']
        assert 'ivfn' in self.registry._functions['batch_func']
        assert 'batch_func' in self.registry._defaults
        
        # 验证所有注册的函数都是同一个函数
        qrofn_func, _ = self.registry._functions['batch_func']['qrofn']
        ivfn_func, _ = self.registry._functions['batch_func']['ivfn']
        default_func, _ = self.registry._defaults['batch_func']
        
        assert qrofn_func is batch_function
        assert ivfn_func is batch_function
        assert default_func is batch_function
    
    def test_batch_extension_different_names(self):
        """测试批量注册不同名称的配置"""
        registrations = [
            {'name': 'func_a', 'mtype': 'qrofn', 'target_classes': ['Fuzznum']},
            {'name': 'func_b', 'mtype': 'qrofn', 'target_classes': ['Fuzzarray']},
        ]
        
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @batch_extension(registrations)
            def multi_name_function():
                return "multi_name"
        
        # 验证不同名称都已注册
        assert 'func_a' in self.registry._functions
        assert 'func_b' in self.registry._functions
        
        func_a, metadata_a = self.registry._functions['func_a']['qrofn']
        func_b, metadata_b = self.registry._functions['func_b']['qrofn']
        
        assert func_a is multi_name_function
        assert func_b is multi_name_function
        assert metadata_a.target_classes == ['Fuzznum']
        assert metadata_b.target_classes == ['Fuzzarray']
    
    def test_batch_extension_different_priorities(self):
        """测试批量注册不同优先级的配置"""
        registrations = [
            {'name': 'priority_func', 'mtype': 'qrofn', 'priority': 5, 'target_classes': ['Fuzznum']},
            {'name': 'priority_func', 'mtype': 'ivfn', 'priority': 10, 'target_classes': ['Fuzznum']},
        ]
        
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @batch_extension(registrations)
            def priority_function():
                return "priority"
        
        # 验证不同优先级都已注册
        _, metadata_qrofn = self.registry._functions['priority_func']['qrofn']
        _, metadata_ivfn = self.registry._functions['priority_func']['ivfn']
        
        assert metadata_qrofn.priority == 5
        assert metadata_ivfn.priority == 10
    
    def test_batch_extension_preserves_function(self):
        """测试批量装饰器保持函数属性"""
        registrations = [
            {'name': 'preserve_func', 'mtype': 'qrofn', 'target_classes': ['Fuzznum']},
        ]
        
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @batch_extension(registrations)
            def preserve_function(x: int) -> str:
                """Preserved docstring"""
                return str(x)
        
        # 验证函数属性保持不变
        assert preserve_function.__name__ == 'preserve_function'
        assert preserve_function.__doc__ == "Preserved docstring"
        
        # 验证函数仍然可调用
        result = preserve_function(42)
        assert result == "42"
    
    def test_batch_extension_empty_registrations(self):
        """测试空注册列表的批量装饰器"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @batch_extension([])
            def empty_function():
                return "empty"
        
        # 验证没有注册任何函数
        assert len(self.registry._functions) == 0
        assert len(self.registry._defaults) == 0
        
        # 但函数本身仍然可用
        assert empty_function() == "empty"
    
    def test_batch_extension_with_kwargs(self):
        """测试批量注册带额外参数的配置"""
        registrations = [
            {
                'name': 'kwargs_batch', 
                'mtype': 'qrofn', 
                'target_classes': ['Fuzznum'],
                'description': 'Batch function with kwargs'
            },
        ]
        
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            @batch_extension(registrations)
            def kwargs_batch_function():
                return "kwargs_batch"
        
        _, metadata = self.registry._functions['kwargs_batch']['qrofn']
        assert metadata.description == 'Batch function with kwargs'


class TestDecoratorIntegration:
    """测试装饰器与注册表的集成"""
    
    def test_extension_uses_global_registry(self):
        """测试装饰器使用全局注册表"""
        # 不使用 patch，测试真实的全局注册表
        original_registry = get_registry_extension()
        
        @extension('global_test', mtype='qrofn', target_classes=['Fuzznum'])
        def global_function():
            return "global"
        
        # 验证函数已注册到全局注册表
        assert 'global_test' in original_registry._functions
        assert 'qrofn' in original_registry._functions['global_test']
        
        # 清理
        if 'global_test' in original_registry._functions:
            del original_registry._functions['global_test']
    
    def test_batch_extension_uses_global_registry(self):
        """测试批量装饰器使用全局注册表"""
        original_registry = get_registry_extension()
        
        registrations = [
            {'name': 'global_batch', 'mtype': 'qrofn', 'target_classes': ['Fuzznum']},
        ]
        
        @batch_extension(registrations)
        def global_batch_function():
            return "global_batch"
        
        # 验证函数已注册到全局注册表
        assert 'global_batch' in original_registry._functions
        
        # 清理
        if 'global_batch' in original_registry._functions:
            del original_registry._functions['global_batch']


class TestDecoratorErrorHandling:
    """测试装饰器错误处理"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.registry = ExtensionRegistry()
    
    def test_extension_duplicate_registration_error(self):
        """测试重复注册相同优先级函数的错误处理"""
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            # 第一次注册
            @extension('duplicate_func', mtype='qrofn', priority=5, target_classes=['Fuzznum'])
            def first_function():
                return "first"
            
            # 第二次注册相同配置应该失败
            with pytest.raises(ValueError, match="already exists with higher or equal priority"):
                @extension('duplicate_func', mtype='qrofn', priority=5, target_classes=['Fuzznum'])
                def second_function():
                    return "second"
    
    def test_batch_extension_duplicate_registration_error(self):
        """测试批量装饰器中重复注册的错误处理"""
        registrations = [
            {'name': 'batch_duplicate', 'mtype': 'qrofn', 'priority': 5, 'target_classes': ['Fuzznum']},
            {'name': 'batch_duplicate', 'mtype': 'qrofn', 'priority': 5, 'target_classes': ['Fuzznum']},  # 重复
        ]
        
        with patch('axisfuzzy.extension.decorator.get_registry_extension', return_value=self.registry):
            with pytest.raises(ValueError, match="already exists with higher or equal priority"):
                @batch_extension(registrations)
                def duplicate_batch_function():
                    return "duplicate"