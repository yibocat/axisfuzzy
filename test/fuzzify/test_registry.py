#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:28
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
from fuzzlab.fuzzify.registry import (
    get_fuzzification_registry,
    register_fuzzification_strategy,
    FuzzificationRegistry
)
from fuzzlab.fuzzify.base import FuzzificationStrategy
from fuzzlab.core import Fuzznum, Fuzzarray
from fuzzlab.membership import MembershipFunction
import numpy as np


class MockStrategy(FuzzificationStrategy):
    """用于测试的模拟策略"""

    def __init__(self, q=2, test_param=None, **kwargs):
        super().__init__(q=q, test_param=test_param, **kwargs)
        self.mtype = "test_type"
        self.method = "mock_method"

    def fuzzify_scalar(self, x, mf=None):
        # 简单模拟实现
        from fuzzlab.core import Fuzznum
        return Fuzznum(mtype='qrofn', q=2).create(md=0.5, nmd=0.3)

    def fuzzify_array(self, x, mf=None):
        # 简单模拟实现
        from fuzzlab.core import Fuzzarray, get_fuzznum_registry
        registry = get_fuzznum_registry()
        backend_cls = registry.get_backend('qrofn')
        backend = backend_cls.from_arrays(md=np.array([0.5]), nmd=np.array([0.3]), q=2)
        return Fuzzarray(backend=backend, mtype='qrofn', q=2)


class TestFuzzificationRegistry:

    def test_registry_singleton(self):
        """测试注册表是单例"""
        registry1 = get_fuzzification_registry()
        registry2 = get_fuzzification_registry()
        assert registry1 is registry2
        assert isinstance(registry1, FuzzificationRegistry)

    def test_register_strategy(self):
        """测试策略注册"""
        registry = FuzzificationRegistry()

        # 注册策略
        registry.register('test_mtype', 'test_method', MockStrategy)

        # 验证注册成功
        strategy_cls = registry.get_strategy('test_mtype', 'test_method')
        assert strategy_cls is MockStrategy

    def test_register_with_default(self):
        """测试设置默认策略"""
        registry = FuzzificationRegistry()

        # 注册为默认策略
        registry.register('test_mtype2', 'default_method', MockStrategy, is_default=True)

        # 验证默认方法
        default_method = registry.get_default_method('test_mtype2')
        assert default_method == 'default_method'

        # 验证可以通过默认方法获取
        strategy_cls = registry.get_strategy('test_mtype2')
        assert strategy_cls is MockStrategy

    def test_register_invalid_strategy(self):
        """测试注册无效策略类型"""
        registry = FuzzificationRegistry()

        class NotAStrategy:
            pass

        with pytest.raises(ValueError, match="must be a subclass of FuzzificationStrategy"):
            registry.register('invalid', 'method', NotAStrategy)

    def test_register_duplicate_warning(self, capfd):
        """测试重复注册警告"""
        registry = FuzzificationRegistry()

        # 首次注册
        registry.register('dup_type', 'dup_method', MockStrategy)

        # 再次注册同一策略
        registry.register('dup_type', 'dup_method', MockStrategy)

        # 检查警告信息
        captured = capfd.readouterr()
        assert "already registered" in captured.out

    def test_get_strategy_nonexistent(self):
        """测试获取不存在的策略"""
        registry = FuzzificationRegistry()

        # 不存在的策略
        strategy_cls = registry.get_strategy('nonexistent_type', 'nonexistent_method')
        assert strategy_cls is None

        # 不存在的默认方法
        strategy_cls = registry.get_strategy('nonexistent_type')
        assert strategy_cls is None

    def test_list_strategies(self):
        """测试列出策略"""
        registry = FuzzificationRegistry()

        # 注册几个策略
        registry.register('type1', 'method1', MockStrategy)
        registry.register('type1', 'method2', MockStrategy)
        registry.register('type2', 'method1', MockStrategy)

        # 列出所有策略
        all_strategies = registry.list_strategies()
        assert ('type1', 'method1') in all_strategies
        assert ('type1', 'method2') in all_strategies
        assert ('type2', 'method1') in all_strategies

        # 列出特定类型的策略
        type1_strategies = registry.list_strategies('type1')
        assert ('type1', 'method1') in type1_strategies
        assert ('type1', 'method2') in type1_strategies
        assert ('type2', 'method1') not in type1_strategies

    def test_get_available_mtypes(self):
        """测试获取可用的模糊数类型"""
        registry = FuzzificationRegistry()

        registry.register('qrofn', 'method1', MockStrategy)
        registry.register('ivfn', 'method1', MockStrategy)

        mtypes = registry.get_available_mtypes()
        assert 'qrofn' in mtypes
        assert 'ivfn' in mtypes

    def test_get_available_methods(self):
        """测试获取特定类型的可用方法"""
        registry = FuzzificationRegistry()

        registry.register('qrofn', 'default', MockStrategy)
        registry.register('qrofn', 'expert', MockStrategy)
        registry.register('ivfn', 'method1', MockStrategy)

        qrofn_methods = registry.get_available_methods('qrofn')
        assert 'default' in qrofn_methods
        assert 'expert' in qrofn_methods
        assert 'method1' not in qrofn_methods

    def test_get_registry_info(self):
        """测试获取注册表信息"""
        registry = FuzzificationRegistry()

        registry.register('qrofn', 'default', MockStrategy, is_default=True)
        registry.register('qrofn', 'expert', MockStrategy)

        info = registry.get_registry_info()

        assert 'total_strategies' in info
        assert 'mtypes' in info
        assert 'defaults' in info
        assert info['total_strategies'] >= 2
        assert 'qrofn' in info['mtypes']
        assert info['defaults']['qrofn'] == 'default'


class TestRegisterDecorator:

    def test_decorator_basic(self):
        """测试装饰器基本功能"""
        registry = FuzzificationRegistry()

        @register_fuzzification_strategy('decorator_type', 'decorator_method')
        class DecoratorStrategy(FuzzificationStrategy):
            def fuzzify_scalar(self, x, mf=None):
                pass
            def fuzzify_array(self, x, mf=None):
                pass

        # 注意：装饰器会注册到全局注册表，这里我们检查全局注册表
        global_registry = get_fuzzification_registry()
        strategy_cls = global_registry.get_strategy('decorator_type', 'decorator_method')
        assert strategy_cls is DecoratorStrategy

    def test_decorator_with_default(self):
        """测试装饰器设置默认策略"""

        @register_fuzzification_strategy('decorator_type2', 'default_via_decorator', is_default=True)
        class DefaultDecoratorStrategy(FuzzificationStrategy):
            def fuzzify_scalar(self, x, mf=None):
                pass
            def fuzzify_array(self, x, mf=None):
                pass

        global_registry = get_fuzzification_registry()
        default_method = global_registry.get_default_method('decorator_type2')
        assert default_method == 'default_via_decorator'


class TestBuiltinStrategies:
    """测试内置策略是否正确注册"""

    def test_qrofn_default_registered(self):
        """测试 qrofn 默认策略已注册"""
        registry = get_fuzzification_registry()

        # 检查 qrofn 默认策略存在
        strategy_cls = registry.get_strategy('qrofn', 'default')
        assert strategy_cls is not None

        # 检查是否为默认方法
        default_method = registry.get_default_method('qrofn')
        assert default_method == 'default'

        # 检查可以通过省略 method 获取
        strategy_cls_default = registry.get_strategy('qrofn')
        assert strategy_cls_default is strategy_cls

    def test_qrofn_in_available_types(self):
        """测试 qrofn 在可用类型列表中"""
        registry = get_fuzzification_registry()

        mtypes = registry.get_available_mtypes()
        assert 'qrofn' in mtypes

        methods = registry.get_available_methods('qrofn')
        assert 'default' in methods


if __name__ == '__main__':
    pytest.main([__file__])
