#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:45
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Tests for the operation system and dispatcher.

This module tests the core operation framework and dispatcher, focusing on:
- OperationMixin abstract interface
- OperationScheduler registration and dispatch
- Operation dispatcher routing logic
- Performance monitoring
- T-norm configuration
- Error handling and validation
"""

import pytest
import threading
import time
from typing import Dict, Any, List, Union, Optional
from unittest.mock import Mock, patch, MagicMock

from axisfuzzy.core.operation import (
    OperationMixin,
    OperationScheduler,
    get_registry_operation,
    register_operation
)
from axisfuzzy.core.dispatcher import operate
from axisfuzzy.core.triangular import OperationTNorm


class MockStrategy:
    """用于测试的模拟策略类。"""
    
    def __init__(self, mtype='mock', md=0.5, nmd=0.3, q=2):
        self.mtype = mtype
        self.md = md
        self.nmd = nmd
        self.q = q
    
    def execute_operation(self, op_name: str, other):
        """模拟策略的操作执行。"""
        if op_name == 'add' and hasattr(other, 'md'):
            return {'md': self.md + other.md, 'nmd': self.nmd + other.nmd, 'q': self.q}
        elif op_name == 'tim' and isinstance(other, (int, float)):
            return {'md': self.md * other, 'nmd': self.nmd * other, 'q': self.q}
        elif op_name == 'complement':
            return {'md': self.nmd, 'nmd': self.md, 'q': self.q}
        elif op_name == 'gt' and hasattr(other, 'md'):
            return {'value': self.md > other.md}
        return {'md': self.md, 'nmd': self.nmd, 'q': self.q}


class MockFuzznum:
    """用于测试的模拟 Fuzznum 类。"""
    
    def __init__(self, mtype='mock', md=0.5, nmd=0.3, q=2):
        self.mtype = mtype
        self.md = md
        self.nmd = nmd
        self.q = q
        self._strategy = MockStrategy(mtype, md, nmd, q)
    
    def get_strategy_instance(self):
        return self._strategy
    
    def create(self, **kwargs):
        return MockFuzznum(
            mtype=kwargs.get('mtype', self.mtype),
            md=kwargs.get('md', self.md),
            nmd=kwargs.get('nmd', self.nmd),
            q=kwargs.get('q', self.q)
        )


class MockFuzzarray:
    """用于测试的模拟 Fuzzarray 类。"""
    
    def __init__(self, mtype='mock', shape=(2,), q=2):
        self.mtype = mtype
        self.shape = shape
        self.q = q
    
    def execute_vectorized_op(self, op_name: str, other):
        """模拟向量化操作。"""
        if op_name == 'add':
            return MockFuzzarray(self.mtype, self.shape, self.q)
        elif op_name == 'tim':
            return MockFuzzarray(self.mtype, self.shape, self.q)
        return self


class MockOperation(OperationMixin):
    """用于测试的模拟操作类。"""
    
    def get_operation_name(self) -> str:
        return 'mock_add'
    
    def get_supported_mtypes(self) -> List[str]:
        return ['mock', 'test']
    
    def _execute_binary_op_impl(self, s1, s2, tnorm: OperationTNorm) -> Dict[str, Any]:
        """模拟二元操作实现。"""
        return {
            'md': s1.md + s2.md,
            'nmd': s1.nmd + s2.nmd,
            'q': s1.q
        }
    
    def _execute_unary_op_operand_impl(self, strategy, operand: Union[int, float], tnorm: OperationTNorm) -> Dict[str, Any]:
        """模拟一元操作实现。"""
        return {
            'md': strategy.md * operand,
            'nmd': strategy.nmd * operand,
            'q': strategy.q
        }
    
    def _execute_comparison_op_impl(self, s1, s2, tnorm: OperationTNorm) -> Dict[str, bool]:
        """模拟比较操作实现。"""
        return {'value': s1.md > s2.md}


class TestOperationMixin:
    """测试 OperationMixin 抽象基类。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.operation = MockOperation()
        self.strategy1 = MockStrategy('mock', 0.6, 0.2, 2)
        self.strategy2 = MockStrategy('mock', 0.4, 0.3, 2)
        self.tnorm = OperationTNorm('algebraic')
    
    def test_abstract_methods(self):
        """测试抽象方法的实现。"""
        assert self.operation.get_operation_name() == 'mock_add'
        assert self.operation.get_supported_mtypes() == ['mock', 'test']
    
    def test_supports_method(self):
        """测试 supports 方法。"""
        assert self.operation.supports('mock') is True
        assert self.operation.supports('test') is True
        assert self.operation.supports('unsupported') is False
    
    def test_preprocess_binary_operands(self):
        """测试二元操作数预处理。"""
        # 正常情况不应该抛出异常
        try:
            self.operation._preprocess_binary_operands(self.strategy1, self.strategy2, self.tnorm)
        except Exception as e:
            pytest.fail(f"Preprocessing should not raise exception: {e}")
    
    def test_preprocess_unary_operand(self):
        """测试一元操作数预处理。"""
        try:
            self.operation._preprocess_unary_operand(self.strategy1, 2.0, self.tnorm)
        except Exception as e:
            pytest.fail(f"Preprocessing should not raise exception: {e}")
    
    def test_execute_binary_op(self):
        """测试二元操作执行。"""
        result = self.operation.execute_binary_op(self.strategy1, self.strategy2, self.tnorm)
        
        assert isinstance(result, dict)
        assert 'md' in result
        assert 'nmd' in result
        assert 'q' in result
        assert result['md'] == 1.0  # 0.6 + 0.4
        assert result['nmd'] == 0.5  # 0.2 + 0.3
    
    def test_execute_unary_op_operand(self):
        """测试带操作数的一元操作执行。"""
        result = self.operation.execute_unary_op_operand(self.strategy1, 2.0, self.tnorm)
        
        assert isinstance(result, dict)
        assert result['md'] == 1.2  # 0.6 * 2.0
        assert result['nmd'] == 0.4  # 0.2 * 2.0
    
    def test_execute_comparison_op(self):
        """测试比较操作执行。"""
        result = self.operation.execute_comparison_op(self.strategy1, self.strategy2, self.tnorm)
        
        assert isinstance(result, dict)
        assert 'value' in result
        assert result['value'] is True  # 0.6 > 0.4
    
    def test_not_implemented_methods(self):
        """测试未实现的方法抛出 NotImplementedError。"""
        with pytest.raises(NotImplementedError):
            self.operation._execute_unary_op_pure_impl(self.strategy1, self.tnorm)
        
        with pytest.raises(NotImplementedError):
            self.operation._execute_fuzzarray_op_impl(None, None, self.tnorm)


class TestOperationScheduler:
    """测试 OperationScheduler 调度器。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.scheduler = OperationScheduler()
        self.operation = MockOperation()
    
    def test_scheduler_initialization(self):
        """测试调度器初始化。"""
        assert hasattr(self.scheduler, '_operations')
        assert hasattr(self.scheduler, '_performance_stats')
        assert hasattr(self.scheduler, '_stats_lock')
        assert isinstance(self.scheduler._operations, dict)
    
    def test_register_operation(self):
        """测试操作注册。"""
        self.scheduler.register(self.operation)
        
        # 检查是否正确注册到所有支持的 mtype
        op_name = self.operation.get_operation_name()
        assert op_name in self.scheduler._operations
        for mtype in self.operation.get_supported_mtypes():
            assert mtype in self.scheduler._operations[op_name]
            assert self.scheduler._operations[op_name][mtype] is self.operation
    
    def test_get_operation(self):
        """测试获取操作。"""
        self.scheduler.register(self.operation)
        
        # 获取已注册的操作
        retrieved_op = self.scheduler.get_operation('mock_add', 'mock')
        assert retrieved_op is self.operation
        
        # 获取不存在的操作
        nonexistent_op = self.scheduler.get_operation('nonexistent', 'mock')
        assert nonexistent_op is None
    
    def test_get_available_ops(self):
        """测试获取可用操作列表。"""
        self.scheduler.register(self.operation)
        
        available_ops = self.scheduler.get_available_ops('mock')
        assert 'mock_add' in available_ops
        
        # 不存在的 mtype
        empty_ops = self.scheduler.get_available_ops('nonexistent')
        assert empty_ops == []
    
    def test_t_norm_configuration(self):
        """测试 T-范数配置。"""
        # 设置 T-范数
        self.scheduler.set_t_norm('algebraic')
        
        # 获取默认配置
        t_norm_type, params = self.scheduler.get_default_t_norm_config()
        assert isinstance(t_norm_type, str)
        assert isinstance(params, dict)
    
    def test_performance_stats(self):
        """测试性能统计。"""
        # 记录一些操作时间
        self.scheduler._record_operation_time('add', 'binary', 0.001)
        self.scheduler._record_operation_time('add', 'binary', 0.002)
        
        # 获取统计信息
        stats = self.scheduler.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'total_operations' in stats
        assert 'average_time_per_total_operation(us)' in stats
        
        # 重置统计
        self.scheduler.reset_performance_stats()
        reset_stats = self.scheduler.get_performance_stats()
        assert reset_stats['total_operations'] == 0
    
    def test_thread_safety(self):
        """测试线程安全性。"""
        results = []
        errors = []
        
        def register_operation(index):
            try:
                class ThreadOperation(OperationMixin):
                    def get_operation_name(self): return f'thread_op_{index}'
                    def get_supported_mtypes(self): return ['thread_test']
                    def _execute_binary_op_impl(self, s1, s2, tnorm): return {}
                
                op = ThreadOperation()
                self.scheduler.register(op)
                results.append(index)
            except Exception as e:
                errors.append((index, e))
        
        threads = [threading.Thread(target=register_operation, args=(i,)) for i in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 所有注册应该成功
        assert len(errors) == 0
        assert len(results) == 5


class TestOperationRegistry:
    """测试操作注册表功能。"""
    
    def test_get_registry_operation(self):
        """测试获取全局操作注册表。"""
        registry1 = get_registry_operation()
        registry2 = get_registry_operation()
        
        # 应该是同一个实例（单例）
        assert registry1 is registry2
        assert isinstance(registry1, OperationScheduler)
    
    def test_register_operation_decorator(self):
        """测试操作注册装饰器。"""
        @register_operation
        class DecoratedOperation(OperationMixin):
            def get_operation_name(self): return 'decorated_op'
            def get_supported_mtypes(self): return ['decorated']
            def _execute_binary_op_impl(self, s1, s2, tnorm): return {}
        
        # 检查是否自动注册
        registry = get_registry_operation()
        retrieved_op = registry.get_operation('decorated_op', 'decorated')
        assert retrieved_op is not None
        assert isinstance(retrieved_op, DecoratedOperation)
    
    def test_register_operation_decorator_eager_false(self):
        """测试非立即注册的装饰器。"""
        @register_operation(eager=False)
        class LazyOperation(OperationMixin):
            def get_operation_name(self): return 'lazy_op'
            def get_supported_mtypes(self): return ['lazy']
            def _execute_binary_op_impl(self, s1, s2, tnorm): return {}
        
        # 应该返回类本身，而不是实例
        assert LazyOperation is not None
        
        # 手动注册
        registry = get_registry_operation()
        registry.register(LazyOperation())
        
        retrieved_op = registry.get_operation('lazy_op', 'lazy')
        assert retrieved_op is not None


class TestDispatcher:
    """测试操作调度器。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 模拟导入的类
        self.mock_fuzznum = MockFuzznum
        self.mock_fuzzarray = MockFuzzarray
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_fuzznum_fuzznum_operation(self):
        """测试 Fuzznum 与 Fuzznum 的操作。"""
        fn1 = MockFuzznum('mock', 0.6, 0.2, 2)
        fn2 = MockFuzznum('mock', 0.4, 0.3, 2)
        
        result = operate('add', fn1, fn2)
        
        assert isinstance(result, MockFuzznum)
        assert result.md == 1.0  # 0.6 + 0.4
        assert result.nmd == 0.5  # 0.2 + 0.3
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_fuzznum_scalar_operation(self):
        """测试 Fuzznum 与标量的操作。"""
        fn = MockFuzznum('mock', 0.5, 0.3, 2)
        
        result = operate('mul', fn, 2.0)
        
        assert isinstance(result, MockFuzznum)
        assert result.md == 1.0  # 0.5 * 2.0
        assert result.nmd == 0.6  # 0.3 * 2.0
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_fuzznum_comparison_operation(self):
        """测试 Fuzznum 比较操作。"""
        fn1 = MockFuzznum('mock', 0.6, 0.2, 2)
        fn2 = MockFuzznum('mock', 0.4, 0.3, 2)
        
        result = operate('gt', fn1, fn2)
        
        assert isinstance(result, bool)
        assert result is True
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_fuzzarray_fuzzarray_operation(self):
        """测试 Fuzzarray 与 Fuzzarray 的操作。"""
        fa1 = MockFuzzarray('mock', (2, 2), 2)
        fa2 = MockFuzzarray('mock', (2, 2), 2)
        
        result = operate('add', fa1, fa2)
        
        assert isinstance(result, MockFuzzarray)
        assert result.shape == (2, 2)
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_fuzzarray_scalar_operation(self):
        """测试 Fuzzarray 与标量的操作。"""
        fa = MockFuzzarray('mock', (3,), 2)
        
        result = operate('mul', fa, 2.0)
        
        assert isinstance(result, MockFuzzarray)
        assert result.shape == (3,)
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_scalar_fuzznum_commutative_operation(self):
        """测试标量与 Fuzznum 的可交换操作。"""
        fn = MockFuzznum('mock', 0.5, 0.3, 2)
        
        result = operate('add', 2.0, fn)
        
        assert isinstance(result, MockFuzznum)
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_unary_operation(self):
        """测试一元操作。"""
        fn = MockFuzznum('mock', 0.5, 0.3, 2)
        
        result = operate('complement', fn, None)
        
        assert isinstance(result, MockFuzznum)
        assert result.md == 0.3  # 原来的 nmd
        assert result.nmd == 0.5  # 原来的 md
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_unsupported_operation(self):
        """测试不支持的操作类型。"""
        with pytest.raises(TypeError, match="Unsupported operand types"):
            operate('add', "string", 123)
    
    @patch('axisfuzzy.core.fuzznums.Fuzznum', MockFuzznum)
    @patch('axisfuzzy.core.fuzzarray.Fuzzarray', MockFuzzarray)
    def test_division_operation_mapping(self):
        """测试除法操作映射到乘法。"""
        fn = MockFuzznum('mock', 0.4, 0.2, 2)
        
        result = operate('div', fn, 2.0)
        
        assert isinstance(result, MockFuzznum)
        # 除法应该转换为乘以倒数
        assert result.md == 0.2  # 0.4 * 0.5
        assert result.nmd == 0.1  # 0.2 * 0.5


class TestOperationIntegration:
    """测试操作系统集成。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_operation()
    
    def test_operation_registration_and_dispatch(self):
        """测试操作注册和调度的完整流程。"""
        # 创建并注册操作
        class IntegrationOperation(OperationMixin):
            def get_operation_name(self): return 'integration_test'
            def get_supported_mtypes(self): return ['integration']
            def _execute_binary_op_impl(self, s1, s2, tnorm): 
                return {'result': 'success'}
        
        operation = IntegrationOperation()
        self.registry.register(operation)
        
        # 验证注册成功
        retrieved_op = self.registry.get_operation('integration_test', 'integration')
        assert retrieved_op is operation
        
        # 验证可用操作列表
        available_ops = self.registry.get_available_ops('integration')
        assert 'integration_test' in available_ops
    
    def test_performance_monitoring(self):
        """测试性能监控功能。"""
        # 创建操作并执行
        class PerformanceOperation(OperationMixin):
            def get_operation_name(self): return 'perf_test'
            def get_supported_mtypes(self): return ['perf']
            def _execute_binary_op_impl(self, s1, s2, tnorm):
                time.sleep(0.001)  # 模拟计算时间
                return {'result': 'done'}
        
        operation = PerformanceOperation()
        strategy1 = MockStrategy('perf')
        strategy2 = MockStrategy('perf')
        tnorm = OperationTNorm('algebraic')
        
        # 执行操作（这会记录性能统计）
        result = operation.execute_binary_op(strategy1, strategy2, tnorm)
        
        # 检查性能统计
        stats = self.registry.get_performance_stats()
        assert stats['total_operations'] > 0
    
    def test_error_handling_in_operations(self):
        """测试操作中的错误处理。"""
        class ErrorOperation(OperationMixin):
            def get_operation_name(self): return 'error_test'
            def get_supported_mtypes(self): return ['error']
            def _execute_binary_op_impl(self, s1, s2, tnorm):
                raise ValueError("Test error")
        
        operation = ErrorOperation()
        strategy1 = MockStrategy('error')
        strategy2 = MockStrategy('error')
        tnorm = OperationTNorm('algebraic')
        
        # 操作应该传播异常
        with pytest.raises(ValueError, match="Test error"):
            operation.execute_binary_op(strategy1, strategy2, tnorm)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])