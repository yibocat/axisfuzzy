#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:45
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Tests for the FuzznumRegistry system.

This module tests the central registry for fuzzy number types, focusing on:
- Singleton pattern implementation
- Thread-safe registration and retrieval
- Strategy and backend registration
- Transaction support
- Observer pattern
- Registry introspection
- Decorator functionality
"""

import pytest
import threading
import time
from typing import Any, Dict, Callable
from unittest.mock import Mock, patch

from axisfuzzy.core.registry import (
    FuzznumRegistry,
    get_registry_fuzztype,
    register_strategy,
    register_backend,
    register_fuzztype
)
from axisfuzzy.core.base import FuzznumStrategy
from axisfuzzy.core.backend import FuzzarrayBackend


class MockStrategy(FuzznumStrategy):
    """用于测试的模拟策略类。"""
    mtype = 'mock_strategy'
    
    def __init__(self, **kwargs):
        super().__init__()
        # 添加必要的属性声明
        self.declare_attribute('md', float, lambda x: max(0.0, min(1.0, float(x))))
        self.declare_attribute('nmd', float, lambda x: max(0.0, min(1.0, float(x))))
    
    def _fuzz_constraint(self, **kwargs) -> bool:
        """模拟模糊约束检查。"""
        md = kwargs.get('md', 0.0)
        nmd = kwargs.get('nmd', 0.0)
        return md + nmd <= 1.0


class MockBackend(FuzzarrayBackend):
    """用于测试的模拟后端类。"""
    mtype = 'mock_backend'
    
    def _initialize_arrays(self):
        """初始化数组。"""
        import numpy as np
        self._md = np.zeros(self.shape, dtype=float)
        self._nmd = np.zeros(self.shape, dtype=float)
    
    def get_fuzznum_view(self, index: Any):
        """获取模糊数视图。"""
        from axisfuzzy.core.fuzznums import fuzznum
        return fuzznum(mtype=self.mtype, md=float(self._md[index]), nmd=float(self._nmd[index]))
    
    def set_fuzznum_data(self, index: Any, fuzznum):
        """设置模糊数数据。"""
        self._md[index] = fuzznum.md
        self._nmd[index] = fuzznum.nmd
    
    def copy(self):
        """创建拷贝。"""
        new_backend = MockBackend(self.shape, self.q, **self.kwargs)
        new_backend._md = self._md.copy()
        new_backend._nmd = self._nmd.copy()
        return new_backend
    
    def slice_view(self, key):
        """创建切片视图。"""
        sliced_md = self._md[key]
        sliced_nmd = self._nmd[key]
        new_backend = MockBackend(sliced_md.shape, self.q, **self.kwargs)
        new_backend._md = sliced_md
        new_backend._nmd = sliced_nmd
        return new_backend
    
    @staticmethod
    def from_arrays(*components, **kwargs):
        """从数组创建后端。"""
        if len(components) != 2:
            raise ValueError("MockBackend requires exactly 2 components")
        md_array, nmd_array = components
        backend = MockBackend(md_array.shape, **kwargs)
        backend._md = md_array.copy()
        backend._nmd = nmd_array.copy()
        return backend
    
    def get_component_arrays(self):
        """获取组件数组。"""
        return (self._md, self._nmd)
    
    def _get_element_formatter(self, format_spec: str) -> Callable:
        """获取元素格式化器。"""
        def formatter(index, md_val, nmd_val):
            return f"({md_val:.3f}, {nmd_val:.3f})"
        return formatter
    
    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """格式化单个元素。"""
        md_val = self._md[index]
        nmd_val = self._nmd[index]
        return formatter(index, md_val, nmd_val)


class TestFuzznumRegistrySingleton:
    """测试 FuzznumRegistry 单例模式。"""
    
    def test_singleton_instance(self):
        """测试单例模式确保只有一个实例。"""
        registry1 = get_registry_fuzztype()
        registry2 = get_registry_fuzztype()
        
        assert registry1 is registry2
        assert isinstance(registry1, FuzznumRegistry)
    
    def test_singleton_thread_safety(self):
        """测试单例模式的线程安全性。"""
        instances = []
        
        def get_instance():
            instances.append(get_registry_fuzztype())
        
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 所有实例应该是同一个对象
        assert len(set(id(instance) for instance in instances)) == 1
    
    def test_registry_initialization(self):
        """测试注册表初始化。"""
        registry = get_registry_fuzztype()
        
        assert hasattr(registry, 'strategies')
        assert hasattr(registry, 'backends')
        assert hasattr(registry, '_observers')
        assert isinstance(registry.strategies, dict)
        assert isinstance(registry.backends, dict)


class TestFuzznumRegistryRegistration:
    """测试 FuzznumRegistry 注册功能。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
        # 清理可能存在的测试数据
        if 'mock_strategy' in self.registry.strategies:
            del self.registry.strategies['mock_strategy']
        if 'mock_backend' in self.registry.backends:
            del self.registry.backends['mock_backend']
        if 'test_type' in self.registry.strategies:
            del self.registry.strategies['test_type']
        if 'test_type' in self.registry.backends:
            del self.registry.backends['test_type']
    
    def test_register_strategy(self):
        """测试策略注册。"""
        result = self.registry.register_strategy(MockStrategy)
        
        assert result['mtype'] == 'mock_strategy'
        assert result['component'] == 'strategy'
        assert result['registered_class'] == 'MockStrategy'
        assert result['overwrote_existing'] is False
        assert 'mock_strategy' in self.registry.strategies
        assert self.registry.strategies['mock_strategy'] is MockStrategy
    
    def test_register_backend(self):
        """测试后端注册。"""
        result = self.registry.register_backend(MockBackend)
        
        assert result['mtype'] == 'mock_backend'
        assert result['component'] == 'backend'
        assert result['registered_class'] == 'MockBackend'
        assert result['overwrote_existing'] is False
        assert 'mock_backend' in self.registry.backends
        assert self.registry.backends['mock_backend'] is MockBackend
    
    def test_register_complete_type(self):
        """测试完整类型注册（策略+后端）。"""
        # 创建匹配的策略和后端
        class TestStrategy(MockStrategy):
            mtype = 'test_type'
        
        class TestBackend(MockBackend):
            mtype = 'test_type'
        
        result = self.registry.register(strategy=TestStrategy, backend=TestBackend)
        
        assert result['mtype'] == 'test_type'
        assert len(result['details']) == 2
        assert 'test_type' in self.registry.strategies
        assert 'test_type' in self.registry.backends
    
    def test_register_strategy_only(self):
        """测试仅注册策略。"""
        result = self.registry.register(strategy=MockStrategy)
        
        assert result['mtype'] == 'mock_strategy'
        assert len(result['details']) == 1
        assert result['details'][0]['component'] == 'strategy'
        assert 'mock_strategy' in self.registry.strategies
        assert 'mock_strategy' not in self.registry.backends
    
    def test_register_backend_only(self):
        """测试仅注册后端。"""
        result = self.registry.register(backend=MockBackend)
        
        assert result['mtype'] == 'mock_backend'
        assert len(result['details']) == 1
        assert result['details'][0]['component'] == 'backend'
        assert 'mock_backend' in self.registry.backends
        assert 'mock_backend' not in self.registry.strategies
    
    def test_register_duplicate_strategy(self):
        """测试重复注册策略。"""
        # 首次注册
        self.registry.register_strategy(MockStrategy)
        
        # 重复注册应该成功，overwrote_existing 为 True
        result = self.registry.register_strategy(MockStrategy)
        
        assert result['mtype'] == 'mock_strategy'
        assert result['overwrote_existing'] is True
    
    def test_register_invalid_strategy(self):
        """测试注册无效策略。"""
        class InvalidStrategy:
            pass  # 不继承 FuzznumStrategy
        
        with pytest.raises(TypeError, match="must be a subclass of FuzznumStrategy"):
            self.registry.register_strategy(InvalidStrategy)
    
    def test_register_invalid_backend(self):
        """测试注册无效后端。"""
        class InvalidBackend:
            pass  # 不继承 FuzzarrayBackend
        
        with pytest.raises(TypeError, match="must be a subclass of FuzzarrayBackend"):
            self.registry.register_backend(InvalidBackend)
    
    def test_register_mismatched_mtypes(self):
        """测试不匹配的 mtype。"""
        class StrategyA(MockStrategy):
            mtype = 'type_a'
        
        class BackendB(MockBackend):
            mtype = 'type_b'
        
        with pytest.raises(ValueError, match="mtype mismatch"):
            self.registry.register(strategy=StrategyA, backend=BackendB)


class TestFuzznumRegistryRetrieval:
    """测试 FuzznumRegistry 检索功能。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
        # 注册测试数据
        self.registry.register_strategy(MockStrategy)
        self.registry.register_backend(MockBackend)
    
    def teardown_method(self):
        """每个测试方法后的清理。"""
        # 清理测试数据
        if 'mock_strategy' in self.registry.strategies:
            del self.registry.strategies['mock_strategy']
        if 'mock_backend' in self.registry.backends:
            del self.registry.backends['mock_backend']
    
    def test_get_strategy(self):
        """测试获取策略。"""
        strategy_class = self.registry.get_strategy('mock_strategy')
        assert strategy_class is MockStrategy
    
    def test_get_backend(self):
        """测试获取后端。"""
        backend_class = self.registry.get_backend('mock_backend')
        assert backend_class is MockBackend
    
    def test_get_nonexistent_strategy(self):
        """测试获取不存在的策略。"""
        with pytest.raises(ValueError, match="Strategy for mtype 'nonexistent' not found"):
            self.registry.get_strategy('nonexistent')
    
    def test_get_nonexistent_backend(self):
        """测试获取不存在的后端。"""
        with pytest.raises(ValueError, match="Backend for mtype 'nonexistent' not found"):
            self.registry.get_backend('nonexistent')
    
    def test_get_registered_mtypes(self):
        """测试获取已注册的类型。"""
        mtypes = self.registry.get_registered_mtypes()
        
        assert isinstance(mtypes, dict)
        assert 'mock_strategy' in mtypes
        assert 'mock_backend' in mtypes
        
        # 检查类型信息
        strategy_info = mtypes['mock_strategy']
        assert strategy_info['has_strategy'] is True
        assert strategy_info['has_backend'] is False
        
        backend_info = mtypes['mock_backend']
        assert backend_info['has_strategy'] is False
        assert backend_info['has_backend'] is True


class TestFuzznumRegistryTransaction:
    """测试 FuzznumRegistry 事务功能。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
    
    def test_successful_transaction(self):
        """测试成功的事务。"""
        class TxStrategy(MockStrategy):
            mtype = 'tx_test'
        
        class TxBackend(MockBackend):
            mtype = 'tx_test'
        
        with self.registry.transaction():
            self.registry.register_strategy(TxStrategy)
            self.registry.register_backend(TxBackend)
        
        # 事务成功，应该能找到注册的类型
        assert 'tx_test' in self.registry.strategies
        assert 'tx_test' in self.registry.backends
        
        # 清理
        del self.registry.strategies['tx_test']
        del self.registry.backends['tx_test']
    
    def test_failed_transaction_rollback(self):
        """测试失败事务的回滚。"""
        class TxStrategy(MockStrategy):
            mtype = 'tx_rollback'
        
        initial_strategies = self.registry.strategies.copy()
        initial_backends = self.registry.backends.copy()
        
        try:
            with self.registry.transaction():
                self.registry.register_strategy(TxStrategy)
                # 故意引发异常
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # 事务失败，应该回滚到初始状态
        assert self.registry.strategies == initial_strategies
        assert self.registry.backends == initial_backends
        assert 'tx_rollback' not in self.registry.strategies


class TestFuzznumRegistryObserver:
    """测试 FuzznumRegistry 观察者模式。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
        self.events = []
        
        def observer(event_type: str, event_data: Dict[str, Any]):
            self.events.append((event_type, event_data))
        
        self.observer = observer
        self.registry.add_observer(self.observer)
    
    def teardown_method(self):
        """每个测试方法后的清理。"""
        self.registry.remove_observer(self.observer)
        # 清理可能的测试数据
        if 'observer_test' in self.registry.strategies:
            del self.registry.strategies['observer_test']
    
    def test_observer_registration_event(self):
        """测试观察者接收注册事件。"""
        class ObserverStrategy(MockStrategy):
            mtype = 'observer_test'
        
        self.registry.register_strategy(ObserverStrategy)
        
        # 应该收到注册事件
        assert len(self.events) > 0
        event_type, event_data = self.events[-1]
        assert event_type == 'register_strategy'
        assert event_data['mtype'] == 'observer_test'
    
    def test_observer_removal(self):
        """测试观察者移除。"""
        self.registry.remove_observer(self.observer)
        
        class RemovalStrategy(MockStrategy):
            mtype = 'removal_test'
        
        initial_event_count = len(self.events)
        self.registry.register_strategy(RemovalStrategy)
        
        # 移除观察者后不应该收到新事件
        assert len(self.events) == initial_event_count
        
        # 清理
        del self.registry.strategies['removal_test']


class TestFuzznumRegistryBatch:
    """测试 FuzznumRegistry 批量操作。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
    
    def test_batch_register_success(self):
        """测试成功的批量注册。"""
        class BatchStrategy1(MockStrategy):
            mtype = 'batch1'
        
        class BatchStrategy2(MockStrategy):
            mtype = 'batch2'
        
        registrations = [
            {'strategy': BatchStrategy1},
            {'strategy': BatchStrategy2}
        ]
        
        results = self.registry.batch_register(registrations)
        
        assert len(results) == 2
        assert 'batch1' in results
        assert 'batch2' in results
        assert results['batch1']['mtype'] == 'batch1'
        assert results['batch2']['mtype'] == 'batch2'
        assert len(results['batch1']['details']) == 1  # 只注册了策略
        assert len(results['batch2']['details']) == 1  # 只注册了策略
        
        # 清理
        del self.registry.strategies['batch1']
        del self.registry.strategies['batch2']
    
    def test_batch_register_partial_failure(self):
        """测试部分失败的批量注册。"""
        class BatchStrategy(MockStrategy):
            mtype = 'batch_partial'
        
        class InvalidStrategy:
            pass  # 无效策略
        
        registrations = [
            {'strategy': BatchStrategy},
            {'strategy': InvalidStrategy}  # 这个会失败
        ]
        
        # 批量注册遇到错误时会抛出异常并回滚
        with pytest.raises(TypeError, match="Strategy must be a subclass of FuzznumStrategy"):
            self.registry.batch_register(registrations)
        
        # 由于事务回滚，第一个策略也不应该被注册
        assert 'batch_partial' not in self.registry.strategies


class TestFuzznumRegistryUnregistration:
    """测试 FuzznumRegistry 注销功能。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
        
        class UnregStrategy(MockStrategy):
            mtype = 'unreg_test'
        
        class UnregBackend(MockBackend):
            mtype = 'unreg_test'
        
        self.registry.register(strategy=UnregStrategy, backend=UnregBackend)
    
    def test_unregister_complete(self):
        """测试完全注销。"""
        result = self.registry.unregister('unreg_test')
        
        assert result['mtype'] == 'unreg_test'
        assert result['strategy_removed'] is True
        assert result['backend_removed'] is True
        assert result['was_complete'] is True
        assert 'unreg_test' not in self.registry.strategies
        assert 'unreg_test' not in self.registry.backends
    
    def test_unregister_strategy_only(self):
        """测试仅注销策略。"""
        result = self.registry.unregister('unreg_test', remove_strategy=True, remove_backend=False)
        
        assert result['mtype'] == 'unreg_test'
        assert result['strategy_removed'] is True
        assert result['backend_removed'] is False
        assert result['was_complete'] is True
        assert 'unreg_test' not in self.registry.strategies
        assert 'unreg_test' in self.registry.backends
        
        # 清理剩余的后端
        del self.registry.backends['unreg_test']
    
    def test_unregister_nonexistent(self):
        """测试注销不存在的类型。"""
        result = self.registry.unregister('nonexistent')
        
        assert result['mtype'] == 'nonexistent'
        assert result['strategy_removed'] is False
        assert result['backend_removed'] is False
        assert result['was_complete'] is False


class TestFuzznumRegistryIntrospection:
    """测试 FuzznumRegistry 内省功能。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
    
    def test_get_statistics(self):
        """测试获取统计信息。"""
        stats = self.registry.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_strategies' in stats
        assert 'total_backends' in stats
        assert 'complete_types' in stats
        assert 'registration_stats' in stats
        assert 'observer_count' in stats
        assert isinstance(stats['total_strategies'], int)
        assert isinstance(stats['total_backends'], int)
    
    def test_get_health_status(self):
        """测试获取健康状态。"""
        health = self.registry.get_health_status()
        
        assert isinstance(health, dict)
        assert 'is_healthy' in health
        assert 'complete_types' in health
        assert 'incomplete_types' in health
        assert 'missing_strategies' in health
        assert 'missing_backends' in health
        assert 'error_rate' in health
        assert isinstance(health['is_healthy'], bool)
        assert isinstance(health['complete_types'], list)
        assert isinstance(health['incomplete_types'], list)


class TestFuzznumRegistryDecorators:
    """测试 FuzznumRegistry 装饰器功能。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
    
    def teardown_method(self):
        """每个测试方法后的清理。"""
        # 清理可能的测试数据
        for mtype in ['decorated_strategy', 'decorated_backend', 'decorated_complete']:
            if mtype in self.registry.strategies:
                del self.registry.strategies[mtype]
            if mtype in self.registry.backends:
                del self.registry.backends[mtype]
    
    def test_register_strategy_decorator(self):
        """测试策略注册装饰器。"""
        @register_strategy
        class DecoratedStrategy(MockStrategy):
            mtype = 'decorated_strategy'
        
        assert 'decorated_strategy' in self.registry.strategies
        assert self.registry.strategies['decorated_strategy'] is DecoratedStrategy
    
    def test_register_backend_decorator(self):
        """测试后端注册装饰器。"""
        @register_backend
        class DecoratedBackend(MockBackend):
            mtype = 'decorated_backend'
        
        assert 'decorated_backend' in self.registry.backends
        assert self.registry.backends['decorated_backend'] is DecoratedBackend
    
    def test_register_fuzztype_function(self):
        """测试 register_fuzztype 便利函数。"""
        class CompleteStrategy(MockStrategy):
            mtype = 'decorated_complete'
        
        class CompleteBackend(MockBackend):
            mtype = 'decorated_complete'
        
        result = register_fuzztype(strategy=CompleteStrategy, backend=CompleteBackend)
        
        assert result['mtype'] == 'decorated_complete'
        assert len(result['details']) == 2  # 策略和后端都注册了
        assert 'decorated_complete' in self.registry.strategies
        assert 'decorated_complete' in self.registry.backends


class TestFuzznumRegistryThreadSafety:
    """测试 FuzznumRegistry 线程安全性。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.registry = get_registry_fuzztype()
        self.results = []
        self.errors = []
    
    def test_concurrent_registration(self):
        """测试并发注册。"""
        def register_strategy(index):
            try:
                class ConcurrentStrategy(MockStrategy):
                    mtype = f'concurrent_{index}'
                
                result = self.registry.register_strategy(ConcurrentStrategy)
                self.results.append((index, result))
            except Exception as e:
                self.errors.append((index, e))
        
        threads = [threading.Thread(target=register_strategy, args=(i,)) for i in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 所有注册应该成功
        assert len(self.errors) == 0
        assert len(self.results) == 10
        
        # 清理
        for i in range(10):
            mtype = f'concurrent_{i}'
            if mtype in self.registry.strategies:
                del self.registry.strategies[mtype]
    
    def test_concurrent_retrieval(self):
        """测试并发检索。"""
        # 先注册一个策略
        class RetrievalStrategy(MockStrategy):
            mtype = 'retrieval_test'
        
        self.registry.register_strategy(RetrievalStrategy)
        
        def get_strategy(index):
            try:
                strategy = self.registry.get_strategy('retrieval_test')
                self.results.append((index, strategy))
            except Exception as e:
                self.errors.append((index, e))
        
        threads = [threading.Thread(target=get_strategy, args=(i,)) for i in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 所有检索应该成功
        assert len(self.errors) == 0
        assert len(self.results) == 10
        assert all(result[1] is RetrievalStrategy for result in self.results)
        
        # 清理
        del self.registry.strategies['retrieval_test']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])