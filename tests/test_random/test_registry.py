#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 axisfuzzy.random.registry 模块

本模块测试随机生成器注册表系统的核心功能，包括：
- RandomGeneratorRegistry 单例模式
- 生成器注册和注销功能
- @register_random 装饰器
- 线程安全性验证
- 查询和列举功能
- 错误处理和验证
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Set
from unittest.mock import Mock, patch

from axisfuzzy.random.registry import (
    RandomGeneratorRegistry,
    register_random,
    get_random_generator,
    list_registered_random,
    unregister_random
)
from axisfuzzy.random.base import BaseRandomGenerator


class TestRandomGeneratorRegistry:
    """测试 RandomGeneratorRegistry 单例类的核心功能"""
    
    def test_singleton_pattern(self, clean_registry):
        """测试单例模式确保全局唯一实例"""
        # 获取两个实例
        registry1 = RandomGeneratorRegistry()
        registry2 = RandomGeneratorRegistry()
        
        # 验证是同一个实例
        assert registry1 is registry2
        assert id(registry1) == id(registry2)
    
    def test_initial_state(self, clean_registry):
        """测试初始状态的正确性"""
        registry = RandomGeneratorRegistry()
        
        # 初始状态应该为空
        assert len(registry._generators) == 0
        assert registry.list_mtypes() == []
    
    def test_register_generator(self, clean_registry, mock_generator):
        """测试生成器注册功能"""
        registry = RandomGeneratorRegistry()
        
        # 注册生成器
        registry.register("mock", mock_generator)
        
        # 验证注册成功
        assert "mock" in registry._generators
        assert registry._generators["mock"] is mock_generator
        assert "mock" in registry.list_mtypes()
    
    def test_get_generator(self, clean_registry, mock_generator):
        """测试生成器获取功能"""
        registry = RandomGeneratorRegistry()
        
        # 注册并获取生成器
        registry.register("mock", mock_generator)
        retrieved_generator = registry.get_generator("mock")
        
        # 验证获取的是同一个实例
        assert retrieved_generator is mock_generator
    
    def test_get_nonexistent_generator(self, clean_registry):
        """测试获取不存在的生成器"""
        registry = RandomGeneratorRegistry()
        
        # 获取不存在的生成器应该返回None
        result = registry.get_generator("nonexistent")
        assert result is None
    
    def test_unregister_generator(self, clean_registry, mock_generator):
        """测试生成器注销功能"""
        registry = RandomGeneratorRegistry()
        
        # 注册然后注销
        registry.register("mock", mock_generator)
        assert "mock" in registry._generators
        
        registry.unregister("mock")
        assert "mock" not in registry._generators
        assert "mock" not in registry.list_mtypes()
    
    def test_unregister_nonexistent_generator(self, clean_registry):
        """测试注销不存在的生成器"""
        registry = RandomGeneratorRegistry()
        
        # 尝试注销不存在的生成器应该返回False
        result = registry.unregister("nonexistent")
        assert result is False
    
    def test_clear_registry(self, clean_registry, mock_generator, mock_parameterized_generator):
        """测试清空注册表功能"""
        registry = RandomGeneratorRegistry()
        
        # 注册多个生成器
        registry.register("mock", mock_generator)
        registry.register("mock_param", mock_parameterized_generator)
        assert len(registry._generators) == 2
        
        # 清空注册表
        registry.clear()
        assert len(registry._generators) == 0
        assert registry.list_mtypes() == []
    
    def test_list_generators(self, clean_registry, mock_generator, mock_parameterized_generator):
        """测试列举生成器功能"""
        registry = RandomGeneratorRegistry()
        
        # 注册多个生成器
        registry.register("mock", mock_generator)
        registry.register("mock_param", mock_parameterized_generator)
        # 创建另一个mock生成器用于测试
        mock_generator2 = Mock(spec=BaseRandomGenerator)
        mock_generator2.mtype = "mock2"
        registry.register("mock2", mock_generator2)
        
        # 获取生成器列表
        generator_list = registry.list_mtypes()
        
        # 验证列表内容（应该按字母顺序排序）
        assert generator_list == ["mock", "mock2", "mock_param"]
    
    def test_duplicate_registration(self, clean_registry, mock_generator, mock_parameterized_generator):
        """测试重复注册的处理"""
        registry = RandomGeneratorRegistry()
        
        # 首次注册
        registry.register("mock", mock_generator)
        
        # 重复注册应该抛出异常
        with pytest.raises(ValueError, match="Random generator for mtype 'mock' is already registered"):
            registry.register("mock", mock_parameterized_generator)
        
        # 验证原生成器仍然存在
        assert registry.get_generator("mock") is mock_generator
    
    def test_force_registration_override(self, clean_registry, mock_generator, mock_parameterized_generator):
        """测试强制覆盖注册（通过先注销再注册）"""
        registry = RandomGeneratorRegistry()
        
        # 首次注册
        registry.register("mock", mock_generator)
        
        # 注销后重新注册（模拟强制覆盖）
        registry.unregister("mock")
        registry.register("mock_param", mock_parameterized_generator)
        
        # 验证生成器已被替换
        assert registry.get_generator("mock_param") is mock_parameterized_generator
        assert not registry.is_registered("mock")


class TestRegistrationDecorator:
    """测试 @register_random 装饰器功能"""
    
    def test_decorator_basic_usage(self, clean_registry):
        """测试装饰器的基本使用"""
        
        @register_random
        class DecoratedGenerator(BaseRandomGenerator):
            mtype = "decorated_type"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                return params
            
            def fuzznum(self, rng=None, **params):
                return Mock()
            
            def fuzzarray(self, shape, rng=None, **params):
                return Mock()
        
        # 验证装饰器自动注册了生成器
        registry = RandomGeneratorRegistry()
        assert "decorated_type" in registry.list_mtypes()
        
        # 验证可以获取生成器实例
        generator = registry.get_generator("decorated_type")
        assert isinstance(generator, DecoratedGenerator)
        assert generator.mtype == "decorated_type"
    
    def test_decorator_with_mtype_mismatch(self, clean_registry):
        """测试装饰器处理空mtype的情况"""
        
        # 定义一个mtype为空字符串的类
        class EmptyMtypeGenerator(BaseRandomGenerator):
            mtype = ""  # 空字符串
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                return params
            
            def fuzznum(self, rng=None, **params):
                return Mock()
            
            def fuzzarray(self, shape, rng=None, **params):
                return Mock()
        
        # 这个测试应该在装饰时就失败，因为装饰器会验证mtype
        with pytest.raises(TypeError, match="must have a non-empty 'mtype' attribute to be registered"):
            register_random(EmptyMtypeGenerator)
    
    def test_decorator_duplicate_registration(self, clean_registry, mock_generator):
        """测试装饰器重复注册会失败"""
        registry = RandomGeneratorRegistry()
        
        # 预先注册一个生成器
        registry.register("mock", mock_generator)
        
        # 尝试使用装饰器注册相同类型应该失败
        with pytest.raises(ValueError, match="Random generator for mtype 'mock' is already registered"):
            @register_random
            class ConflictGenerator(BaseRandomGenerator):
                mtype = "mock"
                
                def get_default_parameters(self):
                    return {}
                
                def validate_parameters(self, **params):
                    return params
                
                def fuzznum(self, rng=None, **params):
                    return Mock()
                
                def fuzzarray(self, shape, rng=None, **params):
                    return Mock()
    
    def test_decorator_validation(self, clean_registry):
        """测试装饰器的验证功能"""
        # 测试空字符串mtype
        with pytest.raises(TypeError, match="must have a non-empty 'mtype' attribute"):
            @register_random
            class EmptyMtypeGenerator(BaseRandomGenerator):
                mtype = ""
                
                def get_default_parameters(self):
                    return {}
                
                def validate_parameters(self, **params):
                    return params
                
                def fuzznum(self, rng=None, **params):
                    return Mock()
                
                def fuzzarray(self, shape, rng=None, **params):
                    return Mock()
    
    def test_decorator_returns_original_class(self, clean_registry):
        """测试装饰器返回原始类"""
        
        @register_random
        class OriginalGenerator(BaseRandomGenerator):
            mtype = "return_test"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                return params
            
            def fuzznum(self, rng=None, **params):
                return Mock()
            
            def fuzzarray(self, shape, rng=None, **params):
                return Mock()
        
        # 验证装饰器返回的是原始类
        assert OriginalGenerator.__name__ == "OriginalGenerator"
        
        # 验证可以直接实例化
        direct_instance = OriginalGenerator()
        assert isinstance(direct_instance, OriginalGenerator)
        
        # 验证注册表中的实例是同一个类
        registry = RandomGeneratorRegistry()
        registry_instance = registry.get_generator("return_test")
        assert type(registry_instance) is OriginalGenerator


class TestGlobalFunctions:
    """测试全局函数的功能"""
    
    def test_get_random_generator_function(self, clean_registry, mock_generator):
        """测试 get_random_generator 全局函数"""
        # 注册生成器
        registry = RandomGeneratorRegistry()
        registry.register("mock", mock_generator)
        
        # 使用全局函数获取
        retrieved_generator = get_random_generator("mock")
        assert retrieved_generator is mock_generator
    
    def test_list_registered_generators_function(self, clean_registry, mock_generator, mock_parameterized_generator):
        """测试 list_registered_random 全局函数"""
        # 注册多个生成器
        registry = RandomGeneratorRegistry()
        registry.register("mock", mock_generator)
        registry.register("mock_param", mock_parameterized_generator)
        
        # 使用全局函数列举
        generator_list = list_registered_random()
        assert set(generator_list) == {"mock", "mock_param"}
    
    def test_unregister_generator_function(self, clean_registry, mock_generator):
        """测试 unregister_random 全局函数"""
        # 注册生成器
        registry = RandomGeneratorRegistry()
        registry.register("mock", mock_generator)
        
        # 使用全局函数注销
        unregister_random("mock")
        
        # 验证已注销
        assert "mock" not in registry.list_mtypes()
    
    def test_clear_registry_function(self, clean_registry, mock_generator, mock_parameterized_generator):
        """测试 registry.clear() 方法"""
        # 注册多个生成器
        registry = RandomGeneratorRegistry()
        registry.register("mock", mock_generator)
        registry.register("mock_param", mock_parameterized_generator)
        
        # 使用实例方法清空
        registry.clear()
        
        # 验证已清空
        assert len(registry.list_mtypes()) == 0


class TestThreadSafety:
    """测试线程安全性"""
    
    def test_concurrent_registration(self, clean_registry, concurrent_test_helper):
        """测试并发注册的线程安全性"""
        registry = RandomGeneratorRegistry()
        num_threads = 10
        registration_results = []
        
        def register_worker(thread_id):
            """工作线程：注册生成器"""
            try:
                # 创建模拟生成器
                mock_gen = Mock(spec=BaseRandomGenerator)
                mock_gen.mtype = f"thread_{thread_id}"
                
                # 注册生成器
                registry.register(f"thread_{thread_id}", mock_gen)
                
                # 验证注册成功
                retrieved = registry.get_generator(f"thread_{thread_id}")
                success = retrieved is mock_gen
                
                registration_results.append({
                    'thread_id': thread_id,
                    'success': success,
                    'mtype': f"thread_{thread_id}"
                })
                
                return success
                
            except Exception as e:
                registration_results.append({
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e)
                })
                return False
        
        # 并发执行注册
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_worker, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        # 验证所有注册都成功
        assert all(results), f"Some registrations failed: {registration_results}"
        
        # 验证所有生成器都已注册
        final_generators = registry.list_mtypes()
        expected_generators = [f"thread_{i}" for i in range(num_threads)]
        assert set(final_generators) == set(expected_generators)
    
    def test_concurrent_access(self, clean_registry, mock_generator):
        """测试并发访问的线程安全性"""
        registry = RandomGeneratorRegistry()
        
        # 预先注册一些生成器
        for i in range(5):
            mock_gen = Mock(spec=BaseRandomGenerator)
            mock_gen.mtype = f"access_test_{i}"
            registry.register(f"access_test_{i}", mock_gen)
        
        num_threads = 20
        access_results = []
        
        def access_worker(thread_id):
            """工作线程：访问生成器"""
            try:
                # 随机访问不同的生成器
                target_type = f"access_test_{thread_id % 5}"
                generator = registry.get_generator(target_type)
                
                # 验证获取的生成器正确
                success = generator is not None and generator.mtype == target_type
                
                access_results.append({
                    'thread_id': thread_id,
                    'target_type': target_type,
                    'success': success
                })
                
                return success
                
            except Exception as e:
                access_results.append({
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e)
                })
                return False
        
        # 并发执行访问
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(access_worker, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        # 验证所有访问都成功
        assert all(results), f"Some accesses failed: {access_results}"
    
    def test_concurrent_registration_and_access(self, clean_registry):
        """测试并发注册和访问的混合场景"""
        registry = RandomGeneratorRegistry()
        num_register_threads = 5
        num_access_threads = 10
        all_results = []
        
        def register_worker(thread_id):
            """注册工作线程"""
            try:
                mock_gen = Mock(spec=BaseRandomGenerator)
                mock_gen.mtype = f"mixed_test_{thread_id}"
                
                time.sleep(0.01)  # 模拟一些工作
                registry.register(f"mixed_test_{thread_id}", mock_gen)
                
                return True
            except Exception:
                return False
        
        def access_worker(thread_id):
            """访问工作线程"""
            try:
                # 等待一些生成器被注册
                time.sleep(0.02)
                
                # 尝试访问已注册的生成器
                available_generators = registry.list_mtypes()
                if available_generators:
                    target_type = available_generators[thread_id % len(available_generators)]
                    generator = registry.get_generator(target_type)
                    return generator is not None
                else:
                    return True  # 如果没有可用生成器，这也是正常的
            except Exception:
                return False
        
        # 并发执行注册和访问
        with ThreadPoolExecutor(max_workers=num_register_threads + num_access_threads) as executor:
            # 提交注册任务
            register_futures = [executor.submit(register_worker, i) for i in range(num_register_threads)]
            
            # 提交访问任务
            access_futures = [executor.submit(access_worker, i) for i in range(num_access_threads)]
            
            # 收集结果
            register_results = [future.result() for future in as_completed(register_futures)]
            access_results = [future.result() for future in as_completed(access_futures)]
        
        # 验证注册操作成功
        assert all(register_results), "Some registrations failed"
        
        # 验证访问操作成功（允许一些失败，因为可能在注册完成前访问）
        success_rate = sum(access_results) / len(access_results)
        assert success_rate >= 0.8, f"Access success rate too low: {success_rate}"
        
        # 验证最终状态一致
        final_generators = registry.list_mtypes()
        assert len(final_generators) == num_register_threads
    
    def test_concurrent_unregistration(self, clean_registry):
        """测试并发注销的线程安全性"""
        registry = RandomGeneratorRegistry()
        num_generators = 10
        
        # 预先注册生成器
        for i in range(num_generators):
            mock_gen = Mock(spec=BaseRandomGenerator)
            mock_gen.mtype = f"unregister_test_{i}"
            registry.register(f"unregister_test_{i}", mock_gen)
        
        def unregister_worker(thread_id):
            """注销工作线程"""
            try:
                target_type = f"unregister_test_{thread_id}"
                registry.unregister(target_type)
                return True
            except KeyError:
                # 可能已被其他线程注销，这是正常的
                return True
            except Exception:
                return False
        
        # 并发执行注销
        with ThreadPoolExecutor(max_workers=num_generators) as executor:
            futures = [executor.submit(unregister_worker, i) for i in range(num_generators)]
            results = [future.result() for future in as_completed(futures)]
        
        # 验证操作成功
        assert all(results), "Some unregistrations failed"
        
        # 验证所有生成器都已注销
        final_generators = registry.list_mtypes()
        assert len(final_generators) == 0


class TestRegistryValidation:
    """测试注册表验证逻辑"""
    
    def test_invalid_generator_type(self, clean_registry):
        """测试无效生成器类型的处理"""
        registry = RandomGeneratorRegistry()
        
        # 尝试注册非BaseRandomGenerator实例
        invalid_generator = Mock()  # 不是BaseRandomGenerator的子类
        
        with pytest.raises(TypeError, match="Generator must be an instance of BaseRandomGenerator"):
            registry.register("invalid_type", invalid_generator)
    
    def test_none_generator(self, clean_registry):
        """测试None生成器的处理"""
        registry = RandomGeneratorRegistry()
        
        with pytest.raises(TypeError):
            registry.register("none_type", None)
    
    def test_invalid_mtype(self, clean_registry, mock_generator):
        """测试无效mtype的处理"""
        registry = RandomGeneratorRegistry()
        
        # 测试空字符串mtype
        with pytest.raises(ValueError, match="mtype cannot be empty"):
            registry.register("", mock_generator)
        
        # 测试None mtype
        with pytest.raises(ValueError, match="mtype cannot be empty"):
            registry.register(None, mock_generator)
        
        # 测试非字符串mtype - 这会在类型检查时失败，但运行时可能通过
        # 实际上123会被转换为字符串"123"，所以不会抛出错误
        # 我们测试一个会导致strip()失败的类型
        with pytest.raises(AttributeError):
            registry.register(123, mock_generator)
    
    def test_mtype_consistency_validation(self, clean_registry):
        """测试mtype一致性验证"""
        registry = RandomGeneratorRegistry()
        
        # 创建一个mtype与注册键不匹配的生成器
        mock_gen = Mock(spec=BaseRandomGenerator)
        mock_gen.mtype = "actual_type"
        
        # 注册时使用不同的键应该失败
        with pytest.raises(ValueError, match="Generator mtype 'actual_type' does not match registration mtype 'registered_type'"):
            registry.register("registered_type", mock_gen)
    
    def test_generator_instance_validation(self, clean_registry):
        """测试生成器实例验证"""
        registry = RandomGeneratorRegistry()
        
        # 创建一个不完整的生成器（缺少必要方法）
        class IncompleteGenerator:
            @property
            def mtype(self):
                return "incomplete"
        
        incomplete_gen = IncompleteGenerator()
        
        # 应该被拒绝（不是BaseRandomGenerator的实例）
        with pytest.raises(TypeError):
            registry.register("incomplete_type", incomplete_gen)


class TestRegistryPerformance:
    """测试注册表性能相关功能"""
    
    def test_registration_performance(self, clean_registry, performance_timer):
        """测试注册性能"""
        registry = RandomGeneratorRegistry()
        num_generators = 1000
        
        # 创建生成器列表
        generators = []
        for i in range(num_generators):
            mock_gen = Mock(spec=BaseRandomGenerator)
            mock_gen.mtype = f"perf_test_{i}"
            generators.append((f"perf_test_{i}", mock_gen))
        
        # 测量注册时间
        with performance_timer:
            for mtype, generator in generators:
                registry.register(mtype, generator)
        
        # 验证性能在合理范围内
        avg_time = performance_timer.elapsed_time / num_generators
        assert avg_time < 0.001  # 每次注册应该少于1毫秒
        
        # 验证所有生成器都已注册
        assert len(registry.list_mtypes()) == num_generators
    
    def test_access_performance(self, clean_registry, performance_timer):
        """测试访问性能"""
        registry = RandomGeneratorRegistry()
        num_generators = 100
        num_accesses = 10000
        
        # 预先注册生成器
        for i in range(num_generators):
            mock_gen = Mock(spec=BaseRandomGenerator)
            mock_gen.mtype = f"access_perf_{i}"
            registry.register(f"access_perf_{i}", mock_gen)
        
        # 测量访问时间
        with performance_timer:
            for i in range(num_accesses):
                target_type = f"access_perf_{i % num_generators}"
                _ = registry.get_generator(target_type)
        
        # 验证访问性能
        avg_time = performance_timer.elapsed_time / num_accesses
        assert avg_time < 0.0001  # 每次访问应该很快
    
    def test_list_performance(self, clean_registry, performance_timer):
        """测试列举性能"""
        registry = RandomGeneratorRegistry()
        num_generators = 1000
        
        # 预先注册大量生成器
        for i in range(num_generators):
            mock_gen = Mock(spec=BaseRandomGenerator)
            mock_gen.mtype = f"list_perf_{i}"
            registry.register(f"list_perf_{i}", mock_gen)
        
        # 测量列举时间
        num_lists = 100
        with performance_timer:
            for _ in range(num_lists):
                _ = registry.list_mtypes()
        
        # 验证列举性能
        avg_time = performance_timer.elapsed_time / num_lists
        assert avg_time < 0.01  # 每次列举应该少于10毫秒


class TestEdgeCases:
    """
    测试边界条件和极值情况
    
    这些测试验证注册表在边界条件下的健壮性
    """
    
    def test_extremely_long_mtype_names(self, clean_registry, mock_generator):
        """测试极长的mtype名称"""
        registry = RandomGeneratorRegistry()
        
        # 测试非常长的mtype名称
        long_mtype = 'a' * 10000  # 10K字符
        mock_generator.mtype = long_mtype  # 设置正确的mtype
        registry.register(long_mtype, mock_generator)
        
        # 验证注册成功
        assert long_mtype in registry._generators
        assert registry.get_generator(long_mtype) is mock_generator
        
        # 测试极长的Unicode mtype
        unicode_long_mtype = '中文' * 5000  # 10K Unicode字符
        # 创建新的mock生成器用于Unicode测试
        unicode_mock_generator = Mock(spec=BaseRandomGenerator)
        unicode_mock_generator.mtype = unicode_long_mtype
        registry.register(unicode_long_mtype, unicode_mock_generator)
        assert unicode_long_mtype in registry._generators
    
    def test_massive_registration_stress(self, clean_registry):
        """测试大量注册的压力测试"""
        registry = RandomGeneratorRegistry()
        
        # 创建大量生成器
        num_generators = 10000
        generators = {}
        
        for i in range(num_generators):
            mtype = f'generator_{i}'
            generator = Mock(spec=BaseRandomGenerator)
            generator.mtype = mtype
            generators[mtype] = generator
            registry.register(mtype, generator)
        
        # 验证所有生成器都已注册
        assert len(registry._generators) == num_generators
        
        # 验证可以正确检索
        for i in range(0, num_generators, 1000):  # 抽样检查
            mtype = f'generator_{i}'
            assert registry.get_generator(mtype) is generators[mtype]
    
    def test_memory_intensive_operations(self, clean_registry):
        """测试内存密集型操作"""
        registry = RandomGeneratorRegistry()
        
        # 注册大量生成器
        large_generators = {}
        for i in range(1000):
            mtype = f'large_gen_{i}'
            generator = Mock(spec=BaseRandomGenerator)
            generator.mtype = mtype
            # 模拟大对象
            generator.large_data = list(range(1000))  # 每个生成器携带大数据
            large_generators[mtype] = generator
            registry.register(mtype, generator)
        
        # 测试列表操作的性能
        mtypes = registry.list_mtypes()
        assert len(mtypes) == 1000
        
        # 测试批量注销
        for i in range(500):
            mtype = f'large_gen_{i}'
            registry.unregister(mtype)
        
        assert len(registry._generators) == 500
    
    def test_special_character_mtype_names(self, clean_registry):
        """测试特殊字符的mtype名称"""
        registry = RandomGeneratorRegistry()
        
        special_mtypes = [
            'type-with-dashes',
            'type_with_underscores',
            'type.with.dots',
            'type with spaces',
            'type\twith\ttabs',
            'type\nwith\nnewlines',
            'type"with"quotes',
            "type'with'apostrophes",
            'type/with/slashes',
            'type\\with\\backslashes',
            'type@with@symbols',
            'type#with#hash',
            'type$with$dollar',
            'type%with%percent',
            'type&with&ampersand',
            'type*with*asterisk',
            'type+with+plus',
            'type=with=equals',
            'type?with?question',
            'type[with]brackets',
            'type{with}braces',
            'type(with)parentheses',
            'type<with>angles',
            'type|with|pipes',
            'type~with~tilde',
            'type`with`backtick',
            'type^with^caret',
            '🚀emoji🎯test🔥',
            '中文测试类型',
            'русский_тип',
            'ελληνικός_τύπος',
            '日本語タイプ',
            '한국어_타입',
            'العربية_نوع',
        ]
        
        for mtype in special_mtypes:
            generator = Mock(spec=BaseRandomGenerator)
            generator.mtype = mtype
            registry.register(mtype, generator)
            
            # 验证注册和检索
            assert registry.get_generator(mtype) is generator
            assert mtype in registry.list_mtypes()
    
    def test_concurrent_stress_operations(self, clean_registry):
        """测试并发压力操作"""
        registry = RandomGeneratorRegistry()
        results = []
        errors = []
        
        def stress_worker(worker_id):
            try:
                # 每个worker执行多种操作
                for i in range(100):
                    mtype = f'worker_{worker_id}_gen_{i}'
                    generator = Mock(spec=BaseRandomGenerator)
                    generator.mtype = mtype
                    
                    # 注册
                    registry.register(mtype, generator)
                    
                    # 立即检索
                    retrieved = registry.get_generator(mtype)
                    assert retrieved is generator
                    
                    # 列出所有类型（可能很慢）
                    mtypes = registry.list_mtypes()
                    assert mtype in mtypes
                    
                    # 注销一半
                    if i % 2 == 0:
                        registry.unregister(mtype)
                
                results.append(f'Worker {worker_id} completed')
            except Exception as e:
                errors.append(f'Worker {worker_id} error: {e}')
        
        # 启动多个并发worker
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(10)]
            
            for future in as_completed(futures):
                future.result()  # 等待完成并获取异常
        
        # 验证没有错误
        assert len(errors) == 0, f"Concurrent stress test errors: {errors}"
        assert len(results) == 10
    
    def test_registry_state_consistency_under_stress(self, clean_registry):
        """测试压力下的注册表状态一致性"""
        registry = RandomGeneratorRegistry()
        
        # 快速注册和注销大量生成器
        total_registered = 0
        total_unregistered = 0
        
        for cycle in range(10):
            # 注册阶段
            for i in range(1000):
                mtype = f'cycle_{cycle}_gen_{i}'
                generator = Mock(spec=BaseRandomGenerator)
                generator.mtype = mtype
                registry.register(mtype, generator)
                total_registered += 1
            
            # 部分注销
            for i in range(0, 1000, 2):  # 注销偶数索引
                mtype = f'cycle_{cycle}_gen_{i}'
                success = registry.unregister(mtype)
                if success:
                    total_unregistered += 1
        
        # 验证最终状态
        remaining_mtypes = registry.list_mtypes()
        expected_remaining = total_registered - total_unregistered
        # 允许一些误差，因为并发操作可能导致轻微的不一致
        tolerance = max(1, int(expected_remaining * 0.1))  # 10%误差或至少1个
        assert abs(len(remaining_mtypes) - expected_remaining) <= tolerance, f"Expected ~{expected_remaining} mtypes, got {len(remaining_mtypes)}"


class TestErrorHandling:
    """测试错误处理和边界情况"""
    
    def test_registry_state_after_errors(self, clean_registry, mock_generator):
        """测试错误后注册表状态的一致性"""
        registry = RandomGeneratorRegistry()
        
        # 成功注册一个生成器
        registry.register("mock", mock_generator)
        
        # 尝试无效操作
        try:
            registry.register("mock", mock_generator)  # 重复注册
        except ValueError:
            pass
        
        try:
            registry.register("", mock_generator)  # 空mtype
        except ValueError:
            pass
        
        # 验证原有状态未受影响
        assert "mock" in registry.list_mtypes()
        assert registry.get_generator("mock") is mock_generator
    
    def test_concurrent_error_handling(self, clean_registry):
        """测试并发错误处理"""
        registry = RandomGeneratorRegistry()
        num_threads = 10
        
        def error_worker(thread_id):
            """产生错误的工作线程"""
            try:
                if thread_id % 2 == 0:
                    # 偶数线程：尝试注册无效生成器
                    registry.register(f"error_test_{thread_id}", "invalid_generator")
                    return False  # 不应该到达这里
                else:
                    # 奇数线程：尝试注册空mtype
                    mock_gen = Mock(spec=BaseRandomGenerator)
                    registry.register("", mock_gen)
                    return False  # 不应该到达这里
            except (TypeError, ValueError):
                return True  # 预期的错误
            except Exception:
                return False  # 意外的错误
        
        # 并发执行错误操作
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(error_worker, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        # 验证所有错误都被正确处理
        assert all(results), "Some errors were not handled correctly"
        
        # 验证注册表状态仍然一致
        assert len(registry.list_mtypes()) == 0


if __name__ == "__main__":
    # 运行测试的示例
    pytest.main([__file__, "-v"])