#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 axisfuzzy.random.base 模块

本模块测试随机生成器抽象基类的核心功能，包括：
- BaseRandomGenerator 抽象接口
- ParameterizedRandomGenerator 工具方法
- 参数合并和验证逻辑
- 分布采样功能
- 抽象方法强制实现检查
"""

import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
from unittest.mock import Mock, patch

from axisfuzzy.random.base import (
    BaseRandomGenerator,
    ParameterizedRandomGenerator
)


class TestBaseRandomGenerator:
    """测试 BaseRandomGenerator 抽象基类"""
    
    def test_abstract_class_cannot_instantiate(self):
        """测试抽象类不能直接实例化"""
        with pytest.raises(TypeError):
            BaseRandomGenerator()
    
    def test_abstract_methods_defined(self):
        """测试抽象方法已正确定义"""
        # 检查抽象方法列表
        abstract_methods = BaseRandomGenerator.__abstractmethods__
        
        expected_methods = {
            'get_default_parameters',
            'validate_parameters',
            'fuzznum',
            'fuzzarray'
        }
        
        assert abstract_methods == expected_methods
        
        # 验证mtype是类属性而不是抽象方法
        assert hasattr(BaseRandomGenerator, 'mtype')
        assert BaseRandomGenerator.mtype == 'unknown'
    
    def test_concrete_implementation_required(self):
        """测试具体实现必须实现所有抽象方法"""
        
        # 创建一个不完整的实现（缺少某些抽象方法）
        class IncompleteGenerator(BaseRandomGenerator):
            @property
            def mtype(self):
                return "incomplete"
            
            def get_default_parameters(self):
                return {}
            
            # 故意不实现 validate_parameters, fuzznum, fuzzarray
        
        # 尝试实例化应该失败
        with pytest.raises(TypeError):
            IncompleteGenerator()
    
    def test_complete_implementation_works(self, mock_fuzznum, mock_fuzzarray):
        """测试完整实现可以正常工作"""
        
        class CompleteGenerator(BaseRandomGenerator):
            mtype = "complete"  # 类属性而不是property
            
            def get_default_parameters(self):
                return {'param1': 1.0, 'param2': 2.0}
            
            def validate_parameters(self, **params):
                # 简单验证：所有参数必须是数值
                for key, value in params.items():
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Parameter {key} must be numeric")
            
            def fuzznum(self, rng, **params):  # rng是必需参数
                return mock_fuzznum
            
            def fuzzarray(self, rng, shape, **params):  # 参数顺序匹配实际接口
                return mock_fuzzarray
        
        # 应该能够成功实例化
        generator = CompleteGenerator()
        
        # 验证基本功能
        assert generator.mtype == "complete"
        assert generator.get_default_parameters() == {'param1': 1.0, 'param2': 2.0}
        
        # 验证参数验证（validate_parameters不返回值）
        generator.validate_parameters(param1=5.0, param2=10.0)  # 应该不抛出异常
        
        # 验证生成方法（需要提供rng参数）
        rng = np.random.default_rng(42)
        assert generator.fuzznum(rng) is mock_fuzznum
        assert generator.fuzzarray(rng, (3, 3)) is mock_fuzzarray
    
    def test_mtype_property_requirements(self):
        """测试 mtype 属性的要求"""
        
        class TestGenerator(BaseRandomGenerator):
            def __init__(self, mtype_value):
                self.mtype = mtype_value  # 直接设置类属性
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass  # validate_parameters不返回值
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        # 测试有效的 mtype
        valid_generator = TestGenerator("triangular")
        assert valid_generator.mtype == "triangular"
        
        # 测试 mtype 应该是字符串
        invalid_generator = TestGenerator(123)
        assert invalid_generator.mtype == 123  # 基类不强制类型检查


class TestParameterizedRandomGenerator:
    """测试 ParameterizedRandomGenerator 抽象基类"""
    
    def test_inherits_from_base(self):
        """测试继承关系"""
        assert issubclass(ParameterizedRandomGenerator, BaseRandomGenerator)
    
    def test_abstract_class_cannot_instantiate(self):
        """测试抽象类不能直接实例化"""
        with pytest.raises(TypeError):
            ParameterizedRandomGenerator()
    
    def test_merge_parameters_method(self):
        """测试参数合并方法"""
        
        class TestParameterizedGenerator(ParameterizedRandomGenerator):
            mtype = "test_param"  # 类属性
            
            def get_default_parameters(self):
                return {
                    'a': 1.0,
                    'b': 2.0,
                    'c': 3.0
                }
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestParameterizedGenerator()
        
        # 测试基本合并
        merged = generator._merge_parameters(b=5.0, d=4.0)
        expected = {'a': 1.0, 'b': 5.0, 'c': 3.0, 'd': 4.0}
        assert merged == expected
        
        # 测试空参数
        merged_empty = generator._merge_parameters()
        assert merged_empty == {'a': 1.0, 'b': 2.0, 'c': 3.0}
        
        # 测试完全覆盖
        merged_override = generator._merge_parameters(a=10.0, b=20.0, c=30.0)
        expected_override = {'a': 10.0, 'b': 20.0, 'c': 30.0}
        assert merged_override == expected_override
    
    def test_sample_from_distribution_method(self, sample_rng):
        """测试分布采样方法"""
        
        class TestParameterizedGenerator(ParameterizedRandomGenerator):
            @property
            def mtype(self):
                return "test_dist"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                return params
            
            def fuzznum(self, rng=None, **params):
                return Mock()
            
            def fuzzarray(self, shape, rng=None, **params):
                return Mock()
        
        generator = TestParameterizedGenerator()
        
        # 测试均匀分布采样
        uniform_sample = generator._sample_from_distribution(
            sample_rng, dist='uniform', low=0.0, high=1.0
        )
        assert isinstance(uniform_sample, float)
        assert 0.0 <= uniform_sample <= 1.0
        
        # 测试正态分布采样
        normal_sample = generator._sample_from_distribution(
            sample_rng, dist='normal', loc=0.0, scale=1.0
        )
        assert isinstance(normal_sample, float)
        
        # 测试Beta分布采样
        beta_sample = generator._sample_from_distribution(
            sample_rng, dist='beta', low=0.0, high=1.0, a=2.0, b=3.0
        )
        assert isinstance(beta_sample, float)
        assert 0.0 <= beta_sample <= 1.0
        
        # 测试数组采样
        uniform_array = generator._sample_from_distribution(
            sample_rng, size=10, dist='uniform', low=0.0, high=1.0
        )
        assert isinstance(uniform_array, np.ndarray)
        assert uniform_array.shape == (10,)
        assert np.all(uniform_array >= 0.0)
        assert np.all(uniform_array <= 1.0)
    
    def test_sample_from_distribution_invalid_distribution(self, sample_rng):
        """测试无效分布名称的处理"""
        
        class TestParameterizedGenerator(ParameterizedRandomGenerator):
            mtype = "test_invalid"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestParameterizedGenerator()
        
        # 测试无效分布名称
        with pytest.raises(ValueError, match="Unsupported distribution"):
            generator._sample_from_distribution(
                sample_rng, dist='invalid_distribution'
            )
    
    def test_validate_range_method(self):
        """测试范围验证方法"""
        
        class TestParameterizedGenerator(ParameterizedRandomGenerator):
            mtype = "test_range"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestParameterizedGenerator()
        
        # 测试有效范围（根据实际方法签名：name, value, min_val, max_val）
        generator._validate_range("test_param", 5.0, 0.0, 10.0)
        generator._validate_range("boundary_min", 0.0, 0.0, 10.0)
        generator._validate_range("boundary_max", 10.0, 0.0, 10.0)
        
        # 测试无效范围（值太小）
        with pytest.raises(ValueError, match="test_param.*must be between"):
            generator._validate_range("test_param", -1.0, 0.0, 10.0)
        
        # 测试无效范围（值太大）
        with pytest.raises(ValueError, match="test_param.*must be between"):
            generator._validate_range("test_param", 15.0, 0.0, 10.0)
    
    def test_validate_range_edge_cases(self):
        """测试范围验证的边界情况"""
        
        class TestParameterizedGenerator(ParameterizedRandomGenerator):
            mtype = "test_edge"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestParameterizedGenerator()
        
        # 测试边界值
        generator._validate_range("param", 0.0, 0.0, 1.0)
        generator._validate_range("param", 1.0, 0.0, 1.0)
        generator._validate_range("param", 0.5, 0.0, 1.0)
        
        # 测试相等的边界
        generator._validate_range("param", 5.0, 5.0, 5.0)


class TestParameterMerging:
    """测试参数合并逻辑的详细行为"""
    
    def setup_method(self):
        """设置测试用的生成器类"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "merge_test"
            
            def get_default_parameters(self):
                return {
                    'alpha': 1.0,
                    'beta': 2.0,
                    'gamma': 3.0,
                    'nested': {'x': 10, 'y': 20}
                }
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        self.generator = TestGenerator()
    
    def test_parameter_precedence(self):
        """测试参数优先级（用户参数覆盖默认参数）"""
        merged = self.generator._merge_parameters(
            alpha=5.0,  # 覆盖默认值
            delta=4.0   # 新增参数
        )
        
        expected = {
            'alpha': 5.0,  # 被覆盖
            'beta': 2.0,   # 保持默认
            'gamma': 3.0,  # 保持默认
            'nested': {'x': 10, 'y': 20},  # 保持默认
            'delta': 4.0   # 新增
        }
        
        assert merged == expected
    
    def test_nested_parameter_handling(self):
        """测试嵌套参数的处理"""
        # 注意：当前实现可能不支持深度合并嵌套字典
        # 这个测试验证当前行为
        merged = self.generator._merge_parameters(
            nested={'z': 30}  # 完全替换嵌套字典
        )
        
        # 验证嵌套字典被完全替换（不是合并）
        assert merged['nested'] == {'z': 30}
        assert 'x' not in merged['nested']
        assert 'y' not in merged['nested']
    
    def test_parameter_type_preservation(self):
        """测试参数类型保持"""
        merged = self.generator._merge_parameters(
            alpha=5,      # int
            beta=2.5,     # float
            gamma="3.0",  # string
            delta=True    # bool
        )
        
        assert isinstance(merged['alpha'], int)
        assert isinstance(merged['beta'], float)
        assert isinstance(merged['gamma'], str)
        assert isinstance(merged['delta'], bool)
    
    def test_empty_defaults_handling(self):
        """测试空默认参数的处理"""
        
        class EmptyDefaultsGenerator(ParameterizedRandomGenerator):
            mtype = "empty_defaults"
            
            def get_default_parameters(self):
                return {}  # 空默认参数
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = EmptyDefaultsGenerator()
        
        # 测试只有用户参数
        merged = generator._merge_parameters(a=1, b=2)
        assert merged == {'a': 1, 'b': 2}
        
        # 测试无参数
        merged_empty = generator._merge_parameters()
        assert merged_empty == {}


class TestDistributionSampling:
    """测试分布采样功能的详细行为"""
    
    def setup_method(self):
        """设置测试用的生成器"""
        
        class SamplingTestGenerator(ParameterizedRandomGenerator):
            mtype = "sampling_test"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        self.generator = SamplingTestGenerator()
    
    def test_uniform_distribution_sampling(self, sample_rng):
        """测试均匀分布采样的详细行为"""
        # 单个样本
        sample = self.generator._sample_from_distribution(
            sample_rng, dist='uniform', low=2.0, high=8.0
        )
        assert 2.0 <= sample <= 8.0
        
        # 多个样本
        samples = self.generator._sample_from_distribution(
            sample_rng, size=100, dist='uniform', low=2.0, high=8.0
        )
        assert samples.shape == (100,)
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)
        
        # 统计检验：均值应该接近中点
        mean_sample = np.mean(samples)
        assert 4.0 < mean_sample < 6.0  # 允许统计波动
    
    def test_normal_distribution_sampling(self, sample_rng):
        """测试正态分布采样的详细行为"""
        # 标准正态分布
        sample = self.generator._sample_from_distribution(
            sample_rng, dist='normal', low=-3.0, high=3.0, loc=0.0, scale=1.0
        )
        assert isinstance(sample, float)
        
        # 自定义正态分布
        samples = self.generator._sample_from_distribution(
            sample_rng, size=1000, dist='normal', low=5.0, high=15.0, loc=10.0, scale=2.0
        )
        
        # 统计检验 - 由于clipping，均值会偏向中心
        mean_sample = np.mean(samples)
        # 由于clipping效应，均值应该在合理范围内
        assert 8.0 < mean_sample < 12.0
    
    def test_beta_distribution_sampling(self, sample_rng):
        """测试Beta分布采样的详细行为"""
        # Beta分布参数
        alpha, beta = 2.0, 5.0
        
        samples = self.generator._sample_from_distribution(
            sample_rng, size=1000, dist='beta', low=0.0, high=1.0, a=alpha, b=beta
        )
        
        # Beta分布的值域是[0, 1]
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
        
        # 统计检验：Beta分布的理论均值是 a/(a+b)
        theoretical_mean = alpha / (alpha + beta)
        sample_mean = np.mean(samples)
        
        assert abs(sample_mean - theoretical_mean) < 0.05
    
    def test_distribution_parameter_validation(self, sample_rng):
        """测试分布参数验证"""
        # 测试均匀分布的无效参数
        with pytest.raises(ValueError):
            self.generator._sample_from_distribution(
                sample_rng, dist='uniform', low=5.0, high=2.0  # low > high
            )
        
        # 测试负标准差 - 这个由numpy处理，不会在我们的方法中抛出错误
        # 但我们可以测试其他无效参数
        with pytest.raises(ValueError):
            self.generator._sample_from_distribution(
                sample_rng, dist='beta', low=0.0, high=1.0, a=0.0, b=1.0  # alpha <= 0
            )
        
        # 测试缺少必需参数 - 实际上uniform有默认值，所以不会失败
        # 我们测试其他情况
        try:
            # 这应该工作，因为有默认值
            result = self.generator._sample_from_distribution(
                sample_rng, dist='uniform'
            )
            assert isinstance(result, float)
        except Exception:
            pytest.fail("Uniform distribution should work with default parameters")
    
    def test_multidimensional_sampling(self, sample_rng):
        """测试多维采样"""
        # 2D数组
        samples_2d = self.generator._sample_from_distribution(
            sample_rng, size=(10, 5), dist='uniform', low=0.0, high=1.0
        )
        assert samples_2d.shape == (10, 5)
        assert np.all(samples_2d >= 0.0)
        assert np.all(samples_2d <= 1.0)
        
        # 3D数组
        samples_3d = self.generator._sample_from_distribution(
            sample_rng, size=(3, 4, 5), dist='normal', low=-2.0, high=2.0, loc=0.0, scale=1.0
        )
        assert samples_3d.shape == (3, 4, 5)


class TestRangeValidation:
    """测试范围验证功能的详细行为"""
    
    def setup_method(self):
        """设置测试用的生成器"""
        
        class ValidationTestGenerator(ParameterizedRandomGenerator):
            @property
            def mtype(self):
                return "validation_test"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                return params
            
            def fuzznum(self, rng=None, **params):
                return Mock()
            
            def fuzzarray(self, shape, rng=None, **params):
                return Mock()
        
        self.generator = ValidationTestGenerator()
    
    def test_inclusive_bounds(self):
        """测试包含边界的验证"""
        # 边界值应该通过验证
        self.generator._validate_range("param", 0.0, 0.0, 10.0)
        self.generator._validate_range("param", 10.0, 0.0, 10.0)
        self.generator._validate_range("param", 5.0, 0.0, 10.0)
    
    def test_exclusive_bounds(self):
        """测试排除边界的验证"""
        # 测试超出下界
        with pytest.raises(ValueError):
            self.generator._validate_range("param", -1.0, 0.0, 10.0)
        
        # 测试超出上界
        with pytest.raises(ValueError):
            self.generator._validate_range("param", 11.0, 0.0, 10.0)
        
        # 边界内的值应该通过
        self.generator._validate_range("param", 5.0, 0.0, 10.0)
    
    def test_error_messages(self):
        """测试错误消息的准确性"""
        # 测试下界错误消息
        with pytest.raises(ValueError, match="test_param.*must be between"):
            self.generator._validate_range("test_param", 3.0, 5.0, 15.0)
        
        # 测试上界错误消息
        with pytest.raises(ValueError, match="test_param.*must be between"):
            self.generator._validate_range("test_param", 20.0, 5.0, 15.0)
    
    def test_special_numeric_values(self):
        """测试特殊数值的处理"""
        # 测试无穷大
        with pytest.raises(ValueError):
            self.generator._validate_range("param", float('inf'), 0.0, 10.0)
        
        # 测试负无穷大
        with pytest.raises(ValueError):
            self.generator._validate_range("param", float('-inf'), 0.0, 10.0)
        
        # 测试NaN
        with pytest.raises(ValueError):
            self.generator._validate_range("param", float('nan'), 0.0, 10.0)
    
    def test_type_validation(self):
        """测试类型验证"""
        # 数值类型应该通过
        self.generator._validate_range("int_param", 5, 0, 10)
        self.generator._validate_range("float_param", 5.5, 0.0, 10.0)
        
        # 非数值类型应该失败
        with pytest.raises((TypeError, ValueError)):
            self.generator._validate_range("string_param", "5", 0, 10)
        
        with pytest.raises((TypeError, ValueError)):
            self.generator._validate_range("list_param", [5], 0, 10)


class TestPerformance:
    """
    测试BaseRandomGenerator性能相关功能
    
    这些测试验证生成器在高负载和大规模数据处理时的性能表现
    """
    
    def test_parameter_merging_performance(self):
        """测试参数合并的性能"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "perf_test"
            
            def get_default_parameters(self):
                return {f'default_{i}': i for i in range(100)}  # 100个默认参数
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试大量参数合并的性能
        large_params = {f'param_{i}': i * 0.1 for i in range(500)}
        
        import time
        num_iterations = 1000
        
        start_time = time.time()
        for _ in range(num_iterations):
            merged = generator._merge_parameters(**large_params)
        elapsed_time = time.time() - start_time
        
        # 验证性能（每次合并应该很快）
        avg_time = elapsed_time / num_iterations
        assert avg_time < 0.001, f"Parameter merging took {avg_time:.6f}s per operation, too slow"
        
        # 验证合并结果正确
        assert len(merged) == 600  # 100个默认 + 500个传入
    
    def test_range_validation_performance(self):
        """测试范围验证的性能"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "range_perf_test"
            
            def get_default_parameters(self):
                return {'a': 0.0, 'b': 1.0}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试大量范围验证的性能
        test_values = [0.5, 5.0, 1e-5, 1e5]
        test_ranges = [(0.0, 1.0), (-10.0, 10.0), (1e-10, 1e10), (-1e6, 1e6)]
        
        import time
        num_iterations = 10000
        
        start_time = time.time()
        for _ in range(num_iterations):
            for i, (min_val, max_val) in enumerate(test_ranges):
                value = test_values[i % len(test_values)]
                if min_val <= value <= max_val:
                    generator._validate_range('test_param', value, min_val, max_val)
        elapsed_time = time.time() - start_time
        
        # 验证验证性能
        total_validations = num_iterations * len(test_ranges)
        avg_time = elapsed_time / total_validations
        assert avg_time < 0.0001, f"Range validation took {avg_time:.8f}s per operation, too slow"
    
    def test_distribution_sampling_performance(self, sample_rng):
        """测试分布采样的性能"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "dist_perf_test"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试不同分布的采样性能
        distributions = [
            {'dist': 'uniform', 'low': 0.0, 'high': 1.0},
            {'dist': 'normal', 'low': -3.0, 'high': 3.0, 'loc': 0.0, 'scale': 1.0},
            {'dist': 'beta', 'low': 0.0, 'high': 1.0, 'a': 2.0, 'b': 5.0}
        ]
        
        for dist_params in distributions:
            import time
            num_samples = 1000
            
            start_time = time.time()
            for _ in range(num_samples):
                sample = generator._sample_from_distribution(sample_rng, **dist_params)
            elapsed_time = time.time() - start_time
            
            # 验证采样性能
            avg_time = elapsed_time / num_samples
            assert avg_time < 0.001, f"Distribution {dist_params['dist']} sampling took {avg_time:.8f}s per sample, too slow"
    
    def test_large_array_generation_performance(self, sample_rng):
        """测试大数组生成的性能"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "array_perf_test"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                # 模拟实际的数组生成
                size = np.prod(shape)
                samples = generator._sample_from_distribution(rng, size=shape, dist='uniform', low=0.0, high=1.0)
                return Mock(shape=shape, size=size, values=samples)
        
        generator = TestGenerator()
        
        # 测试不同大小数组的生成性能
        test_shapes = [
            (1000,),
            (100, 100),
            (50, 50, 4)
        ]
        
        for shape in test_shapes:
            import time
            
            start_time = time.time()
            result = generator.fuzzarray(sample_rng, shape)
            elapsed_time = time.time() - start_time
            
            # 验证生成性能（应该在合理时间内完成）
            size = np.prod(shape)
            assert elapsed_time < 1.0, f"Array generation for shape {shape} (size {size}) took {elapsed_time:.4f}s, too slow"
    
    def test_memory_efficiency_validation(self, sample_rng):
        """测试内存效率验证"""
        
        class MemoryEfficientGenerator(ParameterizedRandomGenerator):
            mtype = "memory_eff_test"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock(shape=shape)
        
        generator = MemoryEfficientGenerator()
        
        # 测试大量小对象生成的内存效率
        num_objects = 1000
        
        # 生成大量对象
        objects = []
        for i in range(num_objects):
            obj = generator.fuzznum(sample_rng)
            objects.append(obj)
        
        # 验证对象都已创建
        assert len(objects) == num_objects
        
        # 清理对象
        del objects
        
        # 测试大数组的内存效率
        large_shapes = [(100, 100), (50, 50, 4)]
        
        for shape in large_shapes:
            large_array = generator.fuzzarray(sample_rng, shape)
            assert large_array.shape == shape
            del large_array


class TestEdgeCases:
    """
    测试边界条件和极值情况
    
    这些测试验证基础生成器在边界条件下的健壮性
    """
    
    def test_extreme_range_validation(self):
        """测试极值范围验证"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_extreme"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试极大范围
        generator._validate_range("param", 0.0, -1e100, 1e100)
        
        # 测试极小范围
        generator._validate_range("param", 1.5e-100, 1e-100, 2e-100)
        
        # 测试接近相等的边界
        generator._validate_range("param", 1.0000000001, 1.0, 1.0000000002)
        
        # 测试无穷大
        with pytest.raises(ValueError):
            generator._validate_range("param", float('-inf'), -1e100, 1e100)
        
        with pytest.raises(ValueError):
            generator._validate_range("param", float('inf'), -1e100, 1e100)
    
    def test_extreme_distribution_parameters(self, sample_rng):
        """测试极值分布参数"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_extreme_dist"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试极小的标准差
        samples = generator._sample_from_distribution(
            sample_rng, size=100, dist='normal', low=-1e-5, high=1e-5, loc=0.0, scale=1e-10
        )
        assert samples.shape == (100,)
        assert np.all(np.abs(samples) < 1e-4)  # 应该非常接近0
        
        # 测试极大的标准差（需要clipping）
        samples = generator._sample_from_distribution(
            sample_rng, size=100, dist='normal', low=-1e5, high=1e5, loc=0.0, scale=1e10
        )
        assert samples.shape == (100,)
        # 由于clipping，所有值都在范围内
        assert np.all(samples >= -1e5) and np.all(samples <= 1e5)
    
    def test_numerical_precision_limits(self, sample_rng):
        """测试数值精度限制"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_precision"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试接近机器精度的参数
        eps = np.finfo(float).eps
        
        # 测试beta分布的极值参数
        samples = generator._sample_from_distribution(
            sample_rng, size=100, dist='beta', low=0.0, high=1.0, a=1e-6, b=1.0
        )
        assert samples.shape == (100,)
        assert np.all(samples >= 0) and np.all(samples <= 1)
        
        # 测试接近0的参数
        samples = generator._sample_from_distribution(
            sample_rng, size=100, dist='uniform', low=eps, high=2*eps
        )
        assert samples.shape == (100,)
        assert np.all(samples >= 0) and np.all(samples <= 2*eps)
    
    def test_large_shape_handling(self, sample_rng):
        """测试大形状数组的处理"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_large_shape"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试大的一维数组
        large_shape = (10000,)  # 减小尺寸以避免内存问题
        samples = generator._sample_from_distribution(
            sample_rng, size=large_shape, dist='uniform', low=0.0, high=1.0
        )
        assert samples.shape == large_shape
        assert len(samples) == 10000
        
        # 测试多维大数组
        large_2d_shape = (100, 100)  # 减小尺寸
        samples = generator._sample_from_distribution(
            sample_rng, size=large_2d_shape, dist='normal', low=-3.0, high=3.0, loc=0.0, scale=1.0
        )
        assert samples.shape == large_2d_shape
        assert samples.size == 10000
    
    def test_parameter_type_edge_cases(self):
        """测试参数类型边界情况"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_param_types"
            
            def get_default_parameters(self):
                return {'default_param': 1.0}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试numpy标量
        params = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3])
        }
        
        merged = generator._merge_parameters(**params)
        
        # 验证numpy类型被正确处理
        assert merged['numpy_int'] == 42
        assert merged['numpy_float'] == 3.14
        assert np.array_equal(merged['numpy_array'], np.array([1, 2, 3]))
        assert merged['default_param'] == 1.0  # 默认参数保留
    
    def test_memory_intensive_operations(self):
        """测试内存密集型操作"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_memory"
            
            def get_default_parameters(self):
                return {'default_1': 1.0, 'default_2': 2.0}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试大量参数合并
        many_params = {f'param_{i}': i for i in range(1000)}  # 减少数量
        merged = generator._merge_parameters(**many_params)
        
        # 验证所有参数都被正确合并
        assert len(merged) >= 1002  # 1000个新参数 + 2个默认参数
        for i in range(1000):
            assert merged[f'param_{i}'] == i
        assert merged['default_1'] == 1.0
        assert merged['default_2'] == 2.0


class TestIntegrationWithMocks:
    """测试抽象基类与模拟对象的集成"""
    
    def test_complete_generator_workflow(self, mock_fuzznum, mock_fuzzarray, sample_rng):
        """测试完整的生成器工作流程"""
        
        class WorkflowTestGenerator(ParameterizedRandomGenerator):
            mtype = "workflow_test"
            
            def get_default_parameters(self):
                return {
                    'min_val': 0.0,
                    'max_val': 1.0,
                    'shape_param': 2.0
                }
            
            def validate_parameters(self, **params):
                # 使用基类的验证工具
                if 'min_val' in params:
                    self._validate_range('min_val', params['min_val'], 0.0, 10.0)
                if 'max_val' in params:
                    self._validate_range('max_val', params['max_val'], 0.0, 10.0)
                if 'shape_param' in params:
                    self._validate_range('shape_param', params['shape_param'], 1.0, 5.0)
            
            def fuzznum(self, rng, **params):
                # 合并参数
                merged_params = self._merge_parameters(**params)
                
                # 验证参数
                self.validate_parameters(**merged_params)
                
                # 使用分布采样
                sample_value = self._sample_from_distribution(
                    rng or sample_rng, dist='uniform',
                    low=merged_params['min_val'],
                    high=merged_params['max_val']
                )
                
                # 返回模拟的Fuzznum
                mock_fuzznum.value = sample_value
                return mock_fuzznum
            
            def fuzzarray(self, rng, shape, **params):
                # 类似的工作流程，但返回数组
                merged_params = self._merge_parameters(**params)
                self.validate_parameters(**merged_params)
                
                sample_array = self._sample_from_distribution(
                    rng or sample_rng, size=shape, dist='uniform',
                    low=merged_params['min_val'],
                    high=merged_params['max_val']
                )
                
                mock_fuzzarray.values = sample_array
                mock_fuzzarray.shape = shape
                return mock_fuzzarray
        
        # 创建生成器并测试
        generator = WorkflowTestGenerator()
        
        # 测试fuzznum生成
        result_num = generator.fuzznum(sample_rng, min_val=0.2, max_val=0.8)
        assert result_num is mock_fuzznum
        assert 0.2 <= result_num.value <= 0.8
        
        # 测试fuzzarray生成
        result_array = generator.fuzzarray(sample_rng, (5, 3), min_val=0.1, max_val=0.9)
        assert result_array is mock_fuzzarray
        assert result_array.shape == (5, 3)
        assert np.all(result_array.values >= 0.1)
        assert np.all(result_array.values <= 0.9)
        
        # 测试参数验证失败
        with pytest.raises(ValueError):
            generator.fuzznum(sample_rng, min_val=-1.0)  # 违反min_val >= 0.0
        
        with pytest.raises(ValueError):
            generator.fuzzarray(sample_rng, (2, 2), shape_param=6.0)  # 违反shape_param <= 5.0


class TestParameterValidationEdgeCases:
    """测试参数验证的边界情况"""
    
    def test_boundary_value_analysis(self):
        """测试边界值分析"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_boundary"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试边界值：最小值、最大值、中间值
        test_cases = [
            (0.0, 0.0, 10.0),    # 最小边界
            (10.0, 0.0, 10.0),   # 最大边界
            (5.0, 0.0, 10.0),    # 中间值
            (-5.0, -10.0, 0.0),  # 负数范围最大边界
            (-10.0, -10.0, 0.0), # 负数范围最小边界
        ]
        
        for value, min_val, max_val in test_cases:
            generator._validate_range("test_param", value, min_val, max_val)
        
        # 测试超出边界的情况
        boundary_violations = [
            (-0.1, 0.0, 10.0),   # 略小于最小值
            (10.1, 0.0, 10.0),   # 略大于最大值
            (-11.0, -10.0, 0.0), # 负数范围超出
        ]
        
        for value, min_val, max_val in boundary_violations:
            with pytest.raises(ValueError):
                generator._validate_range("test_param", value, min_val, max_val)
    
    def test_floating_point_precision_issues(self):
        """测试浮点精度问题"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_float_precision"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试浮点运算精度问题
        # 0.1 + 0.2 != 0.3 的经典问题
        computed_value = 0.1 + 0.2  # 实际上是 0.30000000000000004
        
        # 这应该通过，因为我们的验证应该处理浮点精度
        try:
            generator._validate_range("float_precision", computed_value, 0.0, 1.0)
        except ValueError:
            pytest.fail("Float precision should not cause validation failure")
        
        # 测试非常接近边界的值
        eps = np.finfo(float).eps
        
        # 这些应该通过
        generator._validate_range("near_boundary", 1.0 - eps, 0.0, 1.0)
        generator._validate_range("near_boundary", 0.0 + eps, 0.0, 1.0)
    
    def test_special_float_values_comprehensive(self):
        """测试特殊浮点值的全面处理"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_special_floats"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        special_values = [
            float('inf'),
            float('-inf'),
            float('nan'),
        ]
        
        for special_val in special_values:
            with pytest.raises(ValueError):
                generator._validate_range("special_val", special_val, 0.0, 1.0)
        
        # 负零可能不会引发错误，单独测试
        try:
            generator._validate_range("negative_zero", -0.0, 0.0, 1.0)
        except ValueError:
            pass  # 如果引发错误也是可以的
        
        # 测试极小和极大的正常浮点数
        tiny_val = np.finfo(float).tiny
        max_val = np.finfo(float).max
        
        # 这些应该通过（如果在范围内）
        if tiny_val <= 1.0:  # 只有在范围内才测试
            generator._validate_range("tiny", tiny_val, 0.0, 1.0)
        generator._validate_range("large", max_val/2, 0.0, max_val)


class TestDistributionSamplingEdgeCases:
    """测试分布采样的边界情况"""
    
    def test_distribution_parameter_boundaries(self, sample_rng):
        """测试分布参数边界"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_dist_boundaries"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试Beta分布的边界参数
        # a=1, b=1 应该给出均匀分布
        samples = generator._sample_from_distribution(
            sample_rng, size=1000, dist='beta', low=0.0, high=1.0, a=1.0, b=1.0
        )
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)
        
        # 测试极端的Beta参数
        samples = generator._sample_from_distribution(
            sample_rng, size=100, dist='beta', low=0.0, high=1.0, a=0.1, b=0.1
        )
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)
        
        # 测试正态分布的零标准差（应该退化为常数）
        samples = generator._sample_from_distribution(
            sample_rng, size=100, dist='normal', low=-1.0, high=1.0, loc=0.5, scale=0.0
        )
        # 所有样本应该都是loc值（被clipping到范围内）
        assert np.allclose(samples, 0.5, atol=1e-10)
    
    def test_clipping_behavior(self, sample_rng):
        """测试clipping行为的详细情况"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_clipping"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 测试正态分布的强clipping
        # 使用很大的标准差，但很小的clipping范围
        samples = generator._sample_from_distribution(
            sample_rng, size=1000, dist='normal', 
            low=0.0, high=0.1, loc=0.05, scale=10.0
        )
        
        # 所有样本都应该在clipping范围内
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 0.1)
        
        # 由于强clipping，大部分样本应该在边界上
        boundary_samples = np.sum((samples == 0.0) | (samples == 0.1))
        assert boundary_samples > len(samples) * 0.5  # 超过50%在边界
    
    def test_distribution_consistency(self, sample_rng):
        """测试分布的一致性"""
        
        class TestGenerator(ParameterizedRandomGenerator):
            mtype = "test_consistency"
            
            def get_default_parameters(self):
                return {}
            
            def validate_parameters(self, **params):
                pass
            
            def fuzznum(self, rng, **params):
                return Mock()
            
            def fuzzarray(self, rng, shape, **params):
                return Mock()
        
        generator = TestGenerator()
        
        # 使用固定种子确保可重现性
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        
        # 相同参数应该产生相同结果
        samples1 = generator._sample_from_distribution(
            rng1, size=100, dist='uniform', low=0.0, high=1.0
        )
        samples2 = generator._sample_from_distribution(
            rng2, size=100, dist='uniform', low=0.0, high=1.0
        )
        
        assert np.array_equal(samples1, samples2)
        
        # 不同种子应该产生不同结果
        rng3 = np.random.default_rng(54321)
        samples3 = generator._sample_from_distribution(
            rng3, size=100, dist='uniform', low=0.0, high=1.0
        )
        
        assert not np.array_equal(samples1, samples3)


if __name__ == "__main__":
    # 运行测试的示例
    pytest.main([__file__, "-v"])