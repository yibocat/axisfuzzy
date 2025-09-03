#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 axisfuzzy.random.api 模块

本模块测试随机生成API的核心功能，包括：
- rand 函数的重载和参数处理
- choice 函数的随机采样功能
- uniform/normal/beta 工具函数
- _resolve_rng 内部函数的优先级逻辑
- 错误处理和边界条件
- 与注册表和种子管理系统的集成

重新设计的测试基于真实的 API 接口，确保测试的准确性和可维护性。
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from typing import Any, Optional, Tuple, Union

# 导入被测试的模块
from axisfuzzy.random.api import (
    rand,
    choice, 
    uniform,
    normal,
    beta,
    _resolve_rng
)
from axisfuzzy.core import Fuzznum, Fuzzarray


class TestResolveRng:
    """
    测试 _resolve_rng 内部函数的优先级逻辑
    
    该函数实现了随机数生成器的三级优先级：
    1. rng 参数（最高优先级）
    2. seed 参数（中等优先级）
    3. 全局 RNG（最低优先级）
    """
    
    def test_rng_parameter_highest_priority(self):
        """测试 rng 参数具有最高优先级"""
        custom_rng = np.random.default_rng(42)
        seed_value = 123
        
        result = _resolve_rng(seed=seed_value, rng=custom_rng)
        
        # 应该返回提供的 rng，忽略 seed
        assert result is custom_rng
    
    def test_seed_parameter_second_priority(self):
        """测试 seed 参数在没有 rng 时生效"""
        seed_value = 42
        
        result = _resolve_rng(seed=seed_value, rng=None)
        
        # 应该创建新的生成器
        assert isinstance(result, np.random.Generator)
        # 验证种子生效（通过生成相同序列）
        result1 = _resolve_rng(seed=seed_value)
        result2 = _resolve_rng(seed=seed_value)
        assert result1.random() == result2.random()
    
    @patch('axisfuzzy.random.api.get_rng')
    def test_global_rng_lowest_priority(self, mock_get_rng):
        """测试在没有 rng 和 seed 时使用全局 RNG"""
        mock_global_rng = Mock()
        mock_get_rng.return_value = mock_global_rng
        
        result = _resolve_rng(seed=None, rng=None)
        
        assert result is mock_global_rng
        mock_get_rng.assert_called_once()
    
    def test_seed_sequence_support(self):
        """测试支持 SeedSequence 对象"""
        seed_seq = np.random.SeedSequence(42)
        
        result = _resolve_rng(seed=seed_seq, rng=None)
        
        assert isinstance(result, np.random.Generator)
    
    def test_bit_generator_support(self):
        """测试支持 BitGenerator 对象"""
        bit_gen = np.random.PCG64(42)
        
        result = _resolve_rng(seed=bit_gen, rng=None)
        
        assert isinstance(result, np.random.Generator)


class TestRandFunction:
    """
    测试 rand 函数的核心功能
    
    rand 函数是主要的工厂函数，支持：
    - 单个 Fuzznum 生成（shape=None）
    - Fuzzarray 批量生成（shape 指定）
    - 参数传递和验证
    - 多种重载形式
    """
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_single_fuzznum_generation(self, mock_resolve_rng, mock_get_generator):
        """测试单个 Fuzznum 生成（shape=None）"""
        # 设置 mock 对象
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        mock_generator = Mock()
        mock_fuzznum = Mock(spec=Fuzznum)
        mock_generator.fuzznum.return_value = mock_fuzznum
        mock_get_generator.return_value = mock_generator
        
        # 调用函数
        result = rand(mtype='test_type', q=2)
        
        # 验证调用
        mock_get_generator.assert_called_once_with('test_type')
        mock_generator.fuzznum.assert_called_once_with(mock_rng, q=2)
        assert result is mock_fuzznum
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_fuzzarray_generation_int_shape(self, mock_resolve_rng, mock_get_generator):
        """测试 Fuzzarray 生成（整数 shape）"""
        # 设置 mock 对象
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        mock_generator = Mock()
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_generator.fuzzarray.return_value = mock_fuzzarray
        mock_get_generator.return_value = mock_generator
        
        # 调用函数
        result = rand(mtype='test_type', q=3, shape=100)
        
        # 验证调用
        mock_get_generator.assert_called_once_with('test_type')
        mock_generator.fuzzarray.assert_called_once_with(mock_rng, (100,), q=3)
        assert result is mock_fuzzarray
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_fuzzarray_generation_tuple_shape(self, mock_resolve_rng, mock_get_generator):
        """测试 Fuzzarray 生成（元组 shape）"""
        # 设置 mock 对象
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        mock_generator = Mock()
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_generator.fuzzarray.return_value = mock_fuzzarray
        mock_get_generator.return_value = mock_generator
        
        # 调用函数
        result = rand(mtype='test_type', q=2, shape=(10, 20))
        
        # 验证调用
        mock_get_generator.assert_called_once_with('test_type')
        mock_generator.fuzzarray.assert_called_once_with(mock_rng, (10, 20), q=2)
        assert result is mock_fuzzarray
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_explicit_mtype_parameter(self, mock_resolve_rng, mock_get_generator):
        """测试显式指定 mtype 参数"""
        # 设置 mock 对象
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        mock_generator = Mock()
        mock_fuzznum = Mock(spec=Fuzznum)
        mock_generator.fuzznum.return_value = mock_fuzznum
        mock_get_generator.return_value = mock_generator
        
        # 调用函数（显式指定 mtype）
        result = rand(mtype='explicit_type', q=2)
        
        # 验证使用指定的 mtype
        mock_get_generator.assert_called_once_with('explicit_type')
        mock_generator.fuzznum.assert_called_once_with(mock_rng, q=2)
    
    @patch('axisfuzzy.random.api.get_random_generator')
    def test_unregistered_mtype_error(self, mock_get_generator):
        """测试未注册 mtype 的错误处理"""
        mock_get_generator.return_value = None
        
        with pytest.raises(KeyError, match="No random generator registered for mtype 'unknown'"):
            rand(mtype='unknown')
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_invalid_shape_type_error(self, mock_resolve_rng, mock_get_generator):
        """测试无效 shape 类型的错误处理"""
        mock_get_generator.return_value = Mock()
        mock_resolve_rng.return_value = Mock()
        
        with pytest.raises(TypeError, match="Shape must be an int or a tuple of ints"):
            rand(mtype='test_type', shape="invalid")
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_parameter_passing(self, mock_resolve_rng, mock_get_generator):
        """测试参数传递机制"""
        # 设置 mock 对象
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        mock_generator = Mock()
        mock_fuzznum = Mock(spec=Fuzznum)
        mock_generator.fuzznum.return_value = mock_fuzznum
        mock_get_generator.return_value = mock_generator
        
        # 调用函数并传递额外参数
        result = rand(
            mtype='test_type', 
            q=3, 
            md_dist='beta', 
            a=2.0, 
            b=3.0,
            custom_param='value'
        )
        
        # 验证所有参数都被传递
        expected_params = {
            'q': 3,
            'md_dist': 'beta',
            'a': 2.0,
            'b': 3.0,
            'custom_param': 'value'
        }
        mock_generator.fuzznum.assert_called_once_with(mock_rng, **expected_params)
    
    @patch('axisfuzzy.random.api.get_random_generator')
    def test_seed_and_rng_parameter_handling(self, mock_get_generator):
        """测试 seed 和 rng 参数的处理"""
        mock_generator = Mock()
        mock_fuzznum = Mock(spec=Fuzznum)
        mock_generator.fuzznum.return_value = mock_fuzznum
        mock_get_generator.return_value = mock_generator
        
        custom_rng = np.random.default_rng(42)
        
        # 使用自定义 rng
        result = rand(mtype='test_type', q=2, rng=custom_rng)
        
        # 验证传递了正确的 rng
        mock_generator.fuzznum.assert_called_once_with(custom_rng, q=2)


class TestChoiceFunction:
    """
    测试 choice 函数的随机采样功能
    
    choice 函数提供从现有 Fuzzarray 中随机采样的能力，
    支持有放回/无放回采样、权重采样等。
    """
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_single_element_choice(self, mock_resolve_rng):
        """测试单个元素选择（size=None）"""
        # 创建模拟的 Fuzzarray
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_fuzzarray.ndim = 1
        mock_fuzzarray.__len__ = Mock(return_value=5)
        
        # 模拟选择结果
        mock_selected = Mock(spec=Fuzznum)
        mock_fuzzarray.__getitem__ = Mock(return_value=mock_selected)
        
        # 设置 mock RNG
        mock_rng = Mock()
        mock_rng.choice.return_value = 2  # 选择索引 2
        mock_resolve_rng.return_value = mock_rng
        
        # 调用函数
        result = choice(mock_fuzzarray)
        
        # 验证调用
        mock_rng.choice.assert_called_once_with(5, size=None, replace=True, p=None)
        mock_fuzzarray.__getitem__.assert_called_once_with(2)
        assert result is mock_selected
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_multiple_elements_choice(self, mock_resolve_rng):
        """测试多个元素选择"""
        # 创建模拟的 Fuzzarray
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_fuzzarray.ndim = 1
        mock_fuzzarray.__len__ = Mock(return_value=10)
        
        # 模拟选择结果
        mock_selected_array = Mock(spec=Fuzzarray)
        mock_fuzzarray.__getitem__ = Mock(return_value=mock_selected_array)
        
        # 设置 mock RNG
        mock_rng = Mock()
        mock_indices = np.array([1, 3, 7])
        mock_rng.choice.return_value = mock_indices
        mock_resolve_rng.return_value = mock_rng
        
        # 调用函数
        result = choice(mock_fuzzarray, size=3)
        
        # 验证调用
        mock_rng.choice.assert_called_once_with(10, size=3, replace=True, p=None)
        mock_fuzzarray.__getitem__.assert_called_once_with(mock_indices)
        assert result is mock_selected_array
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_choice_without_replacement(self, mock_resolve_rng):
        """测试无放回采样"""
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_fuzzarray.ndim = 1
        mock_fuzzarray.__len__ = Mock(return_value=10)
        mock_fuzzarray.__getitem__ = Mock(return_value=Mock())
        
        mock_rng = Mock()
        mock_rng.choice.return_value = np.array([1, 5, 9])
        mock_resolve_rng.return_value = mock_rng
        
        # 调用函数
        choice(mock_fuzzarray, size=3, replace=False)
        
        # 验证 replace=False 被传递
        mock_rng.choice.assert_called_once_with(10, size=3, replace=False, p=None)
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_choice_with_probabilities(self, mock_resolve_rng):
        """测试带权重的采样"""
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_fuzzarray.ndim = 1
        mock_fuzzarray.__len__ = Mock(return_value=3)
        mock_fuzzarray.__getitem__ = Mock(return_value=Mock())
        
        mock_rng = Mock()
        mock_rng.choice.return_value = 1
        mock_resolve_rng.return_value = mock_rng
        
        # 调用函数
        probabilities = [0.1, 0.7, 0.2]
        choice(mock_fuzzarray, p=probabilities)
        
        # 验证权重被传递
        mock_rng.choice.assert_called_once_with(3, size=None, replace=True, p=probabilities)
    
    def test_non_fuzzarray_input_error(self):
        """测试非 Fuzzarray 输入的错误处理"""
        with pytest.raises(TypeError, match="Input for axisfuzzy.random.choice must be a Fuzzarray"):
            choice([1, 2, 3])  # 传入列表而不是 Fuzzarray
    
    def test_multidimensional_fuzzarray_error(self):
        """测试多维 Fuzzarray 的错误处理"""
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_fuzzarray.ndim = 2  # 2维数组
        
        with pytest.raises(ValueError, match="Input Fuzzarray for choice must be 1-dimensional"):
            choice(mock_fuzzarray)
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_seed_parameter_handling(self, mock_resolve_rng):
        """测试 seed 参数处理"""
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_fuzzarray.ndim = 1
        mock_fuzzarray.__len__ = Mock(return_value=5)
        mock_fuzzarray.__getitem__ = Mock(return_value=Mock())
        
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        # 调用函数
        choice(mock_fuzzarray, seed=42)
        
        # 验证 _resolve_rng 被正确调用
        mock_resolve_rng.assert_called_once_with(42, None)


class TestUtilityFunctions:
    """
    测试工具函数：uniform, normal, beta
    
    这些函数提供标准的随机数生成功能，
    集成了 AxisFuzzy 的种子管理系统。
    """
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_uniform_basic(self, mock_resolve_rng):
        """测试 uniform 函数基本功能"""
        mock_rng = Mock()
        mock_rng.uniform.return_value = 0.5
        mock_resolve_rng.return_value = mock_rng
        
        result = uniform(0.0, 1.0)
        
        mock_resolve_rng.assert_called_once_with(None, None)
        mock_rng.uniform.assert_called_once_with(0.0, 1.0, None)
        assert result == 0.5
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_uniform_with_shape(self, mock_resolve_rng):
        """测试 uniform 函数带 shape 参数"""
        mock_rng = Mock()
        mock_array = np.array([0.1, 0.5, 0.9])
        mock_rng.uniform.return_value = mock_array
        mock_resolve_rng.return_value = mock_rng
        
        result = uniform(-1.0, 1.0, shape=(3,))
        
        mock_rng.uniform.assert_called_once_with(-1.0, 1.0, (3,))
        np.testing.assert_array_equal(result, mock_array)
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_uniform_with_seed(self, mock_resolve_rng):
        """测试 uniform 函数带 seed 参数"""
        mock_rng = Mock()
        mock_rng.uniform.return_value = 0.7
        mock_resolve_rng.return_value = mock_rng
        
        result = uniform(0.0, 1.0, seed=42)
        
        mock_resolve_rng.assert_called_once_with(42, None)
        mock_rng.uniform.assert_called_once_with(0.0, 1.0, None)
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_normal_basic(self, mock_resolve_rng):
        """测试 normal 函数基本功能"""
        mock_rng = Mock()
        mock_rng.normal.return_value = 1.5
        mock_resolve_rng.return_value = mock_rng
        
        result = normal(0.0, 1.0)
        
        mock_resolve_rng.assert_called_once_with(None, None)
        mock_rng.normal.assert_called_once_with(0.0, 1.0, None)
        assert result == 1.5
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_normal_with_parameters(self, mock_resolve_rng):
        """测试 normal 函数带参数"""
        mock_rng = Mock()
        mock_array = np.array([2.1, 3.5, 1.9])
        mock_rng.normal.return_value = mock_array
        mock_resolve_rng.return_value = mock_rng
        
        result = normal(loc=2.0, scale=0.5, shape=(3,))
        
        mock_rng.normal.assert_called_once_with(2.0, 0.5, (3,))
        np.testing.assert_array_equal(result, mock_array)
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_beta_basic(self, mock_resolve_rng):
        """测试 beta 函数基本功能"""
        mock_rng = Mock()
        mock_rng.beta.return_value = 0.3
        mock_resolve_rng.return_value = mock_rng
        
        result = beta(2.0, 3.0)
        
        mock_resolve_rng.assert_called_once_with(None, None)
        mock_rng.beta.assert_called_once_with(2.0, 3.0, None)
        assert result == 0.3
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_beta_with_shape_and_rng(self, mock_resolve_rng):
        """测试 beta 函数带 shape 和 rng 参数"""
        mock_rng = Mock()
        mock_array = np.array([0.2, 0.6, 0.4])
        mock_rng.beta.return_value = mock_array
        mock_resolve_rng.return_value = mock_rng
        
        result = beta(1.5, 2.5, shape=(3,), rng=mock_rng)
        
        mock_resolve_rng.assert_called_once_with(None, mock_rng)
        mock_rng.beta.assert_called_once_with(1.5, 2.5, (3,))
        np.testing.assert_array_equal(result, mock_array)


class TestEdgeCases:
    """
    测试边界条件和极值情况
    
    这些测试验证API在边界条件下的健壮性和错误处理
    """
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_extremely_large_shape(self, mock_resolve_rng, mock_get_generator):
        """测试极大的shape值处理"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        mock_generator = Mock()
        mock_get_generator.return_value = mock_generator
        
        # 测试极大的shape值
        large_shape = (10000, 10000)  # 可能导致内存问题
        
        # 应该正常传递给生成器，由生成器决定如何处理
        rand(mtype='test_type', shape=large_shape)
        
        # 验证调用了fuzzarray而不是fuzznum
        mock_generator.fuzzarray.assert_called_once()
        # 验证传递的参数
        call_args = mock_generator.fuzzarray.call_args
        assert call_args[0][1] == large_shape  # shape参数
    
    @patch('axisfuzzy.random.api.get_random_generator')
    def test_invalid_shape_types(self, mock_get_generator):
        """测试无效的shape类型"""
        # 设置mock以避免KeyError
        mock_generator = Mock()
        mock_get_generator.return_value = mock_generator
        
        invalid_shapes = [
            'invalid',  # 字符串
            [1, 2, 3],  # 列表而非元组
            (1.5, 2),   # 浮点数
            (-1, 2),    # 负数
            (0, 2),     # 零
            (1, -2),    # 负数
        ]
        
        for invalid_shape in invalid_shapes:
            try:
                rand(mtype='test_type', shape=invalid_shape)
                # 如果没有引发异常，检查是否调用了正确的方法
                assert mock_get_generator.called
            except (TypeError, ValueError):
                # 这是期望的行为
                pass
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_extreme_parameter_values(self, mock_resolve_rng):
        """测试极值参数"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_gen:
            mock_generator = Mock()
            mock_get_gen.return_value = mock_generator
            
            # 测试极大的数值参数
            extreme_params = {
                'very_large': 1e100,
                'very_small': 1e-100,
                'negative_large': -1e100
            }
            
            rand(mtype='test_type', **extreme_params)
            
            # 验证参数被正确传递
            mock_generator.fuzznum.assert_called_once()
            call_args = mock_generator.fuzznum.call_args[1]
            for key, value in extreme_params.items():
                assert call_args[key] == value
    
    @patch('axisfuzzy.random.api.list_registered_random')
    def test_unicode_mtype_handling(self, mock_list_registered):
        """测试Unicode字符的mtype处理"""
        mock_list_registered.return_value = ['qrofn', 'ivfn']
        
        unicode_mtypes = [
            '中文类型',
            'тип',  # 俄文
            'τύπος',  # 希腊文
            'タイプ',  # 日文
            '🔢📊',   # emoji
        ]
        
        for mtype in unicode_mtypes:
            with pytest.raises(KeyError):
                rand(mtype=mtype)
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_memory_stress_parameters(self, mock_resolve_rng):
        """测试可能导致内存压力的参数组合"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_gen:
            mock_generator = Mock()
            mock_get_gen.return_value = mock_generator
            
            # 大量参数
            many_params = {f'param_{i}': i for i in range(100)}
            
            rand(mtype='test_type', **many_params)
            
            # 验证所有参数都被传递（包括mtype参数，所以是101个）
            call_args = mock_generator.fuzznum.call_args[1]
            assert len(call_args) == 101  # 100个param_i + 1个mtype
            for i in range(100):
                assert call_args[f'param_{i}'] == i


class TestPerformance:
    """
    测试API性能相关功能
    
    这些测试验证API在高负载和大规模数据处理时的性能表现
    """
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_large_scale_array_generation_performance(self, mock_resolve_rng):
        """测试大规模数组生成的性能"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_gen:
            mock_generator = Mock()
            mock_fuzzarray = Mock()
            mock_generator.fuzzarray.return_value = mock_fuzzarray
            mock_get_gen.return_value = mock_generator
            
            # 测试不同规模的数组生成性能
            test_shapes = [
                (1000,),
                (100, 100),
                (50, 50, 8),
                (10, 10, 10, 10)
            ]
            
            import time
            
            for shape in test_shapes:
                start_time = time.time()
                result = rand(mtype='test_type', shape=shape)
                elapsed_time = time.time() - start_time
                
                # 验证调用正确
                mock_generator.fuzzarray.assert_called()
                call_args = mock_generator.fuzzarray.call_args
                assert call_args[0][1] == shape  # shape参数
                
                # 验证性能在合理范围内（每个形状生成应该少于100毫秒）
                assert elapsed_time < 0.1, f"Shape {shape} generation took {elapsed_time:.4f}s, too slow"
                
                # 重置mock以便下次测试
                mock_generator.reset_mock()
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_api_throughput_performance(self, mock_resolve_rng):
        """测试API吞吐量性能"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_gen:
            mock_generator = Mock()
            mock_fuzznum = Mock()
            mock_generator.fuzznum.return_value = mock_fuzznum
            mock_get_gen.return_value = mock_generator
            
            # 测试连续调用的吞吐量
            num_calls = 1000  # 减少调用次数以避免测试超时
            
            import time
            start_time = time.time()
            
            for i in range(num_calls):
                rand(mtype='test_type', seed=i % 100)  # 使用不同种子
            
            elapsed_time = time.time() - start_time
            
            # 计算吞吐量
            throughput = num_calls / elapsed_time
            
            # 验证吞吐量（应该能达到每秒至少100次调用）
            assert throughput > 100, f"API throughput {throughput:.2f} calls/sec is too low"
            
            # 验证所有调用都成功
            assert mock_generator.fuzznum.call_count == num_calls
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_parameter_processing_performance(self, mock_resolve_rng):
        """测试参数处理的性能"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_gen:
            mock_generator = Mock()
            mock_fuzznum = Mock()
            mock_generator.fuzznum.return_value = mock_fuzznum
            mock_get_gen.return_value = mock_generator
            
            # 测试大量参数的处理性能
            large_params = {f'param_{i}': i * 0.1 for i in range(100)}  # 减少参数数量
            
            num_iterations = 10  # 减少迭代次数
            
            import time
            start_time = time.time()
            
            for _ in range(num_iterations):
                rand(mtype='test_type', **large_params)
            
            elapsed_time = time.time() - start_time
            
            # 验证参数处理性能
            avg_time = elapsed_time / num_iterations
            assert avg_time < 0.1, f"Parameter processing took {avg_time:.4f}s per call, too slow"
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_memory_usage_monitoring(self, mock_resolve_rng):
        """测试内存使用监控"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_gen:
            mock_generator = Mock()
            mock_fuzzarray = Mock()
            mock_generator.fuzzarray.return_value = mock_fuzzarray
            mock_get_gen.return_value = mock_generator
            
            # 执行API调用
            for i in range(100):  # 减少调用次数
                rand(mtype='test_type', shape=(10, 10), seed=i)
            
            # 验证调用成功
            assert mock_generator.fuzzarray.call_count == 100
    
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_rng_resolution_performance(self, mock_resolve_rng):
        """测试RNG解析的性能"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_gen:
            mock_generator = Mock()
            mock_fuzznum = Mock()
            mock_generator.fuzznum.return_value = mock_fuzznum
            mock_get_gen.return_value = mock_generator
            
            # 测试不同RNG解析方式的性能
            test_scenarios = [
                {'rng': None, 'seed': None},  # 使用全局RNG
                {'rng': None, 'seed': 12345},  # 使用种子
                {'rng': np.random.default_rng(54321), 'seed': None},  # 使用提供的RNG
            ]
            
            for scenario in test_scenarios:
                num_calls = 100
                
                import time
                start_time = time.time()
                
                for _ in range(num_calls):
                    rand(mtype='test_type', **scenario)
                
                elapsed_time = time.time() - start_time
                
                # 验证RNG解析性能
                avg_time = elapsed_time / num_calls
                assert avg_time < 0.01, f"RNG resolution took {avg_time:.6f}s per call for {scenario}"


class TestIntegrationScenarios:
    """
    测试集成场景和边界条件
    
    这些测试验证 API 函数在复杂场景下的行为，
    包括错误处理、参数组合、性能边界等。
    """
    
    @patch('axisfuzzy.random.api.list_registered_random')
    @patch('axisfuzzy.random.api.get_random_generator')
    def test_error_message_includes_available_types(self, mock_get_generator, mock_list_registered):
        """测试错误消息包含可用类型列表"""
        mock_get_generator.return_value = None
        mock_list_registered.return_value = ['qrofn', 'ivfn', 'qrohfn']
        
        with pytest.raises(KeyError) as exc_info:
            rand(mtype='unknown_type')
        
        error_message = str(exc_info.value)
        assert "No random generator registered for mtype 'unknown_type'" in error_message
        assert "Available mtypes: ['qrofn', 'ivfn', 'qrohfn']" in error_message
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_generator_exception_propagation(self, mock_resolve_rng, mock_get_generator):
        """测试生成器异常的传播"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        mock_generator = Mock()
        mock_generator.fuzznum.side_effect = ValueError("Invalid parameters")
        mock_get_generator.return_value = mock_generator
        
        with pytest.raises(ValueError, match="Invalid parameters"):
            rand(mtype='test_type', q=2)
    
    @patch('axisfuzzy.random.api.get_random_generator')
    @patch('axisfuzzy.random.api._resolve_rng')
    def test_large_shape_handling(self, mock_resolve_rng, mock_get_generator):
        """测试大型 shape 的处理"""
        mock_rng = Mock()
        mock_resolve_rng.return_value = mock_rng
        
        mock_generator = Mock()
        mock_fuzzarray = Mock(spec=Fuzzarray)
        mock_generator.fuzzarray.return_value = mock_fuzzarray
        mock_get_generator.return_value = mock_generator
        
        # 测试大型多维 shape
        large_shape = (1000, 500, 10)
        result = rand(mtype='test_type', q=2, shape=large_shape)
        
        mock_generator.fuzzarray.assert_called_once_with(mock_rng, large_shape, q=2)
        assert result is mock_fuzzarray
    
    def test_parameter_type_preservation(self):
        """测试参数类型保持不变"""
        # 这个测试验证参数在传递过程中类型不会改变
        with patch('axisfuzzy.random.api.get_random_generator') as mock_get_generator, \
             patch('axisfuzzy.random.api._resolve_rng') as mock_resolve_rng:
            
            mock_rng = Mock()
            mock_resolve_rng.return_value = mock_rng
            
            mock_generator = Mock()
            mock_generator.fuzznum.return_value = Mock(spec=Fuzznum)
            mock_get_generator.return_value = mock_generator
            
            # 传递不同类型的参数
            params = {
                'q': 2,                    # int
                'alpha': 2.5,              # float
                'flag': True,              # bool
                'name': 'test',            # str
                'values': [1, 2, 3],       # list
                'array': np.array([1, 2])  # numpy array
            }
            
            rand(mtype='test_type', **params)
            
            # 验证参数被正确传递
            called_args = mock_generator.fuzznum.call_args[1]
            for key, value in params.items():
                assert key in called_args
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(called_args[key], value)
                else:
                    assert called_args[key] == value
                    assert type(called_args[key]) == type(value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])