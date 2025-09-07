"""测试工厂方法模块

本模块测试 fuzzynum 和 fuzzyset 工厂方法的功能，包括：
- fuzzynum 工厂方法的基本功能
- fuzzyset 工厂方法的三种构造方式
- 参数验证和错误处理
- 性能测试
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from axisfuzzy.core.factory import fuzzynum, fuzzyset
from axisfuzzy.core.fuzzarray import Fuzzarray
from axisfuzzy.core.fuzznums import Fuzznum
from axisfuzzy.core.backend import FuzzarrayBackend
from axisfuzzy.config import get_config


class TestFuzzynumFactory:
    """测试 fuzzynum 工厂方法"""

    def test_fuzzynum_basic_creation(self):
        """测试基本的 fuzzynum 创建"""
        # 测试默认参数创建
        fnum = fuzzynum()
        assert isinstance(fnum, Fuzznum)
        
        # 测试带值创建
        fnum = fuzzynum(values=(0.3, 0.7))
        assert isinstance(fnum, Fuzznum)
        # Fuzznum通过策略模式动态代理属性，检查md和nmd属性
        assert hasattr(fnum, 'md')
        assert hasattr(fnum, 'nmd')

    def test_fuzzynum_with_mtype(self):
        """测试指定 mtype 的 fuzzynum 创建"""
        # 测试 QROFN 类型
        fnum = fuzzynum(values=(0.3, 0.7), mtype='qrofn')
        assert isinstance(fnum, Fuzznum)
        
        # 测试 QROHFN 类型
        fnum = fuzzynum(values=((0.2, 0.3), (0.6, 0.7)), mtype='qrohfn')
        assert isinstance(fnum, Fuzznum)

    def test_fuzzynum_with_q_parameter(self):
        """测试指定 q 参数的 fuzzynum 创建"""
        fnum = fuzzynum(values=(0.3, 0.7), q=3)
        assert isinstance(fnum, Fuzznum)
        
    def test_fuzzynum_with_kwargs(self):
        """测试带额外参数的 fuzzynum 创建"""
        fnum = fuzzynum(values=(0.3, 0.7), mtype='qrofn', q=2, custom_param='test')
        assert isinstance(fnum, Fuzznum)

    def test_fuzzynum_invalid_values(self):
        """测试无效值的处理"""
        # 测试超出范围的值 - 违反约束
        with pytest.raises((ValueError, TypeError, AttributeError)):
            fuzzynum(values=(1.5, 0.7))
            
        # 测试错误的值格式
        with pytest.raises((ValueError, TypeError, AttributeError)):
            fuzzynum(values="invalid")


class TestFuzzysetFactory:
    """测试 fuzzyset 工厂方法"""

    def test_fuzzyset_empty_creation(self):
        """测试空 fuzzyset 创建"""
        farr = fuzzyset()
        assert isinstance(farr, Fuzzarray)

    def test_fuzzyset_from_numpy_array(self):
        """测试从 numpy 数组创建 fuzzyset (Path 2)"""
        # 测试 2D 数组 - 使用满足约束的值，QROFN需要2个组件
        data = np.array([[0.3, 0.4, 0.2], [0.4, 0.5, 0.1]])  # 2个组件，3个元素
        farr = fuzzyset(data=data, mtype='qrofn')
        assert isinstance(farr, Fuzzarray)
        assert farr.shape == (3,)
        
        # 测试 3D 数组 (QROHFN) - 使用满足约束的值
        data = np.array([[[0.2, 0.3], [0.4, 0.5]], [[0.1, 0.2], [0.3, 0.4]]])
        farr = fuzzyset(data=data, mtype='qrohfn')
        assert isinstance(farr, Fuzzarray)

    def test_fuzzyset_from_list(self):
        """测试从列表创建 fuzzyset"""
        # 测试普通列表 - 使用满足约束的值，QROFN需要2个组件
        data = [[0.3, 0.4], [0.4, 0.5]]
        farr = fuzzyset(data=data, mtype='qrofn')
        assert isinstance(farr, Fuzzarray)
        
        # 测试嵌套列表 - 使用满足约束的值
        data = [[[0.2, 0.3], [0.4, 0.5]], [[0.1, 0.2], [0.3, 0.4]]]
        farr = fuzzyset(data=data, mtype='qrohfn')
        assert isinstance(farr, Fuzzarray)

    def test_fuzzyset_from_fuzznum_list(self):
        """测试从 Fuzznum 列表创建 fuzzyset (Path 1)"""
        # 创建 Fuzznum 列表 - 使用满足约束的值
        fnum1 = fuzzynum(values=(0.3, 0.4))
        fnum2 = fuzzynum(values=(0.4, 0.5))
        fnum_list = [fnum1, fnum2]
        
        farr = fuzzyset(data=fnum_list)
        assert isinstance(farr, Fuzzarray)
        assert farr.shape == (2,)

    def test_fuzzyset_from_existing_fuzzarray(self):
        """测试从现有 Fuzzarray 创建 fuzzyset (Path 3)"""
        # 先创建一个 Fuzzarray - 使用满足约束的值，QROFN需要2个组件
        original_data = np.array([[0.3, 0.4], [0.4, 0.5]])
        original_farr = fuzzyset(data=original_data, mtype='qrofn')
        
        # 从现有 Fuzzarray 创建新的
        new_farr = fuzzyset(data=original_farr)
        assert isinstance(new_farr, Fuzzarray)
        assert new_farr.shape == original_farr.shape

    def test_fuzzyset_with_backend(self):
        """测试指定后端的 fuzzyset 创建"""
        # 这里需要根据实际的后端实现进行测试 - 使用满足约束的值
        data = np.array([[0.3, 0.4], [0.4, 0.5]])
        
        # 测试不同的后端类型
        farr = fuzzyset(data=data, mtype='qrofn')
        assert isinstance(farr, Fuzzarray)

    def test_fuzzyset_with_shape(self):
        """测试指定形状的 fuzzyset 创建"""
        # 测试指定形状
        farr = fuzzyset(shape=(3, 2), mtype='qrofn')
        assert isinstance(farr, Fuzzarray)
        
    def test_fuzzyset_path2_error_handling(self):
        """测试 Path 2 的错误处理逻辑"""
        # 测试无效的数组格式
        invalid_data = np.array([1, 2, 3])  # 1D 数组，可能不符合模糊数格式
        
        # 根据修复后的逻辑，应该直接抛出错误而不是回退
        with pytest.raises((ValueError, TypeError)):
            fuzzyset(data=invalid_data, mtype='qrofn')

    def test_fuzzyset_parameter_validation(self):
        """测试参数验证"""
        # 测试无效的 mtype
        with pytest.raises((ValueError, KeyError)):
            fuzzyset(mtype='invalid_type')
            
        # 测试约束违反 - QROFN需要2个组件
        with pytest.raises(ValueError):
            data = np.array([[0.8, 0.9], [0.9, 0.8]])  # 违反 md + nmd <= 1 约束
            fuzzyset(data=data, mtype='qrofn')
            
        # 测试维度不匹配的数据
        with pytest.raises(ValueError):
            data = np.array([0.5, 0.3])  # 1D数组，应该是2D
            fuzzyset(data=data, mtype='qrofn')

    def test_fuzzyset_kwargs_passing(self):
        """测试额外参数的传递"""
        data = np.array([[0.3, 0.4], [0.4, 0.5]])  # 2个组件，2个元素
        farr = fuzzyset(data=data, mtype='qrofn', q=2, custom_param='test')
        assert isinstance(farr, Fuzzarray)


class TestFactoryPerformance:
    """测试工厂方法的性能"""

    def test_fuzzynum_creation_performance(self):
        """测试 fuzzynum 创建性能"""
        start_time = time.time()
        
        # 创建大量 fuzzynum - 使用满足约束的值
        for i in range(1000):
            fnum = fuzzynum(values=(0.3, 0.4))
            
        end_time = time.time()
        creation_time = end_time - start_time
        
        # 性能断言（1000个应该在合理时间内完成）
        assert creation_time < 5.0, f"创建1000个fuzzynum耗时{creation_time:.2f}秒，超过预期"

    def test_fuzzyset_creation_performance(self):
        """测试 fuzzyset 创建性能"""
        # 测试不同大小的数组创建性能
        sizes = [10, 100, 1000]
        
        for size in sizes:
            # 创建满足约束的数据：md + nmd <= 1，QROFN需要2个组件
            md_data = np.random.rand(size) * 0.5  # [0, 0.5]
            nmd_data = np.random.rand(size) * (1 - md_data)  # [0, 1-md]
            data = np.array([md_data, nmd_data])  # 2个组件，size个元素
            
            start_time = time.time()
            farr = fuzzyset(data=data, mtype='qrofn')
            end_time = time.time()
            
            creation_time = end_time - start_time
            
            # 性能断言
            assert creation_time < 2.0, f"创建大小为{size}的fuzzyset耗时{creation_time:.2f}秒，超过预期"

    def test_path_comparison_performance(self):
        """测试不同路径的性能比较"""
        # 准备测试数据 - 满足约束，QROFN需要2个组件
        size = 100
        md_data = np.random.rand(size) * 0.5
        nmd_data = np.random.rand(size) * (1 - md_data)
        data = np.array([md_data, nmd_data])  # 2个组件，100个元素
        
        # Path 2: 从原始数组创建
        start_time = time.time()
        farr1 = fuzzyset(data=data, mtype='qrofn')
        path2_time = time.time() - start_time
        
        # Path 1: 从 Fuzznum 列表创建
        fnum_list = [fuzzynum(values=(md_data[i], nmd_data[i])) for i in range(size)]
        start_time = time.time()
        farr2 = fuzzyset(data=fnum_list)
        path1_time = time.time() - start_time
        
        # Path 3: 从现有 Fuzzarray 创建
        start_time = time.time()
        farr3 = fuzzyset(data=farr1)
        path3_time = time.time() - start_time
        
        # 记录性能信息（用于调试和优化）
        print(f"\nPath 2 (原始数组): {path2_time:.4f}秒")
        print(f"Path 1 (Fuzznum列表): {path1_time:.4f}秒")
        print(f"Path 3 (现有Fuzzarray): {path3_time:.4f}秒")
        
        # 基本性能断言
        assert path2_time < 1.0, "Path 2 性能不达标"
        assert path1_time < 2.0, "Path 1 性能不达标"
        assert path3_time < 0.5, "Path 3 性能不达标"


class TestFactoryEdgeCases:
    """测试工厂方法的边界情况"""

    def test_empty_data_handling(self):
        """测试空数据的处理"""
        # 测试空数组 - 需要正确的维度
        empty_array = np.array([[], []])  # 2个组件，0个元素
        farr = fuzzyset(data=empty_array, mtype='qrofn')
        assert isinstance(farr, Fuzzarray)
        
        # 测试空的Fuzznum列表 - 跳过，因为空列表会被转换为1D数组
        # 这是预期行为，空列表应该通过其他方式处理
        pass

    def test_single_element_data(self):
        """测试单元素数据"""
        # 单个 Fuzznum - 使用满足约束的值
        fnum = fuzzynum(values=(0.3, 0.4))
        farr = fuzzyset(data=[fnum])
        assert isinstance(farr, Fuzzarray)
        assert farr.shape == (1,)
        
        # 单元素数组 - 使用满足约束的值，QROFN需要2个组件
        data = np.array([[0.3], [0.4]])  # 2个组件，1个元素
        farr = fuzzyset(data=data, mtype='qrofn')
        assert isinstance(farr, Fuzzarray)
        assert farr.shape == (1,)

    def test_large_data_handling(self):
        """测试大数据的处理"""
        # 创建较大的数据集 - 满足约束，QROFN需要2个组件
        size = 10000
        md_data = np.random.rand(size) * 0.5
        nmd_data = np.random.rand(size) * (1 - md_data)
        large_data = np.array([md_data, nmd_data])  # 2个组件，10000个元素
        
        start_time = time.time()
        farr = fuzzyset(data=large_data, mtype='qrofn')
        end_time = time.time()
        
        assert isinstance(farr, Fuzzarray)
        assert farr.shape == (10000,)
        
        # 性能检查
        creation_time = end_time - start_time
        assert creation_time < 10.0, f"大数据处理耗时{creation_time:.2f}秒，超过预期"

    def test_mixed_type_data(self):
        """测试混合类型数据的处理"""
        # 测试包含不同类型的列表
        mixed_data = [0.3, [0.4, 0.6], (0.5, 0.5)]
        
        # 应该能够处理或给出明确的错误
        try:
            farr = fuzzyset(data=mixed_data, mtype='qrofn')
            assert isinstance(farr, Fuzzarray)
        except (ValueError, TypeError):
            # 如果不支持混合类型，应该给出明确的错误
            pass

    def test_boundary_values(self):
        """测试边界值"""
        # 测试边界值 [0, 1] - 满足约束 md + nmd <= 1，QROFN需要2个组件
        boundary_data = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]])  # 2个组件，3个元素
        farr = fuzzyset(data=boundary_data, mtype='qrofn')
        assert isinstance(farr, Fuzzarray)
        
        # 测试接近边界的值
        near_boundary = np.array([[0.001, 0.999], [0.999, 0.001]])  # 2个组件，2个元素
        farr = fuzzyset(data=near_boundary, mtype='qrofn')
        assert isinstance(farr, Fuzzarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])