"""测试后端约束检查模块

本模块测试 QROFN 和 QROHFN 后端的模糊约束检查功能，包括：
- QROFN 约束检查：md^q + nmd^q <= 1
- QROHFN 约束检查：max(md)^q + max(nmd)^q <= 1
- 性能测试
- 边界情况测试
"""

import pytest
import numpy as np
import time
from unittest.mock import patch

from axisfuzzy.fuzztype.qrofs.backend import QROFNBackend
from axisfuzzy.fuzztype.qrohfs.backend import QROHFNBackend
from axisfuzzy.config import get_config


class TestQROFNConstraints:
    """测试 QROFN 后端的约束检查"""

    def test_valid_qrofn_constraints(self):
        """测试有效的 QROFN 约束"""
        # 测试 q=2 的有效约束
        mds = np.array([0.3, 0.5, 0.7])
        nmds = np.array([0.4, 0.5, 0.6])
        q = 2
        
        # 验证约束：0.3^2 + 0.4^2 = 0.25 <= 1 ✓
        # 验证约束：0.5^2 + 0.5^2 = 0.5 <= 1 ✓
        # 验证约束：0.7^2 + 0.6^2 = 0.85 <= 1 ✓
        
        # 应该不抛出异常
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # 测试通过 from_arrays 创建
        backend = QROFNBackend.from_arrays(mds, nmds, q)
        assert isinstance(backend, QROFNBackend)
        assert backend.shape == (3,)

    def test_invalid_qrofn_constraints(self):
        """测试无效的 QROFN 约束"""
        # 测试违反约束的情况
        mds = np.array([0.8, 0.5])
        nmds = np.array([0.8, 0.5])  # 0.8^2 + 0.8^2 = 1.28 > 1
        q = 2
        
        with pytest.raises(ValueError, match="QROFN constraint violation"):
            QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
            
        with pytest.raises(ValueError, match="QROFN constraint violation"):
            QROFNBackend.from_arrays(mds, nmds, q)

    def test_qrofn_boundary_values(self):
        """测试 QROFN 边界值"""
        # 测试边界情况：md^q + nmd^q = 1
        mds = np.array([1.0, 0.0, 0.5])
        nmds = np.array([0.0, 1.0, np.sqrt(0.75)])  # 0.5^2 + (√0.75)^2 = 1
        q = 2
        
        # 应该通过（在误差范围内）
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        backend = QROFNBackend.from_arrays(mds, nmds, q)
        assert isinstance(backend, QROFNBackend)

    def test_qrofn_different_q_values(self):
        """测试不同 q 值的 QROFN 约束"""
        # q=1: md + nmd <= 1 (传统模糊集)
        mds = np.array([0.6, 0.3])
        nmds = np.array([0.4, 0.7])
        q = 1
        
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # q=3: md^3 + nmd^3 <= 1
        mds = np.array([0.8, 0.9])
        nmds = np.array([0.5, 0.4])  # 0.8^3 + 0.5^3 = 0.637 <= 1
        q = 3
        
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # 测试违反 q=3 约束的情况
        mds = np.array([0.9, 0.8])
        nmds = np.array([0.9, 0.8])  # 0.9^3 + 0.9^3 = 1.458 > 1
        
        with pytest.raises(ValueError, match="QROFN constraint violation"):
            QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)

    def test_qrofn_multidimensional_arrays(self):
        """测试多维数组的 QROFN 约束"""
        # 2D 数组
        mds = np.array([[0.3, 0.5], [0.4, 0.6]])
        nmds = np.array([[0.4, 0.5], [0.5, 0.3]])
        q = 2
        
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        backend = QROFNBackend.from_arrays(mds, nmds, q)
        assert backend.shape == (2, 2)
        
        # 3D 数组
        mds = np.random.rand(2, 3, 4) * 0.6  # 确保在有效范围内
        nmds = np.random.rand(2, 3, 4) * 0.6
        
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        backend = QROFNBackend.from_arrays(mds, nmds, q)
        assert backend.shape == (2, 3, 4)

    def test_qrofn_shape_mismatch(self):
        """测试形状不匹配的错误处理"""
        mds = np.array([0.3, 0.5])
        nmds = np.array([0.4, 0.5, 0.6])  # 不同形状
        q = 2
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            QROFNBackend.from_arrays(mds, nmds, q)


class TestQROHFNConstraints:
    """测试 QROHFN 后端的约束检查"""

    def test_valid_qrohfn_constraints(self):
        """测试有效的 QROHFN 约束"""
        # 创建有效的犹豫模糊数
        mds = np.array([
            [0.2, 0.3, 0.4],  # max = 0.4
            [0.1, 0.5],       # max = 0.5
            [0.6]             # max = 0.6
        ], dtype=object)
        
        nmds = np.array([
            [0.3, 0.4],       # max = 0.4, 0.4^2 + 0.4^2 = 0.32 <= 1 ✓
            [0.2, 0.3, 0.4],  # max = 0.4, 0.5^2 + 0.4^2 = 0.41 <= 1 ✓
            [0.2, 0.3]        # max = 0.3, 0.6^2 + 0.3^2 = 0.45 <= 1 ✓
        ], dtype=object)
        
        q = 2
        
        # 应该不抛出异常
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # 测试通过 from_arrays 创建
        backend = QROHFNBackend.from_arrays(mds, nmds, q)
        assert isinstance(backend, QROHFNBackend)
        assert backend.shape == (3,)

    def test_invalid_qrohfn_constraints(self):
        """测试无效的 QROHFN 约束"""
        # 创建违反约束的犹豫模糊数
        mds = np.array([
            [0.8, 0.9],       # max = 0.9
            [0.5, 0.6]
        ], dtype=object)
        
        nmds = np.array([
            [0.7, 0.8],       # max = 0.8, 0.9^2 + 0.8^2 = 1.45 > 1 ✗
            [0.3, 0.4]
        ], dtype=object)
        
        q = 2
        
        with pytest.raises(ValueError, match="QROHFN constraint violation"):
            QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
            
        with pytest.raises(ValueError, match="QROHFN constraint violation"):
            QROHFNBackend.from_arrays(mds, nmds, q)

    def test_qrohfn_negative_values(self):
        """测试 QROHFN 负值检查"""
        # 测试负的隶属度
        mds = np.array([
            [-0.1, 0.3],      # 包含负值
            [0.2, 0.4]
        ], dtype=object)
        
        nmds = np.array([
            [0.2, 0.3],
            [0.1, 0.2]
        ], dtype=object)
        
        q = 2
        
        with pytest.raises(ValueError, match="Negative membership degree"):
            QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # 测试负的非隶属度
        mds = np.array([
            [0.2, 0.3],
            [0.1, 0.4]
        ], dtype=object)
        
        nmds = np.array([
            [0.2, 0.3],
            [-0.1, 0.2]       # 包含负值
        ], dtype=object)
        
        with pytest.raises(ValueError, match="Negative non-membership degree"):
            QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)

    def test_qrohfn_empty_hesitant_sets(self):
        """测试空犹豫集的处理"""
        # 测试包含空集的情况
        mds = np.array([
            [0.2, 0.3],
            [],               # 空集
            [0.4, 0.5]
        ], dtype=object)
        
        nmds = np.array([
            [0.3, 0.4],
            [0.2],
            [0.1, 0.2]
        ], dtype=object)
        
        q = 2
        
        # 应该能够处理空集（跳过验证）
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)

    def test_qrohfn_single_element_hesitant_sets(self):
        """测试单元素犹豫集"""
        mds = np.array([
            [0.5],            # 单元素
            [0.3],
            [0.7]
        ], dtype=object)
        
        nmds = np.array([
            [0.4],
            [0.6],
            [0.2]
        ], dtype=object)
        
        q = 2
        
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        backend = QROHFNBackend.from_arrays(mds, nmds, q)
        assert isinstance(backend, QROHFNBackend)

    def test_qrohfn_different_q_values(self):
        """测试不同 q 值的 QROHFN 约束"""
        # q=1: max(md) + max(nmd) <= 1
        mds = np.array([
            [0.3, 0.4, 0.5],  # max = 0.5
            [0.2, 0.6, 0.4]   # max = 0.6, 保持相同长度
        ], dtype=object)
        
        nmds = np.array([
            [0.2, 0.3, 0.1],  # max = 0.3, 0.5 + 0.3 = 0.8 <= 1 ✓
            [0.1, 0.4, 0.2]   # max = 0.4, 0.6 + 0.4 = 1.0 <= 1 ✓
        ], dtype=object)
        
        q = 1
        
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # q=3: max(md)^3 + max(nmd)^3 <= 1
        mds = np.array([
            [0.7, 0.8, 0.9],  # max = 0.9
            [0.6, 0.7, 0.5]   # 保持相同长度
        ], dtype=object)
        
        nmds = np.array([
            [0.1, 0.2, 0.15], # max = 0.2, 0.9^3 + 0.2^3 = 0.737 <= 1 ✓
            [0.3, 0.4, 0.25]  # 保持相同长度
        ], dtype=object)
        
        q = 3
        
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)

    def test_qrohfn_multidimensional_arrays(self):
        """测试多维数组的 QROHFN 约束"""
        # 2D 数组
        mds = np.array([
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.1, 0.6], [0.3, 0.4]]
        ], dtype=object)
        
        nmds = np.array([
            [[0.3, 0.4], [0.2, 0.3]],
            [[0.2, 0.3], [0.4, 0.5]]
        ], dtype=object)
        
        q = 2
        
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        backend = QROHFNBackend.from_arrays(mds, nmds, q)
        # QROHFN backend的shape应该是(2, 2, 2) - 包含组件维度
        assert backend.shape == (2, 2, 2)

    def test_qrohfn_dtype_validation(self):
        """测试数据类型验证"""
        # 测试非 object 类型的数组
        mds = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float64)  # 错误的 dtype
        nmds = np.array([[0.2, 0.3], [0.1, 0.4]], dtype=np.float64)
        q = 2
        
        with pytest.raises(TypeError, match="must have dtype=object"):
            QROHFNBackend.from_arrays(mds, nmds, q)


class TestConstraintsPerformance:
    """测试约束检查的性能"""

    def test_qrofn_performance_large_arrays(self):
        """测试大数组的 QROFN 约束检查性能"""
        # 创建大数组
        size = 10000
        mds = np.random.rand(size) * 0.6  # 确保在有效范围内
        nmds = np.random.rand(size) * 0.6
        q = 2
        
        start_time = time.time()
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        end_time = time.time()
        
        validation_time = end_time - start_time
        assert validation_time < 1.0, f"大数组约束检查耗时{validation_time:.2f}秒，超过预期"
        
        # 测试创建性能
        start_time = time.time()
        backend = QROFNBackend.from_arrays(mds, nmds, q)
        end_time = time.time()
        
        creation_time = end_time - start_time
        assert creation_time < 2.0, f"大数组后端创建耗时{creation_time:.2f}秒，超过预期"

    def test_qrohfn_performance_large_arrays(self):
        """测试大数组的 QROHFN 约束检查性能"""
        # 创建大的犹豫模糊数组
        size = 1000  # QROHFN 处理更复杂，使用较小的测试大小
        
        mds = np.empty(size, dtype=object)
        nmds = np.empty(size, dtype=object)
        
        for i in range(size):
            # 随机生成犹豫集
            md_size = np.random.randint(1, 4)
            nmd_size = np.random.randint(1, 4)
            
            mds[i] = np.random.rand(md_size) * 0.5  # 确保约束满足
            nmds[i] = np.random.rand(nmd_size) * 0.5
        
        q = 2
        
        start_time = time.time()
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        end_time = time.time()
        
        validation_time = end_time - start_time
        assert validation_time < 2.0, f"大犹豫数组约束检查耗时{validation_time:.2f}秒，超过预期"
        
        # 测试创建性能
        start_time = time.time()
        backend = QROHFNBackend.from_arrays(mds, nmds, q)
        end_time = time.time()
        
        creation_time = end_time - start_time
        assert creation_time < 3.0, f"大犹豫数组后端创建耗时{creation_time:.2f}秒，超过预期"

    def test_constraint_check_vs_creation_performance(self):
        """测试约束检查与后端创建的性能比较"""
        # QROFN 性能比较
        size = 5000
        mds = np.random.rand(size) * 0.6
        nmds = np.random.rand(size) * 0.6
        q = 2
        
        # 仅约束检查
        start_time = time.time()
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        constraint_time = time.time() - start_time
        
        # 完整后端创建
        start_time = time.time()
        backend = QROFNBackend.from_arrays(mds, nmds, q)
        total_time = time.time() - start_time
        
        # 约束检查应该是总时间的一小部分
        assert constraint_time < total_time, "约束检查时间应该小于总创建时间"
        
        print(f"\nQROFN 性能分析:")
        print(f"约束检查: {constraint_time:.4f}秒")
        print(f"总创建时间: {total_time:.4f}秒")
        print(f"约束检查占比: {constraint_time/total_time*100:.1f}%")


class TestConstraintsEdgeCases:
    """测试约束检查的边界情况"""

    def test_epsilon_tolerance(self):
        """测试误差容忍度"""
        # 获取配置的 epsilon
        epsilon = get_config().DEFAULT_EPSILON
        
        # 测试刚好超过 1 但在误差范围内的情况
        mds = np.array([np.sqrt(0.5)])
        nmds = np.array([np.sqrt(0.5 + epsilon/2)])  # 略微超过但在误差范围内
        q = 2
        
        # 应该通过
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # 测试明显超过误差范围的情况
        nmds = np.array([np.sqrt(0.5 + epsilon * 2)])  # 明显超过误差范围
        
        with pytest.raises(ValueError, match="QROFN constraint violation"):
            QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)

    def test_zero_values(self):
        """测试零值的处理"""
        # QROFN 零值测试
        mds = np.array([0.0, 0.0, 1.0])
        nmds = np.array([1.0, 0.0, 0.0])
        q = 2
        
        QROFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)
        
        # QROHFN 零值测试
        mds = np.array([
            [0.0],
            [0.0, 0.0],
            [1.0]
        ], dtype=object)
        
        nmds = np.array([
            [1.0],
            [0.0],
            [0.0, 0.0]
        ], dtype=object)
        
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)

    def test_none_values_in_qrohfn(self):
        """测试 QROHFN 中的 None 值处理"""
        mds = np.array([
            [0.2, 0.3],
            None,             # None 值
            [0.4, 0.5]
        ], dtype=object)
        
        nmds = np.array([
            [0.3, 0.4],
            [0.2],
            None              # None 值
        ], dtype=object)
        
        q = 2
        
        # 应该能够处理 None 值（跳过验证）
        QROHFNBackend._validate_fuzzy_constraints_static(mds, nmds, q)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])