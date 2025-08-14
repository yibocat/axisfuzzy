#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:28
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from fuzzlab.fuzztype.qrofs.fuzzify import QROFNFuzzificationStrategy
from fuzzlab.membership import TriangularMF, GaussianMF
from fuzzlab.core import Fuzzarray


class TestQROFNFuzzificationStrategyArray:

    def test_fuzzify_array_basic(self):
        """测试基本数组模糊化"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.2)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = strategy.fuzzify_array(x, mf)

        assert isinstance(result, Fuzzarray)
        assert result.mtype == 'qrofn'
        assert result.q == 2
        assert result.shape == (5,)

    def test_fuzzify_array_missing_pi(self):
        """测试缺失 pi 参数"""
        strategy = QROFNFuzzificationStrategy(q=2)  # 没有 pi
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)
        x = np.array([0.3, 0.7])

        with pytest.raises(ValueError, match="Parameter 'pi'.*is required"):
            strategy.fuzzify_array(x, mf)

    def test_fuzzify_array_multidimensional(self):
        """测试多维数组"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.15)
        mf = GaussianMF(sigma=0.5, c=0.5)

        x = np.array([[0.1, 0.3], [0.7, 0.9]])
        result = strategy.fuzzify_array(x, mf)

        assert isinstance(result, Fuzzarray)
        assert result.shape == (2, 2)
        assert result.mtype == 'qrofn'
        assert result.q == 2

    def test_fuzzify_array_constraint_satisfaction(self):
        """测试数组所有元素满足 q-rung 约束"""
        strategy = QROFNFuzzificationStrategy(q=3, pi=0.1)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        x = np.linspace(0, 1, 20)  # 20个测试点
        result = strategy.fuzzify_array(x, mf)

        # 获取所有 md 和 nmd 值
        md_values = result.backend.md
        nmd_values = result.backend.nmd

        # 验证所有值都在 [0,1] 范围内
        assert np.all(md_values >= 0) and np.all(md_values <= 1)
        assert np.all(nmd_values >= 0) and np.all(nmd_values <= 1)

        # 验证所有值都满足 q-rung 约束
        constraints = md_values**3 + nmd_values**3 + 0.1**3
        assert np.all(constraints <= 1.001)  # 允许小的浮点误差

    def test_fuzzify_array_different_q_values(self):
        """测试不同 q 值的数组处理"""
        mf = GaussianMF(sigma=0.3, c=0.5)
        x = np.array([0.2, 0.4, 0.6, 0.8])
        pi = 0.15

        for q in [1, 2, 3, 4]:
            strategy = QROFNFuzzificationStrategy(q=q, pi=pi)
            result = strategy.fuzzify_array(x, mf)

            assert result.q == q
            assert result.shape == (4,)

            # 验证约束
            md_values = result.backend.md
            nmd_values = result.backend.nmd
            constraints = md_values**q + nmd_values**q + pi**q
            assert np.all(constraints <= 1.001)

    def test_fuzzify_array_pi_boundary_values(self):
        """测试 pi 边界值"""
        mf = TriangularMF(a=0.2, b=0.5, c=0.8)
        x = np.array([0.3, 0.5, 0.7])

        # pi = 0
        strategy_pi0 = QROFNFuzzificationStrategy(q=2, pi=0.0)
        result_pi0 = strategy_pi0.fuzzify_array(x, mf)

        md_values = result_pi0.backend.md
        nmd_values = result_pi0.backend.nmd

        # pi=0 时，md^2 + nmd^2 应该接近 1
        sum_squares = md_values**2 + nmd_values**2
        assert np.allclose(sum_squares, 1.0, atol=1e-10)

        # pi = 1（极端情况）
        strategy_pi1 = QROFNFuzzificationStrategy(q=2, pi=1.0)
        result_pi1 = strategy_pi1.fuzzify_array(x, mf)

        nmd_values_pi1 = result_pi1.backend.nmd

        # pi=1 时，nmd 应该接近 0
        assert np.all(nmd_values_pi1 < 1e-10)

    def test_fuzzify_array_vectorization_consistency(self):
        """测试向量化结果与标量结果一致性"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.25)
        mf = GaussianMF(sigma=0.4, c=0.6)

        x_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # 向量化结果
        array_result = strategy.fuzzify_array(x_values, mf)
        md_array = array_result.backend.md
        nmd_array = array_result.backend.nmd

        # 逐个标量计算结果
        md_scalars = []
        nmd_scalars = []

        for x in x_values:
            scalar_result = strategy.fuzzify_scalar(float(x), mf)
            info = scalar_result.get_info()
            md_scalars.append(info['md'])
            nmd_scalars.append(info['nmd'])

        # 比较结果
        assert np.allclose(md_array, md_scalars, atol=1e-12)
        assert np.allclose(nmd_array, nmd_scalars, atol=1e-12)

    def test_fuzzify_array_empty_array(self):
        """测试空数组"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.2)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        x = np.array([])
        result = strategy.fuzzify_array(x, mf)

        assert isinstance(result, Fuzzarray)
        assert result.shape == (0,)

    def test_fuzzify_array_single_element(self):
        """测试单元素数组"""
        strategy = QROFNFuzzificationStrategy(q=3, pi=0.1)
        mf = GaussianMF(sigma=0.2, c=0.5)

        x = np.array([0.6])
        result = strategy.fuzzify_array(x, mf)

        assert isinstance(result, Fuzzarray)
        assert result.shape == (1,)

        # 与标量结果比较
        scalar_result = strategy.fuzzify_scalar(0.6, mf)
        scalar_info = scalar_result.get_info()

        assert abs(result.backend.md[0] - scalar_info['md']) < 1e-12
        assert abs(result.backend.nmd[0] - scalar_info['nmd']) < 1e-12

    def test_fuzzify_array_large_array(self):
        """测试大型数组的性能和正确性"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.15)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        # 创建大型数组
        x = np.random.uniform(0, 1, 1000)
        result = strategy.fuzzify_array(x, mf)

        assert isinstance(result, Fuzzarray)
        assert result.shape == (1000,)

        # 验证所有结果的合理性
        md_values = result.backend.md
        nmd_values = result.backend.nmd

        assert np.all(md_values >= 0) and np.all(md_values <= 1)
        assert np.all(nmd_values >= 0) and np.all(nmd_values <= 1)

        constraints = md_values**2 + nmd_values**2 + 0.15**2
        assert np.all(constraints <= 1.001)

    def test_fuzzify_array_numerical_stability(self):
        """测试数值稳定性"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.05)
        mf = TriangularMF(a=0.0, b=0.95, c=1.0)

        # 创建接近边界的值
        x = np.array([0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
        result = strategy.fuzzify_array(x, mf)

        md_values = result.backend.md
        nmd_values = result.backend.nmd

        # 检查没有 NaN 或 Inf
        assert not np.any(np.isnan(md_values))
        assert not np.any(np.isnan(nmd_values))
        assert not np.any(np.isinf(md_values))
        assert not np.any(np.isinf(nmd_values))

        # 检查非负性
        assert np.all(md_values >= 0)
        assert np.all(nmd_values >= 0)

    def test_compute_nmd_from_hesitation_array(self):
        """测试静态方法处理数组"""
        md_array = np.array([0.3, 0.5, 0.7, 0.9])
        pi = 0.2
        q = 2

        nmd_array = QROFNFuzzificationStrategy._compute_nmd_from_hesitation(md_array, pi, q)

        assert isinstance(nmd_array, np.ndarray)
        assert nmd_array.shape == md_array.shape

        # 验证每个元素的计算
        for i, (md, nmd) in enumerate(zip(md_array, nmd_array)):
            expected = (1 - md**q - pi**q)**(1/q)
            assert abs(nmd - expected) < 1e-10

            # 验证约束
            assert md**q + nmd**q + pi**q <= 1.001

    def test_different_array_dtypes(self):
        """测试不同数组数据类型"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.2)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        # 测试不同 dtype
        dtypes = [np.float32, np.float64, np.int32, np.int64]

        for dtype in dtypes:
            x = np.array([0.2, 0.5, 0.8], dtype=dtype)
            result = strategy.fuzzify_array(x, mf)

            assert isinstance(result, Fuzzarray)
            assert result.shape == (3,)

            # 验证结果数据类型为 float
            assert result.backend.md.dtype == np.float64
            assert result.backend.nmd.dtype == np.float64


if __name__ == '__main__':
    pytest.main([__file__])
