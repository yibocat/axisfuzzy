#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:28
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from axisfuzzy.fuzztype.qrofs.fuzzify import QROFNFuzzificationStrategy
from axisfuzzy.membership import TriangularMF, GaussianMF
from axisfuzzy.core import Fuzznum


class TestQROFNFuzzificationStrategyScalar:

    def test_init_valid_parameters(self):
        """测试有效参数初始化"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.3)

        assert strategy.q == 2
        assert strategy.kwargs['pi'] == 0.3
        assert strategy.mtype == 'qrofn'
        assert strategy.method == 'default'

    def test_init_default_q(self):
        """测试默认 q 值"""
        strategy = QROFNFuzzificationStrategy(pi=0.2)
        assert strategy.q == 1

    def test_init_invalid_q(self):
        """测试无效 q 值"""
        with pytest.raises(ValueError, match="Parameter 'q' must be an integer >= 1"):
            QROFNFuzzificationStrategy(q=0, pi=0.2)

        with pytest.raises(ValueError, match="Parameter 'q' must be an integer >= 1"):
            QROFNFuzzificationStrategy(q=-1, pi=0.2)

        with pytest.raises(ValueError, match="Parameter 'q' must be an integer >= 1"):
            QROFNFuzzificationStrategy(q=None, pi=0.2)

    def test_fuzzify_scalar_basic(self):
        """测试基本标量模糊化"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.2)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        result = strategy.fuzzify_scalar(0.5, mf)

        assert isinstance(result, Fuzznum)
        assert result.mtype == 'qrofn'
        assert result.q == 2

        # 验证数值合理性
        info = result.get_info()
        assert 0.0 <= info['md'] <= 1.0
        assert 0.0 <= info['nmd'] <= 1.0
        # 验证 q-rung 约束: md^q + nmd^q + pi^q <= 1
        constraint = info['md']**2 + info['nmd']**2 + 0.2**2
        assert constraint <= 1.001  # 允许小的浮点误差

    def test_fuzzify_scalar_missing_pi(self):
        """测试缺失 pi 参数"""
        strategy = QROFNFuzzificationStrategy(q=2)  # 没有 pi
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        with pytest.raises(ValueError, match="Parameter 'pi'.*is required"):
            strategy.fuzzify_scalar(0.5, mf)

    def test_fuzzify_scalar_pi_zero(self):
        """测试 pi=0 的边界情况"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.0)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        result = strategy.fuzzify_scalar(0.5, mf)
        info = result.get_info()

        # pi=0 时，应该有 md^q + nmd^q = 1
        assert abs(info['md']**2 + info['nmd']**2 - 1.0) < 1e-10

    def test_fuzzify_scalar_pi_one(self):
        """测试 pi=1 的边界情况"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=1.0)
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        result = strategy.fuzzify_scalar(0.5, mf)
        info = result.get_info()

        # pi=1 时，应该有 nmd ≈ 0
        assert info['nmd'] < 1e-10

    def test_fuzzify_scalar_different_q_values(self):
        """测试不同 q 值"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)
        x = 0.5
        pi = 0.2

        for q in [1, 2, 3, 5]:
            strategy = QROFNFuzzificationStrategy(q=q, pi=pi)
            result = strategy.fuzzify_scalar(x, mf)
            info = result.get_info()

            # 验证 q-rung 约束
            constraint = info['md']**q + info['nmd']**q + pi**q
            assert constraint <= 1.001
            assert result.q == q

    def test_fuzzify_scalar_different_membership_values(self):
        """测试不同隶属度值"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.2)

        # 测试不同 x 值产生不同隶属度
        test_cases = [
            (0.0, TriangularMF(a=0.0, b=0.5, c=1.0)),  # md = 0
            (0.5, TriangularMF(a=0.0, b=0.5, c=1.0)),  # md = 1
            (1.0, TriangularMF(a=0.0, b=0.5, c=1.0)),  # md = 0
            (0.25, TriangularMF(a=0.0, b=0.5, c=1.0)), # md = 0.5
        ]

        for x, mf in test_cases:
            result = strategy.fuzzify_scalar(x, mf)
            info = result.get_info()

            # 验证隶属度计算正确
            expected_md = mf.compute(x)
            assert abs(info['md'] - expected_md) < 1e-10

            # 验证约束
            constraint = info['md']**2 + info['nmd']**2 + 0.2**2
            assert constraint <= 1.001

    def test_fuzzify_scalar_gaussian_membership(self):
        """测试高斯隶属函数"""
        strategy = QROFNFuzzificationStrategy(q=3, pi=0.1)
        mf = GaussianMF(sigma=1.0, c=5.0)

        result = strategy.fuzzify_scalar(5.0, mf)
        info = result.get_info()

        # 在高斯中心点，隶属度应为1
        assert abs(info['md'] - 1.0) < 1e-10

        # 验证约束
        constraint = info['md']**3 + info['nmd']**3 + 0.1**3
        assert constraint <= 1.001

    def test_compute_nmd_from_hesitation_static_method(self):
        """测试静态方法 _compute_nmd_from_hesitation"""
        # 测试标量情况
        md = 0.6
        pi = 0.2
        q = 2

        nmd = QROFNFuzzificationStrategy._compute_nmd_from_hesitation(md, pi, q)

        # 验证计算公式: nmd = (1 - md^q - pi^q)^(1/q)
        expected = (1 - md**q - pi**q)**(1/q)
        assert abs(nmd - expected) < 1e-10

        # 验证约束
        assert md**q + nmd**q + pi**q <= 1.001

    def test_compute_nmd_extreme_cases(self):
        """测试极端情况的数值稳定性"""
        # md 接近 1，pi 接近 0
        md = 0.99
        pi = 0.01
        q = 2

        nmd = QROFNFuzzificationStrategy._compute_nmd_from_hesitation(md, pi, q)

        # nmd 应该很小但非负
        assert nmd >= 0
        assert not np.isnan(nmd)
        assert not np.isinf(nmd)

        # 测试 md + pi 的平方和超过1的情况（数值稳定性）
        md = 0.8
        pi = 0.7  # 0.8^2 + 0.7^2 = 1.13 > 1

        nmd = QROFNFuzzificationStrategy._compute_nmd_from_hesitation(md, pi, q)

        # 应该被裁剪到合理范围
        assert nmd >= 0
        assert not np.isnan(nmd)
        assert not np.isinf(nmd)

    def test_fuzzify_scalar_clipping(self):
        """测试隶属度值的裁剪"""
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.2)

        # 创建一个可能返回超出 [0,1] 范围值的模拟隶属函数
        class MockMF:
            def compute(self, x):
                return x  # 不做裁剪，让 strategy 处理

        mf = MockMF()

        # 测试超出范围的输入
        result1 = strategy.fuzzify_scalar(1.5, mf)  # 应该被裁剪到 1.0
        info1 = result1.get_info()
        assert info1['md'] <= 1.0

        result2 = strategy.fuzzify_scalar(-0.5, mf)  # 应该被裁剪到 0.0
        info2 = result2.get_info()
        assert info2['md'] >= 0.0


if __name__ == '__main__':
    pytest.main([__file__])
