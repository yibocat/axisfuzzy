#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:16
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from fuzzlab.membership import SigmoidMF, SMF, ZMF, GeneralizedBellMF, PiMF


class TestSigmoidMF:
    """测试S型（Sigmoid）隶属函数"""

    def test_init_and_compute(self):
        """测试初始化和基本计算"""
        mf = SigmoidMF(a=10, c=0.5)

        # 中心点应该接近0.5
        assert mf.compute(0.5) == pytest.approx(0.5, abs=1e-10)

        # 单调性
        assert mf.compute(0.4) < mf.compute(0.6)

    def test_slope_effect(self):
        """测试斜率参数的影响"""
        x = np.array([0.3, 0.5, 0.7])

        # 陡峭的S型
        mf_steep = SigmoidMF(a=20, c=0.5)
        result_steep = mf_steep.compute(x)

        # 平缓的S型
        mf_gentle = SigmoidMF(a=5, c=0.5)
        result_gentle = mf_gentle.compute(x)

        # 中心点都应该接近0.5
        assert result_steep[1] == pytest.approx(0.5)
        assert result_gentle[1] == pytest.approx(0.5)

        # 陡峭的S型在边缘点差异更大
        steep_diff = result_steep[2] - result_steep[0]
        gentle_diff = result_gentle[2] - result_gentle[0]
        assert steep_diff > gentle_diff


class TestSMF:
    """测试S型隶属函数"""

    def test_init_and_basic_shape(self):
        """测试初始化和基本形状"""
        mf = SMF(a=0.2, b=0.8)

        # 边界点
        assert mf.compute(0.1) == pytest.approx(0.0)
        assert mf.compute(0.9) == pytest.approx(1.0)

        # 中点
        midpoint = (0.2 + 0.8) / 2
        assert mf.compute(midpoint) == pytest.approx(0.5)

    def test_invalid_parameters(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            SMF(a=0.8, b=0.2)  # a > b

    def test_monotonicity(self):
        """测试单调性"""
        mf = SMF(a=0.2, b=0.8)
        x = np.linspace(0, 1, 11)
        result = mf.compute(x)

        # 应该单调递增
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]


class TestZMF:
    """测试Z型隶属函数"""

    def test_init_and_basic_shape(self):
        """测试初始化和基本形状"""
        mf = ZMF(a=0.2, b=0.8)

        # 边界点
        assert mf.compute(0.1) == pytest.approx(1.0)
        assert mf.compute(0.9) == pytest.approx(0.0)

        # 中点
        midpoint = (0.2 + 0.8) / 2
        assert mf.compute(midpoint) == pytest.approx(0.5)

    def test_monotonicity(self):
        """测试单调性"""
        mf = ZMF(a=0.2, b=0.8)
        x = np.linspace(0, 1, 11)
        result = mf.compute(x)

        # 应该单调递减
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]


class TestGeneralizedBellMF:
    """测试广义贝尔隶属函数"""

    def test_init_and_center_value(self):
        """测试初始化和中心值"""
        mf = GeneralizedBellMF(a=2, b=1, c=0.5)

        # 中心点应该为1
        assert mf.compute(0.5) == pytest.approx(1.0)

    def test_symmetry(self):
        """测试对称性"""
        mf = GeneralizedBellMF(a=2, b=2, c=0.5)

        # 关于中心点对称
        assert mf.compute(0.3) == pytest.approx(mf.compute(0.7))
        assert mf.compute(0.4) == pytest.approx(mf.compute(0.6))

    def test_parameter_b_effect(self):
        """测试参数b的影响"""
        x = np.array([0.3, 0.5, 0.7])

        # 较小的b（更宽）
        mf_wide = GeneralizedBellMF(a=2, b=1, c=0.5)
        result_wide = mf_wide.compute(x)

        # 较大的b（更窄）
        mf_narrow = GeneralizedBellMF(a=2, b=4, c=0.5)
        result_narrow = mf_narrow.compute(x)

        # 中心点都为1
        assert result_wide[1] == pytest.approx(1.0)
        assert result_narrow[1] == pytest.approx(1.0)

        # 更大的b在边缘点应该更小
        assert result_narrow[0] < result_wide[0]
        assert result_narrow[2] < result_wide[2]


class TestPiMF:
    """测试Pi型隶属函数"""

    def test_init_and_plateau(self):
        """测试初始化和平台区域"""
        mf = PiMF(a=0.1, b=0.3, c=0.7, d=0.9)

        # 平台区域应该为1
        plateau_x = np.linspace(0.3, 0.7, 5)
        plateau_result = mf.compute(plateau_x)
        np.testing.assert_array_almost_equal(plateau_result, np.ones(5))

    def test_invalid_parameters(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            PiMF(a=0.5, b=0.3, c=0.7, d=0.9)  # a > b

    def test_symmetry_when_possible(self):
        """测试对称Pi型函数"""
        # 对称的Pi型
        mf = PiMF(a=0.0, b=0.4, c=0.6, d=1.0)

        # 上升和下降部分应该对称
        assert mf.compute(0.2) == pytest.approx(mf.compute(0.8))

    def test_edge_values(self):
        """测试边界值"""
        mf = PiMF(a=0.1, b=0.3, c=0.7, d=0.9)

        # 外部区域应该为0
        assert mf.compute(0.05) == pytest.approx(0.0)
        assert mf.compute(0.95) == pytest.approx(0.0)

        # 转折点
        assert mf.compute(0.1) == pytest.approx(0.0)
        assert mf.compute(0.3) == pytest.approx(1.0)
        assert mf.compute(0.7) == pytest.approx(1.0)
        assert mf.compute(0.9) == pytest.approx(0.0)
