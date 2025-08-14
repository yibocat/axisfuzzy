#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:15
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from fuzzlab.membership import GaussianMF, DoubleGaussianMF


class TestGaussianMF:
    """测试高斯隶属函数"""

    def test_init_valid_parameters(self):
        """测试有效参数初始化"""
        mf = GaussianMF(sigma=0.2, c=0.5)
        assert mf.sigma == 0.2
        assert mf.c == 0.5
        assert mf.parameters == {'sigma': 0.2, 'c': 0.5}

    def test_init_invalid_sigma(self):
        """测试无效 sigma 参数"""
        with pytest.raises(ValueError, match="'sigma' must be positive"):
            GaussianMF(sigma=0, c=0.5)

        with pytest.raises(ValueError, match="'sigma' must be positive"):
            GaussianMF(sigma=-0.1, c=0.5)

    def test_compute_scalar_basic_shape(self):
        """测试标量输入的基本高斯形状"""
        mf = GaussianMF(sigma=0.2, c=0.5)

        # 中心点应该为1
        assert abs(mf.compute(0.5) - 1.0) < 1e-10

        # 对称性
        assert abs(mf.compute(0.3) - mf.compute(0.7)) < 1e-10

        # 单调性（远离中心值递减）
        assert mf.compute(0.4) > mf.compute(0.3)
        assert mf.compute(0.6) > mf.compute(0.7)

    def test_compute_array_basic_shape(self):
        """测试数组输入的基本高斯形状"""
        mf = GaussianMF(sigma=1.0, c=0.0)
        x = np.array([-2, -1, 0, 1, 2])
        result = mf.compute(x)

        # 中心点为最大值
        assert result[2] == pytest.approx(1.0)

        # 对称性
        assert result[1] == pytest.approx(result[3])
        assert result[0] == pytest.approx(result[4])

        # 单调递减
        assert result[2] > result[1] > result[0]

    def test_compute_different_sigma(self):
        """测试不同 sigma 值的影响"""
        c = 0.5
        x = np.array([0.3, 0.5, 0.7])

        # 小 sigma（窄）
        mf_narrow = GaussianMF(sigma=0.1, c=c)
        result_narrow = mf_narrow.compute(x)

        # 大 sigma（宽）
        mf_wide = GaussianMF(sigma=0.5, c=c)
        result_wide = mf_wide.compute(x)

        # 中心点都应该为1
        assert result_narrow[1] == pytest.approx(1.0)
        assert result_wide[1] == pytest.approx(1.0)

        # 窄高斯在边缘点应该更小
        assert result_narrow[0] < result_wide[0]
        assert result_narrow[2] < result_wide[2]

    def test_compute_output_range(self):
        """测试输出范围在 (0, 1]"""
        mf = GaussianMF(sigma=0.2, c=0.5)
        x = np.linspace(-2, 2, 100)
        result = mf.compute(x)

        assert np.all(result > 0.0)  # 高斯函数永远大于0
        assert np.all(result <= 1.0)
        assert np.max(result) == pytest.approx(1.0)

    def test_compute_shape_preservation(self):
        """测试输入输出形状保持"""
        mf = GaussianMF(sigma=0.2, c=0.5)

        # 2D 数组
        x2d = np.array([[0.3, 0.5], [0.7, 0.9]])
        result2d = mf.compute(x2d)
        assert result2d.shape == (2, 2)

    def test_set_parameters(self):
        """测试参数设置"""
        mf = GaussianMF(sigma=0.2, c=0.5)

        # 设置有效参数
        mf.set_parameters(sigma=0.3, c=0.7)
        assert mf.sigma == 0.3
        assert mf.c == 0.7
        assert mf.parameters == {'sigma': 0.3, 'c': 0.7}

    def test_set_parameters_invalid_sigma(self):
        """测试设置无效 sigma"""
        mf = GaussianMF(sigma=0.2, c=0.5)

        with pytest.raises(ValueError, match="'sigma' must be positive"):
            mf.set_parameters(sigma=0)

        with pytest.raises(ValueError, match="'sigma' must be positive"):
            mf.set_parameters(sigma=-0.1)

    def test_mathematical_properties(self):
        """测试数学性质"""
        mf = GaussianMF(sigma=1.0, c=0.0)

        # 在中心点 ±σ 处，值应该约为 exp(-0.5) ≈ 0.606
        expected_at_sigma = np.exp(-0.5)
        assert mf.compute(1.0) == pytest.approx(expected_at_sigma, rel=1e-10)
        assert mf.compute(-1.0) == pytest.approx(expected_at_sigma, rel=1e-10)


class TestDoubleGaussianMF:
    """测试双高斯隶属函数"""

    def test_init_valid_parameters(self):
        """测试有效参数初始化"""
        mf = DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.2, c2=0.7)
        assert mf.sigma1 == 0.1
        assert mf.c1 == 0.3
        assert mf.sigma2 == 0.2
        assert mf.c2 == 0.7

    def test_init_invalid_sigma(self):
        """测试无效 sigma 参数"""
        with pytest.raises(ValueError, match="'sigma1' and 'sigma2' must be positive"):
            DoubleGaussianMF(sigma1=0, c1=0.3, sigma2=0.2, c2=0.7)

        with pytest.raises(ValueError, match="'sigma1' and 'sigma2' must be positive"):
            DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=-0.1, c2=0.7)

    def test_compute_two_peaks(self):
        """测试双峰特性"""
        mf = DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.1, c2=0.7)

        # 两个中心点都应该接近1
        assert mf.compute(0.3) == pytest.approx(1.0, abs=1e-10)
        assert mf.compute(0.7) == pytest.approx(1.0, abs=1e-10)

        # 中间的谷应该较低
        valley = mf.compute(0.5)
        assert valley < 1.0

    def test_compute_maximum_operation(self):
        """测试双高斯取最大值的特性"""
        mf = DoubleGaussianMF(sigma1=0.2, c1=0.3, sigma2=0.2, c2=0.7)

        # 在第一个高斯的影响区域
        x1 = 0.2
        gauss1_val = np.exp(-0.5 * ((x1 - 0.3) / 0.2) ** 2)
        gauss2_val = np.exp(-0.5 * ((x1 - 0.7) / 0.2) ** 2)
        expected = max(gauss1_val, gauss2_val)

        assert mf.compute(x1) == pytest.approx(expected)

    def test_set_parameters(self):
        """测试参数设置"""
        mf = DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.2, c2=0.7)

        mf.set_parameters(sigma1=0.15, c2=0.8)
        assert mf.sigma1 == 0.15
        assert mf.c1 == 0.3     # 未改变
        assert mf.sigma2 == 0.2 # 未改变
        assert mf.c2 == 0.8     # 改变

    def test_set_parameters_invalid_sigma(self):
        """测试设置无效 sigma"""
        mf = DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.2, c2=0.7)

        with pytest.raises(ValueError):
            mf.set_parameters(sigma1=0)

        with pytest.raises(ValueError):
            mf.set_parameters(sigma2=-0.1)
