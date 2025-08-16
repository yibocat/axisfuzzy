#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:16
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from axisfuzzy.membership import TrapezoidalMF


class TestTrapezoidalMF:
    """测试梯形隶属函数"""

    def test_init_valid_parameters(self):
        """测试有效参数初始化"""
        mf = TrapezoidalMF(a=0.0, b=0.2, c=0.8, d=1.0)
        assert mf.a == 0.0
        assert mf.b == 0.2
        assert mf.c == 0.8
        assert mf.d == 1.0
        assert mf.parameters == {'a': 0.0, 'b': 0.2, 'c': 0.8, 'd': 1.0}

    def test_init_equal_parameters(self):
        """测试相等参数（边界情况）"""
        # b == c 的情况（退化为三角形）
        mf1 = TrapezoidalMF(a=0.0, b=0.5, c=0.5, d=1.0)
        assert mf1.b == mf1.c == 0.5

        # a == b 的情况
        mf2 = TrapezoidalMF(a=0.0, b=0.0, c=0.8, d=1.0)
        assert mf2.a == mf2.b == 0.0

        # c == d 的情况
        mf3 = TrapezoidalMF(a=0.0, b=0.2, c=1.0, d=1.0)
        assert mf3.c == mf3.d == 1.0

    def test_init_invalid_parameters(self):
        """测试无效参数抛出异常"""
        with pytest.raises(ValueError, match="requires parameters to satisfy a <= b <= c <= d"):
            TrapezoidalMF(a=1.0, b=0.5, c=0.8, d=1.0)  # a > b

        with pytest.raises(ValueError, match="requires parameters to satisfy a <= b <= c <= d"):
            TrapezoidalMF(a=0.0, b=0.8, c=0.5, d=1.0)  # b > c

        with pytest.raises(ValueError, match="requires parameters to satisfy a <= b <= c <= d"):
            TrapezoidalMF(a=0.0, b=0.2, c=1.0, d=0.8)  # c > d

    def test_compute_scalar_basic_shape(self):
        """测试标量输入的基本梯形形状"""
        mf = TrapezoidalMF(a=0.0, b=0.2, c=0.8, d=1.0)

        # 测试关键点
        assert mf.compute(-0.1) == 0.0  # 左侧外部
        assert mf.compute(0.0) == 0.0   # 左端点
        assert mf.compute(0.1) == 0.5   # 左侧斜坡
        assert mf.compute(0.2) == 1.0   # 平台开始
        assert mf.compute(0.5) == 1.0   # 平台中间
        assert mf.compute(0.8) == 1.0   # 平台结束
        assert mf.compute(0.9) == 0.5   # 右侧斜坡
        assert mf.compute(1.0) == 0.0   # 右端点
        assert mf.compute(1.1) == 0.0   # 右侧外部

    def test_compute_array_basic_shape(self):
        """测试数组输入的基本梯形形状"""
        mf = TrapezoidalMF(a=0.0, b=0.2, c=0.8, d=1.0)
        x = np.array([-0.1, 0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0, 1.1])
        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])

        result = mf.compute(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_degenerate_to_triangular(self):
        """测试退化为三角形的情况（b == c）"""
        mf = TrapezoidalMF(a=0.0, b=0.5, c=0.5, d=1.0)

        assert mf.compute(0.0) == 0.0
        assert mf.compute(0.25) == 0.5
        assert mf.compute(0.5) == 1.0   # 峰值
        assert mf.compute(0.75) == 0.5
        assert mf.compute(1.0) == 0.0

    def test_compute_edge_cases(self):
        """测试边界情况"""
        # a == b 的情况（左侧垂直）
        mf1 = TrapezoidalMF(a=0.0, b=0.0, c=0.8, d=1.0)
        assert mf1.compute(0.0) == 1.0
        assert mf1.compute(0.4) == 1.0

        # c == d 的情况（右侧垂直）
        mf2 = TrapezoidalMF(a=0.0, b=0.2, c=1.0, d=1.0)
        assert mf2.compute(0.6) == 1.0
        assert mf2.compute(1.0) == 1.0

    def test_compute_output_range(self):
        """测试输出范围在 [0, 1]"""
        mf = TrapezoidalMF(a=-2.0, b=-1.0, c=1.0, d=2.0)
        x = np.linspace(-5, 5, 100)
        result = mf.compute(x)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert np.max(result) == 1.0  # 平台应该为1

    def test_compute_shape_preservation(self):
        """测试输入输出形状保持"""
        mf = TrapezoidalMF(a=0.0, b=0.2, c=0.8, d=1.0)

        # 1D 数组
        x1d = np.array([0.1, 0.5, 0.9])
        result1d = mf.compute(x1d)
        assert result1d.shape == (3,)

        # 2D 数组
        x2d = np.array([[0.1, 0.5], [0.7, 0.9]])
        result2d = mf.compute(x2d)
        assert result2d.shape == (2, 2)

    def test_set_parameters(self):
        """测试参数设置"""
        mf = TrapezoidalMF(a=0.0, b=0.2, c=0.8, d=1.0)

        # 设置有效参数
        mf.set_parameters(a=0.1, b=0.3, c=0.7, d=0.9)
        assert mf.a == 0.1
        assert mf.b == 0.3
        assert mf.c == 0.7
        assert mf.d == 0.9

        # 部分设置
        mf.set_parameters(b=0.25, c=0.75)
        assert mf.a == 0.1  # 不变
        assert mf.b == 0.25 # 改变
        assert mf.c == 0.75 # 改变
        assert mf.d == 0.9  # 不变

    def test_set_parameters_invalid(self):
        """测试设置无效参数"""
        mf = TrapezoidalMF(a=0.0, b=0.2, c=0.8, d=1.0)

        with pytest.raises(ValueError):
            mf.set_parameters(a=0.5, b=0.2, c=0.8, d=1.0)  # a > b

    def test_plateau_width_variations(self):
        """测试不同平台宽度"""
        # 窄平台
        mf1 = TrapezoidalMF(a=0.0, b=0.45, c=0.55, d=1.0)
        assert mf1.compute(0.5) == 1.0

        # 宽平台
        mf2 = TrapezoidalMF(a=0.0, b=0.1, c=0.9, d=1.0)
        x_plateau = np.linspace(0.1, 0.9, 10)
        result_plateau = mf2.compute(x_plateau)
        np.testing.assert_array_almost_equal(result_plateau, np.ones(10))