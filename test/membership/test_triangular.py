#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:15
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from axisfuzzy.membership import TriangularMF


class TestTriangularMF:
    """测试三角形隶属函数"""

    def test_init_valid_parameters(self):
        """测试有效参数初始化"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)
        assert mf.a == 0.0
        assert mf.b == 0.5
        assert mf.c == 1.0
        assert mf.parameters == {'a': 0.0, 'b': 0.5, 'c': 1.0}

    def test_init_equal_parameters(self):
        """测试相等参数（边界情况）"""
        # a == b 的情况
        mf1 = TriangularMF(a=0.0, b=0.0, c=1.0)
        assert mf1.a == mf1.b == 0.0

        # b == c 的情况
        mf2 = TriangularMF(a=0.0, b=1.0, c=1.0)
        assert mf2.b == mf2.c == 1.0

        # a == b == c 的情况（退化为单点）
        mf3 = TriangularMF(a=0.5, b=0.5, c=0.5)
        assert mf3.a == mf3.b == mf3.c == 0.5

    def test_init_invalid_parameters(self):
        """测试无效参数抛出异常"""
        with pytest.raises(ValueError, match="requires parameters to satisfy a <= b <= c"):
            TriangularMF(a=1.0, b=0.5, c=2.0)  # a > b

        with pytest.raises(ValueError, match="requires parameters to satisfy a <= b <= c"):
            TriangularMF(a=0.0, b=2.0, c=1.0)  # b > c

    def test_compute_scalar_basic_shape(self):
        """测试标量输入的基本三角形形状"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        # 测试关键点
        assert mf.compute(-0.5) == 0.0  # 左侧外部
        assert mf.compute(0.0) == 0.0   # 左端点
        assert mf.compute(0.25) == 0.5  # 左侧斜率
        assert mf.compute(0.5) == 1.0   # 峰值
        assert mf.compute(0.75) == 0.5  # 右侧斜率
        assert mf.compute(1.0) == 0.0   # 右端点
        assert mf.compute(1.5) == 0.0   # 右侧外部

    def test_compute_array_basic_shape(self):
        """测试数组输入的基本三角形形状"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)
        x = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
        expected = np.array([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])

        result = mf.compute(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_edge_cases(self):
        """测试边界情况"""
        # a == b 的情况（左侧垂直）
        mf1 = TriangularMF(a=0.0, b=0.0, c=1.0)
        assert mf1.compute(0.0) == 1.0
        assert mf1.compute(0.5) == 0.5

        # b == c 的情况（右侧垂直）
        mf2 = TriangularMF(a=0.0, b=1.0, c=1.0)
        assert mf2.compute(0.5) == 0.5
        assert mf2.compute(1.0) == 1.0

        # a == b == c 的情况（单点）
        mf3 = TriangularMF(a=0.5, b=0.5, c=0.5)
        assert mf3.compute(0.5) == 1.0
        assert mf3.compute(0.4) == 0.0
        assert mf3.compute(0.6) == 0.0

    def test_compute_output_range(self):
        """测试输出范围在 [0, 1]"""
        mf = TriangularMF(a=-2.0, b=0.0, c=3.0)
        x = np.linspace(-5, 5, 100)
        result = mf.compute(x)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert np.max(result) == 1.0  # 峰值应该为1

    def test_compute_shape_preservation(self):
        """测试输入输出形状保持"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        # 1D 数组
        x1d = np.array([0.1, 0.5, 0.9])
        result1d = mf.compute(x1d)
        assert result1d.shape == (3,)

        # 2D 数组
        x2d = np.array([[0.1, 0.5], [0.7, 0.9]])
        result2d = mf.compute(x2d)
        assert result2d.shape == (2, 2)

        # 标量
        result_scalar = mf.compute(0.5)
        assert np.isscalar(result_scalar) or result_scalar.shape == ()

    def test_set_parameters(self):
        """测试参数设置"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        # 设置有效参数
        mf.set_parameters(a=0.1, b=0.6, c=1.1)
        assert mf.a == 0.1
        assert mf.b == 0.6
        assert mf.c == 1.1
        assert mf.parameters == {'a': 0.1, 'b': 0.6, 'c': 1.1}

        # 部分设置
        mf.set_parameters(b=0.7)
        assert mf.a == 0.1  # 不变
        assert mf.b == 0.7  # 改变
        assert mf.c == 1.1  # 不变

    def test_set_parameters_invalid(self):
        """测试设置无效参数"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        # 无效的参数顺序
        with pytest.raises(ValueError):
            mf.set_parameters(a=0.8, b=0.5, c=1.0)  # a > b

    def test_callable_interface(self):
        """测试可调用接口"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        # 使用 __call__ 方法
        assert mf(0.5) == mf.compute(0.5)

        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(mf(x), mf.compute(x))

    def test_negative_range(self):
        """测试负数范围"""
        mf = TriangularMF(a=-2.0, b=-1.0, c=0.0)

        assert mf.compute(-3.0) == 0.0
        assert mf.compute(-1.0) == 1.0
        assert mf.compute(-1.5) == 0.5
        assert mf.compute(1.0) == 0.0

    def test_large_range(self):
        """测试大范围数值稳定性"""
        mf = TriangularMF(a=0, b=500, c=1000)

        # 测试中点和边界
        assert mf.compute(500) == 1.0
        assert mf.compute(250) == 0.5
        assert mf.compute(750) == 0.5
        assert mf.compute(-100) == 0.0
        assert mf.compute(1200) == 0.0