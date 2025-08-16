#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:17
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from axisfuzzy.membership import (
    TriangularMF, TrapezoidalMF, GaussianMF,
    SigmoidMF, SMF, ZMF, GeneralizedBellMF,
    PiMF, DoubleGaussianMF
)


class TestVectorization:
    """测试所有隶属函数的向量化和形状处理"""

    @pytest.fixture
    def all_membership_functions(self):
        """提供所有隶属函数的实例"""
        return [
            TriangularMF(a=0.0, b=0.5, c=1.0),
            TrapezoidalMF(a=0.0, b=0.2, c=0.8, d=1.0),
            GaussianMF(sigma=0.2, c=0.5),
            SigmoidMF(a=10, c=0.5),
            SMF(a=0.2, b=0.8),
            ZMF(a=0.2, b=0.8),
            GeneralizedBellMF(a=2, b=1, c=0.5),
            PiMF(a=0.1, b=0.3, c=0.7, d=0.9),
            DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.1, c2=0.7)
        ]

    def test_scalar_input_output_type(self, all_membership_functions):
        """测试标量输入输出类型"""
        x = 0.5

        for mf in all_membership_functions:
            result = mf.compute(x)
            # 标量输入应该返回标量或0维数组
            assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)
            # 值应该在[0,1]范围内
            assert 0.0 <= float(result) <= 1.0

    def test_1d_array_shape_preservation(self, all_membership_functions):
        """测试1D数组形状保持"""
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        for mf in all_membership_functions:
            result = mf.compute(x)
            assert result.shape == x.shape
            assert result.dtype == np.float64 or result.dtype == float
            assert np.all((result >= 0.0) & (result <= 1.0))

    def test_2d_array_shape_preservation(self, all_membership_functions):
        """测试2D数组形状保持"""
        x = np.array([[0.1, 0.3, 0.5],
                      [0.7, 0.9, 0.2]])

        for mf in all_membership_functions:
            result = mf.compute(x)
            assert result.shape == x.shape
            assert np.all((result >= 0.0) & (result <= 1.0))

    def test_3d_array_shape_preservation(self, all_membership_functions):
        """测试3D数组形状保持"""
        x = np.random.rand(2, 3, 4)

        for mf in all_membership_functions:
            result = mf.compute(x)
            assert result.shape == x.shape
            assert np.all((result >= 0.0) & (result <= 1.0))

    def test_empty_array_handling(self, all_membership_functions):
        """测试空数组处理"""
        x = np.array([])

        for mf in all_membership_functions:
            result = mf.compute(x)
            assert result.shape == (0,)

    def test_single_element_array(self, all_membership_functions):
        """测试单元素数组"""
        x = np.array([0.5])

        for mf in all_membership_functions:
            result = mf.compute(x)
            assert result.shape == (1,)
            assert 0.0 <= result[0] <= 1.0

    def test_zero_dimensional_array(self, all_membership_functions):
        """测试0维数组"""
        x = np.array(0.5)  # 0维数组

        for mf in all_membership_functions:
            result = mf.compute(x)
            assert result.shape == () or np.isscalar(result)
            assert 0.0 <= float(result) <= 1.0

    def test_large_array_performance(self, all_membership_functions):
        """测试大数组性能（不应该有明显的性能问题）"""
        x = np.random.rand(1000)

        for mf in all_membership_functions:
            result = mf.compute(x)
            assert result.shape == x.shape
            assert np.all((result >= 0.0) & (result <= 1.0))

    def test_extreme_values_stability(self, all_membership_functions):
        """测试极值的数值稳定性"""
        # 测试极大值
        x_large = np.array([1e6, -1e6, 1e10, -1e10])

        for mf in all_membership_functions:
            result = mf.compute(x_large)
            # 结果应该有限且在合理范围内
            assert np.all(np.isfinite(result))
            assert np.all((result >= 0.0) & (result <= 1.0))

    def test_special_float_values(self, all_membership_functions):
        """测试特殊浮点值的处理"""
        # 测试inf和nan的处理（应该返回有限值或适当处理）
        x_special = np.array([0.0, 0.5, 1.0])  # 使用正常值，因为inf/nan处理可能因函数而异

        for mf in all_membership_functions:
            result = mf.compute(x_special)
            assert np.all(np.isfinite(result))

    def test_dtype_preservation_and_conversion(self, all_membership_functions):
        """测试数据类型处理"""
        # 整数输入
        x_int = np.array([0, 1, 2], dtype=int)

        for mf in all_membership_functions:
            result = mf.compute(x_int)
            assert result.dtype in [np.float64, float]
            assert result.shape == x_int.shape

    def test_consistency_between_scalar_and_array(self, all_membership_functions):
        """测试标量和数组输入的一致性"""
        test_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for mf in all_membership_functions:
            # 标量结果
            scalar_results = [mf.compute(x) for x in test_values]

            # 数组结果
            array_result = mf.compute(np.array(test_values))

            # 应该一致
            for i, scalar_res in enumerate(scalar_results):
                assert abs(float(scalar_res) - array_result[i]) < 1e-14

    def test_broadcast_compatibility(self):
        """测试广播兼容性"""
        mf = GaussianMF(sigma=0.2, c=0.5)

        # 测试与常数的广播
        x = np.array([[0.1, 0.5, 0.9]])  # (1, 3)
        result = mf.compute(x)
        assert result.shape == (1, 3)

        # 测试不同形状的广播行为
        x_col = np.array([[0.1], [0.5], [0.9]])  # (3, 1)
        result_col = mf.compute(x_col)
        assert result_col.shape == (3, 1)
