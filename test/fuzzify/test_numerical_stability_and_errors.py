#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:28
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np
from unittest.mock import MagicMock

from axisfuzzy.fuzzify import Fuzzifier, fuzzify
from axisfuzzy.membership import TriangularMF, GaussianMF


class TestNumericalStability:
    """测试数值稳定性和边界情况"""

    def test_qrofn_extreme_pi_values(self):
        """测试极端 pi 值的数值稳定性"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.0,  # 极小犹豫因子
            a=0.0, b=0.5, c=1.0
        )

        # 测试高隶属度情况 (md 接近 1)
        result = fuzzifier(0.5)  # 在峰值点
        assert not np.isnan(result.md)
        assert not np.isnan(result.nmd)
        assert not np.isinf(result.md)
        assert not np.isinf(result.nmd)
        assert result.nmd >= 0.0

        # 验证 QROFN 约束：md^q + nmd^q + pi^q <= 1
        constraint = result.md**2 + result.nmd**2 + 0.0**2
        assert constraint <= 1.0 + 1e-10  # 允许数值误差

    def test_qrofn_extreme_pi_high(self):
        """测试高 pi 值的数值稳定性"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.99,  # 极高犹豫因子
            a=0.0, b=0.5, c=1.0
        )

        result = fuzzifier(0.5)
        assert not np.isnan(result.nmd)
        assert result.nmd >= 0.0

        # 在高 pi 情况下，nmd 应该接近 0
        assert result.nmd <= 0.2  # 合理的上界

    def test_qrofn_different_q_values(self):
        """测试不同 q 值的数值稳定性"""
        test_cases = [
            (1, 0.2),   # q=1 (直觉模糊数)
            (2, 0.3),   # q=2 (毕达哥拉斯模糊数)
            (3, 0.1),   # q=3
            (10, 0.05)  # 高阶 q
        ]

        for q, pi in test_cases:
            with pytest.raises(ValueError, match="Parameter 'q' must be an integer >= 1"):
                # 测试 q=0 的错误处理
                Fuzzifier(mf='trimf', mtype='qrofn', q=0, pi=pi, a=0, b=0.5, c=1)

            # 测试有效的 q 值
            fuzzifier = Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                q=q,
                pi=pi,
                a=0.0, b=0.5, c=1.0
            )

            result = fuzzifier(0.5)
            assert not np.isnan(result.md)
            assert not np.isnan(result.nmd)
            assert result.nmd >= 0.0

            # 验证约束
            constraint = result.md**q + result.nmd**q + pi**q
            assert constraint <= 1.0 + 1e-10

    def test_array_numerical_stability(self):
        """测试数组路径的数值稳定性"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            q=2,
            pi=0.1,
            sigma=0.1, c=0.5  # 窄高斯，会产生接近1的md值
        )

        # 创建包含极值的测试数组
        x = np.array([0.5, 0.49, 0.51, 0.0, 1.0])
        result = fuzzifier(x)

        assert result.shape == (5,)
        assert not np.any(np.isnan(result.md))
        assert not np.any(np.isnan(result.nmd))
        assert not np.any(np.isinf(result.md))
        assert not np.any(np.isinf(result.nmd))
        assert np.all(result.nmd >= 0.0)

        # 验证所有元素满足约束
        constraints = result.md**2 + result.nmd**2 + 0.1**2
        assert np.all(constraints <= 1.0 + 1e-10)

    def test_large_array_stability(self):
        """测试大数组的数值稳定性"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=3,
            pi=0.2,
            a=0.0, b=0.5, c=1.0
        )

        # 大数组测试
        large_x = np.linspace(0, 1, 1000)
        result = fuzzifier(large_x)

        assert result.shape == (1000,)
        assert not np.any(np.isnan(result.md))
        assert not np.any(np.isnan(result.nmd))
        assert np.all(result.nmd >= 0.0)

        # 随机抽样验证约束
        sample_indices = np.random.choice(1000, 100, replace=False)
        sample_constraints = (result.md[sample_indices]**3 +
                            result.nmd[sample_indices]**3 +
                            0.2**3)
        assert np.all(sample_constraints <= 1.0 + 1e-10)

    def test_floating_point_precision(self):
        """测试浮点精度边界情况"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=1e-15,  # 极小值
            a=0.0, b=0.5, c=1.0
        )

        # 测试非常接近的值
        close_values = [0.5, 0.5 + 1e-15, 0.5 - 1e-15]
        for val in close_values:
            result = fuzzifier(val)
            assert not np.isnan(result.md)
            assert not np.isnan(result.nmd)

    def test_clipping_behavior(self):
        """测试隶属度值的裁剪行为"""
        # 创建一个会返回超出[0,1]范围值的模拟隶属函数
        mock_mf = MagicMock()
        mock_mf.compute.return_value = 1.5  # 超出范围的值

        fuzzifier = Fuzzifier(
            mf=mock_mf,
            mtype='qrofn',
            q=2,
            pi=0.2
        )

        result = fuzzifier(0.5)
        # 验证 md 被正确裁剪到 [0,1]
        assert 0.0 <= result.md <= 1.0
        assert result.nmd >= 0.0

        # 测试负值裁剪
        mock_mf.compute.return_value = -0.5
        result = fuzzifier(0.5)
        assert result.md >= 0.0


class TestErrorHandling:
    """测试错误处理和异常情况"""

    def test_missing_pi_parameter(self):
        """测试缺失 pi 参数的错误处理"""
        with pytest.raises(ValueError, match="Parameter 'pi'.*is required"):
            fuzzifier = Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                q=2,
                # 缺失 pi 参数
                a=0.0, b=0.5, c=1.0
            )
            fuzzifier(0.5)

    def test_invalid_q_parameter(self):
        """测试无效 q 参数的错误处理"""
        invalid_q_values = [0, -1, -5]

        for invalid_q in invalid_q_values:
            with pytest.raises(ValueError, match="Parameter 'q' must be an integer >= 1"):
                Fuzzifier(
                    mf='trimf',
                    mtype='qrofn',
                    q=invalid_q,
                    pi=0.2,
                    a=0.0, b=0.5, c=1.0
                )

    def test_none_q_parameter(self):
        """测试 None q 参数的错误处理"""
        with pytest.raises(ValueError, match="Parameter 'q' must be an integer >= 1"):
            Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                q=None,
                pi=0.2,
                a=0.0, b=0.5, c=1.0
            )

    def test_pi_none_vs_zero(self):
        """测试 pi=None 和 pi=0.0 的区别"""
        # pi=0.0 应该正常工作
        fuzzifier_zero_pi = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.0,
            a=0.0, b=0.5, c=1.0
        )
        result = fuzzifier_zero_pi(0.5)
        assert isinstance(result.md, float)
        assert isinstance(result.nmd, float)

        # pi=None 应该抛出错误
        with pytest.raises(ValueError, match="Parameter 'pi'.*is required"):
            fuzzifier_none_pi = Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                q=2,
                pi=None,
                a=0.0, b=0.5, c=1.0
            )
            fuzzifier_none_pi(0.5)

    def test_invalid_input_types(self):
        """测试无效输入类型的错误处理"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.2,
            a=0.0, b=0.5, c=1.0
        )

        invalid_inputs = [
            "string",
            {"dict": "value"},
            None,
            complex(1, 2)
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(TypeError, match="Unsupported input type"):
                fuzzifier(invalid_input)

    def test_unknown_mtype_error(self):
        """测试未知 mtype 的错误处理"""
        with pytest.raises((ValueError, KeyError)):
            Fuzzifier(
                mf='trimf',
                mtype='unknown_mtype',
                q=2,
                pi=0.2,
                a=0.0, b=0.5, c=1.0
            )

    def test_unknown_method_error(self):
        """测试未知 method 的错误处理"""
        with pytest.raises((ValueError, KeyError)):
            Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                method='unknown_method',
                q=2,
                pi=0.2,
                a=0.0, b=0.5, c=1.0
            )

    def test_conflicting_mf_parameters(self):
        """测试隶属函数参数冲突的错误处理"""
        # 传入已实例化的隶属函数，同时传入构造参数应该报错
        mf_instance = TriangularMF(a=0.0, b=0.5, c=1.0)

        with pytest.raises(ValueError, match="Cannot provide.*parameters.*when.*instance"):
            Fuzzifier(
                mf=mf_instance,
                mtype='qrofn',
                q=2,
                pi=0.2,
                a=0.2, b=0.6, c=0.8  # 冲突的参数
            )

    def test_unknown_parameter_error(self):
        """测试未知参数的错误处理"""
        with pytest.raises(ValueError, match="Unknown parameter.*unknown_param"):
            Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                q=2,
                pi=0.2,
                a=0.0, b=0.5, c=1.0,
                unknown_param=123  # 未知参数
            )

    def test_array_dimension_edge_cases(self):
        """测试数组维度边界情况"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.2,
            a=0.0, b=0.5, c=1.0
        )

        # 空数组
        empty_array = np.array([])
        result = fuzzifier(empty_array)
        assert result.shape == (0,)

        # 单元素数组
        single_element = np.array([0.5])
        result = fuzzifier(single_element)
        assert result.shape == (1,)
        assert not np.isnan(result.md[0])
        assert not np.isnan(result.nmd[0])

    def test_zero_dimensional_array(self):
        """测试零维数组的处理"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.2,
            a=0.0, b=0.5, c=1.0
        )

        # 零维数组应该走标量路径
        zero_d_array = np.array(0.5)
        result = fuzzifier(zero_d_array)

        # 应该返回 Fuzznum 而不是 Fuzzarray
        assert hasattr(result, 'md')
        assert hasattr(result, 'nmd')
        assert isinstance(result.md, float)
        assert isinstance(result.nmd, float)


class TestEdgeCases:
    """测试边界情况和特殊场景"""

    def test_convenience_function_error_propagation(self):
        """测试便捷函数的错误传播"""
        with pytest.raises(ValueError, match="Parameter 'pi'.*is required"):
            fuzzify(
                x=0.5,
                mf='trimf',
                mtype='qrofn',
                q=2,
                # 缺失 pi
                a=0.0, b=0.5, c=1.0
            )

    def test_extreme_array_shapes(self):
        """测试极端数组形状"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.1,
            a=0.0, b=0.5, c=1.0
        )

        # 测试多维数组被扁平化处理
        multi_d = np.array([[0.2, 0.5], [0.8, 0.3]])
        result = fuzzifier(multi_d)
        assert result.shape == (4,)  # 被扁平化

        # 验证所有值都有效
        assert not np.any(np.isnan(result.md))
        assert not np.any(np.isnan(result.nmd))

    def test_mixed_input_types_through_numpy(self):
        """测试通过 numpy 转换的混合输入类型"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=0.2,
            a=0.0, b=0.5, c=1.0
        )

        # 混合类型的列表（int, float）
        mixed_list = [0, 0.5, 1, 0.25]
        result = fuzzifier(mixed_list)
        assert result.shape == (4,)
        assert not np.any(np.isnan(result.md))
        assert not np.any(np.isnan(result.nmd))

        # 元组输入
        tuple_input = (0.2, 0.5, 0.8)
        result = fuzzifier(tuple_input)
        assert result.shape == (3,)

    def test_numerical_precision_boundaries(self):
        """测试数值精度边界"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            q=2,
            pi=1e-10,  # 接近机器精度
            a=0.0, b=0.5, c=1.0
        )

        # 测试接近边界的值
        boundary_values = [1e-15, 1.0 - 1e-15, 0.5 + 1e-15, 0.5 - 1e-15]
        for val in boundary_values:
            result = fuzzifier(val)
            assert not np.isnan(result.md)
            assert not np.isnan(result.nmd)
            assert np.isfinite(result.md)
            assert np.isfinite(result.nmd)
