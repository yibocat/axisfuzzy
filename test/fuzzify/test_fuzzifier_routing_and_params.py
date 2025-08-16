#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:28
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np
from unittest.mock import Mock, patch

from axisfuzzy.fuzzify import Fuzzifier, fuzzify
from axisfuzzy.membership import TriangularMF, GaussianMF, MembershipFunction
from axisfuzzy.core import Fuzznum, Fuzzarray


class MockMembershipFunction(MembershipFunction):
    """用于测试的模拟隶属函数"""

    def __init__(self, param1=1.0, param2=2.0, name=None):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
        self.parameters = {'param1': param1, 'param2': param2}

    def compute(self, x):
        return np.clip(np.asarray(x) * 0.5, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value


class TestFuzzifierInitialization:

    def test_init_with_mf_instance(self):
        """测试使用隶属函数实例初始化"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        fuzzifier = Fuzzifier(
            mf=mf,
            mtype='qrofn',
            method='default',
            q=2,
            pi=0.2
        )

        assert fuzzifier.mf is mf
        assert fuzzifier.mtype == 'qrofn'
        assert fuzzifier.method == 'default'
        assert hasattr(fuzzifier, 'strategy')

    def test_init_with_mf_string(self):
        """测试使用隶属函数字符串初始化"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,  # 隶属函数参数
            q=2, pi=0.2           # 策略参数
        )

        assert isinstance(fuzzifier.mf, TriangularMF)
        assert fuzzifier.mf.a == 0.0
        assert fuzzifier.mf.b == 0.5
        assert fuzzifier.mf.c == 1.0

    def test_init_default_mtype(self):
        """测试默认 mtype"""
        with patch('axisfuzzy.config.get_config') as mock_config:
            mock_config.return_value.DEFAULT_MTYPE = 'qrofn'

            fuzzifier = Fuzzifier(
                mf='trimf',
                a=0.0, b=0.5, c=1.0,
                q=2, pi=0.2
            )

            assert fuzzifier.mtype == 'qrofn'

    def test_init_default_method(self):
        """测试默认 method"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        assert fuzzifier.method == 'default'

    def test_parameter_separation(self):
        """测试参数自动分拣"""
        # 这里我们需要 mock 一些内部机制来测试参数分拣
        with patch('axisfuzzy.fuzzify.fuzzifier.inspect') as mock_inspect:
            # Mock strategy 参数
            mock_strategy_sig = Mock()
            mock_strategy_sig.parameters.keys.return_value = ['self', 'q', 'pi']

            # Mock MF 参数
            mock_mf_sig = Mock()
            mock_mf_sig.parameters.keys.return_value = ['self', 'a', 'b', 'c', 'name']

            mock_inspect.signature.side_effect = [mock_strategy_sig, mock_mf_sig]

            # 实际测试中，我们验证构造成功即可
            fuzzifier = Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                a=0.0, b=0.5, c=1.0,  # 应该给隶属函数
                q=2, pi=0.2           # 应该给策略
            )

            assert fuzzifier is not None

    def test_mf_instance_with_constructor_params_error(self):
        """测试传入隶属函数实例时还传构造参数应报错"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)

        with pytest.raises(ValueError, match="already instantiated.*cannot provide.*parameters"):
            Fuzzifier(
                mf=mf,
                mtype='qrofn',
                a=1.0,  # 这应该导致错误
                q=2, pi=0.2
            )

    def test_unknown_parameter_error(self):
        """测试未知参数报错"""
        with pytest.raises(ValueError, match="Unknown parameter.*unknown_param"):
            Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                a=0.0, b=0.5, c=1.0,
                q=2, pi=0.2,
                unknown_param=123  # 这应该导致错误
            )

    def test_unknown_mtype_error(self):
        """测试未知 mtype 报错"""
        with pytest.raises(ValueError, match="No default method.*unknown_mtype"):
            Fuzzifier(
                mf='trimf',
                mtype='unknown_mtype',
                a=0.0, b=0.5, c=1.0
            )

    def test_unknown_method_error(self):
        """测试未知 method 报错"""
        with pytest.raises(ValueError, match="Strategy.*not found"):
            Fuzzifier(
                mf='trimf',
                mtype='qrofn',
                method='unknown_method',
                a=0.0, b=0.5, c=1.0,
                q=2, pi=0.2
            )


class TestFuzzifierCall:

    def test_call_with_scalar_int(self):
        """测试标量 int 输入"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        result = fuzzifier(5)
        assert isinstance(result, Fuzznum)

    def test_call_with_scalar_float(self):
        """测试标量 float 输入"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        result = fuzzifier(0.7)
        assert isinstance(result, Fuzznum)

    def test_call_with_numpy_scalar(self):
        """测试 numpy 标量输入"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        # numpy 各种标量类型
        result1 = fuzzifier(np.float64(0.7))
        result2 = fuzzifier(np.int32(5))
        result3 = fuzzifier(np.float32(0.3))

        assert isinstance(result1, Fuzznum)
        assert isinstance(result2, Fuzznum)
        assert isinstance(result3, Fuzznum)

    def test_call_with_zero_dim_array(self):
        """测试0维数组作为标量处理"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        zero_dim_array = np.array(0.7)  # 0维数组
        result = fuzzifier(zero_dim_array)

        assert isinstance(result, Fuzznum)

    def test_call_with_list(self):
        """测试列表输入"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        result = fuzzifier([0.1, 0.5, 0.9])
        assert isinstance(result, Fuzzarray)
        assert result.shape == (3,)

    def test_call_with_tuple(self):
        """测试元组输入"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        result = fuzzifier((0.2, 0.6))
        assert isinstance(result, Fuzzarray)
        assert result.shape == (2,)

    def test_call_with_ndarray(self):
        """测试 numpy 数组输入"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        x = np.array([0.1, 0.3, 0.7, 0.9])
        result = fuzzifier(x)

        assert isinstance(result, Fuzzarray)
        assert result.shape == (4,)

    def test_call_with_multidim_array(self):
        """测试多维数组输入"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        x = np.array([[0.1, 0.3], [0.7, 0.9]])
        result = fuzzifier(x)

        assert isinstance(result, Fuzzarray)
        assert result.shape == (2, 2)

    def test_call_with_invalid_type(self):
        """测试无效输入类型"""
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        with pytest.raises(TypeError, match="Unsupported input type.*dict"):
            fuzzifier({'invalid': 'input'})


class TestFuzzifyFunction:

    def test_fuzzify_convenience_function(self):
        """测试便捷函数 fuzzify"""
        result = fuzzify(
            x=0.5,
            mf='trimf',
            mtype='qrofn',
            a=0.0, b=0.5, c=1.0,
            q=2, pi=0.2
        )

        assert isinstance(result, Fuzznum)

    def test_fuzzify_with_array(self):
        """测试 fuzzify 函数处理数组"""
        result = fuzzify(
            x=[0.2, 0.5, 0.8],
            mf='gaussmf',
            mtype='qrofn',
            sigma=0.2, c=0.5,
            q=3, pi=0.1
        )

        assert isinstance(result, Fuzzarray)
        assert result.shape == (3,)

    def test_fuzzify_is_equivalent_to_fuzzifier(self):
        """测试 fuzzify 函数与 Fuzzifier 等价"""
        # 相同的参数
        x = 0.6
        mf_name = 'trimf'
        mtype = 'qrofn'
        kwargs = {'a': 0.0, 'b': 0.5, 'c': 1.0, 'q': 2, 'pi': 0.2}

        # 使用 fuzzify 函数
        result1 = fuzzify(x=x, mf=mf_name, mtype=mtype, **kwargs)

        # 使用 Fuzzifier
        fuzzifier = Fuzzifier(mf=mf_name, mtype=mtype, **kwargs)
        result2 = fuzzifier(x)

        # 结果应该相同
        assert type(result1) == type(result2)
        assert isinstance(result1, Fuzznum)

        # 比较数值（可能需要根据实际实现调整）
        info1 = result1.get_info()
        info2 = result2.get_info()
        assert abs(info1['md'] - info2['md']) < 1e-10
        assert abs(info1['nmd'] - info2['nmd']) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__])