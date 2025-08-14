#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:14
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import pytest
import numpy as np

from fuzzlab.membership import (
    get_mf_class, create_mf, MembershipFunction,
    TriangularMF, TrapezoidalMF, GaussianMF,
    SigmoidMF, SMF, ZMF, GeneralizedBellMF,
    PiMF, DoubleGaussianMF
)


class TestMembershipFactory:
    """测试隶属函数工厂方法"""

    def test_get_mf_class_with_aliases(self):
        """测试通过别名获取隶属函数类"""
        # 标准别名
        assert get_mf_class('trimf') is TriangularMF
        assert get_mf_class('trapmf') is TrapezoidalMF
        assert get_mf_class('gaussmf') is GaussianMF
        assert get_mf_class('sigmoid') is SigmoidMF
        assert get_mf_class('smf') is SMF
        assert get_mf_class('zmf') is ZMF
        assert get_mf_class('gbellmf') is GeneralizedBellMF
        assert get_mf_class('pimf') is PiMF
        assert get_mf_class('gauss2mf') is DoubleGaussianMF

    def test_get_mf_class_with_class_names(self):
        """测试通过类名获取隶属函数类"""
        assert get_mf_class('TriangularMF') is TriangularMF
        assert get_mf_class('triangularmf') is TriangularMF  # 小写版本
        assert get_mf_class('TrapezoidalMF') is TrapezoidalMF
        assert get_mf_class('GaussianMF') is GaussianMF

    def test_get_mf_class_case_insensitive(self):
        """测试大小写不敏感"""
        assert get_mf_class('TRIMF') is TriangularMF
        assert get_mf_class('TrimF') is TriangularMF
        assert get_mf_class('gaussMF') is GaussianMF

    def test_get_mf_class_unknown_function(self):
        """测试未知隶属函数抛出异常"""
        with pytest.raises(ValueError, match="Unknown membership function 'unknown'"):
            get_mf_class('unknown')

        # 检查错误消息包含可用函数列表
        with pytest.raises(ValueError, match="Available functions are:"):
            get_mf_class('nonexistent')

    def test_create_mf_basic(self):
        """测试基本的隶属函数创建"""
        # 创建三角形隶属函数
        mf, remaining = create_mf('trimf', a=0.0, b=0.5, c=1.0)
        assert isinstance(mf, TriangularMF)
        assert mf.a == 0.0
        assert mf.b == 0.5
        assert mf.c == 1.0
        assert remaining == {}

    def test_create_mf_with_remaining_kwargs(self):
        """测试创建隶属函数时处理额外参数"""
        mf, remaining = create_mf(
            'gaussmf',
            sigma=1.0, c=0.0,  # 隶属函数参数
            q=2, pi=0.1        # 额外参数
        )
        assert isinstance(mf, GaussianMF)
        assert mf.sigma == 1.0
        assert mf.c == 0.0
        assert remaining == {'q': 2, 'pi': 0.1}

    def test_create_mf_invalid_parameters(self):
        """测试创建隶属函数时参数错误"""
        # 缺少必需参数
        with pytest.raises(TypeError):
            create_mf('trimf', a=0.0, b=0.5)  # 缺少 c

        # 参数顺序错误（对于有校验的函数）
        with pytest.raises(ValueError):
            create_mf('trimf', a=1.0, b=0.5, c=0.0)  # a > b > c

    def test_create_mf_all_builtin_functions(self):
        """测试创建所有内置隶属函数"""
        test_cases = [
            ('trimf', {'a': 0, 'b': 0.5, 'c': 1}),
            ('trapmf', {'a': 0, 'b': 0.25, 'c': 0.75, 'd': 1}),
            ('gaussmf', {'sigma': 0.2, 'c': 0.5}),
            ('sigmoid', {'a': 10, 'c': 0.5}),
            ('smf', {'a': 0.2, 'b': 0.8}),
            ('zmf', {'a': 0.2, 'b': 0.8}),
            ('gbellmf', {'a': 2, 'b': 1, 'c': 0.5}),
            ('pimf', {'a': 0.1, 'b': 0.3, 'c': 0.7, 'd': 0.9}),
            ('gauss2mf', {'sigma1': 0.1, 'c1': 0.3, 'sigma2': 0.1, 'c2': 0.7})
        ]

        for name, params in test_cases:
            mf, remaining = create_mf(name, **params)
            assert isinstance(mf, MembershipFunction)
            assert remaining == {}
