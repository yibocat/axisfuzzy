#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
隶属函数测试的共享配置和 fixtures

本模块提供隶属函数测试所需的共享配置、fixtures 和工具函数。
包括标准测试数据、参数组合、数值精度设置等。
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple

# 数值精度设置
TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8


@pytest.fixture
def tolerance():
    """数值比较的绝对容差"""
    return TOLERANCE


@pytest.fixture
def rtol():
    """数值比较的相对容差"""
    return RELATIVE_TOLERANCE


@pytest.fixture
def standard_x_values():
    """标准测试输入值集合"""
    return np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


@pytest.fixture
def extended_x_values():
    """扩展测试输入值集合（包含边界外值）"""
    return np.array([-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0])


@pytest.fixture
def large_x_array():
    """大型数组用于性能测试"""
    return np.linspace(-2, 3, 10000)


@pytest.fixture
def triangular_params():
    """三角隶属函数的标准参数组合"""
    return {
        'standard': {'a': 0.0, 'b': 0.5, 'c': 1.0},
        'left_shoulder': {'a': 0.0, 'b': 0.0, 'c': 0.5},
        'right_shoulder': {'a': 0.5, 'b': 1.0, 'c': 1.0},
        'narrow': {'a': 0.4, 'b': 0.5, 'c': 0.6},
        'wide': {'a': -1.0, 'b': 0.5, 'c': 2.0}
    }


@pytest.fixture
def trapezoidal_params():
    """梯形隶属函数的标准参数组合"""
    return {
        'standard': {'a': 0.0, 'b': 0.25, 'c': 0.75, 'd': 1.0},
        'triangular_degenerate': {'a': 0.0, 'b': 0.5, 'c': 0.5, 'd': 1.0},
        'left_shoulder': {'a': 0.0, 'b': 0.0, 'c': 0.3, 'd': 0.6},
        'right_shoulder': {'a': 0.4, 'b': 0.7, 'c': 1.0, 'd': 1.0},
        'narrow_core': {'a': 0.2, 'b': 0.45, 'c': 0.55, 'd': 0.8}
    }


@pytest.fixture
def gaussian_params():
    """高斯隶属函数的标准参数组合"""
    return {
        'standard': {'sigma': 0.2, 'c': 0.5},
        'narrow': {'sigma': 0.1, 'c': 0.5},
        'wide': {'sigma': 0.5, 'c': 0.5},
        'left_shifted': {'sigma': 0.2, 'c': 0.2},
        'right_shifted': {'sigma': 0.2, 'c': 0.8}
    }


@pytest.fixture
def sigmoid_params():
    """Sigmoid隶属函数的标准参数组合"""
    return {
        'standard': {'k': 10.0, 'c': 0.5},
        'steep': {'k': 20.0, 'c': 0.5},
        'gentle': {'k': 5.0, 'c': 0.5},
        'left_shifted': {'k': 10.0, 'c': 0.3},
        'right_shifted': {'k': 10.0, 'c': 0.7},
        'negative_slope': {'k': -10.0, 'c': 0.5}
    }


@pytest.fixture
def smf_params():
    """S型隶属函数的标准参数组合"""
    return {
        'standard': {'a': 0.2, 'b': 0.8},
        'narrow': {'a': 0.4, 'b': 0.6},
        'wide': {'a': 0.0, 'b': 1.0},
        'left_shifted': {'a': 0.0, 'b': 0.5},
        'right_shifted': {'a': 0.5, 'b': 1.0}
    }


@pytest.fixture
def zmf_params():
    """Z型隶属函数的标准参数组合"""
    return {
        'standard': {'a': 0.2, 'b': 0.8},
        'narrow': {'a': 0.4, 'b': 0.6},
        'wide': {'a': 0.0, 'b': 1.0},
        'left_shifted': {'a': 0.0, 'b': 0.5},
        'right_shifted': {'a': 0.5, 'b': 1.0}
    }


@pytest.fixture
def pi_params():
    """Π型隶属函数的标准参数组合"""
    return {
        'standard': {'a': 0.1, 'b': 0.3, 'c': 0.7, 'd': 0.9},
        'narrow': {'a': 0.2, 'b': 0.4, 'c': 0.6, 'd': 0.8},
        'wide': {'a': 0.0, 'b': 0.2, 'c': 0.8, 'd': 1.0},
        'symmetric': {'a': 0.0, 'b': 0.25, 'c': 0.75, 'd': 1.0}
    }


@pytest.fixture
def bell_params():
    """广义贝尔隶属函数的标准参数组合"""
    return {
        'standard': {'a': 0.2, 'b': 2.0, 'c': 0.5},
        'narrow': {'a': 0.1, 'b': 4.0, 'c': 0.5},
        'wide': {'a': 0.4, 'b': 1.0, 'c': 0.5},
        'steep': {'a': 0.2, 'b': 5.0, 'c': 0.5},
        'gentle': {'a': 0.3, 'b': 1.0, 'c': 0.5}
    }


@pytest.fixture
def double_gaussian_params():
    """双高斯隶属函数的标准参数组合"""
    return {
        'standard': {'sigma1': 0.1, 'c1': 0.3, 'sigma2': 0.15, 'c2': 0.7},
        'symmetric': {'sigma1': 0.1, 'c1': 0.4, 'sigma2': 0.1, 'c2': 0.6},
        'asymmetric': {'sigma1': 0.05, 'c1': 0.2, 'sigma2': 0.2, 'c2': 0.8},
        'overlapping': {'sigma1': 0.2, 'c1': 0.4, 'sigma2': 0.2, 'c2': 0.6}
    }


@pytest.fixture
def invalid_params():
    """无效参数组合用于测试参数验证"""
    return {
        'triangular': [
            {'a': 1.0, 'b': 0.5, 'c': 0.0},  # 错误顺序
            {'a': 0.0, 'b': 1.0, 'c': 0.5},  # b > c
            {'a': 0.5, 'b': 0.0, 'c': 1.0},  # a > b
        ],
        'trapezoidal': [
            {'a': 1.0, 'b': 0.5, 'c': 0.3, 'd': 0.0},  # 错误顺序
            {'a': 0.0, 'b': 0.8, 'c': 0.5, 'd': 1.0},  # b > c
        ],
        'gaussian': [
            {'sigma': -0.1, 'c': 0.5},  # 负sigma
            {'sigma': 0.0, 'c': 0.5},   # 零sigma
        ],
        'smf': [
            {'a': 0.8, 'b': 0.2},  # a > b
        ],
        'zmf': [
            {'a': 0.8, 'b': 0.2},  # a > b
        ],
        'bell': [
            {'a': -0.1, 'b': 2.0, 'c': 0.5},  # 负a
            {'a': 0.2, 'b': -1.0, 'c': 0.5},  # 负b
            {'a': 0.0, 'b': 2.0, 'c': 0.5},   # 零a
            {'a': 0.2, 'b': 0.0, 'c': 0.5},   # 零b
        ],
        'double_gaussian': [
            {'sigma1': -0.1, 'c1': 0.3, 'sigma2': 0.15, 'c2': 0.7},  # 负sigma1
            {'sigma1': 0.1, 'c1': 0.3, 'sigma2': -0.15, 'c2': 0.7},  # 负sigma2
            {'sigma1': 0.0, 'c1': 0.3, 'sigma2': 0.15, 'c2': 0.7},   # 零sigma1
        ]
    }


@pytest.fixture
def boundary_test_cases():
    """边界条件测试用例"""
    return {
        'extreme_values': np.array([-1e6, -1e3, -10, 10, 1e3, 1e6]),
        'near_zero': np.array([-1e-10, -1e-15, 0.0, 1e-15, 1e-10]),
        'special_values': np.array([np.inf, -np.inf, np.nan])
    }


@pytest.fixture
def membership_function_aliases():
    """隶属函数别名映射"""
    return {
        'triangular': ['trimf', 'triangularmf', 'TriangularMF'],
        'trapezoidal': ['trapmf', 'trapezoidalmf', 'TrapezoidalMF'],
        'gaussian': ['gaussmf', 'gaussianmf', 'GaussianMF'],
        'sigmoid': ['sigmf', 'sigmoidmf', 'SigmoidMF'],
        'smf': ['SMF'],
        'zmf': ['ZMF'],
        'pimf': ['pi', 'PiMF'],
        'gbellmf': ['bell', 'generalizedbell', 'GeneralizedBellMF'],
        'gauss2mf': ['dgauss', 'doublegaussian', 'DoubleGaussianMF']
    }


def assert_membership_properties(mf_func, x_values, tolerance=TOLERANCE):
    """
    验证隶属函数的基本数学性质
    
    Parameters
    ----------
    mf_func : callable
        隶属函数
    x_values : array_like
        测试输入值
    tolerance : float
        数值容差
    """
    y_values = mf_func(x_values)
    
    # 检查输出范围 [0, 1]
    assert np.all(y_values >= -tolerance), "隶属度值不能小于0"
    assert np.all(y_values <= 1 + tolerance), "隶属度值不能大于1"
    
    # 检查输出形状
    assert y_values.shape == np.asarray(x_values).shape, "输出形状必须与输入形状一致"
    
    # 检查数值类型
    assert np.issubdtype(y_values.dtype, np.floating), "输出必须是浮点数类型"


def assert_parameter_update(mf, new_params, tolerance=TOLERANCE):
    """
    验证参数更新功能
    
    Parameters
    ----------
    mf : MembershipFunction
        隶属函数实例
    new_params : dict
        新参数
    tolerance : float
        数值容差
    """
    old_params = mf.get_parameters().copy()
    mf.set_parameters(**new_params)
    updated_params = mf.get_parameters()
    
    # 检查参数是否正确更新
    for key, value in new_params.items():
        if key in updated_params:
            assert abs(updated_params[key] - value) < tolerance, f"参数 {key} 更新失败"
    
    # 检查未更新的参数保持不变
    for key, value in old_params.items():
        if key not in new_params:
            assert abs(updated_params[key] - value) < tolerance, f"参数 {key} 意外改变"


def generate_test_data(x_range=(-2, 3), num_points=1000):
    """
    生成测试数据
    
    Parameters
    ----------
    x_range : tuple
        输入值范围
    num_points : int
        数据点数量
        
    Returns
    -------
    numpy.ndarray
        测试输入数据
    """
    return np.linspace(x_range[0], x_range[1], num_points)