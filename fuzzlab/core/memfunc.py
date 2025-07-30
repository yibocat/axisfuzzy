#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 22:46
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Tuple

import numpy as np


class MembershipFunc:

    def __init__(self):

        self._membership_functions = {
            'sigmf': self._sigmf,
            'trimf': self._trimf,
            'zmf': self._zmf,
            'trapmf': self._trapmf,
            'smf': self._smf,
            'gaussmf': self._gaussmf,
            'gauss2mf': self._gauss2mf,
            'gbellmf': self._gbellmf,
        }

    @staticmethod
    def _sigmf(x: float | np.ndarray,
               params: Tuple[float, float]) -> float | np.ndarray:
        """
        基本的Sigmoid函数

        params 是一个包含两个参数的元组，表示Sigmoid函数的参数。其中第一个值
        表示左右偏差或位移。第二个值表示活跃区域，其值越大，曲线越平缓。

        Args:
            x:  float or np.ndarray
                the independent variable
            *params:
                two parameters, representing the sigmoid
                function parameters.

        Returns:
            y:  np.float64 or np.ndarray
                the sigmoid function value

        """
        assert len(params) == 2, 'parameter list length must be 2.'
        return 1. / (1. + np.exp(- params[0] * (x - params[1])))

    @staticmethod
    def _trimf(x: float | np.ndarray,
               params: Tuple[float, float, float]) -> float | np.ndarray:
        """
        Triangular function, 'x' represents the independent variable, 'abc'
        is a 3-valued array that satisfies a<=b<=c, that is, the three angles
        when the independent variable takes three values.

        Args:
            x:  float or np.ndarray
                the independent variable
            *params:
                3-valued array that satisfies a<=b<=c, that is, the three angles

        Returns:
            y:  np.float np.ndarray
                the triangular function value
        """
        assert len(params) == 3, 'parameter must have exactly three elements.'
        assert params[0] <= params[1] <= params[2], 'parameters requires the three elements a <= b <= c.'

        if isinstance(x, np.ndarray):
            x = np.array([x])
        y = np.zeros(len(x))

        # Left side
        if params[0] != params[1]:
            idx = np.nonzero(np.logical_and(params[0] < x, x < params[1]))[0]
            y[idx] = (x[idx] - params[0]) / float(params[1] - params[0])

        # Right side
        if params[1] != params[2]:
            idx = np.nonzero(np.logical_and(params[1] < x, x < params[2]))[0]
            y[idx] = (params[2] - x[idx]) / float(params[2] - params[1])

        idx = np.nonzero(x == params[1])
        y[idx] = 1
        if len(y) == 1:
            return y[0]
        return y

    @staticmethod
    def _zmf(x: float | np.ndarray,
             params: Tuple[float, float]) -> float | np.ndarray:
        """
        Z 型隶属函数。

        Args:
            x (float or np.ndarray): 独立变量。
            params (list): 包含两个参数的列表 [a, b]，满足 a <= b。

        Returns:
            float or np.ndarray: Z 型函数值。
        """
        assert len(params) == 2, 'zmf: 参数列表长度必须为 2。'
        a, b = params[0], params[1]
        assert a <= b, 'zmf: 要求 a <= b。'

        x_arr = np.atleast_1d(x)
        y = np.ones_like(x_arr, dtype=float)

        # x < a
        y[x_arr < a] = 1.0

        # a <= x < (a+b)/2
        idx = np.logical_and(a <= x_arr, x_arr < (a + b) / 2.)
        y[idx] = 1 - 2. * ((x_arr[idx] - a) / (b - a)) ** 2.

        # (a+b)/2 <= x <= b
        idx = np.logical_and((a + b) / 2. <= x_arr, x_arr <= b)
        y[idx] = 2. * ((x_arr[idx] - b) / (b - a)) ** 2.

        # x > b
        y[x_arr > b] = 0.0

        return y[0] if np.isscalar(x) else y

    @staticmethod
    def _trapmf(x: float | np.ndarray,
                params: Tuple[float, float, float, float]) -> float | np.ndarray:
        """
        梯形隶属函数。

        Args:
            x (float or np.ndarray): 独立变量。
            params (list): 包含四个参数的列表 [a, b, c, d]，满足 a <= b <= c <= d。

        Returns:
            float or np.ndarray: 梯形函数值。
        """
        assert len(params) == 4, 'trapmf: 参数必须恰好有四个元素。'
        a, b, c, d = params[0], params[1], params[2], params[3]
        assert a <= b <= c <= d, 'trapmf: 参数要求四个元素满足 a <= b <= c <= d。'

        x_arr = np.atleast_1d(x)
        y = np.zeros_like(x_arr, dtype=float)

        # x <= a 或 x >= d
        y[np.logical_or(x_arr <= a, x_arr >= d)] = 0.0

        # a < x < b
        if a != b:
            idx = np.logical_and(a < x_arr, x_arr < b)
            y[idx] = (x_arr[idx] - a) / (b - a)
        # x = b
        y[x_arr == b] = 1.0

        # b <= x <= c
        idx = np.logical_and(b <= x_arr, x_arr <= c)
        y[idx] = 1.0

        # c < x < d
        if c != d:
            idx = np.logical_and(c < x_arr, x_arr < d)
            y[idx] = (d - x_arr[idx]) / (d - c)
        # x = c
        y[x_arr == c] = 1.0

        return y[0] if np.isscalar(x) else y

    @staticmethod
    def _smf(x: float | np.ndarray,
             params: Tuple[float, float]) -> float | np.ndarray:
        """
        S 型隶属函数。

        Args:
            x (float or np.ndarray): 独立变量。
            params (list): 包含两个参数的列表 [a, b]，满足 a <= b。

        Returns:
            float or np.ndarray: S 型函数值。
        """
        assert len(params) == 2, 'smf: 参数列表长度必须为 2。'
        a, b = params[0], params[1]
        assert a <= b, 'smf: 要求 a <= b。'

        x_arr = np.atleast_1d(x)
        y = np.zeros_like(x_arr, dtype=float)

        # x <= a
        y[x_arr <= a] = 0.0

        # a <= x < (a+b)/2
        idx = np.logical_and(a <= x_arr, x_arr < (a + b) / 2.)
        y[idx] = 2. * ((x_arr[idx] - a) / (b - a)) ** 2.

        # (a+b)/2 <= x <= b
        idx = np.logical_and((a + b) / 2. <= x_arr, x_arr <= b)
        y[idx] = 1 - 2. * ((x_arr[idx] - b) / (b - a)) ** 2.

        # x > b
        y[x_arr > b] = 1.0

        return y[0] if np.isscalar(x) else y

    @staticmethod
    def _gaussmf(x: float | np.ndarray,
                 params: Tuple[float, float]) -> float | np.ndarray:
        """
        高斯隶属函数。

        Args:
            x (float or np.ndarray): 独立变量。
            params (list): 包含两个参数的列表 [mean, std_dev]，mean 为均值，std_dev 为标准差。

        Returns:
            float or np.ndarray: 高斯函数值。
        """
        assert len(params) == 2, 'gaussmf: 参数列表长度必须为 2。'
        mean, std_dev = params[0], params[1]
        return np.exp(-((x - mean) ** 2.) / (2 * std_dev ** 2.))

    def _gauss2mf(self,
                  x: float | np.ndarray,
                  params: Tuple[float, float, float, float]) -> float | np.ndarray:
        """
        双高斯隶属函数。

        Args:
            x (float or np.ndarray): 独立变量。
            params (list): 包含四个参数的列表 [mean1, std_dev1, mean2, std_dev2]，
                           mean1, std_dev1 为第一个高斯函数的均值和标准差，
                           mean2, std_dev2 为第二个高斯函数的均值和标准差。
                           要求 mean1 <= mean2。

        Returns:
            float or np.ndarray: 双高斯函数值。
        """
        assert len(params) == 4, 'gauss2mf: 参数列表长度必须为 4。'
        mean1, std_dev1, mean2, std_dev2 = params[0], params[1], params[2], params[3]
        assert mean1 <= mean2, 'gauss2mf: 要求 mean1 <= mean2。'

        x_arr = np.atleast_1d(x)
        y = np.ones_like(x_arr, dtype=float)

        # x <= mean1
        idx1 = x_arr <= mean1
        y[idx1] = self._gaussmf(x_arr[idx1], (mean1, std_dev1))

        # x > mean2
        idx2 = x_arr > mean2
        y[idx2] = self._gaussmf(x_arr[idx2], (mean2, std_dev2))

        # mean1 < x <= mean2 的部分 y 保持为 1.0

        return y[0] if np.isscalar(x) else y

    @staticmethod
    def _gbellmf(x: float | np.ndarray,
                 params: Tuple[float, float, float]) -> float | np.ndarray:
        """
        广义钟型隶属函数 (Generalized Bell function)。

        Args:
            x (float or np.ndarray): 独立变量。
            params (list): 包含三个参数的列表 [a, b, c]，a 为宽度，b 为斜率，c 为中心点。

        Returns:
            float or np.ndarray: 广义钟型函数值。
        """
        assert len(params) == 3, 'gbellmf: 参数列表长度必须为 3。'
        a, b, c = params[0], params[1], params[2]
        return 1. / (1. + np.abs((x - c) / a) ** (2 * b))

    def __call__(self,
                 mf_type: str,
                 x: float | np.ndarray,
                 params: Tuple[float, float]
                         | Tuple[float, float, float]
                         | Tuple[float, float, float, float]) -> float | np.ndarray:
        """
        通过 __call__ 方法计算指定隶属函数的隶属度。

        Args:
            mf_type (str): 隶属函数类型，例如 'sigmf', 'trimf' 等。
            x (float or np.ndarray): 独立变量，可以是单个数值或 NumPy 数组。
            params (list): 隶属函数所需的参数列表。

        Returns:
            float or np.ndarray: 对应 x 的隶属度值。

        Raises:
            ValueError: 如果指定的隶属函数类型不存在。
        """
        if mf_type not in self._membership_functions:
            raise ValueError(
                f"不支持的隶属函数类型: {mf_type}. 支持的类型有: {list(self._membership_functions.keys())}")
        return self._membership_functions[mf_type](x, params)
