#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/14 11:41
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import numpy as np
from .base import MembershipFunction


class SigmoidMF(MembershipFunction):
    """Sigmoid隶属函数"""

    def __init__(self, a: float = 1.0, c: float = 0.0, name: str = None):
        super().__init__(name)
        self.a = a  # 斜率参数
        self.c = c  # 中心点
        self.parameters = {'a': a, 'c': c}

    def compute(self, x: np.ndarray) -> np.ndarray:
        """计算Sigmoid隶属度值"""
        return 1 / (1 + np.exp(-self.a * (x - self.c)))

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class TriangularMF(MembershipFunction):
    """三角形隶属函数"""

    def __init__(self, a: float, b: float, c: float, name: str = None):
        super().__init__(name)
        self.a = a  # 左端点
        self.b = b  # 峰值点
        self.c = c  # 右端点
        self.parameters = {'a': a, 'b': b, 'c': c}

    def compute(self, x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # 左侧上升
        mask1 = (x >= self.a) & (x <= self.b)
        result[mask1] = (x[mask1] - self.a) / (self.b - self.a)

        # 右侧下降
        mask2 = (x > self.b) & (x <= self.c)
        result[mask2] = (self.c - x[mask2]) / (self.c - self.b)

        return result

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class TrapezoidalMF(MembershipFunction):
    """梯形隶属函数"""

    def __init__(self, a: float, b: float, c: float, d: float, name: str = None):
        super().__init__(name)
        self.a = a  # 左端点
        self.b = b  # 左平台点
        self.c = c  # 右平台点
        self.d = d  # 右端点
        self.parameters = {'a': a, 'b': b, 'c': c, 'd': d}

    def compute(self, x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # 左侧上升
        mask1 = (x >= self.a) & (x <= self.b)
        result[mask1] = (x[mask1] - self.a) / (self.b - self.a)

        # 平台区域
        mask2 = (x > self.b) & (x <= self.c)
        result[mask2] = 1.0

        # 右侧下降
        mask3 = (x > self.c) & (x <= self.d)
        result[mask3] = (self.d - x[mask3]) / (self.d - self.c)

        return result

    def set_parameters(self, **kwargs):
        for param in ['a', 'b', 'c', 'd']:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]


class GaussianMF(MembershipFunction):
    """高斯隶属函数"""

    def __init__(self, sigma: float, c: float, name: str = None):
        super().__init__(name)
        self.sigma = sigma  # 标准差
        self.c = c  # 中心点
        self.parameters = {'sigma': sigma, 'c': c}

    def compute(self, x):
        return np.exp(-0.5 * ((x - self.c) / self.sigma) ** 2)

    def set_parameters(self, **kwargs):
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
            self.parameters['sigma'] = self.sigma
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class SMF(MembershipFunction):
    """S型隶属函数"""

    def __init__(self, a: float, b: float, name: str = None):
        super().__init__(name)
        self.a = a  # 左端点
        self.b = b  # 右端点
        self.parameters = {'a': a, 'b': b}

        if a >= b:
            raise ValueError("Parameter 'a' must be less than 'b'")

    def compute(self, x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # 计算中点
        mid = (self.a + self.b) / 2

        # x <= a: 隶属度为0
        mask1 = x <= self.a
        result[mask1] = 0.0

        # a < x <= mid: 二次增长
        mask2 = (x > self.a) & (x <= mid)
        result[mask2] = 2 * ((x[mask2] - self.a) / (self.b - self.a)) ** 2

        # mid < x < b: 二次减缓增长
        mask3 = (x > mid) & (x < self.b)
        result[mask3] = 1 - 2 * ((x[mask3] - self.b) / (self.b - self.a)) ** 2

        # x >= b: 隶属度为1
        mask4 = x >= self.b
        result[mask4] = 1.0

        return result

    def set_parameters(self, **kwargs):
        if 'a' in kwargs and 'b' in kwargs:
            if kwargs['a'] >= kwargs['b']:
                raise ValueError("Parameter 'a' must be less than 'b'")

        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b


class ZMF(MembershipFunction):
    """Z型隶属函数"""

    def __init__(self, a: float, b: float, name: str = None):
        super().__init__(name)
        self.a = a  # 左端点
        self.b = b  # 右端点
        self.parameters = {'a': a, 'b': b}

        if a >= b:
            raise ValueError("Parameter 'a' must be less than 'b'")

    def compute(self, x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # 计算中点
        mid = (self.a + self.b) / 2

        # x <= a: 隶属度为1
        mask1 = x <= self.a
        result[mask1] = 1.0

        # a < x <= mid: 二次减少
        mask2 = (x > self.a) & (x <= mid)
        result[mask2] = 1 - 2 * ((x[mask2] - self.a) / (self.b - self.a)) ** 2

        # mid < x < b: 二次加速减少
        mask3 = (x > mid) & (x < self.b)
        result[mask3] = 2 * ((x[mask3] - self.b) / (self.b - self.a)) ** 2

        # x >= b: 隶属度为0
        mask4 = x >= self.b
        result[mask4] = 0.0

        return result

    def set_parameters(self, **kwargs):
        if 'a' in kwargs and 'b' in kwargs:
            if kwargs['a'] >= kwargs['b']:
                raise ValueError("Parameter 'a' must be less than 'b'")

        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b


class DoubleGaussianMF(MembershipFunction):
    """双高斯隶属函数"""

    def __init__(self, sigma1: float, c1: float, sigma2: float, c2: float, name: str = None):
        super().__init__(name)
        self.sigma1 = sigma1  # 第一个高斯的标准差
        self.c1 = c1          # 第一个高斯的中心
        self.sigma2 = sigma2  # 第二个高斯的标准差
        self.c2 = c2          # 第二个高斯的中心
        self.parameters = {'sigma1': sigma1, 'c1': c1, 'sigma2': sigma2, 'c2': c2}

    def compute(self, x):
        x = np.asarray(x)

        # 计算两个高斯函数
        gauss1 = np.exp(-0.5 * ((x - self.c1) / self.sigma1) ** 2)
        gauss2 = np.exp(-0.5 * ((x - self.c2) / self.sigma2) ** 2)

        # 取最大值
        return np.maximum(gauss1, gauss2)

    def set_parameters(self, **kwargs):
        for param in ['sigma1', 'c1', 'sigma2', 'c2']:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]


class GeneralizedBellMF(MembershipFunction):
    """广义贝尔隶属函数"""

    def __init__(self, a: float, b: float, c: float, name: str = None):
        super().__init__(name)
        self.a = a  # 宽度参数
        self.b = b  # 形状参数
        self.c = c  # 中心参数
        self.parameters = {'a': a, 'b': b, 'c': c}

        if a <= 0:
            raise ValueError("Parameter 'a' must be positive")
        if b <= 0:
            raise ValueError("Parameter 'b' must be positive")

    def compute(self, x):
        x = np.asarray(x)

        # 避免除零错误
        with np.errstate(divide='ignore', invalid='ignore'):
            result = 1 / (1 + np.abs((x - self.c) / self.a) ** (2 * self.b))

        # 处理可能的无穷大或NaN值
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

        return result

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            if kwargs['a'] <= 0:
                raise ValueError("Parameter 'a' must be positive")
            self.a = kwargs['a']
            self.parameters['a'] = self.a

        if 'b' in kwargs:
            if kwargs['b'] <= 0:
                raise ValueError("Parameter 'b' must be positive")
            self.b = kwargs['b']
            self.parameters['b'] = self.b

        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class PiMF(MembershipFunction):
    """Pi型隶属函数（S型和Z型的组合）"""

    def __init__(self, a: float, b: float, c: float, d: float, name: str = None):
        super().__init__(name)
        self.a = a  # 左下降开始点
        self.b = b  # 左下降结束点/平台开始点
        self.c = c  # 平台结束点/右下降开始点
        self.d = d  # 右下降结束点
        self.parameters = {'a': a, 'b': b, 'c': c, 'd': d}

        if not (a <= b <= c <= d):
            raise ValueError("Parameters must satisfy: a <= b <= c <= d")

    def compute(self, x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # 左侧S型部分 (a到b)
        if self.a < self.b:
            s_part = SMF(self.a, self.b)
            mask_s = (x >= self.a) & (x <= self.b)
            result[mask_s] = s_part.compute(x[mask_s])

        # 平台部分 (b到c)
        mask_plateau = (x > self.b) & (x <= self.c)
        result[mask_plateau] = 1.0

        # 右侧Z型部分 (c到d)
        if self.c < self.d:
            z_part = ZMF(self.c, self.d)
            mask_z = (x > self.c) & (x <= self.d)
            result[mask_z] = z_part.compute(x[mask_z])

        return result

    def set_parameters(self, **kwargs):
        # 验证参数顺序
        params = self.parameters.copy()
        params.update(kwargs)

        if not (params['a'] <= params['b'] <= params['c'] <= params['d']):
            raise ValueError("Parameters must satisfy: a <= b <= c <= d")

        for param in ['a', 'b', 'c', 'd']:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]
