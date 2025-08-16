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
        if not (a <= b <= c):
            raise ValueError("TriangularMF requires parameters to satisfy a <= b <= c")
        self.a = a
        self.b = b
        self.c = c
        self.parameters = {'a': a, 'b': b, 'c': c}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # 左侧上升段（仅当 b > a）
        if self.b > self.a:
            mask1 = (x >= self.a) & (x < self.b)
            result[mask1] = (x[mask1] - self.a) / (self.b - self.a)

        # 顶点
        result[x == self.b] = 1.0

        # 右侧下降段（仅当 c > b）
        if self.c > self.b:
            mask2 = (x > self.b) & (x <= self.c)
            result[mask2] = (self.c - x[mask2]) / (self.c - self.b)

        # 裁剪到 [0,1]
        return np.clip(result, 0.0, 1.0)

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
        # 重新验证参数顺序
        if not (self.a <= self.b <= self.c):
            raise ValueError("TriangularMF requires parameters to satisfy a <= b <= c")


class TrapezoidalMF(MembershipFunction):
    """梯形隶属函数"""

    def __init__(self, a: float, b: float, c: float, d: float, name: str = None):
        super().__init__(name)
        if not (a <= b <= c <= d):
            raise ValueError("TrapezoidalMF requires parameters to satisfy a <= b <= c <= d")
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.parameters = {'a': a, 'b': b, 'c': c, 'd': d}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # 左侧上升（仅当 b > a）
        if self.b > self.a:
            mask_rise = (x > self.a) & (x < self.b)
            result[mask_rise] = (x[mask_rise] - self.a) / (self.b - self.a)

        # 平台区域 [b, c]
        mask_plateau = (x >= self.b) & (x <= self.c)
        result[mask_plateau] = 1.0

        # 右侧下降（仅当 d > c）
        if self.d > self.c:
            mask_fall = (x > self.c) & (x < self.d)
            result[mask_fall] = (self.d - x[mask_fall]) / (self.d - self.c)

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        for param in ['a', 'b', 'c', 'd']:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]
        # 重新验证参数顺序
        if not (self.a <= self.b <= self.c <= self.d):
            raise ValueError("TrapezoidalMF requires parameters to satisfy a <= b <= c <= d")


class GaussianMF(MembershipFunction):
    """高斯隶属函数"""

    def __init__(self, sigma: float, c: float, name: str = None):
        super().__init__(name)
        if sigma <= 0:
            raise ValueError("GaussianMF parameter 'sigma' must be positive")
        self.sigma = sigma
        self.c = c
        self.parameters = {'sigma': sigma, 'c': c}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.c) / self.sigma) ** 2)

    def set_parameters(self, **kwargs):
        if 'sigma' in kwargs:
            if kwargs['sigma'] <= 0:
                raise ValueError("GaussianMF parameter 'sigma' must be positive")
            self.sigma = kwargs['sigma']
            self.parameters['sigma'] = self.sigma
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class SMF(MembershipFunction):
    """S型隶属函数"""

    def __init__(self, a: float, b: float, name: str = None):
        super().__init__(name)
        if a >= b:
            raise ValueError("SMF requires parameter a < b")
        self.a = a
        self.b = b
        self.parameters = {'a': a, 'b': b}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # x <= a: y = 0
        result[x <= self.a] = 0.0

        # x >= b: y = 1
        result[x >= self.b] = 1.0

        # a < x < b: S型曲线
        mid = (self.a + self.b) / 2
        mask_first = (x > self.a) & (x <= mid)
        mask_second = (x > mid) & (x < self.b)

        # 第一段：2 * ((x - a) / (b - a))^2
        if np.any(mask_first):
            result[mask_first] = 2 * ((x[mask_first] - self.a) / (self.b - self.a)) ** 2

        # 第二段：1 - 2 * ((x - b) / (b - a))^2
        if np.any(mask_second):
            result[mask_second] = 1 - 2 * ((x[mask_second] - self.b) / (self.b - self.a)) ** 2

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if self.a >= self.b:
            raise ValueError("SMF requires parameter a < b")


class ZMF(MembershipFunction):
    """Z型隶属函数"""

    def __init__(self, a: float, b: float, name: str = None):
        super().__init__(name)
        if a >= b:
            raise ValueError("ZMF requires parameter a < b")
        self.a = a
        self.b = b
        self.parameters = {'a': a, 'b': b}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.ones_like(x, dtype=float)

        # x <= a: y = 1
        result[x <= self.a] = 1.0

        # x >= b: y = 0
        result[x >= self.b] = 0.0

        # a < x < b: Z型曲线
        mid = (self.a + self.b) / 2
        mask_first = (x > self.a) & (x <= mid)
        mask_second = (x > mid) & (x < self.b)

        # 第一段：1 - 2 * ((x - a) / (b - a))^2
        if np.any(mask_first):
            result[mask_first] = 1 - 2 * ((x[mask_first] - self.a) / (self.b - self.a)) ** 2

        # 第二段：2 * ((x - b) / (b - a))^2
        if np.any(mask_second):
            result[mask_second] = 2 * ((x[mask_second] - self.b) / (self.b - self.a)) ** 2

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if self.a >= self.b:
            raise ValueError("ZMF requires parameter a < b")


class DoubleGaussianMF(MembershipFunction):
    """双高斯隶属函数"""

    def __init__(self, sigma1: float, c1: float, sigma2: float, c2: float, name: str = None):
        super().__init__(name)
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("DoubleGaussianMF parameters 'sigma1' and 'sigma2' must be positive")
        self.sigma1 = sigma1
        self.c1 = c1
        self.sigma2 = sigma2
        self.c2 = c2
        self.parameters = {'sigma1': sigma1, 'c1': c1, 'sigma2': sigma2, 'c2': c2}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        gauss1 = np.exp(-0.5 * ((x - self.c1) / self.sigma1) ** 2)
        gauss2 = np.exp(-0.5 * ((x - self.c2) / self.sigma2) ** 2)
        return np.maximum(gauss1, gauss2)

    def set_parameters(self, **kwargs):
        for param in ['sigma1', 'c1', 'sigma2', 'c2']:
            if param in kwargs:
                if param in ('sigma1', 'sigma2') and kwargs[param] <= 0:
                    raise ValueError("DoubleGaussianMF parameters 'sigma1' and 'sigma2' must be positive")
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]


class GeneralizedBellMF(MembershipFunction):
    """广义贝尔隶属函数"""

    def __init__(self, a: float, b: float, c: float, name: str = None):
        super().__init__(name)
        if a <= 0:
            raise ValueError("GeneralizedBellMF parameter 'a' must be positive")
        if b <= 0:
            raise ValueError("GeneralizedBellMF parameter 'b' must be positive")
        self.a = a
        self.b = b
        self.c = c
        self.parameters = {'a': a, 'b': b, 'c': c}

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
                raise ValueError("GeneralizedBellMF parameter 'a' must be positive")
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            if kwargs['b'] <= 0:
                raise ValueError("GeneralizedBellMF parameter 'b' must be positive")
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class PiMF(MembershipFunction):
    """Pi型隶属函数（S型和Z型的组合）"""

    def __init__(self, a: float, b: float, c: float, d: float, name: str = None):
        super().__init__(name)
        if not (a <= b <= c <= d):
            raise ValueError("PiMF requires parameters to satisfy a <= b <= c <= d")
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.parameters = {'a': a, 'b': b, 'c': c, 'd': d}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # x <= a 或 x >= d: y = 0
        result[(x <= self.a) | (x >= self.d)] = 0.0

        # b <= x <= c: y = 1 (平台区域)
        result[(x >= self.b) & (x <= self.c)] = 1.0

        # a < x < b: S型上升
        if self.b > self.a:
            mask_rise = (x > self.a) & (x < self.b)
            if np.any(mask_rise):
                # 使用SMF的逻辑
                smf_result = SMF(self.a, self.b).compute(x[mask_rise])
                result[mask_rise] = smf_result

        # c < x < d: Z型下降
        if self.d > self.c:
            mask_fall = (x > self.c) & (x < self.d)
            if np.any(mask_fall):
                # 使用ZMF的逻辑
                zmf_result = ZMF(self.c, self.d).compute(x[mask_fall])
                result[mask_fall] = zmf_result

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        for param in ['a', 'b', 'c', 'd']:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]
        if not (self.a <= self.b <= self.c <= self.d):
            raise ValueError("PiMF requires parameters to satisfy a <= b <= c <= d")
