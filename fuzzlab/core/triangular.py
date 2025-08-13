#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/13 17:16
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab


"""
Fuzzy t-norm Framework Module (FuzzFramework)

This module implements a comprehensive framework for calculating fuzzy t-norms and t-conorms,
supporting various classical fuzzy logic operators with improved type consistency and performance.

Key Features:
----------
1. Supports 12 different types of t-norms and their corresponding t-conorms.
2. Supports operations for q-rung generalized fuzzy numbers (via q-rung isomorphic mapping).
3. Provides generator functions and pseudo-inverse functions for Archimedean t-norms.
4. Automatically verifies mathematical properties of t-norms (axioms, Archimedean property,
   consistency of generators).
5. Visualization capabilities (3D surface plots).
6. De Morgan's Law verification.
7. **NEW**: Unified type handling ensuring consistent float64 output.
8. **NEW**: High-performance reduce operations without numpy ufunc limitations.

Supported t-norm Types:
--------------
- algebraic: Algebraic product t-norm
- lukasiewicz: Łukasiewicz t-norm
- einstein: Einstein t-norm
- hamacher: Hamacher t-norm family
- yager: Yager t-norm family
- schweizer_sklar: Schweizer-Sklar t-norm family
- dombi: Dombi t-norm family
- aczel_alsina: Aczel-Alsina t-norm family
- frank: Frank t-norm family
- minimum: Minimum t-norm (non-Archimedean)
- drastic: Drastic product t-norm (non-Archimedean)
- nilpotent: Nilpotent t-norm (non-Archimedean)

Usage Example:
-------
>>> # Create an algebraic product t-norm instance with q=2
>>> fuzzy_framework = OperationTNorm(norm_type='algebraic', q=2)

>>> # Calculate t-norm and t-conorm
>>> result_t = fuzzy_framework.t_norm(0.6, 0.7)
>>> result_s = fuzzy_framework.t_conorm(0.6, 0.7)

>>> # Verify De Morgan's Laws
>>> demorgan_results = fuzzy_framework.verify_de_morgan_laws()

>>> # Plot 3D surface
>>> fuzzy_framework.plot_t_norm_surface()
"""

import warnings
from typing import Optional, Callable, Union, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from fuzzlab.config import get_config


class TypeNormalizer:
    """统一的类型标准化工具类"""

    @staticmethod
    def to_float(value: Any) -> float:
        """将任意数值类型转换为标准 Python float"""
        if value is None:
            raise TypeError("Cannot convert None to float")
        if isinstance(value, (np.ndarray, np.floating, np.integer)):
            if hasattr(value, 'item'):
                return float(value.item())
            elif np.isscalar(value):
                return float(value)
        return float(value)

    @staticmethod
    def to_int(value: Any) -> int:
        """将任意数值类型转换为标准 Python int"""
        if value is None:
            raise TypeError("Cannot convert None to int")
        if isinstance(value, (np.ndarray, np.floating, np.integer)):
            if hasattr(value, 'item'):
                return int(value.item())
            elif np.isscalar(value):
                return int(value)
        return int(value)

    @staticmethod
    def ensure_array_output(value: Any) -> np.ndarray:
        """确保输出为 float64 numpy 数组"""
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def ensure_scalar_output(value: Any) -> float:
        """确保输出为标准 Python float"""
        if isinstance(value, (np.ndarray, np.floating, np.integer)):
            if np.isscalar(value) or (hasattr(value, 'ndim') and value.ndim == 0):
                return float(value.item()) if hasattr(value, 'item') else float(value)
            else:
                # 对于数组，这里可能需要特殊处理
                raise ValueError(f"Expected scalar output, got array with shape {getattr(value, 'shape', 'unknown')}")
        return float(value)


class BaseNormOperation(ABC):
    """T-范数操作的抽象基类"""

    def __init__(self, q: int = 1, **params):
        self.q = TypeNormalizer.to_int(q)
        self.params = params
        self.is_archimedean = False
        self.is_strict_archimedean = False
        self.supports_q = False

    @abstractmethod
    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T-范数的具体实现"""
        pass

    @abstractmethod
    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T-余范数的具体实现"""
        pass

    def g_func_impl(self, a: np.ndarray) -> Optional[np.ndarray]:
        """生成器函数实现（可选）"""
        return None

    def g_inv_func_impl(self, u: np.ndarray) -> Optional[np.ndarray]:
        """生成器逆函数实现（可选）"""
        return None


class AlgebraicNorm(BaseNormOperation):
    """代数积 T-范数"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.is_archimedean = True
        self.is_strict_archimedean = True
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = a * b"""
        return TypeNormalizer.ensure_array_output(a * b)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = a + b - a * b"""
        return TypeNormalizer.ensure_array_output(a + b - a * b)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = -ln(a)"""
        result = np.where(a > get_config().DEFAULT_EPSILON, -np.log(a), np.inf)
        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = exp(-u)"""
        result = np.where(u < 100, np.exp(-u), 0.0)
        return TypeNormalizer.ensure_array_output(result)


class LukasiewiczNorm(BaseNormOperation):
    """Łukasiewicz T-范数"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.is_archimedean = True
        self.is_strict_archimedean = False
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = max(0, a + b - 1)"""
        return TypeNormalizer.ensure_array_output(np.maximum(0, a + b - 1))

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = min(1, a + b)"""
        return TypeNormalizer.ensure_array_output(np.minimum(1, a + b))

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = 1 - a"""
        return TypeNormalizer.ensure_array_output(1 - a)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = max(0, 1 - u)"""
        return TypeNormalizer.ensure_array_output(np.maximum(0, 1 - u))


class EinsteinNorm(BaseNormOperation):
    """Einstein T-范数"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.is_archimedean = True
        self.is_strict_archimedean = True
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = (a * b) / (1 + (1 - a) * (1 - b))"""
        result = (a * b) / (1 + (1 - a) * (1 - b))
        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = (a + b) / (1 + a * b)"""
        result = (a + b) / (1 + a * b)
        return TypeNormalizer.ensure_array_output(result)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = ln((2-a)/a)"""
        result = np.where(a > get_config().DEFAULT_EPSILON, np.log((2 - a) / a), np.inf)
        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = 2 / (1 + exp(u))"""
        result = np.where(u < 100, 2 / (1 + np.exp(u)), 0.0)
        return TypeNormalizer.ensure_array_output(result)


class HamacherNorm(BaseNormOperation):
    """Hamacher T-范数"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.gamma = params.get('hamacher_param', 1.0)
        if self.gamma <= 0:
            raise ValueError("Hamacher parameter gamma must be positive")
        self.is_archimedean = True
        self.is_strict_archimedean = True
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = (a * b) / (gamma + (1 - gamma) * (a + b - a * b))"""
        result = (a * b) / (self.gamma + (1 - self.gamma) * (a + b - a * b))
        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = (a + b - (2 - gamma) * a * b) / (1 - (1 - gamma) * a * b)"""
        result = (a + b - (2 - self.gamma) * a * b) / (1 - (1 - self.gamma) * a * b)
        return TypeNormalizer.ensure_array_output(result)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = ln((gamma + (1-gamma)*a)/a)"""
        result = np.where(a > get_config().DEFAULT_EPSILON,
                          np.log((self.gamma + (1 - self.gamma) * a) / a), np.inf)
        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = gamma/(exp(u)-1+gamma)"""
        exp_u = np.exp(u)
        result = np.where(exp_u > (1 - self.gamma), self.gamma / (exp_u - (1 - self.gamma)), 0.0)
        return TypeNormalizer.ensure_array_output(result)


class YagerNorm(BaseNormOperation):
    """Yager T-范数族"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.p = params.get('yager_param', 1.0)
        if self.p <= 0:
            raise ValueError("Yager parameter p must be positive")

        self.is_archimedean = True
        self.is_strict_archimedean = (self.p != 1)  # p=1时退化为Łukasiewicz，非严格阿基米德
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = max(0, 1 - ((1-a)^p + (1-b)^p)^(1/p))"""
        term1 = np.power(1 - a, self.p)
        term2 = np.power(1 - b, self.p)
        result = np.maximum(0, 1 - np.power(term1 + term2, 1.0 / self.p))
        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = min(1, (a^p + b^p)^(1/p))"""
        term1 = np.power(a, self.p)
        term2 = np.power(b, self.p)
        result = np.minimum(1, np.power(term1 + term2, 1.0 / self.p))
        return TypeNormalizer.ensure_array_output(result)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = (1-a)^p"""
        result = np.power(1 - a, self.p)
        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = 1 - u^(1/p)"""
        result = 1 - np.power(u, 1.0 / self.p)
        return TypeNormalizer.ensure_array_output(result)


class SchweizerSklarNorm(BaseNormOperation):
    """Schweizer-Sklar T-范数"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.p = params.get('sklar_param', 1.0)
        if self.p == 0:
            raise ValueError("Schweizer-Sklar parameter p cannot be zero")

        self.is_archimedean = True
        self.is_strict_archimedean = True
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Schweizer-Sklar T-范数实现"""
        eps = get_config().DEFAULT_EPSILON

        # 确保输入是数组
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if self.p > 0:
            # T(a,b) = (max(0, a^(-p) + b^(-p) - 1))^(-1/p)
            mask = (a > eps) & (b > eps)
            result = np.zeros_like(a, dtype=np.float64)

            if np.any(mask):
                a_safe = a[mask]
                b_safe = b[mask]

                term1 = np.power(a_safe, -self.p)
                term2 = np.power(b_safe, -self.p)
                inner = np.maximum(0, term1 + term2 - 1)

                # 避免对0取负指数
                result_masked = np.where(inner > eps, np.power(inner, -1.0 / self.p), 0.0)
                result[mask] = result_masked
        else:  # p < 0
            # 当p<0时的处理
            mask = (a < 1.0 - eps) & (b < 1.0 - eps)
            # 确保 result 是一个可写的数组，并用默认值填充
            result = np.minimum(a, b, out=np.empty_like(a, dtype=np.float64))

            if np.any(mask):
                a_safe = a[mask]
                b_safe = b[mask]

                term1 = np.power(a_safe, -self.p)
                term2 = np.power(b_safe, -self.p)
                inner = term1 + term2 - 1

                result_masked = np.where(inner > 0, np.power(inner, -1.0 / self.p), 0.0)
                result[mask] = result_masked

        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Schweizer-Sklar T-余范数实现"""
        eps = get_config().DEFAULT_EPSILON

        # 确保输入是数组
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # 确保 result 是一个可写的数组，并用默认值填充
        result = np.maximum(a, b, out=np.empty_like(a, dtype=np.float64))

        if self.p > 0:
            # S(a,b) = 1 - (max(0, (1-a)^(-p) + (1-b)^(-p) - 1))^(-1/p)
            mask = (1 - a > eps) & (1 - b > eps)

            if np.any(mask):
                a_comp = 1 - a[mask]
                b_comp = 1 - b[mask]

                term1 = np.power(a_comp, -self.p)
                term2 = np.power(b_comp, -self.p)
                inner = np.maximum(0, term1 + term2 - 1)

                result_masked = 1 - np.where(inner > eps, np.power(inner, -1.0 / self.p), 0.0)
                result[mask] = result_masked
        else:  # p < 0
            mask = (a < 1.0 - eps) & (b < 1.0 - eps)

            if np.any(mask):
                a_comp = 1 - a[mask]
                b_comp = 1 - b[mask]

                term1 = np.power(a_comp, -self.p)
                term2 = np.power(b_comp, -self.p)
                inner = term1 + term2 - 1

                result_masked = 1 - np.where(inner > 0, np.power(inner, -1.0 / self.p), 0.0)
                result[mask] = result_masked

        return TypeNormalizer.ensure_array_output(result)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """Schweizer-Sklar 生成器函数"""
        eps = get_config().DEFAULT_EPSILON
        a = np.asarray(a, dtype=np.float64)

        if self.p > 0:
            # g(a) = a^(-p) - 1
            result = np.where(a > eps, np.power(a, -self.p) - 1, np.inf)
        else:  # p < 0
            # g(a) = (1-a)^(-p) - 1
            result = np.where(a < 1.0 - eps, np.power(1 - a, -self.p) - 1, np.inf)

        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """Schweizer-Sklar 生成器逆函数"""
        u = np.asarray(u, dtype=np.float64)

        if self.p > 0:
            # g^(-1)(u) = (u + 1)^(-1/p)
            result = np.where(u > -1, np.power(u + 1, -1.0 / self.p), 0.0)
        else:  # p < 0
            # g^(-1)(u) = 1 - (u + 1)^(-1/p)
            result = np.where(u > -1, 1 - np.power(u + 1, -1.0 / self.p), 0.0)

        return TypeNormalizer.ensure_array_output(result)


class DombiNorm(BaseNormOperation):
    """Dombi T-范数"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.p = params.get('dombi_param', 1.0)
        if self.p <= 0:
            raise ValueError("Dombi parameter p must be positive")

        self.is_archimedean = True
        self.is_strict_archimedean = True
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = 1/(1+(((1-a)/a)^p+((1-b)/b)^p)^(1/p))"""
        eps = get_config().DEFAULT_EPSILON

        # 边界条件处理
        result = np.zeros_like(a, dtype=np.float64)

        # T(a,0)=0, T(0,b)=0
        mask_zero = (a <= eps) | (b <= eps)
        result[mask_zero] = 0.0

        # T(a,1)=a, T(1,b)=b
        mask_a_one = np.abs(a - 1.0) < eps
        mask_b_one = np.abs(b - 1.0) < eps
        result[mask_a_one] = b[mask_a_one]
        result[mask_b_one] = a[mask_b_one]

        # 主要公式应用区域
        calc_mask = (a > eps) & (b > eps) & ~mask_a_one & ~mask_b_one

        if np.any(calc_mask):
            a_calc = a[calc_mask]
            b_calc = b[calc_mask]

            term_a = np.power((1.0 - a_calc) / a_calc, self.p)
            term_b = np.power((1.0 - b_calc) / b_calc, self.p)

            denominator_term = np.power(term_a + term_b, 1.0 / self.p)
            result[calc_mask] = 1.0 / (1.0 + denominator_term)

        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = 1 / (1 + ((a/(1-a))^p + (b/(1-b))^p)^(-1/p))"""
        eps = get_config().DEFAULT_EPSILON

        result = np.zeros_like(a, dtype=np.float64)

        # S(a,0)=a, S(0,b)=b
        mask_a_zero = np.abs(a) < eps
        mask_b_zero = np.abs(b) < eps
        result[mask_a_zero] = b[mask_a_zero]
        result[mask_b_zero] = a[mask_b_zero]

        # S(a,1)=1, S(1,b)=1
        mask_one = (np.abs(a - 1.0) < eps) | (np.abs(b - 1.0) < eps)
        result[mask_one] = 1.0

        # 主要公式
        calc_mask = (a > eps) & (b > eps) & (a < 1.0 - eps) & (b < 1.0 - eps)

        if np.any(calc_mask):
            a_calc = a[calc_mask]
            b_calc = b[calc_mask]

            term_a = np.power(a_calc / (1.0 - a_calc), self.p)
            term_b = np.power(b_calc / (1.0 - b_calc), self.p)

            denominator_term = np.power(term_a + term_b, -1.0 / self.p)
            result[calc_mask] = 1.0 / (1.0 + denominator_term)

        return TypeNormalizer.ensure_array_output(result)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = ((1-a)/a)^p"""
        eps = get_config().DEFAULT_EPSILON

        result = np.where(
            np.abs(a - 1.0) < eps, 0.0,
            np.where(
                np.abs(a - 0.0) < eps, np.inf,
                np.power((1.0 - a) / a, self.p)
            )
        )
        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = 1 / (1 + u^(1/p))"""
        result = np.where(
            np.isinf(u), 0.0,
            np.where(
                np.abs(u - 0.0) < get_config().DEFAULT_EPSILON, 1.0,
                1.0 / (1.0 + np.power(u, 1.0 / self.p))
            )
        )
        return TypeNormalizer.ensure_array_output(result)


class AczelAlsinaNorm(BaseNormOperation):
    """Aczel-Alsina T-范数"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.p = params.get('aa_param', 1.0)
        if self.p <= 0:
            raise ValueError("Aczel-Alsina parameter p must be positive")

        self.is_archimedean = True
        self.is_strict_archimedean = True
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = exp(-(((-ln a)^p + (-ln b)^p)^(1/p)))"""
        eps = get_config().DEFAULT_EPSILON

        mask = (a > eps) & (b > eps)
        result = np.zeros_like(a, dtype=np.float64)

        if np.any(mask):
            a_safe = a[mask]
            b_safe = b[mask]

            term1 = np.power(-np.log(a_safe), self.p)
            term2 = np.power(-np.log(b_safe), self.p)
            inner = np.power(term1 + term2, 1.0 / self.p)

            result[mask] = np.exp(-inner)

        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = 1 - exp(-(((-ln(1-a))^p + (-ln(1-b))^p)^(1/p)))"""
        eps = get_config().DEFAULT_EPSILON

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # 确保 result 是一个可写的数组，并用默认值填充
        result = np.maximum(a, b, out=np.empty_like(a, dtype=np.float64))
        mask = (1 - a > eps) & (1 - b > eps)

        if np.any(mask):
            a_comp = 1 - a[mask]
            b_comp = 1 - b[mask]

            term1 = np.power(-np.log(a_comp), self.p)
            term2 = np.power(-np.log(b_comp), self.p)
            inner = np.power(term1 + term2, 1.0 / self.p)

            result[mask] = 1 - np.exp(-inner)

        return TypeNormalizer.ensure_array_output(result)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = (-ln a)^p"""
        eps = get_config().DEFAULT_EPSILON
        result = np.where(a > eps, np.power(-np.log(a), self.p), np.inf)
        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = exp(-u^(1/p))"""
        result = np.where(u >= 0, np.exp(-np.power(u, 1.0 / self.p)), 1.0)
        return TypeNormalizer.ensure_array_output(result)


class FrankNorm(BaseNormOperation):
    """Frank T-范数"""

    _S_INF_THRESHOLD = 1e5

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.s = params.get('frank_param', np.e)
        if self.s <= 0 or self.s == 1:
            raise ValueError("Frank parameter s must be positive and not equal to 1")

        self.is_archimedean = True
        self.is_strict_archimedean = True
        self.supports_q = True

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = log_s(1 + ((s^a - 1)(s^b - 1))/(s - 1))"""
        # 当 s 足够大时，直接返回 minimum，避免数值不稳定
        if self.s > self._S_INF_THRESHOLD:
            return TypeNormalizer.ensure_array_output(np.minimum(a, b))

        eps = get_config().DEFAULT_EPSILON
        if abs(self.s - 1) < eps:
            return TypeNormalizer.ensure_array_output(np.minimum(a, b))

        with np.errstate(over='ignore'):
            val_a = np.power(self.s, a) - 1
            val_b = np.power(self.s, b) - 1

        denominator = self.s - 1

        # 计算对数的参数，确保大于0
        arg_log = 1 + (val_a * val_b) / denominator
        result = np.where(arg_log <= 0, 0.0, np.log(arg_log) / np.log(self.s))

        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = 1 - log_s(1 + ((s^(1-a) - 1)(s^(1-b) - 1))/(s - 1))"""
        # 当 s 足够大时，直接返回 maximum，避免数值不稳定
        if self.s > self._S_INF_THRESHOLD:
            return TypeNormalizer.ensure_array_output(np.maximum(a, b))

        eps = get_config().DEFAULT_EPSILON
        if abs(self.s - 1) < eps:
            return TypeNormalizer.ensure_array_output(np.maximum(a, b))

        with np.errstate(over='ignore'):
            val_1_a = np.power(self.s, 1 - a) - 1
            val_1_b = np.power(self.s, 1 - b) - 1

        denominator = self.s - 1

        # 计算对数的参数，确保大于0
        arg_log = 1 + (val_1_a * val_1_b) / denominator
        result = np.where(arg_log <= 0, 1.0, 1 - np.log(arg_log) / np.log(self.s))

        return TypeNormalizer.ensure_array_output(result)

    def g_func_impl(self, a: np.ndarray) -> np.ndarray:
        """g(a) = -log((s^a - 1)/(s - 1))"""
        eps = get_config().DEFAULT_EPSILON

        result = np.where(
            a > eps,
            -np.log((np.power(self.s, a) - 1) / (self.s - 1)),
            np.inf
        )
        return TypeNormalizer.ensure_array_output(result)

    def g_inv_func_impl(self, u: np.ndarray) -> np.ndarray:
        """g^(-1)(u) = log_s(1 + (s - 1) exp(-u))"""
        result = np.where(
            u < 100,
            np.log(1 + (self.s - 1) * np.exp(-u)) / np.log(self.s),
            0.0
        )
        return TypeNormalizer.ensure_array_output(result)


class MinimumNorm(BaseNormOperation):
    """最小 T-范数（非阿基米德）"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.is_archimedean = False
        self.is_strict_archimedean = False
        self.supports_q = False

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = min(a,b)"""
        return TypeNormalizer.ensure_array_output(np.minimum(a, b))

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = max(a,b)"""
        return TypeNormalizer.ensure_array_output(np.maximum(a, b))


class DrasticNorm(BaseNormOperation):
    """急剧 T-范数（非阿基米德）"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.is_archimedean = False
        self.is_strict_archimedean = False
        self.supports_q = False

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = a if b=1; b if a=1; 0 otherwise"""
        eps = get_config().DEFAULT_EPSILON
        result = np.where(np.abs(b - 1.0) < eps, a,
                          np.where(np.abs(a - 1.0) < eps, b, 0.0))
        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = a if b=0; b if a=0; 1 otherwise"""
        eps = get_config().DEFAULT_EPSILON
        result = np.where(np.abs(b - 0.0) < eps, a,
                          np.where(np.abs(a - 0.0) < eps, b, 1.0))
        return TypeNormalizer.ensure_array_output(result)


class NilpotentNorm(BaseNormOperation):
    """幂零 T-范数（非阿基米德）"""

    def __init__(self, q: int = 1, **params):
        super().__init__(q, **params)
        self.is_archimedean = False
        self.is_strict_archimedean = False
        self.supports_q = False

    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T(a,b) = min(a,b) if a+b>1; 0 otherwise"""
        result = np.where(a + b > 1, np.minimum(a, b), 0.0)
        return TypeNormalizer.ensure_array_output(result)

    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """S(a,b) = max(a,b) if a+b<1; 1 otherwise"""
        result = np.where(a + b < 1, np.maximum(a, b), 1.0)
        return TypeNormalizer.ensure_array_output(result)


class OperationTNorm:
    """
    重构后的模糊 T-范数操作框架

    主要改进：
    1. 统一的类型处理，确保输出一致性
    2. 模块化的范数实现
    3. 高性能的 reduce 操作
    4. 改进的错误处理和验证
    """

    # 支持的 T-范数类型注册表
    _NORM_REGISTRY = {
        'algebraic': AlgebraicNorm,
        'lukasiewicz': LukasiewiczNorm,
        'einstein': EinsteinNorm,
        'hamacher': HamacherNorm,
        'yager': YagerNorm,
        'schweizer_sklar': SchweizerSklarNorm,
        'dombi': DombiNorm,
        'aczel_alsina': AczelAlsinaNorm,
        'frank': FrankNorm,
        'minimum': MinimumNorm,
        'drastic': DrasticNorm,
        'nilpotent': NilpotentNorm,
    }

    def __init__(self,
                 norm_type: str = None,
                 q: int = 1,
                 **params):
        """
        初始化模糊操作框架

        Args:
            norm_type: T-范数类型
            q: q-阶参数
            **params: 特定范数的参数
        """
        # 类型标准化
        if norm_type is None:
            norm_type = 'algebraic'

        if norm_type not in self._NORM_REGISTRY:
            raise ValueError(f"Unknown t-norm type: {norm_type}. "
                             f"Available types: {', '.join(self._NORM_REGISTRY.keys())}")

        self.norm_type = norm_type
        self.q = TypeNormalizer.to_int(q)
        self.params = params

        if self.q <= 0:
            raise ValueError(f"q must be a positive integer, got q={self.q}")

        # 创建范数操作实例
        norm_class = self._NORM_REGISTRY[norm_type]
        self._norm_op = norm_class(q=self.q, **params)

        # 继承属性
        self.is_archimedean = self._norm_op.is_archimedean
        self.is_strict_archimedean = self._norm_op.is_strict_archimedean
        self.supports_q = self._norm_op.supports_q

        # 初始化函数
        self._initialize_functions()

        # 验证属性
        if get_config().TNORM_VERIFY:
            self._verify_properties()

    def _initialize_functions(self):
        """初始化所有操作函数"""
        # 基础 T-范数和 T-余范数
        if self.supports_q and self.q != 1:
            # q-阶扩展
            self.t_norm = self._create_q_extended_t_norm()
            self.t_conorm = self._create_q_extended_t_conorm()
        else:
            # 标准操作
            self.t_norm = self._create_standard_t_norm()
            self.t_conorm = self._create_standard_t_conorm()

        # 生成器函数
        if self.is_archimedean:
            self.g_func = self._create_generator()
            self.g_inv_func = self._create_generator_inv()
            self.f_func = self._create_dual_generator()
            self.f_inv_func = self._create_dual_generator_inv()
        else:
            self.g_func = None
            self.g_inv_func = None
            self.f_func = None
            self.f_inv_func = None

    def _create_standard_t_norm(self):
        """创建标准 T-范数函数"""

        def t_norm_func(a, b):
            a_arr = np.asarray(a, dtype=np.float64)
            b_arr = np.asarray(b, dtype=np.float64)
            result = self._norm_op.t_norm_impl(a_arr, b_arr)

            # 根据输入类型决定输出格式
            if np.isscalar(a) and np.isscalar(b):
                return TypeNormalizer.ensure_scalar_output(result)
            return result

        return t_norm_func

    def _create_standard_t_conorm(self):
        """创建标准 T-余范数函数"""

        def t_conorm_func(a, b):
            a_arr = np.asarray(a, dtype=np.float64)
            b_arr = np.asarray(b, dtype=np.float64)
            result = self._norm_op.t_conorm_impl(a_arr, b_arr)

            # 根据输入类型决定输出格式
            if np.isscalar(a) and np.isscalar(b):
                return TypeNormalizer.ensure_scalar_output(result)
            return result

        return t_conorm_func

    def _create_q_extended_t_norm(self):
        """创建 q-阶扩展的 T-范数函数"""

        def q_t_norm_func(a, b):
            a_arr = np.asarray(a, dtype=np.float64)
            b_arr = np.asarray(b, dtype=np.float64)

            # q-阶扩展: T_q(a,b) = (T_base(a^q, b^q))^(1/q)
            a_q = np.power(a_arr, self.q)
            b_q = np.power(b_arr, self.q)
            base_result = self._norm_op.t_norm_impl(a_q, b_q)
            result = np.power(base_result, 1.0 / self.q)

            # 根据输入类型决定输出格式
            if np.isscalar(a) and np.isscalar(b):
                return TypeNormalizer.ensure_scalar_output(result)
            return TypeNormalizer.ensure_array_output(result)

        return q_t_norm_func

    def _create_q_extended_t_conorm(self):
        """创建 q-阶扩展的 T-余范数函数"""

        def q_t_conorm_func(a, b):
            a_arr = np.asarray(a, dtype=np.float64)
            b_arr = np.asarray(b, dtype=np.float64)

            # q-阶扩展: S_q(a,b) = (S_base(a^q, b^q))^(1/q)
            a_q = np.power(a_arr, self.q)
            b_q = np.power(b_arr, self.q)
            base_result = self._norm_op.t_conorm_impl(a_q, b_q)
            result = np.power(base_result, 1.0 / self.q)

            # 根据输入类型决定输出格式
            if np.isscalar(a) and np.isscalar(b):
                return TypeNormalizer.ensure_scalar_output(result)
            return TypeNormalizer.ensure_array_output(result)

        return q_t_conorm_func

    def _create_generator(self):
        """创建生成器函数"""
        if not self.is_archimedean or self._norm_op.g_func_impl(np.array([0.5])) is None:
            return None

        def g_func(a):
            a_arr = np.asarray(a, dtype=np.float64)
            if self.supports_q and self.q != 1:
                # q-阶生成器: g_q(a) = g_base(a^q)
                a_q = np.power(a_arr, self.q)
                result = self._norm_op.g_func_impl(a_q)
            else:
                result = self._norm_op.g_func_impl(a_arr)

            if np.isscalar(a):
                return TypeNormalizer.ensure_scalar_output(result)
            return result

        return g_func

    def _create_generator_inv(self):
        """创建生成器逆函数"""
        if not self.is_archimedean or self._norm_op.g_inv_func_impl(np.array([0.5])) is None:
            return None

        def g_inv_func(u):
            u_arr = np.asarray(u, dtype=np.float64)
            base_result = self._norm_op.g_inv_func_impl(u_arr)

            if self.supports_q and self.q != 1:
                # q-阶生成器逆: g_q_inv(u) = (g_base_inv(u))^(1/q)
                result = np.power(base_result, 1.0 / self.q)
            else:
                result = base_result

            if np.isscalar(u):
                return TypeNormalizer.ensure_scalar_output(result)
            return result

        return g_inv_func

    def _create_dual_generator(self):
        """创建对偶生成器函数"""
        if not self.g_func:
            return None

        def f_func(a):
            # f(a) = g((1 - a^q)^(1/q))
            a_arr = np.asarray(a, dtype=np.float64)
            dual_input = np.power(1 - np.power(a_arr, self.q), 1.0 / self.q)
            result = self.g_func(dual_input)

            if np.isscalar(a):
                return TypeNormalizer.ensure_scalar_output(result)
            return result

        return f_func

    def _create_dual_generator_inv(self):
        """创建对偶生成器逆函数"""
        if not self.g_inv_func:
            return None

        def f_inv_func(u):
            # f_inv(u) = (1 - g_inv(u)^q)^(1/q)
            u_arr = np.asarray(u, dtype=np.float64)
            g_inv_result = self.g_inv_func(u_arr)
            result = np.power(1 - np.power(g_inv_result, self.q), 1.0 / self.q)

            if np.isscalar(u):
                return TypeNormalizer.ensure_scalar_output(result)
            return result

        return f_inv_func

    def _pairwise_reduce(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         arr: np.ndarray) -> np.ndarray:
        """
        高效的成对规约操作，使用树形规约减少函数调用次数

        Args:
            func: 二元操作函数
            arr: 输入数组，形状为 (n, ...)，n > 0

        Returns:
            规约后的结果
        """
        data = arr.copy()
        while data.shape[0] > 1:
            n = data.shape[0]
            even = n // 2 * 2
            if even > 0:
                # 向量化批量二元操作
                merged = func(data[0:even:2], data[1:even:2])
                if n % 2 == 1:
                    # 拼接最后一个未配对的元素
                    data = np.concatenate([merged, data[-1:]], axis=0)
                else:
                    data = merged
            else:
                break
        return data[0]

    def _generic_reduce(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                        array: np.ndarray,
                        axis: Optional[Union[int, Tuple[int, ...]]]) -> np.ndarray:
        """
        通用的规约操作，支持多轴规约

        Args:
            func: 二元操作函数
            array: 输入数组
            axis: 规约轴

        Returns:
            规约后的结果
        """
        a = TypeNormalizer.ensure_array_output(array)

        if a.size == 0:
            raise ValueError("Cannot reduce an empty array.")

        if axis is None:
            # 全数组规约
            flat = a.reshape(-1)
            if flat.shape[0] == 1:
                return flat[0]

            # 树形规约
            return self._pairwise_reduce(func, flat.reshape(-1, *flat.shape[1:]))

        # 支持多轴规约
        if isinstance(axis, (tuple, list)):
            # 逐轴规约（从大到小防止轴索引位移）
            result = a
            for ax in sorted(axis, reverse=True):
                result = self._generic_reduce(func, result, ax)
            return result

        # 单轴规约
        ax = int(axis)
        if ax < 0:
            ax = a.ndim + ax

        if ax >= a.ndim or ax < 0:
            raise ValueError(f"axis {axis} is out of bounds for array of dimension {a.ndim}")

        if a.shape[ax] == 0:
            raise ValueError(f"Cannot reduce over axis {ax} with size 0.")

        if a.shape[ax] == 1:
            # 只有一个元素，直接压缩该维度
            return np.squeeze(a, axis=ax)

        # 移动目标轴到前面进行树形规约
        moved = np.moveaxis(a, ax, 0)
        reduced = self._pairwise_reduce(func, moved)

        return TypeNormalizer.ensure_array_output(reduced)

    def t_norm_reduce(self, array: np.ndarray,
                      axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """
        T-范数规约操作

        Args:
            array: 输入数组
            axis: 规约轴，None 表示全数组规约

        Returns:
            规约结果
        """
        if self.t_norm is None:
            raise NotImplementedError(f"T-norm reduction is not supported for {self.norm_type}")

        def _norm_func(x, y):
            # 确保返回 numpy 数组
            result = self.t_norm(x, y)
            return TypeNormalizer.ensure_array_output(result)

        return self._generic_reduce(_norm_func, array, axis)

    def t_conorm_reduce(self, array: np.ndarray,
                        axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """
        T-余范数规约操作

        Args:
            array: 输入数组
            axis: 规约轴，None 表示全数组规约

        Returns:
            规约结果
        """
        if self.t_conorm is None:
            raise NotImplementedError(f"T-conorm reduction is not supported for {self.norm_type}")

        def _conorm_func(x, y):
            # 确保返回 numpy 数组
            result = self.t_conorm(x, y)
            return TypeNormalizer.ensure_array_output(result)

        return self._generic_reduce(_conorm_func, array, axis)

    # ======================= 验证函数 ===========================

    def _verify_properties(self):
        """验证 T-范数的数学性质"""
        try:
            self._verify_t_norm_axioms()
            self._verify_archimedean_property()
            if self.is_archimedean and self.g_func and self.g_inv_func:
                self._verify_generator_properties()
        except Exception as e:
            warnings.warn(f"Property verification failed: {e}", RuntimeWarning)

    def _verify_t_norm_axioms(self):
        """验证 T-范数公理"""
        test_values = [0.2, 0.5, 0.8]
        eps = get_config().DEFAULT_EPSILON

        for a in test_values:
            for b in test_values:
                for c in test_values:
                    # 交换律
                    if abs(self.t_norm(a, b) - self.t_norm(b, a)) >= eps:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}) Commutativity failed: "
                            f"T({a},{b}) ≠ T({b},{a})", UserWarning)

                    # 结合律
                    left = self.t_norm(self.t_norm(a, b), c)
                    right = self.t_norm(a, self.t_norm(b, c))
                    if abs(left - right) >= eps:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}) Associativity failed: "
                            f"T(T({a},{b}),{c}) ≠ T({a},T({b},{c}))", UserWarning)

                    # 单调性
                    if a <= b and self.t_norm(a, c) > self.t_norm(b, c) + eps:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}) Monotonicity failed",
                            UserWarning)

                    # 边界条件
                    if abs(self.t_norm(a, 1.0) - a) >= eps:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}) Boundary condition failed: "
                            f"T({a},1) ≠ {a}", UserWarning)

    def _verify_archimedean_property(self):
        """验证阿基米德性质"""
        if not self.is_archimedean:
            return

        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        eps = get_config().DEFAULT_EPSILON

        for a in test_values:
            t_aa = self.t_norm(a, a)
            if self.is_strict_archimedean:
                if t_aa >= a - eps:
                    warnings.warn(
                        f"({self.norm_type}, q={self.q}) Strict Archimedean property failed: "
                        f"T({a},{a}) = {t_aa:.6f} ≥ {a}", UserWarning)
            else:
                if t_aa > a + eps:
                    warnings.warn(
                        f"({self.norm_type}, q={self.q}) Archimedean property failed: "
                        f"T({a},{a}) = {t_aa:.6f} > {a}", UserWarning)

    def _verify_generator_properties(self):
        """验证生成器性质"""
        if not (self.g_func and self.g_inv_func):
            return

        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        eps = get_config().DEFAULT_EPSILON

        for a in test_values:
            for b in test_values:
                try:
                    g_a = self.g_func(a)
                    g_b = self.g_func(b)

                    if np.isinf(g_a) or np.isinf(g_b):
                        continue

                    via_generator = self.g_inv_func(g_a + g_b)
                    direct = self.t_norm(a, b)

                    if abs(direct - via_generator) >= eps:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}) Generator verification failed: "
                            f"T({a},{b})={direct:.6f} ≠ g^(-1)(g(a)+g(b))={via_generator:.6f}",
                            UserWarning)
                except Exception as e:
                    warnings.warn(
                        f"Error during generator calculation: a={a}, b={b}. Error: {e}",
                        RuntimeWarning)

    def verify_de_morgan_laws(self, a: float = 0.6, b: float = 0.7) -> dict[str, bool]:
        """验证德摩根律"""
        results = {}
        eps = get_config().DEFAULT_EPSILON

        def q_complement(x):
            """q-阶补运算"""
            if not (0 <= x <= 1):
                return x
            return (1 - x ** self.q) ** (1 / self.q)

        # 验证: S(a,b) = N(T(N(a), N(b)))
        s_direct = self.t_conorm(a, b)
        n_a = q_complement(a)
        n_b = q_complement(b)
        s_via_demorgan = q_complement(self.t_norm(n_a, n_b))
        results['de_morgan_1'] = abs(s_direct - s_via_demorgan) < eps

        # 验证: T(a,b) = N(S(N(a), N(b)))
        t_direct = self.t_norm(a, b)
        t_via_demorgan = q_complement(self.t_conorm(n_a, n_b))
        results['de_morgan_2'] = abs(t_direct - t_via_demorgan) < eps

        return results

    # ======================= 信息获取 ============================

    def get_info(self) -> dict:
        """获取当前 T-范数实例的配置信息"""
        return {
            'norm_type': self.norm_type,
            'is_archimedean': self.is_archimedean,
            'is_strict_archimedean': self.is_strict_archimedean,
            'supports_q': self.supports_q,
            'q': self.q,
            'parameters': self.params,
        }

    def plot_t_norm_surface(self, resolution: int = 50):
        """绘制 T-范数和 T-余范数的 3D 表面图"""
        eps = get_config().DEFAULT_EPSILON
        x = np.linspace(eps, 1.0 - eps, resolution)
        y = np.linspace(eps, 1.0 - eps, resolution)
        X, Y = np.meshgrid(x, y)

        Z_t_norm = np.zeros_like(X)
        Z_t_conorm = np.zeros_like(X)

        # 向量化计算
        try:
            Z_t_norm = self.t_norm(X, Y)
            Z_t_conorm = self.t_conorm(X, Y)
        except Exception:
            # 回退到逐点计算
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        Z_t_norm[i, j] = self.t_norm(X[i, j], Y[i, j])
                        Z_t_conorm[i, j] = self.t_conorm(X[i, j], Y[i, j])
                    except Exception:
                        Z_t_norm[i, j] = np.nan
                        Z_t_conorm[i, j] = np.nan

        # 绘图
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z_t_norm, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('a')
        ax1.set_ylabel('b')
        ax1.set_zlabel('T(a,b)')
        ax1.set_title(f'T-Norm: {self.norm_type.title()} (q={self.q})')
        ax1.set_zlim(0, 1)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X, Y, Z_t_conorm, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('a')
        ax2.set_ylabel('b')
        ax2.set_zlabel('S(a,b)')
        ax2.set_title(f'T-Conorm: {self.norm_type.title()} (q={self.q})')
        ax2.set_zlim(0, 1)

        plt.tight_layout()
        plt.show()

    # ======================= 工具方法 ============================

    @staticmethod
    def register_norm(name: str, norm_class: type) -> None:
        """注册新的 T-范数类型"""
        if not issubclass(norm_class, BaseNormOperation):
            raise ValueError("norm_class must be a subclass of BaseNormOperation")
        OperationTNorm._NORM_REGISTRY[name] = norm_class

    @staticmethod
    def list_available_norms() -> list[str]:
        """列出所有可用的 T-范数类型"""
        return list(OperationTNorm._NORM_REGISTRY.keys())

    @classmethod
    def from_generator(cls, g_func: Callable, g_inv_func: Callable,
                       norm_type: str = "custom", q: int = 1, **params):
        """从生成器函数创建 T-范数实例"""
        # 这是一个高级功能，需要更复杂的实现
        # 暂时抛出未实现异常
        raise NotImplementedError("Creating T-norm from custom generators is not yet implemented")


# 为了保持向后兼容性，可以添加一些别名
def create_tnorm(norm_type: str = 'algebraic', q: int = 1, **params) -> OperationTNorm:
    """创建 T-范数实例的便利函数"""
    return OperationTNorm(norm_type=norm_type, q=q, **params)


# 导出主要类和函数
__all__ = [
    'OperationTNorm',
    'BaseNormOperation',
    'TypeNormalizer',
    'create_tnorm',
    'AlgebraicNorm',
    'LukasiewiczNorm',
    'EinsteinNorm',
    'HamacherNorm',
    'YagerNorm',
    'SchweizerSklarNorm',
    'DombiNorm',
    'AczelAlsinaNorm',
    'FrankNorm',
    'MinimumNorm',
    'DrasticNorm',
    'NilpotentNorm'
]
