#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 23:53
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
Fuzzy t-norm Framework Module (FuzzFramework)

This module implements a comprehensive framework for calculating fuzzy t-norms and t-conorms,
supporting various classical fuzzy logic operators.

Key Features:
----------
1. Supports 12 different types of t-norms and their corresponding t-conorms.
2. Supports operations for q-rung generalized fuzzy numbers (via q-rung isomorphic mapping).
3. Provides generator functions and pseudo-inverse functions for Archimedean t-norms.
4. Automatically verifies mathematical properties of t-norms (axioms, Archimedean property,
   consistency of generators).
5. Visualization capabilities (3D surface plots).
6. De Morgan's Law verification.

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
from typing import Optional, Callable, Union

import numpy as np
from matplotlib import pyplot as plt

from axisfuzzy.config import get_config


class OperationTNorm:
    """
    The OperationTNorm class is used to compute and analyze various fuzzy t-norms and t-conorms.
    It supports operations for q-rung generalized fuzzy numbers by extending q-rung
    through generator functions. The class internally defines various common t-norms,
    their generators (if they exist), and related properties and verification methods.

    Attributes:
        t_norm_list (list): A list of supported t-norm types.
        norm_type (str): The currently used t-norm type.
        q (int): The q-rung parameter, used for generalized fuzzy number operations.
        params (dict): A dictionary of parameters specific to certain t-norms.
        is_archimedean (bool): True if the t-norm is Archimedean, False otherwise.
        is_strict_archimedean (bool): True if the t-norm is strictly Archimedean, False otherwise.
        supports_q (bool): True if the t-norm supports q-rung operations, False otherwise.
        g_func (Optional[Callable[[float], float]]): The generator function g(a) for the t-norm,
                                                     potentially q-transformed.
        g_inv_func (Optional[Callable[[float], float]]): The pseudo-inverse function g_inv(u) for the generator,
                                                          potentially q-transformed.
        f_func (Optional[Callable[[float], float]]): The dual generator function f(a) for the t-conorm.
        f_inv_func (Optional[Callable[[float], float]]): The pseudo-inverse function f_inv(u) for the dual generator.
        t_norm (Optional[Callable[[float, float], float]]): The computed t-norm function,
                                                             adapted for the current q value.
        t_conorm (Optional[Callable[[float, float], float]]): The computed t-conorm function,
                                                               adapted for the current q value.

    Core Functionality:
        1.  **Multiple t-norm and t-conorm implementations**: Covers algebraic product, Łukasiewicz,
            Einstein, Hamacher, Yager, Schweizer-Sklar, Dombi, Aczel-Alsina, Frank, Minimum,
            Drastic, and Nilpotent t-norms.
        2.  **q-rung generalized fuzzy number support**: Extends operations to q-rung fuzzy numbers
            via q-rung extension of generator functions. For a base t-norm T_base and t-conorm S_base,
            their q-rung extension is defined as:
            - **Generator q-rung extension**: `g_q(a) = g_base(a^q)`
            - **Generator q-rung pseudo-inverse**: `g_q_inv(u) = (g_base_inv(u))^(1/q)`
            - **Dual generator**: `f(a) = g(1-a^q)^(1/q)`
            - **Dual pseudo-inverse**: `f_inv(u) = (1-g_inv(u)^q)^(1/q)`
        3.  **Generator and Pseudo-inverse Support**: For Archimedean t-norms, provides definitions
            for their generators and pseudo-inverses, and supports their q-rung transformations.
        4.  **Property Verification**: Provides verification for t-norm axioms (commutativity,
            associativity, monotonicity, boundary conditions), Archimedean property, strict
            Archimedean property, and consistency of generator properties.
        5.  **De Morgan's Law Verification**: Verifies if t-norms and t-conorms satisfy De Morgan's
            laws under q-rung isomorphic mapping.
        6.  **Visualization**: Provides functionality to plot 3D surface graphs for t-norms and t-conorms.
        7.  **Utility Methods**: Includes static utility methods for constructing t-norms from generators,
            deriving generators from t-norms (numerical method), and numerically solving for
            generator pseudo-inverses.

    Examples:
        >>> # 1. Initialize an algebraic product t-norm with default q=1
        >>> alg_norm = OperationTNorm(norm_type='algebraic')
        >>> print(f"Algebraic T(0.5, 0.8) = {alg_norm.t_norm(0.5, 0.8):.4f}") # 0.5 * 0.8 = 0.4000
        >>> print(f"Algebraic S(0.5, 0.8) = {alg_norm.t_conorm(0.5, 0.8):.4f}") # 0.5 + 0.8 - 0.5*0.8 = 0.9000

        >>> # 2. Initialize an Łukasiewicz t-norm and use q=2 for q-rung generalization
        >>> luk_norm_q2 = OperationTNorm(norm_type='lukasiewicz', q=2)
        >>> # For Łukasiewicz, T_base(a,b) = max(0, a+b-1)
        >>> # T_q(a,b) = (max(0, a^q + b^q - 1))^(1/q)
        >>> # T_2(0.6, 0.7) = (max(0, 0.6^2 + 0.7^2 - 1))^(1/2) = (max(0, 0.36 + 0.49 - 1))^(1/2) = (max(0, -0.15))^(1/2) = 0
        >>> print(f"Łukasiewicz (q=2) T(0.6, 0.7) = {luk_norm_q2.t_norm(0.6, 0.7):.4f}") # 0.0000
        >>> # S_2(0.6, 0.7) = (min(1, 0.6^2 + 0.7^2))^(1/2) = (min(1, 0.36 + 0.49))^(1/2) = (min(1, 0.85))^(1/2) = sqrt(0.85) approx 0.9220
        >>> print(f"Łukasiewicz (q=2) S(0.6, 0.7) = {luk_norm_q2.t_conorm(0.6, 0.7):.4f}") # 0.9220

        >>> # 3. Initialize a Hamacher t-norm and set parameter gamma
        >>> hamacher_norm = OperationTNorm(norm_type='hamacher', hamacher_param=0.5)
        >>> print(f"Hamacher (gamma=0.5) T(0.3, 0.9) = {hamacher_norm.t_norm(0.3, 0.9):.4f}")

        >>> # 4. Get norm information
        >>> info = alg_norm.get_info()
        >>> for key, value in info.items():
        ...     print(f"  {key}: {value}")

        >>> # 5. Verify De Morgan's Laws
        >>> de_morgan_results = luk_norm_q2.verify_de_morgan_laws(0.6, 0.7)

        >>> # 6. Plot t-norm and t-conorm surface
        >>> # hamacher_norm.plot_t_norm_surface(resolution=30) # Can adjust resolution

    Notes:
        - Parameter `q`: `q` must be a positive integer. When `q=1`, operations
          degenerate to classical fuzzy operations.
        - Specific Norm Parameters: Some t-norms (e.g., Hamacher, Yager, Frank) require
          additional parameters. These parameters are passed via `**params` to the
          constructor. Please refer to the comments of each `_init_xxx` method for
          required parameters and their valid ranges.
            - For example, Hamacher requires `gamma > 0`.
            - Yager requires `p > 0`.
            - Frank requires `s > 0` and `s != 1`.
            - Schweizer-Sklar requires `p != 0`.
            - Dombi and Aczel-Alsina require `p > 0`.
        - Floating Point Precision: Internal calculations use `get_config().DEFAULT_EPSILON`
          (defaulting to 1e-12) for floating-point comparisons to avoid precision issues.
        - Boundary Value Handling: Many t-norm and generator functions may involve `log(0)`
          or division by zero when inputs are close to 0 or 1. The code attempts to handle
          these boundary cases using `get_config().DEFAULT_EPSILON` and conditional checks
          to prevent runtime errors and NaN values.
        - Non-Archimedean Norms: Minimum, Drastic, and Nilpotent are non-Archimedean t-norms.
          They do not have generators and do not support q-rung generalization (`supports_q = False`).
        - Warning Messages: The class's internal verification methods (`_verify_properties`)
          will issue `warnings.UserWarning` or `warnings.RuntimeWarning` if axioms are not
          satisfied or generator properties are inconsistent. These warnings are intended
          to alert the user to potential mathematical inconsistencies or numerical issues
          but do not interrupt program execution.
        - Plotting Performance: The `plot_t_norm_surface` method may require longer
          computation and rendering times at higher `resolution` values.
    """

    # Defines the list of supported t-norm types.
    # This is a class attribute listing all predefined t-norm types that an OperationTNorm instance can be initialized with.
    t_norm_list = [
        'algebraic',
        'lukasiewicz',
        'einstein',
        'hamacher',
        'yager',
        'schweizer_sklar',
        'dombi',
        'aczel_alsina',
        'frank',
        'minimum',
        'drastic',
        'nilpotent'
    ]

    def __init__(self,
                 norm_type: str = None,
                 q: int = 1,  # q-rung parameter, typically a positive integer
                 **params):
        """
        Initializes the fuzzy operation framework.

        Args:
            norm_type (str, optional): The type of t-norm. Defaults to 'algebraic'.
                                       Must be one of the types listed in `OperationTNorm.t_norm_list`.
            q (int, optional): The q-rung pair parameter, used for generalized fuzzy numbers.
                               Defaults to 1. `q` must be a positive integer. When `q=1`,
                               operations degenerate to classical fuzzy operations.
            **params: Additional parameters for specific t-norms.
                      For example:
                      - Hamacher norm: `hamacher_param` (float, must be > 0)
                      - Yager norm: `yager_param` (float, must be > 0)
                      - Schweizer-Sklar norm: `sklar_param` (float, must be != 0)
                      - Dombi norm: `dombi_param` (float, must be > 0)
                      - Aczel-Alsina norm: `aa_param` (float, must be > 0)
                      - Frank norm: `frank_param` (float, must be > 0 and != 1)

        Raises:
            ValueError: If `norm_type` is unknown or `q` does not meet the requirements.
        """

        # If norm_type is not specified, default to 'algebraic'
        if norm_type is None:
            norm_type = 'algebraic'

        # Check if norm_type is in the supported list
        if norm_type not in self.t_norm_list:
            raise ValueError(f"Unknown t-norm type: {norm_type}. Available types: "
                             f"{', '.join(self.t_norm_list)}")

        # Check if q is a positive integer
        if not isinstance(q, int) or q <= 0:
            raise ValueError(f"q must be a positive integer, "
                             f"but received q={q} (type: {type(q)}).")
        # Store instance attributes
        self.norm_type = norm_type  # Current t-norm type
        self.q = q  # q-rung parameter
        self.params: dict = params  # Stores additional parameters for specific norms

        # Raw generator and pseudo-inverse functions for q=1 (used for generator property verification
        # and initial q=1 level operations, deriving t-norm and t-conorm).
        # These are internal base generator functions set by _init_xxx methods.
        self._base_g_func_raw: Optional[Callable[[float], float]] = None
        self._base_g_inv_func_raw: Optional[Callable[[float], float]] = None

        # Final generator and pseudo-inverse functions (potentially q-transformed, used for generator property verification).
        # These are q-transformed generator functions used for internal verification.
        self.g_func: Optional[Callable[[float], float]] = None
        self.g_inv_func: Optional[Callable[[float], float]] = None

        # Dual generator and its pseudo-inverse, used for verifying t-conorm generator properties.
        self.f_func: Optional[Callable[[float], float]] = None
        self.f_inv_func: Optional[Callable[[float], float]] = None

        # Final t-norm and t-conorm calculation functions.
        # These are the main interfaces called externally, directly adapting to the q value.
        # In fact, t-norms and t-conorms are derived from generators and dual generators and their pseudo-inverses.
        self.t_norm: Optional[Callable[[float, float], float]] = None
        self.t_conorm: Optional[Callable[[float, float], float]] = None

        # Norm properties: whether Archimedean, strictly Archimedean, and whether q-rung generalization is supported.
        # These properties are set in _init_xxx methods for internal logic and information queries.
        self.is_archimedean: bool = False
        self.is_strict_archimedean: bool = False
        self.supports_q: bool = False

        # Initialize all operations and properties.
        # This is the core of the constructor, which sets all functions and properties based on norm_type.
        self._initialize_operation()

        # Verify norm properties.
        # After initialization, immediately verify the mathematical properties of the selected t-norm
        # to ensure it meets expectations.
        if get_config().TNORM_VERIFY:
            self._verify_properties()

    def _initialize_operation(self):
        """
        Initializes all base t-norm, t-conorm, generator functions, and their properties,
        and applies q-rung transformation to the generator functions based on the `q` value.

        Process Overview:

        1.  **Initialize Base Operations (q=1)**: Calls the corresponding `_init_xxx` method
            based on `self.norm_type`.
            - Each `_init_xxx` method sets `_base_t_norm_raw`, `_base_t_conorm_raw`,
              `_base_g_func_raw`, `_base_g_inv_func_raw`, as well as `is_archimedean`,
              `is_strict_archimedean`, `supports_q`, and other attributes.
        2.  **Apply q-value transformation to q-rung generators**: Calls `_q_transformation`.
            - This transforms `_base_g_func_raw` and `_base_g_inv_func_raw` into the
              final `self.g_func` and `self.g_inv_func` based on `self.q` and `self.supports_q`.
        3.  **Apply q-rung transformation to dual generators and their pseudo-inverses**: Calls `_init_dual_generators`.
            - This generates `self.f_func` and `self.f_inv_func` based on `self.q` and `self.supports_q`.
        4.  **Perform q-rung isomorphic mapping for t-norms and t-conorms**: Calls `_q_transformation_for_t_norm`.
            - This applies q-rung isomorphic mapping to the t-norm and t-conorm operators based on the `q` value.
        5.  **Transform t-norms directly based on generators and their pseudo-inverses,
            and t-conorms based on dual generators and pseudo-inverses**: Calls `_t_norm_transformation`.
            - This transforms `self.g_func` and `self.g_inv_func` into `self.t_norm`
              based on `self.q` and `self.supports_q`.
            - Simultaneously, it transforms `self.f_func` and `self.f_inv_func` into `self.t_conorm`
              based on `self.q` and `self.supports_q`.
        """

        # 1. Initialize raw functions and properties for base operations (q=1).
        # Calls the corresponding initialization method based on norm_type.
        if self.norm_type == "algebraic":
            self._init_algebraic()
        elif self.norm_type == "lukasiewicz":
            self._init_lukasiewicz()
        elif self.norm_type == "einstein":
            self._init_einstein()
        elif self.norm_type == "hamacher":
            self._init_hamacher()
        elif self.norm_type == "yager":
            self._init_yager()
        elif self.norm_type == "schweizer_sklar":
            self._init_schweizer_sklar()
        elif self.norm_type == "dombi":
            self._init_dombi()
        elif self.norm_type == "aczel_alsina":
            self._init_aczel_alsina()
        elif self.norm_type == "frank":
            self._init_frank()
        elif self.norm_type == "minimum":
            self._init_minimum()
        elif self.norm_type == "drastic":
            self._init_drastic()
        elif self.norm_type == "nilpotent":
            self._init_nilpotent()

        # 2. Apply q-value transformation to q-rung generators.
        # This means that through q-rung transformation, _base_g_func_raw and _base_g_inv_func_raw
        # are transformed into the final g_func and g_inv_func, which are the q-transformed generator functions.
        self._q_transformation()

        # 3. Generate dual generators and their pseudo-inverses.
        self._init_dual_generators()

        # 4. Perform q-rung isomorphic mapping for t-norms and t-conorms.
        self._q_transformation_for_t_norm()

        # 5. Transform t-norms directly based on generators and their pseudo-inverses,
        # and t-conorms based on dual generators and pseudo-inverses.
        self._t_norm_transformation()

    def _q_transformation(self):
        """
        Transforms the generator and pseudo-inverse functions based on the `q` value.
        Before this, it checks if some t-norms and t-conorms do not have generators.

        These transformed generators (`self.g_func`, `self.g_inv_func`)

        Mathematical Expressions (q-rung Generator Generalization):
            - **Generator**: `g_q(a) = g_base(a^q)`
            - **Pseudo-inverse**: `g_q_inv(u) = (g_base_inv(u))^(1/q)`

        Logic:
            - If `self.supports_q` is True and `self.q` is not equal to 1, and the base
              generator is defined, then the q-rung transformation described above is applied.
            - Otherwise, the original base generator is used directly.
            - If the base generator is undefined (e.g., for non-Archimedean norms),
              `self.g_func` and `self.g_inv_func` remain None.
        """
        if self.supports_q and self.q != 1:
            # Check if base generator exists, as non-Archimedean norms do not have generators.
            if self._base_g_func_raw is None or self._base_g_inv_func_raw is None:
                warnings.warn(f"The t-norm {self.norm_type} supports q-tung transformation, "
                              f"but its base generator or pseudo-inverse is empty. "
                              f"Skipping generator transformation.",
                              RuntimeWarning)
                self.g_func = None
                self.g_inv_func = None
                self.f_func = None
                self.f_inv_func = None
                return

            # q-rung generator: g_q(a) = g_base(a^q)
            # For a=0, a^q is still 0, and g_base(0) is usually infinity.
            # q-rung pseudo-inverse: g_q_inv(u) = (g_base_inv(u))^(1/q)
            # For u=inf, g_base_inv(inf) is usually 0.
            self.g_func = lambda a: self._base_g_func_raw(a ** self.q)
            self.g_inv_func = lambda u: (self._base_g_inv_func_raw(u)) ** (1 / self.q)
        else:
            # If q=1 or q-rung is not supported, use the original base generator directly.
            self.g_func = self._base_g_func_raw
            self.g_inv_func = self._base_g_inv_func_raw

    def _init_dual_generators(self):
        """
        Initializes the dual generator and its pseudo-inverse.
        The dual generator is derived from the generator and its pseudo-inverse.
        Together with the generator and its pseudo-inverse, they form the t-norm and t-conorm.

        Mathematical Expressions:
            - **Dual Generator**: `f(a) = g(1-a^q)^(1/q)`
            - **Dual Pseudo-inverse**: `f_inv(u) = (1 - g_inv(u)^q)^(1/q)`

        Logic:
            - The dual generator is initialized only if the current t-norm is Archimedean
              and its generator (`self.g_func`) and pseudo-inverse (`self.g_inv_func`) are defined.
            - Otherwise, the dual generator remains None.
        """
        if self.is_archimedean and self.g_func is not None and self.g_inv_func is not None:
            # Dual generator f(a) = g((1 - a^q)^(1/q))
            # Ensure 1-a is within the valid domain [0,1], otherwise the result is infinity.
            # Dual pseudo-inverse f_inv(u) = (1 - g_inv(u)^q)^(1/q)
            self.f_func = lambda a: self.g_func((1 - a ** self.q) ** (1 / self.q))
            self.f_inv_func = lambda u: (1 - self.g_inv_func(u) ** self.q) ** (1 / self.q)
        else:
            self.f_func = None
            self.f_inv_func = None

    def _q_transformation_for_t_norm(self):
        """
        Applies q-rung isomorphic mapping to the t-norm and t-conorm operators based on the `q` value.
        These transformed t-norms and t-conorms are primarily used to verify if the t-norms and
        t-conorms obtained after generator transformation conform to their mathematical definitions.

        Mathematical Expressions (q-rung Generalization):
            - **t-norm**: `T_q(a,b) = (T_base(a^q, b^q))^(1/q)`
            - **t-conorm**: `S_q(a,b) = (S_base(a^q, b^q))^(1/q)`
        Logic:
            - If `self.supports_q` is True and `self.q` is not equal to 1, then the
              q-rung isomorphic mapping described above is applied.
            - Otherwise (i.e., `q=1` or the norm does not support q-rung generalization),
              the original base operation function is used directly.
            - The t-norm and t-conorm are assigned to `self._check_t_norm` and `self._check_t_conorm`.
        """
        if self.supports_q and self.q != 1:
            # Apply q-rung isomorphic mapping: definition of t-norm T_q(a,b) and t-conorm S_q(a,b)
            self._check_t_norm = lambda a, b: (self._base_t_norm_raw(a ** self.q, b ** self.q)) ** (1 / self.q)
            self._check_t_conorm = lambda a, b: (self._base_t_conorm_raw(a ** self.q, b ** self.q)) ** (1 / self.q)
        else:
            # If q=1 or q-rung is not supported, use the original base operations directly.
            self._check_t_norm = self._base_t_norm_raw
            self._check_t_conorm = self._base_t_conorm_raw

    def _t_norm_transformation(self):
        """
        Directly transforms the t-norm based on its generator and pseudo-inverse,
        and the t-conorm based on its dual generator and pseudo-inverse.
        This method verifies if the transformed t-norm and t-conorm conform to their
        mathematical definitions.

        Logic:
            - Checks if the current t-norm is Archimedean and if its generator and
              pseudo-inverse are defined.
            - Checks if the dual generator and pseudo-inverse exist.
            - Checks if the transformed t-norm and t-conorm conform to their mathematical definitions.

        Note:
            - This method is only used to verify if the transformed t-norm and t-conorm
              conform to their mathematical definitions.
            - In practical applications, this method is usually not called directly;
              instead, the already transformed t-norm and t-conorm are used.
        """
        if not self.is_archimedean:
            self.t_norm = self._base_t_norm_raw
            self.t_conorm = self._base_t_conorm_raw
        else:
            if self.g_func is None or self.g_inv_func is None:
                warnings.warn(f"The t-norm {self.norm_type} supports t-norm transformation, "
                              f"but its generator or pseudo-inverse is empty. "
                              f"t-norm and t-conorm are not transformed and as "
                              f"initial mathematical expression calculation form.",
                              RuntimeWarning)
                self.t_norm = self._base_t_norm_raw
                self.t_conorm = self._base_t_conorm_raw
            else:
                # Calculate q-transformed t-norm and t-conorm using generator, pseudo-inverse,
                # dual generator, and dual pseudo-inverse.
                self.t_norm = lambda a, b: self.g_inv_func(self.g_func(a) + self.g_func(b))
                self.t_conorm = lambda a, b: self.f_inv_func(self.f_func(a) + self.f_func(b))

        # Verify if the transformed t-norm and t-conorm match the initial t-norm and t-conorm.
        if self.is_archimedean:
            test_data = (0.6, 0.4)
            check = self._check_t_norm(*test_data)
            t_norm = self.t_norm(*test_data)

            if abs(check - t_norm) > get_config().DEFAULT_EPSILON:
                warnings.warn(f"Test failed, t-norm {self.norm_type} has a large deviation "
                              f"({abs(check - t_norm)})"
                              f"in the q-rung operation values obtained through the generator and "
                              f"its pseudo-inverse transformation({t_norm}) compared to the q-rung isomorphic "
                              f"mapping values of the t-norm and t-conorm({check}).",
                              RuntimeWarning)

    def _pairwise_reduce(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         arr: np.ndarray) -> np.ndarray:
        """
        高效(避免 Python 逐元素)的成对规约：
        不断两两合并(树形规约)，降低调用次数。
        arr: 形状 (n, ...)，n > 0
        """
        data = arr
        while data.shape[0] > 1:
            n = data.shape[0]
            even = n // 2 * 2
            if even > 0:
                merged = func(data[0:even:2], data[1:even:2])  # 向量化批量二元
                if n % 2 == 1:
                    # 拼回最后一个未配对
                    data = np.concatenate([merged, data[-1:]], axis=0)
                else:
                    data = merged
            else:
                # n=1 直接退出
                break
        return data[0]

    def _generic_reduce(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                        array: np.ndarray,
                        axis: Optional[Union[int, tuple]]):
        a = np.asarray(array, dtype=np.float64)
        if a.size == 0:
            raise ValueError("Cannot reduce an empty array.")
        if axis is None:
            flat = a.reshape(-1)
            res = flat[0]
            for i in range(1, flat.shape[0]):
                res = func(res, flat[i])
            return np.asarray(res, dtype=np.float64)
        # 支持多轴
        if isinstance(axis, tuple):
            # 逐轴规约（从大到小防止轴索引位移）
            for ax in sorted(axis, reverse=True):
                a = self._generic_reduce(func, a, ax)
            return a
        # 单轴
        ax = int(axis)
        if a.shape[ax] == 0:
            raise ValueError(f"Cannot reduce over axis {ax} with size 0.")
        # 移动到前面做树规约
        moved = np.moveaxis(a, ax, 0)  # (n, ...)
        res = self._pairwise_reduce(func, moved)
        return np.asarray(res, dtype=np.float64)

    def t_norm_reduce(self, array: np.ndarray, axis: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """
        自定义规约实现，替代 frompyfunc.reduce，避免 not reorderable 错误。
        """
        if self.t_norm is None:
            raise NotImplementedError(f"T-norm reduction is not supported for {self.norm_type}")
        # self.t_norm 已支持 ndarray 逐元素广播
        def _f(x, y):
            return self.t_norm(x, y)
        return self._generic_reduce(_f, array, axis)

    def t_conorm_reduce(self, array: np.ndarray, axis: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """
        自定义规约实现，替代 frompyfunc.reduce。
        """
        if self.t_conorm is None:
            raise NotImplementedError(f"T-conorm reduction is not supported for {self.norm_type}")
        def _f(x, y):
            return self.t_conorm(x, y)
        return self._generic_reduce(_f, array, axis)


    # ======================= Initialize Base Operations (q=1) ====================
    # Each _init_xxx method is responsible for defining the following for that norm type at q=1:
    # - _base_g_func_raw: The raw generator function g(a).
    # - _base_g_inv_func_raw: The raw generator pseudo-inverse function g_inv(u).
    # - is_archimedean: Whether it is an Archimedean t-norm.
    # - is_strict_archimedean: Whether it is a strictly Archimedean t-norm.
    # - supports_q: Whether it supports q-rung isomorphic mapping generalization.

    def _init_algebraic(self):
        """
        Initializes the Algebraic Product t-norm and its dual t-conorm.
        This is a strictly Archimedean t-norm.
        """
        self._base_t_norm_raw = lambda a, b: a * b
        """Mathematical expression: T(a,b) = a * b"""

        self._base_t_conorm_raw = lambda a, b: a + b - a * b
        """Mathematical expression: S(a,b) = a + b - ab"""

        # TODO: RuntimeWarning: divide by zero encountered in log self._base_g_func_raw = lambda a: np.where(a > get_config().DEFAULT_EPSILON, -np.log(a), np.inf)
        self._base_g_func_raw = lambda a: np.where(a > get_config().DEFAULT_EPSILON, -np.log(a), np.inf)
        """Mathematical expression: g(a) = -ln(a)
        The generator g(a) approaches infinity as a approaches 0, and approaches 0 as a approaches 1.
        Here, get_config().DEFAULT_EPSILON is used to handle the log(0) case, avoiding runtime errors.
        """

        self._base_g_inv_func_raw = lambda u: np.where(u < 100, np.exp(-u), 0.0)
        """Mathematical expression: g^(-1)(u) = exp(-u)
        The pseudo-inverse g_inv(u) approaches 0 as u approaches infinity, and approaches 1 as u approaches 0.
        Here, an upper limit of 100 is set to prevent exp(-u) from becoming too small, leading to floating-point underflow, and directly returns 0.0.
        """

        self.is_archimedean = True
        """Is an Archimedean t-norm: T(x,x) < x for all x ∈ (0,1)."""

        self.is_strict_archimedean = True
        """Is a strictly Archimedean t-norm: In addition to Archimedean property, T(x,y) = 0 if and only if x=0 or y=0."""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization: This norm can be generalized through q-rung transformation."""

    def _init_lukasiewicz(self):
        """
        Initializes the Łukasiewicz t-norm and its dual t-conorm.
        This is an Archimedean t-norm, but not strictly Archimedean.
        """
        self._base_t_norm_raw = lambda a, b: np.maximum(0, a + b - 1)
        """Mathematical expression: T(a,b) = max(0, a + b - 1)"""

        self._base_t_conorm_raw = lambda a, b: np.minimum(1, a + b)
        """Mathematical expression: S(a,b) = min(1, a + b)"""

        self._base_g_func_raw = lambda a: 1 - a
        """Mathematical expression: g(a) = 1 - a"""

        self._base_g_inv_func_raw = lambda u: np.maximum(0, 1 - u)
        """Mathematical expression: g^(-1)(u) = max(0, 1 - u)"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = False
        """Is not a strictly Archimedean t-norm: For example, T(0.5, 0.5) = 0, but 0.5 != 0."""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_einstein(self):
        """
        Initializes the Einstein t-norm and its dual t-conorm.
        This is a strictly Archimedean t-norm.
        """
        self._base_t_norm_raw = lambda a, b: (a * b) / (1 + (1 - a) * (1 - b))
        """Mathematical expression: T(a,b) = (a * b) / (1 + (1-a)*(1-b))"""

        self._base_t_conorm_raw = lambda a, b: (a + b) / (1 + a * b)
        """Mathematical expression: S(a,b) = (a + b)/(1 + a * b)"""

        self._base_g_func_raw = lambda a: np.where(a > get_config().DEFAULT_EPSILON, np.log((2 - a) / a), np.inf)
        """Mathematical expression: g(a) = ln((2-a)/a)"""

        self._base_g_inv_func_raw = lambda u: np.where(u < 100, 2 / (1 + np.exp(u)), 0.0)
        """Mathematical expression: g^(-1)(u) = 2 / (1 + exp(u))"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = True
        """Is a strictly Archimedean t-norm"""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_hamacher(self):
        """
        Initializes the Hamacher t-norm and its dual t-conorm.
        This is a strictly Archimedean t-norm, containing a parameter `gamma`.
        """
        if 'hamacher_param' not in self.params:
            self.params['hamacher_param'] = 1.0
        gamma = self.params.get('hamacher_param')  # Get parameter gamma
        if gamma <= 0:
            raise ValueError("Hamacher parameter gamma must be greater than 0")

        self._base_t_norm_raw = lambda a, b: (a * b) / (gamma + (1 - gamma) * (a + b - a * b))
        """Mathematical expression: T(a,b) = (a * b) / (gamma + (1-gamma)*(a+b-a*b))"""

        self._base_t_conorm_raw = lambda a, b: (a + b - (2 - gamma) * a * b) / (1 - (1 - gamma) * a * b)
        """Mathematical expression: S(a,b) = (a+b-(2-gamma)*ab)/(1-(1-gamma)*ab)"""

        self._base_g_func_raw = lambda a: np.where(a > get_config().DEFAULT_EPSILON,
                                                   np.log((gamma + (1 - gamma) * a) / a), np.inf)
        """Mathematical expression: g(a) = ln((gamma + (1-gamma)*a)/a)"""

        self._base_g_inv_func_raw = lambda u: np.where(np.exp(u) > (1 - gamma), gamma / (np.exp(u) - (1 - gamma)), 0.0)
        """Mathematical expression: g^(-1)(u) = gamma/(exp(u)-1+gamma)"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = True
        """Is a strictly Archimedean t-norm"""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_yager(self):
        """
        Initializes the Yager t-norm family and its dual t-conorm family.
        This is an Archimedean t-norm family, containing a parameter `p`.
        When p=1, the Yager t-norm degenerates to the Łukasiewicz t-norm.
        """
        if 'yager_param' not in self.params:
            self.params['yager_param'] = 1.0
        p = self.params.get('yager_param')  # Get parameter p, defaults to 1.0
        if p <= 0:
            raise ValueError("Yager parameter p must be greater than 0")

        self._base_t_norm_raw = lambda a, b: np.maximum(0, 1 - ((1 - a) ** p + (1 - b) ** p) ** (1 / p))
        """Mathematical expression: T(a,b) = 1 - min(1, ((1-a)^p + (1-b)^p)^{1/p})"""

        self._base_t_conorm_raw = lambda a, b: np.minimum(1, (a ** p + b ** p) ** (1 / p))
        """Mathematical expression: S(a,b) = min(1, (a^p + b^p)^{1/p})"""

        self._base_g_func_raw = lambda a: (1 - a) ** p
        """Mathematical expression: g(a) = (1 - a)^p"""

        # self._base_g_inv_func_raw = lambda u: 1 - min(1, u ** (1 / p))
        self._base_g_inv_func_raw = lambda u: 1 - (u ** (1 / p))
        """Mathematical expression: g^(-1)(u) = 1 - min(1, u^{1/p})"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = (p == 1)
        """Whether it is a strictly Archimedean t-norm; it is strict when p=1.
        When p=1, the Yager t-norm degenerates to the Łukasiewicz t-norm,
        and Łukasiewicz is not strictly Archimedean.
        The comment here is incorrect: it should be that as p approaches infinity,
        the Yager t-norm approaches the Minimum t-norm.
        When p=1, T(a,b) = max(0, a+b-1) and S(a,b) = min(1, a+b), which is exactly the Łukasiewicz norm.
        The Łukasiewicz norm is not strictly Archimedean.
        The Yager t-norm family is generally strictly Archimedean, unless p=1 or p approaches infinity.
        For p=1, T(a,a) = max(0, 2a-1). If a=0.5, T(0.5,0.5)=0, but a!=0, so it is not strictly Archimedean.
        """

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_schweizer_sklar(self):
        """
        Initializes the Schweizer-Sklar t-norm and its dual t-conorm.
        This is a strictly Archimedean t-norm, containing a parameter `p`.
        As p approaches 0, it degenerates to the algebraic product.
        As p approaches infinity, it degenerates to Minimum.
        As p approaches negative infinity, it degenerates to Drastic Product.
        """
        if 'sklar_param' not in self.params:
            self.params['sklar_param'] = 1.0
        p = self.params.get('sklar_param')  # Get parameter p, defaults to 1.0
        if p == 0:
            raise ValueError("Schweizer-Sklar parameter p cannot be 0")

        if p > 0:
            self._base_t_norm_raw = lambda a, b: np.where(
                (a > get_config().DEFAULT_EPSILON) & (b > get_config().DEFAULT_EPSILON),
                (np.maximum(0, a ** (-p) + b ** (-p) - 1)) ** (-1 / p), 0.0)
            """Mathematical expression: T(a,b) = (max(0, a^{-p} + b^{-p} - 1))^{-1/p}
            Handles cases where a or b are close to 0, avoiding division by 0 or negative exponent issues.
            """

            self._base_t_conorm_raw = lambda a, b: np.where(
                ((1 - a) > get_config().DEFAULT_EPSILON) & ((1 - b) > get_config().DEFAULT_EPSILON),
                1 - (np.maximum(0, (1 - a) ** (-p) + (1 - b) ** (-p) - 1)) ** (-1 / p), np.maximum(a, b))
            """Mathematical expression: S(a,b) = 1 - (max(0, (1-a)^{-p} + (1-b)^{-p} - 1))^{-1/p}
            Handles cases where 1-a or 1-b are close to 0.
            """

            self._base_g_func_raw = lambda a: np.where(a > get_config().DEFAULT_EPSILON, a ** (-p) - 1, np.inf)
            """Mathematical expression: g(a) = a^{-p} - 1"""

            self._base_g_inv_func_raw = lambda u: np.where(u > -1, (u + 1) ** (-1 / p), 0.0)
            """Mathematical expression: g^(-1)(u) = (u + 1)^{-1/p}"""

        else:  # p < 0
            # When p < 0, the formula form is slightly different to ensure correct function behavior.
            self._base_t_norm_raw = lambda a, b: np.where(
                (a < 1.0 - get_config().DEFAULT_EPSILON) & (b < 1.0 - get_config().DEFAULT_EPSILON),
                (a ** (-p) + b ** (-p) - 1) ** (-1 / p), np.minimum(a, b))
            """Mathematical expression: T(a,b) = (a^{-p} + b^{-p} - 1)^{-1/p}
            Handles cases where a or b are close to 1.
            """

            self._base_t_conorm_raw = lambda a, b: np.where(
                ((1 - a) < 1.0 - get_config().DEFAULT_EPSILON) & ((1 - b) < 1.0 - get_config().DEFAULT_EPSILON),
                1 - ((1 - a) ** (-p) + (1 - b) ** (-p) - 1) ** (-1 / p), np.maximum(a, b))
            """Mathematical expression: S(a,b) = 1 - ((1-a)^{-p} + (1-b)^{-p} - 1)^{-1/p}
            Handles cases where 1-a or 1-b are close to 1.
            """

            self._base_g_func_raw = lambda a: np.where(a < 1.0 - get_config().DEFAULT_EPSILON, (1 - a) ** (-p) - 1,
                                                       np.inf)
            """Mathematical expression: g(a) = (1 - a)^{-p} - 1"""

            self._base_g_inv_func_raw = lambda u: np.where(u > -1, 1 - (u + 1) ** (-1 / p), 0.0)
            """Mathematical expression: g^(-1)(u) = 1 - (u + 1)^{-1/p}"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = True
        """Is a strictly Archimedean t-norm"""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_dombi(self):
        """
        Initializes the Dombi t-norm and its dual t-conorm.
        This is a strictly Archimedean t-norm, containing a parameter `p`.
        """
        if 'dombi_param' not in self.params:
            self.params['dombi_param'] = 1.0
        p = self.params.get('dombi_param')  # Get parameter p, defaults to 1.0
        if p <= 0:
            raise ValueError("Dombi parameter p must be greater than 0")

        # Dombi t-norm original formula
        def dombi_tnorm(a, b):
            # Boundary condition handling: T(a,0)=0, T(0,b)=0
            res = np.zeros_like(a, dtype=float)

            # Condition for main formula
            mask = (a > get_config().DEFAULT_EPSILON) & (b > get_config().DEFAULT_EPSILON)

            # Handle T(a,1)=a, T(1,b)=b
            mask_a1 = np.abs(a - 1.0) < get_config().DEFAULT_EPSILON
            mask_b1 = np.abs(b - 1.0) < get_config().DEFAULT_EPSILON

            res[mask_a1] = b[mask_a1]
            res[mask_b1] = a[mask_b1]

            # Apply main formula where applicable
            calc_mask = mask & ~mask_a1 & ~mask_b1

            a_calc, b_calc = a[calc_mask], b[calc_mask]

            term_a = np.power((1.0 - a_calc) / a_calc, p)
            term_b = np.power((1.0 - b_calc) / b_calc, p)

            denominator_term = np.power(term_a + term_b, 1 / p)
            res[calc_mask] = 1 / (1 + denominator_term)

            return res

        # Dombi t-conorm original formula
        def dombi_tconorm(a, b):
            res = np.zeros_like(a, dtype=float)

            # S(a,0)=a, S(0,b)=b
            mask_a0 = np.abs(a) < get_config().DEFAULT_EPSILON
            mask_b0 = np.abs(b) < get_config().DEFAULT_EPSILON
            res[mask_a0] = b[mask_a0]
            res[mask_b0] = a[mask_b0]

            # S(a,1)=1, S(1,b)=1
            mask_a1 = np.abs(a - 1.0) < get_config().DEFAULT_EPSILON
            mask_b1 = np.abs(b - 1.0) < get_config().DEFAULT_EPSILON
            res[mask_a1] = 1.0
            res[mask_b1] = 1.0

            # Main formula
            calc_mask = ~mask_a0 & ~mask_b0 & ~mask_a1 & ~mask_b1
            a_calc, b_calc = a[calc_mask], b[calc_mask]

            term_a = np.power(a_calc / (1.0 - a_calc), p)
            term_b = np.power(b_calc / (1.0 - b_calc), p)

            denominator_term = np.power(term_a + term_b, -1 / p)
            res[calc_mask] = 1 / (1 + denominator_term)

            return res

        # Corrected definitions for generator and pseudo-inverse to conform to mathematical definitions at boundaries.
        # Generator g(a) = ((1-a)/a)^p
        def dombi_g_func(a):
            return np.where(
                np.abs(a - 1.0) < get_config().DEFAULT_EPSILON, 0.0,
                np.where(
                    np.abs(a - 0.0) < get_config().DEFAULT_EPSILON, np.inf,
                    np.power((1.0 - a) / a, p)
                )
            )

        # Pseudo-inverse g_inv(u) = 1 / (1 + u^(1/p))
        def dombi_g_inv_func(u):
            # When u approaches infinity, u^(1/p) approaches infinity, 1 + u^(1/p) approaches infinity, and the result approaches 0.
            return np.where(
                np.isinf(u), 0.0,
                np.where(
                    np.abs(u - 0.0) < get_config().DEFAULT_EPSILON, 1.0,
                    1.0 / (1.0 + np.power(u, 1.0 / p))
                )
            )

        self._base_t_norm_raw = dombi_tnorm
        """Mathematical expression: T(a,b) = 1/(1+(((1-a)/a)^p+((1-b)/b)^p)^{1/p})"""

        self._base_t_conorm_raw = dombi_tconorm
        """Mathematical expression: S(a,b) = 1 / (1 + ((a/(1-a))^p + (b/(1/(1-b)))^p)^{-1/p})"""

        self._base_g_func_raw = dombi_g_func
        """Mathematical expression: g(a) = ((1 - a)/a)^p"""

        self._base_g_inv_func_raw = dombi_g_inv_func
        """Mathematical expression: g^(-1)(u) = 1 / (1 + u^{1/p})"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = True
        """Is a strictly Archimedean t-norm"""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_aczel_alsina(self):
        """
        Initializes the Aczel-Alsina t-norm and its dual t-conorm.
        This is a strictly Archimedean t-norm, containing a parameter `p`.
        As p approaches 0, it degenerates to Minimum.
        As p approaches 1, it degenerates to the algebraic product.
        As p approaches infinity, it degenerates to Drastic Product.
        """
        if 'aa_param' not in self.params:
            self.params['aa_param'] = 1.0
        p = self.params.get('aa_param')  # Get parameter p, defaults to 1.0
        if p <= 0:
            raise ValueError("Aczel-Alsina parameter p must be greater than 0")

        self._base_t_norm_raw = lambda a, b: np.where((a > get_config().DEFAULT_EPSILON) & (b > get_config().DEFAULT_EPSILON),
                                                      np.exp(-(((-np.log(a)) ** p + (-np.log(b)) ** p) ** (1 / p))), 0.0)
        """Mathematical expression: T(a,b) = exp(-(((-ln a)^p + (-ln b)^p)^{1/p}))
        Handles cases where a or b are close to 0, avoiding log(0) or negative power issues.
        """

        self._base_t_conorm_raw = lambda a, b: np.where(((1 - a) > get_config().DEFAULT_EPSILON) & ((1 - b) > get_config().DEFAULT_EPSILON),
                                                        1 - np.exp(-(((-np.log(1 - a)) ** p + (-np.log(1 - b)) ** p) ** (1 / p))), np.maximum(a, b))
        """Mathematical expression: S(a,b) = 1 - exp(-(((-ln(1-a))^p + (-ln(1-b))^p)^{1/p}))
        Handles cases where 1-a or 1-b are close to 0.
        """

        self._base_g_func_raw = lambda a: np.where(a > get_config().DEFAULT_EPSILON, (-np.log(a)) ** p, np.inf)
        """Mathematical expression: g(a) = (-ln a)^p"""

        self._base_g_inv_func_raw = lambda u: np.where(u >= 0, np.exp(-(u ** (1 / p))), 1.0)
        """Mathematical expression: g^(-1)(u) = exp(-u^{1/p})"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = True
        """Is a strictly Archimedean t-norm"""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_frank(self):
        """
        Initializes the Frank t-norm and its dual t-conorm.
        This is a strictly Archimedean t-norm family, containing a parameter `s`.
        As s approaches 0, it degenerates to Drastic Product.
        As s approaches 1, it degenerates to Minimum.
        As s approaches infinity, it degenerates to Product.
        """
        if 'frank_param' not in self.params:
            self.params['frank_param'] = np.e
        s = self.params.get('frank_param')  # Get parameter s, defaults to natural logarithm base e
        if s <= 0 or s == 1:
            raise ValueError("Frank parameter s must be greater than 0 and not equal to 1")

        def frank_tnorm(a, b):
            if s == np.inf:  # When s approaches infinity, Frank product degenerates to Minimum product.
                return np.minimum(a, b)
            # Avoid log(0) or division by 0.
            if abs(s - 1) < get_config().DEFAULT_EPSILON:
                return np.minimum(a, b)  # Degenerates to Minimum when s=1.
            val_a = s ** a - 1
            val_b = s ** b - 1
            denominator = s - 1

            # Calculate the argument for log, ensuring it's greater than 0.
            arg_log = 1 + (val_a * val_b) / denominator
            return np.where(arg_log <= 0, 0.0, np.log(arg_log) / np.log(s))

        def frank_tconorm(a, b):
            if s == np.inf:  # When s approaches infinity, Frank conorm degenerates to Maximum conorm.
                return np.maximum(a, b)
            if abs(s - 1) < get_config().DEFAULT_EPSILON:
                return np.maximum(a, b)  # Degenerates to Maximum when s=1.
            val_1_a = s ** (1 - a) - 1
            val_1_b = s ** (1 - b) - 1
            denominator = s - 1

            # Calculate the argument for log, ensuring it's greater than 0.
            arg_log = 1 + (val_1_a * val_1_b) / denominator
            return np.where(arg_log <= 0, 1.0, 1 - np.log(arg_log) / np.log(s))

        self._base_t_norm_raw = frank_tnorm
        """Mathematical expression: T(a,b) = log_s(1 + ((s^a - 1)(s^b - 1))/(s - 1))"""

        self._base_t_conorm_raw = frank_tconorm
        """Mathematical expression: S(a,b) = 1 - log_s(1 + ((s^{1-a} - 1)(s^{1-b} - 1))/(s - 1))"""

        self._base_g_func_raw = lambda a: np.where(a > get_config().DEFAULT_EPSILON, -np.log((s ** a - 1) / (s - 1)), np.inf)
        """Mathematical expression: g(a) = -log_s((s^a - 1)/(s - 1))"""

        self._base_g_inv_func_raw = lambda u: np.where(u < 100, np.log(1 + (s - 1) * np.exp(-u)) / np.log(s), 0.0)
        """Mathematical expression: g^(-1)(u) = log_s(1 + (s - 1) exp(-u))"""

        self.is_archimedean = True
        """Is an Archimedean t-norm"""

        self.is_strict_archimedean = True
        """Is a strictly Archimedean t-norm"""

        self.supports_q = True
        """Supports q-rung isomorphic mapping generalization"""

    def _init_minimum(self):
        """
        Initializes the Minimum t-norm and its dual Maximum t-conorm.
        This is a non-Archimedean t-norm, and also the strongest t-norm.
        """
        self._base_t_norm_raw = lambda a, b: np.minimum(a, b)
        """Mathematical expression: T(a,b) = min(a,b)"""

        self._base_t_conorm_raw = lambda a, b: np.maximum(a, b)
        """Mathematical expression: S(a,b) = max(a,b)"""

        self._base_g_func_raw = None  # Non-Archimedean t-norms do not have generators.
        self._base_g_inv_func_raw = None

        self.is_archimedean = False
        """Is a non-Archimedean t-norm: T(x,x) = x, which does not satisfy T(x,x) < x."""

        self.is_strict_archimedean = False
        """Is a non-strictly Archimedean t-norm"""

        self.supports_q = False
        """Does not support q-rung isomorphic mapping generalization: Non-Archimedean t-norms typically do not support q-rung generalization via generators."""

    def _init_nilpotent(self):
        """
        Initializes the Nilpotent t-norm and its dual t-conorm.
        This is a non-Archimedean t-norm.
        """

        def nilpotent_tnorm(a, b):
            return np.where(a + b > 1, np.minimum(a, b), 0.0)

        def nilpotent_tconorm(a, b):
            return np.where(a + b < 1, np.maximum(a, b), 1.0)

        self._base_t_norm_raw = nilpotent_tnorm
        """Mathematical expression: T(a,b) = min(a,b) if a+b>1; 0 otherwise"""

        self._base_t_conorm_raw = nilpotent_tconorm
        """Mathematical expression: S(a,b) = max(a,b) if a+b<1; 1 otherwise"""

        self._base_g_func_raw = None
        self._base_g_inv_func_raw = None

        self.is_archimedean = False
        """Is a non-Archimedean t-norm"""

        self.is_strict_archimedean = False
        """Is a non-strictly Archimedean t-norm"""

        self.supports_q = False
        """Does not support q-rung isomorphic mapping generalization"""

    def _init_drastic(self):
        """
        Initializes the Drastic Product t-norm and its dual Drastic Sum t-conorm.
        This is a non-Archimedean t-norm, and also the weakest t-norm.
        """

        def drastic_tnorm(a, b):
            return np.where(np.abs(b - 1.0) < get_config().DEFAULT_EPSILON, a,
                            np.where(np.abs(a - 1.0) < get_config().DEFAULT_EPSILON, b, 0.0))

        def drastic_tconorm(a, b):
            return np.where(np.abs(b - 0.0) < get_config().DEFAULT_EPSILON, a,
                            np.where(np.abs(a - 0.0) < get_config().DEFAULT_EPSILON, b, 1.0))

        self._base_t_norm_raw = drastic_tnorm
        """Mathematical expression: T(a,b) = a if b=1; b if a=1; 0 otherwise"""

        self._base_t_conorm_raw = drastic_tconorm
        """Mathematical expression: S(a,b) = a if b=0; b if a=0; 1 otherwise"""

        self._base_g_func_raw = None
        self._base_g_inv_func_raw = None

        self.is_archimedean = False
        """Is a non-Archimedean t-norm"""

        self.is_strict_archimedean = False
        """Is a non-strictly Archimedean t-norm"""

        self.supports_q = False
        """Does not support q-rung isomorphic mapping generalization"""

    # ======================= Verification Functions ===========================

    def _verify_properties(self):
        """
        Verifies the mathematical properties of the current t-norm instance,
        including t-norm axioms, Archimedean property, and consistency between
        the generator and the t-norm.
        Verification results are output as warnings and do not interrupt program execution.

        Verification Contents:
        1.  **t-norm Axioms**: Calls `_verify_t_norm_axioms` to verify commutativity,
            associativity, monotonicity, and boundary conditions.
        2.  **Archimedean Property**: Calls `_verify_archimedean_property` to verify
            whether it satisfies the Archimedean or strictly Archimedean property.
        3.  **Generator Properties**: For Archimedean t-norms where generators are defined,
            calls `_verify_generator_properties` to verify if `T(a,b) = g_inv(g(a) + g(b))` holds.
        """
        # Verify t-norm axioms.
        self._verify_t_norm_axioms()

        # Verify Archimedean property.
        self._verify_archimedean_property()

        # Verify generator properties (only for Archimedean norms where generators are defined).
        if self.is_archimedean and self.g_func is not None and self.g_inv_func is not None:
            self._verify_generator_properties()

    def _verify_t_norm_axioms(self):
        """
        Verifies t-norm axioms: commutativity, associativity, monotonicity, and boundary conditions.
        Uses a set of test values for verification and performs floating-point comparisons
        with a tolerance based on `get_config().DEFAULT_EPSILON`.
        If any axiom is not satisfied, a `UserWarning` is issued.
        """
        test_values = [0.2, 0.5, 0.8]  # Fuzzy values used for testing

        for a in test_values:
            for b in test_values:
                for c in test_values:
                    # 1. Commutativity: T(a,b) = T(b,a)
                    # Check if abs(T(a,b) - T(b,a)) is greater than the tolerance.
                    if abs(self.t_norm(a, b) - self.t_norm(b, a)) >= get_config().DEFAULT_EPSILON:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}).Commutativity failed: T({a},{b}) ≠ T({b},{a}) "
                            f"(T({a},{b})={self.t_norm(a, b):.6f}, T({b},{a})={self.t_norm(b, a):.6f}).",
                            UserWarning
                        )

                    # 2. Associativity: T(T(a,b),c) = T(a,T(b,c))
                    left_assoc = self.t_norm(self.t_norm(a, b), c)
                    right_assoc = self.t_norm(a, self.t_norm(b, c))
                    # Check if abs(left_assoc - right_assoc) is greater than the tolerance.
                    if abs(left_assoc - right_assoc) >= get_config().DEFAULT_EPSILON:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}).Associativity failed: T(T({a},{b}),{c}) ≠ T({a},T({b},{c})) "
                            f"(left={left_assoc:.6f}, right={right_assoc:.6f}).",
                            UserWarning
                        )

                    # 3. Monotonicity: If a <= b, then T(a,c) <= T(b,c)
                    if a <= b:
                        # Check if T(a,c) is strictly greater than T(b,c) + _epsilon.
                        if self.t_norm(a, c) > self.t_norm(b, c) + get_config().DEFAULT_EPSILON:
                            warnings.warn(
                                f"({self.norm_type}, q={self.q}).Monotonicity failed: a≤b but T(a,c)>T(b,c) "
                                f"(T({a},{c})={self.t_norm(a, c):.6f}, T({b},{c})={self.t_norm(b, c):.6f}).",
                                UserWarning
                            )

                    # 4. Boundary Condition: T(a,1) = a
                    # Check if abs(T(a,1) - a) is greater than the tolerance.
                    if abs(self.t_norm(a, 1.0) - a) >= get_config().DEFAULT_EPSILON:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}).Boundary condition failed: T({a},1) ≠ {a} "
                            f"(T({a},1)={self.t_norm(a, 1.0):.6f}).",
                            UserWarning
                        )

    def _verify_archimedean_property(self):
        """
        Verifies the Archimedean Property and Strict Archimedean Property of the t-norm.

        Definitions:
        -   **Archimedean Property**: For all `a ∈ (0,1)`, there exists `n` such that
            `T^n(a) = T(a, ..., a)` (n times) `= 0`.
            Equivalently, for all `a ∈ (0,1)`, `T(a,a) < a`.
        -   **Strict Archimedean Property**: In addition to the Archimedean property,
            `T(x,y) = 0` if and only if `x=0` or `y=0`.
            Equivalently, for all `a ∈ (0,1)`, `T(a,a) < a` and `T(x,y) > 0` for all `x,y ∈ (0,1]`.

        Logic:
        -   Verification is performed only for norms marked as Archimedean (`self.is_archimedean` is True).
        -   For strictly Archimedean norms, it checks `T(a,a) < a`.
        -   For non-strictly Archimedean but Archimedean norms, it checks `T(a,a) <= a`.
        """
        if not self.is_archimedean:
            return  # Only verify for norms marked as Archimedean.

        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for a in test_values:
            t_aa = self.t_norm(a, a)
            if self.is_strict_archimedean:
                # Strict Archimedean property requires T(a,a) < a.
                # Check if t_aa is greater than or equal to a - _epsilon.
                if t_aa >= a - get_config().DEFAULT_EPSILON:  # Use _epsilon for floating-point comparison.
                    warnings.warn(
                        f"({self.norm_type}, q={self.q}).Strict Archimedean property failed: T({a},{a}) = {t_aa:.6f} ≥ {a}.",
                        UserWarning
                    )
            else:
                # Archimedean property requires T(a,a) <= a.
                # Check if t_aa is strictly greater than a + _epsilon.
                if t_aa > a + get_config().DEFAULT_EPSILON:
                    warnings.warn(
                        f"({self.norm_type}, q={self.q}).Archimedean property failed: T({a},{a}) = {t_aa:.6f} > {a}.",
                        UserWarning
                    )

    def _verify_generator_properties(self):
        """
        Verifies generator properties: `T(a,b) = g_inv(g(a) + g(b))`.
        This verification compares the t-norm calculated using the q-transformed
        generator (`self.g_func`, `self.g_inv_func`) with the t-norm calculated
        directly using the q-transformed isomorphic mapping (`self.t_norm`).

        Logic:
        -   If the generator or pseudo-inverse is undefined, skips verification and issues a warning.
        -   Uses a set of test values `a` and `b`.
        -   For each pair `(a,b)`, calculates `g(a)` and `g(b)`.
        -   Calculates the t-norm result via the generator formula `g_inv(g(a) + g(b))` (`via_generator`).
        -   Calculates the t-norm result directly via `self.t_norm(a,b)` (`direct`).
        -   Compares `via_generator` and `direct`. If the difference exceeds
            `get_config().DEFAULT_EPSILON`, a warning is issued.
        -   Includes error handling to prevent numerical issues during generator calculation.
        """
        if self.g_func is None or self.g_inv_func is None:
            warnings.warn(
                f"({self.norm_type}, q={self.q}).Generator or pseudo-inverse is undefined, skipping generator property verification.",
                RuntimeWarning)
            return False

        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for a in test_values:
            for b in test_values:
                try:
                    # Use the current q-transformed g_func and g_inv_func.
                    g_a = self.g_func(a)
                    g_b = self.g_func(b)

                    # Ensure g_a and g_b are finite values to avoid NaN or errors from adding infinities.
                    if np.isinf(g_a) or np.isinf(g_b):
                        # If generator values are infinity, it usually means the input is at a boundary (e.g., a=0 or b=0).
                        # In such cases, the result of T(a,b) is typically 0.
                        # Direct comparison might be inaccurate, so skip this test pair because
                        # the generator formula's behavior at boundaries can be uncertain.
                        continue

                    # Attempt to calculate t-norm via generator.
                    via_generator = self.g_inv_func(g_a + g_b)

                    # Calculate directly using the current q-transformed isomorphic mapped t-norm.
                    direct = self.t_norm(a, b)

                    # Compare the two calculation results, allowing for _epsilon tolerance.
                    if abs(direct - via_generator) >= get_config().DEFAULT_EPSILON:
                        warnings.warn(
                            f"({self.norm_type}, q={self.q}).Generator verification failed: T({a},{b})={direct:.6f} ≠ g^(-1)(g(a)+g(b))={via_generator:.6f}. "
                            f"g(a)={g_a:.6f}, g(b)={g_b:.6f}.",
                            UserWarning
                        )
                        # return False # Do not interrupt, continue checking other values.
                except Exception as e:
                    warnings.warn(
                        f"({self.norm_type}, q={self.q}).Error during generator calculation: a={a}, b={b}. Error message: {e}.",
                        RuntimeWarning
                    )
                    continue  # Skip current iteration to avoid subsequent comparison failures due to calculation errors.
        return True

    def verify_de_morgan_laws(self, a: float = 0.6, b: float = 0.7) -> dict[str, bool]:
        """
        Verifies De Morgan's Laws under q-rung isomorphic mapping.
        For q-rung t-norms and t-conorms defined through isomorphic mapping,
        De Morgan's Laws typically hold.

        De Morgan's Law Forms (Generalized for q-rung fuzzy numbers):
        1.  `S(a,b) = N(T(N(a), N(b)))`
        2.  `T(a,b) = N(S(N(a), N(b)))`
        where `N(x)` is the q-rung complement `(1 - x^q)^(1/q)`.

        Args:
            a (float): The first fuzzy value, defaults to 0.6.
            b (float): The second fuzzy value, defaults to 0.7.

        Returns:
            dict[str, bool]: A dictionary containing two boolean values, indicating
                             whether De Morgan's Laws hold.
                             `'de_morgan_1'`: `S(a,b) == N(T(N(a), N(b)))`
                             `'de_morgan_2'`: `T(a,b) == N(S(N(a), N(b)))`
        """
        # results = {}
        #
        # # Verify: S(a,b) = 1 - T(1-a, 1-b)
        # s_direct = self.t_conorm(a, b)
        # s_via_demorgan = 1 - self.t_norm(1 - a, 1 - b)
        # # Use _epsilon for floating-point comparison
        # results['de_morgan_1'] = abs(s_direct - s_via_demorgan) < get_config().DEFAULT_EPSILON
        #
        # # Verify: T(a,b) = 1 - S(1-a, 1-b)
        # t_direct = self.t_norm(a, b)
        # t_via_demorgan = 1 - self.t_conorm(1 - a, 1 - b)
        # # Use _epsilon for floating-point comparison
        # results['de_morgan_2'] = abs(t_direct - t_via_demorgan) < get_config().DEFAULT_EPSILON
        #
        # return results

        results = {}

        # Define q-rung complement operation
        def q_rung_complement(x):
            if not (0 <= x <= 1):
                return x  # Or raise an error
            return (1 - x ** self.q) ** (1 / self.q)

        # Verify: S(a,b) = N(T(N(a), N(b)))
        s_direct = self.t_conorm(a, b)
        n_a = q_rung_complement(a)
        n_b = q_rung_complement(b)
        s_via_demorgan = q_rung_complement(self.t_norm(n_a, n_b))
        results['de_morgan_1'] = abs(s_direct - s_via_demorgan) < get_config().DEFAULT_EPSILON

        # Verify: T(a,b) = N(S(N(a), N(b)))
        t_direct = self.t_norm(a, b)
        t_via_demorgan = q_rung_complement(self.t_conorm(n_a, n_b))
        results['de_morgan_2'] = abs(t_direct - t_via_demorgan) < get_config().DEFAULT_EPSILON

        return results

    # ======================= Information Retrieval ============================

    def get_info(self) -> dict:
        """
        Retrieves configuration information and properties of the current t-norm instance.

        Returns:
            dict: A dictionary containing information such as norm type, q value,
                  parameters, Archimedean property, etc.
        """
        info = {
            'norm_type': self.norm_type,  # Name of the t-norm type
            'is_archimedean': self.is_archimedean,  # Whether it is Archimedean
            'is_strict_archimedean': self.is_strict_archimedean,  # Whether it is strictly Archimedean
            'supports_q': self.supports_q,  # Whether it supports q-rung generalization
            'q_value': self.q,  # Current q value
            'parameter': self.params,  # Additional parameters for specific norms
        }
        return info

    def plot_t_norm_surface(self, resolution: int = 50):
        """
        Plots the 3D surface of the current t-norm and t-conorm.
        The plot displays the surfaces of t-norm T(a,b) and t-conorm S(a,b)
        on the [0,1]x[0,1] plane.

        Args:
            resolution (int): The resolution of the plotting grid, defaults to 50.
                              Higher resolution results in a smoother surface but
                              longer computation time.
        """
        # Generate ranges for a and b, avoiding boundary values that might cause
        # calculation issues (e.g., log(0) or division by 0).
        x = np.linspace(get_config().DEFAULT_EPSILON, 1.0 - get_config().DEFAULT_EPSILON, resolution)
        y = np.linspace(get_config().DEFAULT_EPSILON, 1.0 - get_config().DEFAULT_EPSILON, resolution)
        X, Y = np.meshgrid(x, y)  # Create meshgrid points.

        # Initialize Z-coordinate matrices to store t-norm and t-conorm results.
        Z_t_norm = np.zeros_like(X, dtype=float)
        Z_t_conorm = np.zeros_like(X, dtype=float)

        # Iterate through meshgrid points and calculate t-norm and t-conorm values for each (a,b) pair.
        for i in range(resolution):
            for j in range(resolution):
                try:
                    Z_t_norm[i, j] = self.t_norm(X[i, j], Y[i, j])
                    Z_t_conorm[i, j] = self.t_conorm(X[i, j], Y[i, j])
                except Exception as e:
                    # Catch potential numerical errors during plotting (e.g., some norms
                    # might cause overflow or NaN for certain parameters).
                    # Set the value of erroneous points to NaN, so these points are skipped
                    # during plotting and do not interrupt the graph.
                    Z_t_norm[i, j] = np.nan
                    Z_t_conorm[i, j] = np.nan
                    # Can choose to issue a warning here, but frequent warnings might affect user experience, so commented out.
                    # warnings.warn(f"Plotting calculation error: ({X[i,j]}, {Y[i,j]}) -> {e}", RuntimeWarning)

        # Create Matplotlib figure and subplots.
        fig = plt.figure(figsize=(14, 6))  # Adjust figure size to display two 3D plots side-by-side.

        # First subplot: t-norm surface plot.
        ax1 = fig.add_subplot(121, projection='3d')  # First subplot in a 1x2 grid, 3D projection.
        ax1.plot_surface(X, Y, Z_t_norm, cmap='viridis', alpha=0.8)  # Plot surface.
        ax1.set_xlabel('a')
        ax1.set_ylabel('b')
        ax1.set_zlabel('T(a,b)')
        ax1.set_title(f'T-Norm: {self.norm_type.title()} (q={self.q})')  # Set title.
        ax1.set_zlim(0, 1)  # Limit Z-axis range to [0,1], as fuzzy values are typically within this range.

        # Second subplot: t-conorm surface plot.
        ax2 = fig.add_subplot(122, projection='3d')  # Second subplot in a 1x2 grid, 3D projection.
        ax2.plot_surface(X, Y, Z_t_conorm, cmap='plasma', alpha=0.8)  # Plot surface.
        ax2.set_xlabel('a')
        ax2.set_ylabel('b')
        ax2.set_zlabel('S(a,b)')
        ax2.set_title(f'T-Conorm: {self.norm_type.title()} (q={self.q})')  # Set title.
        ax2.set_zlim(0, 1)  # Limit Z-axis range to [0,1].

        plt.tight_layout()  # Automatically adjust subplot parameters for a tight layout.
        plt.show()  # Display the figure.

    # ======================= Utility Methods ============================
    # These methods provide general tools for converting between t-norms and generators;
    # they are not directly involved in the core initialization process of OperationTNorm.
    # They are defined as static methods and can be called directly via the class name
    # (e.g., OperationTNorm.generator_to_t_norm(...)).

    @staticmethod
    def t_norm_to_generator(t_norm_func: Callable[[float, float], float],
                            epsilon: float = 1e-6) -> tuple[Callable[[float], float], Callable[[float], float]]:
        """
        Attempts to derive the additive generator g(a) and its pseudo-inverse g_inv(u)
        from a t-norm function (Archimedean t-norm).
        This method uses a numerical approach and may not be accurate for all t-norms
        or across all input ranges.
        **Note**: This is a highly simplified example implementation. For complex t-norms,
        accurate derivation of generators usually requires more advanced numerical
        integration techniques or symbolic derivation.

        This only provides a conceptual framework; actual applications may require
        customization for specific norms.

        Args:
            t_norm_func (Callable): The t-norm function T(a,b).
            epsilon (float): Floating-point comparison precision.

        Returns:
            tuple[Callable, Callable]: A tuple containing (generator function g(a), pseudo-inverse function g_inv(u)).
        """

        # The generator g(x) of an Archimedean t-norm satisfies g(x) = C * integral(1/phi(t) dt)
        # And T(x,y) = g_inv(g(x) + g(y))
        # For strictly Archimedean t-norms, g(x) satisfies g(x) = -ln(x) + ... or g(x) = (1-x)/x + ...
        # Here, a simplified numerical method is used, based on g(x) + g(y) = g(T(x,y)).
        # Set g(1) = 0 (for continuous Archimedean t-norms).

        def g(a: float) -> float:
            """
            Simplified generator function g(a).
            For the Product t-norm (T(a,b)=ab), its generator is g(a) = -ln(a).
            This form is used directly as an example here because it is a common generator.
            However, for other t-norms, the generator form will be different and require
            more complex derivation.
            """
            if a <= epsilon:
                return np.inf  # g(0) is usually infinity.
            if a >= 1.0 - epsilon:
                return 0.0  # g(1) is usually 0.

            # For strictly Archimedean t-norms T(x,y) = g_inv(g(x) + g(y))
            # And g(1)=0, T(x,1)=x
            # Then g(T(x,1)) = g(x) + g(1) => g(x) = g(x) + 0
            # Consider g(x) = integral_x^1 (1/phi(t)) dt
            # Simplified to g(x) = C * (1/x - 1) for some C, or -ln(x) for Product.
            # The numerical derivation here may not be general and needs to be adjusted for specific norms.
            # This is a simplified example; actual applications require more complex numerical integration or specific formulas.
            try:
                # Attempt to use the generator form of the algebraic product as an inspiration.
                # For T(a,b) = ab, g(a) = -ln(a).
                # g(T(a,b)) = -ln(ab) = -ln(a) - ln(b) = g(a) + g(b).
                # This is a simplified attempt and does not guarantee accuracy for all norms.
                # A better approach would be numerical integration or looking up known formulas.
                return -np.log(a)  # Example only, may not apply to all t-norms.
            except Exception:
                return np.inf  # Return infinity on error.

        # Numerical solution for the pseudo-inverse function g_inv(u).
        # Calls the static method generator_to_generator_inv to derive its pseudo-inverse from the g function.
        g_inv = OperationTNorm.generator_to_generator_inv(g, epsilon=epsilon)

        return g, g_inv

    @staticmethod
    def generator_to_t_norm(g_func: Callable[[float], float],
                            g_inv_func: Callable[[float], float]) -> Callable[[float, float], float]:
        """
        Constructs a t-norm T(a,b) from a generator g(a) and its pseudo-inverse g_inv(u).
        This is one of the fundamental definitions of Archimedean t-norms.

        Mathematical Expression:
        `T(a,b) = g_inv(g(a) + g(b))`

        Args:
            g_func (Callable): The generator function g(a).
            g_inv_func (Callable): The pseudo-inverse function g_inv(u).

        Returns:
            Callable: The constructed t-norm function T(a,b).
        """

        def T(a: float, b: float) -> float:
            """
            Internal function implementing the calculation logic for T(a,b).
            """
            try:
                val_g_a = g_func(a)
                val_g_b = g_func(b)
                # Avoid errors from adding infinities (e.g., NaN).
                if np.isinf(val_g_a) or np.isinf(val_g_b):
                    # If g(a) or g(b) is infinity, it usually means a or b is at a boundary (0 or 1).
                    # In such cases, the result of T(a,b) is typically 0 (e.g., T(0,b)=0).
                    # This is a simplified handling; a more rigorous approach would depend on the norm's characteristics.
                    if val_g_a == np.inf and val_g_b == np.inf:
                        return 0.0  # g(0)+g(0) -> g_inv(inf) -> 0
                    if val_g_a == np.inf:
                        return 0.0  # g(0)+g(b) -> g_inv(inf) -> 0
                    if val_g_b == np.inf:
                        return 0.0  # g(a)+g(0) -> g_inv(inf) -> 0

                return g_inv_func(val_g_a + val_g_b)
            except Exception as e:
                warnings.warn(f"Error constructing t-norm from generator: {e}", RuntimeWarning)
                return 0.0  # Return 0 on error.

        return T

    @staticmethod
    def generator_to_generator_inv(g_func: Callable[[float], float],
                                   domain_start: float = 0.0,
                                   domain_end: float = 1.0,
                                   max_iterations: int = 1000,
                                   epsilon: float = 1e-6,
                                   ) -> Callable[[float], float]:
        """
        Derives the pseudo-inverse g_inv_func from a generator g_func using a numerical
        method (bisection method). Applicable to strictly monotonic generators.
        The pseudo-inverse `g_inv(u)` satisfies `g(g_inv(u)) = u`.

        Args:
            g_func (Callable): The generator function, taking x as input and returning g(x).
                               Assumes g(x) is a strictly monotonic function mapping from
                               [domain_start, domain_end] to some range.
            domain_start (float): The starting value of the generator's input domain (usually 0).
            domain_end (float): The ending value of the generator's input domain (usually 1).
            max_iterations (int): Maximum number of iterations to prevent infinite loops.
            epsilon (float): Floating-point comparison precision, used to determine if a
                             sufficiently accurate solution has been found.

        Returns:
            Callable: The generator pseudo-inverse function, taking u as input and returning x,
                      such that g(x) ≈ u.
        """
        # Determine the monotonicity of the generator.
        # Try to pick two points within the domain to determine, avoiding g(0) or g(1) being infinity.
        try:
            # Pick points near domain_start and domain_end, avoiding direct use of boundary values
            # (which might lead to infinity or errors).
            val_at_start = g_func(domain_start + epsilon)
            val_at_end = g_func(domain_end - epsilon)
            # Determine if g(x) is decreasing or increasing. For example, -ln(x) is decreasing.
            is_decreasing = (val_at_start > val_at_end + epsilon)  # g(x) is decreasing (e.g., -ln(x)).
        except Exception:
            # If boundary value calculation fails, assume decreasing (common for many generators).
            is_decreasing = True

        def g_inv(u: float) -> float:
            """
            Pseudo-inverse function implementation: Given a target value u,
            finds x within the range [domain_start, domain_end].
            Uses the bisection method for searching.
            """
            low = domain_start
            high = domain_end

            # Fast path for boundary cases.
            # If the target value u is close to g(domain_start) or g(domain_end),
            # directly return the corresponding boundary value.
            # This helps handle cases where the generator approaches infinity at boundaries.
            if is_decreasing:
                # If target value u is close to g(domain_start) (usually inf), then x approaches domain_start.
                if u >= g_func(domain_start + epsilon) - epsilon:
                    return domain_start
                # If target value u is close to g(domain_end) (usually 0), then x approaches domain_end.
                if u <= g_func(domain_end - epsilon) + epsilon:
                    return domain_end
            else:  # g(x) is increasing.
                if u <= g_func(domain_start + epsilon) + epsilon:
                    return domain_start
                if u >= g_func(domain_end - epsilon) - epsilon:
                    return domain_end

            # Bisection search.
            for _ in range(max_iterations):
                mid = (low + high) / 2.0

                # Avoid mid-being too close to boundaries, which might cause g_func(mid) to overflow or error.
                # Force mid to stay within (domain_start, domain_end) and maintain a certain distance from boundaries.
                if mid <= domain_start + epsilon:
                    mid = domain_start + epsilon
                if mid >= domain_end - epsilon:
                    mid = domain_end - epsilon

                try:
                    g_mid = g_func(mid)
                except Exception:
                    # If g_func(mid) encounters an error (e.g., log(0)), try adjusting mid.
                    # This usually happens when mid is too close to the boundary of the generator function's domain.
                    if is_decreasing:
                        high = mid  # Assume mid is too small, need to increase x, so shrink high boundary.
                    else:
                        low = mid  # Assume mid is too large, need to decrease x, so expand low boundary.
                    continue

                # If g(mid) is sufficiently close to the target value u, then a solution is found.
                if abs(g_mid - u) < epsilon:
                    return mid

                # Adjust search interval based on monotonicity.
                if is_decreasing:  # g(x) is decreasing (e.g., -ln(x)).
                    if g_mid > u:  # g(mid) is greater than target value, meaning mid is too small, need to increase mid (because g is decreasing).
                        low = mid
                    else:  # g(mid) is smaller than target value, meaning mid is too large, need to decrease mid.
                        high = mid
                else:  # g(x) is increasing.
                    if g_mid < u:  # g(mid) is smaller than target value, meaning mid is too small, need to increase mid (because g is increasing).
                        low = mid
                    else:  # g(mid) is greater than target value, meaning mid is too large, need to decrease mid.
                        high = mid

            # Reached maximum iterations, return the current best approximation.
            # Even if an exact solution is not found, return the midpoint of the current bisection interval as an approximation.
            return (low + high) / 2.0

        return g_inv
