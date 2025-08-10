#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 21:56
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Dict

from typing import Dict

import numpy as np

from ....random.core.base import ParameterizedGenerator
from ....core import Fuzznum


class QROFNRandomGenerator(ParameterizedGenerator):
    """
    Random generator for Q-Rung Orthopair Fuzzy Numbers.

    This generator creates random QROFN instances with customizable
    membership and non-membership degrees that satisfy the q-rung constraint:
    μ^q + ν^q ≤ 1, where q ≥ 1.
    """

    @property
    def mtype(self) -> str:
        return 'qrofn'

    def get_default_parameters(self) -> dict:
        """
        Get default parameters for QROFN random generation.

        Returns:
            Dictionary with default parameters:
                - q: q-rung parameter (default: 1)
                - distribution: Sampling distribution (default: 'uniform')
                - mu_range: Range for membership degree (default: [0, 1])
                - nu_range: Range for non-membership degree (default: [0, 1])
                - constraint_method: How to handle q-rung constraint (default: 'conditional')
        """
        return {
            'q': 1,
            'distribution': 'uniform',
            'mu_range': [0., 1.],
            'nu_range': [0., 1.],
            'constraint_method': 'conditional',
            'max_attempts': 1000
        }
    
    def batch_generate_fuzznum(self, rng: np.random.Generator, size, **kwargs):
        """
        批量生成 size 个 QROFN Fuzznum，返回一维 Fuzznum 列表。
        """
        self.validate_parameters(**kwargs)
        params = self._merge_parameters(**kwargs)
        q = params['q']
        mu_range = params['mu_range']
        nu_range = params['nu_range']
        distribution = params['distribution']
        constraint_method = params['constraint_method']
        max_attempts = params['max_attempts']

        # 只实现最常用的 'conditional' 方法的批量生成
        if constraint_method != 'conditional':
            # 其他方法暂时用原有单个生成
            return [self.generate_fuzznum(rng, **kwargs) for _ in range(size)]

        # 批量生成 μ
        dist_params = params.copy()
        dist_params.pop('distribution', None)
        mus = np.array([
            self._sample_constrained(rng, distribution, mu_range[0], mu_range[1], **dist_params)
            for _ in range(size)
        ])
        # 批量计算每个 μ 下 ν 的最大取值
        max_nu_q = 1.0 - mus ** q
        max_nus = np.where(max_nu_q > 0, max_nu_q ** (1.0 / q), 0.0)
        effective_nu_max = np.minimum(nu_range[1], max_nus)
        # 批量生成 ν
        nus = np.array([
            self._sample_constrained(rng, distribution, nu_range[0], effective_nu_max[i], **dist_params)
            if nu_range[0] <= effective_nu_max[i] else nu_range[0]
            for i in range(size)
        ])
        # 批量构造 Fuzznum
        return [Fuzznum(q=q, mtype='qrofn').create(md=mus[i], nmd=nus[i]) for i in range(size)]

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate parameters for QROFN generation.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            True if parameters are valid.

        Raises:
            ValueError: If parameters are invalid.
        """
        params = self._merge_parameters(**kwargs)

        # validate q
        q = params['q']
        if not isinstance(q, int) or q < 1:
            raise ValueError(f"Invalid q value: '{q}'. Must be an integer >= 1.")

        # validate ranges
        mu_range = params['mu_range']
        nu_range = params['nu_range']

        if len(mu_range) != 2 or len(nu_range) != 2:
            raise ValueError("mu_range and nu_range must be lists of two elements.")

        self._validate_range(mu_range[0], 0.0, 1.0, "mu_range[0]")
        self._validate_range(mu_range[1], 0.0, 1.0, "mu_range[1]")
        self._validate_range(nu_range[0], 0.0, 1.0, "nu_range[0]")
        self._validate_range(nu_range[1], 0.0, 1.0, "nu_range[1]")

        if mu_range[0] > mu_range[1]:
            raise ValueError("mu_range[0] must be <= mu_range[1]")
        if nu_range[0] > nu_range[1]:
            raise ValueError("nu_range[0] must be <= nu_range[1]")

        # Validate constraint method
        valid_methods = ['reject_sample', 'normalize', 'conditional']
        if params['constraint_method'] not in valid_methods:
            raise ValueError(f"constraint_method must be one of {valid_methods}")

        return True

    def generate_fuzznum(self, rng: np.random.Generator, **kwargs) -> 'Fuzznum':
        """
        Generate a random QROFN Fuzznum.

        Args:
            rng: NumPy random generator instance.
            **kwargs: Generation parameters.

        Returns:
            A randomly generated QROFN Fuzznum instance.
        """
        # Validate and merge parameters
        self.validate_parameters(**kwargs)
        params = self._merge_parameters(**kwargs)

        # Generate μ and ν satisfying the q-rung constraint
        mu, nu = self._generate_constrained_pair(rng, params)

        return Fuzznum(q=params['q'], mtype='qrofn').create(md=mu, nmd=nu)

    def _generate_constrained_pair(self,
                                   rng: np.random.Generator,
                                   params: dict) -> tuple[float, float]:
        """
        Generate (μ, ν) pair satisfying the q-rung constraint.

        Args:
            rng: Random number generator.
            params: Generation parameters.

        Returns:
            Tuple of (mu, nu) values satisfying μ^q + ν^q ≤ 1.
        """
        method = params['constraint_method']

        if method == 'reject_sample':
            return self._reject_sampling(rng, params)
        elif method == 'normalize':
            return self._normalize_method(rng, params)
        elif method == 'conditional':
            return self._conditional_method(rng, params)
        else:
            raise ValueError(f"Unknown constraint method: {method}")

    def _reject_sampling(self,
                         rng: np.random.Generator,
                         params: dict) -> tuple[float, float]:
        """
        Use rejection sampling to generate valid (μ, ν) pairs.
        """
        q = params['q']
        mu_range = params['mu_range']
        nu_range = params['nu_range']
        max_attempts = params['max_attempts']
        distribution = params['distribution']

        # Prepare keyword arguments for distribution, removing keys passed positionally
        dist_params = params.copy()
        dist_params.pop('distribution', None)

        for _ in range(max_attempts):
            # Sample μ and ν independently
            mu = self._sample_constrained(
                rng, distribution, mu_range[0], mu_range[1], **dist_params
            )
            nu = self._sample_constrained(
                rng, distribution, nu_range[0], nu_range[1], **dist_params
            )

            # Check q-rung constraint
            if mu ** q + nu ** q <= 1.0:
                return mu, nu

        raise RuntimeError(
            f"Failed to generate valid QROFN pair after {max_attempts} attempts. "
            f"Consider relaxing the ranges or using a different constraint method."
        )

    def _normalize_method(self,
                          rng: np.random.Generator,
                          params: dict) -> tuple[float, float]:
        """
        Generate pair and normalize to satisfy constraint.
        """
        q = params['q']
        mu_range = params['mu_range']
        nu_range = params['nu_range']
        distribution = params['distribution']

        # Prepare keyword arguments for distribution, removing keys passed positionally
        dist_params = params.copy()
        dist_params.pop('distribution', None)

        # Sample μ and ν independently
        mu = self._sample_constrained(
            rng, distribution, mu_range[0], mu_range[1], **dist_params
        )
        nu = self._sample_constrained(
            rng, distribution, nu_range[0], nu_range[1], **dist_params
        )

        # Normalize if constraint is violated
        constraint_sum = mu ** q + nu ** q
        if constraint_sum > 1.0:
            scale_factor = (1.0 / constraint_sum) ** (1.0 / q)
            mu *= scale_factor
            nu *= scale_factor

        return mu, nu

    def _conditional_method(self,
                            rng: np.random.Generator,
                            params: dict) -> tuple[float, float]:
        """
        Generate μ first, then ν conditionally based on constraint.
        """
        q = params['q']
        mu_range = params['mu_range']
        nu_range = params['nu_range']
        distribution = params['distribution']

        # Prepare keyword arguments for distribution, removing keys passed positionally
        dist_params = params.copy()
        dist_params.pop('distribution', None)

        # Sample μ first
        mu = self._sample_constrained(
            rng, distribution, mu_range[0], mu_range[1], **dist_params
        )

        # Calculate maximum allowed ν given μ
        max_nu_q = 1.0 - mu ** q
        if max_nu_q <= 0:
            nu = 0.0
        else:
            max_nu = max_nu_q ** (1.0 / q)
            # Constrain ν range based on the q-rung constraint
            effective_nu_max = min(nu_range[1], max_nu)
            if nu_range[0] > effective_nu_max:
                nu = nu_range[0]  # This might violate constraint, but user ranges take precedence
            else:
                nu = self._sample_constrained(
                    rng, distribution, nu_range[0], effective_nu_max, **dist_params
                )

        return mu, nu


# Generator function for registration
def qrofn_random_generator(rng: np.random.Generator, **kwargs) -> 'Fuzznum':
    """
    Generator function for QROFN random generation.

    This function serves as the bridge between the RandomRegistry and
    the QROFNRandomGenerator class.

    Args:
        rng: NumPy random generator instance.
        **kwargs: Generation parameters.

    Returns:
        A randomly generated QROFN Fuzznum instance.
    """
    generator = QROFNRandomGenerator()
    return generator.generate_fuzznum(rng, **kwargs)
