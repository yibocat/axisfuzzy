#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/12 14:16
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional

import numpy as np

from ..core import Fuzznum, Fuzzarray


class BaseRandomGenerator(ABC):

    @property
    @abstractmethod
    def mtype(self) -> str:
        """The fuzzy number type (mtype) this generator handles."""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for random generation.

        Returns:
            A dictionary of default parameters.
        """
        pass

    @abstractmethod
    def validate_parameters(self, **params) -> None:
        """
        Validate the parameters for random generation.

        Args:
            **params: Parameters to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        pass

    @abstractmethod
    def fuzznum(self,
                rng: np.random.Generator,
                **params) -> 'Fuzznum':
        """
        Generate a single random Fuzznum instance.

        Args:
            rng: The NumPy random number generator instance.
            **params: Generation-specific parameters, which may include
                      structural parameters like 'q'.
        """
        pass

    @abstractmethod
    def fuzzarray(self,
                  rng: np.random.Generator,
                  shape: Tuple[int, ...],
                  **params) -> 'Fuzzarray':
        """
        Generate a Fuzzarray of a given shape with random values.

        This method is designed for high-performance batch generation.

        Args:
            rng: The NumPy random number generator instance.
            shape: The shape of the Fuzzarray to generate.
            **params: Generation-specific parameters, which may include
                      structural parameters like 'q'.
        """
        pass


class ParameterizedRandomGenerator(BaseRandomGenerator, ABC):
    """
    A helper base class for generators that use parameterized distributions.

    This class provides common utilities for merging parameters and sampling
    from standard probability distributions in a vectorized manner, simplifying
    the implementation of concrete generators.
    """
    def __init__(self):
        self._default_params = self.get_default_parameters()

    def _merge_parameters(self, **params) -> Dict[str, Any]:
        """
        Merge user-provided parameters with the generator's defaults.

        Args:
            **params: User-provided parameters.

        Returns:
            A dictionary containing the merged parameters.
        """
        params = self._default_params.copy()
        params.update(params)
        return params

    def _validate_range(self, name: str, value: float, min_val: float, max_val: float):
        """
        Validate that a numeric parameter is within a given range.

        Args:
            name: The name of the parameter for error messages.
            value: The value to check.
            min_val: The minimum allowed value (inclusive).
            max_val: The maximum allowed value (inclusive).

        Raises:
            ValueError: If the value is out of the specified range.
        """
        if not (min_val <= value <= max_val):
            raise ValueError(f"Parameter '{name}' must be between "
                             f"{min_val} and {max_val}, but got {value}.")

    def _sample_from_distribution(
        self,
        rng: np.random.Generator,
        size: Optional[int] = None,
        dist: str = 'uniform',
        low: float = 0.0,
        high: float = 1.0,
        **dist_params
    ) -> Union[float, np.ndarray]:
        """
        Sample one or more values from a specified distribution, clipped to a range.

        This is a vectorized helper for generating component arrays efficiently.

        Args:
            rng: The NumPy random number generator.
            size: The number of samples to generate. If None, returns a single float.
            dist: The distribution name ('uniform', 'beta', 'normal').
            low: The lower bound of the output range (inclusive).
            high: The upper bound of the output range (inclusive).
            **dist_params: Additional parameters for the distribution, e.g.,
                         'a' and 'b' for 'beta', 'loc' and 'scale' for 'normal'.

        Returns:
            A NumPy array of samples or a single float if size is None.

        Raises:
            ValueError: If an unsupported distribution is requested.
        """
        if low > high:
            raise ValueError(f"low ({low}) cannot be greater than high ({high}).")

        if dist == 'uniform':
            return rng.uniform(low, high, size)

        elif dist == 'beta':
            a = dist_params.get('a', 2.0)
            b = dist_params.get('b', 2.0)
            samples = rng.beta(a, b, size)
            return low + samples * (high - low)

        elif dist == 'normal':
            loc = dist_params.get('loc', (low + high) / 2)
            scale = dist_params.get('scale', (high - low) / 6)
            samples = rng.normal(loc, scale, size)
            return np.clip(samples, low, high)

        else:
            raise ValueError(f"Unsupported distribution: '{dist}'. "
                             f"Available distributions are 'uniform', 'beta', 'normal'.")
