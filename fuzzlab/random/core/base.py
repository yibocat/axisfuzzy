#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 20:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
Base classes for random generators in FuzzLab.

This module provides abstract base classes that define the interface
for random generation of different fuzzy number types.
"""
from abc import ABC, abstractmethod

from ...core import Fuzznum

import numpy as np


class BaseRandomGenerator(ABC):
    """
    Abstract base class for fuzzy number random generators.

    This class defines the interface that all mtype-specific random
    generators must implement.
    """

    @abstractmethod
    def generate_fuzznum(self, rng: np.random.Generator, **kwargs) -> 'Fuzznum':
        """
        Generate a random Fuzznum instance.

        Args:
            rng: NumPy random generator instance.
            **kwargs: Additional parameters specific to the fuzzy number type.

        Returns:
            A randomly generated Fuzznum instance.
        """
        pass

    @abstractmethod
    def validate_parameters(self, **kwargs):
        """
        Validate the parameters for random generation.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            True if parameters are valid, False otherwise.

        Raises:
            ValueError: If parameters are invalid.
        """
        pass

    def get_default_parameters(self) -> dict:
        """
        Get default parameters for random generation.

        Returns:
            Dictionary of default parameters.
        """
        return {}

    @property
    @abstractmethod
    def mtype(self) -> str:
        """The fuzzy number type this generator handles."""
        pass


class ParameterizedGenerator(BaseRandomGenerator):
    """
    Base class for generators that support parameterized random generation.

    This class provides common functionality for generators that support
    various probability distributions and parameter constraints.
    """

    def __init__(self):
        self._default_params = self.get_default_parameters()

    def _merge_parameters(self, **kwargs) -> dict:
        """
        Merge provided parameters with defaults.

        Args:
            **kwargs: User-provided parameters.

        Returns:
            Merged parameter dictionary.
        """
        params = self._default_params.copy()
        params.update(kwargs)
        return params

    def _validate_range(self,
                        value: float,
                        min_val: float,
                        max_val: float,
                        name: str) -> None:
        """
        Validate that a value is within the specified range.

        Args:
            value: Value to validate.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value is out of range.
        """
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"{name} must be in range [{min_val}, {max_val}], got {value}"
            )

    def _sample_constrained(self,
                            rng: np.random.Generator,
                            distribution: str = 'uniform',
                            low: float = 0.0, high: float = 1.0,
                            **dist_params) -> float:
        """
        Sample a value from a distribution with constraints.

        Args:
            rng: Random number generator.
            distribution: Distribution type ('uniform', 'beta', 'normal').
            low: Lower bound for clipping.
            high: Upper bound for clipping.
            **dist_params: Distribution-specific parameters.

        Returns:
            A random value within [low, high].
        """
        if distribution == 'uniform':
            return rng.uniform(low, high)
        elif distribution == 'beta':
            a = dist_params.get('a', 1.0)
            b = dist_params.get('b', 1.0)
            value = rng.beta(a, b)
            return low + value * (high - low)
        elif distribution == 'normal':
            loc = dist_params.get('loc', (low + high) / 2)
            scale = dist_params.get('scale', (high - low) / 4)
            value = rng.normal(loc, scale)
            return np.clip(value, low, high)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
