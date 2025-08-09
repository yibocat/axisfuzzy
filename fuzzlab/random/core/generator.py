#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 20:43
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import threading
from typing import Optional, Union, Sequence, Any

import numpy as np

from ...core import Fuzznum, Fuzzarray

from .registry import get_random_registry


class RandomGenerator:
    """
    Main random generator class for FuzzLab.

    This class provides a unified interface for all random generation operations
    in FuzzLab, including seeded generation, state management, and dispatching
    to mtype-specific generators.

    Attributes:
        _rng (np.random.Generator): The underlying NumPy random generator.
        _lock (threading.RLock): Thread lock for safe concurrent access.
        _registry: Reference to the random generator registry.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random generator.

        Args:
            seed: Optional seed for reproducible random generation.
        """
        self._lock = threading.RLock()
        self._registry = get_random_registry()

        # Initialize with NumPy's default BitGenerator (PCG64)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

    def seed(self, seed_value: Optional[int] = None) -> None:
        """
        Set the random seed for reproducible results.

        Args:
            seed_value: Integer seed value. If None, use system time.
        """
        with self._lock:
            if seed_value is not None:
                self._rng = np.random.default_rng(seed_value)
            else:
                self._rng = np.random.default_rng()

    def get_state(self) -> dict:
        """
        Get the current random state.

        Returns:
            Dictionary containing the current state information.
        """
        with self._lock:
            return {
                'bit_generator': self._rng.bit_generator.state
            }

    def set_state(self, state: dict) -> None:
        """
        Set the random state.

        Args:
            state: State dictionary returned by get_state().
        """
        with self._lock:
            if 'bit_generator' in state:
                self._rng.bit_generator.state = state['bit_generator']

    def fuzznum(self, mtype: str, **kwargs) -> 'Fuzznum':
        """
        Generate a random Fuzznum of the specified mtype.

        Args:
            mtype: The fuzzy number type (e.g., 'qrofn').
            **kwargs: Additional parameters specific to the mtype.

        Returns:
            A randomly generated Fuzznum instance.

        Raises:
            ValueError: If no generator is registered for the specified mtype.
        """
        generator_func = self._registry.get_fuzznum_generator(mtype)
        if generator_func is None:
            raise ValueError(f"No random generator registered for mtype: {mtype}")

        # Pass the RNG instance to the generator function
        return generator_func(self._rng, **kwargs)

    def fuzzarray(self, mtype: str, shape: Union[int, tuple], **kwargs) -> 'Fuzzarray':
        """
        Generate a random Fuzzarray of the specified mtype and shape.

        Args:
            mtype: The fuzzy number type (e.g., 'qrofn').
            shape: Shape of the array (int or tuple of ints).
            **kwargs: Additional parameters specific to the mtype.

        Returns:
            A randomly generated Fuzzarray instance.
        """
        from fuzzlab.core.fuzzarray import fuzzarray

        # Normalize shape to tuple
        if isinstance(shape, int):
            shape = (shape,)

        # Generate individual Fuzznums and create array
        total_size = np.prod(shape)
        fuzznums = [self.fuzznum(mtype, **kwargs) for _ in range(total_size)]

        # Reshape to desired shape
        data = np.array(fuzznums, dtype=object).reshape(shape)
        return fuzzarray(data)

    def choice(self,
               a: Union[Sequence, 'Fuzzarray'],
               size: Optional[Union[int, tuple]] = None,
               replace: bool = True,
               p: Optional[Sequence[float]] = None) -> Any:
        """
        Choose random elements from a sequence or Fuzzarray.

        Args:
            a: Sequence to sample from (list, Fuzzarray, etc.).
            size: Output shape. If None, return single element.
            replace: Whether sampling is with replacement.
            p: Probabilities for each element. If None, uniform.

        Returns:
            Single element or array of chosen elements.
        """
        with self._lock:
            # Handle Fuzzarray input
            if hasattr(a, '__class__') and a.__class__.__name__ == 'Fuzzarray':
                # Convert Fuzzarray to flat list for sampling
                flat_data = a.data.flatten()
                chosen_indices = self._rng.choice(
                    len(flat_data), size=size, replace=replace, p=p
                )

                if size is None:
                    return flat_data[chosen_indices]
                else:
                    return [flat_data[i] for i in chosen_indices]

            # Handle regular sequences
            if hasattr(a, '__len__'):
                chosen_indices = self._rng.choice(
                    len(a), size=size, replace=replace, p=p
                )

                if size is None:
                    return a[chosen_indices]
                else:
                    return [a[i] for i in chosen_indices]

            raise TypeError(f"Cannot sample from object of type {type(a)}")

    def uniform(self,
                low: float = 0.0,
                high: float = 1.0,
                size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from uniform distribution.

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            size: Output shape.

        Returns:
            Random number(s) from uniform distribution.
        """
        with self._lock:
            return self._rng.uniform(low, high, size)

    def normal(self, loc: float = 0.0, scale: float = 1.0,
               size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from normal distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation.
            size: Output shape.

        Returns:
            Random number(s) from normal distribution.
        """
        with self._lock:
            return self._rng.normal(loc, scale, size)

    def beta(self, a: float, b: float,
             size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from beta distribution.

        Args:
            a: Alpha parameter.
            b: Beta parameter.
            size: Output shape.

        Returns:
            Random number(s) from beta distribution.
        """
        with self._lock:
            return self._rng.beta(a, b, size)

    @property
    def rng(self) -> np.random.Generator:
        """Access to the underlying NumPy random generator."""
        return self._rng
