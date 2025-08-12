#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/12 13:39
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
Random seed management for reproducible fuzzy number generation.

This module provides a global random state manager that ensures reproducible
random generation across the entire FuzzLab library while maintaining
thread safety and performance.
"""

import threading
from typing import Optional, Union

import numpy as np


class GlobalRandomState:
    """
    Global random state manager for FuzzLab.

    This class manages a single numpy.random.Generator instance that is used
    throughout the library for random fuzzy number generation. It provides
    thread-safe access and seed management capabilities.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()
        self._seed = None

    def set_seed(self, seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = None):
        """
        Set the global random seed.

        Args:
            seed: Random seed. Can be an integer, SeedSequence, or BitGenerator.
                 If None, uses system entropy.
        """
        with self._lock:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

    def get_generator(self) -> np.random.Generator:
        """
        Get the global random number generator.

        Returns:
            The current global Generator instance.
        """
        with self._lock:
            return self._rng

    def get_seed(self) -> Union[int, np.random.SeedSequence, np.random.BitGenerator]:
        """
        Get the current global random seed.

        Returns:
            The current seed value.
        """
        with self._lock:
            return self._seed

    def spawn_generator(self) -> np.random.Generator:
        """
        Spawn a new independent generator from the current state.

        This creates a new Generator that is statistically independent
        of the global one, useful for parallel or isolated operations.

        Returns:
            A new independent Generator instance.
        """
        with self._lock:
            # Use the global generator to spawn a new independent one
            spawned_rng = np.random.default_rng()
            # Initialize with entropy from current generator
            seed_sequence = np.random.SeedSequence(self._rng.integers(0, 2**32))
            return np.random.default_rng(seed_sequence)


# Global instance
_global_random_state = GlobalRandomState()


# Public interface functions
def _set_random_seed(seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = None):
    """Set the global random seed for FuzzLab."""
    _global_random_state.set_seed(seed)


def _get_random_generator() -> np.random.Generator:
    """Get the global random number generator."""
    return _global_random_state.get_generator()


def _spawn_random_generator() -> np.random.Generator:
    """Spawn a new independent random generator."""
    return _global_random_state.spawn_generator()


def _get_current_seed():
    """Get the current global seed value."""
    return _global_random_state.get_seed()


# Convenience aliases
get_seed = _get_current_seed
set_seed = _set_random_seed
get_rng = _get_random_generator
spawn_rng = _spawn_random_generator
