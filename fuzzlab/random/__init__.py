#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 20:19
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .core import RandomGenerator, RandomRegistry

_global_generator = RandomGenerator()


# Expose convenient functions
def seed(seed_value: int = None):
    """Set the global random seed for reproducible results."""
    return _global_generator.seed(seed_value)


def get_state():
    """Get the current random state."""
    return _global_generator.get_state()


def set_state(state):
    """Set the random state."""
    return _global_generator.set_state(state)


def fuzznum(mtype: str, **kwargs):
    """Generate a random Fuzznum of the specified mtype."""
    return _global_generator.fuzznum(mtype, **kwargs)


def fuzzarray(mtype: str, shape, **kwargs):
    """Generate a random Fuzzarray of the specified mtype and shape."""
    return _global_generator.fuzzarray(mtype, shape, **kwargs)


def choice(a, size=None, replace=True, p=None):
    """Choose random elements from a sequence."""
    return _global_generator.choice(a, size, replace, p)


def uniform(low=0.0, high=1.0, size=None):
    """Generate random numbers from uniform distribution."""
    return _global_generator.uniform(low, high, size)


def normal(loc=0.0, scale=1.0, size=None):
    """Generate random numbers from normal distribution."""
    return _global_generator.normal(loc, scale, size)


def beta(a, b, size=None):
    """Generate random numbers from beta distribution."""
    return _global_generator.beta(a, b, size)


# Export the global generator for advanced usage
generator = _global_generator

__all__ = [
    'seed', 'get_state', 'set_state',
    'fuzznum', 'fuzzarray', 'choice',
    'uniform', 'normal', 'beta',
    'generator', 'RandomGenerator', 'RandomRegistry'
]
