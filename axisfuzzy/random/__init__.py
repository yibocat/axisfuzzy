#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/12 11:32
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

# Core base classes for extension
from .base import BaseRandomGenerator, ParameterizedRandomGenerator

# Registry management
from .registry import (
    get_registry_random as registry,
    register_random as register_random,
    unregister_random as unregister,
    get_random_generator as get_generator,
    list_registered as list_registered,
    is_registered as is_registered
)

# High-level API
from .api import (
    rand as rand,
    choice as choice,
    uniform as uniform,
    normal as normal,
    beta as beta
)

# Seed management
from .seed import (
    get_seed as get_seed,
    set_seed as set_seed,
    get_rng as get_rng,
    spawn_rng as spawn_rng
)

__all__ = [
    # Base classes
    'BaseRandomGenerator',
    'ParameterizedRandomGenerator',

    # Registry management
    'registry',
    'register_random',
    'unregister',
    'get_generator',
    'list_registered',
    'is_registered',

    # High-level API
    'rand',
    'choice',
    'uniform',
    'normal',
    'beta',

    # Seed management
    'get_seed',
    'set_seed',
    'get_rng',
    'spawn_rng'
]