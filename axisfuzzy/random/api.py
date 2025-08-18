#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
High-level API for random fuzzy number and array generation.

This module provides the main user-facing functions for creating random
Fuzznum and Fuzzarray instances, as well as other utility functions
for random sampling.
"""

from typing import Any, Optional, Sequence, Tuple, Union, overload
import numpy as np

from ..core import Fuzznum, Fuzzarray

from .registry import (
    get_random_generator,
    list_registered_random,
)
from .seed import get_rng


def _resolve_rng(
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = None,
    rng: Optional[np.random.Generator] = None
) -> np.random.Generator:
    """
    Resolve the random number generator to use based on provided inputs.

    The priority is: rng > seed > global_rng.

    Args:
        seed: A specific seed to create a new generator.
        rng: An existing generator instance.

    Returns:
        A NumPy random number generator instance.
    """
    if rng is not None:
        return rng
    if seed is not None:
        return np.random.default_rng(seed)
    return get_rng()


@overload
def random_fuzz(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: None = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzznum: ...


@overload
def random_fuzz(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: Union[int, Tuple[int, ...]] = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzzarray: ...


def random_fuzz(
    mtype: Optional[str] = None,
    q: int = 1,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = None,
    rng: Optional[np.random.Generator] = None,
    **params
) -> Union[Fuzznum, Fuzzarray]:
    """
    Generate a random Fuzznum or Fuzzarray.

    This is the main factory function for random generation in FuzzLab.
    - If `shape` is None, a single Fuzznum is returned.
    - If `shape` is provided, a Fuzzarray of that shape is returned.

    The random seed is determined with the following priority:
    1. `rng`: An existing NumPy generator instance, for advanced control.
    2. `seed`: A local seed for this specific function call, ensuring local reproducibility.
    3. Global seed: Set via `axisfuzzy.random.set_seed()`, for global reproducibility.

    Args:
        mtype: The fuzzy number type to generate (e.g., 'qrofn').
        q: The q-rung for the fuzzy number, if applicable.
        shape: The shape of the Fuzzarray. If None, generates a single Fuzznum.
        seed: A local seed for this call.
        rng: An existing NumPy random generator instance.
        **params: Parameters passed to the mtype-specific generator

    Returns:
        A Fuzznum or Fuzzarray instance.
    """

    if mtype is None:
        from ..config import get_config
        mtype = get_config().DEFAULT_MTYPE

    params['q'] = q  # Ensure 'q' is always included in parameters

    generator = get_random_generator(mtype)
    if generator is None:
        available = list_registered_random()
        raise KeyError(f"No random generator registered for mtype '{mtype}'. "
                       f"Available mtypes: {available}")

    resolved_rng = _resolve_rng(seed, rng)

    # Generate a single Fuzznum
    if shape is None:
        return generator.fuzznum(resolved_rng, **params)

    # Normalize shape to a tuple
    shape = (shape,) if isinstance(shape, int) else shape
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be an int or a tuple of ints, but got {type(shape)}")

    # Generate a Fuzzarray using the high-performance method
    return generator.fuzzarray(resolved_rng, shape, **params)       # type: ignore[return-value]


def choice(
    obj: Fuzzarray,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    replace: bool = True,
    p: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Union[Any, Fuzzarray]:
    """
    Generate a random sample from a given 1-D array or Fuzzarray.

    Args:
        obj: The source Fuzzarray to sample from. Must be 1-dimensional.
        size: Output shape. If None, a single Fuzznum is returned (as a Fuzzarray of shape (1,)).
        replace: Whether the sample is with or without replacement.
        p: The probabilities associated with each element in the Fuzzarray.
        seed: A local seed for this call.
        rng: An existing NumPy random generator instance.

    Returns:
        The generated random samples as a Fuzzarray or NumPy array.
    """
    if not isinstance(obj, Fuzzarray):
        raise TypeError(f"Input for axisfuzzy.random.choice must be a Fuzzarray, got {type(obj)}")

    resolved_rng = _resolve_rng(seed, rng)

    if obj.ndim != 1:
        raise ValueError("Input Fuzzarray for choice must be 1-dimensional.")
    indices = resolved_rng.choice(len(obj), size=size, replace=replace, p=p)
    return obj[indices]


# TODO: 以下三个方法未来应该用于生成基于某种模式的随机模糊数,而不是单一的 float
def uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Union[float, np.ndarray]:
    """Generate random floats from a uniform distribution."""
    resolved_rng = _resolve_rng(seed, rng)
    return resolved_rng.uniform(low, high, shape)


def normal(
    loc: float = 0.0,
    scale: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Union[float, np.ndarray]:
    """Generate random floats from a normal (Gaussian) distribution."""
    resolved_rng = _resolve_rng(seed, rng)
    return resolved_rng.normal(loc, scale, shape)


def beta(
    a: float,
    b: float,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Union[float, np.ndarray]:
    """Generate random floats from a beta distribution."""
    resolved_rng = _resolve_rng(seed, rng)
    return resolved_rng.beta(a, b, shape)


rand = random_fuzz

# rand = random_fuzz
# registry = get_registry_random
# register = register_random_generator
# unregister = unregister_random
# get_generator = get_random_generator
# list_registered = list_registered_random
# is_registered = is_registered_random
