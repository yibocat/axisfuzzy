#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
High-level API for random fuzzy number and array generation.

This module provides the main user-facing interface for creating random
Fuzznum and Fuzzarray instances. It serves as the primary entry point
for all random fuzzy number generation operations in AxisFuzzy, offering
both convenience and flexibility through a unified API.

The module abstracts away the complexity of generator registration,
seed management, and type-specific generation logic, providing users
with simple functions that automatically dispatch to the appropriate
specialized generators based on the fuzzy number type (mtype).

Architecture
------------
The API layer sits at the top of AxisFuzzy's random generation system:

- **Unified Interface**: Single entry point (`random_fuzz`) for all generation
- **Automatic Dispatch**: Routes requests to appropriate type-specific generators
- **Flexible Seeding**: Supports global, local, and generator-specific seeding
- **Overload Support**: Type hints and overloads for better IDE integration
- **Performance Optimized**: Direct backend population for large arrays

Key Functions
-------------
random_fuzz : Main factory function for Fuzznum/Fuzzarray generation
choice : Random sampling from existing Fuzzarray instances
uniform, normal, beta : Utility functions for traditional random sampling
rand : Convenient alias for random_fuzz

See Also
--------
axisfuzzy.random.base : Abstract generator interfaces
axisfuzzy.random.registry : Generator registration system
axisfuzzy.random.seed : Global seed management
axisfuzzy.fuzztype.qrofn.random : Example generator implementation

Examples
--------
Basic random generation:

.. code-block:: python

    import axisfuzzy.random as fr

    # Set global seed for reproducibility
    fr.set_seed(42)

    # Generate single fuzzy number
    num = fr.rand('qrofn', q=2)

    # Generate fuzzy array
    arr = fr.rand('qrofn', shape=(100, 50), q=3)

    # Use local seed (doesn't affect global state)
    local_num = fr.rand('qrofn', q=2, seed=123)

Advanced usage with custom parameters:

.. code-block:: python

    import axisfuzzy.random as fr
    import numpy as np

    # Custom generator with specific parameters
    custom_arr = fr.rand(
        'qrofn',
        shape=(1000,),
        q=4,
        md_dist='beta',
        md_low=0.1,
        md_high=0.9,
        a=2.0,
        b=5.0,
        nu_mode='orthopair'
    )

    # Use your own RNG for full control
    my_rng = np.random.default_rng(456)
    controlled_arr = fr.rand('qrofn', shape=(500,), q=2, rng=my_rng)

Sampling from existing arrays:

.. code-block:: python

    import axisfuzzy.random as fr

    # Create source array
    source = fr.rand('qrofn', shape=(1000,), q=2)

    # Random sampling
    sample = fr.choice(source, size=100, replace=False)

    # Weighted sampling
    weights = fr.uniform(0, 1, size=1000)  # Random weights
    weighted_sample = fr.choice(source, size=50, p=weights)
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

    This internal utility function implements the priority system for
    determining which random number generator to use. It provides a
    consistent interface for all public API functions.

    Parameters
    ----------
    seed : int, numpy.random.SeedSequence, numpy.random.BitGenerator, optional
        A specific seed to create a new generator instance.
    rng : numpy.random.Generator, optional
        An existing generator instance to use directly.

    Returns
    -------
    numpy.random.Generator
        The resolved random number generator instance.

    Notes
    -----
    The priority order is:
    1. `rng`: If provided, use it directly (highest priority)
    2. `seed`: If provided, create new generator with this seed
    3. Global RNG: Use the global generator from seed management (lowest priority)

    This priority system allows for maximum flexibility:
    - Use `rng` for full control over generator state
    - Use `seed` for local reproducibility without affecting global state
    - Use neither to follow global random state for consistency

    Examples
    --------
    .. code-block:: python

        # Uses global generator
        rng1 = _resolve_rng()

        # Creates new generator with local seed
        rng2 = _resolve_rng(seed=42)

        # Uses provided generator directly
        my_rng = np.random.default_rng(123)
        rng3 = _resolve_rng(rng=my_rng)
    """
    if rng is not None:
        return rng
    if seed is not None:
        return np.random.default_rng(seed)
    return get_rng()


@overload
def rand(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: None = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzznum: ...


@overload
def rand(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: Union[int, Tuple[int, ...]] = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzzarray: ...


def rand(
    mtype: Optional[str] = None,
    q: int = 1,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = None,
    rng: Optional[np.random.Generator] = None,
    **params
) -> Union[Fuzznum, Fuzzarray]:
    """
    Generate random Fuzznum or Fuzzarray instances.

    This is the primary factory function for random fuzzy number generation
    in AxisFuzzy. It provides a unified interface that automatically dispatches
    to the appropriate type-specific generator based on the requested mtype.

    The function supports both single fuzzy number generation (when shape=None)
    and high-performance batch generation for arrays of any dimensionality.

    Parameters
    ----------
    mtype : str, optional
        The fuzzy number type identifier (e.g., 'qrofn', 'ivfn').
        If None, uses the default mtype from global configuration.
    q : int, default 1
        The q-rung parameter for q-rung fuzzy numbers. This is a structural
        parameter that defines the mathematical constraints of the fuzzy number.
    shape : int, tuple of int, optional
        The desired shape for batch generation:
        - None: Returns a single Fuzznum instance
        - int: Returns a 1D Fuzzarray with the specified size
        - tuple: Returns a multi-dimensional Fuzzarray with the specified shape
    seed : int, numpy.random.SeedSequence, numpy.random.BitGenerator, optional
        Local seed for this specific generation call. Creates a temporary
        generator that doesn't affect the global random state.
    rng : numpy.random.Generator, optional
        Existing NumPy generator instance to use. Provides full control
        over the random number generation process.
    **params : dict
        Additional parameters passed to the mtype-specific generator.
        These control the distribution and constraints of the generated values.
        Common parameters include:
        - md_dist, nu_dist: Distribution types ('uniform', 'beta', 'normal')
        - md_low, md_high: Range bounds for membership degrees
        - nu_low, nu_high: Range bounds for non-membership degrees
        - a, b: Beta distribution shape parameters
        - loc, scale: Normal distribution parameters

    Returns
    -------
    Fuzznum or Fuzzarray
        - Single Fuzznum if shape is None
        - Fuzzarray with specified shape otherwise

    Raises
    ------
    KeyError
        If no generator is registered for the specified mtype.
    TypeError
        If shape is not None, int, or tuple of int.
    ValueError
        If generator-specific parameters are invalid.

    Notes
    -----
    The random generation system uses a three-tier priority for randomness control:

    1. **rng parameter**: Direct generator control (highest priority)
    2. **seed parameter**: Local reproducibility for this call only
    3. **Global seed**: Library-wide consistency (lowest priority)

    For optimal performance with large arrays, the function uses vectorized
    operations and direct backend population, avoiding Python-level iteration
    over individual Fuzznum objects.

    See Also
    --------
    choice : Random sampling from existing Fuzzarray
    axisfuzzy.random.set_seed : Set global random seed
    axisfuzzy.random.spawn_rng : Create independent generators

    Examples
    --------
    Basic usage:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Single fuzzy number with default parameters
        num = fr.random_fuzz('qrofn', q=2)

        # 1D array of 100 fuzzy numbers
        arr1d = fr.random_fuzz('qrofn', q=3, shape=100)

        # 2D array with custom shape
        arr2d = fr.random_fuzz('qrofn', q=2, shape=(50, 20))

    Advanced parameter control:

    .. code-block:: python

        # Custom distribution parameters
        arr = fr.random_fuzz(
            'qrofn',
            q=4,
            shape=(1000,),
            md_dist='beta',        # Beta distribution for membership
            md_low=0.1,           # Lower bound
            md_high=0.9,          # Upper bound
            a=2.0,                # Beta shape parameter
            b=5.0,                # Beta shape parameter
            nu_mode='orthopair'   # Constraint mode
        )

    Reproducibility control:

    .. code-block:: python

        # Global reproducibility
        fr.set_seed(42)
        arr1 = fr.random_fuzz('qrofn', q=2, shape=(100,))

        # Local reproducibility (doesn't affect global state)
        arr2 = fr.random_fuzz('qrofn', q=2, shape=(100,), seed=123)

        # Full control with custom generator
        import numpy as np
        my_rng = np.random.default_rng(456)
        arr3 = fr.random_fuzz('qrofn', q=2, shape=(100,), rng=my_rng)

    Performance considerations:

    .. code-block:: python

        # Efficient: Single vectorized call
        large_arr = fr.random_fuzz('qrofn', q=2, shape=(10000, 100))

        # Inefficient: Multiple individual calls
        # individual_nums = [fr.random_fuzz('qrofn', q=2) for _ in range(10000)]

    Different fuzzy number types:

    .. code-block:: python

        # Q-rung orthopair fuzzy numbers
        qrofn_arr = fr.random_fuzz('qrofn', q=3, shape=(100,))

        # Interval-valued fuzzy numbers
        ivfn_arr = fr.random_fuzz('ivfn', shape=(100,))

        # Other types as available
        available_types = fr.list_registered_random()
        print(f"Available types: {available_types}")
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
    Generate random samples from a given Fuzzarray.

    This function provides random sampling capabilities similar to numpy.random.choice
    but specifically designed for Fuzzarray objects. It enables random selection
    of fuzzy numbers from existing arrays, with support for weighted sampling
    and sampling with or without replacement.

    Parameters
    ----------
    obj : Fuzzarray
        The source Fuzzarray to sample from. Must be 1-dimensional for
        proper indexing semantics.
    size : int, tuple of int, optional
        Output shape for the sampled results:
        - None: Returns a single Fuzznum
        - int: Returns 1D Fuzzarray with specified size
        - tuple: Returns multi-dimensional Fuzzarray with specified shape
    replace : bool, default True
        Whether sampling is with replacement (True) or without replacement (False).
        Without replacement requires size <= len(obj).
    p : array-like of float, optional
        Probabilities associated with each element in obj. Must have the same
        length as obj and sum to 1.0. If None, assumes uniform distribution.
    seed : int, optional
        Local seed for reproducible sampling that doesn't affect global state.
    rng : numpy.random.Generator, optional
        Existing generator instance for full control over randomness.

    Returns
    -------
    Fuzznum or Fuzzarray
        Random sample(s) from the source array:
        - Single Fuzznum if size is None
        - Fuzzarray with specified shape otherwise

    Raises
    ------
    TypeError
        If obj is not a Fuzzarray instance.
    ValueError
        If obj is not 1-dimensional, or if sampling parameters are invalid.

    Notes
    -----
    This function performs true sampling of Fuzznum objects, not just index
    sampling. The returned objects are independent copies of the selected
    elements, so modifications to the sample won't affect the original array.

    For large-scale sampling operations, consider the memory implications
    since each sampled Fuzznum is a complete copy of the original.

    See Also
    --------
    random_fuzz : Generate new random fuzzy numbers
    numpy.random.Generator.choice : NumPy's choice function

    Examples
    --------
    Basic sampling:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Create source array
        source = fr.rand('qrofn', q=2, shape=(1000,))

        # Sample single element
        single_sample = fr.choice(source)

        # Sample multiple elements with replacement
        samples = fr.choice(source, size=100)

        # Sample without replacement
        unique_samples = fr.choice(source, size=50, replace=False)

    Weighted sampling:

    .. code-block:: python

        # Create source and weights
        source = fr.rand('qrofn', q=2, shape=(100,))
        weights = fr.uniform(0, 1, size=100)
        weights = weights / weights.sum()  # Normalize to sum to 1

        # Weighted sampling
        weighted_samples = fr.choice(source, size=20, p=weights)

    Multi-dimensional sampling:

    .. code-block:: python

        # Create 1D source
        source = fr.rand('qrofn', q=3, shape=(500,))

        # Sample into 2D result
        samples_2d = fr.choice(source, size=(10, 20))
        print(samples_2d.shape)  # Output: (10, 20)

    Reproducible sampling:

    .. code-block:: python

        source = fr.rand('qrofn', q=2, shape=(100,))

        # Reproducible with local seed
        sample1 = fr.choice(source, size=10, seed=42)
        sample2 = fr.choice(source, size=10, seed=42)
        # sample1 and sample2 will be identical

        # Using custom generator
        import numpy as np
        my_rng = np.random.default_rng(123)
        controlled_sample = fr.choice(source, size=10, rng=my_rng)

    Statistical sampling applications:

    .. code-block:: python

        # Bootstrap sampling
        data = fr.rand('qrofn', q=2, shape=(1000,))
        bootstrap_samples = [
            fr.choice(data, size=len(data), replace=True, seed=i)
            for i in range(100)  # 100 bootstrap resamples
        ]

        # Stratified sampling (with custom weights)
        # Assume some scoring function exists
        # scores = [score_function(x) for x in data]
        # weights = np.array(scores) / sum(scores)
        # stratified_sample = fr.choice(data, size=100, p=weights)
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
    """
    Generate random floats from a uniform distribution.

    This utility function provides direct access to NumPy's uniform random
    number generation, integrated with AxisFuzzy's seed management system.
    It's useful for generating auxiliary random data or parameters for
    fuzzy number generation.

    Parameters
    ----------
    low : float, default 0.0
        Lower boundary of the output interval (inclusive).
    high : float, default 1.0
        Upper boundary of the output interval (exclusive).
    shape : int, tuple of int, optional
        Output shape:
        - None: Returns a single float
        - int: Returns 1D array with specified size
        - tuple: Returns multi-dimensional array with specified shape
    seed : int, optional
        Local seed for this specific call.
    rng : numpy.random.Generator, optional
        Existing generator instance to use.

    Returns
    -------
    float or numpy.ndarray
        Random sample(s) from uniform distribution in [low, high).

    Notes
    -----
    This function may be extended in future versions to generate
    fuzzy numbers with uniform membership patterns rather than
    scalar floats.

    See Also
    --------
    normal : Generate from normal distribution
    beta : Generate from beta distribution
    numpy.random.Generator.uniform : NumPy's uniform function

    Examples
    --------
    .. code-block:: python

        import axisfuzzy.random as fr

        # Single random float
        val = fr.uniform(0, 1)

        # Array of random floats
        arr = fr.uniform(-1, 1, shape=(100,))

        # 2D array with specific bounds
        matrix = fr.uniform(0.2, 0.8, shape=(10, 20))

        # Reproducible generation
        reproducible_arr = fr.uniform(0, 1, shape=(50,), seed=42)
    """
    resolved_rng = _resolve_rng(seed, rng)
    return resolved_rng.uniform(low, high, shape)


def normal(
    loc: float = 0.0,
    scale: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Union[float, np.ndarray]:
    """
    Generate random floats from a normal (Gaussian) distribution.

    This utility function provides direct access to NumPy's normal random
    number generation, integrated with AxisFuzzy's seed management system.
    It's commonly used for generating noise, perturbations, or normally
    distributed parameters for fuzzy computations.

    Parameters
    ----------
    loc : float, default 0.0
        Mean of the distribution.
    scale : float, default 1.0
        Standard deviation of the distribution (must be positive).
    shape : int, tuple of int, optional
        Output shape:
        - None: Returns a single float
        - int: Returns 1D array with specified size
        - tuple: Returns multi-dimensional array with specified shape
    seed : int, optional
        Local seed for this specific call.
    rng : numpy.random.Generator, optional
        Existing generator instance to use.

    Returns
    -------
    float or numpy.ndarray
        Random sample(s) from normal distribution N(loc, scale²).

    Notes
    -----
    This function may be extended in future versions to generate
    fuzzy numbers with normal membership patterns rather than
    scalar floats.

    See Also
    --------
    uniform : Generate from uniform distribution
    beta : Generate from beta distribution
    numpy.random.Generator.normal : NumPy's normal function

    Examples
    --------
    .. code-block:: python

        import axisfuzzy.random as fr

        # Standard normal (mean=0, std=1)
        val = fr.normal()

        # Custom mean and standard deviation
        arr = fr.normal(loc=5.0, scale=2.0, shape=(100,))

        # Generate noise for perturbation
        noise = fr.normal(0, 0.1, shape=(1000, 10))

        # Reproducible generation
        reproducible_noise = fr.normal(0, 1, shape=(50,), seed=42)
    """
    resolved_rng = _resolve_rng(seed, rng)
    return resolved_rng.normal(loc, scale, shape)


def beta(
    a: float,
    b: float,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Union[float, np.ndarray]:
    """
    Generate random floats from a beta distribution.

    This utility function provides direct access to NumPy's beta random
    number generation, integrated with AxisFuzzy's seed management system.
    Beta distributions are particularly useful in fuzzy logic due to their
    support on [0, 1] and flexible shape characteristics.

    Parameters
    ----------
    a : float
        Shape parameter alpha (must be positive). Controls the distribution's
        behavior near x=0.
    b : float
        Shape parameter beta (must be positive). Controls the distribution's
        behavior near x=1.
    shape : int, tuple of int, optional
        Output shape:
        - None: Returns a single float
        - int: Returns 1D array with specified size
        - tuple: Returns multi-dimensional array with specified shape
    seed : int, optional
        Local seed for this specific call.
    rng : numpy.random.Generator, optional
        Existing generator instance to use.

    Returns
    -------
    float or numpy.ndarray
        Random sample(s) from Beta(a, b) distribution on [0, 1].

    Notes
    -----
    Beta distributions are commonly used in fuzzy number generation
    because they naturally produce values in [0, 1], which aligns
    with membership degree constraints. Different (a, b) combinations
    produce different distribution shapes:

    - a = b = 1: Uniform distribution
    - a = b > 1: Symmetric, bell-shaped (concentrated around 0.5)
    - a = b < 1: U-shaped (concentrated near 0 and 1)
    - a ≠ b: Asymmetric distributions

    This function may be extended in future versions to generate
    fuzzy numbers with beta-distributed membership patterns.

    See Also
    --------
    uniform : Generate from uniform distribution
    normal : Generate from normal distribution
    numpy.random.Generator.beta : NumPy's beta function

    Examples
    --------
    .. code-block:: python

        import axisfuzzy.random as fr

        # Symmetric beta distribution
        symmetric = fr.beta(2.0, 2.0, shape=(100,))

        # Asymmetric, skewed toward 0
        skewed_left = fr.beta(0.5, 2.0, shape=(100,))

        # Asymmetric, skewed toward 1
        skewed_right = fr.beta(2.0, 0.5, shape=(100,))

        # U-shaped distribution
        u_shaped = fr.beta(0.5, 0.5, shape=(100,))

        # Generate for membership degree parameters
        membership_params = fr.beta(3.0, 1.5, shape=(1000, 2))

        # Reproducible generation
        reproducible_beta = fr.beta(2.0, 5.0, shape=(50,), seed=42)
    """
    resolved_rng = _resolve_rng(seed, rng)
    return resolved_rng.beta(a, b, shape)


# rand = random_fuzz      # type: ignore

# rand = random_fuzz
# registry = get_registry_random
# register = register_random_generator
# unregister = unregister_random
# get_generator = get_random_generator
# list_registered = list_registered_random
# is_registered = is_registered_random
