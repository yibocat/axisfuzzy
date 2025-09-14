#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Abstract base classes for random fuzzy number generation.

This module provides the foundational interfaces and utilities for implementing
random generators for different fuzzy number types (mtypes). The system follows
a plugin architecture where each fuzzy number type can register its own
specialized random generator.

The module defines two main abstract base classes:

1. :class:`BaseRandomGenerator` - Core interface that all generators must implement
2. :class:`ParameterizedRandomGenerator` - Helper class with distribution sampling utilities

Architecture
------------
The random generation system is designed around the following principles:

- **Type Specialization**: Each fuzzy number type (mtype) has its own generator
- **High Performance**: Vectorized batch generation for Fuzzarray creation
- **Flexibility**: Parameterized control over distributions and generation modes
- **Reproducibility**: Integration with NumPy's random number generation system

Random generators are registered globally and can be accessed through the
registry system. Each generator is responsible for creating both individual
Fuzznum instances and batch Fuzzarray instances with optimal performance.

Classes
-------
BaseRandomGenerator
    Abstract interface for all random generators.
ParameterizedRandomGenerator
    Helper base class with distribution sampling utilities.

See Also
--------
axisfuzzy.random.registry : Registration system for random generators
axisfuzzy.random.api : High-level API for random generation
axisfuzzy.fuzztype.qrofn.random : Example implementation for QROFN type

Examples
--------
Implementing a custom random generator:

.. code-block:: python

    from axisfuzzy.random.base import ParameterizedRandomGenerator
    from axisfuzzy.random import register_random

    @register_random
    class CustomRandomGenerator(ParameterizedRandomGenerator):
        mtype = "custom"

        def get_default_parameters(self):
            return {
                'param1': 1.0,
                'param2': 0.5,
                'distribution': 'uniform'
            }

        def validate_parameters(self, **params):
            if 'param1' in params and params['param1'] <= 0:
                raise ValueError("param1 must be positive")

        def fuzznum(self, rng, **params):
            # Implementation for single fuzzy number generation
            merged = self._merge_parameters(**params)
            value = self._sample_from_distribution(
                rng, dist=merged['distribution'],
                low=0, high=merged['param1']
            )
            return Fuzznum(mtype='custom').create(value=value)

        def fuzzarray(self, rng, shape, **params):
            # Implementation for batch generation
            merged = self._merge_parameters(**params)
            # ... batch generation logic
            return Fuzzarray(...)

Using the registered generator:

.. code-block:: python

    import axisfuzzy.random as fr

    # Generate single instance
    num = fr.rand('custom', param1=2.0)

    # Generate batch
    arr = fr.rand('custom', shape=(100,), param2=0.8)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional

import numpy as np

from ..core import Fuzznum, Fuzzarray


class BaseRandomGenerator(ABC):
    """
    Abstract base class for fuzzy number random generators.

    This class defines the interface that all random generators must implement
    to be compatible with the AxisFuzzy random generation system. Each fuzzy
    number type (mtype) should have a corresponding generator that inherits
    from this class.

    The generator is responsible for creating both individual :class:`Fuzznum`
    instances and batch :class:`Fuzzarray` instances with high performance
    vectorized operations.

    Attributes
    ----------
    mtype : str
        The fuzzy number type identifier that this generator handles.
        Must be set by concrete implementations.

    Notes
    -----
    Concrete implementations must set the ``mtype`` class attribute and
    implement all abstract methods. The generator should be stateless
    to ensure thread safety and consistent behavior.

    For high-performance batch generation, implementations should avoid
    creating intermediate Fuzznum objects and instead populate backend
    arrays directly.

    See Also
    --------
    ParameterizedRandomGenerator : Helper base class with distribution utilities
    axisfuzzy.random.registry.register_random : Decorator for automatic registration

    Examples
    --------
    Basic generator structure:

    .. code-block:: python

        class MyGenerator(BaseRandomGenerator):
            mtype = "mytype"

            def get_default_parameters(self):
                return {'param1': 1.0, 'param2': 0.5}

            def validate_parameters(self, **params):
                # Validate parameter values
                pass

            def fuzznum(self, rng, **params):
                # Generate single fuzzy number
                return Fuzznum(mtype=self.mtype).create(...)

            def fuzzarray(self, rng, shape, **params):
                # Generate batch array
                return Fuzzarray(...)
    """

    mtype: str = 'unknown'
    """str: The fuzzy number type identifier handled by this generator."""

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for random generation.

        Returns the default configuration for all parameters that control
        the random generation process. These defaults can be overridden
        when calling the generation methods.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their default values.
            The exact parameters depend on the specific fuzzy number type
            and generation strategy.

        Notes
        -----
        Default parameters should include all necessary configuration
        for the generator to function properly. Structural parameters
        like 'q' (q-rung) are typically not included in defaults as
        they are usually provided explicitly.

        Examples
        --------

        .. code-block:: python

            def get_default_parameters(self):
                return {
                    'md_dist': 'uniform',
                    'md_low': 0.0,
                    'md_high': 1.0,
                    'nu_mode': 'orthopair',
                    'distribution_params': {'a': 2.0, 'b': 2.0}
                }
        
        """
        pass

    @abstractmethod
    def validate_parameters(self, **params) -> None:
        """
        Validate parameters for random generation.

        Checks that all provided parameters have valid values and that
        parameter combinations are mathematically consistent. This method
        should raise appropriate exceptions for invalid configurations.

        Parameters
        ----------
        **params : dict
            Parameter values to validate, including both structural
            parameters (like 'q') and procedural parameters (like
            distribution settings).

        Raises
        ------
        ValueError
            If any parameter value is invalid or if parameter combinations
            violate mathematical constraints.
        TypeError
            If parameters have incorrect types.

        Notes
        -----
        Validation should be comprehensive but efficient, as it may be
        called frequently during batch generation. Consider caching
        validation results for repeated parameter sets.

        Examples
        --------
        
        .. code-block:: python

            def validate_parameters(self, **params):
                if 'q' in params:
                    q = params['q']
                    if not isinstance(q, int) or q <= 0:
                        raise ValueError(f"q must be positive integer, got {q}")

                if 'md_low' in params and 'md_high' in params:
                    if params['md_low'] > params['md_high']:
                        raise ValueError("md_low cannot exceed md_high")

        """
        pass

    @abstractmethod
    def fuzznum(self,
                rng: np.random.Generator,
                **params) -> 'Fuzznum':
        """
        Generate a single random Fuzznum instance.

        Creates one fuzzy number with the specified parameters using the
        provided random number generator for reproducible results.

        Parameters
        ----------
        rng : numpy.random.Generator
            NumPy random number generator instance for sampling.
        **params : dict
            Generation parameters including structural parameters (like 'q')
            and procedural parameters (like distribution settings).
            Parameters not provided will use default values.

        Returns
        -------
        Fuzznum
            A single fuzzy number instance of the appropriate mtype.

        Notes
        -----
        This method should be implemented efficiently but the primary
        performance focus should be on the :meth:`fuzzarray` method
        for batch generation scenarios.

        Examples
        --------
        .. code-block:: python

            def fuzznum(self, rng, **params):
                merged = self._merge_parameters(**params)
                self.validate_parameters(**merged)

                # Generate membership degree
                md = rng.uniform(merged['md_low'], merged['md_high'])

                # Generate non-membership degree with constraints
                max_nmd = (1 - md**merged['q']) ** (1/merged['q'])
                nmd = rng.uniform(0, max_nmd)

                return Fuzznum(mtype=self.mtype, q=merged['q']).create(
                    md=md, nmd=nmd
                )
        """
        pass

    @abstractmethod
    def fuzzarray(self,
                  rng: np.random.Generator,
                  shape: Tuple[int, ...],
                  **params) -> 'Fuzzarray':
        """
        Generate a Fuzzarray of random fuzzy numbers.

        Creates a multi-dimensional array of fuzzy numbers with the specified
        shape and parameters. This method is designed for high-performance
        batch generation using vectorized operations.

        Parameters
        ----------
        rng : numpy.random.Generator
            NumPy random number generator instance for sampling.
        shape : tuple of int
            The desired shape of the output Fuzzarray.
        **params : dict
            Generation parameters including structural parameters (like 'q')
            and procedural parameters (like distribution settings).

        Returns
        -------
        Fuzzarray
            Multi-dimensional array of fuzzy numbers with the specified shape.

        Notes
        -----
        This method should be highly optimized for performance:

        - Use vectorized NumPy operations for sampling
        - Avoid creating intermediate Fuzznum objects
        - Populate backend arrays directly when possible
        - Handle constraints efficiently using array operations

        For large arrays, consider memory-efficient generation strategies
        and avoid operations that scale quadratically with array size.

        Examples
        --------
        .. code-block:: python

            def fuzzarray(self, rng, shape, **params):
                merged = self._merge_parameters(**params)
                self.validate_parameters(**merged)

                size = int(np.prod(shape))

                # Vectorized generation
                mds = rng.uniform(
                    merged['md_low'], merged['md_high'], size=size
                )

                # Constraint handling
                max_nmds = (1 - mds**merged['q']) ** (1/merged['q'])
                nmds = rng.uniform(0, max_nmds)

                # Create backend directly
                backend = MyBackend.from_arrays(
                    mds=mds.reshape(shape),
                    nmds=nmds.reshape(shape),
                    q=merged['q']
                )

                return Fuzzarray(backend=backend)
        """
        pass


class ParameterizedRandomGenerator(BaseRandomGenerator, ABC):
    """
    Helper base class for generators using parameterized distributions.

    This class extends :class:`BaseRandomGenerator` with common utilities
    for parameter management and statistical distribution sampling. It
    simplifies the implementation of concrete generators by providing
    ready-to-use methods for common operations.

    The class provides:
    - Parameter merging and default handling
    - Vectorized sampling from standard distributions
    - Parameter validation utilities
    - Efficient distribution parameter management

    Notes
    -----
    This class is designed for generators that use standard statistical
    distributions (uniform, beta, normal) for sampling fuzzy number
    components. Generators with highly specialized sampling logic may
    inherit directly from :class:`BaseRandomGenerator`.

    The class maintains cached default parameters for efficiency during
    repeated generation calls.

    See Also
    --------
    BaseRandomGenerator : Core interface for all generators

    Examples
    --------
    Using the parameterized base class:

    .. code-block:: python

        @register_random
        class MyGenerator(ParameterizedRandomGenerator):
            mtype = "mytype"

            def get_default_parameters(self):
                return {
                    'md_dist': 'uniform',
                    'md_low': 0.0,
                    'md_high': 1.0,
                    'a': 2.0,  # Beta parameter
                    'b': 2.0   # Beta parameter
                }

            def fuzznum(self, rng, **params):
                merged = self._merge_parameters(**params)

                # Use built-in distribution sampling
                md = self._sample_from_distribution(
                    rng,
                    dist=merged['md_dist'],
                    low=merged['md_low'],
                    high=merged['md_high'],
                    a=merged['a'],
                    b=merged['b']
                )

                return Fuzznum(mtype=self.mtype).create(md=md)
    """

    def __init__(self):
        """
        Initialize the parameterized generator.

        Caches the default parameters for efficient access during
        generation operations.
        """
        self._default_params = self.get_default_parameters()

    def _merge_parameters(self, **params) -> Dict[str, Any]:
        """
        Merge user-provided parameters with generator defaults.

        Combines the default parameters with user-specified overrides,
        giving priority to user values while ensuring all necessary
        parameters are present.

        Parameters
        ----------
        **params : dict
            User-provided parameter overrides.

        Returns
        -------
        dict
            Complete parameter dictionary with user overrides applied
            to default values.

        Examples
        --------
        .. code-block:: python

            # With defaults: {'md_low': 0.0, 'md_high': 1.0, 'dist': 'uniform'}
            merged = self._merge_parameters(md_high=0.8, dist='beta')
            # Result: {'md_low': 0.0, 'md_high': 0.8, 'dist': 'beta'}
        """
        # FIX: Create a copy of defaults, then update with user params.
        merged_params = self._default_params.copy()
        merged_params.update(params)
        return merged_params

    def _validate_range(self, name: str, value: float, min_val: float, max_val: float):
        """
        Validate that a numeric parameter is within a specified range.

        Convenience method for common range validation operations in
        parameter validation routines.

        Parameters
        ----------
        name : str
            Parameter name for error messages.
        value : float
            Value to validate.
        min_val : float
            Minimum allowed value (inclusive).
        max_val : float
            Maximum allowed value (inclusive).

        Raises
        ------
        ValueError
            If value is outside the specified range.

        Examples
        --------
        .. code-block:: python

            def validate_parameters(self, **params):
                if 'md_low' in params:
                    self._validate_range('md_low', params['md_low'], 0.0, 1.0)
                if 'sigma' in params:
                    self._validate_range('sigma', params['sigma'], 0.001, 10.0)
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
        Sample values from a specified distribution with range clipping.

        Provides a unified interface for sampling from common statistical
        distributions with automatic range normalization. This method is
        optimized for vectorized operations and supports both scalar and
        array generation.

        Parameters
        ----------
        rng : numpy.random.Generator
            NumPy random number generator instance.
        size : int, optional
            Number of samples to generate. If None, returns a single float.
        dist : str, default 'uniform'
            Distribution name. Supported values:
            - 'uniform': Uniform distribution over [low, high]
            - 'beta': Beta distribution scaled to [low, high]
            - 'normal': Normal distribution clipped to [low, high]
        low : float, default 0.0
            Lower bound of output range (inclusive).
        high : float, default 1.0
            Upper bound of output range (inclusive).
        **dist_params : dict
            Distribution-specific parameters:
            - For 'beta': 'a' and 'b' (shape parameters)
            - For 'normal': 'loc' (mean) and 'scale' (standard deviation)

        Returns
        -------
        float or numpy.ndarray
            Random sample(s) within the specified range.
            Returns float if size is None, otherwise returns ndarray.

        Raises
        ------
        ValueError
            If low > high or if an unsupported distribution is specified.

        Notes
        -----
        Distribution scaling and clipping behavior:

        - **Uniform**: Direct sampling within [low, high]
        - **Beta**: Samples from Beta(a,b) then linearly scaled to [low, high]
        - **Normal**: Samples from Normal(loc, scale) then clipped to [low, high]

        For 'normal' distribution, default loc is (low+high)/2 and default
        scale is (high-low)/6 to provide reasonable coverage.

        Examples
        --------
        Single value sampling:

        .. code-block:: python

            # Uniform sampling
            val = self._sample_from_distribution(rng, dist='uniform', low=0.2, high=0.8)

            # Beta distribution sampling
            val = self._sample_from_distribution(
                rng, dist='beta', low=0, high=1, a=2.0, b=5.0
            )

            # Normal distribution sampling
            val = self._sample_from_distribution(
                rng, dist='normal', low=0, high=1, loc=0.5, scale=0.15
            )

        Vectorized sampling:

        .. code-block:: python

            # Generate 1000 samples from beta distribution
            samples = self._sample_from_distribution(
                rng, size=1000, dist='beta', low=0.1, high=0.9, a=3.0, b=2.0
            )

            # Generate array for backend initialization
            size = int(np.prod(shape))
            values = self._sample_from_distribution(
                rng, size=size, dist='uniform', low=0, high=1
            ).reshape(shape)
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
