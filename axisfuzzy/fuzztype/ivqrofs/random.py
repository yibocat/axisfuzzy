#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Random generator for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

This module provides a high-performance, vectorized random generator for IVQROFNs,
fully integrated with AxisFuzzy's random generation framework. The generator supports
multiple interval generation modes and distribution types while maintaining mathematical
constraints.

Key Features:
- High-performance vectorized generation for large-scale Fuzzarray creation
- Multiple interval generation modes (symmetric, asymmetric, random)
- Support for various probability distributions (uniform, beta, normal)
- Constraint-aware generation respecting q-rung orthopair constraints
- Flexible parameter configuration for different uncertainty modeling needs

Mathematical Foundation:
    An IVQROFN is characterized by:
    - Membership interval: μ = [μ^L, μ^U] where 0 ≤ μ^L ≤ μ^U ≤ 1
    - Non-membership interval: ν = [ν^L, ν^U] where 0 ≤ ν^L ≤ ν^U ≤ 1
    - Constraint: (μ^U)^q + (ν^U)^q ≤ 1, where q ≥ 1

Examples:
    .. code-block:: python

        import axisfuzzy.random as fr
        
        # Generate single IVQROFN with default parameters
        ivqrofn = fr.rand('ivqrofn', q=2)
        
        # Generate array with custom interval parameters
        arr = fr.rand('ivqrofn', 
                      shape=(100, 50), 
                      q=3,
                      interval_mode='symmetric',
                      md_dist='beta',
                      a=2.0, b=5.0)
"""

from typing import Any, Dict, Tuple, Optional

import numpy as np

from ...config import get_config
from ...core import Fuzznum, Fuzzarray
from ...random import register_random
from ...random.base import ParameterizedRandomGenerator

from .backend import IVQROFNBackend


@register_random
class IVQROFNRandomGenerator(ParameterizedRandomGenerator):
    """
    High-performance random generator for interval-valued q-rung orthopair fuzzy numbers.
    
    This generator leverages vectorized NumPy operations to efficiently create
    large Fuzzarray instances by directly populating the IVQROFNBackend with
    interval-valued membership and non-membership degrees.
    
    The generator supports multiple interval generation strategies:
    - Symmetric intervals: Equal spread around central values
    - Asymmetric intervals: Biased spread with different lower/upper ranges
    - Random intervals: Variable interval widths with stochastic bounds
    
    Mathematical constraints are enforced throughout generation:
    - All interval bounds remain in [0, 1]
    - Interval ordering: lower ≤ upper for both membership and non-membership
    - Q-rung constraint: max(md)^q + max(nmd)^q ≤ 1
    
    Attributes:
        mtype (str): Generator type identifier, set to 'ivqrofn'
    
    Examples:
        .. code-block:: python
        
            # Direct generator usage
            generator = IVQROFNRandomGenerator()
            rng = np.random.default_rng(42)
            
            # Generate single IVQROFN
            single = generator.fuzznum(rng, q=2, interval_mode='symmetric')
            
            # Generate batch of IVQROFNs
            batch = generator.fuzzarray(rng, shape=(1000,), q=3, 
                                       md_dist='beta', a=2.0, b=5.0)
    """
    
    mtype = "ivqrofn"
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Returns the default parameters for IVQROFN generation.
        
        Returns:
            Dict[str, Any]: Default parameter configuration including:
                - Distribution parameters for interval centers
                - Interval generation mode and width parameters
                - Constraint handling configurations
        """
        return {
            # Membership degree interval center distribution
            'md_dist': 'uniform',
            'md_low': 0.0,
            'md_high': 1.0,
            
            # Non-membership degree generation mode
            'nu_mode': 'orthopair',  # 'orthopair', 'independent', 'complementary'
            'nu_dist': 'uniform',
            'nu_low': 0.0,
            'nu_high': 1.0,
            
            # Interval generation parameters
            'interval_mode': 'symmetric',  # 'symmetric', 'asymmetric', 'random'
            'interval_width': 0.1,  # Base interval width (0, 0.5]
            'width_variation': 0.0,  # Random variation in interval width [0, 1]
            
            # Distribution shape parameters
            'a': 2.0, 'b': 2.0,  # Beta distribution parameters
            'loc': 0.5, 'scale': 0.15,  # Normal distribution parameters
            
            # Advanced constraint parameters
            'constraint_margin': 0.01,  # Safety margin for constraint satisfaction
            'min_interval_width': 0.01,  # Minimum allowed interval width
        }
    
    def validate_parameters(self, q: int, **kwargs) -> None:
        """
        Validates parameters for IVQROFN generation.
        
        Parameters:
            q (int): Q-rung parameter for constraint validation
            **kwargs: Additional generation parameters
        
        Raises:
            ValueError: If any parameter is outside valid range or inconsistent
        """
        if not isinstance(q, int) or q <= 0:
            raise ValueError(f"q must be a positive integer, but got {q}")
        
        # Validate range parameters
        if 'md_low' in kwargs and 'md_high' in kwargs:
            if kwargs['md_low'] > kwargs['md_high']:
                raise ValueError("md_low cannot be greater than md_high")
        
        if 'nu_low' in kwargs and 'nu_high' in kwargs:
            if kwargs['nu_low'] > kwargs['nu_high']:
                raise ValueError("nu_low cannot be greater than nu_high")
        
        # Validate interval generation parameters
        if 'interval_width' in kwargs:
            width = kwargs['interval_width']
            if not (0 < width <= 0.5):
                raise ValueError("interval_width must be in (0, 0.5]")
        
        if 'width_variation' in kwargs:
            variation = kwargs['width_variation']
            if not (0 <= variation <= 1):
                raise ValueError("width_variation must be in [0, 1]")
        
        if 'interval_mode' in kwargs:
            mode = kwargs['interval_mode']
            if mode not in ['symmetric', 'asymmetric', 'random']:
                raise ValueError("interval_mode must be 'symmetric', 'asymmetric', or 'random'")
        
        if 'nu_mode' in kwargs:
            nu_mode = kwargs['nu_mode']
            if nu_mode not in ['orthopair', 'independent', 'complementary']:
                raise ValueError("nu_mode must be 'orthopair', 'independent', or 'complementary'")
        
        # Validate constraint parameters
        if 'constraint_margin' in kwargs:
            margin = kwargs['constraint_margin']
            if not (0 <= margin <= 0.1):
                raise ValueError("constraint_margin must be in [0, 0.1]")
    
    def fuzznum(self,
                rng: np.random.Generator,
                q: Optional[int] = None,
                **kwargs) -> 'Fuzznum':
        """
        Generates a single random IVQROFN.
        
        This method creates a single interval-valued fuzzy number by generating
        interval centers and then creating intervals around them according to
        the specified interval generation mode.
        
        Parameters:
            rng (np.random.Generator): Random number generator instance
            q (Optional[int]): Q-rung parameter for constraint validation
            **kwargs: Additional generation parameters
        
        Returns:
            Fuzznum: Single IVQROFN instance with interval-valued components
        """
        params = self._merge_parameters(**kwargs)
        q = q if q is not None else get_config().DEFAULT_Q
        self.validate_parameters(q=q, **params)
        
        # Generate membership degree interval center
        md_center = self._sample_from_distribution(
            rng, size=None, dist=params['md_dist'],
            low=params['md_low'], high=params['md_high'],
            a=params['a'], b=params['b'],
            loc=params['loc'], scale=params['scale']
        )
        
        # Generate membership degree interval
        md_interval = self._generate_interval(
            rng, center=md_center, 
            base_width=params['interval_width'],
            mode=params['interval_mode'],
            variation=params['width_variation'],
            bounds=(0.0, 1.0)
        )
        
        # Generate non-membership degree interval based on mode
        nmd_interval = self._generate_nmd_interval(
            rng, md_interval=md_interval, 
            params=params, q=q
        )
        
        # Create and return Fuzznum
        return Fuzznum(mtype='ivqrofn', q=q).create(md=md_interval, nmd=nmd_interval)
    
    def fuzzarray(self,
                  rng: np.random.Generator,
                  shape: Tuple[int, ...],
                  q: Optional[int] = None,
                  **params) -> 'Fuzzarray':
        """
        Generates a Fuzzarray of IVQROFNs using high-performance vectorized operations.
        
        This method implements efficient batch generation by leveraging NumPy's
        vectorized operations and directly constructing the backend arrays.
        
        Parameters:
            rng (np.random.Generator): Random number generator instance
            shape (Tuple[int, ...]): Output array shape
            q (Optional[int]): Q-rung parameter for constraint validation
            **params: Additional generation parameters
        
        Returns:
            Fuzzarray: Batch of IVQROFN instances with specified shape
        """
        params = self._merge_parameters(**params)
        q = q if q is not None else get_config().DEFAULT_Q
        self.validate_parameters(q=q, **params)
        
        size = int(np.prod(shape))
        
        # 1. Generate membership degree interval centers (vectorized)
        md_centers = self._sample_from_distribution(
            rng, size=size, dist=params['md_dist'],
            low=params['md_low'], high=params['md_high'],
            a=params['a'], b=params['b'],
            loc=params['loc'], scale=params['scale']
        )
        
        # 2. Generate membership degree intervals (vectorized)
        md_intervals = self._generate_intervals_vectorized(
            rng, centers=md_centers,
            base_width=params['interval_width'],
            mode=params['interval_mode'],
            variation=params['width_variation'],
            bounds=(0.0, 1.0)
        )
        
        # 3. Generate non-membership degree intervals (vectorized)
        nmd_intervals = self._generate_nmd_intervals_vectorized(
            rng, md_intervals=md_intervals,
            params=params, q=q
        )
        
        # 4. Reshape arrays to target shape with interval dimension
        final_shape = shape + (2,)
        md_intervals = md_intervals.reshape(final_shape)
        nmd_intervals = nmd_intervals.reshape(final_shape)
        
        # 5. Create backend directly from arrays (High-Performance Path)
        backend = IVQROFNBackend.from_arrays(
            mds=md_intervals, nmds=nmd_intervals, q=q
        )
        
        # 6. Return the final Fuzzarray
        return Fuzzarray(backend=backend)
    
    def _generate_interval(self, 
                          rng: np.random.Generator, 
                          center: float, 
                          base_width: float,
                          mode: str,
                          variation: float,
                          bounds: Tuple[float, float]) -> np.ndarray:
        """
        Generate a single interval around a center value.
        
        Parameters:
            rng (np.random.Generator): Random number generator
            center (float): Center value for interval generation
            base_width (float): Base width for the interval
            mode (str): Interval generation mode
            variation (float): Random variation factor
            bounds (Tuple[float, float]): Valid range bounds
        
        Returns:
            np.ndarray: Generated interval [lower, upper]
        """
        min_bound, max_bound = bounds
        
        # Apply random variation to width
        if variation > 0:
            width_factor = 1.0 + variation * rng.uniform(-1, 1)
            width = base_width * max(0.1, width_factor)
        else:
            width = base_width
        
        if mode == 'symmetric':
            # Symmetric interval around center
            half_width = width / 2
            lower = max(min_bound, center - half_width)
            upper = min(max_bound, center + half_width)
            
            # Adjust if hitting bounds
            if lower == min_bound:
                upper = min(max_bound, min_bound + width)
            elif upper == max_bound:
                lower = max(min_bound, max_bound - width)
                
        elif mode == 'asymmetric':
            # Asymmetric interval with bias
            bias = rng.uniform(-0.3, 0.3)
            lower_width = width * (0.5 - bias)
            upper_width = width * (0.5 + bias)
            
            lower = max(min_bound, center - lower_width)
            upper = min(max_bound, center + upper_width)
            
        else:  # 'random'
            # Random interval positioning
            position_bias = rng.uniform(-0.5, 0.5)
            lower_width = width * (0.5 + position_bias * 0.5)
            upper_width = width * (0.5 - position_bias * 0.5)
            
            lower = max(min_bound, center - lower_width)
            upper = min(max_bound, center + upper_width)
        
        # Ensure proper ordering
        if lower > upper:
            lower, upper = upper, lower
            
        return np.array([lower, upper], dtype=np.float64)
    
    def _generate_intervals_vectorized(self,
                                      rng: np.random.Generator,
                                      centers: np.ndarray,
                                      base_width: float,
                                      mode: str,
                                      variation: float,
                                      bounds: Tuple[float, float]) -> np.ndarray:
        """
        Generate multiple intervals using vectorized operations.
        
        Parameters:
            rng (np.random.Generator): Random number generator
            centers (np.ndarray): Center values for interval generation
            base_width (float): Base width for intervals
            mode (str): Interval generation mode
            variation (float): Random variation factor
            bounds (Tuple[float, float]): Valid range bounds
        
        Returns:
            np.ndarray: Generated intervals with shape (n, 2)
        """
        n = len(centers)
        min_bound, max_bound = bounds
        
        # Apply random variation to widths
        if variation > 0:
            width_factors = 1.0 + variation * rng.uniform(-1, 1, size=n)
            widths = base_width * np.maximum(0.1, width_factors)
        else:
            widths = np.full(n, base_width)
        
        if mode == 'symmetric':
            # Symmetric intervals
            half_widths = widths / 2
            lowers = np.maximum(min_bound, centers - half_widths)
            uppers = np.minimum(max_bound, centers + half_widths)
            
            # Adjust for boundary hits
            at_min = (lowers == min_bound)
            at_max = (uppers == max_bound)
            
            uppers[at_min] = np.minimum(max_bound, min_bound + widths[at_min])
            lowers[at_max] = np.maximum(min_bound, max_bound - widths[at_max])
            
        elif mode == 'asymmetric':
            # Asymmetric intervals with random bias
            biases = rng.uniform(-0.3, 0.3, size=n)
            lower_widths = widths * (0.5 - biases)
            upper_widths = widths * (0.5 + biases)
            
            lowers = np.maximum(min_bound, centers - lower_widths)
            uppers = np.minimum(max_bound, centers + upper_widths)
            
        else:  # 'random'
            # Random interval positioning
            position_biases = rng.uniform(-0.5, 0.5, size=n)
            lower_widths = widths * (0.5 + position_biases * 0.5)
            upper_widths = widths * (0.5 - position_biases * 0.5)
            
            lowers = np.maximum(min_bound, centers - lower_widths)
            uppers = np.minimum(max_bound, centers + upper_widths)
        
        # Ensure proper ordering (swap if necessary)
        swap_mask = lowers > uppers
        lowers[swap_mask], uppers[swap_mask] = uppers[swap_mask], lowers[swap_mask]
        
        # Stack into interval array
        intervals = np.column_stack([lowers, uppers])
        return intervals
    
    def _generate_nmd_interval(self,
                              rng: np.random.Generator,
                              md_interval: np.ndarray,
                              params: Dict[str, Any],
                              q: int) -> np.ndarray:
        """
        Generate non-membership degree interval based on membership interval.
        
        Parameters:
            rng (np.random.Generator): Random number generator
            md_interval (np.ndarray): Membership interval [md_L, md_U]
            params (Dict[str, Any]): Generation parameters
            q (int): Q-rung parameter
        
        Returns:
            np.ndarray: Non-membership interval [nmd_L, nmd_U]
        """
        md_lower, md_upper = md_interval[0], md_interval[1]
        
        if params['nu_mode'] == 'orthopair':
            # Constraint-aware generation
            max_nmd_upper = (1.0 - md_upper ** q) ** (1.0 / q)
            max_nmd_upper = max(0.0, max_nmd_upper - params['constraint_margin'])
            
            # Generate center and create interval
            if max_nmd_upper > 0:
                nmd_center = self._sample_from_distribution(
                    rng, size=None, dist=params['nu_dist'],
                    low=params['nu_low'], high=min(params['nu_high'], max_nmd_upper),
                    a=params['a'], b=params['b'],
                    loc=params['loc'], scale=params['scale']
                )
                
                nmd_interval = self._generate_interval(
                    rng, center=nmd_center,
                    base_width=min(params['interval_width'], max_nmd_upper),
                    mode=params['interval_mode'],
                    variation=params['width_variation'],
                    bounds=(0.0, max_nmd_upper)
                )
            else:
                nmd_interval = np.array([0.0, 0.0])
                
        elif params['nu_mode'] == 'independent':
            # Independent generation with post-constraint
            nmd_center = self._sample_from_distribution(
                rng, size=None, dist=params['nu_dist'],
                low=params['nu_low'], high=params['nu_high'],
                a=params['a'], b=params['b'],
                loc=params['loc'], scale=params['scale']
            )
            
            nmd_interval = self._generate_interval(
                rng, center=nmd_center,
                base_width=params['interval_width'],
                mode=params['interval_mode'],
                variation=params['width_variation'],
                bounds=(0.0, 1.0)
            )
            
            # Apply constraint
            max_nmd_upper = (1.0 - md_upper ** q) ** (1.0 / q)
            if nmd_interval[1] > max_nmd_upper:
                scale_factor = max_nmd_upper / nmd_interval[1] if nmd_interval[1] > 0 else 1.0
                nmd_interval *= scale_factor
                
        else:  # 'complementary'
            # Complementary interval generation
            available_space = 1.0 - md_upper
            max_nmd_upper = (1.0 - md_upper ** q) ** (1.0 / q)
            effective_upper = min(available_space, max_nmd_upper)
            
            if effective_upper > 0:
                nmd_center = effective_upper * 0.5
                nmd_interval = self._generate_interval(
                    rng, center=nmd_center,
                    base_width=min(params['interval_width'], effective_upper),
                    mode=params['interval_mode'],
                    variation=params['width_variation'],
                    bounds=(0.0, effective_upper)
                )
            else:
                nmd_interval = np.array([0.0, 0.0])
        
        return nmd_interval
    
    def _generate_nmd_intervals_vectorized(self,
                                          rng: np.random.Generator,
                                          md_intervals: np.ndarray,
                                          params: Dict[str, Any],
                                          q: int) -> np.ndarray:
        """
        Generate non-membership degree intervals using vectorized operations.
        
        Parameters:
            rng (np.random.Generator): Random number generator
            md_intervals (np.ndarray): Membership degree intervals (n, 2)
            params (Dict[str, Any]): Generation parameters
            q (int): Q-rung parameter
        
        Returns:
            np.ndarray: Non-membership degree intervals (n, 2)
        """
        n = len(md_intervals)
        md_uppers = md_intervals[:, 1]
        
        # Calculate constraint-based upper bounds
        max_nmd_uppers = np.power(1.0 - np.power(md_uppers, q), 1.0 / q)
        max_nmd_uppers = np.maximum(0.0, max_nmd_uppers - params['constraint_margin'])
        
        if params['nu_mode'] == 'orthopair':
            # Vectorized constraint-aware generation
            effective_highs = np.minimum(params['nu_high'], max_nmd_uppers)
            
            # Generate centers with dynamic upper bounds
            nmd_centers = np.zeros(n)
            valid_mask = effective_highs > params['nu_low']
            
            if np.any(valid_mask):
                n_valid = np.sum(valid_mask)
                valid_centers = self._sample_from_distribution(
                    rng, size=n_valid, dist=params['nu_dist'],
                    low=0.0, high=1.0,  # Sample in [0,1] then scale
                    a=params['a'], b=params['b'],
                    loc=params['loc'], scale=params['scale']
                )
                
                # Scale to individual ranges
                ranges = effective_highs[valid_mask] - params['nu_low']
                nmd_centers[valid_mask] = params['nu_low'] + valid_centers * ranges
            
            # Generate intervals with constraint-based bounds
            effective_widths = np.minimum(params['interval_width'], max_nmd_uppers)
            nmd_intervals = self._generate_intervals_vectorized(
                rng, centers=nmd_centers,
                base_width=np.mean(effective_widths),  # Use average for simplicity
                mode=params['interval_mode'],
                variation=params['width_variation'],
                bounds=(0.0, 1.0)  # Will be clipped later
            )
            
            # Apply individual upper bound constraints
            nmd_intervals[:, 1] = np.minimum(nmd_intervals[:, 1], max_nmd_uppers)
            nmd_intervals[:, 0] = np.minimum(nmd_intervals[:, 0], nmd_intervals[:, 1])
                    
        else:  # 'independent' or 'complementary' modes use simplified vectorization
            if params['nu_mode'] == 'independent':
                # Independent generation
                nmd_centers = self._sample_from_distribution(
                    rng, size=n, dist=params['nu_dist'],
                    low=params['nu_low'], high=params['nu_high'],
                    a=params['a'], b=params['b'],
                    loc=params['loc'], scale=params['scale']
                )
            else:  # 'complementary'
                # Complementary generation
                available_spaces = 1.0 - md_uppers
                effective_uppers = np.minimum(available_spaces, max_nmd_uppers)
                nmd_centers = effective_uppers * 0.5
            
            # Generate intervals
            nmd_intervals = self._generate_intervals_vectorized(
                rng, centers=nmd_centers,
                base_width=params['interval_width'],
                mode=params['interval_mode'],
                variation=params['width_variation'],
                bounds=(0.0, 1.0)
            )
            
            # Apply constraint for all modes
            nmd_intervals[:, 1] = np.minimum(nmd_intervals[:, 1], max_nmd_uppers)
            nmd_intervals[:, 0] = np.minimum(nmd_intervals[:, 0], nmd_intervals[:, 1])
        
        return nmd_intervals