#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
IVQROFN Fuzzification Strategy Implementation.

This module implements the fuzzification strategy for Interval-Valued Q-Rung 
Orthopair Fuzzy Numbers (IVQROFNs). The strategy transforms crisp values into 
interval-valued fuzzy numbers where both membership and non-membership degrees 
are represented as intervals [a, b].

Key Features:
- Interval-based membership and non-membership degree calculation
- Constraint validation: max(md)^q + max(nmd)^q ≤ 1
- Support for multiple membership function parameters
- High-performance vectorized operations
- Flexible interval generation modes

Mathematical Foundation:
    For an IVQROFN A and crisp input x:
    - Membership interval: μ_A(x) = [μ_A^L(x), μ_A^U(x)]
    - Non-membership interval: ν_A(x) = [ν_A^L(x), ν_A^U(x)]
    - Constraint: (μ_A^U(x))^q + (ν_A^U(x))^q ≤ 1

Examples:
    .. code-block:: python

        from axisfuzzy.fuzzifier import Fuzzifier
        
        # Create IVQROFN fuzzifier with triangular membership function
        fuzzifier = Fuzzifier(
            mf='trimf',
            mtype='ivqrofn',
            q=2,
            mf_params={'a': 0.2, 'b': 0.5, 'c': 0.8},
            interval_width=0.1,
            interval_mode='symmetric'
        )
        
        # Fuzzify crisp values
        crisp_value = 0.6
        ivqrofn_result = fuzzifier(crisp_value)
        print(ivqrofn_result)  # <[0.75,0.85],[0.05,0.15]>
"""

import numpy as np
from typing import Optional, Dict, List, Union

from ...config import get_config
from ...core import Fuzznum, Fuzzarray, get_registry_fuzztype
from ...fuzzifier import FuzzificationStrategy, register_fuzzifier


@register_fuzzifier(is_default=True)
class IVQROFNFuzzificationStrategy(FuzzificationStrategy):
    """
    IVQROFN Fuzzification Strategy Implementation.
    
    This strategy transforms crisp values into interval-valued q-rung orthopair 
    fuzzy numbers where both membership and non-membership degrees are represented 
    as intervals. The strategy supports multiple interval generation modes and 
    maintains mathematical constraints.
    
    Attributes:
        mtype (str): Fuzzy number type identifier, set to 'ivqrofn'
        method (str): Strategy method identifier, set to 'default'
    
    Parameters:
        q (Optional[int]): Q-rung parameter for constraint validation
        pi (Optional[float]): Base hesitation parameter in range [0, 1]
        interval_width (float): Width parameter for interval generation
        interval_mode (str): Mode for interval generation ('symmetric', 'asymmetric', 'random')
        nmd_generation_mode (str): Mode for non-membership interval generation
    
    Interval Generation Modes:
        - 'symmetric': Intervals are symmetric around the central value
        - 'asymmetric': Intervals use different lower/upper spreads
        - 'random': Intervals are generated with random widths within bounds
    
    Mathematical Properties:
        - Membership interval: md = [md_L, md_U] where 0 ≤ md_L ≤ md_U ≤ 1
        - Non-membership interval: nmd = [nmd_L, nmd_U] where 0 ≤ nmd_L ≤ nmd_U ≤ 1
        - Constraint: md_U^q + nmd_U^q ≤ 1 (applied to upper bounds)
    
    Examples:
        .. code-block:: python
        
            # Create strategy with symmetric intervals
            strategy = IVQROFNFuzzificationStrategy(
                q=2, 
                interval_width=0.1, 
                interval_mode='symmetric'
            )
            
            # Fuzzify using membership function
            result = strategy.fuzzify(
                x=0.6, 
                mf_cls=TriangularMF, 
                mf_params_list=[{'a': 0.2, 'b': 0.5, 'c': 0.8}]
            )
    """
    
    mtype = 'ivqrofn'
    method = 'default'
    
    def __init__(self, 
                 q: Optional[int] = None, 
                 pi: Optional[float] = None,
                 interval_width: float = 0.1,
                 interval_mode: str = 'symmetric',
                 nmd_generation_mode: str = 'orthopair'):
        """
        Initialize IVQROFN fuzzification strategy.
        
        Parameters:
            q (Optional[int]): Q-rung parameter for constraint validation
            pi (Optional[float]): Base hesitation parameter in range [0, 1]
            interval_width (float): Width parameter for interval generation
            interval_mode (str): Mode for interval generation
            nmd_generation_mode (str): Mode for non-membership generation
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        q = q if q is not None else get_config().DEFAULT_Q
        super().__init__(q=q)
        
        self.pi = pi if pi is not None else 0.1
        self.interval_width = interval_width
        self.interval_mode = interval_mode
        self.nmd_generation_mode = nmd_generation_mode
        
        # Parameter validation
        if not (0 <= self.pi <= 1):
            raise ValueError("pi must be in [0, 1]")
        if not (0 < interval_width <= 0.5):
            raise ValueError("interval_width must be in (0, 0.5]")
        if interval_mode not in ['symmetric', 'asymmetric', 'random']:
            raise ValueError("interval_mode must be 'symmetric', 'asymmetric', or 'random'")
        if nmd_generation_mode not in ['orthopair', 'independent', 'complementary']:
            raise ValueError("nmd_generation_mode must be 'orthopair', 'independent', or 'complementary'")

    def fuzzify(self,
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
        """
        Transform crisp input into IVQROFN representation.
        
        This method computes interval-valued membership and non-membership degrees
        from crisp inputs using the specified membership function(s) and interval
        generation parameters.
        
        Parameters:
            x (Union[float, int, list, np.ndarray]): Crisp input value(s)
            mf_cls (type): Membership function class to instantiate
            mf_params_list (List[Dict]): List of parameter dictionaries for MF instances
        
        Returns:
            Union[Fuzznum, Fuzzarray]: Single IVQROFN or array of IVQROFNs
        
        Notes:
            For multiple parameter sets, the strategy computes multiple membership
            evaluations and combines them into a single IVQROFN with expanded intervals.
        """
        # Normalize input to numpy array for vectorized computation
        x = np.asarray(x, dtype=float)
        original_shape = x.shape
        is_scalar = x.ndim == 0
        
        # Flatten for processing if multidimensional
        x_flat = x.flatten() if x.ndim > 0 else np.array([x.item()])
        
        # Initialize result arrays
        if is_scalar:
            md_intervals = np.zeros(2, dtype=np.float64)
            nmd_intervals = np.zeros(2, dtype=np.float64)
        else:
            md_intervals = np.zeros((len(x_flat), 2), dtype=np.float64)
            nmd_intervals = np.zeros((len(x_flat), 2), dtype=np.float64)
        
        # Process each membership function parameter set
        all_md_values = []
        for params in mf_params_list:
            # Instantiate membership function with current parameters
            mf = mf_cls(**params)
            
            # Compute membership degrees for all input values
            md_values = np.clip(mf.compute(x_flat), 0, 1)
            all_md_values.append(md_values)
        
        # Combine multiple membership evaluations into intervals
        if len(all_md_values) == 1:
            # Single parameter set: generate intervals around the values
            md_centers = all_md_values[0]
            if is_scalar:
                md_intervals[:] = self._generate_md_interval(md_centers[0])
            else:
                for i, md_center in enumerate(md_centers):
                    md_intervals[i] = self._generate_md_interval(md_center)
        else:
            # Multiple parameter sets: use min/max with expansion
            all_md_array = np.array(all_md_values)  # Shape: (n_params, n_values)
            md_mins = np.min(all_md_array, axis=0)
            md_maxs = np.max(all_md_array, axis=0)
            
            if is_scalar:
                # Expand interval based on range and width parameter
                interval_range = md_maxs[0] - md_mins[0]
                expansion = max(interval_range, self.interval_width)
                center = (md_mins[0] + md_maxs[0]) / 2
                
                lower = max(0.0, center - expansion / 2)
                upper = min(1.0, center + expansion / 2)
                md_intervals[:] = [lower, upper]
            else:
                for i in range(len(x_flat)):
                    # Expand interval based on range and width parameter
                    interval_range = md_maxs[i] - md_mins[i]
                    expansion = max(interval_range, self.interval_width)
                    center = (md_mins[i] + md_maxs[i]) / 2
                    
                    lower = max(0.0, center - expansion / 2)
                    upper = min(1.0, center + expansion / 2)
                    md_intervals[i] = [lower, upper]
        
        # Generate non-membership intervals based on membership intervals
        if is_scalar:
            nmd_intervals[:] = self._generate_nmd_interval(md_intervals)
        else:
            for i in range(len(x_flat)):
                nmd_intervals[i] = self._generate_nmd_interval(md_intervals[i])
        
        # Reshape back to original dimensions if needed
        if not is_scalar and x.ndim > 1:
            md_intervals = md_intervals.reshape(original_shape + (2,))
            nmd_intervals = nmd_intervals.reshape(original_shape + (2,))
        elif not is_scalar:
            # For 1D arrays, add the interval dimension
            pass  # Already correct shape (n, 2)
        # For scalar, md_intervals and nmd_intervals are already shape (2,)
        
        # Create backend and fuzzy array
        backend_cls = get_registry_fuzztype().get_backend(self.mtype)
        backend = backend_cls.from_arrays(mds=md_intervals, nmds=nmd_intervals, q=self.q)
        fuzzy_array = Fuzzarray(backend=backend, mtype=self.mtype, q=self.q)
        
        # Return single Fuzznum for scalar input, Fuzzarray otherwise
        if is_scalar:
            return fuzzy_array[()]
        return fuzzy_array

    def _generate_md_interval(self, md_center: float) -> np.ndarray:
        """
        Generate membership degree interval around a central value.
        
        Parameters:
            md_center (float): Central membership degree value
        
        Returns:
            np.ndarray: Membership interval [md_lower, md_upper]
        """
        if self.interval_mode == 'symmetric':
            # Symmetric interval around center
            half_width = self.interval_width / 2
            lower = max(0.0, md_center - half_width)
            upper = min(1.0, md_center + half_width)
            
            # Adjust if interval hits boundaries
            if lower == 0.0:
                upper = min(1.0, self.interval_width)
            elif upper == 1.0:
                lower = max(0.0, 1.0 - self.interval_width)
                
        elif self.interval_mode == 'asymmetric':
            # Asymmetric interval with bias towards uncertainty
            lower_width = self.interval_width * 0.3
            upper_width = self.interval_width * 0.7
            
            lower = max(0.0, md_center - lower_width)
            upper = min(1.0, md_center + upper_width)
            
        else:  # 'random'
            # Random interval width within bounds
            random_width = np.random.uniform(0.05, self.interval_width)
            random_bias = np.random.uniform(-0.5, 0.5)
            
            lower = max(0.0, md_center - random_width * (0.5 + random_bias))
            upper = min(1.0, md_center + random_width * (0.5 - random_bias))
        
        # Ensure proper interval ordering
        if lower > upper:
            lower, upper = upper, lower
            
        return np.array([lower, upper], dtype=np.float64)

    def _generate_nmd_interval(self, md_interval: np.ndarray) -> np.ndarray:
        """
        Generate non-membership degree interval based on membership interval.
        
        Parameters:
            md_interval (np.ndarray): Membership interval [md_lower, md_upper]
        
        Returns:
            np.ndarray: Non-membership interval [nmd_lower, nmd_upper]
        """
        md_lower, md_upper = md_interval[0], md_interval[1]
        
        if self.nmd_generation_mode == 'orthopair':
            # Respect q-rung orthopair constraint: md_upper^q + nmd_upper^q ≤ 1
            max_nmd_upper = (1.0 - md_upper ** self.q) ** (1.0 / self.q)
            max_nmd_upper = max(0.0, max_nmd_upper)
            
            # Generate interval with constraint-aware upper bound
            if self.interval_mode == 'symmetric':
                nmd_center = min(self.pi, max_nmd_upper * 0.5)
                half_width = min(self.interval_width / 2, max_nmd_upper / 2)
                
                nmd_lower = max(0.0, nmd_center - half_width)
                nmd_upper = min(max_nmd_upper, nmd_center + half_width)
                
            else:  # asymmetric or random
                # Conservative approach: use constraint-based upper bound
                nmd_upper = min(self.pi, max_nmd_upper * 0.8)
                nmd_lower = max(0.0, nmd_upper - self.interval_width)
                
        elif self.nmd_generation_mode == 'independent':
            # Generate independently, then apply constraint
            if self.interval_mode == 'symmetric':
                nmd_center = self.pi
                half_width = self.interval_width / 2
                
                nmd_lower = max(0.0, nmd_center - half_width)
                nmd_upper = min(1.0, nmd_center + half_width)
            else:
                nmd_lower = max(0.0, self.pi - self.interval_width * 0.7)
                nmd_upper = min(1.0, self.pi + self.interval_width * 0.3)
            
            # Apply constraint post-generation
            max_nmd_upper = (1.0 - md_upper ** self.q) ** (1.0 / self.q)
            if nmd_upper > max_nmd_upper:
                # Scale down interval to satisfy constraint
                scale_factor = max_nmd_upper / nmd_upper if nmd_upper > 0 else 1.0
                nmd_lower *= scale_factor
                nmd_upper = max_nmd_upper
                
        else:  # 'complementary'
            # Complementary approach: nmd intervals complement md intervals
            available_lower = max(0.0, 1.0 - md_upper)
            available_upper = max(0.0, 1.0 - md_lower)
            
            # Apply q-rung constraint
            max_nmd_upper = (1.0 - md_upper ** self.q) ** (1.0 / self.q)
            available_upper = min(available_upper, max_nmd_upper)
            
            # Generate interval within available range
            if available_upper > available_lower:
                interval_range = min(self.interval_width, available_upper - available_lower)
                nmd_lower = available_lower
                nmd_upper = min(available_upper, available_lower + interval_range)
            else:
                nmd_lower = nmd_upper = 0.0
        
        # Ensure proper interval ordering and bounds
        nmd_lower = max(0.0, min(nmd_lower, 1.0))
        nmd_upper = max(nmd_lower, min(nmd_upper, 1.0))
        
        return np.array([nmd_lower, nmd_upper], dtype=np.float64)