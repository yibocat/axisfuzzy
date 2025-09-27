#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) Fuzzification Strategy Implementation.

This module implements the fuzzification strategy for classical fuzzy sets (FS),
providing efficient transformation from crisp values to fuzzy representations
using membership functions based on Zadeh's fuzzy set theory.

The FSFuzzificationStrategy class provides:
- Simple and efficient crisp-to-fuzzy transformation
- Single membership degree calculation (md ∈ [0, 1])
- Vectorized operations for high performance
- Clean integration with AxisFuzzy's fuzzification framework

Mathematical Foundation:
    For crisp input x and membership function μ(x), the fuzzification process
    creates a fuzzy set A where:
    A = {⟨x, μ_A(x)⟩ | x ∈ X}
    
    The membership degree μ_A(x) ∈ [0, 1] represents the degree to which
    element x belongs to fuzzy set A.

Examples:
    .. code-block:: python

        from axisfuzzy.fuzzifier import Fuzzifier
        from axisfuzzy.membership import TriangularMF
        
        # Create FS fuzzifier with triangular membership function
        fuzzifier = Fuzzifier(
            mf=TriangularMF,
            mtype='fs',
            mf_params={'a': 0.2, 'b': 0.5, 'c': 0.8}
        )
        
        # Fuzzify crisp values
        result = fuzzifier([0.3, 0.6, 0.9])
        print(result)  # Array of FS fuzzy numbers
"""

from typing import Union, List, Dict, Optional
import numpy as np

from axisfuzzy.core import Fuzznum, Fuzzarray, get_registry_fuzztype
from axisfuzzy.fuzzifier import FuzzificationStrategy, register_fuzzifier

from .backend import FSBackend


@register_fuzzifier(is_default=True)
class FSFuzzificationStrategy(FuzzificationStrategy):
    """
    Fuzzification strategy for classical Fuzzy Sets (FS).
    
    This strategy implements the most fundamental fuzzification approach based on
    Zadeh's fuzzy set theory, where crisp values are transformed into fuzzy sets
    with only membership degrees. The strategy is optimized for simplicity and
    computational efficiency.
    
    Attributes:
        mtype (str): Membership type identifier, set to 'fs'
        method (str): Strategy method name, set to 'default'
    
    Mathematical Process:
        1. Apply membership function to crisp input: md = μ(x)
        2. Ensure membership degree is in [0, 1]: md = clip(md, 0, 1)
        3. Create FS fuzzy number with computed membership degree
    
    Performance Characteristics:
        - Minimal computational overhead (single membership calculation)
        - Optimal for large-scale fuzzy transformations
        - Vectorized operations for batch processing
        - Direct backend construction for maximum efficiency
    
    Examples:
        .. code-block:: python
        
            # Single value fuzzification
            strategy = FSFuzzificationStrategy()
            mf_cls = GaussianMF
            mf_params = [{'sigma': 0.1, 'c': 0.5}]
            
            result = strategy.fuzzify(0.6, mf_cls, mf_params)
            print(result)  # <0.8825> (approximate)
            
            # Array fuzzification
            values = np.array([0.3, 0.5, 0.7])
            array_result = strategy.fuzzify(values, mf_cls, mf_params)
            print(array_result.shape)  # (3,)
    """
    
    # Strategy identification attributes
    mtype = "fs"
    method = "default"

    def __init__(self, q: Optional[int] = None):
        """
        Initialize FS fuzzification strategy.
        
        Parameters:
            q (Optional[int]): q-rung parameter (inherited for compatibility,
                             not used in basic FS but maintained for framework consistency)
        
        Notes:
            The q parameter is inherited from the base FuzzificationStrategy class
            for compatibility with the framework, but is not used in basic
            fuzzy sets. It's maintained to ensure consistent interface across
            all fuzzy types.
        """
        super().__init__(q=q)

    def fuzzify(self,
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
        """
        Fuzzify crisp input using classical fuzzy set methodology.
        
        This method transforms crisp numerical values into FS fuzzy numbers by
        applying membership functions and creating appropriate fuzzy representations.
        The implementation supports both single values and vectorized batch processing.
        
        Parameters:
            x (Union[float, int, list, np.ndarray]): Crisp input data to be fuzzified
            mf_cls (type): Membership function class to instantiate
            mf_params_list (List[Dict]): List of parameter dictionaries for 
                                       membership function instances
        
        Returns:
            Union[Fuzznum, Fuzzarray]: Single Fuzznum for scalar input, 
                                     Fuzzarray for array-like input, or 
                                     stacked Fuzzarray for multiple parameter sets
        
        Mathematical Process:
            For each parameter set in mf_params_list:
            1. Instantiate membership function: mf = mf_cls(**params)
            2. Compute membership degrees: md = mf.compute(x)
            3. Apply range constraint: md = clip(md, 0, 1)
            4. Create FS backend and wrap in Fuzzarray
        
        Examples:
            .. code-block:: python
            
                # Single membership function
                x = 0.5
                mf_params = [{'a': 0.2, 'b': 0.5, 'c': 0.8}]
                result = strategy.fuzzify(x, TriangularMF, mf_params)
                
                # Multiple membership functions (linguistic terms)
                mf_params = [
                    {'a': 0.0, 'b': 0.2, 'c': 0.4},  # "Low"
                    {'a': 0.3, 'b': 0.5, 'c': 0.7},  # "Medium"  
                    {'a': 0.6, 'b': 0.8, 'c': 1.0}   # "High"
                ]
                stacked_result = strategy.fuzzify(x, TriangularMF, mf_params)
                print(stacked_result.shape)  # (3,) for 3 linguistic terms
        """
        # Normalize input to numpy array for vectorized computation
        x = np.asarray(x, dtype=float)
        is_scalar = x.ndim == 0
        
        # Store original shape for proper result construction
        original_shape = x.shape if not is_scalar else ()
        
        # Flatten for batch processing if needed
        if x.ndim > 0:
            x_flat = x.flatten()
            flat_size = x_flat.size
        else:
            x_flat = x
            flat_size = 1
        
        results = []

        # Process each membership function parameter set
        for params in mf_params_list:
            # Instantiate membership function with current parameters
            mf = mf_cls(**params)

            # Compute membership degrees using vectorized operations
            if flat_size > 1:
                mds = np.clip(mf.compute(x_flat), 0, 1)
                # Reshape back to original array shape
                mds = mds.reshape(original_shape)
            else:
                # Handle scalar input
                md_scalar = float(np.clip(mf.compute(x), 0, 1))
                mds = np.array(md_scalar) if not is_scalar else md_scalar

            # Create backend representation for current fuzzy type
            if is_scalar:
                # For scalar input, create single Fuzznum
                fuzznum = Fuzznum(mtype=self.mtype, q=self.q).create(md=mds)
                results.append(fuzznum)
            else:
                # For array input, create Fuzzarray with backend
                backend = FSBackend.from_arrays(mds=mds, q=self.q)
                fuzzarray = Fuzzarray(backend=backend, mtype=self.mtype, q=self.q)
                results.append(fuzzarray)

        # Return handling based on input type and parameter count
        if len(results) == 1:
            # Single parameter set: return individual result
            return results[0]
        else:
            # Multiple parameter sets: stack results
            if is_scalar:
                # For scalar input with multiple params, return list of Fuzznums
                # Convert to Fuzzarray for consistency
                md_values = np.array([r.md for r in results])
                backend = FSBackend.from_arrays(mds=md_values, q=self.q)
                return Fuzzarray(backend=backend, mtype=self.mtype, q=self.q)
            else:
                # For array input with multiple params, stack along new axis
                from axisfuzzy.mixin.factory import _stack_factory
                return _stack_factory(results[0], *results[1:], axis=0)