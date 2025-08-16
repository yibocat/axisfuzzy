#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/12 15:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab


"""
Random generator for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

This module provides a high-performance, vectorized random generator for QROFNs,
fully integrated with FuzzLab's random generation framework.
"""
from typing import Any, Dict, Tuple

import numpy as np

from ...core import Fuzznum, Fuzzarray
from ...random import register_random
from ...random.base import ParameterizedRandomGenerator

from .backend import QROFNBackend


@register_random
class QROFNRandomGenerator(ParameterizedRandomGenerator):
    """
    High-performance random generator for q-rung orthopair fuzzy numbers.

    This generator leverages vectorized NumPy operations to efficiently create
    large Fuzzarray instances by directly populating the QROFNBackend.
    """

    mtype = "qrofn"

    def get_default_parameters(self) -> Dict[str, Any]:
        """Returns the default parameters for QROFN generation."""
        return {
            'md_dist': 'uniform',
            'md_low': 0.0,
            'md_high': 1.0,
            'nu_mode': 'orthopair',  # 'orthopair' or 'independent'
            'nu_dist': 'uniform',
            'nu_low': 0.0,
            'nu_high': 1.0,
            # Beta distribution parameters
            'a': 2.0,
            'b': 2.0,
            # Normal distribution parameters
            'loc': 0.5,
            'scale': 0.15
        }

    def validate_parameters(self, q: int, **kwargs) -> None:
        """Validates parameters for QROFN generation."""
        if not isinstance(q, int) or q <= 0:
            raise ValueError(f"q must be a positive integer, but got {q}")

        if 'md_low' in kwargs and 'md_high' in kwargs and kwargs['md_low'] > kwargs['md_high']:
            raise ValueError("md_low cannot be greater than md_high")

        if 'nu_low' in kwargs and 'nu_high' in kwargs and kwargs['nu_low'] > kwargs['nu_high']:
            raise ValueError("nu_low cannot be greater than nu_high")

        if 'nu_mode' in kwargs and kwargs['nu_mode'] not in ['orthopair', 'independent']:
            raise ValueError("nu_mode must be 'orthopair' or 'independent'")

    def fuzznum(self, rng: np.random.Generator, q: int = 1, **kwargs) -> 'Fuzznum':
        """
        Generates a single random QROFN.

        This is achieved by generating a Fuzzarray of shape (1,) and extracting
        the single element, ensuring logic reuse and consistency.
        """

        params = self._merge_parameters(**kwargs)
        self.validate_parameters(q=q, **params)

        # Generate a single membership degree
        md = self._sample_from_distribution(
            rng,
            size=None,  # Generate a single float
            dist=params['md_dist'],
            low=params['md_low'],
            high=params['md_high'],
            a=params['a'], b=params['b'],
            loc=params['loc'], scale=params['scale']
        )

        # 'orthopair' mode: Calculate non-membership degree based on the constraint
        if params['nu_mode'] == 'orthopair':
            max_nmd = (1 - md ** q) ** (1 / q)
            effective_high = min(params['nu_high'], max_nmd)

            # Sample in [0,1] then scale
            nmd_sample = self._sample_from_distribution(
                rng, size=None, dist=params['nu_dist'], low=0.0, high=1.0,
                a=params['a'], b=params['b'], loc=params['loc'], scale=params['scale'])

            nmd = params['nu_low'] + nmd_sample * (effective_high - params['nu_low'])
            nmd = max(nmd, params['nu_low'])
        else:  # 'independent' mode
            nmd = self._sample_from_distribution(
                rng, size=None, dist=params['nu_dist'], low=params['nu_low'], high=params['nu_high'],
                a=params['a'], b=params['b'], loc=params['loc'], scale=params['scale']
            )
            if (md ** q + nmd ** q) > 1.0:
                nmd = (1 - md ** q) ** (1 / q)

        return Fuzznum(mtype='qrofn', q=q).create(md=md, nmd=nmd)

    def fuzzarray(self,
                  rng: np.random.Generator,
                  shape: Tuple[int, ...],
                  q: int = 1,
                  **params) -> 'Fuzzarray':
        """
        Generates a Fuzzarray of QROFNs using high-performance vectorized operations.
        """
        params = self._merge_parameters(**params)
        self.validate_parameters(q=q, **params)

        size = int(np.prod(shape))

        # 1. Generate membership degrees (mds) vectorially
        mds = self._sample_from_distribution(
            rng,
            size=size,
            dist=params['md_dist'],
            low=params['md_low'],
            high=params['md_high'],
            a=params['a'], b=params['b'],
            loc=params['loc'], scale=params['scale']
        )

        # 2. Generate non-membership degrees (nmds) vectorially based on nu_mode
        if params['nu_mode'] == 'orthopair':
            # Calculate max allowed nmd for each md to satisfy the constraint
            max_nmd = (1 - mds ** q) ** (1 / q)

            # Effective upper bound for sampling is the minimum of user's high and the calculated max
            effective_high = np.minimum(params['nu_high'], max_nmd)

            # Sample nmds within the constrained range [nu_low, effective_high]
            nmds = self._sample_from_distribution(
                rng,
                size=size,
                dist=params['nu_dist'],
                low=params['nu_low'],
                high=1.0,  # Sample in [0,1] then scale
                a=params['a'], b=params['b'],
                loc=params['loc'], scale=params['scale']
            )
            # Scale samples to the dynamic range [nu_low, effective_high]
            nmds = params['nu_low'] + nmds * (effective_high - params['nu_low'])
            # Ensure we don't exceed the lower bound if effective_high < nu_low
            nmds = np.maximum(nmds, params['nu_low'])

        else:  # 'independent' mode
            nmds = self._sample_from_distribution(
                rng,
                size=size,
                dist=params['nu_dist'],
                low=params['nu_low'],
                high=params['nu_high'],
                a=params['a'], b=params['b'],
                loc=params['loc'], scale=params['scale']
            )
            # Find elements that violate the constraint
            violates_mask = (mds ** q + nmds ** q) > 1.0
            if np.any(violates_mask):
                # For violating elements, clamp nmd to the maximum allowed value
                max_nmd_violating = (1 - mds[violates_mask] ** q) ** (1 / q)
                nmds[violates_mask] = np.minimum(nmds[violates_mask], max_nmd_violating)

        # 3. Reshape arrays to the target shape
        mds = mds.reshape(shape)
        nmds = nmds.reshape(shape)

        # 4. Create backend directly from arrays (High-Performance Path)
        backend = QROFNBackend.from_arrays(mds=mds, nmds=nmds, q=q)

        # 5. Return the final Fuzzarray
        return Fuzzarray(backend=backend)
