#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Standard membership function implementations for AxisFuzzy.

This module provides a comprehensive collection of commonly used membership
functions that form the building blocks of fuzzy sets and fuzzy logic systems.
Each function implements the :class:`MembershipFunction` interface and provides
mathematically sound, numerically stable, and computationally efficient
implementations.

The functions are designed to work seamlessly with the AxisFuzzy fuzzification
system and support both scalar and vectorized array operations for high-performance
batch processing.

Available Functions
-------------------
This module implements the following standard membership functions:

**Basic Shape Functions**:
    - :class:`TriangularMF`: Triangular-shaped membership function with three parameters (a, b, c)
    - :class:`TrapezoidalMF`: Trapezoidal-shaped function with four parameters (a, b, c, d)
    - :class:`GaussianMF`: Bell-shaped Gaussian function with center and spread parameters

**S-Curve and Z-Curve Functions**:
    - :class:`SigmoidMF`: Sigmoid (S-shaped) function with slope and center parameters
    - :class:`SMF`: S-shaped membership function with smooth transitions
    - :class:`ZMF`: Z-shaped membership function (inverse of S-shaped)
    - :class:`PiMF`: Pi-shaped function combining S and Z curves

**Advanced Functions**:
    - :class:`GeneralizedBellMF`: Generalized bell-shaped function with adjustable slope
    - :class:`DoubleGaussianMF`: Combination of two Gaussian functions

Design Principles
-----------------
All membership functions in this module follow consistent design principles:

**Mathematical Correctness**:
    - All functions guarantee output values in the range [0, 1]
    - Smooth transitions and mathematical continuity where appropriate
    - Proper handling of boundary conditions and edge cases

**Numerical Stability**:
    - Robust handling of extreme values and potential division by zero
    - Use of NumPy's error handling for floating-point operations
    - Clipping and sanitization of results to ensure valid outputs

**Performance Optimization**:
    - Vectorized operations using NumPy for efficient array processing
    - Minimal object creation and memory allocation
    - Optimized algorithms for common parameter configurations

**User Experience**:
    - Flexible parameter specification (positional or keyword arguments)
    - Comprehensive parameter validation with clear error messages
    - Consistent naming conventions following MATLAB Fuzzy Logic Toolbox

Parameter Conventions
---------------------
Most functions support dual parameter specification modes:

1. **Positional Arguments**: ``MembershipFunction(param1, param2, ...)``
   - Traditional mathematical notation
   - Compact specification for known parameter orders

2. **Keyword Arguments**: ``MembershipFunction(param1=value1, param2=value2)``
   - Self-documenting parameter specification
   - Partial parameter specification with defaults
   - Better for programmatic generation

**Parameter Validation**:
All functions validate their parameters during construction and when updated
via :meth:`set_parameters`. Common validation includes:
- Range checks (e.g., positive values for standard deviations)
- Ordering constraints (e.g., a ≤ b ≤ c for triangular functions)
- Mathematical feasibility (e.g., non-zero denominators)

Vectorization Support
---------------------
All functions are designed for efficient vectorized operation:

.. code-block:: python

    import numpy as np

    # Single value computation
    mf = TriangularMF(0, 0.5, 1)
    single_result = mf.compute(0.3)  # Returns scalar

    # Vectorized computation
    x_array = np.linspace(0, 1, 1000)
    array_result = mf.compute(x_array)  # Returns array of same shape

Performance Characteristics
---------------------------
The functions are optimized for different usage patterns:

- **TriangularMF, TrapezoidalMF**: Extremely fast, piecewise linear computation
- **GaussianMF, DoubleGaussianMF**: Moderate speed, smooth curves
- **SMF, ZMF, PiMF**: Moderate speed, complex curve shapes
- **GeneralizedBellMF**: Slower due to power operations, high flexibility
- **SigmoidMF**: Fast, good for neural network-style applications

Integration with AxisFuzzy
--------------------------
These functions integrate seamlessly with the broader AxisFuzzy ecosystem:

- **Factory System**: All functions are automatically discovered and available
  through :func:`axisfuzzy.membership.factory.create_mf`
- **Fuzzification**: Used by :mod:`axisfuzzy.fuzzify` for crisp-to-fuzzy conversion
- **Visualization**: Built-in plotting support for function analysis
- **Serialization**: Parameter dictionaries support easy save/load operations

Notes
-----
- All functions are thread-safe for read operations after construction
- Parameter updates via :meth:`set_parameters` are not guaranteed to be thread-safe
- Functions maintain internal parameter dictionaries for introspection
- Plotting requires matplotlib and is optional for core functionality

See Also
--------
axisfuzzy.membership.base : Base class defining the membership function interface
axisfuzzy.membership.factory : Factory functions for creating instances
axisfuzzy.fuzzify : Fuzzification system using these functions

Examples
--------
Basic usage of different membership functions:

.. code-block:: python

    import numpy as np
    from axisfuzzy.membership.function import TriangularMF, GaussianMF, SigmoidMF

    # Create different function types
    tri = TriangularMF(a=0, b=0.5, c=1)
    gauss = GaussianMF(sigma=0.2, c=0.5)
    sigmoid = SigmoidMF(k=5, c=0.5)

    # Evaluate at single points
    x = 0.3
    tri_val = tri.compute(x)     # ≈ 0.6
    gauss_val = gauss.compute(x) # ≈ 0.135
    sig_val = sigmoid.compute(x) # ≈ 0.119

    # Vectorized evaluation
    x_array = np.linspace(0, 1, 100)
    tri_array = tri.compute(x_array)
    gauss_array = gauss.compute(x_array)
    sig_array = sigmoid.compute(x_array)

Parameter flexibility and validation:

.. code-block:: python

    # Multiple ways to create triangular function
    tri1 = TriangularMF(0, 0.5, 1)              # Positional
    tri2 = TriangularMF(a=0, b=0.5, c=1)        # Keyword
    tri3 = TriangularMF(a=0, c=1, b=0.5)        # Mixed order

    # Parameter validation
    try:
        invalid = TriangularMF(1, 0.5, 0)  # Invalid order
    except ValueError as e:
        print(f"Validation error: {e}")

    # Dynamic parameter updates
    tri = TriangularMF(0, 0.5, 1)
    tri.set_parameters(b=0.7)  # Update peak position
    print(tri.get_parameters()) # {'a': 0, 'b': 0.7, 'c': 1}

Advanced function combinations:

.. code-block:: python

    # Create complex membership functions
    double_gauss = DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.15, c2=0.7)
    pi_function = PiMF(a=0.2, b=0.4, c=0.6, d=0.8)

    # Evaluate and compare
    x = np.linspace(0, 1, 200)
    dg_values = double_gauss.compute(x)
    pi_values = pi_function.compute(x)

    # Plot for comparison
    import matplotlib.pyplot as plt
    plt.plot(x, dg_values, label='Double Gaussian')
    plt.plot(x, pi_values, label='Pi Function')
    plt.legend()
    plt.show()

Integration with fuzzification:

.. code-block:: python

    from axisfuzzy.membership.factory import create_mf
    from axisfuzzy.fuzzify import fuzzify

    # Create membership function via factory
    mf, _ = create_mf('trimf', a=0, b=0.5, c=1)

    # Use in fuzzification
    crisp_values = [0.2, 0.5, 0.8]
    fuzzy_result = fuzzify(crisp_values, membership_function=mf, mtype='qrofn')

References
----------
- Zadeh, L.A. (1965). "Fuzzy sets". Information and Control, 8(3), 338-353.
- MATLAB Fuzzy Logic Toolbox documentation for function definitions
"""

import numpy as np
from .base import MembershipFunction


class SigmoidMF(MembershipFunction):
    """
    Sigmoid membership function.

    .. math::

        f(x) = \\frac{1}{1 + \\exp\\bigl(-k\\,(x - c)\\bigr)}

    A smooth S-shaped function commonly used in neural networks and fuzzy logic.
    The function transitions from 0 to 1 with adjustable steepness and center point.

    Parameters
    ----------
    k : float
        Slope (steepness) of the sigmoid function. Positive values create
        ascending curves, negative values create descending curves.
    c : float
        Center (midpoint) of the sigmoid function where output equals 0.5.

    Examples
    --------
    .. code-block:: python

        # Create ascending sigmoid
        mf = SigmoidMF(k=2.0, c=0.5)
        result = mf.compute([0, 0.5, 1])  # [0.12, 0.5, 0.88]

        # Create descending sigmoid
        mf = SigmoidMF(k=-2.0, c=0.5)
        result = mf.compute([0, 0.5, 1])  # [0.88, 0.5, 0.12]
    """

    def __init__(self, *params, k: float = None, c: float = None):
        super().__init__()

        # Support *params calling convention
        if params:
            if len(params) != 2:
                raise ValueError("SigmoidMF requires exactly two parameters: k, c")
            k, c = params

        # If no *params provided, must use keyword arguments
        if k is None or c is None:
            # Set default values (maintain backward compatibility)
            k = 1.0 if k is None else k
            c = 0.0 if c is None else c

        self.k, self.c = k, c
        self.parameters = {"k": k, "c": c}

    def compute(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid membership values."""
        return 1 / (1 + np.exp(-self.k * (x - self.c)))

    def set_parameters(self, **kwargs):
        if 'k' in kwargs:
            self.k = kwargs['k']
            self.parameters['k'] = self.k
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class TriangularMF(MembershipFunction):
    """
    Triangular membership function.

    .. math::

        f(x) = \\max\\!\\left(0,\\; \\min\\!\\left(\\frac{x - a}{\\,b - a\\,},\\; \\frac{c - x}{\\,c - b\,}\\right)\\right)

    A piecewise linear function forming a triangle shape. It rises linearly from
    0 to 1 and then falls linearly back to 0. This is one of the most commonly
    used membership functions due to its simplicity and computational efficiency.

    Parameters
    ----------
    a : float
        Left foot of the triangle (where function starts rising from 0).
    b : float
        Peak of the triangle (where function equals 1).
    c : float
        Right foot of the triangle (where function falls back to 0).

    Constraints: a ≤ b ≤ c


    Examples
    --------
    .. code-block:: python

        # Standard triangular function
        mf = TriangularMF(a=0, b=0.5, c=1)
        result = mf.compute([0, 0.25, 0.5, 0.75, 1])  # [0, 0.5, 1, 0.5, 0]

        # Asymmetric triangle
        mf = TriangularMF(a=0, b=0.2, c=1)  # Peak closer to left
    """

    def __init__(self, *params, a: float = None, b: float = None, c: float = None):
        """
        Initialize TriangularMF.

        Args:
            *params: (a, b, c) required if no keyword args given
            a (float, optional): left foot
            b (float, optional): peak
            c (float, optional): right foot
        """
        super().__init__()

        # Support *params calling convention
        if params:
            if len(params) != 3:
                raise ValueError("TriangularMF requires exactly three parameters: a, b, c")
            a, b, c = params

        a = a if a is not None else 0.0
        b = b if b is not None else 0.5
        c = c if c is not None else 1.0

        if not (a <= b <= c):
            raise ValueError("TriangularMF requires parameters to satisfy a <= b <= c")

        self.a, self.b, self.c = a, b, c
        self.parameters = {"a": a, "b": b, "c": c}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # Left ascending segment (only when b > a)
        if self.b > self.a:
            mask1 = (x >= self.a) & (x < self.b)
            result[mask1] = (x[mask1] - self.a) / (self.b - self.a)

        # Peak point
        result[x == self.b] = 1.0

        # Right descending segment (only when c > b)
        if self.c > self.b:
            mask2 = (x > self.b) & (x <= self.c)
            result[mask2] = (self.c - x[mask2]) / (self.c - self.b)

        # Clip to [0,1]
        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c
        # Re-validate parameter order
        if not (self.a <= self.b <= self.c):
            raise ValueError("TriangularMF requires parameters to satisfy a <= b <= c")


class TrapezoidalMF(MembershipFunction):
    """
    Trapezoidal membership function.

    .. math::

        f(x)=\\max\\!\\left(0,\\; \\min\\!\\left(\\dfrac{x - a}{\\,b - a\\,},\\; 1,\\; \\dfrac{d - x}{\\,d - c\\,}\\right)\\right)

    A piecewise linear function forming a trapezoid shape. It has a flat top
    region between two linear transition regions. This function is useful when
    there's a range of values that should have maximum membership.

    Parameters
    ----------
    a : float
        Left foot (start of ascending region).
    b : float
        Left shoulder (start of flat top).
    c : float
        Right shoulder (end of flat top).
    d : float
        Right foot (end of descending region).

    Constraints: a ≤ b ≤ c ≤ d

    Examples
    --------
    .. code-block:: python

        # Standard trapezoidal function
        mf = TrapezoidalMF(a=0, b=0.2, c=0.8, d=1)
        result = mf.compute([0, 0.1, 0.5, 0.9, 1])  # [0, 0.5, 1, 0.5, 0]

        # Degenerate to triangle when b=c
        mf = TrapezoidalMF(a=0, b=0.5, c=0.5, d=1)  # Acts like triangle
    """

    def __init__(self, *params, a: float = None, b: float = None,
                 c: float = None, d: float = None):
        super().__init__()

        if params:
            if len(params) != 4:
                raise ValueError("TrapezoidalMF requires exactly four parameters: a, b, c, d")
            a, b, c, d = params

        a = a if a is not None else 0.0
        b = b if b is not None else 0.25
        c = c if c is not None else 0.75
        d = d if d is not None else 1.0

        if not (a <= b <= c <= d):
            raise ValueError("TrapezoidalMF requires parameters to satisfy a <= b <= c <= d")

        self.a, self.b, self.c, self.d = a, b, c, d
        self.parameters = {"a": a, "b": b, "c": c, "d": d}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # Left ascending (only when b > a)
        if self.b > self.a:
            mask_rise = (x > self.a) & (x < self.b)
            result[mask_rise] = (x[mask_rise] - self.a) / (self.b - self.a)

        # Plateau region [b, c]
        mask_plateau = (x >= self.b) & (x <= self.c)
        result[mask_plateau] = 1.0

        # Right descending (only when d > c)
        if self.d > self.c:
            mask_fall = (x > self.c) & (x < self.d)
            result[mask_fall] = (self.d - x[mask_fall]) / (self.d - self.c)

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        for param in ['a', 'b', 'c', 'd']:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]
        # Re-validate parameter order
        if not (self.a <= self.b <= self.c <= self.d):
            raise ValueError("TrapezoidalMF requires parameters to satisfy a <= b <= c <= d")


class GaussianMF(MembershipFunction):
    """
    Gaussian membership function.

    .. math::

        f(x) = \\exp(-\\frac{(x-c)^2}{2\\sigma^2})

    A bell-shaped curve based on the normal distribution. It provides smooth
    transitions and is commonly used when gradual membership changes are desired.
    The function is symmetric around its center point.

    Parameters
    ----------
    sigma : float
        Standard deviation controlling the width of the bell curve.
        Must be positive. Smaller values create narrower curves.
    c : float
        Center of the bell curve (mean of the distribution).

    Examples
    --------
    .. code-block:: python

        # Standard Gaussian
        mf = GaussianMF(sigma=0.2, c=0.5)
        result = mf.compute([0.3, 0.5, 0.7])  # Peak at c=0.5

        # Wide Gaussian
        mf = GaussianMF(sigma=0.5, c=0.5)  # Broader curve

        # Narrow Gaussian
        mf = GaussianMF(sigma=0.1, c=0.5)  # Sharper peak
    """

    def __init__(self, *params, sigma: float = None, c: float = None):
        super().__init__()

        if params:
            if len(params) != 2:
                raise ValueError("GaussianMF requires exactly two parameters: sigma, c")
            sigma, c = params

        sigma = sigma if sigma is not None else 1.0
        c = c if c is not None else 0.5

        if sigma <= 0:
            raise ValueError("GaussianMF parameter 'sigma' must be positive")

        self.sigma, self.c = sigma, c
        self.parameters = {"sigma": sigma, "c": c}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.c) / self.sigma) ** 2)

    def set_parameters(self, **kwargs):
        if 'sigma' in kwargs:
            if kwargs['sigma'] <= 0:
                raise ValueError("GaussianMF parameter 'sigma' must be positive")
            self.sigma = kwargs['sigma']
            self.parameters['sigma'] = self.sigma
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class SMF(MembershipFunction):
    """
    S-shaped membership function.

    .. math::
       :label: s-shaped-function

       f(x) =
       \\begin{cases}
         0, & x \\le a,\\\[6pt]
         2\\left(\\dfrac{x-a}{b-a}\\right)^2, & a < x < \\dfrac{a+b}{2},\\\[6pt]
         1 - 2\\left(\\dfrac{x-b}{b-a}\\right)^2, & \\dfrac{a+b}{2} \\le x < b,\\\[6pt]
         1, & x \\ge b.
       \\end{cases}

    A smooth S-curve that transitions from 0 to 1. The function has an inflection
    point at the midpoint between parameters a and b. It's useful for representing
    gradual increase in membership with smooth acceleration and deceleration.

    Parameters
    ----------
    a : float
        Lower bound where function starts transitioning from 0.
    b : float
        Upper bound where function reaches 1.

    Constraints: a < b

    Examples
    --------
    .. code-block:: python

        # Standard S-curve
        mf = SMF(a=0, b=1)
        result = mf.compute([0, 0.25, 0.5, 0.75, 1])  # [0, 0.125, 0.5, 0.875, 1]

        # Shifted S-curve
        mf = SMF(a=0.2, b=0.8)  # Transition between 0.2 and 0.8
    """

    def __init__(self, *params, a: float = None, b: float = None):
        super().__init__()

        if params:
            if len(params) != 2:
                raise ValueError("SMF requires exactly two parameters: a, b")
            a, b = params

        a = a if a is not None else 0.0
        b = b if b is not None else 1.0

        if a >= b:
            raise ValueError("SMF requires parameter a < b")

        self.a, self.b = a, b
        self.parameters = {"a": a, "b": b}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # x <= a: y = 0
        result[x <= self.a] = 0.0

        # x >= b: y = 1
        result[x >= self.b] = 1.0

        # a < x < b: S-curve
        mid = (self.a + self.b) / 2
        mask_first = (x > self.a) & (x <= mid)
        mask_second = (x > mid) & (x < self.b)

        # First segment: 2 * ((x - a) / (b - a))^2
        if np.any(mask_first):
            result[mask_first] = 2 * ((x[mask_first] - self.a) / (self.b - self.a)) ** 2

        # Second segment: 1 - 2 * ((x - b) / (b - a))^2
        if np.any(mask_second):
            result[mask_second] = 1 - 2 * ((x[mask_second] - self.b) / (self.b - self.a)) ** 2

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if self.a >= self.b:
            raise ValueError("SMF requires parameter a < b")


class ZMF(MembershipFunction):
    """
    Z-shaped membership function.

    .. math::
        :label: s-shaped-function

        f(x) =
        \\begin{cases}
         1, & x \\le a,\\\[6pt]
         1 - 2\\left(\\dfrac{x - a}{b - a}\\right)^2, & a < x < \\dfrac{a + b}{2},\\\[6pt]
         2\\left(\\dfrac{x - b}{b - a}\\right)^2, & \\dfrac{a + b}{2} \\le x < b,\\\[6pt]
         0, & x \\ge b.
        \\end{cases}

    A smooth Z-curve that transitions from 1 to 0. It's the inverse of the S-curve
    and is useful for representing gradual decrease in membership. The function
    has an inflection point at the midpoint between parameters a and b.

    Parameters
    ----------
    a : float
        Lower bound where function starts transitioning from 1.
    b : float
        Upper bound where function reaches 0.

    Constraints: a < b

    Examples
    --------
    .. code-block:: python

        # Standard Z-curve
        mf = ZMF(a=0, b=1)
        result = mf.compute([0, 0.25, 0.5, 0.75, 1])  # [1, 0.875, 0.5, 0.125, 0]

        # Shifted Z-curve
        mf = ZMF(a=0.3, b=0.7)  # Transition between 0.3 and 0.7
    """

    def __init__(self, *params, a: float = None, b: float = None):
        super().__init__()

        if params:
            if len(params) != 2:
                raise ValueError("ZMF requires exactly two parameters: a, b")
            a, b = params

        a = a if a is not None else 0.0
        b = b if b is not None else 1.0

        if a >= b:
            raise ValueError("ZMF requires parameter a < b")

        self.a, self.b = a, b
        self.parameters = {"a": a, "b": b}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.ones_like(x, dtype=float)

        # x <= a: y = 1
        result[x <= self.a] = 1.0

        # x >= b: y = 0
        result[x >= self.b] = 0.0

        # a < x < b: Z-curve
        mid = (self.a + self.b) / 2
        mask_first = (x > self.a) & (x <= mid)
        mask_second = (x > mid) & (x < self.b)

        # First segment: 1 - 2 * ((x - a) / (b - a))^2
        if np.any(mask_first):
            result[mask_first] = 1 - 2 * ((x[mask_first] - self.a) / (self.b - self.a)) ** 2

        # Second segment: 2 * ((x - b) / (b - a))^2
        if np.any(mask_second):
            result[mask_second] = 2 * ((x[mask_second] - self.b) / (self.b - self.a)) ** 2

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if self.a >= self.b:
            raise ValueError("ZMF requires parameter a < b")


class DoubleGaussianMF(MembershipFunction):
    """
    Double Gaussian membership function.

    .. math::

        f(x) = \\max(\\exp(-\\frac{(x - c_1)^2}{2\\sigma_1^2}),
                     \\exp(-\\frac{(x - c_2)^2}{2\\sigma_2^2}))

    A combination of two Gaussian curves that takes the maximum value at each point.
    This creates a function with two peaks or a broader, flatter top than a single
    Gaussian. Useful for representing bimodal distributions or wide acceptance regions.

    Parameters
    ----------
    sigma1, sigma2 : float
        Standard deviations of the two Gaussian curves. Must be positive.
    c1, c2 : float
        Centers of the two Gaussian curves.

    Examples
    --------
    .. code-block:: python

        # Two separate peaks
        mf = DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.1, c2=0.7)

        # Overlapping peaks (broader curve)
        mf = DoubleGaussianMF(sigma1=0.2, c1=0.4, sigma2=0.2, c2=0.6)

        # Asymmetric double curve
        mf = DoubleGaussianMF(sigma1=0.1, c1=0.2, sigma2=0.3, c2=0.8)
    """

    def __init__(self, *params, sigma1: float = None, c1: float = None,
                 sigma2: float = None, c2: float = None):
        super().__init__()

        if params:
            if len(params) != 4:
                raise ValueError("DoubleGaussianMF requires exactly four parameters: sigma1, c1, sigma2, c2")
            sigma1, c1, sigma2, c2 = params

        sigma1 = sigma1 if sigma1 is not None else 1.0
        c1 = c1 if c1 is not None else 0.25
        sigma2 = sigma2 if sigma2 is not None else 1.0
        c2 = c2 if c2 is not None else 0.75

        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("DoubleGaussianMF parameters 'sigma1' and 'sigma2' must be positive")

        self.sigma1, self.c1, self.sigma2, self.c2 = sigma1, c1, sigma2, c2
        self.parameters = {"sigma1": sigma1, "c1": c1, "sigma2": sigma2, "c2": c2}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        gauss1 = np.exp(-0.5 * ((x - self.c1) / self.sigma1) ** 2)
        gauss2 = np.exp(-0.5 * ((x - self.c2) / self.sigma2) ** 2)
        return np.maximum(gauss1, gauss2)

    def set_parameters(self, **kwargs):
        for param in ['sigma1', 'c1', 'sigma2', 'c2']:
            if param in kwargs:
                if param in ('sigma1', 'sigma2') and kwargs[param] <= 0:
                    raise ValueError("DoubleGaussianMF parameters 'sigma1' and 'sigma2' must be positive")
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]


class GeneralizedBellMF(MembershipFunction):
    """
    Generalized Bell membership function.

    .. math::

        f(x) = \\frac{1}{1 + \\left|\\dfrac{x - c}{a}\\right|^{2 b}}

    A bell-shaped curve with adjustable steepness and width. It provides more
    flexibility than the Gaussian function by allowing independent control of
    the curve's width and steepness around the center point.

    Parameters
    ----------
    a : float
        Width parameter controlling the curve width. Must be positive.
        Larger values create wider curves.
    b : float
        Slope parameter controlling steepness at the crossover points.
        Must be positive. Larger values create steeper transitions.
    c : float
        Center of the bell curve.

    Examples
    --------
    .. code-block:: python

        # Standard generalized bell
        mf = GeneralizedBellMF(a=0.2, b=2, c=0.5)

        # Wide, gentle curve
        mf = GeneralizedBellMF(a=0.5, b=1, c=0.5)

        # Narrow, steep curve
        mf = GeneralizedBellMF(a=0.1, b=5, c=0.5)
    """

    def __init__(self, *params, a: float = None, b: float = None, c: float = None):
        super().__init__()

        if params:
            if len(params) != 3:
                raise ValueError("GeneralizedBellMF requires exactly three parameters: a, b, c")
            a, b, c = params

        a = a if a is not None else 1.0
        b = b if b is not None else 2.0
        c = c if c is not None else 0.0

        if a <= 0:
            raise ValueError("GeneralizedBellMF parameter 'a' must be positive")
        if b <= 0:
            raise ValueError("GeneralizedBellMF parameter 'b' must be positive")

        self.a, self.b, self.c = a, b, c
        self.parameters = {"a": a, "b": b, "c": c}

    def compute(self, x):
        x = np.asarray(x)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            result = 1 / (1 + np.abs((x - self.c) / self.a) ** (2 * self.b))

        # Handle possible infinity or NaN values
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

        return result

    def set_parameters(self, **kwargs):
        if 'a' in kwargs:
            if kwargs['a'] <= 0:
                raise ValueError("GeneralizedBellMF parameter 'a' must be positive")
            self.a = kwargs['a']
            self.parameters['a'] = self.a
        if 'b' in kwargs:
            if kwargs['b'] <= 0:
                raise ValueError("GeneralizedBellMF parameter 'b' must be positive")
            self.b = kwargs['b']
            self.parameters['b'] = self.b
        if 'c' in kwargs:
            self.c = kwargs['c']
            self.parameters['c'] = self.c


class PiMF(MembershipFunction):
    """
    Pi-shaped membership function.

    .. math::

        f(x) =
       \\begin{cases}
         0, & x \\le a,\\\[6pt]
         2\\left(\\dfrac{x-a}{b-a}\\right)^2, & a < x < \\dfrac{a+b}{2},\\\[6pt]
         1 - 2\\left(\\dfrac{x-b}{b-a}\\right)^2, & \\dfrac{a+b}{2} \\le x < b,\\\[6pt]
         1, & b \\le x \\le c,\\\[6pt]
         1 - 2\\left(\\dfrac{x-c}{d-c}\\right)^2, & c < x < \\dfrac{c+d}{2},\\\[6pt]
         2\\left(\\dfrac{x-d}{d-c}\\right)^2, & \\dfrac{c+d}{2} \\le x < d,\\\[6pt]
         0, & x \\ge d.
       \end{cases}

    A combination of S-shaped and Z-shaped curves creating a bell-like function
    with smooth transitions. It rises smoothly, maintains a flat top, then falls
    smoothly. This function is useful when you need smooth transitions with a
    plateau region of maximum membership.

    Parameters
    ----------
    a : float
        Left foot where function starts rising.
    b : float
        Left shoulder where function reaches maximum.
    c : float
        Right shoulder where function starts falling.
    d : float
        Right foot where function reaches zero.

    Constraints: a ≤ b ≤ c ≤ d

    Examples
    --------
    .. code-block:: python

        # Standard Pi function
        mf = PiMF(a=0, b=0.2, c=0.8, d=1)

        # Narrow plateau
        mf = PiMF(a=0, b=0.4, c=0.6, d=1)

        # Wide plateau
        mf = PiMF(a=0, b=0.1, c=0.9, d=1)
    """

    def __init__(self, *params, a: float = None, b: float = None,
                 c: float = None, d: float = None):
        super().__init__()

        if params:
            if len(params) != 4:
                raise ValueError("PiMF requires exactly four parameters: a, b, c, d")
            a, b, c, d = params

        a = a if a is not None else 0.0
        b = b if b is not None else 0.25
        c = c if c is not None else 0.75
        d = d if d is not None else 1.0

        if not (a <= b <= c <= d):
            raise ValueError("PiMF requires parameters to satisfy a <= b <= c <= d")

        self.a, self.b, self.c, self.d = a, b, c, d
        self.parameters = {"a": a, "b": b, "c": c, "d": d}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        # x <= a or x >= d: y = 0
        result[(x <= self.a) | (x >= self.d)] = 0.0

        # b <= x <= c: y = 1 (plateau region)
        result[(x >= self.b) & (x <= self.c)] = 1.0

        # a < x < b: S-curve ascending
        if self.b > self.a:
            mask_rise = (x > self.a) & (x < self.b)
            if np.any(mask_rise):
                # Use SMF logic
                smf_result = SMF(self.a, self.b).compute(x[mask_rise])
                result[mask_rise] = smf_result

        # c < x < d: Z-curve descending
        if self.d > self.c:
            mask_fall = (x > self.c) & (x < self.d)
            if np.any(mask_fall):
                # Use ZMF logic
                zmf_result = ZMF(self.c, self.d).compute(x[mask_fall])
                result[mask_fall] = zmf_result

        return np.clip(result, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        for param in ['a', 'b', 'c', 'd']:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.parameters[param] = kwargs[param]
        if not (self.a <= self.b <= self.c <= self.d):
            raise ValueError("PiMF requires parameters to satisfy a <= b <= c <= d")
