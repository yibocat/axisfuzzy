#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Abstract base class for membership functions in AxisFuzzy.

This module defines the core interface and behavior that all membership functions
must implement. Membership functions are mathematical functions that map crisp
values to membership degrees in the range [0, 1], forming the foundation of
fuzzy set theory and the AxisFuzzy fuzzification system.

The base class provides:
- Abstract interface specification for all membership function implementations
- Parameter management and introspection capabilities
- Function object behavior through ``__call__`` method
- Built-in visualization support for function plotting
- Standardized metadata handling

Architecture
------------
The membership function system follows a template method pattern where:

1. **Interface Definition**: :class:`MembershipFunction` defines the required
   methods that all concrete implementations must provide.

2. **Parameter Management**: Built-in support for storing, retrieving, and
   updating function parameters with automatic metadata synchronization.

3. **Callable Interface**: All membership functions can be invoked directly
   as callable objects, providing a natural mathematical syntax.

4. **Visualization Support**: Optional plotting capabilities for visual
   inspection and validation of membership function shapes.

Core Methods
------------
All membership function implementations must provide:

- ``compute(x)``: The primary computation method that calculates membership
  degrees for input values. Accepts both scalar and array inputs.
- ``set_parameters(**kwargs)``: Method for updating function parameters
  after instantiation with validation and metadata synchronization.

Additional inherited capabilities include:

- ``get_parameters()``: Returns current parameter dictionary for introspection
- ``__call__(x)``: Enables direct function invocation syntax
- ``plot(x_range, num_points)``: Generates matplotlib visualizations

Usage Patterns
---------------
Membership functions are typically used in three contexts:

1. **Fuzzification**: Converting crisp values to fuzzy membership degrees
   during the construction of :class:`Fuzznum` objects.

2. **Rule Systems**: Defining linguistic variables and fuzzy rules in
   expert systems and control applications.

3. **Data Analysis**: Analyzing uncertainty and partial membership in
   datasets with imprecise or subjective classifications.

Design Principles
-----------------
The membership function design follows several key principles:

- **Mathematical Correctness**: All functions guarantee output in [0, 1]
- **Numerical Stability**: Robust handling of edge cases and numerical errors
- **Performance**: Efficient vectorized computation using NumPy arrays
- **Extensibility**: Easy to subclass for custom membership functions
- **Consistency**: Uniform parameter naming and behavior across all implementations

Notes
-----
- All membership functions are stateful objects that store their parameters
- Thread safety depends on the specific implementation but is not guaranteed by the base class
- Parameter validation is delegated to concrete implementations
- Visualization requires matplotlib and is optional for core functionality

See Also
--------
axisfuzzy.membership.function : Concrete implementations of standard membership functions
axisfuzzy.membership.factory : Factory functions for creating membership function instances
axisfuzzy.fuzzify : Fuzzification system that uses membership functions

Examples
--------
Creating and using a custom membership function:

.. code-block:: python

    import numpy as np
    from axisfuzzy.membership.base import MembershipFunction

    class LinearMF(MembershipFunction):
        def __init__(self, slope=1.0, intercept=0.0):
            super().__init__()
            self.slope = slope
            self.intercept = intercept
            self.parameters = {'slope': slope, 'intercept': intercept}

        def compute(self, x):
            result = self.slope * x + self.intercept
            return np.clip(result, 0.0, 1.0)  # Ensure [0,1] range

        def set_parameters(self, **kwargs):
            if 'slope' in kwargs:
                self.slope = kwargs['slope']
                self.parameters['slope'] = self.slope
            if 'intercept' in kwargs:
                self.intercept = kwargs['intercept']
                self.parameters['intercept'] = self.intercept

    # Usage
    mf = LinearMF(slope=0.5, intercept=0.1)
    x = np.array([0, 0.5, 1.0, 2.0])
    membership = mf(x)  # Calls compute() via __call__
    print(membership)   # [0.1, 0.35, 0.6, 1.0]

Batch processing with arrays:

.. code-block:: python

    # Process multiple values efficiently
    x_values = np.linspace(0, 10, 100)
    membership_degrees = mf.compute(x_values)

    # Visualize the function
    mf.plot(x_range=(0, 10), num_points=200)

Parameter management:

.. code-block:: python

    # Inspect current parameters
    params = mf.get_parameters()
    print(params)  # {'slope': 0.5, 'intercept': 0.1}

    # Update parameters dynamically
    mf.set_parameters(slope=1.0, intercept=0.0)
    new_params = mf.get_parameters()
    print(new_params)  # {'slope': 1.0, 'intercept': 0.0}

References
----------
- Zadeh, L.A. (1965). "Fuzzy sets". Information and Control, 8(3), 338-353.
- Klir, G.J. & Yuan, B. (1995). "Fuzzy Sets and Fuzzy Logic: Theory and Applications"
- Ross, T.J. (2010). "Fuzzy Logic with Engineering Applications"
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np


class MembershipFunction(ABC):
    """
    Abstract base class for all membership functions in AxisFuzzy.

    This class defines the interface and common behavior that all membership
    function implementations must provide. It serves as the foundation for
    the AxisFuzzy fuzzification system and fuzzy set operations.

    Membership functions map input values to membership degrees in the range [0, 1],
    providing the mathematical foundation for fuzzy logic operations and fuzzy
    set theory applications.

    Attributes
    ----------
    name : str
        Human-readable name of the membership function, typically the class name.
    parameters : dict
        Dictionary storing the current parameter values of the function.
        Automatically populated by concrete implementations.

    Notes
    -----
    All concrete implementations must override both :meth:`compute` and
    :meth:`set_parameters` methods. The base class provides parameter storage,
    callable interface, and optional visualization capabilities.

    Thread safety is not guaranteed and depends on the specific implementation.
    Most implementations are read-only after construction but may support
    parameter updates through :meth:`set_parameters`.

    See Also
    --------
    axisfuzzy.membership.function : Standard membership function implementations
    axisfuzzy.membership.factory : Factory functions for creating instances

    Examples
    --------
    Basic usage pattern for subclassing:

    .. code-block:: python

        class CustomMF(MembershipFunction):
            def __init__(self, param1, param2=1.0):
                super().__init__()
                self.param1 = param1
                self.param2 = param2
                self.parameters = {'param1': param1, 'param2': param2}

            def compute(self, x):
                # Implementation specific logic
                return np.clip(some_function(x, self.param1, self.param2), 0, 1)

            def set_parameters(self, **kwargs):
                if 'param1' in kwargs:
                    self.param1 = kwargs['param1']
                    self.parameters['param1'] = self.param1
                # ... handle other parameters
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the base membership function.

        Sets up the basic attributes that all membership functions share,
        including the function name and parameter storage dictionary.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to concrete implementations. The base class
            ignores all arguments but concrete classes may use them for
            parameter initialization.

        Notes
        -----
        Concrete implementations should call ``super().__init__()`` before
        setting their specific parameters and updating the ``parameters`` dict.
        """
        self.name = self.__class__.__name__
        self.parameters = {}

    @abstractmethod
    def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute membership degrees for input values.

        This is the core method that all membership functions must implement.
        It transforms input values into membership degrees in the range [0, 1].

        Parameters
        ----------
        x : float or numpy.ndarray
            Input value(s) for which to compute membership degrees.
            Can be a scalar or array of any shape.

        Returns
        -------
        float or numpy.ndarray
            Membership degree(s) corresponding to the input values.
            Output has the same shape as input and values are in [0, 1].

        Notes
        -----
        Implementations must ensure that:
        - All output values are in the range [0, 1]
        - The function handles both scalar and array inputs correctly
        - Numerical stability is maintained for edge cases
        - The function is vectorized for efficient array processing

        Examples
        --------
        Implementation example for a simple linear membership function:

        .. code-block:: python

            def compute(self, x):
                x = np.asarray(x, dtype=float)
                result = (x - self.min_val) / (self.max_val - self.min_val)
                return np.clip(result, 0.0, 1.0)
        """
        pass

    def get_parameters(self) -> dict:
        """
        Retrieve the current parameters of the membership function.

        Returns a dictionary containing all parameters that define the
        shape and behavior of the membership function. This is useful
        for introspection, serialization, and debugging.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values.
            The exact keys depend on the specific membership function type.

        Examples
        --------

        .. code-block:: python

            # For a triangular membership function
            mf = TriangularMF(a=0, b=0.5, c=1)
            params = mf.get_parameters()
            print(params)  # {'a': 0, 'b': 0.5, 'c': 1}

            # For a Gaussian membership function
            mf = GaussianMF(sigma=1.0, c=0.0)
            params = mf.get_parameters()
            print(params)  # {'sigma': 1.0, 'c': 0.0}

        """
        return self.parameters

    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """
        Update the parameters of the membership function.

        This method allows dynamic modification of function parameters after
        instantiation. Implementations must validate parameters and update
        both the internal state and the ``parameters`` dictionary.

        Parameters
        ----------
        **kwargs
            Parameter names and values to update. Only parameters recognized
            by the specific membership function are processed.

        Raises
        ------
        ValueError
            If invalid parameter values are provided or if parameter
            combinations violate mathematical constraints.

        Notes
        -----
        Implementations should:
        - Validate all parameter values before updating
        - Maintain mathematical constraints (e.g., ordering requirements)
        - Update both internal attributes and the ``parameters`` dict
        - Provide clear error messages for invalid parameters

        Examples
        --------
        
        .. code-block:: python

            # Update triangular function parameters
            mf = TriangularMF(a=0, b=0.5, c=1)
            mf.set_parameters(b=0.6, c=1.2)

            # Update Gaussian function parameters
            mf = GaussianMF(sigma=1.0, c=0.0)
            mf.set_parameters(sigma=1.5)  # Only update sigma

            # Invalid parameter raises ValueError
            try:
                mf.set_parameters(sigma=-1.0)  # Negative sigma
            except ValueError as e:
                print(f"Error: {e}")

        """
        pass

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Enable the membership function to be called as a function object.

        This method provides a convenient interface for computing membership
        degrees using standard mathematical function call syntax. It delegates
        to the :meth:`compute` method for the actual calculation.

        Parameters
        ----------
        x : float or numpy.ndarray
            Input value(s) for membership degree computation.

        Returns
        -------
        float or numpy.ndarray
            Membership degree(s) in the range [0, 1].

        Examples
        --------
        .. code-block:: python

            mf = TriangularMF(a=0, b=0.5, c=1)

            # Direct function call syntax
            result = mf(0.3)        # Single value
            results = mf([0.1, 0.5, 0.8])  # Multiple values

            # Equivalent to calling compute() directly
            result = mf.compute(0.3)
            results = mf.compute([0.1, 0.5, 0.8])
        """
        return self.compute(x)

    def plot(self, x_range: Tuple[float, float] = (0, 1), num_points: int = 1000):
        """
        Generate a matplotlib plot of the membership function.

        Creates a visual representation of the membership function over
        the specified input range. This is useful for function validation,
        parameter tuning, and educational purposes.

        Parameters
        ----------
        x_range : tuple of float, default (0, 1)
            The (min, max) range of input values to plot.
        num_points : int, default 1000
            Number of points to use for plotting. Higher values create
            smoother curves but require more computation.

        Notes
        -----
        This method requires matplotlib to be installed. If matplotlib
        is not available, the method will raise an ImportError.

        The plot shows:
        - X-axis: Input values over the specified range
        - Y-axis: Membership degrees from 0 to 1
        - Title: Function name and type
        - Grid: Enabled for easier reading

        Examples
        --------
        .. code-block:: python

            # Plot with default settings
            mf = TriangularMF(a=0, b=0.5, c=1)
            mf.plot()

            # Plot over custom range with high resolution
            mf.plot(x_range=(-2, 3), num_points=2000)

            # Plot multiple functions for comparison
            mf1 = TriangularMF(a=0, b=0.3, c=0.6)
            mf2 = TriangularMF(a=0.4, b=0.7, c=1.0)

            import matplotlib.pyplot as plt
            mf1.plot()
            mf2.plot()
            plt.legend(['Function 1', 'Function 2'])
            plt.show()

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        ValueError
            If ``x_range`` is invalid or ``num_points`` is not positive.
        """
        import matplotlib.pyplot as plt
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = self.compute(x)
        plt.plot(x, y, label=self.name)
        plt.xlabel('x')
        plt.ylabel('Membership Degree')
        plt.title(f'{self.name} Membership Function')
        plt.grid(True)
        # 只有当 name 存在且不以下划线开头时才显示图例
        if hasattr(self, 'name') and self.name and not self.name.startswith('_'):
            plt.legend()
        # plt.show()
