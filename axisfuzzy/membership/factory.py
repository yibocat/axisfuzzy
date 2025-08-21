#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Factory functions for creating membership function instances in AxisFuzzy.

This module provides a centralized factory system for creating membership function
instances by name, supporting both full class names and convenient aliases. It
automatically discovers available membership function classes and provides a
flexible parameter-passing mechanism that separates membership function parameters
from other system parameters.

The factory system enables:
- Dynamic membership function creation from string names
- Automatic class discovery and registration
- Flexible parameter handling with automatic separation
- User-friendly aliases for common membership functions
- Integration with the broader AxisFuzzy fuzzification system

Architecture
------------
The factory system consists of three key components:

1. **Class Discovery**: Automatically scans :mod:`axisfuzzy.membership.function`
   to build a registry of available membership function classes.

2. **Alias System**: Provides user-friendly short names and alternative spellings
   for membership functions (e.g., 'trimf' for TriangularMF).

3. **Parameter Separation**: Intelligently separates membership function parameters
   from other system parameters during instance creation.

Core Functions
--------------
The factory provides two main functions:

- ``get_mf_class(name)``: Resolves a name/alias to the corresponding
  :class:`MembershipFunction` subclass for direct instantiation.

- ``create_mf(name, **kwargs)``: Creates a membership function instance
  with automatic parameter handling and returns unused parameters.

Automatic Class Discovery
-------------------------
The factory automatically discovers all :class:`MembershipFunction` subclasses
defined in :mod:`axisfuzzy.membership.function` using Python's introspection
capabilities. This means:

- New membership function classes are automatically available
- No manual registration is required for standard functions
- The system stays synchronized with available implementations

Alias System
------------
The factory supports multiple naming conventions:

**Standard Aliases** (manually maintained):
    - 'trimf' → :class:`TriangularMF`
    - 'trapmf' → :class:`TrapezoidalMF`
    - 'gaussmf' → :class:`GaussianMF`
    - 'smf' → :class:`SMF`
    - 'zmf' → :class:`ZMF`
    - 'gbellmf' → :class:`GeneralizedBellMF`
    - 'pimf' → :class:`PiMF`
    - 'gauss2mf' → :class:`DoubleGaussianMF`
    - 'sigmoid' → :class:`SigmoidMF`

**Automatic Aliases** (generated):
    - All class names in lowercase (e.g., 'triangularmf' → :class:`TriangularMF`)
    - Full class names (e.g., 'TriangularMF' → :class:`TriangularMF`)

Parameter Handling
------------------
The :func:`create_mf` function uses introspection to intelligently separate
parameters intended for the membership function constructor from other
system parameters:

1. **Constructor Analysis**: Inspects the target class's ``__init__`` method
   to identify required and optional parameters.

2. **Parameter Separation**: Splits input kwargs into membership function
   parameters and remaining system parameters.

3. **Clean Returns**: Returns both the created instance and unused parameters
   for further processing by calling systems.

This design allows the factory to be embedded in larger systems (like the
fuzzification system) where multiple components may share parameter dictionaries.

Error Handling
--------------
The factory provides clear error messages for common issues:

- **Unknown Functions**: Lists all available functions when an invalid name is provided
- **Parameter Errors**: Preserves and re-raises constructor validation errors
- **Import Errors**: Handles missing dependencies gracefully

Notes
-----
- All function names and aliases are case-insensitive for user convenience
- The factory system is thread-safe for read operations
- New membership functions added to the function module are automatically available
- Parameter validation is delegated to the individual membership function constructors

See Also
--------
axisfuzzy.membership.base : Base class for all membership functions
axisfuzzy.membership.function : Standard membership function implementations
axisfuzzy.fuzzify : Fuzzification system that uses this factory

Examples
--------
Basic factory usage:

.. code-block:: python

    from axisfuzzy.membership.factory import create_mf, get_mf_class

    # Create instances using aliases
    tri_mf, unused = create_mf('trimf', a=0, b=0.5, c=1)
    gauss_mf, unused = create_mf('gaussmf', sigma=1.0, c=0.5)

    # Create using full class names
    trap_mf, unused = create_mf('TrapezoidalMF', a=0, b=0.2, c=0.8, d=1)

    # Case-insensitive names
    sigmoid_mf, unused = create_mf('SIGMOID', k=2.0, c=0.5)

Parameter separation in system integration:

.. code-block:: python

    # Mixed parameters for fuzzification system
    all_params = {
        'a': 0, 'b': 0.5, 'c': 1,           # For membership function
        'mtype': 'qrofn', 'q': 2,           # For fuzzy system
        'method': 'direct'                   # For fuzzification method
    }

    # Factory separates parameters automatically
    mf, system_params = create_mf('trimf', **all_params)
    print(system_params)  # {'mtype': 'qrofn', 'q': 2, 'method': 'direct'}

Direct class access:

.. code-block:: python

    # Get class reference for advanced usage
    MFClass = get_mf_class('triangularmf')

    # Create multiple instances efficiently
    mf1 = MFClass(a=0, b=0.3, c=0.6)
    mf2 = MFClass(a=0.4, b=0.7, c=1.0)

Error handling:

.. code-block:: python

    try:
        mf, unused = create_mf('unknown_function', param=1)
    except ValueError as e:
        print(e)  # Lists all available functions

    try:
        mf, unused = create_mf('trimf', a=1, b=0.5, c=0)  # Invalid order
    except ValueError as e:
        print(e)  # TriangularMF validation error

Integration with other systems:

.. code-block:: python

    # Typical usage in fuzzification system
    def create_fuzzifier(mf_name, **params):
        # Create membership function with parameter separation
        mf, remaining_params = create_mf(mf_name, **params)

        # Use remaining parameters for other components
        mtype = remaining_params.get('mtype', 'qrofn')
        method = remaining_params.get('method', 'direct')

        # Build complete system
        return SomeFuzzificationSystem(mf, mtype, method)

References
----------
- Python ``inspect`` module documentation for introspection techniques
- Factory Method pattern in software design
- Plugin architecture patterns for extensible systems
"""

import inspect
from typing import Dict, Type, Any, Tuple

from .base import MembershipFunction
from . import function as mfs


# Automatic class discovery: Build mapping from class names to class objects
# This uses introspection to find all MembershipFunction subclasses in the function module
# e.g., {'TriangularMF': <class 'TriangularMF'>, 'GaussianMF': <class 'GaussianMF'>, ...}
_mf_class_map: Dict[str, Type[MembershipFunction]] = {
    name: obj for name, obj in inspect.getmembers(mfs, inspect.isclass)
    if issubclass(obj, MembershipFunction) and obj is not MembershipFunction
}

# Manual alias mapping: User-friendly names and alternative spellings
# These are carefully curated aliases that follow MATLAB Fuzzy Logic Toolbox conventions
# and provide convenient short forms for common membership functions
_mf_alias_map: Dict[str, Type[MembershipFunction]] = {
    'sigmoid': mfs.SigmoidMF,           # Standard sigmoid function
    'trimf': mfs.TriangularMF,          # MATLAB-style triangular
    'trapmf': mfs.TrapezoidalMF,        # MATLAB-style trapezoidal
    'gaussmf': mfs.GaussianMF,          # MATLAB-style Gaussian
    'smf': mfs.SMF,                     # S-shaped membership function
    'zmf': mfs.ZMF,                     # Z-shaped membership function
    'gbellmf': mfs.GeneralizedBellMF,   # MATLAB-style generalized bell
    'pimf': mfs.PiMF,                   # Pi-shaped membership function
    'gauss2mf': mfs.DoubleGaussianMF    # MATLAB-style double Gaussian
}

# Automatic alias generation: Add lowercase versions of all class names
# This allows users to use either exact class names or lowercase versions
# e.g., 'TriangularMF' or 'triangularmf' both work
_mf_alias_map.update({k.lower(): v for k, v in _mf_class_map.items()})


def get_mf_class(name: str) -> Type[MembershipFunction]:
    """
    Resolve a membership function name to its corresponding class.

    This function provides a unified interface for obtaining membership function
    classes by name, supporting both full class names and convenient aliases.
    Name matching is case-insensitive for user convenience.

    Parameters
    ----------
    name : str
        Name or alias of the membership function. Can be:
        - Full class name (e.g., 'TriangularMF', 'GaussianMF')
        - Lowercase class name (e.g., 'triangularmf', 'gaussianmf')
        - Standard alias (e.g., 'trimf', 'gaussmf', 'sigmoid')
        Case-insensitive matching is supported.

    Returns
    -------
    Type[MembershipFunction]
        The membership function class corresponding to the given name.

    Raises
    ------
    ValueError
        If the specified name is not found in the registry. The error message
        includes a complete list of available function names and aliases.

    Examples
    --------
    Get classes using different naming conventions:

    .. code-block:: python

        # Using full class names
        TriClass = get_mf_class('TriangularMF')
        GaussClass = get_mf_class('GaussianMF')

        # Using MATLAB-style aliases
        TriClass = get_mf_class('trimf')
        GaussClass = get_mf_class('gaussmf')

        # Using lowercase class names
        TriClass = get_mf_class('triangularmf')
        GaussClass = get_mf_class('gaussianmf')

        # Case-insensitive matching
        TriClass = get_mf_class('TRIMF')
        TriClass = get_mf_class('TriangularMF')
        TriClass = get_mf_class('triangularmf')

    Instantiate classes directly:

    .. code-block:: python

        # Get class and create instances
        MFClass = get_mf_class('trimf')
        mf1 = MFClass(a=0, b=0.3, c=0.6)
        mf2 = MFClass(a=0.4, b=0.7, c=1.0)

    Error handling:

    .. code-block:: python

        try:
            UnknownClass = get_mf_class('unknown_function')
        except ValueError as e:
            print(e)  # Shows list of all available functions
    """
    mf_cls = _mf_alias_map.get(name.lower())
    if mf_cls is None:
        available = ", ".join(sorted(_mf_alias_map.keys()))
        raise ValueError(f"Unknown membership function '{name}'. Available functions are: {available}")
    return mf_cls


def create_mf(name: str, **mf_kwargs: Any) -> Tuple[MembershipFunction, Dict[str, Any]]:
    """
    Factory function for creating membership function instances with parameter separation.

    This function creates a membership function instance of the specified type,
    automatically handling parameter separation between membership function
    parameters and other system parameters. It uses introspection to determine
    which parameters belong to the membership function constructor and which
    should be passed to other system components.

    Parameters
    ----------
    name : str
        Name or alias of the membership function to create. Supports the same
        naming conventions as :func:`get_mf_class`.
    **mf_kwargs
        Mixed parameters that may include:
        - Parameters for the membership function constructor
        - Parameters for other system components (returned as unused)

    Returns
    -------
    tuple of (MembershipFunction, dict)
        A tuple containing:

        - **instance** : MembershipFunction
            The created membership function instance.
        - **remaining_kwargs** : dict
            Dictionary of parameters that were not used in the membership
            function constructor. These can be passed to other system components.

    Raises
    ------
    ValueError
        - If the specified function name is not recognized
        - If the membership function constructor raises validation errors
    TypeError
        If required parameters for the membership function are missing

    Notes
    -----
    The parameter separation works by:
    1. Inspecting the target class's ``__init__`` method signature
    2. Identifying which parameters are accepted by the constructor
    3. Separating input kwargs into constructor and remaining parameters
    4. Creating the instance with only the relevant parameters

    This design enables embedding the factory in larger systems where multiple
    components share parameter dictionaries.

    Examples
    --------
    Basic usage with parameter separation:

    .. code-block:: python

        # Mixed parameters for different system components
        all_params = {
            'a': 0, 'b': 0.5, 'c': 1,        # TriangularMF parameters
            'mtype': 'qrofn',                 # Fuzzy system parameter
            'q': 2,                           # Fuzzy system parameter
            'method': 'centroid'              # Processing parameter
        }

        # Create membership function and separate parameters
        mf, unused = create_mf('trimf', **all_params)
        print(type(mf).__name__)    # 'TriangularMF'
        print(unused)               # {'mtype': 'qrofn', 'q': 2, 'method': 'centroid'}

    Creating different membership function types:

    .. code-block:: python

        # Triangular membership function
        tri_mf, _ = create_mf('trimf', a=0, b=0.5, c=1)

        # Gaussian membership function
        gauss_mf, _ = create_mf('gaussmf', sigma=1.0, c=0.5)

        # Trapezoidal membership function
        trap_mf, _ = create_mf('trapmf', a=0, b=0.2, c=0.8, d=1)

        # Sigmoid membership function
        sig_mf, _ = create_mf('sigmoid', k=2.0, c=0.5)

    Integration with fuzzification systems:

    .. code-block:: python

        def create_fuzzifier(mf_type, **params):
            \"\"\"Example of factory integration in larger system.\"\"\"
            # Create membership function with parameter separation
            mf, system_params = create_mf(mf_type, **params)

            # Extract system parameters
            mtype = system_params.get('mtype', 'qrofn')
            q_value = system_params.get('q', 2)
            method = system_params.get('method', 'direct')

            # Build complete fuzzification system
            return FuzzificationEngine(
                membership_function=mf,
                target_mtype=mtype,
                q=q_value,
                method=method
            )

        # Usage with mixed parameters
        fuzzifier = create_fuzzifier(
            'trimf', a=0, b=0.5, c=1,
            mtype='qrofn', q=3, method='centroid'
        )

    Error handling and validation:

    .. code-block:: python

        # Invalid function name
        try:
            mf, _ = create_mf('invalid_name', param=1)
        except ValueError as e:
            print(f"Function error: {e}")

        # Invalid parameters (caught by membership function)
        try:
            mf, _ = create_mf('trimf', a=1, b=0.5, c=0)  # Invalid order
        except ValueError as e:
            print(f"Parameter error: {e}")

        # Missing required parameters
        try:
            mf, _ = create_mf('trimf')  # No parameters provided
        except TypeError as e:
            print(f"Missing parameters: {e}")

    Advanced usage with class inspection:

    .. code-block:: python

        # Get available parameter names for a function
        MFClass = get_mf_class('trimf')
        signature = inspect.signature(MFClass.__init__)
        param_names = list(signature.parameters.keys())
        print(f"TriangularMF parameters: {param_names}")
        # Output: ['self', 'params', 'a', 'b', 'c']

        # Create with only known parameters
        known_params = {k: v for k, v in all_params.items()
                       if k in param_names}
        mf, unused = create_mf('trimf', **known_params)

    Batch creation for multiple functions:

    .. code-block:: python

        # Configuration for multiple membership functions
        mf_configs = [
            {'name': 'trimf', 'a': 0, 'b': 0.25, 'c': 0.5},
            {'name': 'gaussmf', 'sigma': 0.1, 'c': 0.75},
            {'name': 'trapmf', 'a': 0.5, 'b': 0.6, 'c': 0.9, 'd': 1.0}
        ]

        # Create all functions
        mf_instances = []
        for config in mf_configs:
            name = config.pop('name')  # Remove name from parameters
            mf, _ = create_mf(name, **config)
            mf_instances.append(mf)

    See Also
    --------
    get_mf_class : Get membership function class without creating instance
    axisfuzzy.membership.base.MembershipFunction : Base class for all functions
    axisfuzzy.membership.function : Available membership function implementations
    """
    mf_cls = get_mf_class(name)

    # Use introspection to determine which parameters belong to the constructor
    # This allows automatic separation of membership function parameters from
    # other system parameters in mixed parameter dictionaries
    constructor_signature = inspect.signature(mf_cls.__init__)
    mf_params = constructor_signature.parameters.keys()

    # Separate parameters into those for the membership function and others
    mf_init_kwargs = {}
    remaining_kwargs = {}
    for key, value in mf_kwargs.items():
        if key in mf_params:
            mf_init_kwargs[key] = value
        else:
            remaining_kwargs[key] = value

    # Create the membership function instance with only relevant parameters
    # Any validation errors will be raised by the specific membership function
    instance = mf_cls(**mf_init_kwargs)

    return instance, remaining_kwargs
