.. _quick_start:

Quick Start
===========

This guide provides a brief introduction to the basic functionalities of ``axisfuzzy``. You will learn how to create and manipulate fuzzy numbers and fuzzy arrays, perform basic arithmetic operations, and utilize some of the convenient features for fuzzy-related calculations.

Core Object: Fuzznum
--------------------

The ``Fuzznum`` is the fundamental building block in ``axisfuzzy``, representing a single fuzzy number. Currently, ``axisfuzzy`` supports two types of fuzzy numbers: 
 - q-rung orthopair fuzzy numbers (q-ROFNs, ``mtype='qrofn'``)
 - q-rung orthopair hesitant fuzzy numbers (q-ROHFNs, ``mtype='qrohfn'``)

To create a fuzzy number, you can use the convenient factory function ``fuzzynum``. Here is an example of how to create a q-ROFN:

.. code-block:: python

    from axisfuzzy import fuzzynum

    # Create a q-rung orthopair fuzzy number (q-ROFN) 
    #   and q-rung orthopair hesitant fuzzy number (q-ROHFN)
    # Here, q=3, the membership degree is 0.8, 
    #   and the non-membership degree is 0.6 for q-ROFN
    # And q=1, the membership degrees are [0.6,0.7], 
    #   and the non-membership degrees are [0.2] for q-ROHFN
    fn1 = fuzzynum((0.8, 0.6), q=3)
    fn2 = fuzzynum(([0.6,0.7],[0.2]), mtype='qrohfn')

    print(fn1)
    # Output: <0.8,0.6>
    print(fn2)
    # Output: <[0.6,0.7],[0.2]>

In this example, we specify the q-ROFN fuzzy number type as ``mtype='qrofn'`` (default) and set the value of `q` to 3. The membership degree (`md`) and non-membership degree (`nmd`) are passed as tuple ``(md, nmd)``. The condition :math:`\mu^q + \nu^q <= 1` must be satisfied.
And for q-ROHFN, the membership degrees and non-membership degrees are passed as list ``([mds], [nmds])``.

More fuzzy sets will be expanded in the future, including: fuzzy sets, type-II fuzzy sets, interval-valued fuzzy sets, etc. Alternatively, developers can also customize and develop various fuzzy sets as needed and register them into ``axisfuzzy``.

Core Container: Fuzzarray
-------------------------

The ``Fuzzarray`` is a container for ``Fuzznum`` objects, designed for efficient numerical computation. It is built upon ``numpy.ndarray``, which allows it to leverage the high-performance capabilities of NumPy for vectorized operations. ``Fuzzarray`` provides a convenient way to store and manipulate fuzzy numbers in a structured and efficient manner. Therefore, ``Fuzzarray`` can be regarded as a fuzzy set of ``Fuzznum``.

Similar to ``fuzzynum``, there is a corresponding factory function ``fuzzyset`` for creating fuzzy arrays. You can create a ``Fuzzarray`` from a list of data.

.. code-block:: python

    from axisfuzzy import fuzzyset, fuzzynum

    # Create three Fuzznum with mtype is default qrofn and q is 1
    # As a list and create a Fuzzarray
    fn1 = fuzzynum((0.5,0.3))
    fn2 = fuzzynum((0.6,0.2))
    fn3 = fuzzynum((0.4,0.4))
    
    f = [fn1, fn2, fn3]
    print(f)
    # [<0.5,0.3>,<0.6,0.2>,<0.4,0.4>]

    fa = fuzzyset(f)
    print(fa)
    # Fuzzarray([<0.5,0.3> <0.6,0.2> <0.4,0.4>], mtype='qrofn', q=1, shape=(3,))

In this example, a 3-element ``Fuzzarray`` is created. The ``mtype`` and `q` parameters are applied to all elements in the array. This demonstrates a quick and convenient way to construct fuzzy arrays for numerical calculations.

Basic Operations
----------------

``axisfuzzy`` overloads standard arithmetic operators, allowing you to perform calculations with fuzzy numbers and arrays in an intuitive and straightforward manner, similar to how you would with regular numbers.

Here are some examples of basic operations:

.. code-block:: python

    from axisfuzzy import fuzzynum, fuzzyset

    # Operations with Fuzznum
    fn1 = fuzzynum((0.8, 0.6), q=3)
    fn2 = fuzzynum((0.2, 0.5), q=3)

    # Addition
    add_result = fn1 + fn2
    print(add_result)
    # Output: <0.802,0.3>

    # Multiplication
    mul_result = fn1 * fn2
    print(mul_result)
    # Output: <0.16,0.6797>

    # Operations with Fuzzarray
    fn11 = fuzzynum((0.5,0.3))
    fn12 = fuzzynum((0.6,0.2))
    fn13 = fuzzynum((0.4,0.4))

    fn21 = fuzzynum((0.2,0.6))
    fn22 = fuzzynum((0.4,0.5))
    fn23 = fuzzynum((0.3,0.7))
    
    data1 = [fn11, fn12, fn13]
    data2 = [fn21, fn22, fn23]

    fa1 = fuzzyset(data1)
    fa2 = fuzzyset(data2)

    # Element-wise addition
    add_array_result = fa1 + fa2
    print(add_array_result)

    # Element-wise multiplication
    mul_array_result = fa1 * fa2
    print(mul_array_result)

As you can see, the syntax is clean and closely mirrors standard Python numerical operations, making it easy to integrate ``axisfuzzy`` into your data analysis and modeling workflows.

Convenient Creation
-------------------

The ``axisfuzzy.random`` module provides functions to easily generate random fuzzy numbers and arrays. This is particularly useful for testing algorithms, performing simulations, or initializing models.

The ``rand`` function is the primary tool for this purpose. Here’s how to use it:

.. code-block:: python

    import axisfuzzy as af

    # Generate a single random Fuzznum with default mtype=qrofn and q=1
    random_fn = af.random.rand()
    print(random_fn)

    # Generate a 2x2 Fuzzarray of random fuzzy numbers with mtype=qrohfn and q=3
    random_fa = af.random.rand(mtype='qrohfn', q=3, shape=(2, 2))
    print(random_fa)

By default, the ``rand`` function generates membership and non-membership degrees from a uniform distribution between 0 and 1, while ensuring that the constraints of the fuzzy number type are met. You can also provide a specific random number generator (`rng`) for more control over the generation process.

From Crisp to Fuzzy: Fuzzification
----------------------------------

Fuzzification is the process of converting a crisp (i.e., precise) value into a fuzzy value. ``axisfuzzy`` provides tools to facilitate this process, allowing you to represent crisp data in a fuzzy format.

The ``Fuzzifier`` class is used for this purpose. It takes a membership function and a fuzzy number type to perform the conversion.

.. code-block:: python

    import numpy as np
    from axisfuzzy import Fuzzifier

    # Create a fuzzifier with a gaussian membership function for q-ROFNs
    # The parameters {"sigma":0.5,"c":0.0} define the shape of the gaussian function.
    fuzzifier = Fuzzifier(
        "gaussmf",
        mtype="qrofn",
        mf_params={"sigma":0.5,"c":0.0})

    # Convert a crisp numpy array to a Fuzzarray
    crisp_data = np.array([0.3, 0.5, 0.7])
    fuzzy_data = fuzzifier(crisp_data)

    print(fuzzy_data)

This example demonstrates how to convert a NumPy array of crisp values into a ``Fuzzarray``. The membership degrees of the resulting fuzzy numbers are determined by the `triangular` membership function. This is a powerful feature for integrating real-world data into fuzzy logic systems.