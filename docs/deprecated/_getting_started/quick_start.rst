.. _getting_started_quick_start:

***********
Quick Start
***********

This guide provides a quick example of how to use AxisFuzzy to perform fuzzy logic operations.

Creating Fuzzy Numbers
======================

First, let's create some fuzzy numbers. A fuzzy number is defined by its membership function. AxisFuzzy provides several built-in membership functions, such as triangular, trapezoidal, and Gaussian.

Here's how to create a triangular fuzzy number:

.. code-block:: python

    from axisfuzzy.core import fuzznum

    # Create a triangular fuzzy number with parameters (left, peak, right)
    a = fuzznum.Triangular(1, 5, 9)

    print(a)

Performing Operations
=====================

You can perform arithmetic operations on fuzzy numbers just like regular numbers:

.. code-block:: python

    b = fuzznum.Triangular(2, 6, 10)

    # Addition
    c = a + b
    print(f"Addition: {c}")

    # Subtraction
    d = a - b
    print(f"Subtraction: {d}")

    # Multiplication
    e = a * b
    print(f"Multiplication: {e}")

    # Division
    f = a / b
    print(f"Division: {f}")

Working with Fuzzy Arrays
=========================

AxisFuzzy also supports fuzzy arrays, which allow you to perform vectorized operations on collections of fuzzy numbers.

.. code-block:: python

    import numpy as np
    from axisfuzzy.core import fuzzarray

    # Create a fuzzy array from a list of fuzzy numbers
    arr = fuzzarray.Fuzzarray([a, b])

    # Perform operations on the array
    arr_sum = arr.sum()
    print(f"Sum of array: {arr_sum}")

    arr_mean = arr.mean()
    print(f"Mean of array: {arr_mean}")

This quick start provides a glimpse of what you can do with AxisFuzzy. For more detailed information, refer to the other sections of this guide.