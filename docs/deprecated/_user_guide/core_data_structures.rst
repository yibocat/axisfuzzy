.. _user_guide_core_data_structures:

Core Data Structures: Fuzznum and Fuzzarray
===========================================

At the heart of AxisFuzzy are two fundamental data structures: ``Fuzznum`` and ``Fuzzarray``. They are the primary objects you will interact with when performing fuzzy logic calculations. This guide provides a detailed exploration of their architecture, properties, and capabilities.

Fuzznum: The Atomic Unit of Fuzziness
-------------------------------------

A ``Fuzznum`` represents a single fuzzy number. It encapsulates a membership function and provides methods for operations and analysis.

**Key Concepts:**

*   **Membership Function**: Defines the degree of membership for any given crisp value. It is the mathematical core of the fuzzy number.
*   **Alpha-Cuts**: An alpha-cut of a fuzzy number is the crisp set of elements whose membership grade is greater than or equal to the specified alpha value. It is a fundamental concept for fuzzy arithmetic.

.. math::

   [A]_{\\alpha} = \\{x \in X | \mu_A(x) \ge \\alpha\\}

**Example: Creating a Triangular Fuzznum**

.. code-block:: python

   from axisfuzzy.core import fuzznums

   # Create a triangular fuzzy number with parameters (left, peak, right)
   tri_fuzznum = fuzznums.triangular(1, 5, 10)

   print(tri_fuzznum)

Fuzzarray: High-Performance Fuzzy Arrays
----------------------------------------

A ``Fuzzarray`` is a homogeneous n-dimensional array object that contains ``Fuzznum`` instances of the same type. It is built on top of NumPy, enabling high-performance, vectorized operations on large sets of fuzzy numbers.

**Key Advantages:**

*   **Performance**: Leverages NumPy's C-backend for fast, vectorized operations, avoiding slow Python loops.
*   **Interoperability**: Seamlessly integrates with the scientific Python ecosystem (e.g., NumPy, SciPy, Pandas).
*   **Rich Functionality**: Supports a wide range of array manipulation routines, mathematical operations, and statistical functions.

**Example: Creating and Operating on a Fuzzarray**

.. code-block:: python

   import numpy as np
   from axisfuzzy.core import fuzzarray

   # Create a Fuzzarray from a list of triangular fuzzy numbers
   fuzzy_arr = fuzzarray([
       fuzznums.triangular(1, 5, 10),
       fuzznums.triangular(2, 6, 11),
       fuzznums.triangular(3, 7, 12)
   ])

   # Perform vectorized addition
   result_arr = fuzzy_arr + 5

   print(result_arr)

Internal Representation
-----------------------

Both ``Fuzznum`` and ``Fuzzarray`` use a structured representation based on their membership function's parameters. This allows for efficient storage and computation. For example, a triangular fuzzy number is stored using its three defining points (left, peak, right). This parametric approach is a key design choice that enables high performance.