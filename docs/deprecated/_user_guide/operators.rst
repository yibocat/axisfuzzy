.. _user_guide_operators:

Fuzzy Operators
===============

AxisFuzzy provides a comprehensive suite of mathematical operators for `Fuzznum` and `Fuzzarray` objects, allowing you to perform fuzzy arithmetic and logical operations in an intuitive, Pythonic way. These operators are overloaded to work just like they would with standard numbers or NumPy arrays.

Arithmetic Operations
---------------------

All standard arithmetic operations are supported. When an operation is performed between two fuzzy numbers (or a fuzzy number and a scalar), a new fuzzy number is returned, with its membership and non-membership degrees calculated according to the rules of fuzzy arithmetic.

Here is a summary of the supported operations for two fuzzy numbers, `fuzz1` and `fuzz2`:

*   **Addition**: `fuzz1 + fuzz2`
*   **Subtraction**: `fuzz1 - fuzz2`
*   **Multiplication**: `fuzz1 * fuzz2`
*   **Division**: `fuzz1 / fuzz2`
*   **Power**: `fuzz1 ** scalar`

.. note::
   The specific formulas for these operations depend on the `mtype` (membership type) of the fuzzy numbers. For example, for intuitionistic fuzzy numbers (IFNs), the operations are typically defined using Einstein sums and products to ensure the results remain valid IFNs.

Example:

.. code-block:: python

   import axisfuzzy as af

   # q-Rung Orthopair Fuzzy Numbers (q=3)
   fuzz1 = af.Fuzznum([0.8, 0.4], mtype='qrofn', q=3)
   fuzz2 = af.Fuzznum([0.7, 0.5], mtype='qrofn', q=3)

   # Addition
   add_result = fuzz1 + fuzz2
   # print(f"Addition: {add_result}")

   # Multiplication
   mul_result = fuzz1 * fuzz2
   # print(f"Multiplication: {mul_result}")

Logical Operations
------------------

AxisFuzzy also supports logical operations, which are essential for rule-based fuzzy systems and decision-making processes.

*   **Complement (NOT)**: `~fuzz1`
    This operation swaps the membership and non-membership degrees.

*   **Union (OR)**: `fuzz1 | fuzz2`
    The union is typically implemented using the maximum operator for both membership and non-membership degrees, depending on the fuzzy logic system.

*   **Intersection (AND)**: `fuzz1 & fuzz2`
    The intersection is typically implemented using the minimum operator.

Example:

.. code-block:: python

   import axisfuzzy as af

   fuzz = af.Fuzznum([0.8, 0.2], mtype='ivfn')

   # Complement
   complement = ~fuzz
   # print(f"Complement: {complement}")  # Expected: Fuzznum([0.2, 0.8], ...)

Comparison Operations
---------------------

Comparing two fuzzy numbers is more complex than comparing crisp numbers. AxisFuzzy provides a `score` function to rank fuzzy numbers. The standard comparison operators (`<`, `>`, `<=`, `>=`, `==`, `!=`) are implemented based on this score.

*   `fuzz1 > fuzz2` is equivalent to `fuzz1.score() > fuzz2.score()`
*   `fuzz1 == fuzz2` checks for equality of membership and non-membership degrees, not just the score.

.. code-block:: python

   fuzz1 = af.Fuzznum([0.9, 0.1])
   fuzz2 = af.Fuzznum([0.8, 0.1])

   # print(fuzz1 > fuzz2)  # True, because score of fuzz1 is higher
   # print(fuzz1 == fuzz1) # True

Operations with Scalars
-----------------------

`Fuzznum` objects can also be combined with scalar (crisp) numbers. The scalar is treated as a special type of fuzzy number for the operation.

*   `fuzz1 * scalar`
*   `fuzz1 + scalar`

Batch Operations with `Fuzzarray`
---------------------------------

All the operators mentioned above are vectorized and work seamlessly with `Fuzzarray` objects, allowing for high-performance batch computations.

.. code-block:: python

   data1 = [[0.8, 0.2], [0.7, 0.3]]
   data2 = [[0.6, 0.3], [0.5, 0.4]]
   f_array1 = af.Fuzzarray(data1)
   f_array2 = af.Fuzzarray(data2)

   # Batch addition
   sum_array = f_array1 + f_array2
   # print(sum_array)

   # Scalar multiplication
   scaled_array = f_array1 * 0.5
   # print(scaled_array)