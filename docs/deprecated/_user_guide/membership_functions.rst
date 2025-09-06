.. _user_guide_membership_functions:

Membership Functions
====================

In fuzzy logic, a **membership function** (MF) is a curve that defines how each point in the input space is mapped to a membership value (or degree of membership) between 0 and 1. AxisFuzzy provides a rich set of built-in membership functions and allows for easy extension with custom functions.

Core Concepts
-------------

All membership functions in AxisFuzzy are designed to be:

*   **Callable**: They can be called with a crisp input value to get the degree of membership.
*   **Vectorized**: They are built on NumPy and operate efficiently on arrays of data.
*   **Parameterizable**: They are instantiated with specific parameters that define their shape (e.g., the center and width of a Gaussian function).

Built-in Membership Functions
-----------------------------

AxisFuzzy includes a variety of common membership functions. Here are some of the key ones:

Triangular MF
~~~~~~~~~~~~~

The triangular membership function is defined by three parameters: `a`, `b`, and `c`, representing the vertices of the triangle.

.. math::

   f(x; a, b, c) = \max\left(\min\left(\frac{x-a}{b-a}, \frac{c-x}{c-b}\right), 0\right)

.. code-block:: python

   from axisfuzzy.membership import trimf

   # Define a triangular membership function from x=0 to x=10, with a peak at x=5
   membership_func = trimf(a=0, b=5, c=10)

   # Calculate membership for a single value
   # value = membership_func(4)  # Expected: 0.8

Trapezoidal MF
~~~~~~~~~~~~~~

The trapezoidal membership function is defined by four parameters: `a`, `b`, `c`, and `d`.

.. math::

   f(x; a, b, c, d) = \max\left(\min\left(\frac{x-a}{b-a}, 1, \frac{d-x}{d-c}\right), 0\right)

.. code-block:: python

   from axisfuzzy.membership import trapmf

   # Define a trapezoidal membership function
   membership_func = trapmf(a=0, b=2, c=8, d=10)

   # Calculate membership
   # value = membership_func(5)  # Expected: 1.0

Gaussian MF
~~~~~~~~~~~

The Gaussian membership function is defined by its center `mean` and standard deviation `sigma`.

.. math::

   f(x; \mu, \sigma) = e^{-\frac{(x - \mu)^2}{2\sigma^2}}

.. code-block:: python

   from axisfuzzy.membership import gaussmf

   # Define a Gaussian membership function centered at 5 with std dev 2
   membership_func = gaussmf(mean=5, sigma=2)

   # Calculate membership
   # value = membership_func(5)  # Expected: 1.0

Using Membership Functions
--------------------------

Membership functions are most commonly used in the process of **fuzzification**, where a crisp numerical value is converted into a fuzzy set. This is typically handled by a ``Fuzzifier`` object, which uses one or more membership functions to perform the conversion.

.. seealso::

   - :ref:`user_guide_fuzzifiers` for how to use membership functions in practice.
   - The `axisfuzzy.membership` module for a full list of available functions.

Creating Custom Membership Functions
------------------------------------

While AxisFuzzy provides many built-in functions, you can easily create your own. A custom membership function is any callable that takes a NumPy array of crisp values and returns an array of membership degrees between 0 and 1.

.. code-block:: python

   import numpy as np

   def custom_bell_mf(x, width, center):
       """A simple custom bell-shaped membership function."""
       return 1 / (1 + np.abs((x - center) / width)**(2 * 1))

   # You can then use this function with a Fuzzifier
   # from axisfuzzy.fuzzifier import Fuzzifier
   # fuzzifier = Fuzzifier(membership_functions={'custom_bell': custom_bell_mf})