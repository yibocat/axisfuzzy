.. _user_guide_fuzzifiers:

Fuzzifiers
==========

**Fuzzification** is the process of converting a crisp (real-world, numerical) input value into a fuzzy set. In AxisFuzzy, this critical operation is handled by the ``Fuzzifier`` class, a flexible and powerful tool for translating precise data into the language of fuzzy logic.

What is a Fuzzifier?
--------------------

A ``Fuzzifier`` acts as a bridge between the crisp world and the fuzzy world. It takes a crisp input and, using one or more membership functions, determines the degree to which that input belongs to various fuzzy sets.

For example, if we have a crisp temperature reading of 22°C, a fuzzifier can tell us that this temperature is "0.8 Warm" and "0.2 Cool".

Key Features of the `Fuzzifier`
--------------------------------

*   **Multi-Function Mapping**: A single ``Fuzzifier`` can manage multiple membership functions, each representing a different fuzzy set (e.g., 'Cold', 'Warm', 'Hot').
*   **Dictionary-Based Output**: It returns a dictionary where keys are the names of the fuzzy sets and values are the corresponding membership degrees.
*   **Extensibility**: It can work with both built-in and custom membership functions.

How to Use a Fuzzifier
----------------------

Using a ``Fuzzifier`` involves three main steps:

1.  **Instantiate Membership Functions**: Define the membership functions that represent your fuzzy sets.
2.  **Create a Fuzzifier Instance**: Create a ``Fuzzifier`` and register your membership functions with it.
3.  **Call the Fuzzifier**: Pass a crisp value to the fuzzifier instance to get the fuzzy output.

Example
-------

Let's build a fuzzifier for temperature.

.. code-block:: python

   import axisfuzzy as af
   from axisfuzzy.membership import trapmf, trimf

   # 1. Define membership functions for 'Cold', 'Warm', and 'Hot'
   cold_mf = trapmf(a=-10, b=0, c=5, d=15)
   warm_mf = trimf(a=10, b=20, c=30)
   hot_mf = trapmf(a=25, b=35, c=40, d=50)

   # 2. Create a Fuzzifier with these functions
   temperature_fuzzifier = af.Fuzzifier(
       membership_functions={
           'Cold': cold_mf,
           'Warm': warm_mf,
           'Hot': hot_mf
       }
   )

   # 3. Fuzzify a crisp temperature value
   crisp_temp = 22
   fuzzy_output = temperature_fuzzifier(crisp_temp)

   # print(fuzzy_output)
   # Expected output (approximate):
   # {
   #     'Cold': 0.0,
   #     'Warm': 0.8,
   #     'Hot': 0.0
   # }

Integration with `Fuzznum` and `Fuzzarray`
------------------------------------------

While ``Fuzzifier`` is a powerful standalone tool, its primary role is often to generate the initial membership and non-membership degrees needed to create ``Fuzznum`` or ``Fuzzarray`` objects, especially for Type-1 fuzzy sets where the non-membership degree is simply `1 - membership`.

.. code-block:: python

   # Continuing the example above...
   membership_degree = fuzzy_output['Warm']

   # Create a Fuzznum representing 'Warm' to a degree of 0.8
   # Note: This is a simplified example. In practice, you might combine
   # outputs or use them to initialize more complex fuzzy numbers.
   fuzz_num = af.Fuzznum([membership_degree, 1 - membership_degree], mtype='ivfn')

.. seealso::

   - :ref:`user_guide_membership_functions` for details on creating the functions used by the ``Fuzzifier``.
   - The `axisfuzzy.fuzzifier` module for the full API.