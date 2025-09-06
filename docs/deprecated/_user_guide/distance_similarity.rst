.. _user_guide_distance_similarity:

Distance and Similarity Measures
================================

Measuring the distance or similarity between fuzzy numbers is a fundamental operation in many fuzzy logic applications, such as clustering, classification, and decision-making. AxisFuzzy provides a rich set of methods attached to the `Fuzznum` and `Fuzzarray` objects for these calculations.

Core Concepts
-------------

*   **Distance**: A measure of how far apart two fuzzy numbers are. A smaller distance implies the numbers are more alike. The distance is always non-negative, and the distance from a number to itself is zero.
*   **Similarity**: A measure of how much two fuzzy numbers resemble each other. It is typically a value between 0 and 1, where 1 means the numbers are identical.

In many cases, similarity can be derived from distance (e.g., `similarity = 1 - distance`).

Available Metrics
-----------------

AxisFuzzy implements numerous distance and similarity metrics from academic literature, accessible via the `.distance()` and `.similarity()` methods of a `Fuzznum` object.

You can specify the metric to use with the `method` parameter.

Common Distance Metrics
~~~~~~~~~~~~~~~~~~~~~~~

*   **Hamming Distance (`hamming`)**: One of the most common metrics, it calculates the absolute difference between the membership and non-membership degrees.

    .. math::

       d(A, B) = \frac{1}{2} \left( |\mu_A - \mu_B| + |\nu_A - \nu_B| \right)

*   **Euclidean Distance (`euclidean`)**: Calculates the straight-line distance in the membership/non-membership space.

    .. math::

       d(A, B) = \sqrt{\frac{1}{2} \left( (\mu_A - \mu_B)^2 + (\nu_A - \nu_B)^2 \right)}

*   **Normalized Hamming/Euclidean**: Versions of the above, normalized to ensure the result is within a specific range, often [0, 1].

Example: Calculating Distance
-----------------------------

.. code-block:: python

   import axisfuzzy as af

   # Pythagorean Fuzzy Numbers
   fuzz1 = af.Fuzznum([0.8, 0.3], mtype='pfn')
   fuzz2 = af.Fuzznum([0.7, 0.4], mtype='pfn')

   # Calculate Hamming distance
   ham_dist = fuzz1.distance(fuzz2, method='hamming')
   # print(f"Hamming Distance: {ham_dist}")

   # Calculate Euclidean distance
   euc_dist = fuzz1.distance(fuzz2, method='euclidean')
   # print(f"Euclidean Distance: {euc_dist}")

Similarity Measures
-------------------

Similarity measures are often based on distance metrics. For instance, a common similarity measure is `1 - distance`.

Example: Calculating Similarity
-------------------------------

.. code-block:: python

   # Continuing from the previous example...

   # Calculate similarity based on Hamming distance
   ham_sim = fuzz1.similarity(fuzz2, method='chen') # Chen's similarity is based on a distance metric
   # print(f"Similarity: {ham_sim}")

Batch Operations with `Fuzzarray`
---------------------------------

The `.distance()` and `.similarity()` methods are also available on `Fuzzarray` objects for efficient, vectorized calculations.

When comparing two `Fuzzarray` objects of the same shape, the operation is performed element-wise, returning a NumPy array of distances or similarities.

.. code-block:: python

   import axisfuzzy as af

   data1 = [[0.8, 0.3], [0.9, 0.1]]
   data2 = [[0.7, 0.4], [0.8, 0.2]]
   f_array1 = af.Fuzzarray(data1, mtype='pfn')
   f_array2 = af.Fuzzarray(data2, mtype='pfn')

   # Batch distance calculation
   distances = f_array1.distance(f_array2, method='hamming')
   # print(distances)
   # Expected output: array([0.1, 0.1])

Choosing a Metric
-----------------

The choice of metric depends heavily on the application and the properties of the fuzzy numbers being used. Some metrics may be more sensitive to changes in membership degrees, while others give more weight to non-membership. It is recommended to consult relevant literature for your specific use case.

.. seealso::

   The API documentation for `axisfuzzy.Fuzznum.distance` and `axisfuzzy.Fuzznum.similarity` for a full list of supported methods.