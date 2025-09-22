.. _quick_start:

Quick Start
===========

Welcome to ``axisfuzzy``, a high-performance Python library for fuzzy logic and fuzzy computation. 
This guide provides a comprehensive introduction to the essential features that make ``axisfuzzy`` 
a powerful tool for researchers, engineers, and data scientists working with uncertainty and 
imprecision in their computational models.

``axisfuzzy`` is built around two core principles: **intuitive usability** and **computational efficiency**. 
The library offers a clean, Pythonic API that feels familiar to users of NumPy and pandas, while 
leveraging advanced architectural patterns to deliver exceptional performance for both scalar 
and vectorized fuzzy operations.

Key features that distinguish ``axisfuzzy``:

- **Factory-Based Object Creation**: Streamlined ``fuzzynum()`` and ``fuzzyset()`` functions for effortless fuzzy object instantiation
- **Extensible Type System**: Built-in support for q-Rung Orthopair Fuzzy Numbers (QROFN) and q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFN), with extensible architecture for custom types
- **High-Performance Backend**: ``Fuzzarray`` leverages NumPy's vectorized operations for efficient batch computations
- **Advanced Fuzzification**: Comprehensive membership function library and flexible fuzzification strategies
- **Random Generation**: Sophisticated random fuzzy number generators for simulation and testing

This quick start guide will walk you through these core capabilities, providing practical examples 
that demonstrate how to harness ``axisfuzzy``'s power in your fuzzy computing workflows.

.. contents::
   :local:

Creating Fuzzy Objects with Factory Functions
----------------------------------------------

The foundation of working with ``axisfuzzy`` lies in creating fuzzy numbers and fuzzy sets. 
The library provides two primary factory functions that abstract away the complexity of 
object instantiation while maintaining full control over fuzzy number properties.

Factory Functions Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``axisfuzzy`` employs a factory pattern that simplifies object creation while ensuring 
type safety and validation. The two core factory functions are:

- ``fuzzynum()``: Creates individual fuzzy numbers (``Fuzznum`` objects)
- ``fuzzyset()``: Creates collections of fuzzy numbers (``Fuzzarray`` objects)

These functions automatically handle type detection, parameter validation, and strategy 
selection based on your input specifications.

Creating Individual Fuzzy Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``fuzzynum()`` function supports multiple creation patterns for maximum flexibility:

.. code-block:: python

    from axisfuzzy import fuzzynum

    # Create a q-Rung Orthopair Fuzzy Number (QROFN)
    # Default: q=1, mtype='qrofn'
    fn1 = fuzzynum((0.3, 0.6))
    print(fn1)  # Output: <0.3,0.6>

    # Specify custom q-rung parameter
    fn2 = fuzzynum((0.7, 0.5), q=3)
    print(fn2)  # Output: <0.7,0.5>

    # Create using keyword arguments for clarity
    fn3 = fuzzynum(md=0.5, nmd=0.4, q=2)
    print(fn3)  # Output: <0.5,0.4>

For q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFN), the syntax accommodates multiple membership values:

.. code-block:: python

    # Create a QROHFN with multiple membership degrees
    fn4 = fuzzynum(([0.5, 0.6, 0.7], [0.2, 0.3]), mtype='qrohfn', q=1)
    print(fn4)  # Output: <[0.5,0.6,0.7],[0.2,0.3]>

    # Single membership, multiple non-membership degrees
    fn5 = fuzzynum(([0.6], [0.1, 0.2, 0.3]), mtype='qrohfn')
    print(fn5)  # Output: <[0.6],[0.1,0.2,0.3]>

Creating Fuzzy Sets
~~~~~~~~~~~~~~~~~~~~

The ``fuzzyset()`` function creates ``Fuzzarray`` objects, which are optimized containers 
for batch operations on multiple fuzzy numbers:

**1. Import axisfuzzy**

.. code-block:: python

    from axisfuzzy import fuzzyset, fuzzynum

**2. Create from a list of fuzzy numbers**

.. code-block:: python

    fuzzy_numbers = [
        fuzzynum((0.8, 0.2)),
        fuzzynum((0.6, 0.4)),
        fuzzynum((0.9, 0.1))
    ]
    
    fs1 = fuzzyset(fuzzy_numbers)
    print(fs1)

output::
    
    [<0.8,0.2> <0.6,0.4> <0.9,0.1>]
    
output with ``repr``::

    Fuzzarray([<0.8,0.2> <0.6,0.4> <0.9,0.1>], mtype='qrofn', q=1, shape=(3,))

**3. Create from tuples by converting to fuzzy numbers first**

.. code-block:: python

    tuple_data = [(0.7, 0.3), (0.5, 0.5), (0.8, 0.2)]
    fuzzy_list = [fuzzynum(t, q=2) for t in tuple_data]
    fs2 = fuzzyset(fuzzy_list)
    print(fs2)
    # Output: Fuzzarray([<0.7,0.3> <0.5,0.5> <0.8,0.2>], mtype='qrofn', q=2, shape=(3,))

**4. Multi-dimensional fuzzy arrays**

.. code-block:: python

    import numpy as np

    data_2d = np.array([[[0.8,0.2], [0.6,0.4]],
                        [[0.9,0.1], [0.7,0.3]]])

    fs3 = fuzzyset(data_2d.T)
    print(fs3.shape)  # Output: (2, 2)

**5. High-performance creation from raw arrays (advanced usage)**

.. code-block:: python

    import numpy as np
    md_values = np.array([0.8, 0.6, 0.7])
    nmd_values = np.array([0.1, 0.3, 0.2])
    raw_data = np.array([md_values, nmd_values])  # Shape: (2, 3)
    
    fs4 = fuzzyset(data=raw_data, mtype='qrofn', q=2)
    print(fs4)
    # Output: Fuzzarray([<0.8,0.1> <0.6,0.3> <0.7,0.2>], mtype='qrofn', q=2, shape=(3,))

Type Validation and Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``axisfuzzy`` automatically enforces mathematical constraints for each fuzzy number type. 
For QROFN, the constraint :math:`\mu^q + \nu^q \leq 1` is validated during creation:

.. code-block:: python

    # Valid QROFN (0.8^2 + 0.6^2 = 1.0 ≤ 1)
    valid_fn = fuzzynum((0.8, 0.6), q=2)

    # This would raise a validation error
    # invalid_fn = fuzzynum((0.9, 0.8), q=2)  # 0.9^2 + 0.8^2 = 1.45 > 1

The factory functions provide immediate feedback on constraint violations, ensuring 
mathematical consistency throughout your fuzzy computations.

This factory-based approach provides a clean, intuitive interface for creating fuzzy 
objects while maintaining the flexibility to work with different fuzzy number types 
and configurations. The next section explores how ``axisfuzzy``'s extension system 
provides even more specialized construction methods for advanced use cases.

Extension Methods for Advanced Construction
-------------------------------------------

Beyond the core factory functions, ``axisfuzzy`` provides a sophisticated extension 
system that enables type-specific construction methods and advanced fuzzy number 
operations. This system is particularly powerful for working with specialized fuzzy 
types like QROFN and QROHFN, offering domain-specific functionality that adapts 
automatically to your fuzzy number types.

Extension System Overview
~~~~~~~~~~~~~~~~~~~~~~~~~

The extension system implements a **Register-Dispatch-Inject** architecture that 
provides polymorphic behavior based on the fuzzy number's mathematical type (``mtype``). 
This allows the same method name to have different implementations for different 
fuzzy types, ensuring mathematical correctness while maintaining a unified API.

Key architectural benefits:

- **Type-aware Polymorphism**: Methods automatically dispatch to type-specific implementations
- **Dynamic Registration**: Extensions can be added at runtime without modifying core code
- **Flexible Injection**: Extensions appear as instance methods, properties, or top-level functions
- **Fallback Support**: Default implementations handle unsupported types gracefully

The extension system is accessed through the ``@extension`` decorator, which registers 
functions for specific fuzzy types and automatically makes them available through 
the appropriate interfaces.

QROFN-Specific Construction Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For q-Rung Orthopair Fuzzy Numbers, ``axisfuzzy`` provides specialized construction 
and manipulation methods that leverage the mathematical properties of this fuzzy type:

.. code-block:: python

    from axisfuzzy import fuzzynum
    from axisfuzzy.extension import extension

    # Create QROFN instances
    qrofn1 = fuzzynum((0.8, 0.3), q=2)
    qrofn2 = fuzzynum((0.6, 0.5), q=2)

    # Use built-in QROFN-specific methods
    distance = qrofn1.distance(qrofn2)  # Euclidean distance for QROFN
    print(f"Distance: {distance:.3f}")  # Output: Distance: 0.361

    # Access QROFN-specific properties
    score = qrofn1.score  # Score function: μ² - ν²
    print(f"Score: {score:.3f}")  # Output: Score: 0.550

    # Complement operation specific to QROFN
    complement = ~qrofn1
    print(complement)  # Output: <0.3,0.8>

QROHFN-Specific Construction Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

q-Rung Orthopair Hesitant Fuzzy Numbers require specialized handling due to their 
multiple membership and non-membership degrees. The extension system provides 
tailored methods for this complexity:

.. code-block:: python

    # Create QROHFN instances
    qrohfn1 = fuzzynum(([0.3, 0.5], [0.4, 0.2]), mtype='qrohfn', q=1)
    qrohfn2 = fuzzynum(([0.6, 0.7], [0.3, 0.2]), mtype='qrohfn', q=1)

    # QROHFN-specific addition operation
    addition = qrohfn1 + qrohfn2
    print(addition)

    # Hesitancy degree calculation
    hesitancy = qrohfn1.ind
    print(f"Hesitancy: {hesitancy:.3f}")

Type Registration and Dynamic Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extension system's power lies in its ability to automatically route method calls 
to the appropriate implementation based on the object's ``mtype``. This happens 
transparently at runtime:

This extension architecture ensures that ``axisfuzzy`` remains both mathematically 
rigorous and highly extensible, allowing researchers to implement domain-specific 
fuzzy operations while maintaining type safety and performance.

High-Performance Computing with Fuzzarray
------------------------------------------

``axisfuzzy``'s ``Fuzzarray`` is engineered for high-performance fuzzy computation, 
leveraging NumPy's vectorized operations and optimized memory layouts to deliver 
exceptional performance for large-scale fuzzy data processing. This section explores 
the architectural decisions and computational strategies that make ``Fuzzarray`` 
suitable for demanding scientific and engineering applications.

NumPy Backend Architecture Advantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Fuzzarray`` implements a **Struct of Arrays (SoA)** architecture that fundamentally 
transforms how fuzzy numbers are stored and processed. Unlike naive approaches that 
store fuzzy numbers as individual objects, ``Fuzzarray`` separates components into 
contiguous NumPy arrays, unlocking significant performance benefits.

**Memory Layout Comparison:**

.. code-block:: python

    # Inefficient: Array of Structs (AoS)
    naive_array = [
        fuzzynum((0.8, 0.1)),
        fuzzynum((0.6, 0.3)),
        fuzzynum((0.7, 0.2))
    ]  # Objects scattered in memory

    # Efficient: Struct of Arrays (SoA) in Fuzzarray
    efficient_array = fuzzyset([
        fuzzynum((0.8, 0.1)),
        fuzzynum((0.6, 0.3)),
        fuzzynum((0.7, 0.2))
    ])
    # Internal storage:
    # mds = np.array([0.8, 0.6, 0.7])    # Contiguous memory
    # nmds = np.array([0.1, 0.3, 0.2])   # Contiguous memory

This architecture provides three critical advantages:

- **Cache Locality**: Related data is stored contiguously, minimizing cache misses
- **SIMD Vectorization**: Enables CPU-level parallel processing of multiple elements
- **Memory Bandwidth**: Reduces memory access overhead through efficient data layout

Performance benefits become dramatic with larger datasets:

.. code-block:: python

    import time
    import numpy as np
    from axisfuzzy import fuzzyset, fuzzynum

    # Create large fuzzy arrays for performance comparison
    size = 100000

    # High-performance creation from raw arrays
    md_values = np.random.uniform(0, 0.8, size)
    nmd_values = np.random.uniform(0, 0.6, size)
    raw_data = np.array([md_values, nmd_values])

    start_time = time.perf_counter()
    large_array = fuzzyset(data=raw_data, mtype='qrofn', q=2)
    elapsed_time = time.perf_counter() - start_time

    print(f"Created array with {size} elements efficiently, time elapsed: {elapsed_time * 1000:.3f} ms")

output::

    Created array with 100000 elements efficiently, time elapsed: 0.434 ms

Vectorized Operations and Batch Computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Fuzzarray`` leverages NumPy's vectorized operations to perform computations on 
entire arrays simultaneously, rather than iterating through individual elements. 
This approach can provide speedups of 10x to 100x over naive implementations.

**Element-wise Operations:**

.. code-block:: python

    import axisfuzzy.random as ar

    # Randomly generate two sets of 10,000 qrofn fuzzy sets
    arr1 = ar.rand(shape=(10000,))
    arr2 = ar.rand(shape=(10000,))

    # Vectorized operations (computed in parallel)
    addition = arr1 + arr2  # All 10000 addition computed at once
    multiplication = arr1 * arr2  # Vectorized multiplication

**Performance Statistics**

.. code-block:: python

    import time
    import axisfuzzy.random as ar

    # Randomly generate two sets of 10,000 qrofn fuzzy sets
    arr1 = ar.rand(shape=(10000,))
    arr2 = ar.rand(shape=(10000,))

    # Vectorized operations (computed in parallel)
    add_start_time = time.perf_counter()
    addition = arr1 + arr2  # All 10000 addition computed at once
    add_elapsed_time = time.perf_counter() - add_start_time

    multi_start_time = time.perf_counter()
    multiplication = arr1 * arr2  # Vectorized multiplication
    multi_elapsed_time = time.perf_counter() - multi_start_time


    print(f"10,000 addition operations time elapsed: {add_elapsed_time * 1000:.3f} ms")
    print(f"10,000 multiplications time elapsed: {multi_elapsed_time * 1000:.3f} ms")

output::

    10,000 addition operations time elapsed: 0.548 ms
    10,000 multiplications time elapsed: 0.537 ms

**Broadcasting and Shape Manipulation:**

.. code-block:: python

    # Broadcasting enables operations between different shapes
    single_fuzzy = fuzzynum((0.5, 0.4), q=2)
    array_fuzzy = fuzzyset([fuzzynum((0.8, 0.1), q=2) for _ in range(100)])
    
    # Broadcast single fuzzy number across entire array
    broadcast_distances = array_fuzzy.distance(single_fuzzy)
    
    # Reshape operations maintain performance
    matrix_array = array_fuzzy.reshape((10, 10))
    column_means = matrix_array.mean(axis=0)

**Batch Processing Workflows:**

.. code-block:: python

    # Process multiple datasets efficiently
    datasets = [
        fuzzyset(data=np.random.uniform(0, 1, (2, 1000)), mtype='qrofn', q=2)
        for _ in range(10)
    ]
    
    # Vectorized analysis across all datasets
    results = []
    for dataset in datasets:
        # Each operation is vectorized internally
        scores = dataset.score  # Property access is vectorized
        mean_score = scores.mean()  # Compute mean of scores
        std_score = scores.std()    # Compute standard deviation of scores
        results.append((mean_score, std_score))

Memory Efficiency and Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Fuzzarray`` implements several optimization strategies to minimize memory usage 
and maximize computational throughput, making it suitable for large-scale applications.

**Memory-Efficient Creation Patterns:**

.. code-block:: python

    # Pre-allocate arrays for maximum efficiency
    large_size = 1000000
    
    # Method 1: Direct raw array creation (fastest)
    md_data = np.random.beta(2, 2, large_size) * 0.8  # Ensure valid range
    nmd_data = np.random.beta(2, 2, large_size) * 0.6
    raw_array = np.array([md_data, nmd_data])
    
    efficient_array = fuzzyset(data=raw_array, mtype='qrofn', q=2)
    
    # Method 2: Empty array with subsequent filling
    empty_array = fuzzyset(fuzzynum((0.5, 0.3), q=2), shape=(large_size,))
    # Fill with vectorized operations...

**Chained Operations and Memory Efficiency:**

.. code-block:: python

    # Efficient chained operations
    large_array = fuzzyset(data=raw_array, mtype='qrofn', q=2)
    
    # Chained operations with minimal memory overhead
    # Note: These operations create new arrays but are optimized internally
    normalized_array = ~large_array
    
    # Reshape operations maintain performance
    result = normalized_array.reshape((1000, 1000))

The combination of SoA architecture, NumPy backend, and vectorized operations makes 
``Fuzzarray`` capable of processing millions of fuzzy numbers efficiently, enabling 
applications in machine learning, decision support systems, and large-scale data 
analysis where performance is critical.

Membership Functions and Fuzzification
--------------------------------------

AxisFuzzy provides a comprehensive membership function library and flexible 
fuzzification system for converting crisp values into fuzzy representations. 
This enables modeling of uncertainty and linguistic variables in real-world applications.

Built-in Membership Function Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library includes a rich collection of standard membership functions, accessible 
through intuitive string aliases for rapid prototyping and development.

**Common Membership Functions:**

.. code-block:: python

    from axisfuzzy.membership import create_mf
    
    # Triangular membership function
    tri_mf, _ = create_mf('trimf', a=0, b=0.5, c=1)
    
    # Gaussian membership function  
    gauss_mf, _ = create_mf('gaussmf', sigma=0.15, c=0.5)
    
    # Trapezoidal membership function
    trap_mf, _ = create_mf('trapmf', a=0, b=0.2, c=0.8, d=1)

**Vectorized Membership Computation:**

.. code-block:: python

    import numpy as np
    
    # Test data for membership evaluation
    x_values = np.linspace(0, 1, 100)
    
    # Compute membership degrees (vectorized)
    tri_degrees = tri_mf(x_values)
    gauss_degrees = gauss_mf(x_values)
    
    # All functions support multi-dimensional arrays
    matrix_input = np.random.uniform(0, 1, (10, 10))
    membership_matrix = gauss_mf(matrix_input)

The Fuzzifier Class
~~~~~~~~~~~~~~~~~~~~

The ``Fuzzifier`` class provides a high-level interface for converting crisp inputs 
into fuzzy numbers, supporting various fuzzification strategies and membership functions.

**Basic Fuzzification:**

.. code-block:: python

    from axisfuzzy.fuzzifier import Fuzzifier
    
    # Create fuzzifier with Gaussian membership function
    fuzzifier = Fuzzifier(
        mf='gaussmf', 
        mf_params={'sigma': 0.1, 'c': 0.5},
        mtype='qrofn',
        q=2
    )
    
    # Convert crisp values to fuzzy numbers
    crisp_data = [0.3, 0.6, 0.9]
    fuzzy_results = fuzzifier(crisp_data)

**Advanced Fuzzification Strategies:**

.. code-block:: python

    # Hesitant fuzzy number fuzzification
    hesitant_fuzzifier = Fuzzifier(
        mf='trimf',
        mf_params={'a': 0, 'b': 0.5, 'c': 1},
        mtype='qrohfn',
        q=3,
        method='default'
    )
    
    # Process arrays efficiently
    score_data = np.array([0.2, 0.7, 0.4, 0.8])
    hesitant_result = hesitant_fuzzifier(score_data)

From Crisp to Fuzzy: Practical Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fuzzification enables modeling of real-world uncertainty and linguistic concepts, 
making it valuable for decision support and data analysis applications.

**Temperature Classification Example:**

.. code-block:: python

    # Define linguistic temperature categories
    cold_mf, _ = create_mf('trimf', a=0, b=0, c=0.3)
    warm_mf, _ = create_mf('trimf', a=0.2, b=0.5, c=0.8) 
    hot_mf, _ = create_mf('trimf', a=0.7, b=1, c=1)
    
    # Normalize temperature readings (0-1 scale)
    temperatures = np.array([0.15, 0.45, 0.75, 0.95])
    
    # Compute membership degrees for each category
    cold_degrees = cold_mf(temperatures)
    warm_degrees = warm_mf(temperatures)
    hot_degrees = hot_mf(temperatures)
    
    # Create fuzzy temperature representations
    temp_fuzzifier = Fuzzifier(mf='trimf', mf_params={'a': 0.2, 'b': 0.5, 'c': 0.8})
    fuzzy_temps = temp_fuzzifier(temperatures)

This fuzzification system seamlessly integrates with AxisFuzzy's computational 
framework, enabling sophisticated fuzzy logic applications while maintaining 
high performance through vectorized operations.

Random Generation and Simulation
--------------------------------

AxisFuzzy's random generation system provides powerful tools for simulation, Monte Carlo 
analysis, and uncertainty modeling. The system ensures reproducibility while offering 
high-performance vectorized operations for large-scale simulations.

Basic Random Generation
~~~~~~~~~~~~~~~~~~~~~~~

Generate random fuzzy numbers using the unified API:

.. code-block:: python

    import axisfuzzy.random as fr
    
    # Set seed for reproducibility
    fr.set_seed(42)
    
    # Generate single random fuzzy number
    single_fuzz = fr.rand(mtype='qrofn', q=2)
    print(f"Random QROFN: {single_fuzz}")
    
    # Generate array of random fuzzy numbers
    fuzz_array = fr.rand(shape=(5, 3), mtype='qrofn', q=2)
    print(f"Random array shape: {fuzz_array.shape}")

Distribution-Based Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create fuzzy numbers from specific probability distributions:

.. code-block:: python

    # Uniform distribution
    uniform_fuzzy = fr.uniform(low=0.2, high=0.8, shape=(100,))
    
    # Normal distribution
    normal_fuzzy = fr.normal(loc=0.5, scale=0.1, shape=(100,))
    
    # Beta distribution for bounded uncertainty
    beta_fuzzy = fr.beta(a=2.0, b=5.0, shape=(100,))
    
    print(f"Generated {len(uniform_fuzzy)} fuzzy numbers from each distribution")

Simulation and Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure reproducible simulations for scientific computing:

.. code-block:: python

    # Reproducible simulation
    def run_simulation(seed=None):
        if seed is not None:
            fr.set_seed(seed)
        
        # Generate test data
        data = fr.rand(shape=(1000,), mtype='qrofn', q=2)
        
        # Perform analysis
        mean_score = data.score.mean()
        return mean_score
    
    # Run reproducible experiments
    result1 = run_simulation(seed=123)
    result2 = run_simulation(seed=123)  # Same result
    result3 = run_simulation(seed=456)  # Different result
    
    print(f"Reproducible: {result1 == result2}")
    print(f"Different seeds: {result1 != result3}")

Independent Random Streams
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create independent generators for parallel processing:

.. code-block:: python

    # Set global seed
    fr.set_seed(42)
    
    # Spawn independent generators
    rng1 = fr.spawn_rng()
    rng2 = fr.spawn_rng()
    
    # Each generator produces independent sequences
    data1 = fr.rand(shape=(100,), rng=rng1)
    data2 = fr.rand(shape=(100,), rng=rng2)
    
    # Verify independence
    correlation = np.corrcoef(data1.score, data2.score)[0, 1]
    print(f"Correlation between streams: {correlation:.4f}")

The random generation system enables sophisticated uncertainty modeling, Monte Carlo 
simulations, and statistical analysis while maintaining scientific reproducibility 
and high computational performance.

Conclusion
----------

This quick start guide has introduced you to the core capabilities of AxisFuzzy, 
a powerful Python library for fuzzy logic and uncertainty modeling. You've learned 
how to work with the fundamental building blocks and leverage the library's key features.

Key Takeaways
~~~~~~~~~~~~~

**Core Data Structures**: AxisFuzzy's ``Fuzznum`` and ``Fuzzarray`` provide intuitive 
interfaces for fuzzy number operations while maintaining high computational performance 
through vectorized operations and optimized memory management.

**Flexible Architecture**: The modular design supports multiple fuzzy number types 
(QROFN, QROHFN, etc.) and extensible components, allowing you to adapt the library 
to your specific research or application needs.

**Production-Ready Features**: From memory-efficient batch processing to reproducible 
random generation, AxisFuzzy provides the tools needed for both research prototyping 
and production deployment.

Next Steps
~~~~~~~~~~

To deepen your understanding and explore advanced features:

- **Explore Fuzzy Types**: Learn about specialized fuzzy number representations 
  in the fuzzy types documentation
- **Advanced Operations**: Discover distance metrics, aggregation functions, and 
  statistical analysis tools in the user guide
- **Custom Extensions**: Create your own membership functions, fuzzification strategies, 
  and random generators using the development guides
- **Performance Optimization**: Learn about memory management, vectorization, and 
  large-scale processing techniques

AxisFuzzy's comprehensive documentation, extensive examples, and active community 
support will guide you through implementing sophisticated fuzzy logic solutions. 
Whether you're conducting academic research, developing decision support systems, 
or building uncertainty-aware applications, AxisFuzzy provides the foundation for 
robust and efficient fuzzy computing.

Welcome to the world of fuzzy logic with AxisFuzzy!