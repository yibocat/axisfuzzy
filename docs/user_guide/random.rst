.. _random:

Random Generation System
========================

The random generation system in ``axisfuzzy`` is a cornerstone for simulation, 
analysis, and testing in fuzzy logic applications. It provides a powerful and 
flexible framework for creating stochastic fuzzy numbers, which is essential for 
modeling uncertainty and variability in real-world systems. Whether you are 
performing Monte Carlo simulations, generating synthetic data for machine learning 
models, or testing the robustness of your fuzzy algorithms, this system offers the 
tools you need for reproducible and statistically sound random generation.

This guide will walk you through the architecture and usage of the random generation 
system, covering everything from basic high-level APIs to the underlying mechanics 
of reproducibility and extensibility. You will learn how to:

- **Ensure reproducibility** of your random experiments using a sophisticated seeding mechanism.
- **Leverage the high-level API** to effortlessly generate various types of random fuzzy numbers.
- **Understand the core components** of the system, including the generator blueprint and the 
  discovery mechanism.
- **Extend the system** by creating and registering your own custom random generators.

By the end of this guide, you will have a comprehensive understanding of how to 
harness the full potential of ``axisfuzzy``'s random generation capabilities.

.. contents::
    :local:

Architectural Deep Dive
-----------------------

The ``axisfuzzy.random`` system is built upon a philosophy of modularity, extensibility, 
and reproducibility. Its architecture is composed of four core components that work 
in concert to provide a powerful yet easy-to-use framework for random fuzzy number 
generation. Understanding this architecture is key to leveraging the system's full 
potential and extending it with custom functionality.

Design Philosophy
~~~~~~~~~~~~~~~~~

- **Modularity & Extensibility**: The system is designed as a "plug-in" architecture. 
  New random generators for different fuzzy membership types (``mtype``) can be defined 
  and registered without altering any core code. This ensures that the system can 
  easily adapt to future fuzzy number representations.
- **Reproducibility**: Scientific computing demands reproducible results. A 
  centralized seed management system ensures that the same seed will always produce 
  the exact same sequence of random fuzzy numbers, which is critical for debugging, 
  validation, and sharing experiments.
- **Unified API**: Despite the complexity of the underlying implementations for 
  different fuzzy types, the system exposes a simple and consistent high-level API 
  (e.g., ``axisfuzzy.random.rand()``). Users can generate any supported fuzzy type 
  through the same interface, significantly lowering the barrier to entry.
- **High Performance**: For large-scale simulations, performance is paramount. The system 
  leverages vectorized operations and interacts directly with the Struct of Arrays (SoA) 
  backend of ``Fuzzarray``, avoiding inefficient Python-level loops and achieving 
  performance close to that of NumPy.

Core Components
~~~~~~~~~~~~~~~

The system's functionality is divided among four key modules:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                        User Layer (e.g., your script)           │
    ├─────────────────────────────────────────────────────────────────┤
    │  fr.rand()  │ fr.choice() │ fr.uniform() │ fr.set_seed() │ ...  │
    └─────────────┬───────────────────────────────────────────────────┘
                  │
    ┌─────────────▼───────────────────────────────────────────────────┐
    │                       API Layer (api.py)                        │
    │  • Unified entry points (rand, choice, etc.)                    │
    │  • Parameter parsing and validation                             │
    │  • Dispatches to the appropriate generator                      │
    └─────────────┬───────────────────────────────────────────────────┘
                  │
        ┌─────────▼─────────┐           ┌──────────────────────┐
        │   Registry Layer  │           │   Seed Management    │
        │  (registry.py)    │           │      (seed.py)       │
        │                   │           │                      │
        │• Manages generator│◄─────────►│• Global random state │
        │  registration     │           │• RNG instantiation   │
        │• Maps `mtype` to  │           │• Spawns independent  │
        │  generator        │           │  streams for parallel│
        └─────────┬─────────┘           └──────────────────────┘
                  │                               │
    ┌─────────────▼───────────────────────────────▼───────────────────┐
    │                  Generator Layer (base.py)                      │
    │                                                                 │
    │  BaseRandomGenerator          ParameterizedRandomGenerator      │
    │  ├─ Defines the common        ├─ _merge_parameters()            │
    │  │  interface for all         ├─ _sample_from_distribution()    │
    │  │  generators                ├─ _validate_range()              │
    │  └─ (fuzznum, fuzzarray)      └─ ... and other utilities        │
    └─────────────┬───────────────────────────────────────────────────┘
                  │
    ┌─────────────▼───────────────────────────────────────────────────┐
    │               Concrete Implementation Layer                     │
    │                                                                 │
    │  QROFNRandomGenerator    │  IVFNRandomGenerator    │  ...       │
    │  ├─ mtype = "qrofn"      │  ├─ mtype = "ivfn"      │            │
    │  ├─ Handles specific     │  ├─ Implements interval-│            │
    │  │  constraints          │  │  based logic         │            │
    └─────────────────────────────────────────────────────────────────┘


1.  **The API Layer** (`api.py`): This is the primary user-facing entry point. 
    It provides the high-level functions like ``rand()`` and ``choice()``. Its 
    role is to interpret the user's request (e.g., ``mtype``, ``shape``, and other parameters) 
    and coordinate the other components to fulfill it.

2.  **The Registry Layer** (`registry.py`): This acts as the system's "dispatch center." 
    It maintains a mapping from an ``mtype`` string (like ``'qrofn'``) to a specific generator 
    instance. When you request a random number of a certain type, the registry looks up and 
    provides the correct generator to the API layer. The ``@register_random`` decorator allows 
    new generators to be added to this registry automatically.

3.  **The Generator Blueprint** (`base.py`): This module defines the "abstract blueprint" that 
    all random generators must follow. The ``BaseRandomGenerator`` class establishes a common 
    interface, requiring every generator to implement methods for creating both single fuzzy 
    numbers (``fuzznum()``) and batches of them (``fuzzarray()``). It also provides the 
    ``ParameterizedRandomGenerator`` subclass, which includes helpful utilities for parameter 
    management and sampling from standard distributions, simplifying the development of new generators.

4.  **The Seed Management Layer** (`seed.py`): This component is the guardian of reproducibility. 
    It manages a global ``numpy.random.Generator`` instance that all random generation tasks draw 
    from. By setting a global seed with ``set_seed()``, you fix the starting point of this random 
    number generator, guaranteeing that every subsequent run of your code will produce identical 
    results. It also supports creating independent random streams for advanced use cases like 
    parallel computing.

A Call's Lifecycle
~~~~~~~~~~~~~~~~~~

To see how these components collaborate, let's trace a typical call:

.. code-block:: python

    import axisfuzzy.random as fr

    # User requests 100 q-Rung Orthopair Fuzzy Numbers
    arr = fr.rand('qrofn', shape=(100,), q=3)

1.  **API Layer**: The ``rand()`` function in ``api.py`` receives the request with 
    ``mtype='qrofn'`` and ``shape=(100,)``.
2.  **Registry Lookup**: The API layer asks the ``registry.py`` to find the generator 
    for ``'qrofn'``. The registry returns an instance of the ``QROFNRandomGenerator``.
3.  **RNG Provision**: The API layer retrieves the global random number generator (RNG) 
    instance from ``seed.py`` to ensure the operation is part of the reproducible sequence.
4.  **Generator Execution**: Since a ``shape`` is specified, the API layer calls the 
    high-performance ``fuzzarray()`` method on the ``QROFNRandomGenerator`` instance. 
    This method uses the provided RNG to perform vectorized sampling, efficiently generates the 
    membership and non-membership degrees according to the ``q-ROFN`` constraints, and directly 
    constructs a ``Fuzzarray`` object for maximum performance.

The Generator Blueprint (base.py)
---------------------------------

At the core of the ``axisfuzzy.random`` system's extensibility is the "generator blueprint" 
defined in ``base.py``. This module establishes a clear and consistent contract that all 
random generators must follow, ensuring that new fuzzy number types can be seamlessly 
integrated into the high-level API. It provides two key abstract base classes: 
:class:`BaseRandomGenerator` and :class:`ParameterizedRandomGenerator`.

BaseRandomGenerator: The Fundamental Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`BaseRandomGenerator` is an abstract base class (ABC) that defines the 
fundamental interface for any random generator. It cannot be instantiated directly. 
Instead, its purpose is to enforce a "contract" that guarantees every generator, 
regardless of the fuzzy number type it produces, will have a consistent structure 
and set of capabilities.

Any class that inherits from :class:`BaseRandomGenerator` must implement the following 
four components:

1.  ``mtype`` Attribute
    A class attribute that declares the fuzzy number type (e.g., ``'qrofn'``, ``'ivfn'``) this 
    generator is responsible for. The registry system uses this string to map a user's request 
    to the correct generator.

    .. code-block:: python

        class QROFNRandomGenerator(BaseRandomGenerator):
            mtype = 'qrofn'

2.  ``get_default_parameters()`` Method
    This method must return a dictionary containing the default parameters for the generation 
    process. These values are used when the user does not explicitly provide them, ensuring 
    predictable behavior.

    .. code-block:: python

        def get_default_parameters(self) -> Dict[str, Any]:
            return {
                'md_dist': 'uniform',
                'md_low': 0.0,
                'md_high': 1.0,
                'nu_mode': 'orthopair'
            }

3.  ``validate_parameters(**params)`` Method
    Before generation, the system calls this method to validate the user-provided and default 
    parameters. It should raise a ``ValueError`` or ``TypeError`` if any parameter is invalid, 
    preventing errors during the generation process. For example, it might check that ``low`` 
    is not greater than ``high``.

4.  ``fuzznum(rng, **params)`` and ``fuzzarray(rng, shape, **params)`` Methods
    These are the core generation methods. The distinction is critical for performance:

    - ``fuzznum()``: Generates a **single** :class:`Fuzznum` instance. Its implementation 
      is typically straightforward and easy to debug.
    
    - ``fuzzarray()``: Generates a batch of fuzzy numbers as a :class:`Fuzzarray` of a given 
      ``shape``. This method is designed for high performance and **must** be implemented using 
      vectorized operations (e.g., with NumPy) to avoid slow Python loops.

This dual-method design allows for both simple, readable logic for single instances and highly optimized, 
scalable logic for large-scale simulations.

ParameterizedRandomGenerator: The Developer's Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While :class:`BaseRandomGenerator` defines the contract, implementing every generator from scratch would 
involve repetitive boilerplate code for common tasks like merging parameters or sampling from statistical 
distributions.

The :class:`ParameterizedRandomGenerator` is a helper class that inherits from :class:`BaseRandomGenerator` 
and provides a powerful toolkit to solve these common problems. It is also an abstract class but comes 
with pre-built utility methods. Developers are strongly encouraged to inherit from it whenever the 
generation logic is based on sampling from standard statistical distributions.

Key utility methods provided by :class:`ParameterizedRandomGenerator`:

1.  ``_merge_parameters(**params)``
    Automatically merges the user-provided parameters with the default parameters returned 
    by ``get_default_parameters()``. User parameters always take precedence. This simplifies 
    parameter management significantly.

2.  ``_validate_range(name, value, min_val, max_val)``
    A convenient helper to check if a given parameter ``value`` falls within a specified 
    ``[min_val, max_val]`` range, raising a ``ValueError`` if it doesn't.

3.  ``_sample_from_distribution(...)``
    This is the most powerful utility. It provides a unified interface to sample from various 
    statistical distributions and automatically scales the output to a desired ``[low, high]`` range. 
    Supported distributions include:

    - ``'uniform'``: Uniform distribution.
    - ``'beta'``: Beta distribution (requires ``a`` and ``b`` shape parameters).
    - ``'normal'``: Normal distribution (requires ``loc`` and ``scale`` parameters). The output is 
      clipped to the ``[low, high]`` range to prevent out-of-bounds values.

    This method abstracts away the complexities of different NumPy sampling functions, allowing 
    developers to focus on the generation logic itself.

Choosing the Right Base Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-   Inherit from **:class:`BaseRandomGenerator`** directly if your generation logic is highly specialized, 
    does not rely on standard statistical distributions, or requires complete control over every step of the process.

-   Inherit from **:class:`ParameterizedRandomGenerator`** (the recommended approach for most cases) 
    if your generation logic involves sampling values from uniform, beta, or normal distributions 
    to construct the fuzzy number's components. This dramatically reduces development time and 
    ensures consistency.


Reproducibility and State Management (seed.py)
----------------------------------------------

In scientific computing and machine learning, **reproducibility** is a fundamental requirement. 
Whether you are validating an algorithm, debugging code, or comparing experimental results, 
the ability to control random processes is essential. The ``seed.py`` module is designed to 
solve this problem by providing a robust and centralized system for managing the random state 
across the entire ``axisfuzzy`` library.

The Core Concept: GlobalRandomState
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the heart of the ``seed.py`` module is the ``GlobalRandomState`` class, which operates 
as a thread-safe singleton. This design ensures that:

- **Single Source of Truth**: The entire library draws random numbers from a single, 
  globally managed ``numpy.random.Generator`` instance.

- **Controllability**: You can set a seed at any point to make subsequent random operations deterministic.

- **Thread Safety**: A built-in ``threading.Lock`` protects the internal state, 
  making it safe to use in multi-threaded applications.

This centralized approach prevents inconsistencies and makes it easy to track and control 
the source of randomness in your applications.


The GlobalRandomState Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

While users typically interact with the random state through the high-level functions described 
below (like ``set_seed()``), the ``GlobalRandomState`` class is the engine that powers the 
entire system. It is implemented as a singleton, meaning a single instance of this class 
manages the random state for the entire application lifecycle.

This class encapsulates a ``numpy.random.Generator`` instance and provides thread-safe methods 
to manage it. Its primary responsibilities are:

- **Initialization**: When first instantiated, it initializes a ``Generator`` with entropy from 
  the operating system, ensuring that initial random numbers are unpredictable until a seed is explicitly set.

- **Seed Management**: It handles the logic for setting and retrieving the seed, and re-creating 
  the generator when the seed is changed via ``set_seed()``.

- **Generator Access**: It provides methods to get the global generator (``get_generator()``) or 
  spawn new, independent generators (``spawn_generator()``).

The class is designed for internal use within the ``axisfuzzy`` library, but understanding its structure can 
be helpful for advanced debugging or extension. The key methods of the class are wrapped and exposed by 
the module-level functions for ease of use.


Core Functions for Random State Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module exposes a simple yet powerful API for interacting with the global random state.

set_seed()
++++++++++

This is the most crucial function for ensuring reproducibility. By calling ``set_seed()`` 
with a specific integer, you initialize the global random number generator to a known state. 
Every time you run your script with the same seed, you will get the exact same sequence of random numbers.

.. code-block:: python

    import axisfuzzy.random as fr

    # Set the global seed to ensure reproducibility
    fr.set_seed(42)

    # All subsequent random generations are now deterministic
    num1 = fr.rand('qrofn', q=2)
    arr1 = fr.rand('ivfn', shape=(5,))

    # Resetting to the same seed will produce the exact same results
    fr.set_seed(42)
    num2 = fr.rand('qrofn', q=2)  # num2 is identical to num1
    arr2 = fr.rand('ivfn', shape=(5,))  # arr2 is identical to arr1

    assert num1 == num2
    assert (arr1 == arr2).all()

get_rng()
+++++++++

For more advanced use cases where you need to perform custom random operations, ``get_rng()`` 
provides direct access to the global ``numpy.random.Generator`` instance. This is useful when you 
need to integrate with other NumPy-based libraries while maintaining a consistent random state.

.. note::
    Using the generator returned by ``get_rng()`` will advance the global random state, affecting 
    all subsequent random operations within ``axisfuzzy``.

.. code-block:: python

    import axisfuzzy.random as fr
    import numpy as np

    # Set the global seed to ensure reproducibility
    fr.set_seed(123)

    # Get the global random number generator
    rng = fr.get_rng()

    # Use it for custom random sampling
    # This advances the global state
    custom_noise = rng.normal(loc=0, scale=0.1, size=10)

    # The next call to fr.rand() will use the advanced state
    fuzzy_num = fr.rand('qrofn', q=2)

spawn_rng()
+++++++++++

This powerful function allows you to create statistically independent random number generators derived 
from the global state. It is essential for parallel computing scenarios where you need to ensure 
that multiple processes or threads are generating random numbers without interfering with each other. 
Each spawned generator manages its own independent state.

.. code-block:: python

    import axisfuzzy.random as fr
    from concurrent.futures import ThreadPoolExecutor

    fr.set_seed(456)

    def worker_task(worker_id):
        # Each worker gets its own independent random stream
        worker_rng = fr.spawn_rng()
        # Generate data using the independent generator
        # This does NOT affect the global state or other workers
        data = worker_rng.uniform(0, 1, size=5)
        return f"Worker {worker_id} data: {data.round(2)}"

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(worker_task, range(2)):
            print(result)

    # The global state remains unaffected by the worker tasks
    global_num = fr.rand('qrofn', q=2)
    print(f"Global random number after parallel tasks: {global_num}")


get_seed()
++++++++++

This utility function allows you to retrieve the seed that was last used to initialize the global random 
state. This is useful for logging, debugging, or storing the configuration of an experiment for later replication.

.. code-block:: python

    import axisfuzzy.random as fr

    fr.set_seed(789)
    current_seed = fr.get_seed()
    print(f"Experiment started with seed: {current_seed}")

    # ... run experiment ...
    results = fr.rand('qrofn', shape=(10,), q=2)

    print(f"Experiment finished. Seed was: {fr.get_seed()}")


The Discovery Mechanism (registry.py)
---------------------------------------

The ``axisfuzzy.random`` system is designed to be extensible, allowing developers to add support for 
new fuzzy number types without modifying the core library. This "plug-in" architecture is powered by 
the discovery and registration mechanism in ``registry.py``. It acts as a central directory, or 
"phone book," that maps each fuzzy number type (``mtype``) to its corresponding random generator.

This system ensures that when you call ``fr.rand('my_new_type')``, the library automatically knows 
where to find and how to use the generator for ``'my_new_type'``.

The Core: RandomGeneratorRegistry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``RandomGeneratorRegistry`` is a thread-safe singleton class that serves as the heart of the 
discovery mechanism. It maintains an internal dictionary mapping ``mtype`` strings to generator instances. 
Its key responsibilities are:

- **Centralized Management**: Provides a single, global source of truth for all available random generators.

- **Automatic Discovery**: Works with the ``@register_random`` decorator to automatically find and register 
  new generators when they are imported.

- **Dynamic Dispatch**: Enables the high-level API to look up and retrieve the correct generator instance at 
  runtime based on the user's request.

Seamless Registration with @register_random
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest and recommended way to add a new generator to the system is by using the ``@register_random`` 
class decorator. When you apply this decorator to a generator class, it automatically performs the 
following steps at module import time:

1.  Reads the ``mtype`` class attribute to identify which fuzzy type the generator handles.
2.  Creates an instance of the generator class.
3.  Registers this instance with the global ``RandomGeneratorRegistry``.

This process is completely transparent and requires no manual intervention.

.. code-block:: python

    from axisfuzzy.random.base import ParameterizedRandomGenerator
    from axisfuzzy.random.registry import register_random

    # The decorator automatically registers this class upon import
    @register_random
    class MyNewTypeGenerator(ParameterizedRandomGenerator):
        # The mtype that this generator is responsible for
        mtype = "my_new_type"

        # --- Implement the required methods ---
        def get_default_parameters(self):
            return {'param': 0.5}

        def validate_parameters(self, **params):
            pass

        def fuzznum(self, rng, **params):
            # ... logic to generate a single fuzzy number
            pass

        def fuzzarray(self, rng, shape, **params):
            # ... logic to generate an array of fuzzy numbers
            pass

Once this module is imported anywhere in your project, the ``'my_new_type'`` generator becomes immediately 
available through the high-level API.

Querying and Using Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``registry.py`` module also provides a set of user-friendly global functions to interact with the registry.

get_random_generator()
++++++++++++++++++++++

This function is the primary way to retrieve a generator instance for a specific ``mtype``. The high-level 
API uses it internally to dispatch requests.

.. code-block:: python

    from axisfuzzy.random.registry import get_random_generator

    # Retrieve the generator for q-Rung Orthopair Fuzzy Numbers
    qrofn_generator = get_random_generator('qrofn')

    if qrofn_generator:
        # You can now use the generator's methods directly
        print(qrofn_generator.get_default_parameters())

list_registered_random() & is_registered_random() & get_registry_random()
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To discover which generators are available or to check for the existence of a specific one, you can 
use these utility functions.

.. code-block:: python

    from axisfuzzy.random.registry import list_registered_random, is_registered_random, get_registry_random

    # List all available random generators
    available_generators = list_registered_random()
    print(f"Available generators: {available_generators}")
    # Expected output might include: ['qrofn', 'ivfn', 'qrohfn', ...]

    # Check if a specific generator is registered
    if is_registered_random('qrofn'):
        print("The 'qrofn' generator is ready to use.")
    else:
        print("The 'qrofn' generator is not available.")

    # Get the global registry instance for advanced operations
    registry = get_registry_random()
    print(f"Registry contains {len(registry._generators)} generators")

This registration and discovery system makes ``axisfuzzy`` highly modular and easy to extend, 
encouraging community contributions and custom adaptations without compromising the stability of the core framework.

The High-Level API (api.py)
---------------------------

The ``api.py`` module serves as the primary user-facing interface for all random generation tasks 
in ``axisfuzzy``. It abstracts the underlying complexity of seed management, generator registration, 
and type-specific logic into a set of powerful yet intuitive functions. This is the recommended entry 
point for most users.

The architecture places this API at the top, providing a unified gateway to the entire random generation subsystem.

Key Functions:

- **rand**: The main factory function for creating random ``Fuzznum`` and ``Fuzzarray`` instances.
- **choice**: A function for random sampling from an existing ``Fuzzarray``, similar to NumPy's equivalent.


The Primary Factory: rand()
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``rand()`` function is the cornerstone of the high-level API. It is a versatile factory that can 
generate both single fuzzy numbers and large, multi-dimensional fuzzy arrays with high performance.

.. code-block:: python

    rand(
        mtype: Optional[str] = None,
        q: int = 1,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
        rng: Optional[np.random.Generator] = None,
        **params
    ) -> Union[Fuzznum, Fuzzarray]:

**Core Parameters:**

- ``mtype``: The type of fuzzy number to generate (e.g., ``'qrofn'``, ``'ivfn'``). If omitted, the library 
  uses the default `mtype` from the global configuration.
- ``q``: The q-rung parameter, a structural property for types like q-Rung Orthopair Fuzzy Numbers.
- ``shape``: Defines the output dimensions. If ``None`` (default), it returns a single ``Fuzznum``. If 
  an ``int`` or ``tuple``, it returns a ``Fuzzarray`` of the specified shape.
- ``seed`` & ``rng``: Control reproducibility. See the note on randomness control below.
- ``**params``: Keyword arguments passed directly to the type-specific generator. These control the statistical 
  properties of the generated numbers, such as the distribution (``md_dist``) or range (``md_low``, ``md_high``).

**Randomness Control Priority**

The API provides a flexible, two-tiered system for managing random state:

1.  **`rng` (Highest Priority)**: Pass an existing ``numpy.random.Generator`` instance for full 
    control. All randomness will be drawn from this generator.
2.  **Global State (Lowest Priority)**: If `rng` is not provided, the function uses the library's global generator, 
    which is managed by ``fr.set_seed()``.

*Reproducible Sampling:*

.. code-block:: python

    # Reproducible sampling with seed
    fr.set_seed(123)
    sample1 = fr.choice(population, size=5)
    fr.set_seed(123)
    sample2 = fr.choice(population, size=5)  # Identical to sample1

**Examples**

*Basic Generation:*

.. code-block:: python

    import axisfuzzy.random as fr

    # Generate a single q-Rung Orthopair Fuzzy Number (q=2)
    num = fr.rand('qrofn', q=2)

    # Generate a 1D array of 50 Interval-Valued Fuzzy Numbers
    arr = fr.rand('ivfn', shape=50)

    # Generate a 10x20 2D array
    arr_2d = fr.rand('qrofn', q=3, shape=(10, 20))

*Advanced Parameter Control:*

This example generates 1,000 fuzzy numbers where the membership degree follows a Beta distribution.

.. code-block:: python

    arr_beta = fr.rand(
        'qrofn',
        q=4,
        shape=(1000,),
        md_dist='beta',       # Membership degree from Beta distribution
        md_low=0.2,           # Clipped to a minimum of 0.2
        md_high=0.8,          # Clipped to a maximum of 0.8
        a=2.0,                # Beta distribution shape parameter `a`
        b=5.0,                # Beta distribution shape parameter `b`
        nu_mode='orthopair'   # Non-membership calculated to satisfy constraint
    )


Random Sampling with choice()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``choice()`` function allows you to draw random samples from an existing 1-dimensional ``Fuzzarray``. 
It supports sampling with or without replacement and allows for weighted probabilities, mirroring the 
functionality of ``numpy.random.choice``.

.. code-block:: python

    choice(
        obj: Fuzzarray,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True,
        p: Optional[Sequence[float]] = None,
        rng: Optional[np.random.Generator] = None
    ) -> Union[Any, Fuzzarray]:

**Core Parameters:**

- ``obj``: The 1D ``Fuzzarray`` to sample from.
- ``size``: The desired shape of the output. If ``None``, returns a single ``Fuzznum``.
- ``replace``: If ``True`` (default), samples are drawn with replacement. If ``False``, they are drawn without replacement.
- ``p``: An array-like sequence of probabilities associated with each element in `obj`.

**Examples**

*Basic Sampling:*

.. code-block:: python

    import axisfuzzy.random as fr

    # 1. Create a source array
    source_arr = fr.rand('qrofn', q=2, shape=(1000,))

    # 2. Sample a single element
    single_sample = fr.choice(source_arr)

    # 3. Sample 50 elements with replacement
    samples_with_replacement = fr.choice(source_arr, size=50)

    # 4. Sample 50 unique elements without replacement
    samples_without_replacement = fr.choice(source_arr, size=50, replace=False)

*Weighted Sampling:*

This is useful when certain fuzzy numbers in the source array are more important than others.

.. code-block:: python

    import numpy as np

    # Assume we have a source array of 100 elements
    source_arr = fr.rand('qrofn', q=2, shape=(100,))

    # Create random weights and normalize them to sum to 1
    weights = np.random.rand(100)
    weights /= weights.sum()

    # Draw a weighted sample of 20 elements
    weighted_sample = fr.choice(source_arr, size=20, p=weights)


Utility Functions
~~~~~~~~~~~~~~~~~

The API also includes helper functions like ``uniform()``, ``normal()``, and ``beta()``, which are convenient 
wrappers around NumPy's random generation functions. They are integrated with AxisFuzzy's seed management 
system, making them useful for generating auxiliary numerical data for your fuzzy logic models.

Additionally, the API provides several utility functions for advanced random generation scenarios:

- **``get_rng()``**: Returns the global random number generator instance
- **``set_seed()``**: Sets the global random seed for reproducible generation
- **``spawn_rng()``**: Creates an independent random generator for parallel operations
- **``get_seed()``**: Retrieves the current global seed value

.. note::

    Currently, these utility functions return standard floating-point numbers (``float`` or ``numpy.ndarray``). 
    In the future, they may be enhanced to generate fuzzy numbers that follow these distributions directly.


Extending the System
--------------------

The ``axisfuzzy.random`` framework is engineered for extensibility. Its plug-in architecture allows you to 
add custom random generators for new or existing fuzzy number types without modifying the core library. 
This section provides a comprehensive, end-to-end tutorial on how to create, register, and use a new 
random generator.

As a practical case study, we will develop an alternative random generator for the existing ``qrofn`` 
(q-Rung Orthopair Fuzzy Number) type. This new generator will have a different default behavior, 
demonstrating how you can tailor generation logic to specific experimental needs.

The Three-Step Process
~~~~~~~~~~~~~~~~~~~~~~

Adding a new generator follows a consistent, three-step pattern that leverages the concepts from 
``base.py`` and ``registry.py``:

1.  **Implement the Generator Class**: Create a Python class that inherits from a suitable base class, 
    typically ``ParameterizedRandomGenerator``. This class will contain all the logic for generating your fuzzy numbers.
2.  **Register the Generator**: Apply the ``@register_random`` decorator to your class. This automatically 
    makes the generator available to the high-level API (e.g., ``fr.rand()``) as soon as the module containing it is imported.
3.  **Ensure Module Import**: Make sure the Python module containing your new generator is imported at 
    an appropriate place in your project, so the registration process is triggered.

End-to-End Example: An Alternative Generator for `qrofn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's imagine we need a ``qrofn`` generator that, by default, produces numbers with membership and 
non-membership degrees clustered around the middle of the range, rather than uniformly distributed. 
We can achieve this by using a Beta distribution as the default.

We will name our new generator ``'qrofn_beta'`` to distinguish it from the standard ``'qrofn'`` generator.

**Step 1 & 2: Implement and Register the Generator**

We create a new Python file (e.g., in your project's utility module) and add the following code. 
This single block of code defines the class and registers it.

.. code-block:: python

    import numpy as np
    from typing import Any, Dict, Tuple

    from axisfuzzy.core import Fuzznum, Fuzzarray
    from axisfuzzy.random.base import ParameterizedRandomGenerator
    from axisfuzzy.random.registry import register_random
    from axisfuzzy.fuzztype.qrofs.backend import QROFNBackend

    @register_random  # The decorator that makes it all work
    class QROFNBetaRandomGenerator(ParameterizedRandomGenerator):
        """
        An alternative random generator for q-Rung Orthopair Fuzzy Numbers
        that defaults to a Beta distribution for more centralized values.
        """
        # The mtype key that users will use to call this generator
        mtype = "qrofn_beta"

        def get_default_parameters(self) -> Dict[str, Any]:
            """
            Define default parameters. Here, we set the distribution to 'beta'.
            """
            return {
                'md_dist': 'beta',      # Default to Beta distribution
                'md_low': 0.0,
                'md_high': 1.0,
                'nu_mode': 'orthopair', # Generate non-membership under the constraint
                'nu_dist': 'uniform',
                'nu_low': 0.0,
                'nu_high': 1.0,
                # Default shape parameters for the Beta distribution
                'a': 2.0,
                'b': 2.0,
                # Default parameters for other distributions (e.g., normal)
                'loc': 0.5,
                'scale': 0.15
            }

        def validate_parameters(self, **params) -> None:
            """
            Validate the structural and distributional parameters.
            """
            # Validate the q-rung parameter
            if 'q' not in params or not isinstance(params['q'], int) or params['q'] <= 0:
                raise ValueError(f"q must be a positive integer, got {params.get('q')}")

            # Use the built-in helper to validate ranges
            self._validate_range('md_low', params.get('md_low', 0.0), 0.0, 1.0)
            self._validate_range('md_high', params.get('md_high', 1.0), 0.0, 1.0)
            if params.get('md_low', 0.0) > params.get('md_high', 1.0):
                raise ValueError("md_low cannot be greater than md_high")

        def fuzznum(self, rng: np.random.Generator, **params) -> 'Fuzznum':
            """
            Generate a single 'qrofn_beta' fuzzy number.
            """
            # 1. Merge user-provided parameters with our defaults
            p = self._merge_parameters(**params)
            self.validate_parameters(**p)

            # 2. Generate membership degree (md) using our sampling utility
            md = self._sample_from_distribution(
                rng, dist=p['md_dist'], low=p['md_low'], high=p['md_high'],
                a=p['a'], b=p['b'], loc=p['loc'], scale=p['scale']
            )

            # 3. Generate non-membership degree (nmd) based on the q-ROFN constraint
            max_nmd = (1 - md**p['q'])**(1 / p['q'])
            nmd = rng.uniform(0, max_nmd) # Sample uniformly within the valid range

            # 4. Create the Fuzznum instance
            return Fuzznum(mtype=self.mtype, q=p['q']).create(md=md, nmd=nmd)

        def fuzzarray(self, rng: np.random.Generator, shape: Tuple[int, ...], **params) -> 'Fuzzarray':
            """
            High-performance batch generation of 'qrofn_beta' fuzzy arrays.
            """
            # 1. Merge and validate parameters
            p = self._merge_parameters(**params)
            self.validate_parameters(**p)
            size = int(np.prod(shape))

            # 2. Vectorized generation of membership degrees
            mds = self._sample_from_distribution(
                rng, size=size, dist=p['md_dist'], low=p['md_low'], high=p['md_high'],
                a=p['a'], b=p['b'], loc=p['loc'], scale=p['scale']
            )

            # 3. Vectorized generation of non-membership degrees
            max_nmds = (1 - mds**p['q'])**(1 / p['q'])
            # Efficiently sample within the constraint for the whole array
            nmds = rng.uniform(0, 1, size=size) * max_nmds

            # 4. Directly construct the backend for maximum performance
            backend = QROFNBackend.from_arrays(
                mds=mds.reshape(shape),
                nmds=nmds.reshape(shape),
                q=p['q']
            )
            return Fuzzarray(backend=backend)

**Step 3: Use the New Generator**

Now, as long as the Python module containing ``AlternativeQROFNGenerator`` is imported, you can use it 
directly with ``fr.rand()`` by specifying ``mtype='qrofn_beta'``.

.. code-block:: python

    import axisfuzzy.random as fr
    # Make sure the module with AlternativeQROFNGenerator is imported, e.g.:
    # from .my_custom_generators import QROFNBetaRandomGenerator

    # Generate a 100-element array using our new generator
    # It will use a Beta(2,2) distribution by default
    beta_arr = fr.rand('qrofn_beta', q=3, shape=100)

    # We can still override the defaults, just like with any other generator
    uniform_arr = fr.rand('qrofn_beta', q=3, shape=100, md_dist='uniform')

    # Check which generators are available
    print(fr.list_registered())
    # Expected output will now include: ['qrofn', 'ivfn', ..., 'qrofn_beta']

Key Takeaways from the Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Inherit from `ParameterizedRandomGenerator`**: This provides powerful helpers like ``_merge_parameters`` 
  and ``_sample_from_distribution``, saving you from writing boilerplate code.
- **Set the `mtype` Attribute**: This unique string is the key that links your generator to the ``fr.rand()`` function.
- **Implement the Four Core Methods**: ``get_default_parameters``, ``validate_parameters``, ``fuzznum``, 
  and ``fuzzarray``.
- **Vectorize `fuzzarray` for Performance**: The most critical performance aspect is to use vectorized 
  NumPy operations for batch generation and construct the backend directly, avoiding slow Python loops.
- **The `@register_random` Decorator Does the Magic**: This simple decorator handles all the complexity 
  of making your generator discoverable by the rest of the library.


Conclusion
----------

This guide has provided a comprehensive tour of the ``axisfuzzy.random`` system, a powerful and 
flexible framework for generating random fuzzy numbers. We have explored its modular architecture, 
designed for extensibility and performance, and delved into the core components that ensure 
reproducibility and ease of use.

You have learned how to:

- **Leverage the high-level API**: Use functions like ``fr.rand()`` and ``fr.choice()`` to 
  effortlessly generate various types of random fuzzy numbers.
- **Ensure Reproducibility**: Control the random state with ``fr.set_seed()`` to guarantee that your 
  experiments are deterministic and verifiable.
- **Understand the Architecture**: Appreciate how the API, Registry, Generator Blueprint, and Seed Management 
  layers work together to provide a seamless experience.
- **Extend the System**: Create and register your own custom random generators by inheriting from 
  ``ParameterizedRandomGenerator`` and using the ``@register_random`` decorator.

By mastering these concepts, you are now equipped to harness the full potential of ``axisfuzzy`` for a 
wide range of applications, from sophisticated Monte Carlo simulations to robust algorithm testing. 
The system's design not only meets current needs but also provides a solid foundation for future extensions, 
ensuring that ``axisfuzzy`` remains at the forefront of fuzzy logic research and development.


