.. _user_guide_core_data_structures:

Core Data Structures: Fuzznum and Fuzzarray
===========================================

The heart of the ``axisfuzzy`` library lies in its two fundamental data 
structures: ``Fuzznum`` and ``Fuzzarray``. These components provide the foundation 
for representing and manipulating fuzzy numbers, offering both an intuitive, 
object-oriented interface and a high-performance, vectorized computation engine. 
Understanding their core concepts is key to leveraging the full power of ``axisfuzzy``.

This guide provides an in-depth exploration of ``Fuzznum`` and ``Fuzzarray``, 
detailing their architecture, their symbiotic relationship, and the design 
patterns that enable their efficiency and extensibility.

.. contents::
   :local:

The Core Concepts
--------------------------

The architecture of ``axisfuzzy.core`` is built on a crucial principle: the **Separation of Concerns**. 
The user-facing API (the "what") is cleanly decoupled from the underlying implementation (the "how").

- **User-Facing Layer**: ``Fuzznum`` and ``Fuzzarray`` provide a clean, 
  pythonic API for users. They are designed to feel familiar, behaving much like 
  Python's built-in numeric types or NumPy's ``ndarray``.
- **Implementation Layer**: ``FuzznumStrategy`` and ``FuzzarrayBackend`` 
  contain the specific logic for each fuzzy number type. They handle data storage, 
  validation, and the mathematical heavy lifting.

This separation allows researchers and developers to work with fuzzy numbers at a 
high level of abstraction, without needing to worry about the complex implementation 
details that ensure performance and correctness.

Fuzznum: The Atomic Unit of Fuzziness
-------------------------------------

A ``Fuzznum`` represents a single, individual fuzzy number. It serves as the primary interface 
for interacting with fuzzy scalars. However, its design is more sophisticated than a simple 
data container; it acts as a **Facade** and a **Dynamic Proxy**.

The Facade and Strategy Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Fuzznum`` object itself is lightweight and holds no fuzzy data (like membership or 
non-membership degrees). Instead, it holds a reference to a **strategy** object—an 
instance of a class that inherits from ``FuzznumStrategy``.

- ``Fuzznum`` (The Facade): Provides a simple, unified interface to the user. 
  When you create a fuzzy number, you are interacting with a ``Fuzznum`` instance. 
  The underlying interaction is actually a ``FuzznumStrategy``, which contains 
  complex validation logic and modification callbacks.

- ``FuzznumStrategy`` (The Strategy): Defines the "how" for a specific type of fuzzy number. 
  It is an abstract base class that establishes a contract for all concrete fuzzy number 
  implementations. For each fuzzy number type (e.g., q-Rung Orthopair Fuzzy Number, or 'qrofn'), 
  there is a corresponding strategy class (e.g., ``QROFNStrategy``) that is responsible for:
    
 - **Data Storage**: Defining the attributes that constitute the fuzzy number (e.g., ``md``, ``nmd``, ``q``).
 - **Validation**: Ensuring that any data assigned to the attributes is valid (e.g., membership degrees are between 0 and 1).
 - **Constraint Enforcement**: Applying the mathematical rules of the fuzzy 
   set theory (e.g., for a 'qrofn', ensuring that ``md**q + nmd**q <= 1``).

This combination of Facade and Strategy patterns means that the ``Fuzznum`` object 
you interact with is just an elegant "shell" that delegates all real work to its 
internal strategy object.

The Dynamic Proxy in Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The magic that connects the ``Fuzznum`` facade to its ``FuzznumStrategy`` is implemented 
using Python's special methods ``__getattr__`` and ``__setattr__``. This makes ``Fuzznum`` a dynamic proxy.

- When you access an attribute, like ``my_fuzznum.md``, the ``Fuzznum`` 
  object's ``__getattr__`` method intercepts the call and forwards it to its 
  internal strategy: ``self._strategy.md``.
- When you set an attribute, like ``my_fuzznum.md = 0.9``, the ``Fuzznum`` 
  object's ``__setattr__`` method forwards the assignment to the strategy: 
  ``self._strategy.md = 0.9``. The strategy's own ``__setattr__`` then triggers 
  its validation and constraint-checking logic.

To the user, it feels like ``Fuzznum`` has properties like ``md`` and ``nmd``, 
but in reality, all state and logic are managed by the strategy. This provides a 
consistent user experience regardless of the complexity of the underlying fuzzy number type.

Creating a Fuzznum
~~~~~~~~~~~~~~~~~~

The recommended way to create a ``Fuzznum`` is via the ``axisfuzzy.fuzzynum`` factory function. 
This function looks up the appropriate ``FuzznumStrategy`` from the central registry 
based on the ``mtype`` (membership type) you provide, instantiates it with your data, 
and wraps it in a ``Fuzznum`` object.

The ``axisfuzzy.fuzzynum`` factory function provides two ways to create fuzzy numbers: 
one is to pass membership values in order within a tuple, and the other is to use keyword arguments.

- **Tuple-Based Creation**: You can pass membership values in order as a tuple. 
  The factory will automatically detect the membership type based on the number of arguments.
- **Keyword Argument Creation**: You can pass membership values as keyword arguments. 
  The factory will match the arguments to the expected parameters of the strategy.

.. note::

    Although a fuzzy number can be created through a class instance 
    like ``Fuzznum(mtype='qrofn', q=1).create(md=0.5, nmd=0.2)``, the ``axisfuzzy.fuzzynum`` factory 
    function is the recommended approach for creating ``Fuzznum`` objects. It ensures 
    that the correct ``FuzznumStrategy`` is instantiated and wrapped, and it handles validation and constraint checking logic.

Suppose we want to create a q-ROFN fuzzy number with ``md=0.8``, ``nmd=0.1``, 
and ``q=3``; we have three methods to create this fuzzy number.

.. code-block:: python

   from axisfuzzy.core import fuzzynum, Fuzznum

   # Create a q-Rung Orthopair Fuzzy Number (q-ROFN) with q=3
   # The factory finds the 'qrofn' strategy, instantiates it,
   # and wraps it in a Fuzznum object.

   # Method One
   my_fuzznum = Fuzznum(mtype='qrofn', q=3).create(md=0.8, nmd=0.1)

   # Method Two(Recommended)
   my_fuzznum = fuzzynum((0.8,0.1), q=3)
   
   # Method Three(Recommended)
   my_fuzznum = fuzzynum(md=0.8, nmd=0.1, q=3)

   # Accessing .md is proxied to the underlying QROFNStrategy
   print(my_fuzznum.md)
   # >>> 0.8

   # Setting .md triggers validation and constraints in the strategy
   try:
       # This will fail the validation rule (must be <= 1)
       my_fuzznum.md = 1.1
   except ValueError as e:
       print(e)

The same applies to the creation of other types of fuzzy numbers, such as q-ROHFN. 
Suppose we want to create a q-ROHFN with ``md=[0.8,0.6]``, ``nmd=[0.1]``, and ``q=3``.

.. code-block:: python

   from axisfuzzy.core import fuzzynum, Fuzznum

   # Create a q-Rung Orthopair Fuzzy Number (q-ROFN) with q=3
   # The factory finds the 'qrofn' strategy, instantiates it,
   # and wraps it in a Fuzznum object.

   # Method One
   my_fuzznum = Fuzznum(mtype='qrohfn', q=3).create(md=[0.8, 0.6], nmd=[0.1])

   # Method Two(Recommended)
   my_fuzznum = fuzzynum(([0.8, 0.6], [0.1]), mtype='qrohfn', q=3)
   
   # Method Three(Recommended)
   my_fuzznum = fuzzynum(md=[0.8, 0.6], nmd=[0.1], mtype='qrohfn' q=3)

   # Accessing .md is proxied to the underlying QROFNStrategy
   print(my_fuzznum.md)
   # >>> [0.8, 0.6]

   # Setting .md triggers validation and constraints in the strategy
   try:
       # This will fail the validation rule (must be <= 1)
       my_fuzznum.md = [1.1, 0.9]
   except ValueError as e:
       print(e)

.. note::

    When default ``mtype`` and ``q`` values are set, the ``mtype`` and ``q`` parameters 
    can be omitted when creating a ``Fuzznum``. Otherwise, ``mtype`` and ``q`` must be specified.

    .. code-block:: python

        my_fuzznum = fuzzynum((0.8, 0.1))
        # or
        my_fuzznum = fuzzynum(md=0.8, nmd=0.1)

Fuzzarray: High-Performance Fuzzy Computation Container
-------------------------------------------------------

While ``Fuzznum`` is the atomic unit, most real-world applications require 
computations on large collections of fuzzy numbers. This is the role of ``Fuzzarray``, 
a high-performance container designed to be the fuzzy equivalent of NumPy's ``ndarray``.

The Performance Dilemma: AoS vs. SoA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A naive approach to creating a fuzzy array would be to use a standard Python 
list or a NumPy array of ``Fuzznum`` objects. This is known as an **Array of 
Structs (AoS)** architecture.

.. code-block:: python

   # Array of Structs (AoS) - Intuitive but inefficient
   aos_array = [
       Fuzznum(md=0.8, nmd=0.1),
       Fuzznum(md=0.6, nmd=0.3),
       # ... many more objects
   ]

This approach is a performance disaster for numerical computing:

1.  **Memory Fragmentation**: Each ``Fuzznum`` is a separate Python object, 
    scattered across different locations in memory.
2.  **Poor Cache Locality**: When performing a vectorized operation (e.g., summing all ``md`` values), 
    the CPU must jump around in memory to access the data for each object, leading to frequent cache misses.
3.  **No SIMD Vectorization**: Modern CPUs rely on Single Instruction, Multiple Data (SIMD) 
    operations to perform calculations on contiguous blocks of data in parallel. The AoS layout makes this impossible.

``axisfuzzy`` solves this by adopting a **Struct of Arrays (SoA)** architecture, implemented by the ``FuzzarrayBackend``.

FuzzarrayBackend and the SoA Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Fuzzarray`` object, much like a ``Fuzznum``, is also a facade. It delegates all 
data storage and computation to an internal backend object, which is an instance of a ``FuzzarrayBackend`` subclass.

The backend implements the SoA pattern. Instead of one array of objects, 
it is an object containing multiple arrays. Each array stores a single component 
of all elements in the collection.

For a ``Fuzzarray`` of 'qrofn' numbers, the ``QROFNBackend`` would look like this conceptually:

.. code-block:: python

   # Struct of Arrays (SoA) - The key to performance
   class QROFNBackend:
       # All membership degrees are stored in one contiguous NumPy array
       mds: np.ndarray = np.array([0.8, 0.6, 0.7, ...])

       # All non-membership degrees are in another contiguous array
       nmds: np.ndarray = np.array([0.1, 0.3, 0.2, ...])

       # The 'q' parameter is stored once
       q: int = 3

The advantages of SoA are immense:

- **Memory Locality**: All values for a given component (e.g., ``mds``) are 
  packed together in a contiguous memory block.
- **Cache Efficiency**: When a computation needs all ``md`` values, the CPU 
  can load the entire ``mds`` array into its cache, dramatically speeding up access.
- **Vectorization**: This layout is exactly what NumPy is designed for. 
  Operations on the arrays can be translated to highly optimized, 
  low-level C or Fortran code that leverages SIMD instructions.

Creating a Fuzzarray: The Three Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Fuzzarray`` constructor is designed with three distinct initialization paths, 
balancing user convenience with internal performance. These paths are optimized for 
different use cases: direct backend assignment for maximum performance, high-performance 
raw array creation for efficient data processing, and user-friendly creation from 
Fuzznum objects for convenience.

Path 1: The User-Friendly Path - Creating from Fuzznum Objects
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is the most common way for a user to create a ``Fuzzarray``. The ``axisfuzzy.fuzzyset`` 
factory function (which is an alias for the ``Fuzzarray`` class) is the primary entry point 
for this path. You pass it an array-like object, such as a list of ``Fuzznum`` objects.

.. code-block:: python

   from axisfuzzy import fuzzyset, fuzzynum

   # Create a Fuzzarray from a list of Fuzznum objects
   arr = fuzzyset([
       fuzzynum((0.8, 0.1), q=2),
       fuzzynum((0.6, 0.3), q=2)
   ])

   print(arr)
   # >>> Fuzzarray([<0.8,0.1> <0.6,0.3>], mtype='qrofn', q=2, shape=(2,))

   # You can also create an empty array and fill it
   arr = fuzzyset(fuzzynum((0.6, 0.3), q=2), shape=(1000,))

When you use this path, the constructor performs several steps:

1.  **Infers Parameters**: It inspects the input data to determine the ``mtype`` and ``q``.
2.  **Creates Backend**: It looks up the appropriate backend class (e.g., ``QROFNBackend``) 
    from the registry and instantiates it with the correct shape.
3.  **Populates Data**: It iterates through the input data, taking each ``Fuzznum``, 
    and "scatters" its components into the correct SoA NumPy arrays within the backend.

This path is convenient but involves overhead due to data inspection and iteration. 
It is ideal for initial array creation from user data.

Path 2: High-Performance Raw Array Creation
++++++++++++++++++++++++++++++++++++++++++++

The ``fuzzyset`` factory function provides a highly optimized path for creating 
``Fuzzarray`` objects directly from raw NumPy arrays or nested lists. This path 
is designed for scenarios where you have structured component data (e.g., membership 
and non-membership degrees) and want to bypass the overhead of individual ``Fuzznum`` 
object creation.

.. code-block:: python

   import numpy as np
   from axisfuzzy import fuzzyset

   # Create QROFN array from raw component arrays
   # First array: membership degrees, Second array: non-membership degrees
   md_values = np.array([0.8, 0.6, 0.7])
   nmd_values = np.array([0.1, 0.3, 0.2])
   raw_data = np.array([md_values, nmd_values])  # Shape: (2, 3)
   
   # High-performance creation (Path 2)
   arr = fuzzyset(data=raw_data, mtype='qrofn', q=2)
   print(arr)
   # >>> Fuzzarray([<0.8,0.1> <0.6,0.3> <0.7,0.2>], mtype='qrofn', q=2, shape=(3,))

   # For QROHFN with hesitant values
   md_hesitant = np.array([[0.2,0.4], [0.5,0.2], [0.7,0.8,0.9]], dtype=object)
   nmd_hesitant = np.array([[0.1], [0.1,0.2], [0.1, 0.05]], dtype=object)
   hesitant_data = np.array([md_hesitant, nmd_hesitant]) # Shape: (2, 3)
   
   arr_hesitant = fuzzyset(data=hesitant_data, mtype='qrohfn', q=2)
   print(arr_hesitant)
   # >>> Fuzzarray([<[0.2, 0.4],[0.1]> <[0.2, 0.5],[0.1, 0.2]> <[0.7, 0.8, 0.9],[0.05, 0.1]>], 
   #                mtype='qrohfn', q=2, shape=(3,))

This path offers several advantages:

- **Maximum Performance**: Bypasses individual ``Fuzznum`` object creation and directly 
  constructs the backend from raw arrays.
- **Memory Efficiency**: No intermediate object allocation, direct array-to-backend transfer.
- **Batch Processing**: Ideal for processing large datasets or results from vectorized operations.
- **Type Safety**: Automatic validation ensures the raw data conforms to the fuzzy type constraints.

The ``fuzzyset`` function intelligently detects when the input data represents raw 
component arrays versus a collection of ``Fuzznum`` objects, automatically selecting 
the appropriate creation path for optimal performance.

.. note::
    When creating a ``Fuzzarray`` using Path 2, the array construction is critical. 
    The shape of the numpy.ndarray passed to ``fuzzyset`` must comply with the backend 
    contracts ``cmpnum``, ``cmpnames``, and ``dtype``. This means that ``data.shape[0]`` 
    must equal ``cmpnum``, representing the names of membership degrees defined in ``cmpnames``.
    The ``dtype`` specifies the array's data type - for special fuzzy sets like ``qrohfn``, 
    it must be set to object to ensure correct representation. 
    Notably, this method works with high-dimensional arrays as long as they satisfy the Backend's established contracts.

Path 3: The High-Performance Path - Creating from a Backend
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

When performance is critical, especially during internal computations, 
``axisfuzzy`` uses a much faster method. A ``Fuzzarray`` can be instantiated directly 
from a pre-constructed ``FuzzarrayBackend`` object.

.. code-block:: python

   # This is a conceptual example. In practice, `new_backend` would be
   # the result of a vectorized operation.
   import numpy as np

   from axisfuzzy.core import Fuzzarray
   from axisfuzzy.fuzztype.qrofs import QROFNBackend

   # 1. Assume a vectorized operation produced these new data arrays
   new_mds = np.array([0.9, 0.7])
   new_nmds = np.array([0.05, 0.25])

   # 2. Create a new backend instance directly from these arrays (very fast)
   new_backend = QROFNBackend.from_arrays(mds=new_mds, nmds=new_nmds, q=2)

   # 3. Create the final Fuzzarray by passing the backend (extremely fast, O(1))
   #    This is the "fast path".
   result_array = Fuzzarray(backend=new_backend)

   print(result_array)
   # >>> [<0.9,0.05>, <0.7,0.25>]

This "fast path" is the key to ``axisfuzzy``'s performance. When the constructor 
receives a ``backend`` argument, it skips all data processing and simply assigns the 
provided backend to its internal data structure. This is an O(1) operation with almost zero overhead.

As we will see in the next section, this path is crucial for completing the 
high-performance computation loop, allowing the results of one vectorized 
operation to be seamlessly and efficiently fed into the next.

FuzznumStrategy: The Brains Behind Individual Fuzzy Numbers
-----------------------------------------------------------

The ``FuzznumStrategy`` is the intelligent core that governs the behavior of every 
individual fuzzy number. It's an abstract base class that acts as a blueprint, 
defining the data structure, validation logic, and operational capabilities for 
a specific fuzzy number type. Think of it as the "strategy" in the Strategy Design 
Pattern, where each concrete fuzzy number type (like QROFN or QROHFN) implements 
this strategy to manage its own unique logic.

Core Responsibilities
~~~~~~~~~~~~~~~~~~~~~

A ``FuzznumStrategy`` subclass is responsible for:

1.  **Attribute Declaration**: It declaratively defines the components of 
    a fuzzy number (e.g., ``md`` for membership, ``nmd`` for non-membership). 
    This is done simply by defining class attributes or using type hints, 
    which are then automatically collected by the base class.
2.  **Validation Lifecycle Management**: This is the most critical role. 
    The strategy implements a sophisticated, multi-stage validation process to 
    guarantee the mathematical integrity of the fuzzy number at all times.
3.  **Operation Dispatch**: It serves as the entry point for operations 
    (like addition or comparison), delegating the actual computation to the 
    appropriate registered function.

The Validation Lifecycle: A Three-Stage Guardian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure robustness, ``FuzznumStrategy`` provides a powerful three-stage lifecycle 
for attribute assignment. When you attempt to set a value (e.g., ``my_fuzznum.md = 0.9``), 
the following sequence is triggered automatically:

1.  **Validator**: The first line of defense. This is a simple function that 
    performs a stateless, atomic check on the new value. For example, it ensures a 
    membership degree is a number between 0 and 1. If the validator returns ``False``, 
    the assignment is immediately rejected with a ``ValueError``.

    *   **Purpose**: Fast, simple, context-free checks.
    *   **Example**: ``self.add_attribute_validator('md', lambda x: 0 <= x <= 1)``

2.  **Transformer**: If the validator passes, the transformer is executed. 
    This function can modify the incoming value, normalizing or converting it to the required 
    internal format. For instance, it might convert an input list into a sorted NumPy array. 
    The transformed value is then used for the actual assignment.

    *   **Purpose**: Data normalization and type conversion.
    *   **Example**: ``self.add_attribute_transformer('md', lambda x: np.asarray(x, dtype=float))``

3.  **Change Callback**: The final stage, executed *after* the new value has been assigned. 
    This function is used for complex, stateful validation that may involve multiple attributes. 
    For example, after ``md`` is updated, a callback checks if the core constraint (e.g., ``md**q + nmd**q <= 1``) 
    is still satisfied. If the constraint is violated, it can raise an exception to effectively "undo" the change.

    *   **Purpose**: Complex, multi-attribute, stateful constraint checking.
    *   **Example**: ``self.add_change_callback('md', self._check_q_rung_constraint)``

This lifecycle, demonstrated in the ``QROFNStrategy``, ensures that a ``Fuzznum`` is always 
in a valid state, providing exceptional data integrity.

.. code-block:: python

   # axisfuzzy/fuzztype/qrofs/qrofn.py
   @register_strategy
   class QROFNStrategy(FuzznumStrategy):
       mtype = 'qrofn'
       md: Optional[float] = None
       nmd: Optional[float] = None

       def __init__(self, q: Optional[int] = None):
           super().__init__(q=q)
           
           # 1. Validator: Is the value between 0 and 1?
           self.add_attribute_validator('md', lambda x: x is None or 0 <= x <= 1)
           self.add_attribute_validator('nmd', lambda x: x is None or 0 <= x <= 1)

           # 2. Transformer: (Not needed for this simple type)

           # 3. Change Callback: Does the new value satisfy the q-rung constraint?
           self.add_change_callback('md', self._on_membership_change)
           self.add_change_callback('nmd', self._on_membership_change)

       def _fuzz_constraint(self):
           """Enforce the q-rung orthopair constraint: md^q + nmd^q <= 1"""
           if self.md is not None and self.nmd is not None and self.q is not None:
               if self.md ** self.q + self.nmd ** self.q > 1.0:
                   raise ValueError("q-rung constraint violated")
       
       def _on_membership_change(self, attr_name, old_value, new_value):
           self._fuzz_constraint()


FuzzarrayBackend: The High-Performance Engine for Fuzzy Arrays
----------------------------------------------------------------

While ``FuzznumStrategy`` manages individual numbers, ``FuzzarrayBackend`` is the powerhouse that 
enables high-speed computations on entire arrays of them. It is an abstract base class that 
mandates a **Struct-of-Arrays (SoA)** architecture, a design choice that is fundamental to AxisFuzzy's performance.

The backend is central to all three ``Fuzzarray`` creation paths:

- **Path 1** (Fuzznum Creation): Traditional creation from ``Fuzznum`` objects, where the backend 
  is populated through repeated calls to ``set_fuzznum_data``.
- **Path 2** (Raw Array Creation): The factory function ``fuzzyset`` provides a highly optimized path for creating 
  ``Fuzzarray`` objects directly from raw NumPy arrays, bypassing individual ``Fuzznum`` object creation for maximum efficiency.
- **Path 3** (Backend Creation): Directly instantiates a ``Fuzzarray`` from a pre-constructed backend, 
  offering O(1) performance for internal operations.

This architecture ensures that regardless of the creation path, all ``Fuzzarray`` objects benefit 
from the same high-performance SoA data layout and vectorized operations.

The SoA Architecture
~~~~~~~~~~~~~~~~~~~~

Instead of storing an array of ``Fuzznum`` objects (Array-of-Structs), which leads to 
scattered memory and poor performance, the SoA architecture stores each component of the 
fuzzy numbers in its own contiguous NumPy array.

-   **AoS (Slow)**: ``[Fuzznum(md=0.8, nmd=0.1), Fuzznum(md=0.7, nmd=0.2)]``
-   **SoA (Fast)**: ``mds = [0.8, 0.7]``, ``nmds = [0.1, 0.2]``

This layout is cache-friendly and allows NumPy's underlying C/Fortran code to leverage 
SIMD (Single Instruction, Multiple Data) instructions for massive parallelization.

Core Abstract Methods
~~~~~~~~~~~~~~~~~~~~~

Every backend must implement a set of abstract methods that define its interaction with 
the ``Fuzzarray`` container. These methods are the bridge between the high-level, 
user-friendly array and the low-level, high-performance data store.

-   ``_initialize_arrays(self)``: This is where the backend creates its component arrays. 
    The data type of these arrays is crucial.
    
    - For scalar components like in ``QROFNBackend``, 
      it creates float arrays: ``self.mds = np.zeros(self.shape, dtype=np.float64)``.
    - For set-based components like in ``QROHFNBackend``, it must use object arrays to 
      hold other arrays: ``self.mds = np.empty(self.shape, dtype=object)``.

-   ``get_fuzznum_view(self, index)`` : Extracts data from the SoA arrays at a given 
    ``index`` and reconstructs it into a single ``Fuzznum`` object for the user to inspect. 
    This is a "view" and should be a lightweight operation.

-   ``set_fuzznum_data(self, index, fuzzynum)`` : The reverse of ``get_fuzznum_view``. 
    It deconstructs a ``Fuzznum`` object and writes its components into the correct 
    positions in the backend's SoA arrays.

-   ``copy(self)``: Creates a deep copy of the backend, ensuring that the new instance has 
    its own separate data arrays. This is vital for immutability and preventing unintended side effects.

-   ``slice_view(self, key)``: A performance-critical method that returns a new backend 
    representing a slice of the original. Crucially, this should be a *view* (sharing memory 
    with the original) whenever possible to avoid costly data duplication, which is the secret 
    to ``Fuzzarray``'s fast slicing.

-   ``from_arrays(*components, **kwargs)``: A factory class method that efficiently constructs 
    a new backend instance directly from a set of component arrays. This is the "fast path" 
    used internally after a vectorized operation computes new result arrays.

Implementation Examples
~~~~~~~~~~~~~~~~~~~~~~~

The difference in implementing these methods for scalar vs. set-based fuzzy numbers is illustrative.
Each backend must implement essential contract properties and methods for proper integration.

**QROFNBackend (Scalar Components)**

.. code-block:: python

   # axisfuzzy/fuzztype/qrofs/backend.py
   @register_backend
   class QROFNBackend(FuzzarrayBackend):
       mtype = 'qrofn'

       @property
       def cmpnum(self) -> int:
           return 2  # Two components: md and nmd

       @property
       def cmpnames(self) -> Tuple[str, ...]:
           return 'md', 'nmd'  # Component names

       @property
       def dtype(self) -> np.dtype:
           return np.dtype(np.float64)  # Scalar values

       def _initialize_arrays(self):
           self.mds = np.zeros(self.shape, dtype=np.float64)
           self.nmds = np.zeros(self.shape, dtype=np.float64)

       def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
           md_value = float(self.mds[index])
           nmd_value = float(self.nmds[index])
           return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

       def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
           self.mds[index] = fuzznum.md
           self.nmds[index] = fuzznum.nmd

       @classmethod
       def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int, **kwargs):
           """Create backend directly from component arrays with validation."""
           cls._validate_fuzzy_constraints_static(mds, nmds, q=q)
           backend = cls(mds.shape, q, **kwargs)
           backend.mds = mds.copy()
           backend.nmds = nmds.copy()
           return backend

**Contract Properties and High-Performance Integration**

The contract properties (``cmpnum``, ``cmpnames``, ``dtype``) are essential for:

- **Path 2 Integration**: The ``fuzzyset`` factory function uses these properties to validate 
  raw array shapes and automatically select the appropriate backend type.
- **Type Safety**: ``dtype`` ensures proper array allocation and prevents type mismatches 
  during high-performance operations.
- **Component Mapping**: ``cmpnames`` provides semantic meaning to array dimensions, 
  enabling clear documentation and debugging.
- **Validation Efficiency**: ``cmpnum`` allows fast shape validation without backend instantiation.

The ``from_arrays`` class method is specifically designed for Path 2, providing:

- **Direct Construction**: Bypasses individual ``Fuzznum`` object creation for maximum performance.
- **Constraint Validation**: Uses static methods for efficient fuzzy logic constraint checking.
- **Memory Optimization**: Minimizes array copying through careful memory management.

**QROHFNBackend (Set Components)**

Note the use of ``dtype=object`` for hesitant sets and enhanced constraint validation.

.. code-block:: python

   # axisfuzzy/fuzztype/qrohfs/backend.py
   @register_backend
   class QROHFNBackend(FuzzarrayBackend):
       mtype = 'qrohfn'

       @property
       def cmpnum(self) -> int:
           return 2  # Two components: md and nmd hesitant sets

       @property
       def cmpnames(self) -> Tuple[str, ...]:
           return 'md', 'nmd'  # Component names

       @property
       def dtype(self) -> np.dtype:
           return np.dtype(object)  # Object arrays for hesitant sets

       def _initialize_arrays(self):
           self.mds = np.empty(self.shape, dtype=object)
           self.nmds = np.empty(self.shape, dtype=object)

       def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
           md_value = self.mds[index]  # Value is already an array
           nmd_value = self.nmds[index]
           return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

       def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
           # The strategy ensures fuzznum.md is already an ndarray
           self.mds[index] = fuzznum.md
           self.nmds[index] = fuzznum.nmd

       @classmethod
       def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int, **kwargs):
           """Create backend from object arrays with enhanced validation."""
           if mds.dtype != object or nmds.dtype != object:
               raise TypeError(f"Input arrays must have dtype=object. Got {mds.dtype} and {nmds.dtype}.")
           cls._validate_fuzzy_constraints_static(mds, nmds, q=q)
           backend = cls(shape=mds.shape, q=q, **kwargs)
           backend.mds = mds
           backend.nmds = nmds
           return backend

Fuzzy Number Type Registry: The Central Hub for Extensibility
-------------------------------------------------------------

The **Fuzzy Number Type Registry** is the architectural cornerstone that makes `AxisFuzzy` a 
truly extensible framework. It acts as a central, thread-safe directory responsible for 
managing all available fuzzy number types. This registry is implemented as a singleton class, 
:class:`~axisfuzzy.core.registry.FuzznumRegistry`, ensuring a single source of truth 
throughout the application.

The core function of the registry is to map a unique string identifier, the ``mtype``, 
to the two classes that define a fuzzy number's complete behavior: 
its :class:`~axisfuzzy.core.base.FuzznumStrategy` and its :class:`~axisfuzzy.core.backend.FuzzarrayBackend`. 
This decoupled design allows developers to introduce entirely new fuzzy number types without 
modifying the core library code.

How It Works: The ``mtype`` Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The registry maintains two critical dictionaries:

-   ``strategies``: Maps an ``mtype`` string (e.g., ``'qrofn'``) to its corresponding ``FuzznumStrategy`` class.
-   ``backends``: Maps the same ``mtype`` to its corresponding ``FuzzarrayBackend`` class.

When you create a fuzzy number or array, for instance, via ``fuzzynum(mtype='qrofn', ...)``, 
`AxisFuzzy` internally queries the registry using the provided ``mtype``. 
It retrieves the appropriate ``QROFNStrategy`` and ``QROFNBackend`` classes to instantiate the objects, 
ensuring the correct logic, constraints, and data structures are used.

Registering a New Type: A Practical Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new fuzzy number type is a straightforward process. `AxisFuzzy` provides both a simple, 
declarative approach using decorators and a more explicit programmatic API.

The Decorator-Based Approach (Recommended)
++++++++++++++++++++++++++++++++++++++++++

The easiest and most common way to register a new type is by using the 
``@register_strategy`` and ``@register_backend`` decorators. 
You apply these directly to your new strategy and backend class definitions.

.. code-block:: python

   from axisfuzzy.core import FuzznumStrategy, FuzzarrayBackend
   from axisfuzzy.core import register_strategy, register_backend

   # 1. Define and register the strategy for the new type
   @register_strategy
   class MyNewTypeStrategy(FuzznumStrategy):
       mtype = 'mynewtype'
       # ... implementation with validators, transformers, etc. ...

   # 2. Define and register the backend for the new type
   @register_backend
   class MyNewTypeBackend(FuzzarrayBackend):
       mtype = 'mynewtype'
       # ... implementation of _initialize_arrays, copy, etc. ...

Behind the scenes, these decorators automatically call the registry's registration methods, 
making your new type immediately available throughout the `AxisFuzzy` ecosystem.

Programmatic Registration: The FuzznumStrategy API
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

For more dynamic scenarios, you can interact with the registry directly. 
First, you need to get the global registry instance using the 
:func:`~axisfuzzy.core.registry.get_registry_fuzztype` factory function.

.. code-block:: python

   from axisfuzzy.core.registry import get_registry_fuzztype

   # Get the singleton registry instance
   registry = get_registry_fuzztype()

   # Programmatically register the components
   registry.register(strategy=MyNewTypeStrategy, backend=MyNewTypeBackend)

The :meth:`~axisfuzzy.core.registry.FuzznumRegistry.register` method is the primary entry point for this. 
It can register a strategy, a backend, or both simultaneously. It also performs crucial validation, 
such as ensuring the ``mtype`` attributes of the strategy and backend match.

Advanced Features of the Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The registry offers several advanced features for robust and flexible type management.

Transactional Integrity
+++++++++++++++++++++++

When registering multiple components or types at once, it's vital to ensure atomicity. 
The registry provides a transaction context manager that guarantees that all operations 
within the block either complete successfully or are all rolled back upon failure.

.. code-block:: python

   registry = get_registry_fuzztype()

   try:
       with registry.transaction():
           # Registering a valid type
           registry.register(strategy=MyStrategy, backend=MyBackend)
           # This next line will fail and cause a rollback
           registry.register(strategy=InvalidStrategy)
   except ValueError:
       print("Transaction failed and was rolled back.")

   # Because of the rollback, 'my_mtype' will not be registered
   assert 'my_mtype' not in registry.get_registered_mtypes()

For convenience, the :meth:`~axisfuzzy.core.registry.FuzznumRegistry.batch_register` method wraps this 
logic, allowing you to register a list of components within a single transaction.

Introspection and Management
++++++++++++++++++++++++++++

The registry is not a black box. You can inspect its state and manage its contents dynamically:

-   **List all types**: ``registry.get_registered_mtypes()`` returns a list of all ``mtype`` 
    strings for which at least one component is registered.
-   **Retrieve a class**: ``registry.get_strategy('qrofn')`` or ``registry.get_backend('qrofn')`` 
    retrieves the specific class associated with an ``mtype``.
-   **Check for completeness**: ``registry.is_complete('qrofn')`` checks if both a strategy and a 
    backend are registered for a given ``mtype``.
-   **Unregister a type**: ``registry.unregister('mynewtype')`` allows you to dynamically remove 
    a type and its components from the registry.

This powerful, centralized registry system is what makes `AxisFuzzy` a flexible and extensible platform, 
empowering you to tailor it for novel research and complex applications.

Complete registration sample code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following displays the registration codes for ``qrofn`` and ``qrohfn``.

.. Note::

    The main difference between ``QROFNStrategy(FuzznumStrategy)`` and ``QROHFNStrategy(FuzznumStrategy)`` 
    lies in the fact that the ``QROHFNStrategy`` of ``qrohfn`` requires the involvement of the transformer 
    ``add_attribute_transformer``, as it involves multiple possible membership degree combinations.

FuzznumStrategy Registration
++++++++++++++++++++++++++++++++

1. Example 1: q-Rung Orthopair Fuzzy Number (qrofn)
   
   This is a fuzzy number type where membership (md) and
   non-membership (nmd) are single floating-point values.

.. code-block:: python
   :emphasize-lines: 10-13, 15-17

   @register_strategy
   class QROFNStrategy(FuzznumStrategy):
       mtype = 'qrofn'
       md: Optional[float] = None
       nmd: Optional[float] = None

       def __init__(self, q: Optional[int] = None):
           super().__init__(q=q)
           # Add validators to ensure md and nmd are floats between 0 and 1
           self.add_attribute_validator(
               'md', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)
           self.add_attribute_validator(
               'nmd', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)
           # Add callbacks to check constraints when attributes change
           self.add_change_callback('md', self._on_membership_change)
           self.add_change_callback('nmd', self._on_membership_change)
           self.add_change_callback('q', self._on_q_change)

       def _fuzz_constraint(self):
           # Constraint: md^q + nmd^q <= 1
           if self.md is not None and self.nmd is not None and self.q is not None:
               sum_of_powers = self.md ** self.q + self.nmd ** self.q
               if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                   raise ValueError(f"Constraint violation for qrofn")

       def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
           if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
               self._fuzz_constraint()

       def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
           if self.md is not None and self.nmd is not None and new_value is not None:
               self._fuzz_constraint()

2. Example 2: q-Rung Orthopair Hesitant Fuzzy Number (qrohfn)
   
   This is a more complex type where membership (md) and non-membership
   (nmd) are sets (lists or arrays) of possible values.

.. code-block:: python
   :emphasize-lines: 13-16, 18-19

   @register_strategy
   class QROHFNStrategy(FuzznumStrategy):
       mtype = 'qrohfn'
       md: Optional[Union[np.ndarray, List]] = None
       nmd: Optional[Union[np.ndarray, List]] = None

       def __init__(self, q: Optional[int] = None):
           super().__init__(q=q)

           # KEY DIFFERENCE: Use a transformer to automatically convert
           # input (like lists) into a consistent internal format (NumPy array).
           # This simplifies the rest of the logic.
           def _to_ndarray(x):
               if x is None:
                   return None
               return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=np.float64)

           self.add_attribute_transformer('md', _to_ndarray)
           self.add_attribute_transformer('nmd', _to_ndarray)

           # Validator now works with NumPy arrays
           def _attr_validator(x):
               if x is None:
                   return True
               # The transformer has already converted x to an ndarray
               if x.ndim == 1 and np.max(x) <= 1 and np.min(x) >= 0:
                   return True
               return False

           self.add_attribute_validator('md', _attr_validator)
           self.add_attribute_validator('nmd', _attr_validator)

           self.add_change_callback('md', self._on_membership_change)
           self.add_change_callback('nmd', self._on_membership_change)
           self.add_change_callback('q', self._on_q_change)

       def _fuzz_constraint(self):
           # Constraint for hesitant sets: max(md)^q + max(nmd)^q <= 1
           if self.md is not None and self.nmd is not None and self.q is not None:
               if len(self.md) > 0 and len(self.nmd) > 0:
                   sum_of_powers = np.max(self.md) ** self.q + np.max(self.nmd) ** self.q
                   if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                       raise ValueError(f"Constraint violation for qrohfn")

       def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
           if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
               self._fuzz_constraint()

       def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
           if self.md is not None and self.nmd is not None and new_value is not None:
               self._fuzz_constraint()

FuzzarrayBackend Registration
+++++++++++++++++++++++++++++++++

3. Example 3: q-Rung Orthopair Fuzzy Number (qrofn) backend
   
   This is a standard fuzzy number type where membership (md) and
   non-membership (nmd) are single floating-point values.

.. code-block:: python
   :emphasize-lines: 7,8

   @register_backend
   class QROFNBackend(FuzzarrayBackend):
       mtype = 'qrofn'

       def _initialize_arrays(self):
           # Use efficient NumPy float arrays for storage
           self.mds = np.zeros(self.shape, dtype=np.float64)
           self.nmds = np.zeros(self.shape, dtype=np.float64)

       def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
           md_value = float(self.mds[index])
           nmd_value = float(self.nmds[index])
           return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

       def set_fuzznum_data(self, index: Any, fuzzynum: 'Fuzznum'):
           self.mds[index] = fuzzynum.md
           self.nmds[index] = fuzzynum.nmd

4. Example 4: q-Rung Orthopair Hesitant Fuzzy Number (qrohfn) backend
   
   This is a more complex type where membership (md) and non-membership
   (nmd) are sets (lists or arrays) of possible values.

.. code-block:: python
   :emphasize-lines: 7,8

   @register_backend
   class QROHFNBackend(FuzzarrayBackend):
       mtype = "qrohfn"

       def _initialize_arrays(self):
           # Use NumPy arrays with dtype=object to store other arrays (the hesitant sets)
           self.mds = np.empty(self.shape, dtype=object)
           self.nmds = np.empty(self.shape, dtype=object)

       def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
           md_value = self.mds[index]
           nmd_value = self.nmds[index]
           return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

       def set_fuzznum_data(self, index: Any, fuzzynum: 'Fuzznum'):
           # The strategy's transformer ensures fuzzynum.md and fuzzynum.nmd are already ndarrays
           self.mds[index] = fuzzynum.md
           self.nmds[index] = fuzzynum.nmd

The Lifecycle of a Computation: Effortless Performance
------------------------------------------------------

When you perform an operation like ``result = arr + my_fuzznum``, `AxisFuzzy`'s
design ensures the process is both intuitive and highly performant. Here's a
high-level overview of what happens behind the scenes:

1.  **Automatic Dispatch**: The ``+`` operation is automatically routed to
    `AxisFuzzy`'s central computation engine. The framework intelligently
    recognizes that you are performing a vectorized operation between a
    ``Fuzzarray`` and a ``Fuzznum``.

2.  **Backend-Powered Calculation**: Instead of looping through each element in
    Python (which would be slow), the operation is delegated directly to the
    ``Fuzzarray``'s high-performance backend. The backend leverages the power of
    NumPy to perform the calculation on the underlying data arrays (the ``mds``
    and ``nmds`` arrays from the SoA architecture). This happens at C-level
    speed, complete with optimizations like broadcasting.

3.  **Efficient Result Creation**: The result of the computation is a new set of
    data arrays. A new ``Fuzzarray`` is then constructed to wrap these results
    using the most efficient "fast path". This means the final ``result`` object
    is created almost instantly, without the overhead of creating each
    ``Fuzznum`` one by one.

In essence, the framework seamlessly translates your simple, high-level Python
code into a highly optimized, low-level computation. You get the readability of
Python and the performance of a compiled language, all thanks to the
synergistic design of the ``Fuzzarray`` (the user-friendly facade) and its
``FuzzarrayBackend`` (the performance engine).
