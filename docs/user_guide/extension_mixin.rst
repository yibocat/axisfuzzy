.. _user_guide_extension_mixin:

Extension and Mixin Systems: Extending AxisFuzzy Functionality
==============================================================

AxisFuzzy provides two complementary systems for extending the functionality of 
``Fuzznum`` and ``Fuzzarray`` objects: the **Extension System** and the **Mixin Operations**. 
These systems enable developers to add new capabilities without modifying the core 
library code, following different architectural patterns optimized for distinct use cases.

This guide explains when and how to use each system, providing practical examples 
and clear decision criteria to help you choose the right approach for your needs.

.. contents::
   :local:

Overview of the Dual Architecture
---------------------------------

The two systems serve complementary purposes:

- **Extension System**: Provides **type-aware** functionality where behavior depends 
  on the specific fuzzy number type (``mtype``). Examples include distance calculations, 
  similarity measures, and scoring functions that vary between qrofn, qrohfn, and other types.

- **Mixin Operations**: Provides **type-agnostic** structural operations that work 
  uniformly across all fuzzy number types. Examples include array reshaping, 
  transposition, and concatenation operations.

System Architecture Diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    AxisFuzzy Extension Architecture
    ================================

    ┌─────────────────────────────────────────────────────────────────┐
    │                    User Interface Layer                         │
    │  Fuzznum.distance()  │  Fuzzarray.reshape()  │  af.similarity() │
    └─────────────────────┬───────────────────────┬───────────────────┘
                          │                       │
    ┌─────────────────────▼───────────────────────▼───────────────────┐
    │                Extension System             │  Mixin Operations │
    │  ┌──────────────────────────────────────┐   │  ┌─────────────┐  │
    │  │        Runtime Dispatch              │   │  │   Static    │  │
    │  │  ┌──────────────────────────────┐    │   │  │  Injection  │  │
    │  │  │    ExtensionDispatcher       │    │   │  │             │  │
    │  │  │   ┌─────────────────────┐    │    │   │  │ ┌─────────┐ │  │
    │  │  │   │  mtype='qrofs'      │    │    │   │  │ │ Factory │ │  │
    │  │  │   │  → qrofs_distance() │    │    │   │  │ │Functions│ │  │
    │  │  │   └─────────────────────┘    │    │   │  │ └─────────┘ │  │
    │  │  │   ┌─────────────────────┐    │    │   │  │             │  │
    │  │  │   │  mtype='qrohfs'     │    │    │   │  │ ┌─────────┐ │  │
    │  │  │   │  → qrohfs_distance()│    │    │   │  │ │Registry │ │  │
    │  │  │   └─────────────────────┘    │    │   │  │ │ System  │ │  │
    │  │  │   ┌──────────────────────┐   │    │   │  │ └─────────┘ │  │
    │  │  │   │  default=True        │   │    │   │  └─────────────┘  │
    │  │  │   │  → default_distance()│   │    │   │                   │
    │  │  │   └──────────────────────┘   │    │   │                   │
    │  │  └──────────────────────────────┘    │   │                   │
    │  └──────────────────────────────────────┘   │                   │
    └─────────────────────────────────────────────┴───────────────────┘
                          │                       │
    ┌─────────────────────▼───────────────────────▼───────────────────┐
    │                    Core Data Layer                              │
    │        Fuzznum        │        Fuzzarray        │    Backend    │
    │    (Scalar Fuzzy)     │    (Array of Fuzzy)     │      SoA      │
    └─────────────────────────────────────────────────────────────────┘

Target Audience and Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

**Extension System** - For External Users and Developers:

- **Primary audience**: External users, researchers, and third-party developers
- **Purpose**: Add custom fuzzy operations and mathematical functions
- **Use cases**: Domain-specific fuzzy logic operations, custom distance metrics, specialized aggregation functions
- **Accessibility**: Public API designed for ease of use

**Mixin Operations** - For AxisFuzzy Maintainers and Core Developers:

- **Primary audience**: AxisFuzzy core maintainers and internal developers
- **Purpose**: Implement universal structural operations and container behaviors
- **Use cases**: Array manipulation, data structure operations, NumPy-like functionality
- **Accessibility**: Internal API for framework development (static integration, not user-extensible)

.. note::
   
   **For most users**: You will primarily use the **Extension System** to add custom functionality.
   The Mixin Operations is documented here for completeness and for those contributing to AxisFuzzy's core.
   The Mixin system is **not user-extensible**. It provides a static set of operations 
   that are integrated into the framework during initialization. Users cannot dynamically register 
   new mixin functions.

.. code-block:: python

    # Extension System - for external users
    @extension(name='custom_distance', mtype='qrofn')
    def my_distance_metric(x, y, p=2):
        """Custom distance metric for Q-rung orthopair fuzzy numbers."""
        return ((abs(x.md**x.q - y.md**y.q)**p + 
                abs(x.nmd**x.q - y.nmd**y.q)**p) / 2)**(1/p)
    
    # Mixin Operations - for core developers (internal use)
    # These are statically integrated during framework initialization
    @register_mixin('reshape', target_classes=['Fuzzarray'])
    def reshape_impl(self, *shape):
        """Internal implementation of reshape operation."""
        return _reshape_factory(self, *shape)

Extension System: Dynamic, Type-Aware Function Registration
-----------------------------------------------------------

The `extension` system is the cornerstone of AxisFuzzy's dynamic functionality, 
designed to address operations whose logic is intrinsically tied to the mathematical 
definition of a fuzzy number type (``mtype``). It allows developers to register 
multiple implementations for a single function name, with the framework automatically 
dispatching to the correct one at runtime based on the object's type. This polymorphic 
behavior is essential for building a robust and extensible fuzzy logic ecosystem.

Architectural Pillars
~~~~~~~~~~~~~~~~~~~~~

The power of the `extension` system stems from a clean, decoupled architecture comprising three pillars:

1. **The** ``@extension`` **Decorator**: A declarative API for registering functions.
2. **The Extension Registry**: A central, thread-safe registry that indexes all registered 
   implementations and their metadata.
3. **The Dynamic Injector**: A mechanism that injects the registered functions as methods or 
   properties into target classes (like ``Fuzznum`` and ``Fuzzarray``) at runtime.

Declarative Registration
~~~~~~~~~~~~~~~~~~~~~~~~

Registering a type-specific implementation is achieved declaratively using the ``@extension`` 
decorator. This approach cleanly separates the core logic of your function from the registration process.

.. code-block:: python

    from axisfuzzy.extension import extension
    
    @extension(name='similarity', mtype='qrofn')
    def qrofn_similarity(x, y):
        """Cosine similarity for q-rung orthopair fuzzy numbers."""
        numerator = x.md * y.md + x.nmd * y.nmd
        denominator = ((x.md**2 + x.nmd**2) * (y.md**2 + y.nmd**2))**0.5
        return numerator / denominator if denominator > 0 else 0
    
    @extension(name='similarity', mtype='qrohfn')
    def qrohfn_similarity(x, y):
        """Similarity for q-rung orthopair hesitant fuzzy numbers."""
        # Different implementation for hesitant fuzzy numbers
        return calculate_hesitant_similarity(x, y)

Specializing by Fuzzy Type (mtype)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary strength of the `extension` system is its ability to specialize behavior 
based on ``mtype``. Below are examples demonstrating how to provide distinct implementations 
for various fuzzy set types.

**Q-rung Orthopair Fuzzy Numbers (qrofn)**

.. code-block:: python

    @extension(name='distance', mtype='qrofn')
    def qrofn_distance(x, y, p=2):
        """Calculate distance between two Q-rung orthopair fuzzy numbers."""
        return ((abs(x.md**x.q - y.md**y.q)**p + 
                abs(x.nmd**x.q - y.nmd**y.q)**p) / 2)**(1/p)

**Q-rung Orthopair Hesitant Fuzzy Numbers (qrohfn)**

.. code-block:: python

    @extension(name='aggregation', mtype='qrohfn')
    def qrohfn_aggregation(x, weights=None):
        """Aggregate Q-rung orthopair hesitant fuzzy values."""
        # Implementation for QROHFN aggregation
        return aggregate_hesitant_values(x, weights)

**Classical Fuzzy Sets (fs)**

.. code-block:: python

    @extension(name='defuzzify', mtype='fs')
    def fs_defuzzify(x, method='centroid'):
        """Defuzzify a classical fuzzy set."""
        # Implementation for defuzzification
        return defuzzification_result(x, method)

Controlling API Exposure: Injection Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `extension` system provides fine-grained control over how a function is exposed 
to the end-user through the ``injection_type`` parameter. This flexibility allows for 
crafting intuitive and consistent APIs.

**Instance Methods** (default): Functions are added as methods to instances

.. code-block:: python

    @extension(name='normalize', mtype='qrofn', injection_type='instance_method')
    def normalize_qrofn(self):
        """Normalize the Q-rung orthopair fuzzy number."""
        # Access instance data via self
        total = self.md + self.nmd
        if total > 0:
            return af.fuzzynum(self.md/total, self.nmd/total, mtype=self.mtype)
        return self
    
    # Usage
    qrofn_value = af.fuzzynum(md=0.8, nmd=0.3, mtype='qrofn', q=3)
    normalized = qrofn_value.normalize()  # Called as instance method

**Top-level Functions**: Functions are available as standalone functions:

.. code-block:: python

    @extension(name='distance', mtype='qrofn', injection_type='top_level_function')
    def qrofn_distance(a, b, metric='euclidean'):
        """Calculate distance between two Q-rung orthopair fuzzy numbers."""
        # Implementation
        return distance_value
    
    # Usage
    import axisfuzzy as af
    dist = af.distance(qrofn1, qrofn2)  # Called as top-level function

**Both Types**: Functions are available both ways

.. code-block:: python

    @extension(name='complement', mtype='qrofn', injection_type='both')
    def qrofn_complement(self):
        """Calculate the complement of a Q-rung orthopair fuzzy number."""
        # Implementation
        return af.fuzzynum(self.nmd, self.md, mtype=self.mtype, q=self.q)
    
    # Usage - both ways work
    comp1 = qrofn_value.complement()     # Instance method
    comp2 = af.complement(qrofn_value)   # Top-level function

**Instance Properties**: Functions can be exposed as properties

.. code-block:: python

    @extension(name='score', mtype='qrofn', injection_type='instance_property')
    def qrofn_score(self):
        """Calculate the score of a Q-rung orthopair fuzzy number."""
        return self.md**self.q - self.nmd**self.q
    
    # Usage
    score = qrofn_value.score  # Accessed as property (no parentheses)

The ``@extension`` Decorator API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@extension`` decorator is the primary interface for registration, offering a rich set 
of parameters to precisely control a function's behavior and metadata.

- ``name`` (str): The function name that will be available on objects
- ``mtype`` (str, optional): Target fuzzy number type (e.g., ``'qrofn'``, ``'qrohfn'``)
- ``target_classes`` (list, optional): Classes to inject into ``Fuzznum``, ``Fuzzarray`` or ``[Fuzznum, Fuzzarray]``
- ``injection_type`` (str): How the function is exposed:
  
  - ``'instance_method'`` : Available as ``obj.function()``
  - ``'top_level_function'`` : Available as ``axisfuzzy.function()``
  - ``'both'`` : Available in both ways (default)
  - ``'instance_property'`` : Available as ``obj.property``

- ``is_default`` (bool): Whether this is a fallback implementation
- ``priority`` (int): Resolution priority for conflicting registrations

Seamless Dispatch in Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once an extension is registered, the framework's dynamic dispatcher handles the rest. 
Calls made via instance methods or top-level functions are automatically routed to the 
appropriate implementation based on the object's ``mtype``, making the process transparent to the user.

.. code-block:: python

    import axisfuzzy as af
    
    # Create fuzzy numbers
    x = af.fuzzynum(md=0.8, nmd=0.3, mtype='qrofn', q=2)
    y = af.fuzzynum(md=0.6, nmd=0.5, mtype='qrofn', q=2)
    
    # Use as instance method
    sim = x.similarity(y)
    
    # Use as top-level function
    sim = af.similarity(x, y)
    
    # Both calls automatically dispatch to qrofn_similarity

Defining Fallback Behavior: Default Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enhance robustness, you can provide a generic implementation that serves as a fallback 
when no ``mtype``-specific version is found. This is achieved by setting the ``is_default=True`` flag.

.. code-block:: python

    @extension(name='normalize', is_default=True)
    def default_normalize(x):
        """Default normalization for any fuzzy number type."""
        total = x.md + x.nmd
        if total > 0:
            return af.fuzzynum(x.md/total, x.nmd/total, mtype=x.mtype)
        return x

Advanced Capabilities
~~~~~~~~~~~~~~~~~~~~~

Beyond basic registration, the `extension` system offers advanced features for managing 
complex scenarios, such as plugin architectures and conditional logic.

**Conflict Resolution with Priority**

In modular systems, it's possible for multiple libraries to register an implementation 
for the same ``(name, mtype)`` pair. The ``priority`` parameter resolves such conflicts 
deterministically: the implementation with the highest priority wins. This prevents 
accidental overwrites and ensures predictable behavior.

.. code-block:: python

    @extension(name='distance', mtype='qrofn', priority=1)
    def euclidean_distance(x, y):
        """Standard Euclidean distance."""
        return standard_euclidean(x, y)
    
    @extension(name='distance', mtype='qrofn', priority=2)  # Higher priority
    def improved_distance(x, y):
        """Improved distance calculation."""
        return improved_euclidean(x, y)  # This will be used

**Conditional Registration**

Registration can be guarded by conditional logic, allowing you to create extensions that 
depend on optional dependencies, such as `NumPy`.

.. code-block:: python

    # Only register if NumPy is available
    try:
        import numpy as np
        
        @extension(name='to_numpy', mtype='qrofn')
        def qrofn_to_numpy(self):
            """Convert to NumPy array representation."""
            return np.array([self.md, self.nmd, self.q])
    except ImportError:
        pass

Built-in Extensions
~~~~~~~~~~~~~~~~~~~

`AxisFuzzy` ships with a rich set of pre-registered extensions for common fuzzy number types, 
providing out-of-the-box functionality for a wide range of tasks.

**For qrofn (q-Rung Orthopair Fuzzy Numbers)**:

- **Constructors**: ``empty``, ``positive``, ``negative``, ``full``, ``empty_like``, ``positive_like``, ``negative_like``, ``full_like``
- **I/O Operations**: ``to_csv``, ``read_csv``, ``to_json``, ``read_json``, ``to_npy``, ``read_npy``
- **Measurement**: ``distance``
- **String Conversion**: ``str2fuzznum``
- **Aggregation**: ``sum``, ``mean``, ``max``, ``min``, ``prod``, ``var``, ``std``
- **Instance Properties**: ``score``, ``acc``, ``ind``

**For qrohfn (q-Rung Orthopair Hesitant Fuzzy Numbers)**:

- **Constructors**: ``empty``, ``positive``, ``negative``, ``full``, ``empty_like``, ``positive_like``, ``negative_like``, ``full_like``
- **I/O Operations**: ``to_csv``, ``read_csv``, ``to_json``, ``read_json``, ``to_npy``, ``read_npy``
- **Measurement**: ``distance``, ``normalize``
- **String Conversion**: ``str2fuzznum``
- **Aggregation**: ``sum``, ``mean``, ``max``, ``min``, ``prod``, ``var``, ``std``
- **Instance Properties**: ``score``, ``acc``, ``ind``

Development Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure your extensions are robust, maintainable, and integrate seamlessly with the 
`AxisFuzzy` ecosystem, adhere to the following best practices.

**1. Adopt Clear Naming Conventions**

.. code-block:: python

    # Good: descriptive and specific
    @extension(name='cosine_similarity', mtype='qrofn')
    def qrofn_cosine_similarity(x, y):
        pass
    
    # Avoid: generic or ambiguous names
    @extension(name='calc', mtype='qrofn')
    def some_calculation(x, y):
        pass

**2. Write Comprehensive Documentation**

.. code-block:: python

    @extension(name='weighted_distance', mtype='qrofn')
    def qrofn_weighted_distance(x, y, weights=None, p=2):
        """Calculate weighted Minkowski distance between Q-rung orthopair fuzzy numbers.
        
        Parameters
        ----------
        x, y : Fuzznum
            Q-rung orthopair fuzzy numbers to compare
        weights : array-like, optional
            Weights for membership and non-membership degrees
        p : float, default=2
            Minkowski distance parameter (p=2 for Euclidean)
            
        Returns
        -------
        float
            Weighted distance value
        """
        if weights is None:
            weights = [0.5, 0.5]
        # Implementation here
        pass

**3. Leverage Backend-Based High-Performance Computing**

For optimal performance, especially when working with ``Fuzzarray`` objects, 
design your extensions to leverage the underlying **Struct of Arrays (SoA)** 
architecture provided by ``FuzzarrayBackend``. This approach ensures:

- **Memory Locality**: Operations on component arrays (e.g., membership degrees) 
  benefit from contiguous memory layout
- **Vectorization**: NumPy-based operations can utilize SIMD instructions for 
  parallel computation
- **Cache Efficiency**: Reduced memory fragmentation leads to better CPU cache utilization

.. code-block:: python

    @extension(name='batch_operation', mtype='qrofn')
    def qrofn_batch_operation(fuzz_array):
        """Example of backend-aware high-performance extension."""
        # Access backend directly for vectorized operations
        backend = fuzz_array._backend
        
        # Perform vectorized computation on component arrays
        result_mds = np.sqrt(backend.mds)  # Vectorized operation
        result_nmds = np.sqrt(backend.nmds)
        
        # Create new backend with results (fast path)
        from axisfuzzy.fuzztype.qrofs import QROFNBackend
        new_backend = QROFNBackend.from_arrays(
            mds=result_mds, nmds=result_nmds, q=backend.q
        )
        
        # Return new Fuzzarray using the fast path (O(1) operation)
        from axisfuzzy.core import Fuzzarray
        return Fuzzarray(backend=new_backend)

This pattern follows the same high-performance principles used throughout 
``AxisFuzzy``'s core, ensuring your extensions scale efficiently with large datasets.




Mixin Operations: Universal Structural Operations
-------------------------------------------------

The Mixin Operations offers a suite of universal, NumPy-inspired structural operations 
that are seamlessly integrated across all fuzzy data types within the `AxisFuzzy` ecosystem. 
These operations are designed for data manipulation and structural transformation, 
rather than fuzzy-specific arithmetic, providing a consistent and predictable API.

Core Philosophy
~~~~~~~~~~~~~~~

The design of the Mixin Operations is guided by a distinct set of principles compared to the Extension System:

* **Universality**: Implement functions that are logically applicable to any fuzzy 
  type, ensuring consistent behavior.
* **Structural Focus**: Prioritize operations on the data container (e.g., shape, 
  size, layout) over the fuzzy values themselves.
* **NumPy-like Interface**: Adopt familiar and powerful array manipulation patterns 
  from NumPy to lower the learning curve.
* **Composition over Inheritance**: Dynamically compose mixins into core classes, 
  promoting flexibility and avoiding rigid class hierarchies.

Demonstration of Core Mixin Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All fuzzy objects automatically inherit mixin functionalities, enabling direct and intuitive use.

.. code-block:: python

    import axisfuzzy as af
    import numpy as np

    md = np.array([0.8, 0.6, 0.9])
    nmd = np.array([0.1, 0.3, 0.05])

    fuzzy_array = af.fuzzyarray(np.array([md,nmd]), mtype='qrofn')

    # Mixin operations work regardless of type
    shape = fuzzy_array.shape          # Shape information
    reshaped = fuzzy_array.reshape(3, 1)  # Reshape operation
    flattened = fuzzy_array.flatten()  # Flatten to 1D
    copied = fuzzy_array.copy()        # Deep copy

Key Functionality Groups
~~~~~~~~~~~~~~~~~~~~~~~~

**Shape and Dimensionality**:

.. code-block:: python

    import axisfuzzy as af
    import numpy as np

    arr = af.fuzzyarray(np.array([[[0.8, 0.6], [0.7, 0.9]],
                                [[0.1, 0.3], [0.2, 0.1]]]), mtype='qrofn')

    # Reshape array
    reshaped = arr.reshape(4)  # or af.reshape(arr, 4)

    # Flatten to 1D
    flat = arr.flatten()

    # Remove single dimensions
    squeezed = arr.squeeze()

    # Return flattened view
    raveled = arr.ravel()

**Data Transformation**:

.. code-block:: python

    # Transpose array
    transposed = arr.T  # or af.transpose(arr)
    
    # Broadcast to new shape
    broadcasted = arr.broadcast_to((3, 2, 2))


**Container Manipulation**:

.. code-block:: python

    arr1 = af.fuzzyarray(np.array([[0.8, 0.6], [0.1, 0.3]]), mtype='qrofn')
    arr2 = af.fuzzyarray(np.array([[0.7, 0.9], [0.2, 0.1]]), mtype='qrofn')

    # Concatenate arrays
    combined = arr1.concat(arr2)  # or af.concat(arr1, arr2)

    # Stack arrays along new axis
    stacked = arr1.stack(arr2, axis=0)

    # Append elements
    extended = arr1.append(af.fuzzynum(md=0.5, nmd=0.4, mtype='qrofn'))

    # Remove and return elements
    item = arr1.pop(0)

**Utilities and Inspection**:

.. code-block:: python

    # Create deep copy
    copied = af.copy(arr)
    
    # Extract scalar item
    scalar = arr.flatten().item(0,1)
    
    # Boolean testing
    has_any = arr.any()  # True if any element is "truthy"
    all_true = arr.all()  # True if all elements are "truthy"

Advanced Structural Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Shape and Attribute Inspection**

.. code-block:: python

    # Shape operations work on any fuzzy type
    data = af.fuzzyarray(np.array([[[0.8, 0.6], [0.7, 0.9]], [[0.1, 0.3], [0.2, 0.1]]]), mtype='qrofn')

    print(data.shape)           # (2, 2)
    print(data.size)            # 4
    print(data.ndim)            # 2

    # Reshape operations
    reshaped = data.reshape(4)  # Flatten to (4,)
    expanded = data.reshape(2, 2, 1)  # Add dimension

**Advanced Indexing and Slicing**

.. code-block:: python

    # Advanced indexing works uniformly
    data = af.fuzzyarray(np.array([[0.1, 0.5, 0.8, 0.3, 0.9], [0.8, 0.4, 0.1, 0.6, 0.05]]), mtype='qrohfn')
    
    # Boolean indexing
    high_values = data[data > 0.5]  # Elements > 0.5
    
    # Fancy indexing
    selected = data[[0, 2, 4]]      # Select specific indices
    
    # Slice operations
    subset = data[1:4]              # Slice notation

Type-Agnostic by Design
~~~~~~~~~~~~~~~~~~~~~~~

A core strength of the Mixin Operations is its type-agnostic nature, ensuring operational 
consistency across the entire fuzzy ecosystem.

.. code-block:: python

    # Same operations, different types
    qrofn_data = af.random.rand('qrofn', shape=3)
    qrohfn_data = af.random.rand('qrohfn', shape=3)

    # All support the same mixin operations
    for data in [qrofn_data, qrohfn_data]:
        print(f"Shape: {data.shape}")
        print(f"Max: {data.max()}")
        print(f"Mean: {data.mean()}")

Built-in Mixin Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

The Mixin system provides a comprehensive set of pre-implemented operations that are 
available across all fuzzy number types. These operations are statically integrated 
into the core classes during framework initialization.

.. code-block:: python

    # Example of using built-in mixin operations
    data = af.fuzzyarray(np.array([[0.8, 0.6, 0.9], [0.1, 0.3, 0.05]]), mtype='qrofn')

    # Structural operations (available via mixin system)
    reshaped = data.reshape(3, 1)  # Shape manipulation
    transposed = data.T            # Transposition
    flattened = data.flatten()     # Flattening

    # Statistical operations
    mean_val = data.mean()         # Mean calculation

    # Utility operations
    copied = data.copy()           # Deep copy

Seamless Integration with Core Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mixin functions are automatically injected into the ``Fuzznum`` and ``Fuzzarray`` 
base classes, making them feel like native methods.

.. code-block:: python

    # All these work regardless of mtype
    num = af.fuzzynum(md=0.8, nmd=0.2, mtype='qrofn')
    arr = af.fuzzyarray(np.array([[0.7, 0.6], [0.2, 0.3]]), mtype='qrofn')

    # Shape operations work on both
    num_reshaped = num.reshape(1, 1)
    arr_reshaped = arr.reshape(2, 1)

    # # Copying works uniformly
    num_copy = af.copy(num)
    arr_copy = af.copy(arr)



Developer Guide: Creating Custom Extensions
-------------------------------------------

This guide provides a comprehensive walkthrough for developers who wish to extend 
`AxisFuzzy`'s capabilities by creating custom, type-specific functionalities through the Extension System.

.. note::
   
   The Extension System is the designated pathway for external contributions and 
   domain-specific customizations. In contrast, the Mixin Operations is reserved for 
   internal, universal structural operations.

When to Create an Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend creating extensions for functionalities that are inherently tied to a 
specific fuzzy logic type. Key use cases include:

1. **Custom Fuzzy Operations**: Novel arithmetic, aggregation, or logical operators.
2. **Domain-Specific Algorithms**: Specialized algorithms for fields like decision-making, 
   control systems, or image processing.
3. **Specialized Mathematical Functions**: Unique distance metrics, similarity measures, 
   or membership function evaluators.
4. **Type-Specific Behaviors**: Any functionality that depends on the unique properties 
   of a fuzzy type (e.g., the 'q' parameter in q-rung orthopair fuzzy sets).

Illustrative Examples
~~~~~~~~~~~~~~~~~~~~~~

**Custom Aggregation Functions**:

.. code-block:: python

    # Custom aggregation for interval-valued fuzzy sets
    @extension(name='weighted_avg', mtype='ivfs')
    def interval_weighted_average(x, weights):
        """Weighted average for interval-valued fuzzy sets."""
        lower = np.average([iv.lower for iv in x], weights=weights)
        upper = np.average([iv.upper for iv in x], weights=weights)
        return IntervalValue(lower, upper)

**Domain-Specific Distance Metrics**:

.. code-block:: python

    # Hamming distance for type-II fuzzy sets
    @extension(name='hamming_distance', mtype='t2fs')
    def t2fs_hamming(x, y):
        """Hamming distance for type-II fuzzy sets."""
        return np.sum(np.abs(x.primary - y.primary) + 
                     np.abs(x.secondary - y.secondary))

**Specialized Membership Functions**:

.. code-block:: python

    # Custom membership evaluation
    @extension(name='evaluate_membership', mtype='qrofs')
    def qrofs_membership(x, element):
        """Evaluate membership for Q-rung orthopair fuzzy sets."""
        md_dist = abs(x.md - element.md) ** x.q
        nmd_dist = abs(x.nmd - element.nmd) ** x.q
        return 1 - ((md_dist + nmd_dist) / 2) ** (1/x.q)

Best Practices for Extension Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Adopt Descriptive Naming Conventions**:

.. code-block:: python

    # GOOD: Descriptive and specific
    @extension(name='euclidean_distance', mtype='qrofn')
    def qrofn_euclidean_distance(x, y):
        pass
    
    # BAD: Too generic
    @extension(name='distance', mtype='qrofn')
    def distance(x, y):
        pass

**2. Write Comprehensive Docstrings**:

.. code-block:: python

    @extension(name='custom_similarity', mtype='fs')
    def fuzzy_similarity(x, y, method='cosine'):
        """
        Calculate similarity between fuzzy sets.
        
        Parameters
        ----------
        x, y : FuzzySet
            Input fuzzy sets
        method : str, default 'cosine'
            Similarity method ('cosine', 'jaccard', 'dice')
            
        Returns
        -------
        float
            Similarity value in [0, 1]
        """
        pass

**3. Implement Robust Edge-Case Handling**:

.. code-block:: python

    @extension(name='safe_division', mtype='qrofn')
    def safe_qrofn_division(x, y, default=0.0):
        """Division with zero-handling for Q-rung numbers."""
        if y.md == 0 and y.nmd == 0:
            return default
        # Implement division logic
        pass

**4. Prioritize Performance Optimization**:

.. code-block:: python

    @extension(name='fast_aggregation', mtype='fs')
    def optimized_aggregation(fuzzy_sets):
        """Optimized aggregation using vectorized operations."""
        # Use NumPy vectorization when possible
        values = np.array([fs.membership_values for fs in fuzzy_sets])
        return np.mean(values, axis=0)

Common Pitfalls to Avoid
~~~~~~~~~~~~~~~~~~~~~~~~

**1. Avoid Creating "Universal" Extensions**: Extensions should be type-specific. 
If a function is universally applicable, it belongs in the Mixin Operations.

.. code-block:: python

    # BAD: This should be a mixin, not a multi-type extension
    @extension(name='reshape', mtype='qrofn')
    @extension(name='reshape', mtype='fs')
    @extension(name='reshape', mtype='ivfs')
    def reshape_for_each_type(self, *shape):
        # Same implementation for all types
        return self._data.reshape(*shape)

**2. Keep Interfaces Clean and Simple**: Avoid over-engineering simple 
operations with excessive parameters.

.. code-block:: python

    # BAD: Too many parameters for a simple operation
    @extension(name='simple_add', mtype='fs')
    def overcomplicated_addition(x, y, normalize=True, 
                                method='algebraic', 
                                confidence=0.95,
                                validation_level='strict'):
        # Keep extensions focused and simple
        pass

**3. Enforce Type Safety**: Always validate input types to prevent unexpected behavior.

.. code-block:: python

    # GOOD: Type checking
    @extension(name='type_safe_operation', mtype='qrofn')
    def safe_operation(x, y):
        if not isinstance(x, QRungOrthopairFuzzyNumber):
            raise TypeError(f"Expected QRungOrthopairFuzzyNumber, got {type(x)}")
        # Implementation
        pass


System Comparison and Architecture Understanding
------------------------------------------------

Extension vs Mixin Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This comparison helps understand the architectural differences between the two systems:

+---------------------------+------------------+------------------+
| Characteristic            | Extension System | Mixin Operations |
+===========================+==================+==================+
| **Type Dependency**       | mtype-sensitive  | mtype-agnostic   |
+---------------------------+------------------+------------------+
| **Mathematical Logic**    | Varies by type   | Uniform across   |
+---------------------------+------------------+------------------+
| **Performance**           | Slight dispatch  | Zero overhead    |
+---------------------------+------------------+------------------+
| **Use Cases**             | Distance, score, | Reshape, concat, |
|                           | similarity       | transpose        |
+---------------------------+------------------+------------------+
| **Extensibility**         | User-extensible  | Internal only    |
|                           | (plugins)        | (not extensible) |
+---------------------------+------------------+------------------+

Understanding Extension Use Cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Extensions are designed for:**

1. **Type-specific Operations**: The function behavior fundamentally depends on the fuzzy type

   .. code-block:: python

       # Q-rung specific similarity using q-parameter
       @extension(name='similarity', mtype='qrofn')
       def qrofn_similarity(x, y):
           # Q-rung specific similarity using q-parameter
           return ((x.md * y.md)**(1/x.q) + (x.nmd * y.nmd)**(1/x.q)) / 2
       
       @extension(name='similarity', mtype='fs')
       def fs_similarity(x, y):
           # Classical fuzzy similarity
           return min(x.membership, y.membership)

2. **Mathematical Operations**: For operations rooted in fuzzy set theory

   .. code-block:: python

       # Fuzzy complement depends on the type's mathematical definition
       @extension(name='complement', mtype='qrofn')
       def qrofn_complement(self):
           # Q-rung orthopair complement
           return af.fuzzynum(self.nmd, self.md, mtype=self.mtype, q=self.q)
       
       @extension(name='complement', mtype='fs')
       def fs_complement(self):
           # Classical fuzzy complement
           return af.fuzzynum(1 - self.membership, mtype='fs')

3. **Domain Expertise Operations**: Operations requiring deep understanding of fuzzy theory

   .. code-block:: python

       @extension(name='score_function', mtype='qrofn')
       def qrofn_score(self):
           """Calculate Q-rung orthopair score function."""
           return self.md**self.q - self.nmd**self.q
       
       @extension(name='hesitancy_degree', mtype='qrohfn')
       def qrohfn_hesitancy(self):
           """Calculate hesitancy degree for Q-rung orthopair hesitant fuzzy numbers."""
           return calculate_hesitancy_degree(self)

**Mixins are designed for:**

.. note::
   Mixins are part of the core library's internal architecture and are not user-extensible. 
   The following examples illustrate the design principles behind built-in mixin operations.

1. **Universal Operations**: When the operation works identically across all fuzzy types

   .. code-block:: python

       # Shape operations are universal (internal implementation)
       def reshape_array(self, *shape):
           """Reshape works the same for any fuzzy type."""
           return self._reshape_implementation(*shape)
       
       # Copying is universal (internal implementation)
       def copy_array(self):
           """Deep copy works the same for any fuzzy type."""
           return self._copy_implementation()

2. **Data Manipulation**: For operations that manipulate the container structure

   .. code-block:: python

       # Indexing operations (internal implementation)
       def take_elements(self, indices):
           """Take elements at specified indices."""
           return self[indices]
       
       # Concatenation operations (internal implementation)
       def concatenate_arrays(arrays, axis=0):
           """Concatenate arrays along specified axis."""
           return generic_concatenate(arrays, axis)

3. **NumPy-like Interface**: When you want familiar array programming patterns

   .. code-block:: python

       # Statistical operations that work on any numeric data (internal implementation)
       def calculate_mean(self, axis=None):
           """Calculate mean along specified axis."""
           return self._statistical_mean(axis)
       
       # Sorting operations (internal implementation)
       def sort_array(self, axis=-1):
           """Sort array along specified axis."""
           return self._sort_implementation(axis)

Understanding the Distinction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To help users understand when functionality belongs to Extensions vs Mixins, 
here are key architectural principles:

**Type-Dependent vs Type-Agnostic Operations**

.. code-block:: python

    # Type-dependent: Use Extensions
    @extension(name='similarity', mtype='qrofn')
    def qrofn_similarity(x, y):
        # Q-rung specific similarity calculation
        return ((x.md * y.md)**(1/x.q) + (x.nmd * y.nmd)**(1/x.q)) / 2
    
    @extension(name='similarity', mtype='fs')
    def fs_similarity(x, y):
        # Classical fuzzy similarity
        return min(x.membership, y.membership)
    
    # Type-agnostic: Built into Mixins (not user-extensible)
    # Example: arr.reshape(), arr.transpose(), arr.flatten()
    # These work identically regardless of fuzzy type

**Mathematical vs Structural Operations**

.. code-block:: python

    # Mathematical operations: Use Extensions
    @extension(name='complement', mtype='qrofn')
    def qrofn_complement(self):
        # Mathematical complement depends on fuzzy type definition
        return af.fuzzynum(self.nmd, self.md, mtype=self.mtype, q=self.q)
    
    # Structural operations: Built into Mixins
    # Example: arr.copy(), arr.take(), arr.concatenate()
    # These manipulate data structure, not mathematical content

Best Practices for Extension Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When developing extensions for AxisFuzzy:

1. **Identify Type Dependency**: Does your operation behavior fundamentally depend on the fuzzy type?
   
   - Yes → Develop as Extension
   - No → Consider if it should be a regular utility function

2. **Assess Mathematical Foundation**: Is this operation rooted in fuzzy set theory?
   
   - Yes → Develop as Extension
   - No → Consider alternative approaches

3. **Evaluate Domain Specificity**: Does the operation require deep understanding of specific fuzzy types?
   
   - Yes → Develop as Extension
   - No → Consider more general solutions

4. **Consider Reusability**: Will this operation be useful across different projects?
   
   - Yes → Develop as Extension with proper documentation
   - No → Consider project-specific implementation

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Extension System**: First call has minimal dispatch overhead; subsequent calls are cached
- **Mixin Operations**: Zero runtime overhead, equivalent to native method calls
- **Memory**: Both systems have negligible memory impact
- **Scalability**: Both scale well with large arrays and complex operations

Development Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**For Extension Development**:

1. Always provide clear ``mtype`` specifications
2. Include default implementations when appropriate
3. Use descriptive function names that indicate purpose
4. Document mathematical formulations in docstrings
5. Test with multiple fuzzy number types

**For Mixin Development**:

1. Ensure operations work uniformly across all mtypes
2. Follow NumPy conventions for parameter names and behavior
3. Delegate to factory functions for actual implementation
4. Maintain consistency with existing array operations
5. Consider both Fuzznum and Fuzzarray use cases

**Available Mixin Operations**

**Shape and Structure**:

- ``shape``: Array dimensions
- ``size``: Total number of elements
- ``ndim``: Number of dimensions
- ``reshape()``: Change array shape
- ``flatten()``: Flatten to 1D
- ``squeeze()``: Remove single-dimensional entries
- ``expand_dims()``: Add new dimensions

**Indexing and Access**:

- ``item()``: Extract single element
- ``take()``: Take elements along axis
- ``compress()``: Select elements using condition
- ``choose()``: Choose elements from multiple arrays

**Data Manipulation**:

- ``copy()``: Deep copy
- ``view()``: Memory view
- ``astype()``: Type conversion
- ``fill()``: Fill with value
- ``repeat()``: Repeat elements
- ``tile()``: Tile array

**Aggregation**:

- ``min()``, ``max()``: Minimum/maximum values
- ``mean()``, ``std()``: Statistical measures
- ``sum()``, ``prod()``: Reduction operations
- ``any()``, ``all()``: Boolean aggregation

**Sorting and Searching**:

- ``sort()``: Sort array
- ``argsort()``: Sort indices
- ``searchsorted()``: Binary search
- ``partition()``: Partial sort



Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Extensions**:

- Type-specific optimizations possible
- Direct access to type internals
- Can leverage specialized algorithms
- Minimal dispatch overhead

**Mixins**:

- Generic implementations
- May require type adaptation
- Optimized for common patterns
- Slightly higher dispatch overhead

**Recommendation**: Choose based on correctness first, then optimize if needed.



Summary and Best Practices
---------------------------

Key Takeaways for Extension Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Extensions** are the primary way for external users to extend AxisFuzzy
2. **Focus on type-specific** mathematical and fuzzy logic operations
3. **Keep extensions simple** and focused on a single responsibility
4. **Document thoroughly** with clear examples and parameter descriptions
5. **Test comprehensively** including edge cases and performance scenarios
6. **Follow naming conventions** that clearly describe the operation

Development Workflow for Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Identify the fuzzy operation** you want to implement
2. **Determine the target fuzzy type(s)** (mtype parameter)
3. **Design the function signature** with clear parameters
4. **Implement with proper error handling** and type checking
5. **Write comprehensive tests** for various scenarios
6. **Document the extension** with examples and use cases
7. **Consider performance optimization** if needed

Extension System Benefits
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Type Safety**: Automatic dispatch to correct implementation
- **Performance**: Direct access to type-specific optimizations
- **Flexibility**: Easy to add new operations without modifying core
- **Maintainability**: Clear separation between core and custom functionality
- **Extensibility**: Support for future fuzzy types and operations

Conclusion
~~~~~~~~~~

The Extension System provides a powerful and flexible way to extend AxisFuzzy
with custom fuzzy logic operations. By following the guidelines in this document,
you can create robust, efficient, and maintainable extensions that integrate
seamlessly with AxisFuzzy's architecture.

The dual-track architecture ensures that:

- **External users** can easily add custom functionality through extensions
- **Core developers** can maintain universal operations through mixins
- **Both systems** work together to provide a comprehensive fuzzy computing platform

For most users, the pre-built extensions will be sufficient for common fuzzy
operations. Advanced users and researchers can create custom extensions for
specialized domain-specific operations, mathematical functions, and novel
fuzzy logic algorithms.

By leveraging the Extension System effectively, you can build powerful,
maintainable fuzzy computing applications tailored to your specific needs.

See Also
--------

- :doc:`core_data_structures` - Core AxisFuzzy data structures
- :doc:`fuzzy_operations` - Mathematical operations framework
- :doc:`../api/extension/index` - Extension system API reference
- :doc:`../api/mixin/index` - Mixin system API reference