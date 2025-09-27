=======================================
Extension Methods Development Guide
=======================================

This development guide provides comprehensive instructions for implementing
custom extension methods for fuzzy number types in AxisFuzzy. The guide
demonstrates the complete development process using Q-Rung Orthopair Fuzzy
Numbers (QROFN) as a reference example, covering architecture design,
implementation patterns, registration workflows, and integration procedures.

The extension system enables developers to create type-specific methods that
seamlessly integrate with AxisFuzzy's core functionality, providing polymorphic
behavior based on the fuzzy number's mathematical type (mtype). This guide
focuses on the practical aspects of developing, registering, and deploying
extension methods for custom fuzzy number types.

.. note::
   
   **Enhanced External Extension Support**: Starting with AxisFuzzy v0.2.0, external 
   extension registration has been streamlined with new APIs that automatically handle 
   extension injection. See :ref:`external_extension_development` for comprehensive 
   guidance on external extension development.

Extension System Architecture Overview
--------------------------------------

AxisFuzzy's extension system implements a sophisticated **Register-Dispatch-Inject** architecture pattern that enables 
seamless integration of type-specific functionality for fuzzy number types. This system provides a clean separation 
between extension registration, runtime dispatch, and method injection, ensuring both flexibility and performance.

Design Principles and Architectural Pillars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extension system is built upon three fundamental design principles:

1. **Type-based Polymorphism**: Extensions are dispatched based on the ``mtype`` attribute of fuzzy objects, enabling 
   specialized implementations for different fuzzy number types (e.g., ``qrofn``, ``ifn``, ``pfn``).

2. **Declarative Registration**: The ``@extension`` decorator provides a clean, declarative API for registering 
   extension functions without requiring manual registry manipulation.

3. **Flexible Injection**: Extensions can be exposed as instance methods, instance properties, top-level functions, 
   or both, depending on the intended usage pattern.

**Architectural Flow:**

.. code-block:: text

   Registration Phase:    @extension decorator → ExtensionRegistry
   Initialization Phase:  ExtensionRegistry → ExtensionDispatcher → ExtensionInjector
   Runtime Phase:         Method call → Dispatcher → Type-specific implementation

Core Components Overview
~~~~~~~~~~~~~~~~~~~~~~~~~

ExtensionRegistry
+++++++++++++++++

The :class:`~axisfuzzy.extension.registry.ExtensionRegistry` serves as the central repository for all registered 
extension functions. It maintains a thread-safe mapping between function names, fuzzy types (``mtype``), and their 
corresponding implementations.

**Key Features:**

- Thread-safe registration and lookup operations
- Support for both specialized (type-specific) and default implementations
- Priority-based resolution for handling multiple implementations
- Comprehensive metadata storage including target classes and injection types

**Registration Example:**

.. code-block:: python

   from axisfuzzy.extension import extension
   
   @extension(name='distance', mtype='qrofn', target_classes=['Fuzznum'])
   def qrofn_distance(x, y, p=2):
       # QROFN-specific distance implementation
       return ((abs(x.md**x.q - y.md**x.q)**p + abs(x.nmd**x.q - y.nmd**x.q)**p) / 2)**(1/p)

ExtensionDispatcher
+++++++++++++++++++

The :class:`~axisfuzzy.extension.dispatcher.ExtensionDispatcher` creates dynamic proxy callables that resolve 
the correct implementation at runtime based on the ``mtype`` of involved fuzzy objects.

**Proxy Types:**

- **Instance Method Proxies**: Callable as ``obj.method(...)`` with automatic ``mtype`` resolution from ``obj``
- **Instance Property Proxies**: Accessed as ``obj.property`` for read-only computed attributes
- **Top-level Function Proxies**: Callable as ``axisfuzzy.function(obj, ...)`` with ``mtype`` resolution from arguments

**Dispatch Resolution Logic:**

.. code-block:: python

   # Runtime dispatch example
   def dispatched_distance(self, other, **kwargs):
       mtype = self.mtype  # Extract from instance
       impl = registry.get_function('distance', mtype)  # Lookup implementation
       if impl is None:
           impl = registry.get_function('distance', None)  # Fallback to default
       return impl(self, other, **kwargs)  # Invoke resolved implementation

ExtensionInjector
++++++++++++++++++

The :class:`~axisfuzzy.extension.injector.ExtensionInjector` orchestrates the final step of making extensions 
available to users by attaching dispatcher-created proxies to target classes and the module namespace.

**Injection Process:**

1. Scan registry metadata to determine target classes and injection types
2. Create appropriate dispatcher proxies for each extension
3. Attach proxies to ``Fuzznum``, ``Fuzzarray`` classes or ``axisfuzzy`` module namespace
4. Avoid overwriting existing attributes to prevent conflicts

The @extension Decorator API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@extension`` decorator provides the primary interface for registering extension functions. It accepts several 
parameters that control how the extension is registered and exposed:

**Core Parameters:**

.. code-block:: python

   @extension(
       name='function_name',           # Extension name (required)
       mtype='qrofn',                 # Target fuzzy type (None for default)
       target_classes=['Fuzznum'],    # Target classes for injection
       injection_type='both',         # How to expose: 'instance_method', 'instance_property', 
                                     # 'top_level_function', or 'both'
       is_default=False,              # Whether this is a fallback implementation
       priority=0                     # Resolution priority (higher wins)
   )

**Usage Patterns:**

.. code-block:: python

   # Instance method for specific type
   @extension('distance', mtype='qrofn', target_classes=['Fuzznum'])
   def qrofn_distance(x, y): ...
   
   # Instance property
   @extension('score', mtype='qrofn', injection_type='instance_property')
   def qrofn_score(obj): ...
   
   # Top-level function only
   @extension('read_csv', mtype='qrofn', injection_type='top_level_function')
   def qrofn_read_csv(filename): ...
   
   # Default fallback implementation
   @extension('normalize', is_default=True)
   def default_normalize(x): ...

Type-based Polymorphic Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extension system achieves polymorphism through runtime ``mtype`` resolution. When an extension method is called, 
the dispatcher examines the ``mtype`` attribute of the primary fuzzy object to select the appropriate implementation.

**Dispatch Priority:**

1. **Exact Match**: Look for implementation registered with specific ``mtype``
2. **Default Fallback**: Use implementation registered with ``mtype=None`` if available
3. **Error**: Raise informative error listing available types and suggesting alternatives

**Example Dispatch Flow:**

.. code-block:: python

   # User calls: my_qrofn.distance(other_qrofn)
   # 1. Dispatcher extracts mtype='qrofn' from my_qrofn
   # 2. Registry lookup: get_function('distance', 'qrofn')
   # 3. Found qrofn_distance implementation
   # 4. Invoke: qrofn_distance(my_qrofn, other_qrofn)

Integration with Core Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extension system integrates seamlessly with AxisFuzzy's core ``Fuzznum`` and ``Fuzzarray`` classes through 
the injection mechanism. Extensions become first-class methods and properties of these classes, providing a 
natural and intuitive user experience.

**Integration Points:**

- **Fuzznum Class**: Single fuzzy number operations (distance, comparison, properties)
- **Fuzzarray Class**: Array-based operations (aggregation, I/O, broadcasting)
- **Module Namespace**: Factory functions and utilities (constructors, file I/O)

This architecture ensures that custom fuzzy types can leverage the full power of AxisFuzzy's extension ecosystem 
while maintaining clean separation of concerns and optimal runtime performance.

Extension Method Implementation Development
-------------------------------------------

This section provides a comprehensive guide for implementing extension methods using QROFN (q-rung Orthopair Fuzzy Numbers) 
as a practical example. The implementation follows a structured approach that ensures consistency, maintainability, and 
optimal performance.

Extension Method Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AxisFuzzy extension methods are organized into five primary categories, each serving distinct computational purposes:

**Constructor Methods**
    Create new fuzzy objects with specific initialization patterns. These methods provide convenient factory functions 
    for common object creation scenarios.

**I/O Operations**
    Handle data serialization and deserialization across multiple formats (CSV, JSON, NumPy binary). These methods 
    enable seamless data exchange and persistence.

**Mathematical Operations**
    Implement aggregation functions and statistical computations using fuzzy-specific algorithms that respect 
    the mathematical properties of each fuzzy type.

**Measurement Functions**
    Calculate distances, similarities, and other metrics between fuzzy objects using type-appropriate formulas.

**Property Accessors**
    Provide computed properties that extract meaningful characteristics from fuzzy objects, such as scores and 
    indeterminacy measures.

Implementation Structure: ext/ Directory Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The QROFN implementation demonstrates the recommended modular organization pattern:

.. code-block:: text

   axisfuzzy/fuzztype/qrofs/
   ├── ext/
   │   ├── constructor.py    # Factory methods for object creation
   │   ├── io.py            # Serialization and data exchange
   │   ├── ops.py           # Mathematical and aggregation operations
   │   ├── measure.py       # Distance and similarity calculations
   │   └── string.py        # String parsing and conversion utilities
   └── extension.py         # Extension registration and decorator usage

This modular structure promotes code reusability, simplifies maintenance, and provides clear separation of concerns.

Core Extension Method Types with QROFN Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table summarizes all required extension methods for a complete fuzzy type implementation:

.. list-table:: QROFN Extension Methods Reference
   :header-rows: 1
   :widths: 20 15 15 50

   * - Method Name
     - Category
     - Injection Type
     - Purpose
   * - ``empty``
     - Constructor
     - Top-level Function
     - Create uninitialized QROFN objects
   * - ``positive``
     - Constructor
     - Top-level Function
     - Create objects with maximum membership (md=1, nmd=0)
   * - ``negative``
     - Constructor
     - Top-level Function
     - Create objects with maximum non-membership (md=0, nmd=1)
   * - ``full``
     - Constructor
     - Top-level Function
     - Create objects filled with specific values
   * - ``empty_like``
     - Constructor
     - Top-level Function
     - Create uninitialized objects matching input shape
   * - ``positive_like``
     - Constructor
     - Top-level Function
     - Create positive objects matching input shape
   * - ``negative_like``
     - Constructor
     - Top-level Function
     - Create negative objects matching input shape
   * - ``full_like``
     - Constructor
     - Top-level Function
     - Create filled objects matching input shape
   * - ``to_csv``
     - I/O Operations
     - Instance Method
     - Export fuzzy arrays to CSV format
   * - ``read_csv``
     - I/O Operations
     - Top-level Function
     - Import fuzzy arrays from CSV files
   * - ``to_json``
     - I/O Operations
     - Instance Method
     - Export fuzzy arrays to JSON format
   * - ``read_json``
     - I/O Operations
     - Top-level Function
     - Import fuzzy arrays from JSON files
   * - ``to_npy``
     - I/O Operations
     - Instance Method
     - Export fuzzy arrays to NumPy binary format
   * - ``read_npy``
     - I/O Operations
     - Top-level Function
     - Import fuzzy arrays from NumPy binary files
   * - ``sum``
     - Mathematical
     - Instance Method
     - Aggregate using t-conorm reduction
   * - ``mean``
     - Mathematical
     - Instance Method
     - Calculate fuzzy arithmetic mean
   * - ``max``
     - Mathematical
     - Instance Method
     - Find maximum based on score function
   * - ``min``
     - Mathematical
     - Instance Method
     - Find minimum based on score function
   * - ``prod``
     - Mathematical
     - Instance Method
     - Aggregate using t-norm reduction
   * - ``var``
     - Mathematical
     - Instance Method
     - Calculate fuzzy variance
   * - ``std``
     - Mathematical
     - Instance Method
     - Calculate fuzzy standard deviation
   * - ``distance``
     - Measurement
     - Instance Method
     - Compute distance between fuzzy objects
   * - ``score``
     - Property
     - Instance Property
     - Calculate membership score (md^q - nmd^q)
   * - ``acc``
     - Property
     - Instance Property
     - Calculate accuracy measure
   * - ``ind``
     - Property
     - Instance Property
     - Calculate indeterminacy degree
   * - ``str2fuzznum``
     - String Conversion
     - Top-level Function
     - Parse string representation to Fuzznum

Implementation Patterns and Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Performance Optimization**
    Leverage backend component arrays directly for vectorized operations. The QROFN implementation demonstrates 
    this pattern in I/O operations:

.. code-block:: python

   def _qrofn_to_csv(arr: Fuzzarray, path: str, **kwargs) -> None:
       # Get component arrays directly from backend for efficiency
       mds, nmds = arr.backend.get_component_arrays()
       
       # Use vectorized string operations
       str_data = np.char.add(
           np.char.add('<', mds.astype(str)),
           np.char.add(',', np.char.add(nmds.astype(str), '>'))
       )

**Type-Specific Algorithm Integration**
    Mathematical operations should utilize the appropriate t-norm/t-conorm operations for the fuzzy type:

.. code-block:: python

   def _qrofn_sum(arr: Union[Fuzznum, Fuzzarray], axis=None):
       op_registry = get_registry_operation()
       norm_type, params = op_registry.get_default_t_norm_config()
       tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)
       
       mds, nmds = arr.backend.get_component_arrays()
       md_sum = tnorm.t_conorm_reduce(mds, axis=axis)
       nmd_sum = tnorm.t_norm_reduce(nmds, axis=axis)

**Error Handling and Validation**
    Implement comprehensive input validation and provide meaningful error messages:

.. code-block:: python

   def _qrofn_distance(fuzz_1, fuzz_2, p_l=2, indeterminacy=True):
       if fuzz_1.q != fuzz_2.q:
           raise ValueError(f"Q-rung mismatch: {fuzz_1.q} != {fuzz_2.q}")
       if fuzz_1.mtype != fuzz_2.mtype:
           raise ValueError(f"Type mismatch: {fuzz_1.mtype} != {fuzz_2.mtype}")

Type-Specific Algorithm Development Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When developing algorithms for custom fuzzy types, consider these essential principles:

1. **Mathematical Consistency**: Ensure all operations respect the mathematical constraints of your fuzzy type
2. **Backend Integration**: Utilize the backend's component array access for optimal performance
3. **Axis-Aware Operations**: Support axis-specific reductions for multi-dimensional arrays
4. **Fallback Handling**: Provide graceful degradation for edge cases (empty arrays, single elements)
5. **Parameter Validation**: Validate type-specific parameters (e.g., q-rung values for QROFN)

Extension Registration and Integration Workflow
------------------------------------------------

This section outlines the complete workflow for registering and integrating extension methods into the AxisFuzzy framework. 
The registration process transforms individual implementation functions into dynamically accessible methods and properties 
on core fuzzy objects.

Extension Method Registration Using @extension Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@extension`` decorator serves as the primary interface for registering extension functions. Each extension function 
must be wrapped with this decorator to become part of the AxisFuzzy extension system.

**Basic Registration Pattern**

.. code-block:: python

   from axisfuzzy.extension import extension
   from .ext import _qrofn_sum  # Import the implementation function
   
   @extension(
       name='sum',
       mtype='qrofn',
       target_classes=['Fuzzarray', 'Fuzznum']
   )
   def qrofn_sum_ext(fuzz, axis=None):
       """Aggregate QROFN values using t-conorm reduction."""
       return _qrofn_sum(fuzz, axis=axis)

**Decorator Parameter Configuration**

The ``@extension`` decorator accepts several parameters that control registration behavior:

.. list-table:: @extension Decorator Parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``name``
     - str (required)
     - The method/function name as it will appear to users
   * - ``mtype``
     - str or None
     - Target fuzzy type ('qrofn', 'ivfs', etc.). None for default implementations
   * - ``target_classes``
     - List[str]
     - Classes to inject into: ['Fuzznum'], ['Fuzzarray'], or ['Fuzznum', 'Fuzzarray']
   * - ``injection_type``
     - str
     - How to expose: 'instance_method', 'instance_property', 'top_level_function', or 'both'
   * - ``is_default``
     - bool
     - Whether this serves as a fallback implementation for unspecified mtypes
   * - ``priority``
     - int
     - Resolution priority when multiple implementations exist (higher wins)

Parameter Configuration for Different Injection Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Instance Method Injection**
    Creates methods callable on Fuzznum/Fuzzarray instances. This is the most common injection type for operations 
    that act on existing fuzzy objects:

.. code-block:: python

   @extension(name='distance', mtype='qrofn', injection_type='instance_method')
   def qrofn_distance_ext(self, other, p_l=2, indeterminacy=True):
       return _qrofn_distance(self, other, p_l, indeterminacy)
   
   # Usage: fuzz_obj.distance(other_obj, p_l=3)

**Instance Property Injection**
    Creates read-only properties accessible via attribute access. Ideal for computed characteristics:

.. code-block:: python

   @extension(name='score', mtype='qrofn', injection_type='instance_property')
   def qrofn_score_ext(self):
       return _qrofn_score(self)
   
   # Usage: fuzz_obj.score

**Top-level Function Injection**
    Creates functions in the axisfuzzy module namespace. Used for constructor functions and static operations:

.. code-block:: python

   @extension(name='empty', mtype='qrofn', injection_type='top_level_function')
   def qrofn_empty_ext(shape=None, q=None):
       return _qrofn_empty(shape, q)
   
   # Usage: axisfuzzy.empty(shape=(3, 3), q=2)

**Both Injection Type**
    Combines instance method and top-level function injection for maximum accessibility:

.. code-block:: python

   @extension(name='sum', mtype='qrofn', injection_type='both')
   def qrofn_sum_ext(fuzz, axis=None):
       return _qrofn_sum(fuzz, axis)
   
   # Usage: fuzz_obj.sum(axis=0) or axisfuzzy.sum(fuzz_obj, axis=0)

Integration with Target Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extension system integrates with AxisFuzzy's core classes through dynamic method injection. The integration process 
respects the existing class hierarchy and avoids conflicts with built-in methods.

**Target Class Specification**
    Use ``target_classes`` to control which classes receive the extension:

.. code-block:: python

   # Only inject into Fuzzarray (for array-specific operations)
   @extension(name='to_csv', mtype='qrofn', target_classes=['Fuzzarray'])
   
   # Inject into both classes (for universal operations)
   @extension(name='sum', mtype='qrofn', target_classes=['Fuzznum', 'Fuzzarray'])

**Method Resolution and Dispatch**
    At runtime, the extension system automatically resolves the correct implementation based on the object's ``mtype``:

.. code-block:: python

   qrofn_obj = Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.3)
   ivfs_obj = Fuzznum('ivfs').create(lower=0.6, upper=0.9)
   
   # Automatically dispatches to QROFN-specific implementation
   qrofn_score = qrofn_obj.score
   
   # Automatically dispatches to IVFS-specific implementation  
   ivfs_score = ivfs_obj.score

The apply_extensions() Function and Its Critical Role
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``apply_extensions()`` function serves as the master activation switch for the entire extension system. This function 
must be called to make registered extensions available to users.

**Function Signature and Purpose**

.. code-block:: python

   def apply_extensions() -> bool:
       """
       Applies all registered extension functions to their respective targets.
       
       Returns:
           bool: True if extensions were applied successfully, False otherwise.
       """

**Integration Process**
    The function performs these critical steps:

1. **Dynamic Class Discovery**: Locates ``Fuzznum`` and ``Fuzzarray`` classes at runtime to avoid circular imports
2. **Module Namespace Resolution**: Identifies the ``axisfuzzy`` module for top-level function injection
3. **Extension Injection**: Delegates to ``ExtensionInjector`` to attach all registered extensions
4. **Idempotency Guarantee**: Ensures safe multiple calls without duplicate injection

**Typical Usage Pattern**
    The function is automatically called during AxisFuzzy initialization, but can be invoked manually when needed:

.. code-block:: python

   # Automatic call during import (typical case)
   import axisfuzzy  # apply_extensions() called internally
   
   # Manual call after registering new extensions
   from axisfuzzy.extension import apply_extensions
   
   # Register your custom extensions here...
   
   # Activate the extensions
   success = apply_extensions()
   if not success:
       print("Warning: Extension application failed")

Testing and Validation of Registered Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive testing ensures that registered extensions function correctly and integrate seamlessly with the framework.

**Basic Functionality Testing**

.. code-block:: python

   def test_qrofn_sum_extension():
       """Test QROFN sum extension registration and functionality."""
       # Create test data
       arr = axisfuzzy.empty((2, 2), mtype='qrofn', q=2)
       arr[0, 0] = Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.2)
       arr[0, 1] = Fuzznum('qrofn', q=2).create(md=0.6, nmd=0.3)
       
       # Test instance method access
       result = arr.sum(axis=0)
       assert isinstance(result, Fuzzarray)
       assert result.mtype == 'qrofn'
       
       # Test top-level function access
       result2 = axisfuzzy.sum(arr, axis=0)
       assert np.allclose(result.md, result2.md)

**Integration Testing**

.. code-block:: python

   def test_extension_injection_completeness():
       """Verify all required extensions are properly injected."""
       required_methods = ['sum', 'mean', 'max', 'min', 'distance']
       required_properties = ['score', 'acc', 'ind']
       required_functions = ['empty', 'positive', 'negative', 'read_csv']
       
       # Test instance methods
       fuzz_obj = Fuzznum('qrofn', q=2).create(md=0.7, nmd=0.2)
       for method in required_methods:
           assert hasattr(fuzz_obj, method), f"Missing method: {method}"
       
       # Test instance properties  
       for prop in required_properties:
           assert hasattr(fuzz_obj, prop), f"Missing property: {prop}"
       
       # Test top-level functions
       import axisfuzzy
       for func in required_functions:
           assert hasattr(axisfuzzy, func), f"Missing function: {func}"

Deployment Considerations and Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Successfully deploying extension methods requires careful attention to several critical aspects that can significantly 
impact both development efficiency and runtime performance. This section provides practical guidance based on real-world 
experience developing the QROFN extension system.

Managing Extension Loading and Initialization
++++++++++++++++++++++++++++++++++++++++++++++

The timing of extension registration is crucial for proper system initialization. Extensions must be registered before 
the ``apply_extensions()`` function is called, which typically occurs during AxisFuzzy's import process. The QROFN 
implementation demonstrates the recommended approach:

.. code-block:: python

   # In axisfuzzy/fuzztype/qrofs/extension.py
   from axisfuzzy.extension import extension
   from .ext import (
       _qrofn_sum, _qrofn_mean, _qrofn_distance,
       _qrofn_empty, _qrofn_to_csv, _qrofn_from_str
   )
   
   # All extensions are registered at module import time
   @extension(name='sum', mtype='qrofn', target_classes=['Fuzzarray', 'Fuzznum'])
   def qrofn_sum_ext(fuzz, axis=None):
       return _qrofn_sum(fuzz, axis=axis)
   
   # Additional registrations follow...

This pattern ensures that when users import AxisFuzzy, all QROFN extensions are immediately available. The key insight 
is that extension registration happens at import time, but the actual method injection occurs later when 
``apply_extensions()`` is called.

**Debugging Extension Loading Issues**

When extensions don't appear to be working, the most common cause is import order problems. You can verify extension 
registration status by checking the registry directly:

.. code-block:: python

   from axisfuzzy.extension.registry import get_extension_registry
   
   registry = get_registry_extension()
   print(f"Registered functions: {list(registry.list_functions())}")
   print(f"QROFN sum available: {registry.get_metadata('sum', 'qrofn')}")

Optimizing Performance in Extension Functions
++++++++++++++++++++++++++++++++++++++++++++++

Extension functions often become performance bottlenecks because they're called frequently during computations. 
The QROFN implementation incorporates several optimization strategies that significantly improve runtime performance.

**Leveraging Backend Component Arrays**

Direct access to backend component arrays eliminates unnecessary object creation and enables vectorized operations. 
The actual implementation in ``ops.py`` demonstrates this pattern:

.. code-block:: python

   def _qrofn_sum(arr: Union[Fuzznum, Fuzzarray], axis=None):
       # Efficient: Direct backend access for component arrays
       mds, nmds = arr.backend.get_component_arrays()
       
       # Use t-norm/t-conorm operations for proper fuzzy aggregation
       op_registry = get_registry_operation()
       norm_type, params = op_registry.get_default_t_norm_config()
       tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)
       
       md_sum = tnorm.t_conorm_reduce(mds, axis=axis)
       nmd_sum = tnorm.t_norm_reduce(nmds, axis=axis)
       
       # Return appropriate type based on axis parameter
       if axis is None:
           return Fuzznum('qrofn', q=arr.q).create(md=md_sum, nmd=nmd_sum)
       else:
           backend_cls = arr.backend.__class__
           new_backend = backend_cls.from_arrays(md_sum, nmd_sum, q=arr.q)
           return Fuzzarray(backend=new_backend)

**Efficient String-Based I/O Operations**

The CSV I/O implementation in ``io.py`` uses efficient string operations without external dependencies:

.. code-block:: python

   def _qrofn_to_csv(arr: Fuzzarray, path: str, **kwargs) -> None:
       """High-performance CSV export using backend arrays directly."""
       # Get component arrays directly from backend
       mds, nmds = arr.backend.get_component_arrays()
       
       # Create string representation efficiently using numpy char operations
       str_data = np.char.add(
           np.char.add('<', mds.astype(str)),
           np.char.add(',', np.char.add(nmds.astype(str), '>'))
       )
       
       # Write directly to CSV without pandas dependency
       with open(path, 'w', newline='', encoding='utf-8') as f:
           writer = csv.writer(f, **kwargs)
           if str_data.ndim == 1:
               writer.writerow(str_data)
           else:
               writer.writerows(str_data)

**Vectorized Distance Computations**

The distance calculation in ``measure.py`` demonstrates efficient vectorized operations for different input combinations:

.. code-block:: python

   def _qrofn_distance(fuzz_1, fuzz_2, p_l=2, indeterminacy=True):
       """High-performance distance calculation with vectorized operations."""
       # Handle Fuzzarray vs Fuzzarray case with full vectorization
       if isinstance(fuzz_1, Fuzzarray) and isinstance(fuzz_2, Fuzzarray):
           mds1, nmds1 = fuzz_1.backend.get_component_arrays()
           mds2, nmds2 = fuzz_2.backend.get_component_arrays()
           
           # Vectorized indeterminacy calculation
           pi1 = (1 - mds1 ** q - nmds1 ** q) ** (1 / q)
           pi2 = (1 - mds2 ** q - nmds2 ** q) ** (1 / q)
           pi = np.abs(pi1 ** q - pi2 ** q) ** p_l
           
           # Vectorized distance computation
           if indeterminacy:
               distance = (0.5 * (np.abs(mds1 ** q - mds2 ** q) ** p_l +
                                  np.abs(nmds1 ** q - nmds2 ** q) ** p_l + pi)) ** (1 / p_l)
           return distance

Robust Error Handling and User Guidance
++++++++++++++++++++++++++++++++++++++++

Extension functions should provide clear, actionable error messages that help users understand and resolve issues quickly. 
The QROFN implementation demonstrates several effective error handling patterns.

**Parameter Validation with Contextual Messages**

.. code-block:: python

   def _qrofn_distance(fuzz_1, fuzz_2, p_l=2, indeterminacy=True):
       # Validate q-rung compatibility
       if fuzz_1.q != fuzz_2.q:
           raise ValueError(
               f"Cannot compute distance between QROFN objects with different q-rungs: "
               f"{fuzz_1.q} and {fuzz_2.q}. Consider converting to the same q-rung first."
           )
       
       # Validate distance parameter
       if p_l <= 0:
           raise ValueError(
               f"Distance parameter p_l must be positive, got {p_l}. "
               f"Common values are 1 (Manhattan), 2 (Euclidean), or inf (Chebyshev)."
           )

**Graceful Handling of Edge Cases**

.. code-block:: python

   def _qrofn_mean(arr: Union[Fuzznum, Fuzzarray], axis=None):
       if arr.size == 0:
           raise ValueError(
               "Cannot compute mean of empty array. "
               "Use axisfuzzy.empty() to create arrays with default values."
           )
       
       if arr.size == 1:
           # Single element case - return copy to maintain consistency
           return arr.copy()
       
       # Normal computation for multiple elements
       mds, nmds = arr.backend.get_component_arrays()
       # ... rest of implementation

This comprehensive approach to deployment ensures that your extension methods integrate seamlessly with AxisFuzzy 
while providing a robust, performant, and maintainable foundation for users.

.. _external_extension_development:

External Extension Development: Quick Guide for Third-Party Developers
----------------------------------------------------------------------

This guide shows how to create and deploy external extensions for AxisFuzzy 
libraries and applications. External extensions allow you to add custom 
functionality without modifying the core AxisFuzzy codebase.

Quick Start: Creating External Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Choose the Right Decorator**

For external projects, use ``@external_extension`` (recommended):

.. code-block:: python

    # my_fuzzy_extensions.py
    import axisfuzzy as af
    from axisfuzzy.extension import external_extension
    
    @external_extension('custom_distance', mtype='qrofn')
    def my_distance(self, other):
        """Custom distance function."""
        return abs(self.md - other.md) + abs(self.nmd - other.nmd)
    
    # Automatically available - no setup needed!
    fuzz1 = af.Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.3)
    fuzz2 = af.Fuzznum('qrofn', q=2).create(md=0.6, nmd=0.4)
    dist = fuzz1.custom_distance(fuzz2)

**Step 2: Choose Injection Type**

.. code-block:: python

    # Instance method (default)
    @external_extension('my_method', mtype='qrofn')
    def method_func(self):
        return self.md + self.nmd
    
    # Usage: fuzz.my_method()
    
    # Top-level function
    @external_extension('my_function', mtype='qrofn', 
                        injection_type='top_level_function')
    def function_func(x, y):
        return x.md * y.md
    
    # Usage: af.my_function(fuzz1, fuzz2)
    
    # Instance property
    @external_extension('my_property', mtype='qrofn', 
                        injection_type='instance_property')
    def property_func(self):
        return self.md ** 2
    
    # Usage: fuzz.my_property (no parentheses)
    
    # Both method and function
    @external_extension('my_both', mtype='qrofn', injection_type='both')
    def both_func(x):
        return x.md * 2
    
    # Usage: fuzz.my_both() OR af.my_both(fuzz)

Common Use Cases
~~~~~~~~~~~~~~~~

**Custom Similarity Measures**:

.. code-block:: python

    @external_extension('cosine_similarity', mtype='qrofn')
    def cosine_sim(self, other):
        """Cosine similarity for QROFN."""
        numerator = self.md * other.md + self.nmd * other.nmd
        denom = ((self.md**2 + self.nmd**2) * (other.md**2 + other.nmd**2))**0.5
        return numerator / denom if denom > 0 else 0

**Custom Aggregation Functions**:

.. code-block:: python

    @external_extension('weighted_mean', mtype='qrofn', 
                        injection_type='top_level_function')
    def weighted_aggregation(fuzzy_list, weights):
        """Weighted mean aggregation."""
        total_weight = sum(weights)
        md_sum = sum(f.md * w for f, w in zip(fuzzy_list, weights))
        nmd_sum = sum(f.nmd * w for f, w in zip(fuzzy_list, weights))
        return af.Fuzznum('qrofn', q=fuzzy_list[0].q).create(
            md=md_sum/total_weight, nmd=nmd_sum/total_weight)

**Custom Properties**:

.. code-block:: python

    @external_extension('entropy', mtype='qrofn', 
                        injection_type='instance_property')
    def qrofn_entropy(self):
        """Calculate entropy measure."""
        import math
        if self.md > 0 and self.nmd > 0:
            return -(self.md * math.log(self.md) + self.nmd * math.log(self.nmd))
        return 0

Library Packaging
~~~~~~~~~~~~~~~~~

**Package Structure**:

.. code-block:: text

    my_fuzzy_lib/
    ├── __init__.py          # Main package
    ├── extensions.py        # Extension definitions
    ├── utils.py            # Helper functions
    └── tests/              # Test suite
        └── test_extensions.py

**Main Package (``__init__.py``)**:

.. code-block:: python

    # my_fuzzy_lib/__init__.py
    """My Fuzzy Extensions Library."""
    
    __version__ = "1.0.0"
    
    # Import extensions to register them
    from . import extensions
    
    # Verify AxisFuzzy is available
    try:
        import axisfuzzy as af
    except ImportError:
        raise ImportError("my_fuzzy_lib requires axisfuzzy")
    
    # Optional: verify extensions loaded
    def verify_extensions():
        """Check if extensions are available."""
        from axisfuzzy.extension import apply_extensions
        return apply_extensions(force_reapply=True)

**Extensions Module (``extensions.py``)**:

.. code-block:: python

    # my_fuzzy_lib/extensions.py
    """Extension definitions."""
    
    import axisfuzzy as af
    from axisfuzzy.extension import external_extension
    
    @external_extension('lib_distance', mtype='qrofn')
    def custom_distance(self, other, method='euclidean'):
        """Custom distance with multiple methods."""
        if method == 'euclidean':
            return ((self.md - other.md)**2 + (self.nmd - other.nmd)**2)**0.5
        elif method == 'manhattan':
            return abs(self.md - other.md) + abs(self.nmd - other.nmd)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @external_extension('lib_score', mtype='qrofn', 
                        injection_type='instance_property')
    def custom_score(self):
        """Custom scoring function."""
        return self.md**self.q - self.nmd**self.q

Advanced Options
~~~~~~~~~~~~~~~~

**Priority Control**:

.. code-block:: python

    # Higher priority overrides existing implementations
    @external_extension('distance', mtype='qrofn', priority=10)
    def improved_distance(self, other):
        return "Better distance algorithm"

**Manual Application**:

.. code-block:: python

    # Defer automatic application
    @external_extension('batch_method', mtype='qrofn', auto_apply=False)
    def batch_operation(self):
        return "Batch processing"
    
    # Apply when ready
    from axisfuzzy.extension import apply_extensions
    apply_extensions(force_reapply=True)

**Conditional Extensions**:

.. code-block:: python

    # Only register if dependencies available
    try:
        import numpy as np
        
        @external_extension('numpy_op', mtype='qrofn')
        def numpy_operation(self):
            return np.array([self.md, self.nmd])
    except ImportError:
        pass  # Skip if NumPy not available

Best Practices
~~~~~~~~~~~~~~

1. **Use descriptive names**: Choose names that clearly indicate functionality
2. **Add docstrings**: Document parameters, returns, and examples
3. **Handle errors**: Check inputs and provide meaningful error messages
4. **Test thoroughly**: Test with different fuzzy types and edge cases
5. **Version compatibility**: Specify minimum AxisFuzzy version requirements

Testing Your Extensions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # test_extensions.py
    import pytest
    import axisfuzzy as af
    
    def test_custom_distance():
        """Test custom distance function."""
        fuzz1 = af.Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.3)
        fuzz2 = af.Fuzznum('qrofn', q=2).create(md=0.6, nmd=0.4)
        
        # Test method exists
        assert hasattr(fuzz1, 'custom_distance')
        
        # Test functionality
        dist = fuzz1.custom_distance(fuzz2)
        assert isinstance(dist, float)
        assert dist >= 0
    
    def test_custom_score():
        """Test custom score property."""
        fuzz = af.Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.3)
        
        # Test property exists
        assert hasattr(fuzz, 'custom_score')
        
        # Test value
        score = fuzz.custom_score
        assert isinstance(score, float)

Deployment
~~~~~~~~~~

**Installation Order**:

.. code-block:: bash

    pip install axisfuzzy>=0.2.0
    pip install my-fuzzy-lib

**Usage**:

.. code-block:: python

    import axisfuzzy as af
    import my_fuzzy_lib  # Extensions auto-register
    
    # Use extensions immediately
    fuzz = af.Fuzznum('qrofn', q=2).create(md=0.8, nmd=0.3)
    result = fuzz.custom_distance(other_fuzz)

Conclusion
----------

This development guide demonstrates the complete workflow for implementing custom fuzzy type extensions in AxisFuzzy. 
Using QROFN as a reference implementation, developers can follow the established patterns to integrate new fuzzy 
number types seamlessly.

The extension development process follows three essential phases: **Implementation** (creating type-specific algorithms 
in modular ext/ files), **Registration** (using ``@extension`` decorators with appropriate parameters), and 
**Integration** (calling ``apply_extensions()`` to inject methods into core classes).

Key implementation requirements include 22 core extension methods spanning constructors, I/O operations, mathematical 
functions, measurements, and properties. The Register-Dispatch-Inject architecture ensures type-safe polymorphic 
behavior while maintaining optimal performance through backend component array access and vectorized operations.

Successful extension development requires adherence to AxisFuzzy's architectural principles: modular organization, 
comprehensive error handling, performance optimization, and thorough testing. The ``apply_extensions()`` function 
serves as the critical integration point, transforming individual implementations into accessible instance methods 
and top-level functions.

.. note::
   For foundational concepts, refer to :doc:`../user_guide/core_data_structures` and 
   :doc:`../user_guide/extension_mixin` documentation.

.. warning::
   Extension methods directly modify core class behavior. Comprehensive testing and validation are mandatory 
   before production deployment.