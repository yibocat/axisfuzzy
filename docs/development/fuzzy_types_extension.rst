=========================
Fuzzy Types Extension
=========================

This development guide provides comprehensive instructions for extending AxisFuzzy with custom fuzzy number types. 
The guide demonstrates the complete implementation process using Q-Rung Orthopair Fuzzy Numbers (QROFN) as a 
reference example, covering both the strategy pattern implementation and backend architecture.

Overview
--------

AxisFuzzy employs a sophisticated architecture that separates the user interface layer from the computational 
backend through the Strategy Pattern. This design enables developers to create new fuzzy number types by 
implementing two core components:

1. **FuzznumStrategy Subclass**: Defines the mathematical properties, validation rules, and user interface behavior
2. **FuzzarrayBackend Subclass**: Implements the high-performance computational backend using Structure-of-Arrays (SoA) architecture

This modular approach ensures that new fuzzy types integrate seamlessly with AxisFuzzy's existing operation 
framework while maintaining optimal performance characteristics.

Creating FuzznumStrategy Subclass
---------------------------------

The foundation of any custom fuzzy number type in AxisFuzzy is the 
:class:`~axisfuzzy.core.base.FuzznumStrategy` subclass. This section demonstrates how 
to implement a complete strategy class using the Q-Rung Orthopair Fuzzy Number (QROFN) 
as our reference example.

Conceptual Foundation
~~~~~~~~~~~~~~~~~~~~~

The :class:`~axisfuzzy.core.base.FuzznumStrategy` abstract base class provides the core 
infrastructure for single-element fuzzy number implementations. It enforces a strict 
attribute declaration contract, provides validation frameworks, and offers operation 
dispatch capabilities with caching support.

**Basic Structure Example:**

.. code-block:: python

   from typing import Optional, Any
   import numpy as np
   from axisfuzzy.core import FuzznumStrategy, register_strategy
   from axisfuzzy.config import get_config
   
   @register_strategy
   class QROFNStrategy(FuzznumStrategy):
       """Q-Rung Orthopair Fuzzy Number Strategy Implementation."""
       
       # Attribute declarations using type annotations
       mtype = 'qrofn'
       md: Optional[float] = None
       nmd: Optional[float] = None

Attribute Declaration System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AxisFuzzy uses type annotations and class-level assignments for attribute declarations. 
The framework automatically collects these declarations during class creation via ``__init_subclass__``.

**Attribute Declaration Patterns:**

.. code-block:: python

   class QROFNStrategy(FuzznumStrategy):
       # Core fuzzy components with type annotations
       mtype = 'qrofn'
       md: Optional[float] = None
       nmd: Optional[float] = None
       
       # Inherited from base class
       q: Optional[int] = None

**Initialization and Validator Registration:**

.. code-block:: python

   def __init__(self, q: Optional[int] = None):
       super().__init__(q=q)
       
       # Register attribute validators using lambda functions
       self.add_attribute_validator(
           'md', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)
       self.add_attribute_validator(
           'nmd', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)

Validation Framework Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The validation framework operates through the ``add_attribute_validator()`` method, 
which registers callable validators that are automatically invoked during attribute assignment.

**Validator Registration Pattern:**

.. code-block:: python

   def __init__(self, q: Optional[int] = None):
       super().__init__(q=q)
       
       # Range validation for membership degree
       self.add_attribute_validator(
           'md', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)
       
       # Range validation for non-membership degree  
       self.add_attribute_validator(
           'nmd', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)

Constraint Implementation Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex fuzzy number types often require cross-attribute constraints. The QROFN 
constraint :math:`\mu^q + \nu^q \leq 1` exemplifies this pattern.

**Instance-Level Constraint Validation:**

.. code-block:: python

   def _fuzz_constraint(self):
       """Validate q-rung orthopair constraints."""
       if self.md is not None and self.nmd is not None and self.q is not None:
           sum_of_powers = self.md ** self.q + self.nmd ** self.q
           if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
               raise ValueError(
                   f"violates fuzzy number constraints: "
                   f"md^q ({self.md}^{self.q}) + nmd^q ({self.nmd}^{self.q})"
                   f"={sum_of_powers: .4f} > 1.0."
                   f"(q: {self.q}, md: {self.md}, nmd: {self.nmd})")

**Override Base Validation Method:**

.. code-block:: python

   def _validate(self) -> None:
       """Override base validation to include fuzzy constraints."""
       super()._validate()
       self._fuzz_constraint()

Change Callbacks and Reactive Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework supports reactive programming patterns through change callbacks registered via ``add_change_callback()`` that trigger when attribute values are modified.

**Implementing Change Callbacks:**

.. code-block:: python

   def __init__(self, q: Optional[int] = None):
       super().__init__(q=q)
       
       # Register validators first
       self.add_attribute_validator('md', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)
       self.add_attribute_validator('nmd', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)
       
       # Register change callbacks
       self.add_change_callback('md', self._on_membership_change)
       self.add_change_callback('nmd', self._on_membership_change)
       self.add_change_callback('q', self._on_q_change)

**Callback Implementation:**

.. code-block:: python

   def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
       """Callback triggered when membership or non-membership degree changes."""
       if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
           self._fuzz_constraint()
   
   def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
       """Callback triggered when q parameter changes."""
       if self.md is not None and self.nmd is not None and new_value is not None:
           self._fuzz_constraint()

Formatting and String Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Proper string representation is crucial for debugging and user interaction. The framework provides multiple formatting options through specific methods.

**Standard Representation Methods:**

.. code-block:: python

   def format_from_components(self, md: float, nmd: float, format_spec: str = "") -> str:
       """Format fuzzy number from component values."""
       if md is None and nmd is None:
           return "<>"
       precision = get_config().DEFAULT_PRECISION
       if format_spec == 'p':
           return f"({md}, {nmd})"
       if format_spec == 'j':
           import json
           return json.dumps({'mtype': self.mtype, 'md': md, 'nmd': nmd, 'q': self.q})
       
       def strip_trailing_zeros(x: float) -> str:
           s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
           return s if s else "0"
       
       md_str = strip_trailing_zeros(md)
       nmd_str = strip_trailing_zeros(nmd)
       return f"<{md_str},{nmd_str}>"
   
   def report(self) -> str:
       """Generate report representation."""
       return self.format_from_components(self.md, self.nmd)
   
   def str(self) -> str:
       """Generate string representation."""
       return self.format_from_components(self.md, self.nmd)
   
   def __format__(self, format_spec: str) -> str:
       """Custom formatting support."""
       if format_spec and format_spec not in ['r', 'p', 'j']:
           return format(self.str(), format_spec)
       return self.format_from_components(self.md, self.nmd, format_spec)


Implementing FuzzarrayBackend
-----------------------------------------

The :class:`~axisfuzzy.core.backend.FuzzarrayBackend` provides high-performance array storage 
using the Struct-of-Arrays (SoA) architecture. This section demonstrates how to implement 
a concrete backend for your custom fuzzy number type, using :class:`~axisfuzzy.fuzztype.qrofs.backend.QROFNBackend` 
as the reference implementation.

SoA Architecture Design
~~~~~~~~~~~~~~~~~~~~~~~

The SoA pattern separates fuzzy number components into independent NumPy arrays, enabling 
vectorized operations and memory-efficient storage. For QROFN, this means storing membership 
degrees and non-membership degrees in separate arrays.

.. code-block:: python

   from typing import Any, Tuple, Optional, Callable
   import numpy as np
   from axisfuzzy.core import FuzzarrayBackend, register_backend

   @register_backend
   class QROFNBackend(FuzzarrayBackend):
       """SoA backend for q-rung orthopair fuzzy numbers."""
       
       mtype = 'qrofn'
       
       def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
           super().__init__(shape, q, **kwargs)

The backend inherits from :class:`~axisfuzzy.core.backend.FuzzarrayBackend` and must define 
the ``mtype`` class attribute to identify the fuzzy number type it supports.

Component Properties
~~~~~~~~~~~~~~~~~~~~

Define the structural properties that describe your fuzzy number's components:

.. code-block:: python

   @property
   def cmpnum(self) -> int:
       """Number of components (2 for QROFN: md, nmd)."""
       return 2

   @property
   def cmpnames(self) -> Tuple[str, ...]:
       """Component names for display and access."""
       return 'md', 'nmd'

   @property
   def dtype(self) -> np.dtype:
       """Data type for component arrays."""
       return np.dtype(np.float64)

These properties inform the framework about your fuzzy number's structure and enable 
proper array initialization and element formatting.

Array Initialization
~~~~~~~~~~~~~~~~~~~~

Implement ``_initialize_arrays`` to create the component storage arrays:

.. code-block:: python

   def _initialize_arrays(self):
       """Initialize membership and non-membership degree arrays."""
       self.mds = np.zeros(self.shape, dtype=np.float64)
       self.nmds = np.zeros(self.shape, dtype=np.float64)

The method creates NumPy arrays with the backend's shape and appropriate data type. 
Each component gets its own array for optimal memory layout and vectorization.

Element Access Methods
~~~~~~~~~~~~~~~~~~~~~~~

Implement bidirectional conversion between array elements and :class:`~axisfuzzy.core.fuzznums.Fuzznum` objects:

.. code-block:: python

   def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
       """Create a Fuzznum object from array data at the given index."""
       md_value = float(self.mds[index])
       nmd_value = float(self.nmds[index])
       
       return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

   def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
       """Set array data from a Fuzznum object at the given index."""
       if fuzznum.mtype != self.mtype:
           raise ValueError(f"Mtype mismatch: expected {self.mtype}, got {fuzznum.mtype}")
       
       if fuzznum.q != self.q:
           raise ValueError(f"Q parameter mismatch: expected {self.q}, got {fuzznum.q}")
       
       self.mds[index] = fuzznum.md
       self.nmds[index] = fuzznum.nmd

These methods handle the conversion between the high-level :class:`~axisfuzzy.core.fuzznums.Fuzznum` 
interface and the low-level array storage, including proper type validation.

Memory Management
~~~~~~~~~~~~~~~~~

Implement efficient copying and slicing operations:

.. code-block:: python

   def copy(self) -> 'QROFNBackend':
       """Create a deep copy of the backend."""
       new_backend = QROFNBackend(self.shape, self.q, **self.kwargs)
       new_backend.mds = self.mds.copy()
       new_backend.nmds = self.nmds.copy()
       return new_backend

   def slice_view(self, key) -> 'QROFNBackend':
       """Create a view of the backend with the given slice."""
       new_shape = self.mds[key].shape
       new_backend = QROFNBackend(new_shape, self.q, **self.kwargs)
       new_backend.mds = self.mds[key]
       new_backend.nmds = self.nmds[key]
       return new_backend

The ``copy`` method creates independent copies for safe mutation, while ``slice_view`` 
creates memory-efficient views that share data with the original backend.

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Implement formatting methods for efficient display operations:

.. code-block:: python

   def _get_element_formatter(self, format_spec: str) -> Callable:
       """Get element formatting function based on format specification."""
       precision = get_config().DEFAULT_PRECISION
       
       if format_spec in ('p', 'j', 'r'):
           # Use Strategy formatting for special formats
           def strategy_formatter(md: float, nmd: float) -> str:
               fuzznum = Fuzznum(mtype=self.mtype, q=self.q).create(md=md, nmd=nmd)
               return fuzznum.format(format_spec)
           return strategy_formatter
       else:
           # Use default numeric formatting
           return self._create_default_formatter(precision)

   def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
       """Format a single element using the provided formatter."""
       md_val = self.mds[index]
       nmd_val = self.nmds[index]
       return formatter(md_val, nmd_val)

The formatting system delegates to the :class:`~axisfuzzy.core.base.FuzznumStrategy` for 
complex formats while providing efficient numeric formatting for simple cases.

Factory Methods
~~~~~~~~~~~~~~~

Implement class methods for convenient backend creation:

.. code-block:: python

   @classmethod
   def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int, **kwargs) -> 'QROFNBackend':
       """Create backend from existing NumPy arrays."""
       if mds.shape != nmds.shape:
           raise ValueError("Membership and non-membership arrays must have the same shape")
       
       backend = cls(mds.shape, q, **kwargs)
       backend.mds = mds
       backend.nmds = nmds
       return backend

   def fill_from_values(self, md_value: float, nmd_value: float):
       """Fill all elements with the specified values."""
       self.mds.fill(md_value)
       self.nmds.fill(nmd_value)

These factory methods provide convenient ways to create and populate backend instances 
 from various data sources, supporting different initialization patterns.


Registration and Integration
---------------------------------------

The final step in implementing a custom fuzzy number type is registering both the strategy 
and backend with AxisFuzzy's type system. This section demonstrates the registration process 
using the decorator system and validates successful integration.

Strategy Registration
~~~~~~~~~~~~~~~~~~~~~

Use the :func:`~axisfuzzy.core.registry.register_strategy` decorator to register your 
:class:`~axisfuzzy.core.base.FuzznumStrategy` subclass:

.. code-block:: python

   from axisfuzzy.core import FuzznumStrategy, register_strategy

   @register_strategy
   class QROFNStrategy(FuzznumStrategy):
       """Strategy for q-rung orthopair fuzzy numbers."""
       
       mtype = 'qrofn'
       
       def __init__(self, q: Optional[int] = None):
           super().__init__(q)
           # Add attribute validators and change callbacks
           self.add_attribute_validator('md', self._validate_membership)
           self.add_attribute_validator('nmd', self._validate_non_membership)
           self.add_change_callback('md', self._on_membership_change)
           self.add_change_callback('nmd', self._on_membership_change)

The decorator automatically registers the strategy with the global registry when the class 
is defined. The ``mtype`` attribute must match between strategy and backend implementations.

Backend Registration
~~~~~~~~~~~~~~~~~~~~

Similarly, use the :func:`~axisfuzzy.core.registry.register_backend` decorator for your 
:class:`~axisfuzzy.core.backend.FuzzarrayBackend` subclass:

.. code-block:: python

   from axisfuzzy.core import FuzzarrayBackend, register_backend

   @register_backend
   class QROFNBackend(FuzzarrayBackend):
       """SoA backend for q-rung orthopair fuzzy numbers."""
       
       mtype = 'qrofn'
       
       def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
           super().__init__(shape, q, **kwargs)

The registration process validates that the backend class properly implements all required 
abstract methods and has a valid ``mtype`` attribute.

Type Validation
~~~~~~~~~~~~~~~

The registry system performs comprehensive validation during registration:

.. code-block:: python

   # Automatic validation checks performed by decorators:
   # 1. mtype attribute presence and validity
   # 2. Required method implementations
   # 3. Class inheritance from correct base classes
   # 4. No conflicts with existing registrations

   # Manual validation example:
   from axisfuzzy.core.registry import get_registry_fuzztype

   registry = get_registry_fuzztype()
   
   # Check if both strategy and backend are registered
   registered_types = registry.get_registered_mtypes()
   if 'qrofn' in registered_types:
       qrofn_info = registered_types['qrofn']
       print(f"Strategy registered: {qrofn_info['has_strategy']}")
       print(f"Backend registered: {qrofn_info['has_backend']}")
       print(f"Complete type: {qrofn_info['is_complete']}")

The registry ensures type consistency and prevents registration conflicts that could 
compromise system stability.

Registry System
~~~~~~~~~~~~~~~

The :class:`~axisfuzzy.core.registry.FuzznumRegistry` provides a centralized type management system:

.. code-block:: python

   # Access the global registry
   from axisfuzzy.core.registry import get_registry_fuzztype
   
   registry = get_registry_fuzztype()
   
   # Query registered types
   all_types = registry.get_registered_mtypes()
   for mtype, info in all_types.items():
       print(f"{mtype}: Strategy={info['has_strategy']}, Backend={info['has_backend']}")
   
   # Get specific implementations
   qrofn_strategy = registry.get_strategy('qrofn')
   qrofn_backend = registry.get_backend('qrofn')
   
   # Registry statistics
   stats = registry.get_statistics()
   print(f"Total strategies: {stats['total_strategies']}")
   print(f"Total backends: {stats['total_backends']}")
   print(f"Complete types: {stats['complete_types']}")

The registry supports thread-safe operations and provides comprehensive introspection 
capabilities for debugging and system monitoring.

Integration Testing
~~~~~~~~~~~~~~~~~~~

Verify successful registration and functionality with comprehensive tests:

.. code-block:: python

   # Test 1: Verify registration
   from axisfuzzy.core.registry import get_registry_fuzztype
   
   registry = get_registry_fuzztype()
   assert 'qrofn' in registry.get_registered_mtypes()
   
   # Test 2: Create Fuzznum instances
   from axisfuzzy.core import Fuzznum
   
   # Single fuzzy number creation
   qrofn = Fuzznum(mtype='qrofn', q=2).create(md=0.8, nmd=0.3)
   assert qrofn.mtype == 'qrofn'
   assert qrofn.q == 2
   assert qrofn.md == 0.8
   assert qrofn.nmd == 0.3
   
   # Test 3: Create Fuzzarray instances
   from axisfuzzy.core import Fuzzarray
   import numpy as np
   
   # Array creation and manipulation
   arr = Fuzzarray(mtype='qrofn', shape=(2, 3), q=2)
   arr.backend.fill_from_values(0.7, 0.2)
   
   # Verify array properties
   assert arr.mtype == 'qrofn'
   assert arr.shape == (2, 3)
   assert arr.q == 2
   
   # Test element access
   element = arr[0, 0]
   assert isinstance(element, Fuzznum)
   assert element.md == 0.7
   assert element.nmd == 0.2

These tests ensure that your custom fuzzy number type integrates properly with AxisFuzzy's 
core functionality and can be used in all standard operations.

Complete Implementation
~~~~~~~~~~~~~~~~~~~~~~~

A complete fuzzy number type implementation requires both strategy and backend registration:

.. code-block:: python

   # File: my_fuzzy_type.py
   from typing import Optional, Tuple, Any
   import numpy as np
   from axisfuzzy.core import (
       FuzznumStrategy, FuzzarrayBackend, 
       register_strategy, register_backend
   )

   @register_strategy
   class MyFuzzyStrategy(FuzznumStrategy):
       mtype = 'my_fuzzy'
       
       def __init__(self, q: Optional[int] = None):
           super().__init__(q)
           # Implementation details...

   @register_backend  
   class MyFuzzyBackend(FuzzarrayBackend):
       mtype = 'my_fuzzy'
       
       def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
           super().__init__(shape, q, **kwargs)
           # Implementation details...

   # Automatic registration occurs when the module is imported
   # Your fuzzy type is now available throughout AxisFuzzy

Once both components are registered, your custom fuzzy number type becomes available 
throughout the AxisFuzzy ecosystem for creation, manipulation, and computation.

Conclusion
----------

This development guide provides a comprehensive framework for extending AxisFuzzy 
with custom fuzzy number types. The systematic approach outlined here ensures both 
mathematical correctness and seamless integration with the existing ecosystem.

**Key Implementation Steps:**

1. **Strategy Development**: Implement ``FuzznumStrategy`` with proper attribute 
   management, validation, and mathematical operations specific to your fuzzy type.

2. **Backend Architecture**: Create ``FuzzarrayBackend`` following the SoA pattern 
   for efficient array operations and memory management.

3. **Registration Integration**: Use ``@register_strategy`` and ``@register_backend`` 
   decorators to make your implementation discoverable throughout AxisFuzzy.

4. **Comprehensive Testing**: Establish robust test suites covering unit tests, 
   constraint validation, integration testing, performance benchmarks, and error handling.

**Best Practices:**

- Maintain mathematical rigor in constraint validation and boundary condition handling
- Follow the established patterns for attribute management and data access
- Ensure performance characteristics meet production requirements
- Implement comprehensive error handling for robust operation

By following this guide, developers can confidently extend AxisFuzzy's capabilities 
while maintaining the library's standards for correctness, performance, and usability. 
The modular architecture ensures that custom implementations integrate seamlessly 
with existing fuzzy logic operations and computational workflows.

**Next Steps**: After implementation, consider contributing your fuzzy type back to 
the AxisFuzzy community through the established contribution guidelines, enabling 
broader adoption and collaborative improvement of your mathematical model.