======================================
Fuzzy Operations Development Guide
======================================

This guide provides comprehensive instructions for developing custom fuzzy operations in AxisFuzzy. 
Using Q-Rung Orthopair Fuzzy Numbers (QROFNs) as a practical example, you will learn how to implement 
the four core operation types, integrate high-performance backends, and ensure proper registration 
within the AxisFuzzy framework.

Understanding the Operation Framework
--------------------------------------

The AxisFuzzy operation framework is built around the ``OperationMixin`` abstract base class, which 
provides a standardized interface for implementing fuzzy number operations. This framework supports 
four distinct operation categories, each with specific implementation requirements and use cases.

Core Architecture
~~~~~~~~~~~~~~~~~

The operation framework follows a strategy pattern where each operation type inherits from 
``OperationMixin`` and implements specific abstract methods. The framework handles preprocessing, 
validation, performance monitoring, and result formatting automatically.

.. code-block:: python

    from axisfuzzy.core import OperationMixin, register_operation
    
    @register_operation
    class CustomOperation(OperationMixin):
        def get_operation_name(self) -> str:
            return 'custom_op'
        
        def get_supported_mtypes(self) -> List[str]:
            return ['qrofn']

Operation Types and Implementation Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework defines four core operation interfaces that correspond to different mathematical 
operation patterns:

**Binary Operations** (``_execute_binary_op_impl``)
    Handle operations between two fuzzy numbers, such as addition, multiplication, or intersection.
    These operations take two strategy instances and return a dictionary containing the result 
    components.

.. code-block:: python

    def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm):
        # Example: QROFN Addition
        md_result = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        nmd_result = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)
        return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

**Unary Operations with Operand** (``_execute_unary_op_operand_impl``)
    Process operations involving a fuzzy number and a scalar value, such as power or scalar 
    multiplication. The operand parameter provides the scalar value.

.. code-block:: python

    def _execute_unary_op_operand_impl(self, strategy, operand, tnorm):
        # Example: QROFN Power
        md_result = strategy.md ** operand
        nmd_result = strategy.nmd ** operand
        return {'md': md_result, 'nmd': nmd_result, 'q': strategy.q}

**Pure Unary Operations** (``_execute_unary_op_pure_impl``)
    Implement operations that transform a single fuzzy number without additional parameters, 
    such as complement or negation operations.

.. code-block:: python

    def _execute_unary_op_pure_impl(self, strategy, tnorm):
        # Example: QROFN Complement
        return {'md': strategy.nmd, 'nmd': strategy.md, 'q': strategy.q}

**Comparison Operations** (``_execute_comparison_op_impl``)
    Handle comparison operations between fuzzy numbers, returning boolean results for 
    relationships like greater than, less than, or equality.

.. code-block:: python

    def _execute_comparison_op_impl(self, strategy_1, strategy_2, tnorm):
        # Example: QROFN Greater Than
        score_1 = strategy_1.md ** strategy_1.q - strategy_1.nmd ** strategy_1.q
        score_2 = strategy_2.md ** strategy_2.q - strategy_2.nmd ** strategy_2.q
        return {'value': score_1 > score_2}

Operation Registration Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@register_operation`` decorator provides automatic registration of operation classes with 
the global operation scheduler. This decorator supports both eager and lazy registration patterns.

.. code-block:: python

    # Immediate registration (default)
    @register_operation
    class QROFNAddition(OperationMixin):
        pass
    
    # Lazy registration
    @register_operation(eager=False)
    class QROFNMultiplication(OperationMixin):
        pass

The registration process validates that classes properly inherit from ``OperationMixin`` and 
instantiates operation objects for immediate availability in the operation scheduler.

T-norm and T-conorm Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operations receive a ``tnorm`` parameter that provides access to triangular norm and conorm 
functions. These mathematical operators are essential for fuzzy set operations and ensure 
consistent behavior across different operation implementations.

.. code-block:: python

    # T-norm application (intersection-like behavior)
    result = tnorm.t_norm(value1, value2)
    
    # T-conorm application (union-like behavior)  
    result = tnorm.t_conorm(value1, value2)

The framework supports multiple T-norm families including algebraic, Einstein, Hamacher, and 
Frank norms, allowing operations to adapt their behavior based on the configured norm type.

QROFN Framework Example
~~~~~~~~~~~~~~~~~~~~~~~

Q-Rung Orthopair Fuzzy Numbers demonstrate the framework's capabilities through their dual-component 
structure (membership and non-membership degrees) and q-rung parameter. Each QROFN operation must 
preserve the q-rung constraint while applying appropriate T-norm/T-conorm combinations.

.. code-block:: python

    @register_operation
    class QROFNAddition(OperationMixin):
        def get_operation_name(self) -> str:
            return 'add'
        
        def get_supported_mtypes(self) -> List[str]:
            return ['qrofn']
        
        def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm):
            # Addition: md = S(md1, md2), nmd = T(nmd1, nmd2)
            md_result = tnorm.t_conorm(strategy_1.md, strategy_2.md)
            nmd_result = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)
            return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

This framework design ensures type safety, performance optimization, and consistent behavior 
across all fuzzy number types while providing flexibility for custom mathematical formulations.

Complete QROFN Operations Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AxisFuzzy framework provides a comprehensive suite of operations for Q-Rung Orthopair Fuzzy Numbers. 
The following table catalogs all available operations, their types, and implementation characteristics:

+------------------------+------------------+--------------------------------------------------+
| Operation Name         | Operation Type   | Notes                                            |
+------------------------+------------------+--------------------------------------------------+
| **Arithmetic Operations**                                                                    |
+------------------------+------------------+--------------------------------------------------+
| ``add``                | Binary           | Addition using t-conorm for MD, t-norm for NMD   |
+------------------------+------------------+--------------------------------------------------+
| ``sub``                | Binary           | Subtraction with constraint validation           |
+------------------------+------------------+--------------------------------------------------+
| ``mul``                | Binary           | Multiplication using t-norm for MD, t-conorm     |
|                        |                  | for NMD                                          |
+------------------------+------------------+--------------------------------------------------+
| ``truediv``            | Binary           | Division with zero-division protection           |
+------------------------+------------------+--------------------------------------------------+
| ``pow``                | Unary+Operand    | Power operation using t-norm generator functions |
+------------------------+------------------+--------------------------------------------------+
| ``times``              | Unary+Operand    | Scalar multiplication with t-norm generators     |
+------------------------+------------------+--------------------------------------------------+
| ``exp``                | Unary+Operand    | Exponential operation (experimental)             |
+------------------------+------------------+--------------------------------------------------+
| ``log``                | Unary+Operand    | Logarithmic operation (experimental)             |
+------------------------+------------------+--------------------------------------------------+
| **Comparison Operations**                                                                    |
+------------------------+------------------+--------------------------------------------------+
| ``gt``                 | Binary           | Greater than using score function (md - nmd)     |
+------------------------+------------------+--------------------------------------------------+
| ``lt``                 | Binary           | Less than using score function                   |
+------------------------+------------------+--------------------------------------------------+
| ``eq``                 | Binary           | Equality with epsilon tolerance                  |
+------------------------+------------------+--------------------------------------------------+
| ``ge``                 | Binary           | Greater than or equal with score function        |
+------------------------+------------------+--------------------------------------------------+
| ``le``                 | Binary           | Less than or equal with score function           |
+------------------------+------------------+--------------------------------------------------+
| ``ne``                 | Binary           | Not equal with epsilon tolerance                 |
+------------------------+------------------+--------------------------------------------------+
| **Set-Theoretic Operations**                                                                 |
+------------------------+------------------+--------------------------------------------------+
| ``intersection``       | Binary           | Fuzzy intersection using t-norm                  |
+------------------------+------------------+--------------------------------------------------+
| ``union``              | Binary           | Fuzzy union using t-conorm                       |
+------------------------+------------------+--------------------------------------------------+
| ``complement``         | Unary            | Fuzzy complement (swap MD and NMD)               |
+------------------------+------------------+--------------------------------------------------+
| ``difference``         | Binary           | Set difference A ∩ ¬B                            |
+------------------------+------------------+--------------------------------------------------+
| ``symmetric_diff``     | Binary           | Symmetric difference (A ∪ B) ∩ ¬(A ∩ B)          |
+------------------------+------------------+--------------------------------------------------+
| **Logical Operations**                                                                       |
+------------------------+------------------+--------------------------------------------------+
| ``implication``        | Binary           | Fuzzy implication ¬A ∪ B                         |
+------------------------+------------------+--------------------------------------------------+
| ``equivalence``        | Binary           | Fuzzy equivalence (A → B) ∩ (B → A)              |
+------------------------+------------------+--------------------------------------------------+
| **Matrix Operations**                                                                        |
+------------------------+------------------+--------------------------------------------------+
| ``matmul``             | Binary           | Matrix multiplication for Fuzzarray objects      |
+------------------------+------------------+--------------------------------------------------+

**Operation Type Categories:**

- **Binary**: Operations between two fuzzy numbers or arrays
- **Unary**: Operations on a single fuzzy number or array  
- **Unary+Operand**: Operations on a fuzzy number/array with a scalar operand

**Method Categories:**

- **Arithmetic**: Mathematical operations following fuzzy arithmetic principles
- **Comparison**: Ordering operations using score functions and epsilon tolerance
- **Set Theory**: Classical fuzzy set operations (intersection, union, complement)
- **Logic**: Fuzzy logical operations (implication, equivalence)
- **Linear Algebra**: Matrix and tensor operations for high-dimensional fuzzy data

**Implementation Notes:**

- All operations support both ``Fuzznum`` (scalar) and ``Fuzzarray`` (vectorized) execution
- Experimental operations (``exp``, ``log``) may have limited stability guarantees
- Comparison operations use the score function :math:`S(A) = \mu_A - \nu_A` for ordering
- Set-theoretic operations leverage configurable t-norm and t-conorm families
- Matrix operations preserve fuzzy constraints while enabling linear algebraic computations


Implementing Binary Operations
-----------------------------------------

Binary operations form the foundation of fuzzy arithmetic and set operations. This section demonstrates how to implement high-performance binary operations using the QROFN framework as a comprehensive example.

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

QROFN binary operations are based on T-norm and T-conorm pairs that preserve the orthopair constraint :math:`md^q + nmd^q \leq 1`. The fundamental patterns are:

**Addition Pattern:**

.. math::
   
   A \oplus B = (S(md_A, md_B), T(nmd_A, nmd_B))

**Multiplication Pattern:**

.. math::
   
   A \otimes B = (T(md_A, md_B), S(nmd_A, nmd_B))

Where :math:`T` is a T-norm, :math:`S` is a T-conorm, and the choice of T-norm/T-conorm pair determines the specific algebraic properties.

Implementation Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary operations require implementing two core methods:

.. code-block:: python

   class QROFNAddition(OperationMixin):
       def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm):
           """Single Fuzznum operation"""
           pass
           
       def _execute_fuzzarray_op_impl(self, fuzzarray_1, other, tnorm):
           """Vectorized Fuzzarray operation"""
           pass

The dual implementation ensures both scalar and vectorized operations maintain consistent semantics while optimizing for their respective use cases.

QROFN Addition Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Addition demonstrates the S-T pattern where membership degrees use T-conorm (disjunctive) and non-membership degrees use T-norm (conjunctive):

.. code-block:: python

   @register_operation
   class QROFNAddition(OperationMixin):
       def get_operation_name(self) -> str:
           return 'add'
           
       def get_supported_mtypes(self) -> List[str]:
           return ['qrofn']
           
       def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm):
           # Addition: md = S(md1, md2), nmd = T(nmd1, nmd2)
           md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
           nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)
           return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

The T-conorm increases membership (optimistic combination) while T-norm decreases non-membership (conservative combination), reflecting additive semantics.

QROFN Multiplication Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiplication uses the T-S pattern, inverting the T-norm/T-conorm roles:

.. code-block:: python

   def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm):
       # Multiplication: md = T(md1, md2), nmd = S(nmd1, nmd2)
       md = tnorm.t_norm(strategy_1.md, strategy_2.md)
       nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)
       return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

This pattern reflects multiplicative semantics where both operands must contribute to membership (conjunctive) while non-membership accumulates (disjunctive).

High-Performance Fuzzarray Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vectorized operations leverage NumPy broadcasting and the ``_prepare_operands`` helper for optimal performance:

.. code-block:: python
   :emphasize-lines: 10,11

   def _execute_fuzzarray_op_impl(self, fuzzarray_1, other, tnorm):
       # Extract and broadcast component arrays
       mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
       
       # Vectorized T-norm/T-conorm operations
       md_res = tnorm.t_conorm(mds1, mds2)  # Addition pattern
       nmd_res = tnorm.t_norm(nmds1, nmds2)
       
       # Construct result backend
       backend_cls = get_registry_fuzztype().get_backend('qrofn')
       new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
       return Fuzzarray(backend=new_backend)

The ``_prepare_operands`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This critical helper handles operand preprocessing, type validation, and broadcasting:

.. code-block:: python

   def _prepare_operands(fuzzarray_1, other):
       mds1, nmds1 = fuzzarray_1.backend.get_component_arrays()
       
       if isinstance(other, Fuzzarray):
           # Validate compatibility
           if other.mtype != fuzzarray_1.mtype:
               raise ValueError(f"Mtype mismatch: {fuzzarray_1.mtype} vs {other.mtype}")
           if other.q != fuzzarray_1.q:
               raise ValueError(f"Q-rung mismatch: {fuzzarray_1.q} vs {other.q}")
               
           mds2, nmds2 = other.backend.get_component_arrays()
           return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)
           
       elif isinstance(other, Fuzznum):
           # Handle scalar broadcasting
           mds2 = np.full((1,), other.md, dtype=mds1.dtype)
           nmds2 = np.full((1,), other.nmd, dtype=nmds1.dtype)
           return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)

Error Handling and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust operations require comprehensive validation:

**Type Compatibility:** Ensure operands share the same mtype and q-rung parameter.

**Shape Broadcasting:** Leverage NumPy's broadcasting rules with clear error messages for incompatible shapes.

**Numerical Stability:** T-norm/T-conorm operations maintain the orthopair constraint automatically.

**Performance Considerations:** Use ``np.broadcast_arrays`` for efficient memory layout and vectorization.

Advanced Binary Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond arithmetic, QROFN supports set-theoretic operations:

**Intersection (Minimum):**

.. math::
   
   A \cap B = (T(md_A, md_B), S(nmd_A, nmd_B))

**Union (Maximum):**

.. math::
   
   A \cup B = (S(md_A, md_B), T(nmd_A, nmd_B))

These operations use the same implementation pattern but with different T-norm/T-conorm semantics, demonstrating the framework's flexibility.

Integration with Operation Scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@register_operation`` decorator automatically integrates operations with the global scheduler:

.. code-block:: python

   from axisfuzzy.core import fuzzynum
   
   # Automatic registration enables operator overloading
   a = fuzzynum(md=0.8, nmd=0.3, mtype='qrofn', q=3)
   b = fuzzynum(md=0.6, nmd=0.4, mtype='qrofn', q=3)
   result = a + b  # Dispatches to QROFNAddition

This seamless integration allows natural mathematical syntax while maintaining the high-performance vectorized backend.



Unary and Comparison Operations
------------------------------------------

Unary and comparison operations form the foundation of advanced fuzzy logic computations, 
enabling scalar transformations and ordering relationships between fuzzy numbers. This section 
explores the implementation patterns for power operations, scalar multiplication, complement 
operations, and score-based comparison strategies.

Mathematical Foundations for Unary Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unary operations in AxisFuzzy fall into two distinct categories: **operand-based** operations 
that require an additional scalar parameter, and **pure** unary operations that transform 
the fuzzy number independently.

**Power Operations** implement scalar exponentiation using T-norm generator and pseudo-inverse functions:

.. math::
   
   A^n = (g^{-1}(n \cdot g(md_A)), f^{-1}(n \cdot f(nmd_A)))

where :math:`g` and :math:`f` are the dual generator and generator functions of the 
T-norm respectively, and :math:`g^{-1}` and :math:`f^{-1}` are the corresponding 
pseudo-inverse functions.

**Times Operations** provide scalar multiplication using the T-norm generator framework:

.. math::
   
   n \cdot A = (f^{-1}(n \cdot f(md_A)), g^{-1}(n \cdot g(nmd_A)))

Note that the times operation uses the generator functions in reverse order compared to 
the power operation, ensuring correct implementation of different operation semantics.

**Complement Operations** implement fuzzy negation by swapping membership degrees:

.. math::
   
   \neg A = (nmd_A, md_A)

Implementation Architecture for Unary Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unary operations utilize two specialized execution methods depending on their mathematical nature:

**Operand-Based Unary Operations** (``_execute_unary_op_operand_impl``)
    Handle operations requiring a scalar operand, such as power and times operations.

.. code-block:: python

    def _execute_unary_op_operand_impl(self, 
                                       strategy: Any, 
                                       operand: Union[int, float], 
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        # Example: QROFN Power Operation using T-norm generators
        md = tnorm.g_inv_func(operand * tnorm.g_func(strategy.md))
        nmd = tnorm.f_inv_func(operand * tnorm.f_func(strategy.nmd))
        return {'md': md, 'nmd': nmd, 'q': strategy.q}

**Pure Unary Operations** (``_execute_unary_op_pure_impl``)
    Handle operations that transform the fuzzy number without additional parameters.

.. code-block:: python

    def _execute_unary_op_pure_impl(self, 
                                    strategy: Any, 
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        # Example: QROFN Complement
        return {'md': strategy.nmd, 'nmd': strategy.md, 'q': strategy.q}

QROFN Power and Times Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``QROFNPower`` and ``QROFNTimes`` classes demonstrate operand-based unary operations:

.. code-block:: python

    @register_operation
    class QROFNPower(OperationMixin):
        """
        Implements the power operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).
        
        This operation calculates A^operand using T-norm generator functions.
        """
        
        def get_operation_name(self) -> str:
            return 'pow'
        
        def get_supported_mtypes(self) -> List[str]:
            return ['qrofn']
        
        def _execute_unary_op_operand_impl(self,
                                           strategy: Any,
                                           operand: Union[int, float],
                                           tnorm: OperationTNorm) -> Dict[str, Any]:
            """
            Executes the unary power operation using T-norm generator functions.
            
            The power operation uses the mathematical formula:
            A^n = (g^{-1}(n·g(md_A)), f^{-1}(n·f(nmd_A)))
            """
            # Use T-norm generator functions for mathematically consistent power operation
            md = tnorm.g_inv_func(operand * tnorm.g_func(strategy.md))
            nmd = tnorm.f_inv_func(operand * tnorm.f_func(strategy.nmd))
            
            return {'md': md, 'nmd': nmd, 'q': strategy.q}

    @register_operation
    class QROFNTimes(OperationMixin):
        """
        Implements the times (scalar multiplication) operation for QROFNs.
        
        This operation calculates n·A using T-norm generator functions.
        Note: Times operation swaps the generator functions compared to power.
        """
        
        def get_operation_name(self) -> str:
            return 'times'
        
        def get_supported_mtypes(self) -> List[str]:
            return ['qrofn']
        
        def _execute_unary_op_operand_impl(self,
                                           strategy: Any,
                                           operand: Union[int, float],
                                           tnorm: OperationTNorm) -> Dict[str, Any]:
            """
            Executes the unary times operation using T-norm generator functions.
            
            The times operation uses the mathematical formula:
            n·A = (f^{-1}(n·f(md_A)), g^{-1}(n·g(nmd_A)))
            """
            # Note: Times operation swaps f and g functions compared to power
            md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
            nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))
            
            return {'md': md, 'nmd': nmd, 'q': strategy.q}

**Key Implementation Features:**

- **Type Validation**: Ensures operands are valid numeric types
- **Constraint Preservation**: Maintains q-rung orthopair properties
- **Vectorization Support**: Compatible with NumPy broadcasting for arrays

High-Performance Fuzzarray Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unary operations on ``Fuzzarray`` objects leverage vectorized operations with T-norm generators for optimal performance:

.. code-block:: python

    @register_operation
    class QROFNPower(OperationMixin):
        # get_operation_name():...
        # get_supported_mtypes():...
        # _execute_unary_op_operand_impl():...

        def _execute_fuzzarray_op_impl(self,
                                    fuzzarray: Fuzzarray,
                                    operand: Union[int, float],
                                    tnorm: OperationTNorm) -> Fuzzarray:
            """
            Executes vectorized power operation on Fuzzarray.
            """
            mds, nmds = fuzzarray.backend.get_component_arrays()
            
            md_res = tnorm.g_inv_func(operand * tnorm.g_func(mds))
            nmd_res = tnorm.f_inv_func(operand * tnorm.f_func(nmds))
            
            backend_cls = get_registry_fuzztype().get_backend('qrofn')
            new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
            return Fuzzarray(backend=new_backend)

    @register_operation
    class QROFNTimes(OperationMixin):
        # get_operation_name():...
        # get_supported_mtypes():...
        # _execute_unary_op_operand_impl():...

        def _execute_fuzzarray_op_impl(self,
                                fuzzarray: Fuzzarray,
                                operand: Union[int, float],
                                tnorm: OperationTNorm) -> Fuzzarray:
            """
            Executes vectorized times operation on Fuzzarray.
            """
            mds, nmds = fuzzarray.backend.get_component_arrays()
            
            md_res = tnorm.f_inv_func(operand * tnorm.f_func(mds))
            nmd_res = tnorm.g_inv_func(operand * tnorm.g_func(nmds))
            
            backend_cls = get_registry_fuzztype().get_backend('qrofn')
            new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
            return Fuzzarray(backend=new_backend)

**Performance Optimizations:**

- **Direct Array Access**: Bypasses object overhead for component arrays
- **In-Place Operations**: Minimizes memory allocation where possible
- **Broadcasting Support**: Handles scalar-array operations efficiently

Comparison Operations and Score Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparison operations implement ordering relationships using mathematical score functions 
that map fuzzy numbers to comparable scalar values.

**Score Function Definition** for QROFNs:

.. math::
   
   Score(A) = md_A^q - nmd_A^q

**Accuracy Function** for tie-breaking:

.. math::
   
   Accuracy(A) = md_A^q + nmd_A^q

Implementation of QROFN Comparison Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``QROFNGreaterThan`` class demonstrates the comparison operation pattern:

.. code-block:: python

    @register_operation
    class QROFNGreaterThan(OperationMixin):
        def get_operation_name(self) -> str:
            return 'gt'
        
        def get_supported_mtypes(self) -> List[str]:
            return ['qrofn']
        
        def _execute_comparison_op_impl(self, strategy_1, strategy_2, tnorm):
            # Simple comparison using membership degree difference
            # This provides a direct score-based comparison for QROFN values
            return {'value': strategy_1.md - strategy_1.nmd > strategy_2.md - strategy_2.nmd}

**Comparison Features:**

- **Epsilon Tolerance**: Handles floating-point precision issues
- **Hierarchical Comparison**: Uses accuracy function for tie-breaking
- **Boolean Return Format**: Standardized dictionary format for results

Vectorized Array Comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Array-level comparisons return boolean arrays for element-wise analysis:

.. code-block:: python

    def _execute_fuzzarray_op_impl(self, fuzzarray_1, fuzzarray_2, tnorm):
        # Extract component arrays using helper function
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)
        
        # Vectorized comparison using membership degree differences
        return np.where(mds1 - nmds1 > mds2 - nmds2, True, False)

**Vectorization Benefits:**

- **Batch Processing**: Handles large arrays efficiently
- **Memory Efficiency**: Minimizes temporary array creation
- **Broadcasting Support**: Automatic shape compatibility handling

Error Handling and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust error handling ensures mathematical consistency and user-friendly diagnostics:

.. code-block:: python

    def _execute_unary_op_operand_impl(self, strategy, operand, tnorm):
        """Execute unary operation with operand validation"""
        # Type validation for operand
        if not isinstance(operand, (int, float)):
            raise TypeError("Operand must be numeric (int or float)")
        
        # Value validation for specific operations
        if operand < 0:
            raise ValueError("Operand must be non-negative for power operations")
        
        # Execute the actual operation logic
        # Implementation depends on specific operation type

**Validation Features:**

- **Type Safety**: Ensures operands match expected types
- **Mathematical Constraints**: Validates domain restrictions
- **Descriptive Errors**: Provides clear diagnostic messages

This comprehensive approach to unary and comparison operations establishes a robust foundation 
for advanced fuzzy computations while maintaining the performance characteristics essential 
for scientific computing applications.



High-Performance Backend Development
-----------------------------------------------

The FuzzarrayBackend architecture provides the computational foundation for AxisFuzzy's 
high-performance fuzzy array operations. This section covers essential backend 
implementation patterns and practical application examples for developing custom 
fuzzy number backends.

Backend Architecture Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AxisFuzzy employs a Structure-of-Arrays (SoA) design where each fuzzy number component 
is stored in separate NumPy arrays, enabling efficient vectorized operations:

.. code-block:: python

    # Memory layout comparison
    # AoS: [md₁,nmd₁] [md₂,nmd₂] [md₃,nmd₃] ...
    # SoA: [md₁,md₂,md₃,...] [nmd₁,nmd₂,nmd₃,...]

Essential Backend Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A minimal backend implementation requires these core components:

.. code-block:: python

    from axisfuzzy.core import FuzzarrayBackend, register_backend
    
    @register_backend
    class CustomBackend(FuzzarrayBackend):
        mtype = 'custom_type'
        
        @property
        def cmpnum(self) -> int:
            return 2  # Number of component arrays
            
        @property
        def cmpnames(self) -> Tuple[str, ...]:
            return ('md', 'nmd')  # Component names
            
        def _initialize_arrays(self):
            self.mds = np.zeros(self.shape, dtype=self.dtype)
            self.nmds = np.zeros(self.shape, dtype=self.dtype)
        
        def get_component_arrays(self) -> Tuple[np.ndarray, ...]:
            return self.mds, self.nmds

Vectorized Constraint Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement efficient constraint checking for mathematical validity:

.. code-block:: python

    @staticmethod
    def _validate_fuzzy_constraints_static(mds: np.ndarray, nmds: np.ndarray, 
                                         q: int) -> None:
        """Vectorized QROFN constraint: md^q + nmd^q ≤ 1"""
        epsilon = get_config().DEFAULT_EPSILON
        sum_of_powers = np.power(mds, q) + np.power(nmds, q)
        violations = sum_of_powers > (1.0 + epsilon)
        
        if np.any(violations):
            violation_indices = np.where(violations)
            first_idx = tuple(idx[0] for idx in violation_indices)
            raise ValueError(f"Constraint violation at {first_idx}")

Practical Application Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Backend Registration and Usage**:

.. code-block:: python

    # Automatic registration via decorator
    @register_backend
    class QROFNBackend(FuzzarrayBackend):
        mtype = 'qrofn'
    
    # Factory function automatically selects backend
    from axisfuzzy.core.fuzzarray import fuzzarray
    arr = fuzzarray(data, mtype='qrofn', q=2)

**High-Performance Array Operations**:

.. code-block:: python

    # Efficient element access and modification
    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        md_val = float(self.mds[index])
        nmd_val = float(self.nmds[index])
        return Fuzznum(mtype=self.mtype, q=self.q).create(
            md=md_val, nmd=nmd_val)
    
    # Memory-efficient operations
    def copy(self) -> 'QROFNBackend':
        new_backend = QROFNBackend(self.shape, self.q, **self.kwargs)
        new_backend.mds = self.mds.copy()
        new_backend.nmds = self.nmds.copy()
        return new_backend

**Integration with Operations**:

.. code-block:: python

    # Backend provides arrays for vectorized operations
    def _execute_fuzzarray_op_impl(self, fuzzarray_1, other, tnorm):
        mds1, nmds1 = fuzzarray_1.backend.get_component_arrays()
        mds2, nmds2 = other.backend.get_component_arrays()
        
        # Vectorized computation using t-norms
        result_mds = tnorm.t_conorm(mds1, mds2)
        result_nmds = tnorm.t_norm(nmds1, nmds2)
        
        return fuzzarray_1.backend.from_arrays(
            result_mds, result_nmds, q=fuzzarray_1.q)

The SoA architecture enables AxisFuzzy to achieve optimal performance for large-scale 
fuzzy computations while maintaining mathematical correctness through vectorized 
constraint validation and efficient NumPy integration.

Operation Development Guide
---------------------------

This section demonstrates the complete development workflow for implementing QROFN operations in AxisFuzzy, using actual code examples from the library to illustrate best practices for operation registration, backend integration, and testing patterns.

Operation Implementation Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QROFN operations follow a standardized implementation pattern. Here's the complete implementation of QROFN addition:

.. code-block:: python

    @register_operation
    class QROFNAddition(OperationMixin):
        """
        Implements the addition operation for Q-Rung Orthopair Fuzzy Numbers.
        
        The addition formula: md = S(md_A, md_B), nmd = T(nmd_A, nmd_B)
        where S is a t-conorm and T is a t-norm.
        """
        
        def get_operation_name(self) -> str:
            return 'add'
        
        def get_supported_mtypes(self) -> List[str]:
            return ['qrofn']
        
        def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm):
            # Core mathematical operation
            md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
            nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)
            return {'md': md, 'nmd': nmd, 'q': strategy_1.q}
        
        def _execute_fuzzarray_op_impl(self, fuzzarray_1, other, tnorm):
            # High-performance vectorized implementation
            mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
            
            md_res = tnorm.t_conorm(mds1, mds2)
            nmd_res = tnorm.t_norm(nmds1, nmds2)
            
            backend_cls = get_registry_fuzztype().get_backend('qrofn')
            new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
            return Fuzzarray(backend=new_backend)

Key implementation principles:

- **Decorator Registration**: ``@register_operation`` enables automatic discovery
- **Type Safety**: ``get_supported_mtypes()`` ensures operation compatibility
- **Dual Implementation**: Both single fuzzy number and vectorized array operations
- **Mathematical Correctness**: Operations follow established fuzzy logic formulas

High-performance computation based on the backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``_execute_fuzzarray_op_impl`` method provides vectorized operations for high-performance computation on fuzzy arrays. This method leverages NumPy's broadcasting and vectorization capabilities to process entire arrays efficiently.

**Core Implementation Pattern**

.. code-block:: python

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized operation for QROFN fuzzy arrays.
        
        Args:
            fuzzarray_1: Primary fuzzy array operand
            other: Secondary operand (Fuzzarray, Fuzznum, or scalar)
            tnorm: T-norm/T-conorm operations handler
            
        Returns:
            Fuzzarray: Result of vectorized operation
        """
        # Step 1: Prepare operands with broadcasting
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        
        # Step 2: Apply vectorized fuzzy operations
        # For addition: md = S(md1, md2), nmd = T(nmd1, nmd2)
        md_res = tnorm.t_conorm(mds1, mds2)  # T-conorm for membership
        nmd_res = tnorm.t_norm(nmds1, nmds2)  # T-norm for non-membership
        
        # Step 3: Create result backend and return Fuzzarray
        backend_cls = get_registry_fuzztype().get_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)

**Advanced Vectorized Operations with Conditions**

For complex operations like subtraction and division, conditional logic is vectorized using NumPy masks:

.. code-block:: python

    def _execute_fuzzarray_op_impl(self, fuzzarray_1, other, tnorm):
        """Vectorized subtraction with conditional validation."""
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        q = fuzzarray_1.q
        epsilon = get_config().DEFAULT_EPSILON
        
        # Vectorized condition checking
        with np.errstate(divide='ignore', invalid='ignore'):
            condition_1 = np.divide(nmds1, nmds2)
            condition_2 = ((1 - mds1**q) / (1 - mds2**q))**(1/q)
        
        # Boolean mask for valid operations
        valid_mask = (
            (condition_1 >= epsilon) & (condition_1 <= 1 - epsilon) &
            (condition_2 >= epsilon) & (condition_2 <= 1 - epsilon) &
            (condition_1 <= condition_2)
        )
        
        # Vectorized computation with conditional results
        md_res_valid = ((mds1**q - mds2**q) / (1 - mds2**q))**(1/q)
        nmd_res_valid = np.divide(nmds1, nmds2)
        
        # Apply results only where conditions are met
        md_res = np.where(valid_mask, md_res_valid, 0.0)  # Default: (0, 1)
        nmd_res = np.where(valid_mask, nmd_res_valid, 1.0)
        
        # Handle numerical errors
        np.nan_to_num(md_res, copy=False, nan=0.0)
        np.nan_to_num(nmd_res, copy=False, nan=1.0)
        
        backend_cls = get_registry_fuzztype().get_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=q)
        return Fuzzarray(backend=new_backend)

**Performance Optimization Techniques**

1. **Broadcasting Strategy**: Uses ``_prepare_operands`` for automatic shape compatibility
2. **Error State Management**: ``np.errstate`` handles division by zero gracefully  
3. **Conditional Vectorization**: ``np.where`` and boolean masks replace loops
4. **Memory Efficiency**: In-place operations with ``copy=False`` parameters
5. **Numerical Stability**: ``np.nan_to_num`` ensures robust floating-point handling

Operand Preparation Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``_prepare_operands`` function handles type checking and broadcasting for vectorized operations:

.. code-block:: python

    def _prepare_operands(fuzzarray_1, other):
        """Helper to get component arrays from operands with broadcasting."""
        mds1, nmds1 = fuzzarray_1.backend.get_component_arrays()
        
        if isinstance(other, Fuzzarray):
            # Validate compatibility
            if other.mtype != fuzzarray_1.mtype:
                raise ValueError(f"Cannot operate on different mtypes: "
                               f"{fuzzarray_1.mtype} and {other.mtype}")
            if other.q != fuzzarray_1.q:
                raise ValueError(f"Cannot operate on different q values: "
                               f"{fuzzarray_1.q} and {other.q}")
            
            mds2, nmds2 = other.backend.get_component_arrays()
            return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)
        
        elif isinstance(other, Fuzznum):
            # Handle Fuzznum broadcasting
            mds2 = np.full((1,), other.md, dtype=mds1.dtype)
            nmds2 = np.full((1,), other.nmd, dtype=nmds1.dtype)
            return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)


Conclusion
----------

This development guide provides a comprehensive framework for implementing custom fuzzy 
operations in AxisFuzzy. The systematic approach demonstrated through QROFN examples 
ensures both mathematical correctness and high-performance execution across scalar and 
vectorized computations.

**Key Implementation Principles:**

1. **Operation Framework Mastery**: Understand the four operation types (binary, unary with 
   operand, pure unary, comparison) and implement appropriate ``_execute_*_impl`` methods 
   for your mathematical requirements.

2. **Dual Implementation Strategy**: Provide both ``_execute_binary_op_impl`` for scalar 
   operations and ``_execute_fuzzarray_op_impl`` for vectorized computations, ensuring 
   semantic consistency while optimizing for performance.

3. **Registration Integration**: Use ``@register_operation`` decorators to seamlessly 
   integrate operations with AxisFuzzy's dispatch system and enable natural operator 
   overloading syntax.

4. **Performance Optimization**: Leverage ``_prepare_operands`` utilities, NumPy broadcasting, 
   and SoA backend architecture for efficient memory usage and vectorized execution.

**Best Practices:**

- Maintain mathematical rigor in T-norm/T-conorm applications and constraint preservation
- Implement comprehensive error handling for type compatibility and numerical stability  
- Follow established patterns for operand preparation and result construction
- Ensure consistent behavior between scalar and array operation implementations

By following this guide, developers can confidently extend AxisFuzzy's operation capabilities 
while maintaining the library's standards for correctness, performance, and mathematical 
precision. The modular architecture ensures that custom operations integrate seamlessly 
with existing fuzzy number types and computational workflows.

**Next Steps**: After implementation, consider contributing your operations back to the 
AxisFuzzy community through established contribution guidelines, enabling broader adoption 
and collaborative improvement of your mathematical models.




