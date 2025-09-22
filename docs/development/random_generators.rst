=============================
Random Generators Development
=============================

This development guide provides comprehensive instructions for implementing
custom random generators for fuzzy number types in AxisFuzzy. The guide
demonstrates the
complete development process using Q-Rung Orthopair Fuzzy Numbers (QROFN) as a
reference example, covering architecture design, implementation patterns, and
integration workflows.

Understanding the Random Generator
----------------------------------

AxisFuzzy's random generation system follows a plugin-based architecture that
enables developers to create specialized random generators for custom fuzzy
number types. The system is built around three core principles:

1. **Type Specialization**: Each fuzzy number type (mtype) has its own dedicated
   random generator
2. **High Performance**: Vectorized batch generation optimized for large-scale
   Fuzzarray creation
3. **Unified Interface**: Consistent API across all generator implementations
   through abstract base classes

The random generation framework consists of two main abstract base classes:

- :class:`~axisfuzzy.random.base.BaseRandomGenerator`: Core interface defining
  the essential methods
- :class:`~axisfuzzy.random.base.ParameterizedRandomGenerator`: Enhanced base
  class with distribution sampling utilities

This modular design ensures that custom generators integrate seamlessly with
AxisFuzzy's existing random generation API while maintaining optimal performance
characteristics.

Architecture and Interface Design
---------------------------------

The AxisFuzzy random generation framework provides a robust architecture for
implementing custom fuzzy number generators. This section introduces the core
interfaces and design patterns that enable seamless integration with the
framework's high-performance random generation system.

Understanding the Generator Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework defines two primary abstract base classes that form the foundation
of all random generators. Understanding their roles and relationships is
essential
for implementing custom generators.

**BaseRandomGenerator Interface**

The ``BaseRandomGenerator`` class establishes the fundamental contract that all
generators must fulfill. This interface ensures consistency across different
fuzzy
number types while maintaining flexibility for type-specific implementations.

.. code-block:: python

    from axisfuzzy.random.base import BaseRandomGenerator
    
    class CustomGenerator(BaseRandomGenerator):
        # Type identifier for the fuzzy number type
        mtype = "custom_type"
        
        def get_default_parameters(self) -> Dict[str, Any]:
            """Return default parameter values for generation."""
            return {'param1': 1.0, 'param2': 0.5}
        
        def validate_parameters(self, **params) -> None:
            """Validate parameter values before generation."""
            # Implementation-specific validation logic
            pass
        
        def fuzznum(self, rng: np.random.Generator, **params) -> 'Fuzznum':
            """Generate a single fuzzy number instance."""
            # Single instance generation logic
            pass
        
        def fuzzarray(self, rng: np.random.Generator, 
                     shape: Tuple[int, ...], **params) -> 'Fuzzarray':
            """Generate a batch of fuzzy numbers with vectorized operations."""
            # High-performance batch generation logic
            pass

**ParameterizedRandomGenerator Enhancement**

The ``ParameterizedRandomGenerator`` extends the base interface with utilities
for
parameter management and distribution sampling. This class provides common
functionality that most generators require, reducing implementation complexity.

.. code-block:: python

    from axisfuzzy.random.base import ParameterizedRandomGenerator
    
    class EnhancedGenerator(ParameterizedRandomGenerator):
        mtype = "enhanced_type"
        
        def get_default_parameters(self) -> Dict[str, Any]:
            return {
                'low': 0.0, 'high': 1.0, 'dist': 'uniform',
                'a': 2.0, 'b': 5.0  # Beta distribution parameters
            }
        
        def validate_parameters(self, **params) -> None:
            # Leverage built-in validation utilities
            self._validate_range('low', params['low'], 0.0, 1.0)
            self._validate_range('high', params['high'], 0.0, 1.0)

Parameter Management Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Effective parameter management ensures robust and user-friendly generators. The
framework provides standardized patterns for parameter declaration, validation,
and merging.

**Parameter Declaration Patterns**

Parameters should be declared with meaningful defaults and comprehensive
validation. The framework supports both simple range validation and complex
cross-parameter constraints.

.. code-block:: python

    def get_default_parameters(self) -> Dict[str, Any]:
        """Define comprehensive parameter set with sensible defaults."""
        return {
            # Core generation parameters
            'md_low': 0.0, 'md_high': 1.0, 'md_dist': 'uniform',
            'nu_low': 0.0, 'nu_high': 1.0, 'nu_dist': 'uniform',
            
            # Distribution-specific parameters
            'a': 2.0, 'b': 5.0,  # Beta distribution shape parameters
            'loc': 0.0, 'scale': 1.0,  # Normal distribution parameters
            
            # Generation mode controls
            'nu_mode': 'orthopair'  # 'orthopair' or 'independent'
        }

**Validation Implementation Strategy**

Parameter validation should occur at multiple levels: individual parameter
ranges, cross-parameter consistency, and mathematical constraint satisfaction.

.. code-block:: python

    def validate_parameters(self, q: int, **kwargs) -> None:
        """Implement comprehensive parameter validation."""
        params = self._merge_parameters(**kwargs)
        
        # Range validation using framework utilities
        self._validate_range('md_low', params['md_low'], 0.0, 1.0)
        self._validate_range('md_high', params['md_high'], 0.0, 1.0)
        
        # Cross-parameter consistency checks
        if params['md_low'] >= params['md_high']:
            raise ValueError("md_low must be less than md_high")
        
        # Distribution parameter validation
        if params['md_dist'] == 'beta' and (params['a'] <= 0 or params['b'] <= 0):
            raise ValueError("Beta distribution requires positive shape "
                           "parameters")

Distribution Sampling Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework integrates seamlessly with NumPy's random generation system,
providing built-in sampling utilities for common distributions while maintaining
extensibility for custom sampling strategies.

**Leveraging Built-in Sampling Utilities**

The ``ParameterizedRandomGenerator`` provides the ``_sample_from_distribution``
method that supports multiple probability distributions with consistent
parameter interfaces.

.. code-block:: python

    def fuzznum(self, rng: np.random.Generator, **params) -> 'Fuzznum':
        """Generate single instance using distribution sampling."""
        params = self._merge_parameters(**params)
        
        # Sample membership degree using specified distribution
        md = self._sample_from_distribution(
            rng, size=None, dist=params['md_dist'],
            low=params['md_low'], high=params['md_high'],
            a=params['a'], b=params['b'],
            loc=params['loc'], scale=params['scale']
        )
        
        # Apply mathematical constraints for valid fuzzy numbers
        # ... constraint application logic
        
        return Fuzznum(mtype=self.mtype).create(md=md, **other_components)

**High-Performance Vectorized Generation**

For batch generation, the framework emphasizes vectorized operations that avoid
creating intermediate objects and leverage NumPy's optimized array operations.

.. code-block:: python

    def fuzzarray(self, rng: np.random.Generator, 
                 shape: Tuple[int, ...], **params) -> 'Fuzzarray':
        """Implement high-performance batch generation."""
        params = self._merge_parameters(**params)
        size = int(np.prod(shape))
        
        # Vectorized sampling for all components
        mds = self._sample_from_distribution(
            rng, size=size, dist=params['md_dist'],
            low=params['md_low'], high=params['md_high'],
            a=params['a'], b=params['b']
        )
        
        # Apply vectorized constraints and reshape
        mds = mds.reshape(shape)
        
        # Create backend directly from arrays (avoid intermediate objects)
        backend = CustomBackend.from_arrays(mds=mds, **other_arrays)
        return Fuzzarray(backend=backend)

NumPy Integration and Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework's integration with NumPy's random generation system ensures
reproducibility, performance, and compatibility with the broader scientific
Python ecosystem.

**Random Number Generator Management**

All generation methods receive a ``np.random.Generator`` instance, ensuring
proper seed management and statistical independence across different generation
calls.

.. code-block:: python

    # Framework handles RNG lifecycle automatically
    def fuzznum(self, rng: np.random.Generator, **params) -> 'Fuzznum':
        # Use provided RNG for all random operations
        sample = rng.uniform(low=0.0, high=1.0)
        # Never create new RNG instances within generators

**Performance Optimization Guidelines**

- **Vectorization**: Use NumPy array operations instead of Python loops
- **Memory Efficiency**: Avoid creating intermediate Fuzznum objects in batch
  generation
- **Backend Integration**: Populate backend arrays directly for optimal
  performance
- **Constraint Application**: Apply mathematical constraints using vectorized
  operations

QROFN Random Generator Implementation  
-------------------------------------

This section demonstrates the complete implementation of a QROFN random
generator, serving as a practical example for developing custom fuzzy number
generators. The QROFN generator showcases advanced features including orthopair
constraint handling, multiple distribution support, and high-performance
vectorized generation.

Class Structure and Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The QROFN generator implementation begins with proper class declaration and
automatic registration with the framework's generator registry.

.. code-block:: python

    from axisfuzzy.random import register_random
    from axisfuzzy.random.base import ParameterizedRandomGenerator
    from axisfuzzy.core import Fuzznum, Fuzzarray
    from .backend import QROFNBackend
    
    @register_random
    class QROFNRandomGenerator(ParameterizedRandomGenerator):
        """
        Random generator for Q-Rung Orthopair Fuzzy Numbers (QROFNs).
        
        Supports multiple probability distributions and orthopair constraint
        handling for generating mathematically valid QROFN instances.
        """
        mtype = "qrofn"

The ``@register_random`` decorator automatically registers the generator with
the global registry, enabling access through the unified
``axisfuzzy.random.rand()`` interface.

Parameter Definition and Default Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QROFN generation requires comprehensive parameter management to support flexible
distribution sampling and constraint handling modes.

.. code-block:: python

    def get_default_parameters(self) -> Dict[str, Any]:
        """Define comprehensive parameter set for QROFN generation."""
        return {
            # Membership degree (μ) distribution parameters
            'md_dist': 'uniform', 'md_low': 0.0, 'md_high': 1.0,
            
            # Non-membership degree (ν) distribution parameters  
            'nu_mode': 'orthopair',  # 'orthopair' or 'independent'
            'nu_dist': 'uniform', 'nu_low': 0.0, 'nu_high': 1.0,
            
            # Distribution-specific shape parameters
            'a': 2.0, 'b': 2.0,  # Beta distribution parameters
            'loc': 0.5, 'scale': 0.15,  # Normal distribution parameters
        }

The parameter set supports multiple probability distributions (uniform, beta,
normal) and provides two constraint handling modes: ``orthopair`` mode ensures
mathematical validity by dynamically adjusting sampling ranges, while
``independent`` mode samples components independently and applies post-generation
constraint enforcement.

Validation Logic Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter validation ensures mathematical consistency and prevents invalid
generation configurations that could produce malformed QROFN instances.

.. code-block:: python

    def validate_parameters(self, q: int, **kwargs) -> None:
        """Implement comprehensive parameter validation for QROFN generation."""
        params = self._merge_parameters(**kwargs)
        
        # Range validation for membership and non-membership bounds
        self._validate_range('md_low', params['md_low'], 0.0, 1.0)
        self._validate_range('md_high', params['md_high'], 0.0, 1.0)
        self._validate_range('nu_low', params['nu_low'], 0.0, 1.0)
        self._validate_range('nu_high', params['nu_high'], 0.0, 1.0)
        
        # Cross-parameter consistency validation
        if params['md_low'] >= params['md_high']:
            raise ValueError("md_low must be less than md_high")
        if params['nu_low'] >= params['nu_high']:
            raise ValueError("nu_low must be less than nu_high")
        
        # Distribution-specific parameter validation
        if params['md_dist'] == 'beta' and (params['a'] <= 0 or params['b'] <= 0):
            raise ValueError("Beta distribution requires positive shape "
                           "parameters")
        
        # Constraint mode validation
        if params['nu_mode'] not in ['orthopair', 'independent']:
            raise ValueError("nu_mode must be 'orthopair' or 'independent'")

Single Instance Generation (fuzznum method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``fuzznum`` method demonstrates constraint-aware generation for individual
QROFN instances, showcasing the orthopair constraint :math:`\mu^q + \nu^q \leq 1`
handling.

.. code-block:: python

    def fuzznum(self, rng: np.random.Generator, 
                q: Optional[int] = None, **kwargs) -> 'Fuzznum':
        """Generate a single QROFN instance with constraint handling."""
        params = self._merge_parameters(**kwargs)
        q = q if q is not None else get_config().DEFAULT_Q
        self.validate_parameters(q=q, **params)
        
        # Sample membership degree using specified distribution
        md = self._sample_from_distribution(
            rng, size=None, dist=params['md_dist'],
            low=params['md_low'], high=params['md_high'],
            a=params['a'], b=params['b'],
            loc=params['loc'], scale=params['scale']
        )
        
        # Handle non-membership degree based on constraint mode
        if params['nu_mode'] == 'orthopair':
            # Calculate maximum allowed non-membership degree
            max_nmd = (1 - md ** q) ** (1 / q)
            effective_high = min(params['nu_high'], max_nmd)
            
            # Sample within constrained range
            nmd_sample = self._sample_from_distribution(
                rng, size=None, dist=params['nu_dist'], 
                low=0.0, high=1.0,
                a=params['a'], b=params['b'],
                loc=params['loc'], scale=params['scale']
            )
            nmd = params['nu_low'] + nmd_sample * (effective_high -
                                                   params['nu_low'])
            nmd = max(nmd, params['nu_low'])
            
        else:  # 'independent' mode
            nmd = self._sample_from_distribution(
                rng, size=None, dist=params['nu_dist'],
                low=params['nu_low'], high=params['nu_high'],
                a=params['a'], b=params['b'],
                loc=params['loc'], scale=params['scale']
            )
            # Apply constraint enforcement if violated
            if (md ** q + nmd ** q) > 1.0:
                nmd = (1 - md ** q) ** (1 / q)
        
        return Fuzznum(mtype='qrofn', q=q).create(md=md, nmd=nmd)

High-Performance Batch Generation (fuzzarray method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``fuzzarray`` method implements vectorized generation for optimal
performance when creating large batches of QROFN instances.

.. code-block:: python

    def fuzzarray(self, rng: np.random.Generator,
                  shape: Tuple[int, ...], q: Optional[int] = None,
                  **params) -> 'Fuzzarray':
        """Generate QROFN batch using high-performance vectorized operations."""
        params = self._merge_parameters(**params)
        q = q if q is not None else get_config().DEFAULT_Q
        self.validate_parameters(q=q, **params)
        
        size = int(np.prod(shape))
        
        # Vectorized membership degree generation
        mds = self._sample_from_distribution(
            rng, size=size, dist=params['md_dist'],
            low=params['md_low'], high=params['md_high'],
            a=params['a'], b=params['b'],
            loc=params['loc'], scale=params['scale']
        )
        
        # Vectorized non-membership degree generation with constraint handling
        if params['nu_mode'] == 'orthopair':
            # Calculate element-wise maximum allowed non-membership degrees
            max_nmd = (1 - mds ** q) ** (1 / q)
            effective_high = np.minimum(params['nu_high'], max_nmd)
            
            # Sample and scale to dynamic ranges
            nmds = self._sample_from_distribution(
                rng, size=size, dist=params['nu_dist'],
                low=params['nu_low'], high=1.0,
                a=params['a'], b=params['b'],
                loc=params['loc'], scale=params['scale']
            )
            nmds = params['nu_low'] + nmds * (effective_high - params['nu_low'])
            nmds = np.maximum(nmds, params['nu_low'])
            
        else:  # 'independent' mode with vectorized constraint enforcement
            nmds = self._sample_from_distribution(
                rng, size=size, dist=params['nu_dist'],
                low=params['nu_low'], high=params['nu_high'],
                a=params['a'], b=params['b'],
                loc=params['loc'], scale=params['scale']
            )
            # Vectorized constraint enforcement
            violates_mask = (mds ** q + nmds ** q) > 1.0
            if np.any(violates_mask):
                max_nmd_violating = (1 - mds[violates_mask] ** q) ** (1 / q)
                nmds[violates_mask] = np.minimum(nmds[violates_mask],
                                                 max_nmd_violating)
        
        # Reshape and create backend directly from arrays
        mds = mds.reshape(shape)
        nmds = nmds.reshape(shape)
        backend = QROFNBackend.from_arrays(mds=mds, nmds=nmds, q=q)
        
        return Fuzzarray(backend=backend)

Mathematical Constraint Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The QROFN implementation demonstrates sophisticated constraint handling that
maintains mathematical validity while supporting flexible generation modes.

**Orthopair Mode**: Dynamically calculates valid sampling ranges based on the
constraint :math:`\mu^q + \nu^q \leq 1`, ensuring all generated instances
satisfy the mathematical requirements without post-generation rejection.

**Independent Mode**: Allows independent sampling followed by constraint
enforcement, providing greater flexibility for specific distribution
requirements while maintaining mathematical validity through post-generation
adjustment.

Registration and Integration
----------------------------

This section demonstrates how to integrate your custom random generator into
AxisFuzzy's global registry system, enabling seamless access through the unified
API. The QROFN generator serves as our reference implementation.

Automatic Registration with Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AxisFuzzy provides the ``@register_random`` decorator for automatic generator
registration. This decorator handles all registration logic and validation:

.. code-block:: python

    from axisfuzzy.random import register_random
    from axisfuzzy.random.base import ParameterizedRandomGenerator

    @register_random
    class QROFNRandomGenerator(ParameterizedRandomGenerator):
        """
        High-performance random generator for q-rung orthopair fuzzy numbers.
        """
        mtype = "qrofn"  # Required: unique identifier

        def get_default_parameters(self) -> Dict[str, Any]:
            return {
                'md_dist': 'uniform',
                'md_low': 0.0,
                'md_high': 1.0,
                'nu_mode': 'orthopair',
                # ... other parameters
            }

        # Implementation methods...

The decorator performs several critical operations:

1. **Type Validation**: Ensures the class inherits from ``BaseRandomGenerator``
2. **mtype Registration**: Maps the generator's ``mtype`` to the class instance
3. **Singleton Creation**: Instantiates and registers a single generator
   instance
4. **Thread Safety**: Uses locks to ensure safe concurrent registration

Registry System Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once registered, generators become accessible through AxisFuzzy's unified API.
The registry system provides several access patterns:

.. code-block:: python

    from axisfuzzy.random.registry import (
        get_random_generator,
        list_registered_random,
        is_registered_random
    )

    # Check if generator is available
    if is_registered_random('qrofn'):
        generator = get_random_generator('qrofn')
        
    # List all registered generators
    available_types = list_registered_random()
    print(f"Available generators: {available_types}")

The registry maintains thread-safe access to all generators and supports
dynamic registration during runtime.

API Integration and Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

Registered generators automatically integrate with the high-level API functions:

.. code-block:: python

    import axisfuzzy.random as fr

    # Single fuzzy number generation
    num = fr.rand('qrofn', q=2)

    # Array generation with custom parameters
    arr = fr.rand('qrofn', 
                  shape=(1000, 100), 
                  q=3,
                  md_dist='beta',
                  a=2.0, b=5.0,
                  nu_mode='orthopair')

    # Seeded generation for reproducibility
    reproducible_arr = fr.rand('qrofn', 
                               shape=(500,), 
                               q=2, 
                               seed=42)

Testing and Validation Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive testing ensures generator reliability and performance. Key testing
areas include:

**Parameter Validation Testing**:

.. code-block:: python

    def test_parameter_validation():
        generator = get_random_generator('qrofn')
        
        # Test valid parameters
        generator.validate_parameters(q=2, md_low=0.0, md_high=1.0)
        
        # Test invalid parameters (should raise exceptions)
        with pytest.raises(ValueError):
            generator.validate_parameters(q=0)  # Invalid q value

**Output Correctness Testing**:

.. code-block:: python

    def test_output_constraints():
        arr = fr.rand('qrofn', shape=(1000,), q=2, seed=42)
        
        # Verify mathematical constraints
        assert np.all(arr.md >= 0) and np.all(arr.md <= 1)
        assert np.all(arr.nmd >= 0) and np.all(arr.nmd <= 1)
        assert np.all(arr.md**2 + arr.nmd**2 <= 1)  # q=2 constraint

**Performance Benchmarking**:

.. code-block:: python

    def benchmark_generation():
        import time
        
        sizes = [1000, 10000, 100000]
        for size in sizes:
            start = time.time()
            arr = fr.rand('qrofn', shape=(size,), q=2)
            duration = time.time() - start
            print(f"Generated {size} QROFNs in {duration:.4f}s")

Best Practices for Production Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Error Handling**: Implement robust error handling for edge cases:

.. code-block:: python

    def safe_generation(mtype, **params):
        try:
            if not is_registered_random(mtype):
                raise ValueError(f"Generator '{mtype}' not registered")
            return fr.rand(mtype, **params)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

**2. Parameter Caching**: Cache validated parameters for repeated operations:

.. code-block:: python

    class CachedGenerator:
        def __init__(self, mtype):
            self.generator = get_random_generator(mtype)
            self._param_cache = {}
        
        def generate(self, **params):
            param_key = tuple(sorted(params.items()))
            if param_key not in self._param_cache:
                self.generator.validate_parameters(**params)
                self._param_cache[param_key] = params
            return self.generator.fuzzarray(rng, **params)

**3. Memory Management**: For large-scale generation, consider memory-efficient
patterns:

.. code-block:: python

    def generate_batches(total_size, batch_size=10000, **params):
        """Generate large arrays in batches to manage memory usage."""
        batches = []
        for i in range(0, total_size, batch_size):
            current_batch_size = min(batch_size, total_size - i)
            batch = fr.rand('qrofn', shape=(current_batch_size,), **params)
            batches.append(batch)
        return np.concatenate(batches)

Conclusion
----------

This guide has demonstrated the complete workflow for developing custom random
generators in AxisFuzzy, using the QROFN generator as a comprehensive example.
The development process follows a structured three-phase approach:

**Phase 1: Architecture and Interface Design** establishes the foundation
through proper inheritance from ``ParameterizedRandomGenerator``, comprehensive
parameter management, and integration with AxisFuzzy's distribution sampling
framework.

**Phase 2: Implementation** focuses on core generation logic, including robust
parameter validation, efficient single-instance and batch generation methods,
and proper handling of mathematical constraints specific to your fuzzy number
type.

**Phase 3: Registration and Integration** ensures seamless integration with
AxisFuzzy's ecosystem through automatic registration, comprehensive testing
workflows, and production-ready deployment practices.

**Key Design Principles**:

- **Modularity**: Each generator is self-contained with clear interfaces
- **Performance**: Vectorized operations and efficient memory management
- **Extensibility**: Plugin-style architecture supports diverse fuzzy number
  types
- **Reliability**: Comprehensive validation and error handling throughout

**Recommendations for Extension**:

1. Follow the established patterns demonstrated by existing generators
2. Implement comprehensive parameter validation for your specific mathematical
   constraints
3. Optimize for vectorized operations to handle large-scale generation
   efficiently
4. Include thorough testing covering edge cases and performance benchmarks
5. Document your generator's mathematical foundations and usage patterns

By following this development framework, you can create robust, high-performance
random generators that integrate seamlessly with AxisFuzzy's unified API,
contributing to the library's extensible architecture while maintaining
consistency and reliability across all fuzzy number types.
