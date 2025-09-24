.. _config_guide:

Configuration system
====================

The :mod:`axisfuzzy.config` module provides a sophisticated, centralized configuration
management system designed specifically for scientific computing libraries. This system
serves as the unified control center for all configurable behaviors within `AxisFuzzy`,
from numerical precision and default fuzzy number types to performance optimization
and debugging features.

As a professional computational library for fuzzy mathematics, `AxisFuzzy` requires
precise control over various computational parameters and behavioral settings.
The configuration system addresses this need by providing a type-safe, validated,
and persistent configuration framework that maintains consistency across all
library operations.

.. contents::
   :local:



Configuration System: Centralized Management for `AxisFuzzy`
------------------------------------------------------------

Core Design Philosophy
~~~~~~~~~~~~~~~~~~~~~~

The configuration system is built upon four fundamental design principles:

**Centralized Management**
    All configuration parameters are consolidated within a single :class:`~axisfuzzy.config.config_file.Config`
    dataclass, providing a unified source of truth for library behavior. This design
    eliminates configuration fragmentation and ensures consistent parameter access
    across all `AxisFuzzy` modules.

**Type Safety and Validation**
    Every configuration parameter includes comprehensive metadata with validation
    functions that enforce type constraints and value ranges. This prevents invalid
    configurations that could compromise computational accuracy or system stability.

**Simplicity Through Abstraction**
    While the underlying architecture is sophisticated, the user-facing API consists
    of intuitive functions like :func:`~axisfuzzy.config.get_config` and
    :func:`~axisfuzzy.config.set_config`. Users interact with a clean interface
    without needing to understand the complex validation and management logic.

**Persistence and Portability**
    Configuration states can be serialized to and loaded from JSON files, enabling
    reproducible computational environments across different projects, systems,
    and research contexts.

Architectural Overview
~~~~~~~~~~~~~~~~~~~~~~

The configuration system employs a layered architecture that separates concerns
while maintaining tight integration:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────┐
    │                    User Interface Layer                 │
    │  axisfuzzy.config.get_config() | set_config() | load()  │
    └─────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────────┐
    │                   API Abstraction Layer                 │
    │              (axisfuzzy.config.api)                     │
    │  • Global convenience functions                         │
    │  • Parameter validation delegation                      │
    └─────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────────┐
    │                 Management Layer                        │
    │             (ConfigManager Singleton)                   │
    │   • Thread-safe configuration state                     │
    │   • Validation orchestration                            │
    │   • File I/O operations                                 │
    └─────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────────┐
    │                   Data Model Layer                      │
    │               (Config Dataclass)                        │
    │   • Parameter definitions and defaults                  │
    │   • Validation rules and metadata                       │
    │   • Type annotations and documentation                  │
    └─────────────────────────────────────────────────────────┘

Component Relationships
~~~~~~~~~~~~~~~~~~~~~~~

The system consists of four primary components that work in concert:

:mod:`axisfuzzy.config.config_file`
    Defines the :class:`~axisfuzzy.config.config_file.Config` dataclass containing
    all configuration parameters with their default values, type annotations,
    and validation metadata. This serves as the authoritative schema for all
    configurable behaviors.

:mod:`axisfuzzy.config.manager`
    Implements the :class:`~axisfuzzy.config.manager.ConfigManager` singleton
    that maintains the global configuration state. Provides thread-safe operations
    for parameter validation, file persistence, and state management.

:mod:`axisfuzzy.config.api`
    Exposes high-level convenience functions that abstract the complexity of
    the manager interface. These functions serve as the primary user interaction
    points with the configuration system.

:mod:`axisfuzzy.config.__init__`
    Orchestrates the public API by importing and exposing the essential functions
    and classes, creating a clean namespace for user consumption.

This architectural design ensures that the configuration system remains both
powerful enough to handle complex scientific computing requirements and simple
enough for everyday use by researchers and developers.





Core Architecture and Components
--------------------------------

The `AxisFuzzy` configuration system is built upon a carefully designed
three-layer architecture that separates concerns while maintaining tight
integration. Each component serves a specific purpose in the overall system,
from data definition to user interaction.

Config Dataclass: Schema and Metadata Foundation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~axisfuzzy.config.config_file.Config` dataclass defines all
configuration parameters with metadata for validation and categorization.

.. code-block:: python

    DEFAULT_PRECISION: int = field(
        default=4,
        metadata={
            'category': 'basic',
            'description': 'Default calculation precision (decimal places)',
            'validator': lambda x: isinstance(x, int) and x >= 0,
            'error_msg': "Must be a non-negative integer."
        }
    )

The metadata dictionary contains four essential components:

- **Category Classification**
    Groups related parameters for logical organization and documentation.
    Current categories include ``basic``, ``performance``, ``debug``, and
    ``display``.

- **Descriptive Documentation**
    Provides human-readable explanations of each parameter's purpose and
    impact on system behavior.

- **Validation Logic**
    Lambda functions that enforce type safety and value constraints,
    preventing invalid configurations from corrupting system state.

- **Error Messaging**
    User-friendly error descriptions that guide users toward correct
    parameter values when validation fails.

Configuration parameters are organized into categories:

**Basic Configuration** (``basic``)
    Core parameters like default fuzzy number types and numerical precision.

**Performance Configuration** (``performance``)
    Parameters controlling computational efficiency and memory usage.

**Debug Configuration** (``debug``)
    Development and diagnostic parameters for verification and logging.

**Display Configuration** (``display``)
    Parameters controlling the presentation of arrays and data structures.

ConfigManager
~~~~~~~~~~~~~

The :class:`~axisfuzzy.config.manager.ConfigManager` implements a thread-safe
singleton pattern that maintains global configuration state and handles
validation using the metadata from the Config dataclass.

Key features:
- Singleton pattern ensures consistent state across all modules
- Automatic validation using embedded metadata
- JSON-based persistence for configuration sharing
- Thread-safe operations for multi-threaded environments

API Layer
~~~~~~~~~

The :mod:`axisfuzzy.config.api` module provides a simple functional interface:

:func:`~axisfuzzy.config.api.get_config`
    Returns the current configuration instance.

:func:`~axisfuzzy.config.api.set_config`
    Updates configuration parameters with validation.

:func:`~axisfuzzy.config.api.load_config_file`
    Loads configuration from JSON files.

:func:`~axisfuzzy.config.api.save_config_file`
    Saves current configuration to JSON format.

:func:`~axisfuzzy.config.api.reset_config`
    Restores all parameters to default values.

The configuration system is designed for thread-safe operation in
multi-threaded scientific computing environments. The singleton manager
uses appropriate locking mechanisms to ensure that configuration changes
are atomic and visible across all threads.

This design enables safe use of `AxisFuzzy` in parallel computing scenarios,
including distributed fuzzy number operations and concurrent analysis
pipelines.



Configuration Categories and Parameters
---------------------------------------

The `AxisFuzzy` configuration system organizes parameters into logical
categories, each serving specific aspects of the library's behavior.
This section provides comprehensive documentation of all configuration
parameters, their purposes, validation rules, and interdependencies.

Configuration Taxonomy
~~~~~~~~~~~~~~~~~~~~~~

Parameters are systematically categorized using metadata-driven classification:

.. code-block:: python

   @dataclass
   class Config:
       PARAMETER_NAME: type = field(
           default=value,
           metadata={
               'category': 'basic|performance|debug|display',
               'description': 'Detailed parameter description',
               'validator': lambda x: validation_logic(x),
               'error_msg': 'User-friendly error message'
           }
       )

This metadata-driven approach enables automatic validation, categorization,
and documentation generation while maintaining type safety.

Basic Configuration Parameters
++++++++++++++++++++++++++++++

Fundamental parameters that define core computational behavior:

**DEFAULT_MTYPE: str**
    Default fuzzy number type for object construction.
    
    :Default: ``'qrofn'``
    :Impact: Affects all ``Fuzznum`` instantiations without explicit type

**DEFAULT_Q: int**
    Default q-rung parameter for orthopair fuzzy numbers.
    
    :Default: ``1``
    :Impact: Controls generalization level (μᵍ + νᵍ ≤ 1)

**DEFAULT_PRECISION: int**
    Numerical precision for computational operations.
    
    :Default: ``4``
    :Impact: Affects display formatting and numerical stability

**DEFAULT_EPSILON: float**
    Numerical tolerance for floating-point comparisons.
    
    :Default: ``1e-12``
    :Impact: Determines equality thresholds and zero-value detection

Performance Configuration Parameters
++++++++++++++++++++++++++++++++++++

Parameters that impact computational efficiency and memory usage:

**CACHE_SIZE: int**
    Maximum entries in operation result caches.
    
    :Default: ``256``
    :Impact: Controls memory vs. speed trade-off (0 disables caching)

Debug Configuration Parameters
++++++++++++++++++++++++++++++

Parameters for development, testing, and verification:

**TNORM_VERIFY: bool**
    Enables T-norm mathematical properties verification.
    
    :Default: ``False``
    :Impact: Significantly affects initialization performance (1-5x slower)

Display Configuration Parameters
++++++++++++++++++++++++++++++++

The **display** category manages array visualization and output formatting
for large-scale fuzzy computations.

**Array Size Thresholds**
    The system defines four size categories with corresponding display strategies:
    
   .. list-table:: Display Thresholds
      :header-rows: 1

      * - Parameter
        - Value
        - Description
      * - DISPLAY_THRESHOLD_SMALL
        - 1,000
        - Arrays below this threshold are displayed in full
      * - DISPLAY_THRESHOLD_MEDIUM  
        - 10,000
        - Medium arrays show edge elements with central truncation
      * - DISPLAY_THRESHOLD_LARGE
        - 100,000 
        - Large arrays use aggressive truncation with minimal edge display
      * - DISPLAY_THRESHOLD_HUGE
        - 1,000,000
        - Huge arrays show only essential structural information

**Edge Display Parameters**
    Control the number of elements shown at array boundaries:
    
    .. list-table:: Display Edge Items Configuration
       :header-rows: 1

       * - Parameter
         - Elements per Dimension
         - Description
       * - MEDIUM
         - 3
         - Shows 3 elements at each edge for medium-sized arrays
       * - LARGE
         - 3
         - Shows 3 elements at each edge for large arrays
       * - HUGE
         - 2
         - Shows only 2 elements at each edge for huge arrays
    
    These parameters balance information content with readability,
    ensuring that users can understand array structure without
    overwhelming console output.

Parameter Validation Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system implements a sophisticated validation framework
that ensures type safety and logical consistency.

**Metadata-Driven Validation**
    Each parameter includes a ``validator`` function in its metadata:
    
    .. code-block:: python
    
       'validator': lambda x: isinstance(x, int) and x >= 0
    
    These validators are automatically invoked during configuration updates,
    providing immediate feedback for invalid values.

**Error Message Design**
    Validation errors include:
    
    - **Parameter name** and **invalid value**
    - **Specific validation rule** that failed
    - **Suggested correction** when applicable
    
    Example error message:
    
    .. code-block:: text
    
       ValueError: Invalid value for 'DEFAULT_PRECISION': -2.
       Must be a non-negative integer.

**Constraint Rules and Dependencies**
    While most parameters are independent, some logical relationships exist:
    
    - Display thresholds should maintain ordering: SMALL < MEDIUM < LARGE < HUGE
    - Edge items should be reasonable relative to threshold sizes
    - Epsilon values should be appropriate for the chosen precision
    
    Future versions may implement cross-parameter validation to enforce
    these relationships automatically.

Configuration Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Performance Optimization**
    - Set ``CACHE_SIZE`` based on available memory and usage patterns
    - Use higher precision only when numerical accuracy is critical
    - Disable ``TNORM_VERIFY`` in production environments

**Memory Management**
    - Monitor cache memory usage in long-running applications
    - Adjust display thresholds for memory-constrained environments
    - Consider precision impact on memory footprint

**Development Workflow**
    - Enable ``TNORM_VERIFY`` during algorithm development
    - Use configuration files for reproducible research
    - Document configuration choices in scientific publications

This comprehensive parameter system provides fine-grained control over
`AxisFuzzy`'s behavior while maintaining ease of use through sensible
defaults and robust validation mechanisms.





Configuration Management Operations
-----------------------------------

The `AxisFuzzy` configuration system provides a comprehensive set of operations
for managing configuration state throughout the application lifecycle. These
operations are exposed through a high-level API that abstracts the underlying
manager implementation while preserving full functionality.

Configuration Reading Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system provides multiple approaches for accessing current
configuration values, each optimized for different use cases.

**Primary Access Method: get_config()**
    The :func:`~axisfuzzy.config.api.get_config` function returns the active
    :class:`~axisfuzzy.config.config_file.Config` instance, providing direct
    access to all configuration parameters:
    
    .. code-block:: python
    
       from axisfuzzy.config.api import get_config
       
       # Get the complete configuration object
       config = get_config()
       
       # Access individual parameters
       precision = config.DEFAULT_PRECISION
       cache_size = config.CACHE_SIZE
       
       # Check configuration state
       print(f"Current precision: {precision}")
       print(f"Cache size: {cache_size}")

**Attribute Access Pattern**
    Configuration parameters are accessed as standard Python attributes,
    leveraging the dataclass implementation for type safety and IDE support:
    
    .. code-block:: python
    
       config = get_config()
       
       # Direct attribute access
       if config.TNORM_VERIFY:
           print("T-norm verification enabled")
       
       # Use in conditional logic
       threshold = (config.DISPLAY_THRESHOLD_MEDIUM 
                   if array_size < 50000 
                   else config.DISPLAY_THRESHOLD_LARGE)

Configuration Modification Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration updates are performed through the :func:`~axisfuzzy.config.api.set_config`
function, which provides atomic updates with comprehensive validation.

**Single Parameter Updates**
    
    .. code-block:: python
    
       from axisfuzzy.config.api import set_config
       
       # Update calculation precision
       set_config(DEFAULT_PRECISION=6)
       
       # Enable debug verification
       set_config(TNORM_VERIFY=True)

**Batch Parameter Updates**
    Multiple parameters can be updated atomically in a single operation:
    
    .. code-block:: python
    
       # Configure for high-precision scientific computing
       set_config(
           DEFAULT_PRECISION=8,
           DEFAULT_EPSILON=1e-15,
           CACHE_SIZE=1024
       )

**Validation and Error Handling**
    All parameter updates undergo validation using metadata-driven rules:
    
    .. code-block:: python
    
       try:
           set_config(DEFAULT_PRECISION=-1)  # Invalid value
       except ValueError as e:
           print(f"Validation error: {e}")
           # Output: Invalid value for 'DEFAULT_PRECISION': -1.
           #         Must be a non-negative integer.

Configuration Persistence Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system supports loading and saving configuration state
to JSON files, enabling reproducible research and deployment scenarios.

**Loading Configuration Files**
    The :func:`~axisfuzzy.config.api.load_config_file` function reads
    configuration from JSON files with comprehensive error handling:
    
    .. code-block:: python
    
       from axisfuzzy.config.api import load_config_file
       
       # Load from file path
       load_config_file("/path/to/config.json")
       
       # Load with error handling
       try:
           load_config_file("research_config.json")
       except FileNotFoundError:
           print("Configuration file not found")
       except ValueError as e:
           print(f"Invalid configuration: {e}")

**Saving Configuration Files**
    Current configuration state can be persisted using
    :func:`~axisfuzzy.config.api.save_config_file`:
    
    .. code-block:: python
    
       from axisfuzzy.config.api import save_config_file
       
       # Save current configuration
       save_config_file("current_config.json")
       
       # Save with automatic directory creation
       save_config_file("/experiments/run_001/config.json")

**JSON Format Structure**
    Configuration files use a flat JSON structure mapping parameter names
    to values:
    
    .. code-block:: json
    
       {
         "DEFAULT_PRECISION": 6,
         "DEFAULT_EPSILON": 1e-12,
         "CACHE_SIZE": 512,
         "TNORM_VERIFY": false
       }

Configuration Reset Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system provides reset functionality to restore default
state and clear modification tracking.

**Complete Reset**
    The :func:`~axisfuzzy.config.api.reset_config` function restores
    all parameters to their default values:
    
    .. code-block:: python
    
       from axisfuzzy.config.api import reset_config
       
       # Reset to defaults
       reset_config()
       
       # Verify reset
       config = get_config()
       assert config.DEFAULT_PRECISION == 4  # Default value

**State Management**
    Reset operations clear internal state tracking:
    
    - **Configuration source**: Cleared to None
    - **Modification flags**: Reset to False
    - **Parameter values**: Restored to dataclass defaults
    
    This ensures clean state for subsequent operations and testing scenarios.

**Integration with Manager**
    All API functions delegate to the underlying
    :class:`~axisfuzzy.config.manager.ConfigManager` singleton, ensuring
    consistent state management across the application:
    
    .. code-block:: python
    
       from axisfuzzy.config.api import get_config_manager
       
       # Access manager directly if needed
       manager = get_config_manager()
       
       # Check modification state
       if manager.is_modified():
           print("Configuration has been modified")
       
       # Get source information
       source = manager.get_config_source()
       if source:
           print(f"Loaded from: {source}")

These operations provide a complete configuration management solution,
supporting both interactive development and automated deployment workflows
while maintaining data integrity through comprehensive validation.





Advanced Usage Patterns and Best Practices
-------------------------------------------

This section covers advanced configuration management patterns that enable
efficient development workflows, environment-specific configurations, and
robust deployment strategies.

Configuration File Creation and Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system provides sophisticated file management capabilities
through template generation and structured configuration workflows.

**Template Generation**

The :class:`~axisfuzzy.config.manager.ConfigManager` provides a static method
for creating configuration templates populated with default values

.. code-block:: python

    from axisfuzzy.config.manager import ConfigManager
    
    # Create a template with all default values
    ConfigManager.create_config_template('config_template.json')

The generated template includes metadata fields for documentation

.. code-block:: json

    {
        "_comment": "AxisFuzzy Configuration File Template",
        "_description": "Modify parameters as needed",
        "_version": "1.0",
        "DEFAULT_MTYPE": "qrofn",
        "DEFAULT_PRECISION": 4,
        "CACHE_SIZE": 256
        // ... other parameters
    }

**Configuration Validation Workflow**

Best practice involves validating configurations before deployment

.. code-block:: python

    import axisfuzzy.config as config
    
    # Load configuration
    config.load_config_file('production.json')
    
    # Validate all parameters
    manager = config.get_config_manager()
    errors = manager.validate_all_config()
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validated successfully")

Temporary Configuration Modification Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temporary configuration changes are essential for testing, debugging, and
context-specific computations without affecting global state.

**Context-Based Configuration Pattern**

Implement temporary modifications using save/restore patterns

.. code-block:: python

    # Save current state
    original_config = config.get_config()
    original_precision = original_config.DEFAULT_PRECISION
    original_cache = original_config.CACHE_SIZE
    
    try:
        # Apply temporary settings
        config.set_config(
            DEFAULT_PRECISION=8,
            CACHE_SIZE=0  # Disable caching for testing
        )
        
        # Perform operations requiring specific configuration
        result = perform_high_precision_calculation()
        
    finally:
        # Restore original configuration
        config.set_config(
            DEFAULT_PRECISION=original_precision,
            CACHE_SIZE=original_cache
        )

**Configuration Snapshot Management**

For complex temporary modifications, use configuration snapshots

.. code-block:: python

    def with_temporary_config(**temp_settings):
        """Context manager for temporary configuration changes."""
        # Save current configuration to temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as tmp:
            temp_file = tmp.name
        
        try:
            config.save_config_file(temp_file)
            config.set_config(**temp_settings)
            yield
        finally:
            config.load_config_file(temp_file)
            os.unlink(temp_file)

Multi-Environment Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Production systems require environment-specific configurations while
maintaining consistency across development, testing, and deployment stages.

**Environment-Specific Configuration Files**

Organize configurations by environment

.. code-block:: text

    configs/
    ├── base.json          # Common settings
    ├── development.json   # Development overrides
    ├── testing.json       # Testing-specific settings
    └── production.json    # Production optimizations

**Configuration Inheritance Pattern**

Implement configuration layering for environment management

.. code-block:: python

    import json
    from pathlib import Path
    
    def load_environment_config(env_name='development'):
        """Load configuration with environment-specific overrides."""
        config_dir = Path('configs')
        
        # Load base configuration
        base_config = {}
        base_file = config_dir / 'base.json'
        if base_file.exists():
            with open(base_file) as f:
                base_config = json.load(f)
        
        # Load environment-specific overrides
        env_file = config_dir / f'{env_name}.json'
        if env_file.exists():
            with open(env_file) as f:
                env_config = json.load(f)
                base_config.update(env_config)
        
        # Apply merged configuration
        config.reset_config()
        config.set_config(**base_config)
        
        return base_config

**Environment Detection and Auto-Configuration**

Implement automatic environment detection

.. code-block:: python

    import os
    
    def auto_configure_environment():
        """Automatically configure based on environment variables."""
        env = os.getenv('AXISFUZZY_ENV', 'development')
        
        env_configs = {
            'development': {
                'DEFAULT_PRECISION': 4,
                'CACHE_SIZE': 128,
                'TNORM_VERIFY': True
            },
            'testing': {
                'DEFAULT_PRECISION': 6,
                'CACHE_SIZE': 64,
                'TNORM_VERIFY': True
            },
            'production': {
                'DEFAULT_PRECISION': 4,
                'CACHE_SIZE': 512,
                'TNORM_VERIFY': False
            }
        }
        
        if env in env_configs:
            config.set_config(**env_configs[env])
            print(f"Configured for {env} environment")
        else:
            print(f"Unknown environment: {env}, using defaults")

**Configuration Monitoring and Validation**

Implement configuration monitoring for production environments

.. code-block:: python

    def monitor_configuration_health():
        """Monitor configuration state and detect issues."""
        manager = config.get_config_manager()
        
        # Check modification status
        if manager.is_modified():
            print("Warning: Configuration modified since last save")
        
        # Validate current configuration
        errors = manager.validate_all_config()
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        # Check configuration source
        source = manager.get_config_source()
        if source:
            print(f"Configuration loaded from: {source}")
        else:
            print("Using default configuration")
        
        return True

These advanced patterns enable robust configuration management across
different deployment scenarios while maintaining code clarity and
operational reliability.




Integration with `AxisFuzzy` Ecosystem
--------------------------------------

The configuration system serves as the foundational layer that orchestrates
the behavior of all `AxisFuzzy` components. This chapter explores how configuration
parameters influence core data structures, computational performance, and
module interactions, providing guidance for extending and customizing the
configuration framework to meet specialized requirements.

Configuration System and Core Data Structures Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system deeply integrates with `AxisFuzzy`'s core data structures
(``Fuzznum`` and ``Fuzzarray``) through strategic parameter injection and
runtime behavior modification. This integration ensures consistent behavior
across all fuzzy number operations while maintaining flexibility for
specialized use cases.

Default Type Resolution and Factory Integration
+++++++++++++++++++++++++++++++++++++++++++++++

The configuration system controls the default behavior of fuzzy number creation
through the ``DEFAULT_MTYPE`` and ``DEFAULT_Q`` parameters. These settings
influence the :func:`~axisfuzzy.core.fuzzynum` factory function's type
resolution mechanism.

.. code-block:: python

    import axisfuzzy.config as config
    from axisfuzzy.core import fuzzynum
    
    # Configuration affects default fuzzy number creation
    config.set_config(DEFAULT_MTYPE='qrohfn', DEFAULT_Q=3)
    
    # Factory uses configuration defaults
    fnum = fuzzynum(([0.8, 0.6], [0.1, 0.2]))  # Creates q-ROHFN with q=3
    print(f"Type: {fnum.mtype}, Q-value: {fnum.q}")

Precision and Numerical Consistency
++++++++++++++++++++++++++++++++++++

The ``DEFAULT_PRECISION`` parameter ensures numerical consistency across
all fuzzy number operations, affecting both display formatting and
internal calculations.

.. code-block:: python

    # Precision affects all numerical operations
    config.set_config(DEFAULT_PRECISION=6)
    
    fnum = fuzzynum((0.123456789, 0.087654321))
    print(fnum)  
    # Output respects precision setting: <0.123457,0.087654>

Validation and Constraint Enforcement
++++++++++++++++++++++++++++++++++++++

The ``DEFAULT_EPSILON`` parameter controls the tolerance for constraint
validation in fuzzy number strategies, ensuring mathematical consistency
while accommodating floating-point precision limitations.

.. code-block:: python

    # Epsilon affects constraint validation
    config.set_config(DEFAULT_EPSILON=1e-10)
    
    # Stricter validation for constraint checking
    fnum = fuzzynum(md=0.9, nmd=0.4, q=2)
    # Validates: md^q + nmd^q <= 1 + epsilon

Configuration Impact on Fuzzy Operations Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration parameters significantly influence the computational performance
of fuzzy operations through caching mechanisms, validation controls, and
debugging features. Understanding these impacts enables optimal performance
tuning for different deployment scenarios.

Caching and Memory Management
+++++++++++++++++++++++++++++

The ``CACHE_SIZE`` parameter controls the operation result cache, directly
impacting performance for repetitive computations. Proper cache sizing
balances memory usage with computational efficiency.

.. code-block:: python

    # Configure cache for high-performance scenarios
    config.set_config(CACHE_SIZE=1024)  # Larger cache for better performance
    
    # Performance-critical operations benefit from caching
    from axisfuzzy.core.triangular import OperationTNorm
    
    op = OperationTNorm(norm_type='einstein', q=2)
    # Repeated operations use cached results
    result1 = op.t_norm(0.8, 0.6)
    result2 = op.t_norm(0.8, 0.6)  # Retrieved from cache

Validation Performance Trade-offs
+++++++++++++++++++++++++++++++++

The ``TNORM_VERIFY`` parameter enables comprehensive mathematical validation
of t-norm operations at the cost of computational overhead. This setting
should be carefully managed based on deployment requirements.

.. code-block:: python

    # Development: Enable verification for correctness
    config.set_config(TNORM_VERIFY=True)
    
    # Production: Disable for optimal performance
    config.set_config(TNORM_VERIFY=False)
    
    # Verification affects t-norm initialization time
    op = OperationTNorm(norm_type='hamacher', p=2)
    # With TNORM_VERIFY=True: validates commutativity, associativity, etc.

Precision vs. Performance Balance
++++++++++++++++++++++++++++++++++

Higher precision settings may impact performance in computation-intensive
scenarios. The configuration system allows dynamic adjustment based on
accuracy requirements.

.. code-block:: python

    # High-precision scientific computing
    config.set_config(DEFAULT_PRECISION=12, DEFAULT_EPSILON=1e-15)
    
    # Performance-optimized settings
    config.set_config(DEFAULT_PRECISION=4, DEFAULT_EPSILON=1e-8)

Collaboration Patterns with Other Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system establishes standardized collaboration patterns
with `AxisFuzzy`'s extension system, mixin operations, and external integrations.
These patterns ensure consistent behavior across the entire ecosystem while
maintaining modularity and extensibility.

Extension System Integration
++++++++++++++++++++++++++++

Configuration parameters influence the behavior of extension functions,
particularly those involving numerical computations and type-specific
operations. Extensions can access configuration through the global API.

.. code-block:: python

    # Extension functions respect global configuration
    from axisfuzzy.core import fuzzynum
    import axisfuzzy.config as config
    
    # Configure precision for extension operations
    config.set_config(DEFAULT_PRECISION=8)
    
    # Extensions use configuration for consistent behavior
    fnum1 = fuzzynum(md=0.8, nmd=0.2, q=2)
    fnum2 = fuzzynum(md=0.7, nmd=0.3, q=2)
    
    # Distance calculations respect precision settings
    distance = fnum1.distance(fnum2, method='euclidean')

Mixin Operations Coordination
+++++++++++++++++++++++++++++

Mixin operations leverage configuration parameters for array manipulations
and structural transformations, ensuring consistent behavior across
different fuzzy number types.

.. code-block:: python

    from axisfuzzy.core import fuzzyarray
    import numpy as np

    # Configuration affects array operations
    config.set_config(DEFAULT_MTYPE='qrofn', DEFAULT_Q=3)

    data = np.array([[[0.2, 0.3], [0.4, 0.5]], [[0.1, 0.2], [0.3, 0.4]]], dtype=object)
    farr = fuzzyarray(data=data)
    reshaped = farr.reshape((4, 1))  # Respects type configuration

Third-Party Integration Patterns
++++++++++++++++++++++++++++++++

The configuration system provides standardized interfaces for third-party
libraries and external tools, enabling seamless integration while
maintaining configuration consistency.

.. code-block:: python

    # Configuration export for external tools
    from dataclasses import asdict
    
    manager = config.get_config_manager()
    config_obj = manager.get_config()
    config_dict = asdict(config_obj)
    
    # External libraries can access AxisFuzzy configuration
    external_tool_config = {
        'precision': config_dict['DEFAULT_PRECISION'],
        'epsilon': config_dict['DEFAULT_EPSILON']
    }

Extension and Customization Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system is designed for extensibility, allowing developers
to add custom configuration parameters and validation logic. This section
provides guidelines for extending the configuration framework while
maintaining compatibility and consistency.

Adding Custom Configuration Parameters
++++++++++++++++++++++++++++++++++++++

Custom parameters can be added by extending the ``Config`` dataclass
with proper metadata and validation functions.

.. code-block:: python

    from dataclasses import dataclass, field
    from axisfuzzy.config.config_file import Config
    
    @dataclass
    class ExtendedConfig(Config):
        CUSTOM_THRESHOLD: float = field(
            default=0.5,
            metadata={
                'category': 'custom',
                'description': 'Custom threshold for specialized operations',
                'validator': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
                'error_msg': "Must be a number between 0 and 1."
            }
        )

Custom Validation Logic
+++++++++++++++++++++++

Complex validation scenarios can be implemented through custom validator
functions that integrate with the configuration system's validation framework.

.. code-block:: python

    def validate_performance_config(precision, cache_size):
        """Custom validator for performance-related parameters."""
        if precision > 10 and cache_size > 512:
            raise ValueError("High precision with large cache may cause memory issues")
        return True

Configuration Profiles and Presets
++++++++++++++++++++++++++++++++++

Developers can create configuration profiles for different use cases,
providing convenient presets for common scenarios.

.. code-block:: python

    # Performance-optimized profile
    PERFORMANCE_PROFILE = {
        'DEFAULT_PRECISION': 4,
        'CACHE_SIZE': 1024,
        'TNORM_VERIFY': False,
        'DEFAULT_EPSILON': 1e-8
    }
    
    # Scientific computing profile
    SCIENTIFIC_PROFILE = {
        'DEFAULT_PRECISION': 12,
        'CACHE_SIZE': 256,
        'TNORM_VERIFY': True,
        'DEFAULT_EPSILON': 1e-15
    }
    
    # Apply profile
    config.set_config(**PERFORMANCE_PROFILE)

Conclusion
----------

The `AxisFuzzy` configuration system provides a comprehensive framework for
managing library behavior across all computational scenarios. Through its
integration with core data structures, performance optimization mechanisms,
and extensible architecture, it enables both novice users and advanced
researchers to tailor the library's behavior to their specific requirements.

Key benefits of the configuration system include:

- **Centralized Control**: All library behavior is controlled through a
  single, consistent interface
- **Performance Optimization**: Fine-grained control over computational
  trade-offs between accuracy and speed
- **Ecosystem Integration**: Seamless coordination with extensions, mixins,
  and external tools
- **Extensibility**: Clear patterns for adding custom configuration
  parameters and validation logic

By leveraging these capabilities, users can optimize `AxisFuzzy` for their
specific use cases while maintaining the reliability and consistency
that makes it a premier tool for fuzzy set theory research and application
development.