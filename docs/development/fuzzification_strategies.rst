=======================================
Fuzzification Strategies Development
=======================================

This development guide provides comprehensive instructions for creating and integrating custom fuzzification 
strategies within the AxisFuzzy framework. The guide demonstrates the complete implementation process using 
Q-Rung Orthopair Fuzzy Numbers (QROFN) as a reference example, covering strategy implementation, registration 
mechanisms, and integration workflows.

The Fuzzification Strategy Framework
------------------------------------

Introduction
~~~~~~~~~~~~

Fuzzification strategies in AxisFuzzy serve as the computational engines that transform crisp numerical 
data into fuzzy representations. These strategies operate within a sophisticated architectural framework 
that separates concerns between user interface, strategy selection, and computational execution. The 
framework enables developers to create custom fuzzification algorithms that seamlessly integrate with 
AxisFuzzy's existing infrastructure while maintaining optimal performance and extensibility.

The strategy-based architecture ensures that new fuzzification methods can be added without modifying 
existing code, following the Open-Closed Principle. Each strategy encapsulates specific mathematical 
algorithms and computational optimizations tailored to particular fuzzy number types or application domains.

Core Architectural Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fuzzification framework consists of three fundamental components that work together to provide a 
flexible and extensible system:

**The FuzzificationStrategy Abstract Base Class**

The :class:`~axisfuzzy.fuzzifier.strategy.FuzzificationStrategy` serves as the foundational interface 
that all custom strategies must implement. This abstract base class defines the essential contract:

.. code-block:: python
   
   class FuzzificationStrategy(ABC):
       """Abstract base class for all fuzzification strategies."""
       
       @abstractmethod
       def fuzzify(self, 
                   x: Union[float, int, list, np.ndarray],
                   mf_cls: Type[MembershipFunction],
                   mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
           """Transform crisp input into fuzzy representation."""
           pass

The abstract ``fuzzify`` method enforces a consistent interface across all strategy implementations, 
ensuring that input data (scalars, lists, or arrays) can be processed uniformly regardless of the 
underlying computational approach.

**The Fuzzifier Class**

The :class:`~axisfuzzy.fuzzifier.Fuzzifier` class acts as the primary user-facing entry point that 
orchestrates the entire fuzzification process. It provides a high-level interface that abstracts the 
complexity of strategy selection and execution:

.. code-block:: python

   # Example usage demonstrating the Fuzzifier's role
   from axisfuzzy.fuzzifier import Fuzzifier
   from axisfuzzy.membership import TriangularMF
   
   # Fuzzifier automatically selects appropriate strategy
   fuzzifier = Fuzzifier(
       mf=TriangularMF,
       mtype='qrofn',
       method='default',
       mf_params={'a': 0.2, 'b': 0.5, 'c': 0.8}
   )
   
   # Strategy execution is transparent to the user
   result = fuzzifier([0.3, 0.6, 0.9])

The Fuzzifier handles membership function instantiation, parameter validation, and strategy delegation, 
providing a clean separation between user interface and computational implementation.

**The Strategy Registry**

The :class:`~axisfuzzy.fuzzifier.registry.FuzzificationStrategyRegistry` implements a centralized 
discovery and management system for available strategies. The registry uses the ``@register_fuzzifier`` 
decorator to automatically detect and register new strategies:

.. code-block:: python

   from axisfuzzy.fuzzifier.registry import register_fuzzifier
   
   @register_fuzzifier
   class CustomFuzzificationStrategy(FuzzificationStrategy):
       """Custom strategy automatically registered upon definition."""

       mtype = "custion_fuzzy_type"
       method = "custom"
       
       def fuzzify(self, x, mf_cls, mf_params_list):
           # Implementation details
           pass

The registry maintains mappings between fuzzy number types (``mtype``) and available methods, enabling 
dynamic strategy selection based on user requirements and system configuration.

Execution Flow
~~~~~~~~~~~~~~

The fuzzification process follows a well-defined execution flow that ensures consistent behavior across 
different strategies and input types:

**1. Strategy Resolution Phase**

When a Fuzzifier is instantiated, it queries the registry to resolve the appropriate strategy based on 
the specified ``mtype`` and ``method`` parameters. If no method is specified, the registry provides 
the default strategy for the given fuzzy number type.

**2. Membership Function Preparation**

The Fuzzifier processes the provided membership function specification (class, instance, or string 
identifier) and standardizes the parameter format into a list of dictionaries. This normalization 
ensures consistent input to the strategy's ``fuzzify`` method.

**3. Strategy Instantiation and Execution**

The resolved strategy class is instantiated with any additional parameters, and its ``fuzzify`` method 
is invoked with the prepared inputs. The strategy handles the computational details of transforming 
crisp data into fuzzy representations.

**4. Result Generation**

Based on the input type and dimensionality, the strategy returns either a :class:`~axisfuzzy.core.Fuzznum` 
for scalar inputs or a :class:`~axisfuzzy.core.Fuzzarray` for array-like inputs. The result maintains 
full compatibility with AxisFuzzy's operation framework.

This architectural design provides several key advantages: strategy implementations remain focused on 
computational logic without concerning themselves with user interface details, new strategies can be 
added through simple inheritance and decoration, and the system maintains backward compatibility while 
supporting extensibility.

The following sections will demonstrate how to implement custom strategies within this framework, using 
the QROFN fuzzification strategy as a concrete example that illustrates best practices and implementation 
patterns.


Implementing a Custom Fuzzification Strategy
--------------------------------------------

This section provides a comprehensive guide for implementing custom fuzzification strategies within the 
AxisFuzzy framework. We will use the QROFN (q-Rung Orthopair Fuzzy Number) strategy as our primary example, 
demonstrating each implementation step with practical code examples.

Step 1: Subclassing FuzzificationStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The foundation of any custom fuzzification strategy is creating a subclass of the abstract 
:class:`~axisfuzzy.fuzzifier.FuzzificationStrategy` base class. This class defines the essential 
interface that all strategies must implement.

.. code-block:: python

    from axisfuzzy.fuzzifier import FuzzificationStrategy
    from typing import Optional, Dict, List, Union
    import numpy as np

    class QROFNFuzzificationStrategy(FuzzificationStrategy):
        """
        QROFN fuzzification strategy implementation.
        
        This strategy generates q-rung orthopair fuzzy numbers from crisp inputs
        using membership functions and a hesitation parameter.
        """
        pass

The inheritance relationship ensures that your custom strategy integrates seamlessly with the framework's 
architecture and provides access to the base class's configuration management and utility methods.

Step 2: Defining Strategy Identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each strategy must define unique class attributes that serve as its identifier within the registry system. 
These attributes determine how the strategy is referenced and accessed by users.

.. code-block:: python

    class QROFNFuzzificationStrategy(FuzzificationStrategy):
        """QROFN fuzzification strategy implementation."""
        
        mtype = "qrofn"      # Fuzzy number type identifier
        method = "default"   # Strategy method name

The ``mtype`` attribute specifies the type of fuzzy number this strategy produces, while ``method`` 
provides a unique name for this particular implementation approach. Together, they form a composite 
key ``(mtype, method)`` that uniquely identifies the strategy in the registry.

For fuzzy types with multiple implementation approaches, you might have strategies like:

.. code-block:: python

    # Different methods for the same fuzzy type
    mtype = "qrofn"
    method = "expert"      # Expert-based fuzzification
    
    mtype = "qrofn" 
    method = "hesitation"  # Hesitation-aware fuzzification

Step 3: Implementing the __init__ Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initialization method handles strategy-specific configuration parameters and performs necessary 
validation. It should call the parent constructor and set up any additional parameters required by 
your strategy.

.. code-block:: python

    def __init__(self, q: Optional[int] = None, pi: Optional[float] = None):
        """
        Initialize the QROFN fuzzification strategy.
        
        Parameters
        ----------
        q : int, optional
            The q parameter for q-rung orthopair fuzzy numbers.
            If None, uses the global default configuration.
        pi : float, optional
            Hesitation parameter in range [0, 1]. Default is 0.1.
            
        Raises
        ------
        ValueError
            If pi is not in the valid range [0, 1].
        """
        super().__init__(q=q)
        self.pi = pi if pi is not None else 0.1
        
        # Validate strategy-specific parameters
        if not (0 <= self.pi <= 1):
            raise ValueError("pi must be in [0,1]")

The parent constructor handles the ``q`` parameter using the global configuration system, while the 
strategy-specific ``pi`` parameter is validated and stored. This pattern ensures consistent parameter 
handling across all strategies while allowing for strategy-specific customization.

Step 4: Implementing the Core fuzzify Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``fuzzify`` method contains the core computational logic of your strategy. This method must handle 
various input types, utilize membership functions dynamically, and construct appropriate output objects.

.. code-block:: python

    def fuzzify(self,
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
        """
        Fuzzify crisp input using QROFN methodology.
        
        Parameters
        ----------
        x : Union[float, int, list, np.ndarray]
            Input data to be fuzzified
        mf_cls : type
            Membership function class to instantiate
        mf_params_list : List[Dict]
            List of parameter dictionaries for membership function instances
            
        Returns
        -------
        Union[Fuzznum, Fuzzarray]
            Single Fuzzarray for one parameter set, or stacked Fuzzarray for multiple sets
        """
        # Normalize input to numpy array for vectorized computation
        x = np.asarray(x, dtype=float)
        results = []

        # Process each membership function parameter set
        for params in mf_params_list:
            # Instantiate membership function with current parameters
            mf = mf_cls(**params)

            # Compute membership degrees using vectorized operations
            mds = np.clip(mf.compute(x), 0, 1)
            
            # Calculate non-membership degrees using QROFN formula
            nmds = np.maximum(1 - mds**self.q - self.pi**self.q, 0.0) ** (1/self.q)

            # Retrieve appropriate backend for this fuzzy type
            from axisfuzzy.core import get_registry_fuzztype
            backend_cls = get_registry_fuzztype().get_backend(self.mtype)
            backend = backend_cls.from_arrays(mds=mds, nmds=nmds, q=self.q)
            
            # Create Fuzzarray with the computed backend
            results.append(Fuzzarray(backend=backend, mtype=self.mtype, q=self.q))

        # Return single result or stacked array based on input parameters
        if len(results) == 1:
            return results[0]
        else:
            from axisfuzzy.mixin.factory import _stack_factory
            return _stack_factory(results[0], *results[1:], axis=0)

This implementation demonstrates several key patterns:

**Input Processing**: The method accepts flexible input types and normalizes them to NumPy arrays for 
efficient vectorized computation.

**Dynamic Membership Function Usage**: Membership functions are instantiated dynamically using the 
provided class and parameters, allowing the strategy to work with any compatible membership function.

**Vectorized Computation**: All mathematical operations use NumPy's vectorized functions for optimal 
performance, especially important when processing large datasets.

**Backend Integration**: The strategy retrieves the appropriate backend class for its fuzzy type and 
uses it to construct the final fuzzy objects, ensuring compatibility with the framework's type system.

**Flexible Output**: The method returns either a single :class:`~axisfuzzy.core.Fuzzarray` or a stacked 
array depending on the number of parameter sets, providing intuitive behavior for different use cases.


Registration and Integration
----------------------------

Once your custom strategy is implemented, it must be registered with the framework to become available 
for use. This section covers the registration process and verification methods.

The @register_fuzzifier Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~axisfuzzy.fuzzifier.register_fuzzifier` decorator provides the primary mechanism for 
registering custom strategies with the framework. This decorator should be applied to your strategy 
class to enable automatic registration during module import.

.. code-block:: python

    from axisfuzzy.fuzzifier import register_fuzzifier

    @register_fuzzifier(is_default=True)
    class QROFNFuzzificationStrategy(FuzzificationStrategy):
        """QROFN fuzzification strategy implementation."""
        
        mtype = "qrofn"
        method = "default"
        
        # ... rest of implementation

The decorator automatically extracts the ``mtype`` and ``method`` attributes from your class and 
registers the strategy with the global registry. This registration occurs when the module containing 
your strategy is imported, making the strategy immediately available for use.

Registration Parameters
~~~~~~~~~~~~~~~~~~~~~~~

The ``@register_fuzzifier`` decorator accepts several parameters that control the registration behavior:

**is_default** : bool, optional (default: False)
    When set to ``True``, this strategy becomes the default method for its fuzzy type. If multiple 
    strategies for the same ``mtype`` are registered with ``is_default=True``, the last one registered 
    takes precedence. This parameter is particularly important for the primary implementation of each 
    fuzzy type.

.. code-block:: python

    # Register as the default strategy for QROFN
    @register_fuzzifier(is_default=True)
    class QROFNFuzzificationStrategy(FuzzificationStrategy):
        mtype = "qrofn"
        method = "default"

    # Register an alternative strategy for QROFN
    @register_fuzzifier(is_default=False)
    class QROFNExpertFuzzificationStrategy(FuzzificationStrategy):
        mtype = "qrofn"
        method = "expert"

When users specify only the fuzzy type without a specific method, the framework automatically selects 
the default strategy, providing a seamless user experience while maintaining flexibility for advanced 
use cases.

Verifying Integration
~~~~~~~~~~~~~~~~~~~~~

After implementing and registering your strategy, it's important to verify that the integration was 
successful. The framework provides several methods for programmatically querying the registry and 
confirming strategy availability.

.. code-block:: python

    from axisfuzzy.fuzzifier import get_registry_fuzzify

    # Get the global registry instance
    registry = get_registry_fuzzify()

    # Check if your strategy is registered
    strategy_cls = registry.get_strategy("qrofn", "default")
    if strategy_cls is not None:
        print(f"Strategy successfully registered: {strategy_cls.__name__}")

    # List all available strategies for your fuzzy type
    qrofn_strategies = registry.list_strategies("qrofn")
    print(f"Available QROFN strategies: {qrofn_strategies}")

    # Verify default method assignment
    default_method = registry.get_default_method("qrofn")
    print(f"Default method for QROFN: {default_method}")

    # Get comprehensive registry information
    registry_info = registry.get_registry_info()
    print(f"Total registered strategies: {registry_info['total_strategies']}")
    print(f"Supported fuzzy types: {registry_info['supported_mtypes']}")

output::

   Strategy successfully registered: QROFNFuzzificationStrategy
   Available QROFN strategies: [('qrofn', 'default')]
   Default method for QROFN: default
   Total registered strategies: 2
   Supported fuzzy types: ['qrofn', 'qrohfn']

For end-to-end verification, you can test the complete fuzzification pipeline:

.. code-block:: python

    from axisfuzzy import Fuzzifier

    # Create a fuzzifier instance with your custom strategy
    fuzzifier = Fuzzifier(
        mf='trimf',
        mtype="qrofn", 
        method="default", 
        pi=0.2,
        mf_params={"a": 0, "b": 0.5, "c": 1}
    )

    # Test fuzzification with sample data
    x = [0.1, 0.5, 0.9]
    
    result = fuzzifier(x)
    print(f"Fuzzification result: {result}")
    print(f"Result type: {type(result)}")

This verification process ensures that your custom strategy is properly integrated and functions correctly 
within the AxisFuzzy ecosystem, providing confidence in your implementation before deployment in 
production environments.

Complete Strategy Implementation Example
----------------------------------------

This section demonstrates a complete fuzzification strategy implementation using the built-in QROFN 
(q-Rung Orthopair Fuzzy Number) strategy as a comprehensive example. The QROFN implementation showcases 
advanced features including parameter validation, vectorized computation, and flexible output handling.

Strategy Overview
~~~~~~~~~~~~~~~~~

The QROFN fuzzification strategy represents one of the most sophisticated implementations in AxisFuzzy, 
incorporating mathematical rigor with computational efficiency. This strategy transforms crisp values 
into q-rung orthopair fuzzy numbers, characterized by membership and non-membership degrees that satisfy 
the constraint: :math:`\mu^q + \nu^q ≤ 1`, where :math:`q ≥ 1` is the rung parameter.

Key design features of the QROFN strategy include:

- **Mathematical Foundation**: Implements q-rung orthopair fuzzy logic with configurable rung parameter
- **Vectorized Operations**: Leverages NumPy for efficient batch processing
- **Flexible Output**: Returns single Fuzzarray or stacked arrays based on parameter configuration
- **Parameter Validation**: Ensures mathematical constraints and input validity

Complete Implementation
~~~~~~~~~~~~~~~~~~~~~~~

The following code presents the complete QROFN fuzzification strategy implementation:

.. code-block:: python

    @register_fuzzifier(is_default=True)
    class QROFNFuzzificationStrategy(FuzzificationStrategy):
        """
        QROFN Fuzzification Strategy Implementation
        
        Transforms crisp values into q-rung orthopair fuzzy numbers with:
        - Single parameter set → Returns Fuzzarray
        - Multiple parameter sets → Returns stacked Fuzzarray
        """

        # Strategy identification
        mtype = "qrofn"
        method = "default"

        def __init__(self, q: Optional[int] = None, pi: Optional[float] = None):
            """
            Initialize QROFN strategy with mathematical parameters.
            
            Parameters:
            - q: Rung parameter (inherited from base class)
            - pi: Hesitation parameter for non-membership calculation
            """
            super().__init__(q=q)
            self.pi = pi if pi is not None else 0.1
            
            # Validate hesitation parameter constraints
            if not (0 <= self.pi <= 1):
                raise ValueError("pi must be in [0,1]")

        def fuzzify(self,
                    x: Union[float, int, list, np.ndarray],
                    mf_cls: type,
                    mf_params_list: List[Dict]) -> Fuzzarray:
            """
            Core fuzzification logic with vectorized computation.
            
            Mathematical Process:
            1. Compute membership degrees using provided membership function
            2. Calculate non-membership degrees: ν = max(1 - μ^q - π^q, 0)^(1/q)
            3. Create backend representation and wrap in Fuzzarray
            4. Handle single/multiple parameter sets appropriately
            """
            
            # Ensure input is numpy array for vectorized operations
            x = np.asarray(x, dtype=float)
            results = []

            # Process each parameter set independently
            for params in mf_params_list:
                # Instantiate membership function with current parameters
                mf = mf_cls(**params)

                # Vectorized membership degree computation
                mds = np.clip(mf.compute(x), 0, 1)
                
                # Calculate non-membership degrees using q-rung constraint
                # ν = max(1 - μ^q - π^q, 0)^(1/q)
                nmds = np.maximum(1 - mds**self.q - self.pi**self.q, 0.0) ** (1/self.q)

                # Create backend representation for current fuzzy type
                backend_cls = get_registry_fuzztype().get_backend(self.mtype)
                backend = backend_cls.from_arrays(mds=mds, nmds=nmds, q=self.q)
                
                # Wrap backend in Fuzzarray with type metadata
                results.append(Fuzzarray(backend=backend, mtype=self.mtype, q=self.q))

            # Return handling based on parameter count
            if len(results) == 1:
                # Single parameter set: return individual Fuzzarray
                return results[0]
            else:
                # Multiple parameter sets: stack along new axis
                from ...mixin.factory import _stack_factory
                return _stack_factory(results[0], *results[1:], axis=0)

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

**Mathematical Rigor**: The implementation strictly adheres to q-rung orthopair fuzzy logic constraints, 
ensuring that membership and non-membership degrees satisfy the fundamental mathematical relationship.

**Computational Efficiency**: Vectorized operations using NumPy eliminate explicit loops, providing 
significant performance improvements for large datasets while maintaining numerical stability.

**Flexible Architecture**: The strategy seamlessly handles both single and multiple parameter scenarios, 
automatically determining the appropriate return type based on input configuration.

**Error Handling**: Comprehensive parameter validation prevents mathematical inconsistencies and provides 
clear error messages for debugging purposes.

**Integration Patterns**: The use of decorators, type metadata, and factory patterns demonstrates best 
practices for extending the AxisFuzzy framework while maintaining backward compatibility.

This implementation serves as a template for developing sophisticated fuzzification strategies that 
require advanced mathematical computations, efficient processing, and flexible output handling within 
the AxisFuzzy ecosystem.

Conclusion
----------

The AxisFuzzy fuzzification strategy framework provides a robust and extensible foundation for implementing 
custom fuzzy transformation algorithms. Through the systematic approach demonstrated in this guide, developers 
can create sophisticated strategies that seamlessly integrate with the framework's architecture while 
maintaining optimal performance and mathematical rigor.

The key principles established throughout this development guide ensure that custom implementations:

- **Maintain Architectural Consistency**: By adhering to the abstract base class interface and registration 
  patterns, strategies integrate naturally with existing framework components.

- **Optimize Computational Performance**: Vectorized operations and efficient backend utilization enable 
  processing of large datasets with minimal computational overhead.

- **Preserve Mathematical Integrity**: Proper parameter validation and constraint enforcement ensure that 
  fuzzy number properties remain mathematically sound across all operations.

- **Support Framework Evolution**: The modular design allows strategies to evolve independently while 
  maintaining backward compatibility with existing user code.

The QROFN implementation example demonstrates how complex mathematical concepts can be translated into 
efficient, maintainable code that serves both research and production environments. As the AxisFuzzy 
ecosystem continues to expand, this framework provides the foundation for incorporating emerging fuzzy 
logic methodologies and computational innovations.

For developers seeking to extend AxisFuzzy's capabilities, this guide establishes the essential patterns 
and best practices necessary for creating high-quality, production-ready fuzzification strategies that 
contribute meaningfully to the broader fuzzy computing community.