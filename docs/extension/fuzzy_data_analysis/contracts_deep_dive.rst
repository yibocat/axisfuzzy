.. _contracts_deep_dive:

====================================
Contracts Deep Dive and Architecture
====================================

Data contracts form the backbone of the `axisfuzzy.analysis` system, providing a formal specification 
framework for defining data structure, type constraints, and validation rules throughout the analysis 
pipeline. By introducing data contracts, the system achieves **compile-time validation** rather than 
runtime error detection, dramatically improving the robustness and reliability of fuzzy analysis workflows.

This comprehensive guide explores the contract system's architecture, implementation details, and 
practical applications within the `axisfuzzy.analysis` ecosystem.

.. contents::
   :local:

Introduction and Overview
-------------------------

What are Data Contracts?
~~~~~~~~~~~~~~~~~~~~~~~~

Data contracts are formal specifications that define the **structure**, **type**, and **constraints** 
of data flowing through analysis pipelines. Unlike traditional type hints that provide static analysis 
benefits, data contracts in `axisfuzzy.analysis` are **runtime-enforceable** specifications that can 
validate complex data structures and business logic constraints.

A data contract encapsulates three fundamental aspects:

- **Identity**: A unique name for referencing and registry management
- **Validation Logic**: Callable functions that determine data compliance
- **Inheritance Hierarchy**: Parent-child relationships enabling polymorphic compatibility

Consider this conceptual example:

.. code-block:: python

   import numpy as np
   from axisfuzzy.analysis.contracts import contract, Contract
   
   # Get the weight vector contract
   ContractWeightVector = Contract.get('ContractWeightVector')
   
   # Traditional approach - no validation
   def analyze_weights(weights):
       return weights.sum()  # May fail if weights is not numeric
   
   # Contract-driven approach - validated input
   @contract
   def analyze_weights(weights: ContractWeightVector) -> float:
       return weights.sum()  # Guaranteed to work with valid weight vectors

The Role of Contracts in axisfuzzy.analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within the `axisfuzzy.analysis` system, data contracts serve multiple critical functions:

**Pipeline Validation**: During pipeline construction, contracts ensure that component connections 
are semantically valid. When connecting two analysis components, the system verifies that the 
output contract of the upstream component is **compatible** with the input contract of the 
downstream component.

**Type Safety**: Contracts provide stronger guarantees than Python's native type system by 
validating not just types but also data structure constraints, value ranges, and business rules.

**Documentation**: Contract names and hierarchies serve as living documentation, making pipeline 
behavior explicit and self-documenting.

**Extensibility**: The contract system enables seamless integration of custom data types and 
validation rules without modifying core system components.

Contract System Architecture Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contract system is built on three foundational components:

1. **Contract Class** (:class:`~axisfuzzy.analysis.contracts.base.Contract`)
   
   The core abstraction representing a single data contract with validation logic and metadata.

2. **Contract Decorator** (:func:`~axisfuzzy.analysis.contracts.decorator.contract`)
   
   A Python decorator that integrates contracts with function type hints, enabling seamless 
   contract-driven development.

3. **Built-in Contract Library** (:mod:`~axisfuzzy.analysis.build_in`)
   
   A comprehensive collection of pre-defined contracts for common fuzzy analysis data types.

The architecture follows a **registry pattern** where all contracts are globally registered and 
accessible through a centralized lookup mechanism. This design enables:

- **Decoupled Development**: Components can reference contracts by name without direct dependencies
- **Dynamic Discovery**: New contracts can be registered at runtime
- **Consistent Validation**: All validation logic follows the same interface contract

Core Contract Structure
^^^^^^^^^^^^^^^^^^^^^^^

The fundamental structure of a Contract object can be conceptualized as follows:

.. code-block:: python

   class Contract:
       """
       A unified data contract object combining semantic naming, 
       runtime validation, and inheritance relationships.
       """
       
       def __init__(self, name: str, validator: Callable, parent: Optional[Contract] = None):
           self.name = name                    # Unique contract identifier
           self.validator = validator          # Runtime validation function
           self.parent = parent                # Inheritance relationship
           self._registry[name] = self         # Global registration
       
       def validate(self, obj: Any) -> bool:
           """Execute runtime validation for this contract."""
           return self.validator(obj)
       
       def is_compatible_with(self, required_contract: Contract) -> bool:
           """Check compatibility through inheritance chain."""
           # Implementation checks self and parent hierarchy
           pass
       
       @classmethod
       def get(cls, name: Union[str, Contract]) -> Contract:
           """Retrieve contract from global registry."""
           return cls._registry[name]

This structure provides the foundation for type-safe data validation and seamless integration 
with Python's type annotation system.

Benefits and Design Philosophy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contract system embodies several key design principles:

**Fail Fast**: By validating data contracts during pipeline construction rather than execution, 
errors are caught early in the development cycle, reducing debugging time and improving developer 
productivity.

**Explicit Contracts**: Rather than relying on implicit assumptions about data structure, the 
system makes all data requirements explicit through named contracts.

**Compositional Design**: Contracts can be composed and inherited, enabling the creation of 
specialized data types that maintain compatibility with more general contracts.

**Zero-Runtime Overhead**: Contract validation occurs during pipeline construction, not during 
data processing, ensuring that production performance is unaffected by the validation framework.

The contract system transforms fuzzy analysis development from an error-prone, trial-and-error 
process into a **contract-driven development** methodology where data requirements are explicit, 
validated, and enforced throughout the entire analysis lifecycle.


The Contract Class: Foundation of the System
--------------------------------------------

The :class:`~axisfuzzy.analysis.contracts.base.Contract` class serves as the fundamental building 
block of the entire contract system. Every data contract in `axisfuzzy.analysis` is an instance 
of this class, encapsulating validation logic, metadata, and inheritance relationships in a 
unified, extensible framework.

Core Components: name, validator, parent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each :class:`Contract` instance is defined by three essential attributes that collectively provide 
the contract's identity, behavior, and relationships:

**name (str): Unique Contract Identity**

The `name` attribute serves as the contract's unique identifier within the global registry system. 
This string-based identifier enables:

- **Global Registration**: Contracts are registered in a centralized registry using their names as keys
- **Reference Resolution**: Components can reference contracts by name using :meth:`Contract.get('ContractName')`
- **Debugging Support**: Error messages include contract names for clear problem identification
- **Collision Detection**: The system prevents duplicate contract names during registration

.. code-block:: python

   # Contract registration and retrieval
   weight_contract = Contract.get('ContractWeightVector')
   crisp_contract = Contract.get('ContractCrispTable')

**validator (Callable[[Any], bool]): Validation Logic**

The `validator` is a callable that implements the contract's validation rules. This function:

- **Encapsulates Business Logic**: Defines what constitutes valid data for the contract
- **Provides Runtime Safety**: Can be invoked for runtime validation when needed
- **Enables Complex Validation**: Supports arbitrary validation logic beyond simple type checking

.. code-block:: python

   # Example validator for weight vectors
   def validate_weight_vector(obj):
       return (isinstance(obj, np.ndarray) and 
               obj.ndim == 1 and 
               np.all(obj >= 0) and 
               len(obj) > 0)

**parent (Optional[Contract]): Inheritance Hierarchy**

The `parent` attribute establishes inheritance relationships between contracts, enabling:

- **Polymorphic Compatibility**: Derived contracts are compatible with their parent contracts
- **Semantic Relationships**: Expresses "is-a" relationships between data types
- **Flexible Pipeline Design**: Components accepting general contracts can process specialized data

Contract Registry and Global Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contract system employs a **singleton registry pattern** for global contract management. 
This centralized approach provides:

**Centralized Access**: All contracts are accessible through the :meth:`Contract.get()` class method, 
eliminating the need for direct imports and enabling loose coupling between components.

**Automatic Registration**: Contracts are automatically registered upon instantiation, ensuring 
immediate availability throughout the system.

**Name Collision Prevention**: The registry enforces unique naming, preventing conflicts that 
could lead to ambiguous contract resolution.

.. code-block:: python

   # Global contract access
   try:
       contract = Contract.get('ContractCustomType')
   except KeyError:
       # Contract not found in registry
       pass

Inheritance and Compatibility Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contract inheritance system implements a **structural subtyping** model where compatibility 
is determined by the inheritance chain rather than nominal typing:

**Compatibility Rules**: A contract `A` is compatible with contract `B` if:

1. `A` and `B` are the same contract instance
2. `B` appears in `A`'s parent chain (A → parent → parent's parent → ... → B)

**Transitive Inheritance**: Compatibility relationships are transitive, allowing deep inheritance 
hierarchies while maintaining type safety.

.. code-block:: python
   :emphasize-lines: 7

   # Inheritance hierarchy example
   base_contract = Contract('ContractWeightVector', 
                            validate_weights)

   normalized_contract = Contract('ContractNormalizedWeights', 
                                  validate_normalized, 
                                  parent=base_contract)
   
   # Compatibility check
   assert normalized_contract.is_compatible_with(base_contract)  # True

Key Methods: ``validate()`` and ``is_compatible_with()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ``validate(obj: Any) -> bool``

Executes the contract's validation logic against a data object:

.. code-block:: python

   contract = Contract.get('ContractWeightVector')
   weights = np.array([0.3, 0.5, 0.2])
   
   if contract.validate(weights):
       # Proceed with validated data...

2. ``is_compatible_with(required_contract: Contract) -> bool``

Determines compatibility between contracts for pipeline validation:

.. code-block:: python

   provided = Contract.get('ContractNormalizedWeights')
   required = Contract.get('ContractWeightVector')
   
   if provided.is_compatible_with(required):
       # Safe to connect components
       connect_pipeline_components(upstream, downstream)

Creating Custom Contracts
~~~~~~~~~~~~~~~~~~~~~~~~~

Custom contracts can be created by instantiating the :class:`Contract` class with appropriate 
parameters:

.. code-block:: python

   # 1. Write the validation function first
   def validate_correlation_matrix(obj):
       """Validate correlation matrix properties."""
       return (isinstance(obj, np.ndarray) and 
               obj.ndim == 2 and 
               obj.shape[0] == obj.shape[1] and
               np.allclose(obj, obj.T) and  # Symmetric
               np.allclose(np.diag(obj), 1.0))  # Unit diagonal
   
   # 2. Create custom contract
   correlation_contract = Contract(
       name='ContractCorrelationMatrix',
       validator=validate_correlation_matrix,
       parent=Contract.get('ContractCrispTable')  # Inherits from base table
   )

This approach enables domain-specific contract creation while maintaining compatibility with 
the existing contract ecosystem.

The ``@contract`` Decorator: Type Hints Integration
---------------------------------------------------

The :func:`~axisfuzzy.analysis.contracts.decorator.contract` decorator bridges the gap between 
Python's type hint system and the contract validation framework. This decorator automatically 
infers data contracts from function type annotations and attaches contract metadata to functions, 
enabling seamless integration with the pipeline system.

Decorator Mechanism and Type Hint Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@contract`` decorator operates through **introspection** of function type annotations, 
automatically mapping type hints to corresponding contract names:

**Type Annotation Analysis**: The decorator uses Python's :mod:`typing` module to extract 
type information from function signatures, converting type objects into contract identifiers.

**Contract Resolution**: Type hints are resolved to contract names using a mapping system that 
associates Python types with their corresponding contract implementations.

**Metadata Attachment**: The decorator attaches contract information as function attributes, 
making them accessible to the pipeline construction system.

.. code-block:: python

   import numpy as np
   from axisfuzzy.analysis.contracts import contract
   
   @contract
   def normalize_weights(weights: np.ndarray) -> np.ndarray:
       """Normalize weight vector to sum to 1.0."""
       return weights / weights.sum()
   
   # Decorator automatically infers:
   # - input_contract: 'ContractWeightVector' (from np.ndarray annotation)
   # - output_contract: 'ContractWeightVector' (from np.ndarray annotation)

Single Input/Output vs Multi Input/Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decorator supports both simple and complex function signatures:

**Single Input/Output Functions**:

Functions with single parameters and return values are directly mapped to their corresponding contracts:

.. code-block:: python

   @contract
   def calculate_entropy(probabilities: np.ndarray) -> float:
       """Calculate Shannon entropy of probability distribution."""
       return -np.sum(probabilities * np.log2(probabilities + 1e-10))
   
   # Inferred contracts:
   # - input_contract: 'ContractWeightVector'
   # - output_contract: None (primitive types not contracted)

**Multi Input/Output Functions**:

Functions with multiple parameters or return values use type annotations directly:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from typing import Dict
   from axisfuzzy.analysis.dataframe import FuzzyDataFrame
   from axisfuzzy.analysis.contracts import contract
   from axisfuzzy.analysis.build_in import (
       ContractCrispTable, 
       ContractWeightVector, 
       ContractFuzzyTable,
       ContractScoreVector
   )
   
   # Multi-input, single output
   @contract
   def fuzzify_table(crisp_data: ContractCrispTable, 
                    weights: ContractWeightVector) -> ContractFuzzyTable:
       """Convert crisp table to fuzzy representation using weights."""
       # Implementation details...
       pass
   
   # Single input, multi-output using Dict return type
   @contract
   def analyze_data(data: ContractCrispTable) -> Dict[str, ContractScoreVector]:
       """Analyze data and return multiple score vectors."""
       # Implementation details...
       # Return dictionary with named outputs
       return {
           'primary_scores': primary_vector,
           'secondary_scores': secondary_vector
       }

Function Metadata Attachment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decorator attaches several metadata attributes to decorated functions:

**Contract Metadata**:

- `_contract_inputs`: Dictionary mapping input parameter names to contract names
- `_contract_outputs`: Dictionary mapping output names to contract names  
- `_is_contract_method`: Boolean flag indicating contract decoration

**Pipeline Integration**:

These metadata attributes enable the pipeline system to:

- **Validate Connections**: Check contract compatibility during pipeline construction
- **Generate Documentation**: Automatically document data flow requirements
- **Enable Type Checking**: Provide static analysis capabilities for pipeline validation

.. code-block:: python

   @contract
   def process_data(data: ContractCrispTable) -> ContractFuzzyTable:
       """Process crisp data into fuzzy representation."""
       pass
   
   # Accessing metadata
   print(process_data._contract_inputs)    # {'data': 'ContractCrispTable'}
   print(process_data._contract_outputs)   # {'output': 'ContractFuzzyTable'}
   print(process_data._is_contract_method) # True

Integration with Python Type System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contract decorator maintains full compatibility with Python's type system while extending 
it with runtime validation capabilities:

**Type Hint Preservation**: Original type hints are preserved, ensuring compatibility with 
static type checkers like mypy and IDE type inference.

**Runtime Contract Mapping**: Type hints are mapped to contracts at decoration time, not 
runtime, ensuring zero performance overhead during function execution.

**Generic Type Support**: The decorator handles generic types and complex type annotations:

.. code-block:: python

   from typing import List, Optional, Union
   
   @contract
   def aggregate_tables(tables: List[pd.DataFrame], 
                        weights: Optional[np.ndarray] = None) -> pd.DataFrame:
       """Aggregate multiple crisp tables with optional weighting."""
       # Implementation handles optional parameters and list types
       pass

Advanced Usage Patterns
~~~~~~~~~~~~~~~~~~~~~~~

**Explicit Contract Override**:

When working with custom data types, contracts are specified through type annotations:

.. code-block:: python

   from axisfuzzy.analysis.contracts import Contract
   
   # Define custom contracts
   ContractCustomDataType = Contract.get('ContractCustomDataType')
   ContractProcessedData = Contract.get('ContractProcessedData')
   
   @contract
   def custom_processor(data: ContractCustomDataType) -> ContractProcessedData:
       """Process custom data type with explicit contracts."""
       pass

**Conditional Contract Application**:

Contracts can be conditionally applied based on runtime parameters:

.. code-block:: python

   @contract
   def adaptive_processor(data: Union[pd.DataFrame, FuzzyDataFrame]) -> FuzzyDataFrame:
       """Process either crisp or fuzzy input to fuzzy output."""
       if isinstance(data, pd.DataFrame):
           # Handle crisp input...
           pass
       else:
           # Handle fuzzy input...
           pass

**Contract Inheritance in Decorators**:

The decorator respects contract inheritance relationships, enabling polymorphic function design:

.. code-block:: python

   @contract
   def general_processor(weights: np.ndarray) -> np.ndarray:
       """Process any weight vector type."""
       pass
   
   # Can accept ContractNormalizedWeights, ContractScoreVector, etc.
   # as long as they inherit from ContractWeightVector

The `@contract` decorator transforms ordinary Python functions into contract-aware components 
that integrate seamlessly with the `axisfuzzy.analysis` pipeline system, providing both 
development-time clarity and runtime safety.

Built-in Contract Library
-------------------------

The `axisfuzzy.analysis` system provides a comprehensive library of pre-defined contracts 
covering common data types used in fuzzy analysis workflows. These built-in contracts are 
defined in :mod:`~axisfuzzy.analysis.build_in` and form the foundation for most analysis 
pipelines.

Base Contracts (ContractAny, ContractCrispTable, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ContractAny**

The most permissive contract that accepts any data type:

.. code-block:: python

   ContractAny = Contract('Any', lambda obj: True)

This contract serves as the root of the inheritance hierarchy and is useful for components 
that can process arbitrary data types.

**ContractCrispTable**

Validates pandas DataFrames containing crisp (non-fuzzy) numerical data:

.. code-block:: python

   def _validate_crisp_table(obj: Any) -> bool:
       """Validate crisp table with numerical data."""
       return (_is_pandas_df(obj) and 
               obj.select_dtypes(include=[np.number]).shape[1] > 0)

This contract ensures that input data is a pandas DataFrame with at least one numerical column, 
making it suitable for quantitative analysis operations.

**ContractFuzzyTable**

Validates :class:`~axisfuzzy.analysis.dataframe.FuzzyDataFrame` instances:

.. code-block:: python

   ContractFuzzyTable = Contract(
       'ContractFuzzyTable',
       lambda obj: isinstance(obj, FuzzyDataFrame)
   )

This contract is essential for operations that require fuzzy data structures with membership 
functions and uncertainty representations.

**ContractWeightVector**

Validates one-dimensional numerical arrays representing weight or score vectors:

.. code-block:: python

   ContractWeightVector = Contract(
       'ContractWeightVector',
       lambda obj: ((_is_numpy_array(obj) and obj.ndim == 1) or 
                   _is_pandas_series(obj))
   )

Accepts both NumPy arrays and pandas Series, providing flexibility in data representation 
while ensuring dimensional consistency.

**ContractMatrix**

Validates two-dimensional numerical arrays or DataFrames:

.. code-block:: python

   ContractMatrix = Contract(
       'ContractMatrix',
       lambda obj: ((_is_numpy_array(obj) and obj.ndim == 2) or 
                   _is_pandas_df(obj))
   )

This contract is fundamental for matrix operations in multi-criteria decision analysis and 
pairwise comparison methods.

Derived Contracts and Inheritance Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in library includes several specialized contracts that inherit from base contracts:

**ContractNormalizedWeights**

Extends :class:`ContractWeightVector` with normalization constraints:

.. code-block:: python

   def _validate_normalized_weights(obj: Any) -> bool:
       """Validate normalized weight vector (sums to 1.0)."""
       return (ContractWeightVector.validate(obj) and 
               len(obj) > 0 and 
               np.isclose(np.sum(obj), 1.0))
   
   ContractNormalizedWeights = Contract(
       'ContractNormalizedWeights',
       _validate_normalized_weights,
       parent=ContractWeightVector
   )

This contract ensures that weight vectors are properly normalized for probability and 
decision-making applications.

**ContractScoreVector**

A semantic alias for :class:`ContractWeightVector` used in scoring contexts:

.. code-block:: python

   ContractScoreVector = Contract(
       'ContractScoreVector',
       ContractWeightVector.validate,
       parent=ContractWeightVector
   )

While functionally identical to its parent, this contract provides semantic clarity in 
pipeline documentation and error messages.

**ContractPairwiseMatrix**

Extends :class:`ContractMatrix` with square matrix constraints:

.. code-block:: python

   ContractPairwiseMatrix = Contract(
       'ContractPairwiseMatrix',
       lambda obj: _is_pandas_df(obj) and obj.shape[0] == obj.shape[1],
       parent=ContractMatrix
   )

Essential for pairwise comparison methods like AHP (Analytic Hierarchy Process) where 
square matrices represent comparison relationships.

Validation Rules and Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fuzzy Data Type Contracts**:

.. code-block:: python

   ContractFuzzyNumber = Contract(
       'ContractFuzzyNumber',
       lambda obj: isinstance(obj, Fuzznum)
   )
   
   ContractFuzzyArray = Contract(
       'ContractFuzzyArray',
       lambda obj: isinstance(obj, Fuzzarray)
   )

These contracts validate core fuzzy data structures from the `axisfuzzy.core` module.

**Primitive Type Contracts**:

.. code-block:: python

   ContractNumericValue = Contract(
       'ContractNumericValue',
       lambda obj: isinstance(obj, (int, float)) and not isinstance(obj, bool)
   )
   
   ContractStringList = Contract(
       'ContractStringList',
       lambda obj: isinstance(obj, list) and all(isinstance(i, str) for i in obj)
   )

These contracts handle basic data types with specific validation constraints.

**Result Type Contracts**:

.. code-block:: python

   ContractRankingResult = Contract(
       'ContractRankingResult',
       lambda obj: (_is_pandas_series(obj) or 
                   (isinstance(obj, list) and 
                    all(isinstance(i, (str, int)) for i in obj)))
   )
   
   ContractThreeWayResult = Contract(
       'ContractThreeWayResult',
       lambda obj: (isinstance(obj, dict) and 
                   all(k in obj for k in ['accept', 'reject', 'defer']))
   )

These contracts validate specific output formats used in decision analysis results.

Contract Compatibility Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inheritance relationships create a compatibility matrix where derived contracts are 
compatible with their ancestors:

.. code-block:: text

   ContractAny
   ├── ContractWeightVector
   │   ├── ContractNormalizedWeights
   │   └── ContractScoreVector
   ├── ContractMatrix
   │   └── ContractPairwiseMatrix
   ├── ContractCrispTable
   ├── ContractFuzzyTable
   ├── ContractFuzzyNumber
   ├── ContractFuzzyArray
   ├── ContractNumericValue
   └── ContractStringList
       ├── ContractCriteriaList
       └── ContractAlternativeList

**Compatibility Examples**:

- :class:`ContractNormalizedWeights` is compatible with :class:`ContractWeightVector`
- :class:`ContractPairwiseMatrix` is compatible with :class:`ContractMatrix`
- All contracts are compatible with :class:`ContractAny`

Extending the Built-in Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in library can be extended by creating new contracts that inherit from existing ones:

.. code-block:: python

   # Custom contract for correlation matrices
   def validate_correlation_matrix(obj):
       """Validate correlation matrix properties."""
       return (ContractPairwiseMatrix.validate(obj) and
               np.allclose(obj, obj.T) and  # Symmetric
               np.allclose(np.diag(obj), 1.0) and  # Unit diagonal
               np.all(np.abs(obj) <= 1.0))  # Values in [-1, 1]
   
   ContractCorrelationMatrix = Contract(
       'ContractCorrelationMatrix',
       validate_correlation_matrix,
       parent=ContractPairwiseMatrix
   )

This approach maintains compatibility with existing components while adding domain-specific 
validation rules for specialized analysis requirements.


Custom Contract Development
---------------------------

Design Principles for Custom Contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When developing custom contracts, follow these core principles to ensure maintainability 
and compatibility with the AxisFuzzy analysis framework:

.. code-block:: python

   import numpy as np
   from axisfuzzy.analysis.contracts import Contract
   
   def validate_positive_matrix(obj) -> bool:
       """Validator for matrices with all positive values."""
       return (isinstance(obj, np.ndarray) and 
               obj.ndim == 2 and 
               np.all(obj > 0))
   
   ContractPositiveMatrix = Contract(
       'ContractPositiveMatrix',
       validate_positive_matrix,
       parent=ContractMatrix
   )

Validator Function Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Effective validator functions should be deterministic, efficient, and provide clear 
validation logic:

.. code-block:: python

   def validate_correlation_bounds(obj) -> bool:
       """Validates correlation matrix properties."""
       if not isinstance(obj, np.ndarray) or obj.ndim != 2:
           return False
       
       # Check symmetry and diagonal properties
       return (obj.shape[0] == obj.shape[1] and
               np.allclose(obj, obj.T) and
               np.allclose(np.diag(obj), 1.0) and
               np.all(np.abs(obj) <= 1.0))

Establishing Contract Hierarchies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contract inheritance enables flexible validation chains and compatibility checking:

.. code-block:: python

   # Base contract for decision matrices
   ContractDecisionMatrix = Contract(
       'ContractDecisionMatrix',
       lambda obj: isinstance(obj, pd.DataFrame) and obj.shape[0] > 0,
       parent=ContractCrispTable
   )
   
   # Specialized contract for normalized decision matrices
   def validate_normalized_decision(obj) -> bool:
       return (ContractDecisionMatrix.validate(obj) and
               np.all((obj >= 0) & (obj <= 1)))
   
   ContractNormalizedDecision = Contract(
       'ContractNormalizedDecision',
       validate_normalized_decision,
       parent=ContractDecisionMatrix
   )

Testing and Validation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive testing ensures contract reliability across diverse data scenarios:

.. code-block:: python

   def test_custom_contract():
       """Test suite for custom contract validation."""
       # Valid cases
       valid_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
       assert ContractCorrelationMatrix.validate(valid_matrix)
       
       # Invalid cases
       invalid_matrix = np.array([[1.0, 1.5], [0.5, 1.0]])
       assert not ContractCorrelationMatrix.validate(invalid_matrix)
       
       # Compatibility testing
       assert ContractCorrelationMatrix.is_compatible_with(ContractMatrix)

Real-world Examples
~~~~~~~~~~~~~~~~~~~

Domain-specific contracts enhance analysis pipeline robustness:

.. code-block:: python

   # Financial risk assessment contract
   def validate_risk_profile(obj) -> bool:
       required_cols = ['risk_score', 'volatility', 'return_rate']
       return (isinstance(obj, pd.DataFrame) and
               all(col in obj.columns for col in required_cols) and
               obj['risk_score'].between(0, 1).all())
   
   ContractRiskProfile = Contract(
       'ContractRiskProfile',
       validate_risk_profile,
       parent=ContractCrispTable
   )

Custom contracts integrate seamlessly with the ``@contract`` decorator, enabling 
type-safe function definitions and automatic validation within analysis pipelines.

Contract Integration in Data Flow
---------------------------------

The contract system forms the backbone of data flow validation in AxisFuzzy's 
analysis framework. This section explores how contracts integrate with pipeline 
construction, model systems, and runtime execution to ensure type safety and 
data integrity throughout the analysis workflow.

Pipeline Construction and Contract Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During pipeline construction, contracts enable **graph-time validation** that 
catches type mismatches before execution begins. The ``FuzzyPipeline`` class 
performs comprehensive contract checking when components are connected:

.. code-block:: python

    from axisfuzzy.analysis import FuzzyPipeline
    from axisfuzzy.analysis.build_in import ContractCrispTable, ContractWeightVector
    from axisfuzzy.analysis.component.basic import ToolNormalization, ToolStatistics

    # Create pipeline with input contracts
    pipeline = FuzzyPipeline("analysis_workflow")
    data_input = pipeline.input("data", contract=ContractCrispTable)
    weights_input = pipeline.input("weights", contract=ContractWeightVector)

    # Add components with automatic contract checking
    normalization = ToolNormalization(method='min_max')
    normalized_data = pipeline.add(normalization.run, data=data_input)

    # Contract compatibility is verified at graph construction time
    statistics = ToolStatistics()  # Statistical analysis component
    stats_result = pipeline.add(statistics.run, data=normalized_data)

The pipeline validates that each component's input requirements match the output 
promises of its dependencies. This validation uses the ``is_compatible_with()`` 
method to check contract inheritance hierarchies, enabling polymorphic data flow.

Model System Integration
~~~~~~~~~~~~~~~~~~~~~~~~

The ``Model`` class leverages contracts for automatic pipeline generation from 
high-level ``forward()`` method definitions. During the build process, contracts 
are extracted from type annotations and used to construct the underlying pipeline:

.. code-block:: python

   from axisfuzzy.analysis.app import Model
   from axisfuzzy.analysis.contracts import contract
   from axisfuzzy.analysis.component.basic import ToolNormalization, ToolFuzzification
   from axisfuzzy.analysis.build_in import ContractCrispTable, ContractFuzzyTable
   from axisfuzzy.fuzzifier import Fuzzifier
   
   class FuzzyAnalysisModel(Model):
       def __init__(self):
           super().__init__()
           # Create a fuzzifier for the fuzzification component
           self.fuzzifier_engine = Fuzzifier(
               mf='gaussmf',
               mtype='qrofn',
               pi=0.2,
               mf_params=[{'sigma': 0.15, 'c': 0.5}]
           )
           # Initialize analysis components
           self.normalizer = ToolNormalization(method='min_max', axis=0)
           self.fuzzifier = ToolFuzzification(fuzzifier=self.fuzzifier_engine)
       
       def get_config(self):
           """Return the model configuration for serialization."""
           return {
               'fuzzifier_config': self.fuzzifier_engine.get_config(),
               'normalization_method': 'min_max',
               'normalization_axis': 0
           }
       
       @contract
       def forward(self, data: ContractCrispTable) -> ContractFuzzyTable:
           normalized_data = self.normalizer(data)
           return self.fuzzifier(normalized_data)

The model system automatically traces component calls during symbolic execution, 
building a ``FuzzyPipeline`` that preserves all contract relationships and 
validation logic from the original method definition.

Runtime vs Compile-time Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contract system operates at two distinct phases:

**Graph-time (Compile-time) Validation:**
- Performed during pipeline construction via ``_add_step()``
- Checks contract compatibility between connected components
- Validates input/output contract mappings
- Prevents incompatible component connections

**Runtime Validation:**
- Optional validation during actual data processing
- Executed by calling ``contract.validate(data)`` on real data objects
- Provides final safety net for dynamic data scenarios
- Can be enabled/disabled for performance optimization

Error Handling and Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contract violations generate descriptive error messages that pinpoint the exact 
source of type mismatches:

.. code-block:: python

   # Example error during pipeline construction
   TypeError: Contract incompatibility for 'FuzzyAnalyzer.run' on input 'data'. 
   Expected compatible with 'ContractFuzzyTable', but received a promise for 
   'ContractCrispTable' from step 'DataLoader.run'.

The error handling system provides:
- **Component identification**: Names the specific component and input parameter
- **Contract details**: Shows expected vs. provided contract types  
- **Source tracing**: Identifies which upstream component produced the incompatible output
- **Inheritance awareness**: Considers contract parent relationships in error messages

Performance Implications
~~~~~~~~~~~~~~~~~~~~~~~~

Contract integration is designed for minimal runtime overhead:

**Graph-time Costs:**
- One-time validation during pipeline construction
- Metadata extraction from decorated functions
- Contract compatibility checking via inheritance traversal

**Runtime Costs:**
- Zero overhead when runtime validation is disabled
- Optional validation calls only when explicitly requested
- Efficient contract lookup via global registry system

The system prioritizes early detection of type errors over runtime performance, 
following the principle that catching errors during development is preferable 
to runtime failures in production environments.

Conclusion
----------

The contract system in `axisfuzzy.analysis` represents a paradigm shift from traditional 
runtime error handling to **compile-time validation** for fuzzy analysis workflows. By 
integrating formal data specifications with Python's type system, contracts provide:

**Development Benefits**: Type-safe pipeline construction with early error detection, 
comprehensive validation of data flow compatibility, and self-documenting component 
interfaces that reduce integration complexity.

**Architectural Advantages**: Modular contract inheritance enabling polymorphic data 
handling, centralized registry management for global contract access, and seamless 
integration with existing Python development tools.

**Production Reliability**: Zero-overhead runtime performance through graph-time 
validation, robust error reporting with precise component identification, and 
extensible validation framework supporting custom data types.

The contract system transforms fuzzy analysis development from an error-prone, 
trial-and-error process into a **contract-driven methodology** where data requirements 
are explicit, validated, and enforced throughout the analysis lifecycle. This foundation 
enables the construction of complex, multi-component analysis pipelines with confidence 
in their correctness and maintainability.

For developers building custom analysis components, the contract system provides the 
tools necessary to create robust, interoperable modules that integrate seamlessly 
with the broader `axisfuzzy.analysis` ecosystem. The combination of built-in contracts, 
decorator-based integration, and inheritance-aware validation creates a powerful 
framework for scientific computing applications requiring both flexibility and reliability.