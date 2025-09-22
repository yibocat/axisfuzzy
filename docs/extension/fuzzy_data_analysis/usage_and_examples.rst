===================
Usage and Examples
===================

This comprehensive guide demonstrates practical usage patterns and real-world examples 
of the AxisFuzzy analysis framework. Through hands-on examples and step-by-step tutorials, 
you'll learn how to leverage the full power of fuzzy data analysis, from basic contract 
development to sophisticated model applications.

Each section builds upon previous concepts while providing standalone examples that you 
can adapt to your specific use cases. Whether you're developing custom contracts, building 
reusable components, or creating complex analysis workflows, this guide provides the 
practical knowledge you need to succeed.

.. contents::
   :local:

Contract Development Fundamentals
---------------------------------

This section introduces the foundational concepts of contract-driven development 
within the AxisFuzzy analysis framework. You'll learn how to leverage the contract 
system to build robust, type-safe analysis components that ensure data integrity 
throughout the analysis pipeline.

**Learning Outcomes:**

- Understanding the Contract base architecture and validation mechanisms
- Creating custom contracts for domain-specific data structures  
- Utilizing contract decorators for automatic type inference
- Implementing contract inheritance and compatibility patterns
- Applying built-in contracts for common analysis scenarios

**Key Files:** ``base.py``, ``decorator.py``, ``build_in.py``

**Reference Documentation:** :doc:`contracts_deep_dive`

.. note::
   Code examples in this section correspond to practical implementations 
   found in ``examples/analysis/`` directory.

Understanding Contract Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Contract`` class serves as the fundamental building block for data validation 
in the AxisFuzzy analysis system. Each contract encapsulates three core elements:

- **Identity**: A unique name for registration and identification
- **Validation Logic**: Runtime validation functions for data compliance
- **Inheritance Hierarchy**: Parent-child relationships for polymorphic compatibility

.. code-block:: python

   from axisfuzzy.analysis.contracts import Contract
   
   # Basic contract creation
   def validate_positive_number(obj):
       return isinstance(obj, (int, float)) and obj > 0
   
   # Create a custom contract
   ContractPositiveNumber = Contract(
       name='PositiveNumber',
       validator=validate_positive_number
   )
   
   # Validate data
   result = ContractPositiveNumber.validate(5.0)  # Returns True
   invalid = ContractPositiveNumber.validate(-1)  # Returns False

Contract Registry and Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contract system maintains a global registry that prevents naming conflicts 
and enables contract lookup by name:

.. code-block:: python

   # Contracts are automatically registered upon creation
   print(Contract._registry.keys())  # Shows all registered contract names
   
   # Retrieve contracts by name
   retrieved_contract = Contract._registry['PositiveNumber']
   
   # Contract names must be unique
   try:
       duplicate = Contract('PositiveNumber', lambda x: True)
   except NameError as e:
       print(f"Error: {e}")  # Contract already registered

Custom Contract Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating domain-specific contracts involves defining validation logic that 
captures your data requirements:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from axisfuzzy.analysis.dataframe import FuzzyDataFrame
   
   # Custom validator for decision matrices
   def validate_decision_matrix(obj):
       if not isinstance(obj, pd.DataFrame):
           return False
       # Must have numeric data only
       return obj.select_dtypes(include=[np.number]).shape[1] == obj.shape[1]
   
   # Create specialized contract
   ContractDecisionMatrix = Contract(
       name='DecisionMatrix',
       validator=validate_decision_matrix
   )
   
   # Fuzzy-specific contract
   def validate_fuzzy_weights(obj):
       return (isinstance(obj, FuzzyDataFrame) and 
               obj.shape[1] == 1 and 
               obj.mtype is not None)
   
   ContractFuzzyWeights = Contract(
       name='FuzzyWeights',
       validator=validate_fuzzy_weights
   )

Contract Inheritance and Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contracts support inheritance relationships that enable polymorphic data handling:

.. code-block:: python

   # Base contract for numeric vectors
   ContractVector = Contract(
       name='Vector',
       validator=lambda obj: isinstance(obj, (list, np.ndarray, pd.Series))
   )
   
   # Specialized contract inheriting from base
   ContractNormalizedVector = Contract(
       name='NormalizedVector',
       validator=lambda obj: (ContractVector.validate(obj) and 
                             abs(np.sum(obj) - 1.0) < 1e-6),
       parent=ContractVector
   )
   
   # Check compatibility
   normalized_data = np.array([0.3, 0.4, 0.3])
   is_compatible = ContractNormalizedVector.is_compatible_with(ContractVector)
   print(f"Compatibility: {is_compatible}")  # True

Contract Decorators for Type Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@contract`` decorator automatically extracts contract information from 
function type annotations:

.. code-block:: python

   from axisfuzzy.analysis.contracts import contract
   from axisfuzzy.analysis.build_in import ContractCrispTable, ContractWeightVector, ContractScoreVector
   import numpy as np
   
   @contract
   def calculate_weighted_scores(data: ContractCrispTable, weights: ContractWeightVector) -> ContractScoreVector:
       """Calculate weighted scores for decision alternatives."""
       return np.dot(data.values, weights)
   
   # The decorator attaches contract metadata
   print(f"Input contracts: {calculate_weighted_scores._contract_inputs}")
   print(f"Output contract: {calculate_weighted_scores._contract_outputs}")

Built-in Contract Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AxisFuzzy provides a comprehensive set of built-in contracts for common analysis 
scenarios:

.. code-block:: python

   from axisfuzzy.analysis.build_in import (
       ContractCrispTable, ContractFuzzyTable, ContractWeightVector,
       ContractMatrix, ContractFuzzyNumber, ContractScoreVector
   )
   
   # Using built-in contracts for validation
   crisp_data = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
   weights = np.array([0.3, 0.4, 0.3])
   
   # Validate inputs before processing
   if (ContractCrispTable.validate(crisp_data) and 
       ContractWeightVector.validate(weights)):
       scores = np.dot(crisp_data.values, weights)
       print(f"Calculated scores: {scores}")

Contract Composition and Advanced Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex validation scenarios can be addressed through contract composition:

.. code-block:: python

   # Composite validation for multi-criteria data
   def validate_mcda_dataset(obj):
       return (ContractCrispTable.validate(obj) and 
               obj.shape[0] >= 2 and  # At least 2 alternatives
               obj.shape[1] >= 2 and  # At least 2 criteria
               not obj.isnull().any().any())  # No missing values
   
   ContractMCDADataset = Contract(
       name='MCDADataset',
       validator=validate_mcda_dataset,
       parent=ContractCrispTable
   )
   
   # Usage in analysis functions
   @contract
   def perform_mcda_analysis(dataset: ContractMCDADataset) -> 'ContractRankingResult':
       """Perform multi-criteria decision analysis."""
       # Implementation details...
       pass

Component Development Mastery
-----------------------------

.. note::
   This section covers component development patterns, including custom component 
   creation, lifecycle management, and contract integration.

**Learning Outcomes:**

* Master the AnalysisComponent architecture and design principles
* Implement custom components with proper contract integration
* Understand component lifecycle and configuration management
* Apply best practices for reusable component development

**Key Files:**

* :file:`component/base.py` - Base component architecture
* :file:`component/basic.py` - Built-in utility components

**Reference Documentation:**

* :doc:`components_and_pipeline` - Comprehensive component framework guide

Component Architecture Foundation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~axisfuzzy.analysis.component.base.AnalysisComponent` serves as the foundational 
base class for all analysis tools. This marker pattern promotes object-oriented design, 
enabling better state management and configuration serialization.

.. code-block:: python

   from axisfuzzy.analysis.component.base import AnalysisComponent
   from axisfuzzy.analysis.contracts import contract, Contract

   class CustomProcessor(AnalysisComponent):
       def __init__(self, threshold: float = 0.5):
           self.threshold = threshold
       
       def get_config(self) -> dict:
           return {'threshold': self.threshold}
       
       @contract
       def run(self, data: 'ContractCrispTable') -> 'ContractCrispTable':
           # Implementation logic here
           return processed_data

The architecture enforces two essential methods: ``run()`` for execution logic and 
``get_config()`` for configuration serialization.

Built-in Component Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides utility components for common operations. The 
:class:`~axisfuzzy.analysis.component.basic.ToolNormalization` demonstrates typical usage:

.. code-block:: python

   from axisfuzzy.analysis.component.basic import ToolNormalization
   
   # Initialize with configuration
   normalizer = ToolNormalization(method='min_max', axis=1)
   
   # Execute with contract validation
   normalized_data = normalizer.run(crisp_data)
   
   # Access configuration for serialization
   config = normalizer.get_config()  # {'method': 'min_max', 'axis': 1}

Built-in components include **ToolNormalization**, **ToolWeightNormalization**, 
**ToolStatistics**, and **ToolFuzzification**.

Component Lifecycle Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components follow a structured lifecycle pattern supporting both immediate execution 
and pipeline integration:

.. code-block:: python

   # 1. Initialization Phase
   component = ToolNormalization(method='z_score', axis=0)
   
   # 2. Configuration Access
   config = component.get_config()
   
   # 3. Execution Phase
   result = component.run(input_data)
   
   # 4. Serialization Support
   import json
   serialized_config = json.dumps(config)
   restored_component = ToolNormalization(**json.loads(serialized_config))

The ``get_config()`` method ensures components can be serialized and reconstructed, 
enabling model persistence and pipeline reproducibility.

Contract Integration and Custom Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components leverage the contract system for type safety. The ``@contract`` decorator 
automatically validates inputs and outputs:

.. code-block:: python

   from axisfuzzy.analysis.component.base import AnalysisComponent
   from axisfuzzy.analysis.contracts import contract, Contract
   import numpy as np
   
   class OutlierDetector(AnalysisComponent):
       def __init__(self, method: str = 'iqr', threshold: float = 1.5):
           self.method = method
           self.threshold = threshold
       
       def get_config(self) -> dict:
           return {'method': self.method, 'threshold': self.threshold}
       
       @contract
       def run(self, data: 'ContractCrispTable') -> 'ContractCrispTable':
           """Detect outliers using IQR method."""
           if self.method == 'iqr':
               Q1 = np.percentile(data, 25, axis=0)
               Q3 = np.percentile(data, 75, axis=0)
               IQR = Q3 - Q1
               lower_bound = Q1 - self.threshold * IQR
               upper_bound = Q3 + self.threshold * IQR
               outlier_mask = (data < lower_bound) | (data > upper_bound)
               # Process outliers as needed
           return data

Best Practices Summary
~~~~~~~~~~~~~~~~~~~~~~

Effective component development follows these principles:

* **Configuration Management**: Implement ``get_config()`` with JSON-serializable parameters
* **Contract Compliance**: Use type annotations with contract decorators for validation
* **State Encapsulation**: Store configuration in instance variables during initialization
* **Error Handling**: Implement graceful error handling with informative messages
* **Documentation**: Provide comprehensive docstrings following NumPy style conventions

Pipeline Core Architecture
--------------------------

.. note::
   **Learning Outcomes:** Master DAG-based pipeline construction, data flow orchestration, 
   and performance optimization for complex fuzzy analysis workflows.

Pipeline Design Philosophy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FuzzyPipeline`` implements a **Directed Acyclic Graph (DAG)** execution engine that 
separates workflow definition from execution. This architecture provides:

**Declarative Construction**

Pipelines use a fluent API describing *what* should happen, not *how*:

.. code-block:: python

   from axisfuzzy.analysis.pipeline import FuzzyPipeline
   from axisfuzzy.analysis.contracts import Contract
   
   pipeline = FuzzyPipeline(name="Analysis Workflow")
   
   # Declare inputs with type contracts
   raw_data = pipeline.input("raw_data", contract=Contract.get('ContractCrispTable'))
   weights = pipeline.input("weights", contract=Contract.get('ContractWeightVector'))
   
   # Define processing steps
   normalized = pipeline.add(normalizer.run, data=raw_data)
   scored = pipeline.add(scorer.run, data=normalized, weights=weights)

**Lazy Evaluation Benefits**

- **Optimization**: Engine analyzes entire graph before execution
- **Validation**: Type contracts checked before computation begins
- **Debugging**: Complete workflow can be inspected and visualized

Pipeline Construction Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linear Processing**

.. code-block:: python

   from axisfuzzy.analysis.component.basic import ToolNormalization, ToolSimpleAggregation
   
   pipeline = FuzzyPipeline(name="Linear Processing")
   data = pipeline.input("data", contract=Contract.get('ContractCrispTable'))
   
   normalizer = ToolNormalization(method='min_max')
   aggregator = ToolSimpleAggregation(operation='mean')
   
   normalized = pipeline.add(normalizer.run, data=data)
   result = pipeline.add(aggregator.run, data=normalized)

**Multi-Input Convergence**

.. code-block:: python

   pipeline = FuzzyPipeline(name="Multi-Input Analysis")
   
   criteria_data = pipeline.input("criteria", contract=Contract.get('ContractCrispTable'))
   expert_weights = pipeline.input("weights", contract=Contract.get('ContractWeightVector'))
   
   normalized_criteria = pipeline.add(normalizer.run, data=criteria_data)
   weighted_analysis = pipeline.add(
       weighted_scorer.run,
       data=normalized_criteria,
       weights=expert_weights
   )

**Parallel Processing**

.. code-block:: python

   raw_data = pipeline.input("data", contract=Contract.get('ContractCrispTable'))
   
   # Multiple parallel branches
   statistical_summary = pipeline.add(statistics.run, data=raw_data)
   fuzzy_analysis = pipeline.add(fuzzifier.run, data=raw_data)
   correlation_matrix = pipeline.add(correlator.run, data=raw_data)

Data Flow Management
~~~~~~~~~~~~~~~~~~~~

**StepOutput Objects**

Pipeline steps return ``StepOutput`` objects as symbolic references:

.. code-block:: python

   # StepOutput represents a "promise" of future data
   normalized_data = pipeline.add(normalizer.run, data=input_data)
   final_result = pipeline.add(analyzer.run, data=normalized_data)

**Contract-Based Validation**

.. code-block:: python

   from axisfuzzy.analysis.contracts import contract
   from axisfuzzy.analysis.build_in import ContractCrispTable, ContractScoreVector
   
   class TypeSafeComponent(AnalysisComponent):
       @contract
       def run(self, data: ContractCrispTable) -> ContractScoreVector:
           return self.process(data)  # Automatic validation

**Execution State Management**

.. code-block:: python

   # Step-by-step execution control
   initial_state = pipeline.start_execution(input_data)
   
   while not initial_state.is_complete():
       initial_state = initial_state.run_next()
       print(f"Completed: {initial_state.latest_step_id}")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Execution Order Optimization**

The engine automatically determines optimal execution order using topological sorting:

.. code-block:: python

   execution_order = pipeline.get_execution_order()
   print(f"Optimized order: {[step.display_name for step in execution_order]}")

**Memory-Efficient Execution**

.. code-block:: python

   # Iterative execution for large datasets
   for step_info in pipeline.step_by_step(input_data):
       print(f"Step: {step_info.step_name} - {step_info.execution_time} ms")

**Pipeline Visualization**

.. code-block:: python

   # Debug and optimize with visualization
   pipeline.visualize(
       output_path="pipeline_graph.png",
       show_contracts=True,
       highlight_critical_path=True
   )

Advanced Patterns and Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Nested Pipeline Composition**

.. code-block:: python

   from axisfuzzy.analysis.component.basic import ToolNormalization
   
   # Hierarchical pipeline composition
   preprocessing_pipeline = FuzzyPipeline(name="Preprocessing")
   raw_input = preprocessing_pipeline.input("raw_data", contract=Contract.get('ContractCrispTable'))
   
   cleaner = ToolNormalization(method='min_max')
   cleaned_data = preprocessing_pipeline.add(cleaner.run, data=raw_input)
   
   main_pipeline = FuzzyPipeline(name="Main Analysis")
   main_input = main_pipeline.input("main_data", contract=Contract.get('ContractCrispTable'))
   preprocessed = main_pipeline.add(preprocessing_pipeline, raw_data=main_input)

**Error Handling**

.. code-block:: python

   from axisfuzzy.analysis.component.base import AnalysisComponent
   from axisfuzzy.analysis.contracts import contract, Contract

   class RobustComponent(AnalysisComponent):
       @contract
       def run(self, data: 'ContractCrispTable') -> 'ContractCrispTable':
           try:
               return self.process_data(data)
           except Exception as e:
               return self.fallback_processing(data)

**Best Practices**

- **Single Responsibility**: Each step has one clear purpose
- **Type Safety**: Use contract annotations for validation
- **Immutability**: Return new objects instead of modifying inputs
- **Error Handling**: Implement graceful degradation patterns
- **Documentation**: Use descriptive names for steps and components

Model API Applications
----------------------

The Model API provides a PyTorch-inspired interface for building sophisticated 
fuzzy analysis workflows. Models serve as high-level blueprints that automatically 
generate optimized pipeline execution graphs, focusing on *what* your analysis 
should accomplish rather than *how* it should be executed.

Understanding Model Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Model Abstraction Layer**

Models automatically register analysis components and handle dependency tracking:

.. code-block:: python

    from axisfuzzy.analysis.app.model import Model
    from axisfuzzy.analysis.component.basic import (
        ToolNormalization, ToolFuzzification, ToolSimpleAggregation
    )
    
    class BasicAnalysisModel(Model):
        def __init__(self, fuzzifier):
            super().__init__()
            self.normalizer = ToolNormalization(method='min_max', axis=0)
            self.fuzzifier = ToolFuzzification(fuzzifier=fuzzifier)
            self.aggregator = ToolSimpleAggregation(operation='mean', axis=1)
        
        def get_config(self) -> dict:
            return {'fuzzifier': self.fuzzifier.fuzzifier}
        
        def forward(self, data):
            normalized_data = self.normalizer(data)
            fuzzy_data = self.fuzzifier(normalized_data)
            return self.aggregator(fuzzy_data)

Forward Method Design Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linear Processing Workflows**

Sequential data transformation through analysis components:

.. code-block:: python

    class LinearAnalysisModel(Model):
        def forward(self, input_data):
            x = self.step1(input_data)
            x = self.step2(x)
            return self.step3(x)

**Branching and Parallel Processing**

Parallel data processing paths that converge at aggregation points:

.. code-block:: python

    class BranchingAnalysisModel(Model):
        def __init__(self):
            super().__init__()
            self.data_normalizer = ToolNormalization(method='min_max')
            self.weight_normalizer = ToolNormalization(method='sum_to_one')
            self.aggregator = ToolWeightedAggregation(operation='topsis')
        
        def forward(self, data, weights):
            # Process data and weight branches separately
            processed_data = self.data_normalizer(data)
            processed_weights = self.weight_normalizer(weights)
            
            # Combine branches
            return self.aggregator(processed_data, processed_weights)

**Conditional Processing Logic**

Adaptive processing based on input characteristics:

.. code-block:: python

    class AdaptiveAnalysisModel(Model):
        def forward(self, data):
            normalized = self.normalizer(data)
            
            # Choose processing path based on data size
            if data.shape[0] < 100:
                return self.light_processor(normalized)
            else:
                return self.heavy_processor(normalized)

Model Composition Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hierarchical Model Architecture**

Complex workflows decomposed into specialized model layers:

.. code-block:: python

    class PreprocessingModel(Model):
        def forward(self, raw_data):
            cleaned = self.cleaner(raw_data)
            return self.normalizer(cleaned)
    
    class MainAnalysisModel(Model):
        def __init__(self, preprocessing_model):
            super().__init__()
            self.preprocessor = preprocessing_model
            self.fuzzifier = ToolFuzzification(fuzzifier=my_fuzzifier)
            self.analyzer = ToolComplexAggregation(operation='electre')
        
        def forward(self, raw_data):
            preprocessed = self.preprocessor(raw_data)
            fuzzy_data = self.fuzzifier(preprocessed)
            return self.analyzer(fuzzy_data)

**Model Ensemble Patterns**

Combining multiple models for robust analysis results:

.. code-block:: python

    class EnsembleAnalysisModel(Model):
        def forward(self, data, weights=None):
            result_a = self.model_a(data)
            result_b = self.model_b(data, weights)
            return self.ensemble_aggregator([result_a, result_b])

Execution and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Automatic Pipeline Generation**

Models automatically generate optimized execution pipelines:

.. code-block:: python

    # Model execution builds and optimizes pipeline automatically
    model = MainAnalysisModel(PreprocessingModel())
    result = model(input_data)
    
    # Access generated pipeline for inspection
    pipeline = model.pipeline
    print(f"Pipeline steps: {len(pipeline.components)}")

**Performance Optimization**

Built-in optimization strategies for large datasets:

.. code-block:: python

    class OptimizedAnalysisModel(Model):
        def forward(self, large_dataset):
            # Automatic batching for memory efficiency
            batched_results = self.batch_processor(large_dataset)
            # Parallel aggregation for speed
            return self.parallel_aggregator(batched_results)

**Model Serialization**

Models support serialization for deployment:

.. code-block:: python

    # Save and reconstruct model configuration
    model_config = model.get_config()
    new_model = MainAnalysisModel.from_config(model_config)

Data Structure Integration
--------------------------

FuzzyDataFrame serves as the cornerstone data structure for fuzzy analysis workflows, 
providing seamless integration between pandas-style data manipulation and AxisFuzzy's 
analysis framework.

FuzzyDataFrame in Analysis Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From Pandas to Fuzzy Analysis**

Convert existing pandas DataFrames using the `.fuzzy` accessor:

.. code-block:: python

    import pandas as pd
    from axisfuzzy.fuzzifier import Fuzzifier
    
    # Start with traditional pandas DataFrame
    crisp_data = pd.DataFrame({
        'performance': [0.85, 0.72, 0.91, 0.68],
        'reliability': [0.78, 0.89, 0.82, 0.75]
    })
    
    # Configure fuzzification strategy
    fuzzifier = Fuzzifier(mf='gaussmf', mtype='qrofn', pi=0.2)
    
    # Convert to FuzzyDataFrame using accessor
    fuzzy_data = crisp_data.fuzzy.to_fuzz_dataframe(fuzzifier=fuzzifier)

**Direct FuzzyDataFrame Construction**

Construct FuzzyDataFrame directly from Fuzzarray objects:

.. code-block:: python

    from axisfuzzy.core import Fuzzarray
    from axisfuzzy.analysis.dataframe import FuzzyDataFrame
    
    # Create Fuzzarray columns with specific fuzzy numbers
    performance_array = Fuzzarray([
        (0.85, 0.10), (0.72, 0.20), (0.91, 0.05), (0.68, 0.25)
    ], mtype='qrofn')
    
    # Construct FuzzyDataFrame
    fuzzy_df = FuzzyDataFrame({
        'performance': performance_array
    }, index=['Project_A', 'Project_B', 'Project_C', 'Project_D'])

Contract Integration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FuzzyDataFrame as Contract Types**

FuzzyDataFrame integrates seamlessly with AxisFuzzy's contract system:

.. code-block:: python

    from axisfuzzy.analysis.build_in import ContractFuzzyTable
    
    # FuzzyDataFrame automatically satisfies fuzzy table contracts
    def process_fuzzy_data(data: ContractFuzzyTable) -> ContractFuzzyTable:
        aggregator = ToolSimpleAggregation(operation='mean', axis=1)
        return aggregator.run(data)
    
    result = process_fuzzy_data(fuzzy_data)

**Custom Contract Validation**

Create specialized contracts for domain-specific requirements:

.. code-block:: python

    from axisfuzzy.analysis.contracts import Contract
    
    class ContractProjectEvaluation(Contract):
        def validate(self, data) -> bool:
            if not isinstance(data, FuzzyDataFrame):
                return False
            
            required_columns = {'performance', 'reliability'}
            return required_columns.issubset(set(data.columns))

Component Integration Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FuzzyDataFrame-Aware Components**

Design components that leverage FuzzyDataFrame's structure:

.. code-block:: python

    from axisfuzzy.analysis.component import AnalysisComponent
    
    class FuzzyDataFrameProcessor(AnalysisComponent):
        def __init__(self, weight_column: str = None):
            super().__init__()
            self.weight_column = weight_column
        
        @contract(input_data=ContractFuzzyTable, output=ContractFuzzyTable)
        def run(self, input_data: FuzzyDataFrame) -> FuzzyDataFrame:
            # Leverage FuzzyDataFrame's column structure
            if self.weight_column and self.weight_column in input_data.columns:
                weights = input_data[self.weight_column]
                # Process weighted operations
                return self.process_weighted(input_data, weights)
            return input_data

**Pipeline Integration Patterns**

FuzzyDataFrame flows naturally through analysis pipelines:

.. code-block:: python

    from axisfuzzy.analysis.pipeline import FuzzyPipeline
    
    # Create pipeline that processes FuzzyDataFrame
    pipeline = FuzzyPipeline("fuzzy_dataframe_analysis")
    input_data = pipeline.input("fuzzy_data", contract=ContractFuzzyTable)
    
    # Components work with FuzzyDataFrame structure
    normalized = pipeline.add(
        ToolNormalization(method='min_max', axis=0).run,
        data=input_data
    )
    
    result = pipeline.run(fuzzy_data=fuzzy_data)

Performance Optimization Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory-Efficient Operations**

FuzzyDataFrame leverages Fuzzarray's backend for optimized memory usage:

.. code-block:: python

    def efficient_fuzzy_processing(fuzzy_df: FuzzyDataFrame) -> FuzzyDataFrame:
        # Column-wise operations are optimized
        processed_columns = {}
        for column_name in fuzzy_df.columns:
            column_data = fuzzy_df[column_name]  # Returns Fuzzarray
            # Fuzzarray operations are vectorized and memory-efficient
            processed_columns[column_name] = column_data
        
        return FuzzyDataFrame(processed_columns, index=fuzzy_df.index)

**Batch Processing Strategies**

Handle large datasets efficiently:

.. code-block:: python

    def process_large_fuzzy_dataset(large_fuzzy_df: FuzzyDataFrame, 
                                   chunk_size: int = 1000) -> FuzzyDataFrame:
        results = []
        total_rows = len(large_fuzzy_df)
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = large_fuzzy_df.iloc[start_idx:end_idx]
            processed_chunk = efficient_fuzzy_processing(chunk)
            results.append(processed_chunk)
        
        return combine_fuzzy_dataframes(results)

Advanced Integration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Model-FuzzyDataFrame Integration**

Combine FuzzyDataFrame with the Model API:

.. code-block:: python

    from axisfuzzy.analysis.app.model import Model
    
    class FuzzyDataFrameAnalysisModel(Model):
        def __init__(self):
            super().__init__()
            self.processor = FuzzyDataFrameProcessor(weight_column='importance')
            self.aggregator = ToolSimpleAggregation(operation='weighted_mean')
        
        def forward(self, fuzzy_data: ContractFuzzyTable):
            processed = self.processor(fuzzy_data)
            return self.aggregator(processed)

**Interoperability with External Systems**

FuzzyDataFrame maintains compatibility with pandas ecosystem:

.. code-block:: python

    def export_fuzzy_results(fuzzy_df: FuzzyDataFrame) -> pd.DataFrame:
        # Extract membership degrees for traditional analysis
        membership_data = {}
        for column in fuzzy_df.columns:
            membership_data[f"{column}_membership"] = [
                fn.membership for fn in fuzzy_df[column]
            ]
        
        return pd.DataFrame(membership_data, index=fuzzy_df.index)

Complete Model Implementation Example
-------------------------------------

This section demonstrates a comprehensive Model implementation that integrates 
multiple analysis components into a cohesive workflow. The example showcases 
advanced Model patterns including multi-input processing, component composition, 
and structured output generation.

**Multi-Component Analysis Model**

The following example implements a complete fuzzy data analysis model that 
processes crisp data through normalization and aggregation stages:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from axisfuzzy.analysis.app.model import Model
    from axisfuzzy.analysis.component.basic import (
        ToolNormalization, ToolSimpleAggregation
    )
    from axisfuzzy.analysis.build_in import ContractCrispTable

    class ComprehensiveAnalysisModel(Model):
        """
        A multi-stage analysis model demonstrating advanced Model patterns.
        
        This model processes crisp data through normalization and aggregation
        stages, demonstrating proper Model usage patterns.
        """
        
        def __init__(self, norm_method: str = 'min_max'):
            super().__init__()
            
            # Initialize analysis components
            self.data_normalizer = ToolNormalization(method=norm_method, axis=0)
            self.aggregator = ToolSimpleAggregation(operation='mean', axis=1)
        
        def get_config(self) -> dict:
            """Return model configuration for reproducibility."""
            return {
                'normalization_method': self.data_normalizer.method,
                'aggregation_operation': self.aggregator.operation
            }
        
        def forward(self, data: ContractCrispTable):
            """
            Define the analysis workflow through component composition.
            
            The forward method demonstrates Model's declarative approach to
            workflow definition, where component calls are automatically
            translated into pipeline operations.
            
            IMPORTANT: In forward(), all variables are symbolic placeholders
            (StepOutput objects). You cannot access their properties like .columns
            or perform direct data operations. The actual computation happens
            when model.run() is called.
            """
            # Primary data processing branch - these are symbolic operations
            normalized_data = self.data_normalizer(data)
            aggregated_result = self.aggregator(normalized_data)
            
            # Return the final symbolic result
            return aggregated_result

**Model Usage and Execution**

The following demonstrates complete model instantiation, building, and execution:

.. code-block:: python

    # Prepare sample data
    sample_data = pd.DataFrame({
        'Criterion_A': [0.8, 0.6, 0.9, 0.7, 0.5],
        'Criterion_B': [0.7, 0.8, 0.6, 0.9, 0.8],
        'Criterion_C': [0.9, 0.7, 0.8, 0.6, 0.7]
    })
    
    # Initialize and build model
    model = ComprehensiveAnalysisModel(norm_method='z_score')
    model.build()  # Automatic pipeline construction
    
    # Method 1: Execute using pandas fuzzy accessor (recommended)
    results = sample_data.fuzzy.run(model, return_intermediate=True)
    final_result = results[0]  # Final aggregated scores
    intermediate_steps = results[1]  # Dictionary of intermediate results
    
    # Method 2: Direct model execution
    direct_result = model.run(sample_data)
    
    # Display results
    print(f"Final aggregated scores: {final_result}")
    print(f"Number of intermediate steps: {len(intermediate_steps)}")
    print(f"Direct execution result: {direct_result}")
    
    # Access model configuration
    config = model.get_config()
    print(f"Model configuration: {config}")

This example demonstrates Model's proper usage patterns:

- **Symbolic Computation**: The forward() method defines computation graph structure
- **Execution Separation**: Actual data processing occurs during run() calls
- **Multiple Execution Methods**: Support for both accessor and direct execution
- **Configuration Management**: Serializable model configuration for reproducibility