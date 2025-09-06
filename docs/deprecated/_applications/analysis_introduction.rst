.. _applications_analysis_introduction:

Application Module: `axisfuzzy.analysis`
========================================

Welcome to the world of AxisFuzzy Application Modules. These are high-level, specialized toolkits built upon the core AxisFuzzy library to solve specific, complex problems in domains like data analysis, control systems, and machine learning.

The `axisfuzzy.analysis` module is the first of these application modules, designed to seamlessly integrate fuzzy logic into modern data science workflows.

Core Philosophy
---------------

The design of `axisfuzzy.analysis` is inspired by leading data science libraries like `scikit-learn` and `PyTorch`, focusing on:

*   **Modularity**: Each step in an analysis process is encapsulated in a reusable ``AnalysisComponent``.
*   **Pipelining**: Complex workflows are constructed by chaining these components together into a ``FuzzyPipeline``.
*   **Contract-Driven Design**: A powerful ``Contract`` system ensures data integrity and validation as data flows through the pipeline, preventing errors before they happen.
*   **High-Level Abstraction**: A ``Model`` API allows for the creation of complex, reusable analysis models with a clean, declarative interface.

Key Components
--------------

*   **`FuzzyDataFrame`**: The central data structure, a subclass of ``pandas.DataFrame`` with a special ``.fuzzy`` accessor to store and manage fuzzy data alongside crisp data.
*   **`AnalysisComponent`**: The atomic building block of an analysis. It is a self-contained unit that performs a specific task, such as data selection, scaling, or fuzzification.
*   **`FuzzyPipeline`**: An orchestrator that manages the execution of a sequence of ``AnalysisComponent`` instances.
*   **`Contract`**: A decorator-based mechanism for enforcing data validation rules on the inputs and outputs of components.

Example: A Simple Analysis Pipeline
-----------------------------------

Here is a conceptual example of how you might build a pipeline to analyze sensor data:

.. code-block:: python

   from axisfuzzy.analysis.component import Selector, Fuzzifier, Aggregator
   from axisfuzzy.analysis.pipeline import FuzzyPipeline

   # Define a pipeline
   pipeline = FuzzyPipeline([
       # 1. Select relevant sensor columns
       Selector(columns=['temperature', 'pressure']),

       # 2. Fuzzify the temperature data
       Fuzzifier(column='temperature', membership_function='triangular', params=...),

       # 3. Aggregate the fuzzy results
       Aggregator(column='temperature', method='mean')
   ])

   # Execute the pipeline on a FuzzyDataFrame
   # result = pipeline.run(my_fuzzy_dataframe)

Future Application Modules
--------------------------

The architecture demonstrated by `axisfuzzy.analysis` serves as a blueprint for future high-level modules, including:

*   Fuzzy Clustering Systems
*   Fuzzy Inference Systems (FIS)
*   Fuzzy Neural Networks
*   Fuzzy Measure and Integral Systems
*   Fuzzy Reinforcement Learning

This modular approach allows AxisFuzzy to grow into a comprehensive ecosystem for a wide range of fuzzy logic applications, while keeping the core library lean and focused.