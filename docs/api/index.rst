=========
API
=========

Welcome to the comprehensive AxisFuzzy API Reference Documentation, your definitive guide
to understanding and utilizing every component, class, method, and function within the
AxisFuzzy framework. This documentation provides detailed technical specifications,
parameter descriptions, return values, and usage examples for all public APIs.

The API documentation is organized into seven core modules that represent the fundamental
building blocks of AxisFuzzy's architecture. Each module contains complete class
hierarchies, method signatures, and implementation details necessary for advanced
development and integration tasks. The documentation follows standard Python docstring
conventions and includes comprehensive type annotations for enhanced development experience.

From configuration management and core data structures to specialized fuzzy operations
and random number generation, this reference covers every aspect of AxisFuzzy's
programmatic interface. Whether you're implementing custom fuzzy logic algorithms,
extending the framework with new functionality, or integrating AxisFuzzy into larger
systems, this API documentation provides the technical depth required for professional
development.

Each module section includes inheritance diagrams, cross-references to related
components, and practical code examples demonstrating real-world usage patterns.
The documentation is designed to serve both as a learning resource for newcomers
and a comprehensive reference for experienced developers working with complex
fuzzy logic implementations.

..  toctree::
    :maxdepth: 1

    config/index
    core/index
    membership/index
    fuzzifier/index
    extension/index
    mixin/index
    random/index

Core Component Overview
========================

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`config/index`
     - Configure the system, providing global configuration and default values
   * - :doc:`core/index`
     - Core Data Structures (Fuzznum, Fuzzarray, FuzznumStrategy)
   * - :doc:`fuzzifier/index`
     - Fuzzification system, converting precise values into fuzzy numbers
   * - :doc:`extension/index`
     - Expand the system to add specialized functions for different types.
   * - :doc:`mixin/index`
     - Mixin system provides array operations similar to NumPy
   * - :doc:`random/index`
     - Random Fuzzy Number Generation System