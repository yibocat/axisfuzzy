=========
API
=========

Complete AxisFuzzy API documentation and user guide.

..  toctree::
    :maxdepth: 1

    config/index
    core/index
    membership/index
    fuzzify
    extension/index
    mixin/index
    random/index

Core Component Overview
============

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`config/index`
     - Configure the system, providing global configuration and default values
   * - :doc:`core/index`
     - Core Data Structures (Fuzznum, Fuzzarray, FuzznumStrategy)
   * - :doc:`fuzzify`
     - Fuzzification system, converting precise values into fuzzy numbers
   * - :doc:`extension/index`
     - Expand the system to add specialized functions for different types.
   * - :doc:`mixin/index`
     - Mixin system provides array operations similar to NumPy
   * - :doc:`random`
     - Random Fuzzy Number Generation System