.. AxisFuzzy documentation master file

===============
AxisFuzzy
===============

*A Professional Python Framework for Fuzzy Logic Computing*

----

.. container:: intro-section

   AxisFuzzy provides high-performance, modular, and scalable fuzzy mathematical operations
   for researchers and engineers. The framework is designed with extensibility and efficiency
   in mind, enabling seamless fuzzy number computations and advanced fuzzy logic operations.

Quick Start
===========

.. tab-set::

   .. tab-item:: Installation

      .. code-block:: bash

         pip install axisfuzzy

   .. tab-item:: Basic Example

      .. code-block:: python
         
         import axisfuzzy as af

         # Create fuzzy numbers
         fuzz1 = af.Fuzznum([0.8, 0.2], mtype='qrofn')
         fuzz2 = af.Fuzznum([0.7, 0.3], mtype='qrofn')

         # Fuzzy operations
         result = fuzz1 + fuzz2
         distance = fuzz1.distance(fuzz2)

         print(f"Result: {result}")
         print(f"Distance: {distance}")

   .. tab-item:: Array Operations

      .. code-block:: python

         import axisfuzzy as af

         # Create fuzzy arrays
         data = [[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]]
         fuzz_array = af.Fuzzarray(data, mtype='qrofn')

         # Batch operations
         mean_result = fuzz_array.mean()
         reshaped = fuzz_array.reshape(3, 1, 2)

Navigation
==========

.. container:: nav-cards

   .. grid:: 2 2 2 2
      :gutter: 4
      :class-container: nav-grid

      .. grid-item-card:: Getting Started
         :class-card: nav-card
         :link: getting_started/index
         :link-type: doc

         📚

         Installation guide, basic concepts, and first steps with AxisFuzzy

      .. grid-item-card:: User Guide
         :class-card: nav-card
         :link: user_guide/index
         :link-type: doc

         📖

         Comprehensive tutorials, examples, and advanced usage patterns

      .. grid-item-card:: Developer Guide
         :class-card: nav-card
         :link: development/index
         :link-type: doc

         🔧

         Extending AxisFuzzy, contributing guidelines, and architecture details

      .. grid-item-card:: API Reference
         :class-card: nav-card
         :link: api/index
         :link-type: doc

         📋

         Complete API documentation with detailed function and class references

      .. grid-item-card:: Extension Systems
         :class-card: nav-card
         :link: extension/index
         :link-type: doc

         🚀

         High-level extension modules for specific domains and applications

.. raw:: html

   <div class="footer-spacing"></div>

.. toctree::
   :hidden:

   getting_started/index
   user_guide/index
   development/index
   fuzzy_types/index
   api/index
   extension/index
