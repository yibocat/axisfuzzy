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

   .. tab-item:: Factory Functions

      .. code-block:: python
         
         from axisfuzzy import fuzzynum, fuzzyarray

         # Create fuzzy numbers with modern factory functions
         fn1 = fuzzynum((0.8, 0.2), q=2)  # q-Rung Orthopair Fuzzy Number
         fn2 = fuzzynum((0.7, 0.3), q=2)

         # Fuzzy operations
         result = fn1 + fn2
         distance = fn1.distance(fn2)
         score = fn1.score  # Score function

         print(f"Result: {result}")
         print(f"Distance: {distance:.3f}")
         print(f"Score: {score:.3f}")

   .. tab-item:: High-Performance Arrays

      .. code-block:: python

         from axisfuzzy import fuzzyarray, fuzzynum
         import axisfuzzy.random as ar

         # Create fuzzy arrays efficiently
         fuzzy_numbers = [
             fuzzynum((0.8, 0.2)),
             fuzzynum((0.6, 0.4)),
             fuzzynum((0.9, 0.1))
         ]
         fs = fuzzyarray(fuzzy_numbers)

         # Vectorized operations (10x-100x faster)
         mean_result = fs.mean()
         distances = fs.distance(fuzzynum((0.5, 0.4)))

         # Random generation for simulation
         random_array = ar.rand(shape=(1000,))
         print(f"Generated {len(random_array)} fuzzy numbers")

   .. tab-item:: Advanced Features

      .. code-block:: python

         from axisfuzzy import fuzzynum
         from axisfuzzy.membership import create_mf
         from axisfuzzy.fuzzifier import Fuzzifier

         # Hesitant fuzzy numbers
         hesitant_fn = fuzzynum(
             ([0.5, 0.6, 0.7], [0.2, 0.3]), 
             mtype='qrohfn', q=1
         )

         # Membership functions and fuzzification
         gauss_mf, _ = create_mf('gaussmf', sigma=0.15, c=0.5)
         fuzzifier = Fuzzifier(mf='gaussmf', 
                              mf_params={'sigma': 0.1, 'c': 0.5})
         
         # Convert crisp values to fuzzy
         crisp_data = [0.3, 0.6, 0.9]
         fuzzy_results = fuzzifier(crisp_data)

Navigation
==========

.. container:: nav-cards

   .. grid:: 2 2 3 3
      :gutter: 4
      :class-container: nav-grid

      .. grid-item-card:: 🚀 Getting Started
         :class-card: nav-card
         :link: getting_started/index
         :link-type: doc

         **Quick Start & Fundamentals**

         Installation guide, core concepts, and hands-on tutorials to get you productive with AxisFuzzy in minutes

      .. grid-item-card:: 📖 User Guide
         :class-card: nav-card
         :link: user_guide/index
         :link-type: doc

         **Comprehensive Tutorials**

         In-depth guides covering data structures, operations, membership functions, fuzzification, and advanced patterns

      .. grid-item-card:: 🧮 Fuzzy Types
         :class-card: nav-card
         :link: fuzzy_types/index
         :link-type: doc

         **Advanced Mathematical Frameworks**

         q-Rung Orthopair Fuzzy Numbers (QROFN) and Hesitant Fuzzy Numbers (QROHFN) for complex uncertainty modeling

      .. grid-item-card:: 🔧 Developer Guide
         :class-card: nav-card
         :link: development/index
         :link-type: doc

         **Extend & Customize**

         Create custom fuzzy types, operations, fuzzification strategies, and integrate with existing systems

      .. grid-item-card:: 📋 API Reference
         :class-card: nav-card
         :link: api/index
         :link-type: doc

         **Complete Technical Reference**

         Detailed documentation for all classes, methods, and functions with type annotations and examples

      .. grid-item-card:: ⚡ Extension Systems
         :class-card: nav-card
         :link: extension/index
         :link-type: doc

         **Domain-Specific Modules**

         High-level specialized extensions for machine learning, decision support, and scientific computing

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
