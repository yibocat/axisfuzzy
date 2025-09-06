.. _getting_started_core_concepts:

*************
Core Concepts
*************

This section delves into the fundamental concepts of AxisFuzzy, providing a deeper understanding of its architecture and components.

Fuzzy Numbers (Fuzznum)
=======================

A `Fuzznum` is the scalar representation of a fuzzy number. It is the most basic data structure in AxisFuzzy. A fuzzy number is characterized by a membership function that assigns a degree of membership, between 0 and 1, to each possible value.

.. math::

   \mu_A(x) \rightarrow [0, 1]

Where :math:`\mu_A(x)` is the membership function of the fuzzy set A.

AxisFuzzy supports various types of fuzzy numbers, including:

- **Triangular Fuzzy Numbers**: Defined by a lower bound, a peak, and an upper bound.
- **Trapezoidal Fuzzy Numbers**: Defined by four points that form a trapezoid.
- **Gaussian Fuzzy Numbers**: Defined by a mean and a standard deviation.

Fuzzy Arrays (Fuzzarray)
========================

A `Fuzzarray` is a homogeneous array of `Fuzznum` objects. It is designed to be a high-performance data structure for vectorized operations on fuzzy numbers, similar to NumPy arrays.

Key features of `Fuzzarray` include:

- **Vectorized Operations**: Perform arithmetic and logical operations on entire arrays at once.
- **Aggregation**: Functions like `sum()`, `mean()`, and `std()` are available.
- **Broadcasting**: Supports broadcasting rules similar to NumPy for operations between arrays of different shapes.

Membership Functions
====================

Membership functions are at the heart of fuzzy logic. They define how a crisp value is mapped to a degree of membership in a fuzzy set. AxisFuzzy provides a flexible factory for creating and customizing membership functions.

Fuzzifier
=========

A `Fuzzifier` is a component responsible for converting crisp (non-fuzzy) data into fuzzy data. This process, known as fuzzification, is the first step in many fuzzy logic systems.

AxisFuzzy provides different strategies for fuzzification, allowing you to choose the most appropriate method for your application.

Extension System
================

The extension system is a powerful feature of AxisFuzzy that allows for the dynamic addition of new functionalities. You can create custom components, such as new types of fuzzy numbers or membership functions, and integrate them seamlessly into the library.

This modular and extensible design makes AxisFuzzy a versatile tool for a wide range of fuzzy logic applications.