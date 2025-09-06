.. _getting_started_introduction:

************
Introduction
************

Welcome to AxisFuzzy! This guide provides an introduction to the core concepts of the library and helps you get started with fuzzy logic programming.

What is AxisFuzzy?
==================

AxisFuzzy is a powerful Python library for fuzzy logic and fuzzy computing, designed with modularity, extensibility, and performance in mind. The entire system is built around two core data structures: ``Fuzznum`` (fuzzy number) and ``Fuzzarray`` (fuzzy array), and is extended through a series of pluggable sub-modules and extension systems.

Core Concepts
=============

- **Fuzznum**: A scalar representation of a fuzzy number, which is the basic unit of fuzzy data.
- **Fuzzarray**: An array-like data structure for handling collections of fuzzy numbers, supporting efficient vectorized operations.
- **Membership Functions**: Functions that define the degree of membership of an element in a fuzzy set.
- **Fuzzifier**: A component that converts crisp (non-fuzzy) data into fuzzy data.
- **Extension System**: A flexible mechanism for extending the functionality of AxisFuzzy with custom components.

Why Use AxisFuzzy?
==================

- **High Performance**: Leverages NumPy and Numba for JIT compilation to accelerate computations.
- **Modular Design**: Allows you to use only the components you need, keeping your code clean and efficient.
- **Extensible**: Easily extend the library with your own custom logic and components.
- **Rich Functionality**: Provides a wide range of tools for fuzzy logic, from basic operations to complex analysis.

This guide will walk you through the installation process, core concepts, and practical examples to help you master AxisFuzzy.