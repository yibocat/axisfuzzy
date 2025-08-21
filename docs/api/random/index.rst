:html_theme.sidebar_secondary.remove:

==========================
random (axisfuzzy.random)
==========================

Introduction
------------
The `axisfuzzy.random` module provides a unified, extensible interface for
reproducible and high-performance random generation of fuzzy numbers and arrays.
It centralizes seed management, exposes a registry for mtype-specific random
generators, and offers a concise API for creating both single `Fuzznum` objects
and vectorized `Fuzzarray` instances. The design favors NumPy's modern RNG,
thread-safe global state, and direct backend population to ensure scalability
and deterministic behavior for scientific workflows.

.. toctree::

    :maxdepth: 1
    :caption: Extension Modules

    base
    registry
    seed
    api