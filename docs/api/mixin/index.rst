:html_theme.sidebar_secondary.remove:

========================
mixin (axisfuzzy.mixin)
========================

Introduction
------------
The AxisFuzzy mixin system provides mtype-agnostic structural operations for :class:`~.base.Fuzznum` and :class:`~.base.Fuzzarray` classes.
It enables NumPy-like functionality such as ``reshape``, ``flatten``, ``transpose``, and ``concatenate`` that work uniformly
across all fuzzy number types without requiring mtype-specific dispatch logic. Functions are dynamically injected as both
instance methods and top-level functions during library initialization, offering seamless integration with the core data structures.

.. toctree::

    :maxdepth: 1
    :caption: Extension Modules

    registry
    register
    factory