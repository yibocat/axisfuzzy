==========================================
extension(axisfuzzy.extension)
==========================================

Introduction
----------------
The AxisFuzzy extension system is a highly flexible mechanism that allows developers to dynamically add and manage functionalities for different types of fuzzy numbers (`mtype`). Its core idea is a `mtype`-based pluggable architecture, enabling AxisFuzzy to easily extend support for new fuzzy number types or provide specialized operations for existing ones, without modifying the core code.

.. toctree::

    registry
    decorator
    dispatcher