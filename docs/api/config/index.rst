:html_theme.sidebar_secondary.remove:

config(axisfuzzy.config)
========================

Introduction
------------
The configuration subsystem of AxisFuzzy is responsible for maintaining global and local configuration items (such as default `mtype`, numerical precision, backend selection, etc.), and provides a unified access interface for various subsystems of the library. The configuration module includes user-facing APIs (convenience functions) and a lower-level configuration manager (used for programmatic and persistent operations).

.. toctree::

    :maxdepth: 1
    :caption: Configuration Modules

    manager
    api
    defaults
