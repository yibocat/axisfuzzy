=================================
membership(axisfuzzy.membership)
=================================

Introduction
------------
The membership module supplies a concise, extensible library of fuzzy membership
functions used for fuzzification in AxisFuzzy. It includes an abstract
:class:`~.membership.MembershipFunction` base class, a comprehensive set of standard implementations
(triangular, trapezoidal, Gaussian, sigmoid, S/Z/Pi, generalized bell, double
Gaussian, etc.), and a factory for creating functions by name or alias. Each
function supports NumPy-vectorized evaluation, parameter validation and updates,
optional plotting, and careful numerical handling. Designed for easy integration
with the fuzzification pipeline, the module also allows users to add custom
membership functions that are discovered and instantiated by the factory.

.. toctree::

    base
    function
    factory