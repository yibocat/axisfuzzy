=========================
Fuzzy Sets Type
=========================

This section provides comprehensive documentation for the specialized fuzzy set types
implemented in AxisFuzzy, focusing on advanced mathematical frameworks that extend
traditional fuzzy logic capabilities. These sophisticated fuzzy set types enable
researchers and practitioners to model complex uncertainty scenarios with enhanced
precision and flexibility.

The documentation covers four primary categories of fuzzy sets, ranging from classical
foundational types to advanced hesitant fuzzy structures. Each type addresses specific
uncertainty modeling requirements and provides unique computational advantages for
different application domains.

Classical Fuzzy Sets (FS) represent the foundational framework introduced by
Lotfi A. Zadeh, providing the theoretical basis for all fuzzy logic systems.
These sets model uncertainty through single membership degrees and serve as
the computational foundation for more complex fuzzy types.

Q-Rung Orthopair Fuzzy Numbers (QROFN) extend classical fuzzy sets by introducing
non-membership degrees with parameterized constraints, enabling more flexible
uncertainty representation through the q-rung parameter that controls the
relationship between membership and non-membership assessments.

Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFN) further enhance this
framework by representing both membership and non-membership degrees as intervals
rather than point values, enabling the modeling of uncertainty in the precision
of fuzzy assessments themselves.

Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFN) incorporate hesitation elements
into the q-rung framework, allowing decision-makers to express multiple possible
membership and non-membership values simultaneously, particularly valuable in
group decision-making scenarios where consensus may be difficult to achieve.

Each fuzzy type includes detailed mathematical definitions, implementation guidelines,
practical examples, and integration methods with AxisFuzzy's core functionality.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   fs
   qrofn
   ivqrofn
   qrohfn


