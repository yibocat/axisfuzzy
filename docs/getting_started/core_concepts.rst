.. _core_concepts:

Core Concepts
=============

This section delves into the fundamental concepts of `AxisFuzzy`, providing a deeper understanding of its architecture and components.

What is a classical fuzzy set?
---------------------------------
Classic fuzzy sets, also known as Type-1 Fuzzy Sets, are the foundation of fuzzy theory.

A fuzzy set A is a mathematical object defined on a universe of discourse :math:`U`. Its core feature is the membership function :math:`\mu_A(x)`, which maps each element x in the universe U to a real number in the interval [0, 1].

.. math::

   \mu_A(x) \rightarrow [0, 1]

Here, :math:`\mu_A(x)` denotes the membership degree of element :math:`x` in the fuzzy set :math:`A`, with values ranging from [0, 1].

If :math:`\mu_A(x) = 0`, then :math:`x` is not a member of fuzzy set :math:`A`; if :math:`\mu_A(x) = 1`, then :math:`x` is a member of fuzzy set :math:`A`; if :math:`0 < \mu_A(x) < 1`, then the membership degree of :math:`x` in fuzzy set :math:`A` is :math:`\mu_A(x)`.

Therefore, a fuzzy set :math:`A` can be represented as :math:`A = \{(x, \mu_A(x))| x \in U \}`

Where:

- :math:`U` is the set of all possible elements (universe of discourse), such as all possible heights, temperatures, etc.
- :math:`x` is a specific element in the universe U, such as height 175cm.
- :math:`\mu_A(x)` is the membership degree of element :math:`x` in fuzzy set :math:`A`, representing the degree of :math:`x` belonging to :math:`A`.

Relationship with `AxisFuzzy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the simplest case, if you have an exact numerical array (e.g., numpy.ndarray ) and use ``Fuzzifier`` to fuzzify it, you are actually creating a classical fuzzy set.

.. code-block:: python

    import numpy as np
    from axisfuzzy import Fuzzifier

    # Universe of Discourse U: A set of height values
    crisp_heights = np.array([150, 160, 170, 180, 190, 200])

    # Define a fuzzy set A: "Tall people"
    # We use a Gaussian membership function to define it, with sigma as 4 and center as 170
    fuzzifier = Fuzzifier(mf='gaussmf', mf_params={"sigma": 4.0, "c": 170.0})

    # Calculate the membership degree of each height value in the "Tall people" set
    membership_degrees = fuzzifier(crisp_heights)

    # Now, `membership_degrees` represents a classical fuzzy set
    # It stores the membership degrees of each element in `crisp_heights`
    print(membership_degrees)

In this example, `crisp_heights` is the universe of discourse U, `fuzzifier` defines the membership function :math:`\mu_A(x)`, and the final ``membership_degrees`` (an array of numerical values between [0, 1]) is a representation of fuzzy set A.

`AxisFuzzy` abstracts fuzzy sets at a high level, using ``Fuzzarray`` to represent fuzzy sets. This allows `AxisFuzzy` to support various types of fuzzy sets, including classical fuzzy sets, 2-type fuzzy sets, intuitionistic fuzzy sets, etc.


What is an extended fuzzy set?
-------------------------------------

A classical fuzzy set uses a single numerical value :math:`\mu_A(x)` to describe the membership relation, but this is not sufficient in many complex scenarios. Extended fuzzy sets introduce more parameters to more accurately characterize uncertainty.

In `AxisFuzzy`, the ``Fuzzarray`` are used to represent a fuzzy set. Each element in the ``Fuzzarray`` is a fuzzy number ``Fuzznum``.


Type-II Fuzzy Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is another important type of generalized fuzzy set. If the membership degree of a first-class fuzzy set is an exact numerical value, then the membership degree of a second-class fuzzy set is itself a fuzzy set.

This means that the membership degree of a second-class fuzzy set is no longer a point (such as 0.7), but an interval or fuzzy number with "width". This is called the Footprint of Uncertainty (FOU). It is used to handle situations where the membership function itself is uncertain. For example, different people's definitions of "tall" (i.e., membership function) are different.

Intuitionistic fuzzy set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
IFS considers not only the membership degree :math:`\mu(x)`, but also introduces the non-membership degree :math:`\nu(x)`.

- :math:`\mu(x)`: The degree to which x belongs to set A.
- :math:`\nu(x)`: The degree to which x does not belong to set A.
- :math:`\pi(x)`: The degree of hesitation or uncertainty.

These two satisfy the constraint: :math:`\mu(x) + \nu(x) ≤ 1`.

Where :math:`\mu(x)` is called the membership degree, and :math:`\nu(x)` is called the non-membership degree.

The remaining part :math:`\pi(x) = 1 - \mu(x) - \nu(x)` is called the hesitation degree or uncertainty degree. It represents that we cannot be sure whether x belongs to A or not.

Relationship with `AxisFuzzy`: The q-rung orthopair fuzzy numbers (q-ROFN) in `AxisFuzzy` are a direct generalization of IFS. When :math:`q=1`, a q-ROFN becomes an intuitionistic fuzzy number.

q-Rung Orthopair Fuzzy Sets(q-ROFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a further relaxation and generalization of IFS. It changes the constraint from linear to nonlinear: :math:`\mu(x)^q + \nu(x)^q ≤ 1`

Where :math:`q` is a non-negative integer.

- When :math:`q=1`, it is an intuitionistic fuzzy set (IFS).
- When :math:`q=2`, it is called a Pythagorean fuzzy set (PFS), with the constraint :math:`\mu(x)^2 + \nu(x)^2 ≤ 1` .
As q increases, the range of :math:`(\mu, \nu)` pairs that satisfy the conditions also expands, enabling q-ROFS to describe a broader range of uncertainties, which classical IFS cannot achieve.

Hesitant Fuzzy Sets(HFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In decision-making, experts may give multiple possible values for the membership degree of an element, rather than a single value. For example, for the height 178cm, expert A thinks the membership degree is 0.6, and expert B thinks it is 0.7.

Hesitant fuzzy set allows the membership degree of an element to be a set of values in [0, 1]. h_A(x) = {0.6, 0.7}

Where :math:`h_A(x)` is the membership degree of element :math:`x` in fuzzy set :math:`A`, representing the degree of :math:`x` belonging to :math:`A`.

Relationship with `AxisFuzzy`: `AxisFuzzy` supports q-rung orthopair hesitant fuzzy sets (q-ROHFS) through q-rung orthopair hesitant fuzzy numbers (q-ROHFN). A q-ROHFN contains a membership degree set and a non-membership degree set.

Where :math:`q` is the q-rung, :math:`md` is the membership degree set, and :math:`nmd` is the non-membership degree set.

A Fuzzarray composed of q-ROHFNs can be regarded as a q-ROHFS, which is one of the most expressive extensions of fuzzy sets in current fuzzy theory.

Fuzznum(Fuzzy number)
-----------------------

A ``Fuzznum`` is the scalar representation of a fuzzy number. It is the most basic data structure in `AxisFuzzy`. A fuzzy number is characterized by a membership function that assigns a degree of membership, between 0 and 1, to each possible value.

Currently, the advanced extended fuzzy number types supported by `AxisFuzzy` include:

 - q-rung orthopair fuzzy sets(q-ROFS, ``mtype='qrofn'``)
 - Intuitionistic Fuzzy Set (IFS, when ``mtype='qrofn'`` and ``q=1``)
 - Pythagorean Fuzzy Set (PFS, when ``mtype='qrofn'`` and ``q=2``)
 - Fermatean Fuzzy Set (FFS, when ``mtype='qrofn'`` and ``q=3``)
 - q-rung orthopair hesitant fuzzy sets(q-ROHFS, when ``mtype='qrohfn'``)

The advanced extended fuzzy number types that the future `AxisFuzzy` plans to support include:

 - Classic Fuzzy Sets(Type-I Fuzzy Sets, FS, when ``mtype='fs'``)
 - Type-II fuzzy sets (Type-II Fuzzy Sets, Type-IIFS, when ``mtype='type2fs'``)
 - q-rung interval-valued fuzzy sets (iv-qrofn, when ``mtype='ivqfs'``)
 - Hesitant fuzzy sets (HFS, when ``mtype='hfs'``)

Fuzzarray(Fuzzy number array, fuzzy sets)
------------------------

A ``Fuzzarray`` is a homogeneous array of ``Fuzznum`` objects. It is designed to be a high-performance data structure for vectorized operations on fuzzy numbers, similar to NumPy arrays. In `AxisFuzzy`, ``Fuzzarray`` is used as a collection of fuzzy numbers and supports vectorized operations, thereby improving computational efficiency.

Key features of ``Fuzzarray`` include:

- **Advanced High-Dimensional Fuzzy Number Container**: The core container of `AxisFuzzy`, which hosts all the basic operations of `AxisFuzzy`.
- **Vectorized Operations**: Perform arithmetic and logical operations on entire arrays at once.
- **Aggregation**: Functions like ``sum()``, ``mean()``, and ``std()`` are available.
- **Broadcasting**: Supports broadcasting rules similar to NumPy for operations between arrays of different shapes.

Membership Functions
--------------------

What is a membership function?
~~~~~~~~~~~~~

In simple terms, a membership function is a mathematical function that defines the degree to which an element belongs to a "fuzzy set." This "degree" is called the Degree of Membership and its value falls within the range [0, 1].

- A membership degree of 0 means the element is completely not a member of the fuzzy set.
- A membership degree of 1 means the element is completely a member of the fuzzy set.
- A membership degree between 0 and 1 means the element is partially a member of the fuzzy set.
This is in stark contrast to classical set theory (also known as "crisp sets"). In classical set theory, an element either belongs to a set or does not belong to it, with no intermediate states.

A Simple Example: "Tall"
~~~~~~~~~~~~~~~~~~~~~~~~

Let's use the concept of "tall" to understand this.

1. In classical set theory:
   We might set an exact threshold, such as a person taller than 180 cm is defined as "tall."
   
   - A person 180.1 cm tall, belongs to the "tall" set (membership degree is 1).
   - A person 179.9 cm tall, does not belong to the "tall" set (membership degree is 0).
     This definition is very "crisp," which is not natural in the real world because a 0.2 cm difference leads to a fundamental change.

2. In fuzzy logic:
   We can use a membership function to define the fuzzy set "tall." This function will assign a membership degree between 0 and 1 to a height value.
   
   - A person 160 cm tall, may have a membership degree of 0 (not tall at all).
   - A person 175 cm tall, may have a membership degree of 0.6 (a bit tall).
   - A person 185 cm tall, may have a membership degree of 0.95 (very tall).
   - A person 200 cm tall, may have a membership degree of 1 (very very tall).

The Role of Membership Functions in Fuzzy Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Membership functions are the cornerstone of fuzzy logic systems. They play a key role in the following three stages:

1. Fuzzification: This is the first step in a fuzzy system. It converts an exact, real-world input value (for example, a temperature sensor reading of 22°C) into fuzzy membership degrees. For instance, for 22°C, the system might calculate, based on predefined membership functions, that it belongs to the category "cold" with a degree of 0.1, to "moderate" with a degree of 0.8, and to "hot" with a degree of 0.
2. Fuzzy Inference: Once the input is fuzzified, the system performs reasoning based on a series of fuzzy rules in the form of "IF-THEN." For example, a rule might be: "If the temperature is 'moderate,' then the fan speed should be 'medium.'" The inference engine uses the input's degree of membership (0.8) to evaluate the firing strength of this rule.
3. Defuzzification: This is the final step. The result of fuzzy reasoning is one or more fuzzy sets. The defuzzification process converts this fuzzy output (e.g., "medium" speed) back into a precise, executable value (e.g., 1200 RPM) for controlling the device.

Membership Functions in `AxisFuzzy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`AxisFuzzy` provides a membership function factory for creating and customizing membership functions. It includes the following membership functions:
 
 - Triangular membership function(``TriangularMF``)
 - Trapezoidal membership function(``TrapezoidalMF``)
 - Gaussian membership function(``GaussianMF``)
 - Sigmoid membership function(``SigmoidMF``)
 - S shape membership function(``SMF``)
 - Z shape membership function(``ZMF``)
 - Pi shape membership function(``PiMF``)
 - Generalized bell membership function(``GeneralizedBellMF``)
 - Double Gaussian membership function(``DoubleGaussianMF``)

Fuzzifier
---------

A ``Fuzzifier`` is a component responsible for converting crisp (non-fuzzy) data into fuzzy data. This process, known as fuzzification, is the first step in many fuzzy logic systems.

`AxisFuzzy` provides a powerful fuzzification system that allows highly flexible and customizable configuration of various types of fuzzifiers, offering robust functionality and high customizability.

Extension System
----------------

The extension system is a powerful feature of `AxisFuzzy` that allows for the dynamic addition of new functionalities. You can create custom components, such as new types of fuzzy numbers or membership functions, and integrate them seamlessly into the library.

This modular and extensible design makes `AxisFuzzy` a versatile tool for a wide range of fuzzy logic applications.
