.. _development_architecture:

System Architecture
===================

Understanding the architecture of AxisFuzzy is key to contributing effectively. The system is designed to be modular, extensible, and high-performing, with several interconnected components working together.

Core Components
---------------

The entire framework is built around two fundamental data structures:

*   **`Fuzznum`**: Represents a single fuzzy number. It encapsulates the membership degree, non-membership degree, and the membership type (`mtype`), such as 'pfn' (Pythagorean) or 'qrofn' (q-Rung Orthopair).
*   **`Fuzzarray`**: A multi-dimensional array of `Fuzznum` objects, built on top of `numpy.ndarray`. It enables highly efficient, vectorized operations on large sets of fuzzy numbers.

These core structures are intentionally kept lean and focused on the mathematical definitions of fuzzy numbers.

Key Subsystems
--------------

Several key subsystems provide the framework's power and flexibility:

1.  **Registration System (`axisfuzzy.core.register`)**
    ---------------------------------------------------

    The registration system is the heart of AxisFuzzy's modularity. It is a singleton registry that dynamically discovers and registers different types of fuzzy number logic (known as `mtype` logic).

    *   **How it Works**: When AxisFuzzy is imported, the registry scans for modules that contain `mtype` definitions. Each `mtype` module defines the specific mathematical formulas for arithmetic operations, distance calculations, etc., for a particular class of fuzzy numbers.
    *   **Why it Matters**: This allows new types of fuzzy numbers to be added to the library without modifying the core `Fuzznum` or `Fuzzarray` code. A developer can simply create a new `mtype` module, define the logic, and the registry will make it available throughout the system.

2.  **Configuration System (`axisfuzzy.config`)**
    --------------------------------------------

    The configuration system provides a centralized way to manage global settings for the library.

    *   **`config` object**: A global `config` object allows users to set parameters such as the default `mtype` or the `q` value for q-Rung Orthopair Fuzzy Numbers.
    *   **`set_config` function**: Provides a safe way to update the configuration at runtime.

    .. code-block:: python

       import axisfuzzy as af

       # Set the default fuzzy number type for the entire session
       af.set_config(default_mtype='pfn')

       # Now, creating a Fuzznum without specifying mtype will use 'pfn'
       fuzz = af.Fuzznum([0.8, 0.3])

3.  **Extension System (`axisfuzzy.extension`)**
    --------------------------------------------

    The extension system provides two powerful mechanisms for adding new functionality to `Fuzznum` and `Fuzzarray` objects at runtime: `extension` and `mixin`.

    *   **`@extension`**: A decorator used to add new **methods** to the core classes. This is ideal for adding new computational logic (e.g., a new type of aggregation).
    *   **`@mixin`**: A decorator used to add new **attributes** or properties. This is useful for attaching metadata or derived properties.

    This dual-track system allows for clean separation of concerns when extending the core objects.

    .. seealso:: :ref:`user_guide_extension_system` for a detailed user-facing guide.

4.  **High-Performance Engine (`numba`)**
    -------------------------------------

    To achieve high performance, especially for `Fuzzarray` operations, AxisFuzzy heavily utilizes the **Numba** library. Most of the core mathematical functions found in the `mtype` logic modules are JIT-compiled (Just-In-Time) with Numba's `@njit` decorator. This translates the Python code into highly optimized machine code, resulting in performance that is often on par with compiled languages like C or Fortran.

Overall Data Flow
-----------------

A typical operation in AxisFuzzy follows this flow:

1.  A user creates a `Fuzznum` or `Fuzzarray`, specifying an `mtype`.
2.  An operation is called (e.g., `fuzz1 + fuzz2`).
3.  The `Fuzznum` object looks up the appropriate `mtype` logic from the **registration system**.
4.  The call is dispatched to the specific, **Numba-optimized** function for that `mtype` (e.g., the addition formula for Pythagorean Fuzzy Numbers).
5.  The function computes the result and returns a new `Fuzznum` instance.

This architecture ensures that the system is both easy to extend and fast in execution.