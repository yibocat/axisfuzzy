.. _development_testing_guide:

Testing Guide
=============

A robust testing strategy is crucial for maintaining the quality and reliability of AxisFuzzy. This guide outlines the testing framework, how to run tests, and how to add new tests.

Testing Framework
-----------------

We use **pytest** as our primary testing framework. Pytest is a mature, feature-rich framework that makes it easy to write small, readable tests, and can scale to support complex functional testing.

Key features of our testing setup:

*   **Directory Structure**: All tests are located in the `/tests` directory at the root of the project.
*   **Modular Tests**: Tests are organized into subdirectories that mirror the structure of the `axisfuzzy` library itself (e.g., `tests/core`, `tests/membership`).
*   **Fixtures**: We use pytest fixtures to set up reusable objects and data for our tests, such as pre-configured `Fuzznum` and `Fuzzarray` instances.

How to Run Tests
----------------

To run the full test suite, you first need to install the development dependencies, which include `pytest`.

1.  **Install Dependencies**:

    .. code-block:: bash

       # Make sure you are in the root of the project directory
       pip install -e .[dev]

2.  **Run Pytest**:

    Once the dependencies are installed, you can run the entire test suite with a single command:

    .. code-block:: bash

       pytest

    Pytest will automatically discover and run all files of the format `test_*.py` or `*_test.py` in the `tests/` directory.

Running Specific Tests
~~~~~~~~~~~~~~~~~~~~~~

You can also run specific tests:

*   **Run tests in a specific file**:

    .. code-block:: bash

       pytest tests/core/test_fuzznum.py

*   **Run a specific test function**:

    .. code-block:: bash

       pytest tests/core/test_fuzznum.py::test_addition

*   **Run tests with a keyword expression**:

    .. code-block:: bash

       pytest -k "distance and not euclidean"

Writing New Tests
-----------------

When you contribute new code, you should also contribute corresponding tests. Here are some guidelines for writing tests for AxisFuzzy:

1.  **Follow the Structure**: Place your new test file in the appropriate subdirectory within `tests/`. For example, a test for a new membership function should go in `tests/membership/`.

2.  **Use Descriptive Names**: Test function names should be descriptive of what they are testing, e.g., `test_qrofn_addition_with_scalars`.

3.  **Keep Tests Small and Focused**: Each test function should ideally test one specific piece of functionality.

4.  **Use Fixtures for Setup**: If you need to create `Fuzznum` or other objects for your tests, check if an existing fixture can provide them. If not, consider creating a new reusable fixture.

5.  **Assert Expected Outcomes**: Use `assert` statements to check that the code behaves as expected. For floating-point comparisons, use `pytest.approx` to avoid precision issues.

Example Test
~~~~~~~~~~~~

Here is a simplified example of what a test function might look like:

.. code-block:: python

   import pytest
   import axisfuzzy as af

   def test_pfn_complement():
       """Test the complement operation for Pythagorean Fuzzy Numbers."""
       fuzz = af.Fuzznum([0.8, 0.3], mtype='pfn')
       complement_fuzz = ~fuzz

       # The complement should swap membership and non-membership
       assert complement_fuzz.membership == pytest.approx(0.3)
       assert complement_fuzz.non_membership == pytest.approx(0.8)

Continuous Integration (CI)
---------------------------

We use a Continuous Integration (CI) service (like GitHub Actions) to automatically run the full test suite for every pull request. This ensures that new contributions do not break existing functionality.

A pull request will not be merged unless all tests pass.