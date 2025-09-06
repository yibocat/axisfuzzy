.. _development_contributing_guide:

Contributing to AxisFuzzy
=========================

Thank you for your interest in contributing to AxisFuzzy! We welcome contributions from the community to help make this library even better. This guide outlines the process and best practices for contributing.

Git Flow Workflow
-----------------

We use the **Git Flow** branching model to manage our development process. This ensures a clean and organized commit history.

**Main Branches:**

*   ``master``: Contains the latest stable, production-ready release.
*   ``develop``: The main development branch where all new features are integrated.

**Supporting Branches:**

*   ``feature/*``: For developing new features. Branched from ``develop``.
*   ``release/*``: For preparing a new production release. Branched from ``develop``.
*   ``hotfix/*``: For fixing critical bugs in a production release. Branched from ``master``.

**Key Commands:**

.. code-block:: bash

   # Start a new feature
   git flow feature start <feature-name>

   # Finish a feature (merges to develop)
   git flow feature finish <feature-name>

   # Start a release
   git flow release start <version>

   # Finish a release (merges to master and develop)
   git flow release finish <version>

Coding Standards
----------------

To maintain code quality and consistency, we adhere to the following standards:

*   **PEP 8**: All Python code must follow the PEP 8 style guide.
*   **Type Hinting**: All functions and methods must include type hints.
*   **NumPy Style Docstrings**: All public modules, classes, functions, and methods must have comprehensive docstrings in the NumPy style. This is crucial for auto-generating our API documentation.
*   **Clear Comments**: For complex logic, algorithms, or design choices, provide clear and concise comments.

Testing
-------

We use ``pytest`` for our testing framework. A thorough test suite is essential for ensuring the stability and correctness of the library.

**Requirements for Contributions:**

*   Every new feature must be accompanied by corresponding unit tests.
*   Any bug fix must include a regression test that fails without the fix and passes with it.
*   All tests must pass before a pull request can be merged.

**Running Tests:**

.. code-block:: bash

   # Run the full test suite
   pytest

   # Run tests for a specific module
   pytest tests/test_core/

Submission Process
------------------

1.  **Fork the Repository**: Create your own fork of the AxisFuzzy repository on GitHub.
2.  **Create a Branch**: Create a new feature or hotfix branch using the Git Flow conventions.
3.  **Implement and Test**: Write your code, ensuring it adheres to the coding standards and is fully tested.
4.  **Update Documentation**: If your changes affect the public API or introduce new concepts, update the relevant documentation in the ``docs/`` directory.
5.  **Submit a Pull Request (PR)**: Push your changes to your fork and open a pull request against the ``develop`` branch of the main repository. Provide a clear description of your changes in the PR.