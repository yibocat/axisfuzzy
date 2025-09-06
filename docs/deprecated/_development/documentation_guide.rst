.. _development_documentation_guide:

Documentation Guide
==================

High-quality documentation is crucial for any software project. This guide explains how to write and maintain documentation for AxisFuzzy, including docstrings, user guides, and API references.

Documentation Structure
----------------------

AxisFuzzy's documentation is organized into several key sections:

*   **Getting Started**: Basic introduction, installation, and quick start guides.
*   **User Guide**: In-depth tutorials and explanations of core concepts.
*   **Development Guide**: Information for contributors and developers.
*   **API Reference**: Auto-generated documentation from docstrings.
*   **Applications**: Documentation for high-level application modules.

Writing Documentation
--------------------

We use **Sphinx** with **reStructuredText** (reST) markup for our documentation. The source files are located in the `docs/` directory.

Key Files and Directories
~~~~~~~~~~~~~~~~~~~~~~~~~

*   `docs/conf.py`: Sphinx configuration file
*   `docs/index.rst`: Main documentation landing page
*   `docs/getting_started/`: Basic tutorials and installation guides
*   `docs/user_guide/`: In-depth guides and tutorials
*   `docs/development/`: Development and contribution guides
*   `docs/api/`: Auto-generated API documentation
*   `docs/applications/`: Documentation for high-level modules

Building Documentation
---------------------

To build the documentation locally:

1.  **Install Dependencies**:

    .. code-block:: bash

       pip install -e .[docs]

2.  **Build HTML Documentation**:

    .. code-block:: bash

       cd docs
       make html

The built documentation will be available in `docs/_build/html/`.

Writing Docstrings
------------------

We use NumPy-style docstrings for all Python code. Here's a template:

.. code-block:: python

   def my_function(param1, param2):
       """Short description of the function.

       A more detailed description that can span multiple lines and provide
       more context about what the function does.

       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type
           Description of param2

       Returns
       -------
       type
           Description of return value

       Notes
       -----
       Any additional notes or mathematical explanations.

       Examples
       --------
       >>> result = my_function(1, 2)
       >>> print(result)
       3
       """

Example: Class Docstring
~~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of a class docstring:

.. code-block:: python

   class MyClass:
       """A brief description of the class.

       A more detailed description that explains what this class is for
       and how it should be used.

       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type, optional
           Description of param2 (default is None)

       Attributes
       ----------
       attr1 : type
           Description of attr1
       attr2 : type
           Description of attr2

       Notes
       -----
       Any additional notes about the implementation or mathematical
       background.

       Examples
       --------
       >>> obj = MyClass(1, 2)
       >>> obj.some_method()
       """

Mathematical Formulas
--------------------

When documenting mathematical concepts, use LaTeX math notation within reST's math directives:

.. code-block:: rst

   The distance between two fuzzy numbers is calculated as:

   .. math::

      d(A, B) = \sqrt{\frac{1}{2}((\mu_A - \mu_B)^2 + (\nu_A - \nu_B)^2)}

   where :math:`\mu_A` and :math:`\nu_A` are the membership and
   non-membership degrees of fuzzy number A.

Cross-Referencing
----------------

Use Sphinx's cross-referencing features to link between different parts of the documentation:

*   **Reference a Section**: `:ref:`section-label``
*   **Reference a Module**: `:mod:`module_name``
*   **Reference a Class**: `:class:`class_name``
*   **Reference a Function**: `:func:`function_name``

Example:

.. code-block:: rst

   See :ref:`user_guide_membership_functions` for more information about
   membership functions, or :class:`axisfuzzy.Fuzznum` for the API reference.

Documentation Style Guide
------------------------

1.  **Be Clear and Concise**: Write in clear, simple language. Avoid jargon unless necessary.

2.  **Use Active Voice**: Prefer "The function returns..." over "A value is returned..."

3.  **Include Examples**: Provide practical examples, especially for complex features.

4.  **Document Exceptions**: List any exceptions that might be raised.

5.  **Keep It Updated**: Update documentation when you change code.

6.  **Use Type Hints**: Include Python type hints in function signatures.

7.  **Include Mathematical Background**: When relevant, include the mathematical formulas and theory behind the implementation.

Continuous Documentation
-----------------------

Documentation is built automatically as part of our CI pipeline. Pull requests that affect documentation will show a preview of the changes.

Remember that good documentation is as important as good code. Take the time to write clear, comprehensive documentation that will help others understand and use your contributions effectively.