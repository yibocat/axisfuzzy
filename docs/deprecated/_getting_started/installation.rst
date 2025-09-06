.. _getting_started_installation:

************
Installation
************

This guide will walk you through the steps to install AxisFuzzy and set up your development environment.

Prerequisites
=============

Before you begin, ensure you have the following installed:

- Python 3.12 or higher
- pip (Python package installer)

Installation
============

You can install AxisFuzzy using pip:

.. code-block:: bash

    pip install axisfuzzy

To install the full set of optional dependencies for analysis, development, and documentation, you can specify the extras:

.. code-block:: bash

    pip install axisfuzzy[all]

This will install all the necessary libraries, including:

- **Core**: numpy, numba
- **Analysis**: pandas, matplotlib, networkx
- **Development**: pytest, notebook
- **Documentation**: sphinx and related extensions

Development Environment
=======================

For developers who want to contribute to AxisFuzzy, it is recommended to set up a virtual environment to manage dependencies.

1. **Create a virtual environment**:

   .. code-block:: bash

       python -m venv .venv

2. **Activate the virtual environment**:

   - On macOS and Linux:

     .. code-block:: bash

         source .venv/bin/activate

   - On Windows:

     .. code-block:: bash

         .venv\Scripts\activate

3. **Install dependencies**:

   Install the required dependencies from the `requirements` directory:

   .. code-block:: bash

       pip install -r requirements/all_requirements.txt

By following these steps, you will have a fully functional environment for both using and developing AxisFuzzy.