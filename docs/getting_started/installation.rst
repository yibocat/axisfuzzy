.. _installation:

Installation
============

AxisFuzzy is available on the Python Package Index (PyPI) and can be installed using standard package managers like ``pip`` or its high-performance alternative, ``uv``. We strongly recommend using a virtual environment (e.g., ``venv`` or ``conda``) to manage dependencies and avoid conflicts with system-wide packages.

Dependency Philosophy
---------------------

AxisFuzzy is designed with a modular dependency system to keep the core library lightweight and flexible. The base installation includes only the essential packages required for fuzzy computation. Optional functionalities, such as data analysis and extension tools, are available as "extras" that can be installed on demand.

Standard Installation with ``pip``
----------------------------------

You can install AxisFuzzy directly from PyPI.

*   **Core Installation**: For the core required functionality:
    Installs the essential components for fuzzy sets and computation.

    .. code-block:: bash

        pip install axisfuzzy


    *   **Dependencies**: ``numpy``, ``numba``


*   **Installation with Analysis Tools**
    Includes the core library plus tools for data analysis and visualization. Ideal for users who want to integrate AxisFuzzy with data science workflows.
    
    .. code-block:: bash

        pip install axisfuzzy[analysis]

    *   **Dependencies**: Core + ``pandas``, ``matplotlib``, ``networkx``, ``pydot``



*   **Full Installation**
    Installs the core library along with all optional dependencies for analysis and development. This provides the most complete user experience.
    
    .. code-block:: bash

        pip install axisfuzzy[all]

    *   **Dependencies**: Includes all dependencies from ``core`` and ``analysis``, and additional ``notebook`` and ``pytest``.

Using ``uv`` for Faster Installation
------------------------------------

For a significantly faster installation experience, you can use ``uv``, a modern Python package installer. The commands are analogous to ``pip``:

*   **Core Installation**:

    .. code-block:: bash

        uv pip install axisfuzzy

*   **Full Installation**:

    .. code-block:: bash

        uv pip install axisfuzzy[all]

You can use ``uv`` for any of the installation options described above (e.g., ``uv pip install axisfuzzy[analysis]``).


Installing from Source
----------------------

To get the latest development version, you can install AxisFuzzy directly from the GitHub repository. This is the recommended approach for contributors.

1.  Clone the repository:

    .. code-block:: bash

        git clone https://github.com/yibocat/axisfuzzy.git

2.  Navigate into the project directory:

    .. code-block:: bash

        cd axisfuzzy

3.  Install in editable mode. For a complete development environment, we recommend installing with the ``all`` extra, which bundles dependencies for analysis and development:

    .. code-block:: bash

        pip install -e .[all]


Development Environment
-----------------------

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