#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import os
import tomllib
from importlib.metadata import version, PackageNotFoundError


try:
    # This will work when the package is installed (the standard way).
    __version__ = version("axisfuzzy")
except PackageNotFoundError:
    # Fallback for development mode when the package is not installed.
    # This reads the version directly from pyproject.toml.
    pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
        __version__ = data['project']['version']
    except FileNotFoundError:
        # If pyproject.toml is not found, we can't determine the version.
        __version__ = "unknown"
