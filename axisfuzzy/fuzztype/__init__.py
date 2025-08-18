#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzztype Package Initializer.

This module dynamically discovers and imports all available fuzzy number type
subpackages within this directory (e.g., 'qrofs', 'ivfs').

This ensures that all mtype-specific components (strategies, backends,
operations, extensions, etc.) are automatically registered with their respective
systems when the `axisfuzzy` library is imported. This makes the entire
architecture pluggable.
"""

import os
import importlib

# Dynamically discover and import all subpackages to trigger their registrations.
# A subpackage is any directory in the current path that is not a __pycache__
# and contains an __init__.py file.
for module_name in os.listdir(os.path.dirname(__file__)):
    module_path = os.path.join(os.path.dirname(__file__), module_name)
    if (os.path.isdir(module_path) and
            '__pycache__' not in module_name and
            os.path.exists(os.path.join(module_path, '__init__.py'))):
        # The import triggers the __init__.py within the subpackage (e.g., qrofs),
        # which in turn should import its own components to register them.
        importlib.import_module(f'.{module_name}', __name__)

# This package is primarily for loading and does not export any symbols itself.
# The individual mtype implementations are accessed via their own submodules.
__all__ = []
