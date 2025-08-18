#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import os
import tomllib


def get_version():
    """
    Retrieve the version from the pyproject.toml file.

    `from axisfuzzy.version import __version__`
    """
    pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    with open(pyproject_path, 'rb') as f:
        data = tomllib.load(f)
    return data['project']['version']


__version__ = get_version()
