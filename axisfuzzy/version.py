#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import os
import tomllib


# 从pyproject.toml文件中获取版本号
# TODO: 这段代码问题很大, 不应该从外部获取版本号.
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
