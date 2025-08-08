#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 18:54
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import os
import pkgutil
import importlib

# 动态导入此包中的所有子模块。
# 这可以确保模块中的任何 @extension 装饰器都能被自动注册，
# 而无需显式导入。

# 获取当前包的路径和名称
package_path = os.path.dirname(__file__)
package_name = __name__

for _, module_name, _ in pkgutil.iter_modules([package_path]):
    # 构建完整的模块路径并导入它。
    # 导入操作会触发模块中定义的任何扩展的注册。
    importlib.import_module(f'.{module_name}', package=package_name)
