#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 18:54
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import os
import pkgutil
import importlib

# Dynamically import all submodules in this package。
# This ensures that any @extension decorator in the module can be automatically registered.
# Without the need for explicit importing.

# Get the path and name of the current package
package_path = os.path.dirname(__file__)
package_name = __name__

for _, module_name, _ in pkgutil.iter_modules([package_path]):
    # Construct the complete module path and import it.
    # The import operation will trigger the registration of any extended definitions specified in the module.
    importlib.import_module(f'.{module_name}', package=package_name)
