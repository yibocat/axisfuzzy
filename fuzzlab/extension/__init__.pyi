#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 16:41
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from typing import Any, Callable

# Import the class types for accurate return type hinting of the getter functions.
from .registry import ExtensionRegistry
from .dispatcher import ExtensionDispatcher
from .injector import ExtensionInjector

# Re-export the public decorators from the decorator submodule.
# Their full signatures are defined in `decorator.pyi`.
from .decorator import extension as extension
from .decorator import batch_extension as batch_extension

# Define the signatures for the public functions provided by this package.

def get_extension_registry() -> ExtensionRegistry: ...

def get_extension_dispatcher() -> ExtensionDispatcher: ...

def get_extension_injector() -> ExtensionInjector: ...

def apply_extensions() -> None: ...
