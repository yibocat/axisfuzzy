#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Type, Optional, List, Union

import numpy as np

from .base import BaseRandomGenerator
from .registry import RandomGeneratorRegistry


def registry() -> RandomGeneratorRegistry: ...
def register_random(cls: Type[BaseRandomGenerator]) -> Type[BaseRandomGenerator]: ...
def unregister(mtype: str) -> bool: ...
def get_generator(mtype: str) -> Optional[BaseRandomGenerator]: ...
def list_registered() -> List[str]: ...
def is_registered(mtype: str) -> bool: ...


from .api import (
    rand as rand,
    choice as choice,
    uniform as uniform,
    normal as normal,
    beta as beta
)

def get_seed() -> Union[int, np.random.SeedSequence, np.random.BitGenerator, None]: ...
def set_seed(seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...) -> None: ...
def get_rng() -> np.random.Generator: ...
def spawn_rng() -> np.random.Generator: ...

__all__: List[str]
