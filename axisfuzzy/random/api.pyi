#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 17:12
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from typing import Any, Optional, Sequence, Tuple, Union, List, Type, overload
import numpy as np

from ..core import Fuzznum, Fuzzarray
from .base import BaseRandomGenerator
from .registry import RandomGeneratorRegistry

# --- Seed Management ---
# def get_seed() -> Union[int, np.random.SeedSequence, np.random.BitGenerator, None]: ...
# def set_seed(seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...) -> None: ...
# def get_rng() -> np.random.Generator: ...
# def spawn_rng() -> np.random.Generator: ...

# --- Core Generation API ---
@overload
def random_fuzz(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: None = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzznum: ...

@overload
def random_fuzz(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: Union[int, Tuple[int, ...]] = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzzarray: ...

def choice(
    obj: Fuzzarray,
    size: Optional[Union[int, Tuple[int, ...]]] = ...,
    replace: bool = ...,
    p: Optional[Sequence[float]] = ...,
    seed: Optional[int] = ...,
    rng: Optional[np.random.Generator] = ...
) -> Fuzzarray: ...

# --- Distribution Helpers ---
def uniform(
    low: float = ...,
    high: float = ...,
    shape: Optional[Union[int, Tuple[int, ...]]] = ...,
    seed: Optional[int] = ...,
    rng: Optional[np.random.Generator] = ...
) -> Union[float, np.ndarray]: ...

def normal(
    loc: float = ...,
    scale: float = ...,
    shape: Optional[Union[int, Tuple[int, ...]]] = ...,
    seed: Optional[int] = ...,
    rng: Optional[np.random.Generator] = ...
) -> Union[float, np.ndarray]: ...

def beta(
    a: float,
    b: float,
    shape: Optional[Union[int, Tuple[int, ...]]] = ...,
    seed: Optional[int] = ...,
    rng: Optional[np.random.Generator] = ...
) -> Union[float, np.ndarray]: ...

# --- Registry Management API ---
def registry() -> RandomGeneratorRegistry: ...
def register(mtype: str, generator: BaseRandomGenerator) -> None: ...
def unregister(mtype: str) -> bool: ...
def get_generator(mtype: str) -> Optional[BaseRandomGenerator]: ...
def list_registered() -> List[str]: ...
def is_registered(mtype: str) -> bool: ...

# --- Alias ---
# Remove the old alias assignment:
# rand = random_fuzz

# Explicitly define the alias 'rand' with the full signature of 'random_fuzz'
@overload
def rand(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: None = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzznum: ...

@overload
def rand(
    mtype: Optional[str] = ...,
    q: int = ...,
    shape: Union[int, Tuple[int, ...]] = ...,
    seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = ...,
    rng: Optional[np.random.Generator] = ...,
    **params: Any
) -> Fuzzarray: ...
