# FuzzLab Fuzzification System Guide

This document describes the FuzzLab fuzzification system (`fuzzify`) and the membership subsystem (`membership`): architecture, core APIs, extensibility, performance notes, and usage examples.

Target branch: 2025-08-14 and later

---

## 1. Module Overview

FuzzLab provides a unified entry to convert crisp values into fuzzy numbers/arrays: the fuzzification system. It builds on:

- Membership functions: x → membership degree md ∈ `[0, 1]`
- Fuzzification strategies: md (+ strategy params) → target fuzzy-number components
- High-performance batch path using a Struct-of-Arrays (SoA) backend for Fuzzarray

Current built-in mtype: `qrofn` (q-rung orthopair fuzzy number)

Related files:
- fuzzlab/fuzzify/
  - `base.py`: abstract strategy base class FuzzificationStrategy
  - `fuzzifier.py`: scheduler Fuzzifier and convenience function fuzzify
  - `registry.py`: strategy registry and decorator
- fuzzlab/membership/
  - `base.py`: membership base class MembershipFunction
  - `function.py`: built-ins (`TriangularMF`, `GaussianMF`, ...)
  - `factory.py`: factory and alias resolution for membership functions
- fuzzlab/fuzztype/qrofs/fuzzify.py: `QROFNFuzzificationStrategy` (default for `qrofn`)

---

## 2. Design & Responsibilities

- `Fuzzifier` (scheduler)
  - Separates “configuration” and “execution”:
    - Configure once: choose `mtype`/`method`, membership function, and strategy params
    - Execute many times: `__call__` with data, return `Fuzznum` or `Fuzzarray`
  - Uses Python introspection (`inspect`) to split kwargs into strategy-init vs membership-init params

- `FuzzificationStrategy` (abstract base)
  - Unified interface: `fuzzify_scalar(x, mf)` and `fuzzify_array(x, mf)`
  - Stores strategy params in `__init__(q: Optional[int] = None, **kwargs)` as `self.q` and `self.kwargs`
  - `get_strategy_info()` exposes meta info for logging/debug

- `Registry`
  - Maps `(mtype, method)` → StrategyClass
  - Default `method` per `mtype` supported
  - Decorator` @register_fuzzification_strategy(mtype, method, is_default=False)`

- `Membership`
  - `MembershipFunction`: `compute(x)` → `[0, 1]`
  - Built-ins: `TriangularMF`, `TrapezoidalMF`, `GaussianMF`, `SMF`, `ZMF`, `GeneralizedBellMF`, `PiMF`, `DoubleGaussianMF`, `SigmoidMF`
  - Factory helpers: `create_mf(name, **kwargs)`, `get_mf_class(name)`
  - Aliases: trimf, trapmf, gaussmf, smf, zmf, gbellmf, pimf, gauss2mf, sigmoid

- `QROFNFuzzificationStrategy` (default for `qrofn`)
  - `method='default'`; parameters: `q` (int, default 1), `pi` (hesitation, required)
  - Constraint: $md^q + nmd^q + pi^q ≤ 1$
  - Formula: $nmd = (1 − md^q − pi^q)^{(1/q)}$
  - Vectorized scalar/array paths; array path constructs `Fuzzarray` via SoA backend

---

## 3. Quick Start

Use `fuzzify` for one-off conversions; use `Fuzzifier` if you’ll reuse the same configuration.

Example 1: Single value → `qrofn` (default strategy, triangular MF)

````python
from fuzzlab.fuzzify import fuzzify

x = 0.7
fz = fuzzify(
    x=x,
    mf='trimf',
    mtype='qrofn',
    q=2,          # strategy param
    pi=0.2,       # strategy param (required)
    a=0.0, b=0.8, c=1.0  # membership params
)
print(fz.get_info())
````

Example 2: Batch → `Fuzzarray` (Gaussian MF)

````python
import numpy as np
from fuzzlab.fuzzify import fuzzify

X = np.array([10, 25, 40], dtype=float)
fa = fuzzify(
    x=X,
    mf='gaussmf',
    mtype='qrofn',
    q=3,
    pi=0.1,
    sigma=5.0, c=25.0
)
print(fa.shape, fa.mtype)
````

Example 3: Reusable engine Fuzzifier

````python
from fuzzlab.fuzzify import Fuzzifier
from fuzzlab.membership import GaussianMF

mf = GaussianMF(sigma=4.0, c=20.0)
engine = Fuzzifier(
    mf=mf,
    mtype='qrofn',
    q=3,
    pi=0.15
)
print(engine(18.0))           # -> Fuzznum
print(engine([15.0, 20.0]))   # -> Fuzzarray
````

Notes:
- If you pass an already-instantiated membership function, do not pass its constructor params again; a clear error will be raised.
- Unknown params (neither in strategy `__init__` nor in MF `__init__`) raise an error early.

---

## 4. API Reference

### 4.1 Fuzzifier (scheduler)

Constructor:
- `Fuzzifier(mf, mtype: Optional[str] = None, method: Optional[str] = None, **kwargs)`
  - `mf`: MembershipFunction instance or its name ('trimf', 'gaussmf', ...)
  - `mtype`: target fuzzy number type (e.g., `'qrofn'`); defaults to `get_config().DEFAULT_MTYPE`
  - `method`: strategy name; defaults to the mtype’s registered default
  - `kwargs`: automatically split via introspection into strategy-init and MF-init params

Call:
- `__call__(x)` -> `Fuzznum | Fuzzarray`
  - x: scalar (int/float) or array (list/np.ndarray)

Behavior:
- Scalar goes to `fuzzify_scalar`; arrays to `fuzzify_array`
- Array path is vectorized and builds a SoA backend-backed Fuzzarray

### 4.2 `fuzzify` (convenience function)

- `fuzzify(x, mf, mtype=None, method=None, **kwargs) -> Fuzznum | Fuzzarray`
  - Equivalent to: return `Fuzzifier(mf, mtype, method, **kwargs)(x)`

### 4.3 `FuzzificationStrategy` (abstract base)

Signature:
- `__init__(q: Optional[int] = None, **kwargs)`
  - `self.q` for q-rung family; `self.kwargs` for strategy-specific params (e.g., pi)
- `fuzzify_scalar(x, mf)` -> `Fuzznum` (abstract)
- `fuzzify_array(x, mf)` -> `Fuzzarray` (abstract)
- `get_strategy_info()` -> `dict`

Implementation tips:
- Vectorize the array path; avoid Python for-loops
- Build SoA backend via get_fuzznum_registry() and from_arrays if applicable

### 4.4 `Registry`

- `get_fuzzification_registry()` -> `FuzzificationRegistry`
- `@register_fuzzification_strategy(mtype, method, is_default=False)`
- `FuzzificationRegistry.register(...)`
- `FuzzificationRegistry.get_strategy(mtype, method=None)`
- `FuzzificationRegistry.get_default_method(mtype)`
- `FuzzificationRegistry.get_available_mtypes()`
- `FuzzificationRegistry.get_available_methods(mtype)`
- `FuzzificationRegistry.get_registry_info()`

### 4.5 `Membership`

- `MembershipFunction`:
  - `compute(x)` -> `[0, 1]`
  - `set_parameters(...)`, `get_parameters()`
- `Factory`:
  - `get_mf_class(name)`, `create_mf(name, **kwargs)`
- Built-ins (aliases):
  - `TriangularMF` (trimf), `TrapezoidalMF` (trapmf), `GaussianMF` (gaussmf)
  - `SMF` (smf), `ZMF` (zmf), `GeneralizedBellMF` (gbellmf)
  - `PiMF` (pimf), `DoubleGaussianMF` (gauss2mf), `SigmoidMF` (sigmoid)

---

## 5. qrofn Default Strategy: `QROFNFuzzificationStrategy`

Location: fuzzlab/fuzztype/qrofs/fuzzify.py  
Registration: `@register_fuzzification_strategy('qrofn', 'default')`

Parameters:
- `q: int = 1` (optional)
- `pi: float` (required, hesitation factor)

Math:
- Constraint: $md^q + nmd^q + pi^q ≤ 1$
- Conversion: `md = mf.compute(x)`; $nmd = (1 − md^q − pi^q)^{(1/q)}$

Implementation:
- Scalar path returns Fuzznum
- Array path computes `md`/`nmd` in a vectorized way and returns a SoA-backed Fuzzarray

Stability:
- Validate presence of pi; clip md, pi to `[0, 1]`
- Apply `np.maximum(1 − md^q − pi^q, 0.0)` to suppress tiny negative round-off

---

## 6. Extensibility

### 6.1 Add a new strategy (for the same mtype)

````python
from fuzzlab.fuzzify import FuzzificationStrategy, register_fuzzification_strategy
from fuzzlab.membership import MembershipFunction
from fuzzlab.core import Fuzznum, Fuzzarray, get_fuzznum_registry
import numpy as np

@register_fuzzification_strategy('qrofn', 'my_method')
class QROFNMyStrategy(FuzzificationStrategy):
    def __init__(self, q: int = 2, alpha: float = 0.1):
        super().__init__(q=q, alpha=alpha)

    def fuzzify_scalar(self, x: float, mf: MembershipFunction) -> Fuzznum:
        md = mf.compute(x)
        pi = np.clip(self.kwargs['alpha'] * (1.0 - md), 0.0, 1.0)
        nmd = np.maximum(1.0 - md**self.q - pi**self.q, 0.0)**(1.0/self.q)
        return Fuzznum(mtype='qrofn', q=self.q).create(md=float(md), nmd=float(nmd))

    def fuzzify_array(self, x: np.ndarray, mf: MembershipFunction) -> Fuzzarray:
        md = mf.compute(x)
        pi = np.clip(self.kwargs['alpha'] * (1.0 - md), 0.0, 1.0)
        nmd = np.maximum(1.0 - md**self.q - pi**self.q, 0.0)**(1.0/self.q)

        backend_cls = get_fuzznum_registry().get_backend('qrofn')
        backend = backend_cls.from_arrays(md=md, nmd=nmd, q=self.q)
        return Fuzzarray(backend=backend, mtype='qrofn', q=self.q)
````

Usage:
````python
from fuzzlab.fuzzify import Fuzzifier
fzr = Fuzzifier(mf='gaussmf', mtype='qrofn', method='my_method',
                q=2, alpha=0.2, sigma=3.0, c=10.0)
y = fzr([9.0, 10.0, 11.0])
````

### 6.2 Add a new mtype

- Implement your mtype under fuzzlab/fuzzy/<your_mtype>/:
  - `backend.py` (SoA backend)
  - `fuzzify.py` with strategy implementations; register them via the decorator
- Register the mtype in the core registry if needed
- Reuse the Fuzzifier config/dispatch model

### 6.3 Add a new membership function

````python
from fuzzlab.membership import MembershipFunction
import numpy as np

class MyMF(MembershipFunction):
    def __init__(self, p: float = 2.0, name: str = None):
        super().__init__(name)
        self.p = p
        self.parameters = {'p': p}

    def compute(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip(1.0 - np.abs(x)/self.p, 0.0, 1.0)

    def set_parameters(self, **kwargs):
        if 'p' in kwargs:
            self.p = kwargs['p']
            self.parameters['p'] = self.p
````

- To enable string alias creation, add it in `membership/factory.py`.

---

## 7. Performance & Best Practices

- Prefer the array path (vectorization + SoA) over Python loops
- Membership compute should accept np.ndarray and return same-shaped arrays
- Avoid temporary Python objects in hot paths; favor vector expressions
- Choose `q`, `pi` carefully to avoid numerical overflow/underflow
- Keep conditionals minimal in vectorized code

---

## 8. Troubleshooting

- Q: I passed `pi` but still get “missing pi”?
  - A: Ensure `pi` is a strategy parameter, not a membership parameter. Fuzzifier splits kwargs by inspecting strategy/MF `__init__` signatures. Strategy params must match the strategy’s `__init__`.
- Q: Can I pass a constructed MF and its constructor params together?
  - A: No. Either pass an instance (no constructor params), or pass a name and constructor params.
- Q: Unknown parameter 'xxx'?
  - A: It doesn’t belong to the strategy or the MF constructor. Fix its name or ownership.
- Q: Can I change params at `__call__`?
  - A: No by design. Configuration and execution are separated for predictability and reuse.

---

## 9. Relation to Core Structures

- `Fuzznum`: facade for a single fuzzy number; created on scalar path
- `Fuzzarray`: SoA-based batch container; constructed on array path
- `FuzznumRegistry`/`Backend`: array path typically builds a backend via registry APIs

---

## 10. Roadmap

- Linguistic variables & terms
- More mtypes and strategies (e.g., ivfn, pfs)
- Richer membership library and parameter validators
- Benchmarks and numerical stability test suites

---

## 11. Full Examples

````python
import numpy as np
from fuzzlab.fuzzify import Fuzzifier, fuzzify
from fuzzlab.membership import TriangularMF

# 1) One-shot convenience
fz = fuzzify(x=0.4, mf='trimf', mtype='qrofn',
             q=2, pi=0.25, a=0.0, b=0.5, c=1.0)
print("Single:", fz.get_info())

# 2) Reusable engine
engine = Fuzzifier(mf='gaussmf', mtype='qrofn',
                   q=3, pi=0.1, sigma=5.0, c=25.0)
arr = np.array([10.0, 25.0, 40.0], dtype=float)
fa = engine(arr)
print("Array:", fa.shape, fa.mtype)

# 3) Custom MF instance
mf_ins = TriangularMF(a=0.0, b=1.0, c=2.0)
engine2 = Fuzzifier(mf=mf_ins, mtype='qrofn', q=2, pi=0.2)
print(engine2(1.3))
````
