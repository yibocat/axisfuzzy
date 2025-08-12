# FuzzLab Random Fuzzy Number Generation System

> Version: Draft 2025-08-12  
> Scope: Current master (SoA Backend architecture)  
> Language: English (see `random_zh.md` for Chinese)

---
## Table of Contents
1. Goals & Principles  
2. Layered Architecture  
3. Core Concepts  
4. Quick Start  
5. API Reference  
6. qrofn Parameter Details  
7. Performance Techniques  
8. Extending with a New Generator  
9. Design Principles Recap  
10. FAQ  
11. Example Gallery  
12. Future Enhancements (Roadmap)

---
## 1. Goals & Principles
The random subsystem provides a unified yet extensible way to generate fuzzy numbers:
- **High performance**: Struct-of-Arrays (SoA) backend with direct vectorized filling.
- **Extensibility**: Per-mtype specialization via a registry.
- **Reproducibility**: Global seed + optional per-call overrides.
- **Parameterized control**: Distributions, modes, structural vs procedural parameters.
- **Single entry point**: `random_fuzz` / alias `rand` auto-selects scalar vs batch.
- **Minimal ceremony**: One class + one registration call for new types.

---
## 2. Layered Architecture
| Layer | File | Responsibility |
|-------|------|----------------|
| Seed State | `fuzzlab/random/seed.py` | Global RNG lifecycle (set / get / spawn) |
| Registry | `fuzzlab/random/registry.py` | mtype → generator instance mapping |
| Abstraction | `fuzzlab/random/base.py` | Interfaces: `fuzznum` & `fuzzarray` |
| Public API | `fuzzlab/random/api.py` + `__init__.py` | Factory & helpers |
| Type Logic | `fuzzlab/fuzzy/<mtype>/random.py` | Concrete generation (e.g. qrofn) |
| Backend | `fuzzlab/fuzzy/<mtype>/backend.py` | SoA component arrays |

Batch path: `rand()` → resolve RNG → lookup generator → `generator.fuzzarray()` → build backend → return Fuzzarray.

---
## 3. Core Concepts
| Term | Meaning |
|------|---------|
| mtype | Fuzzy number type identifier (e.g. `qrofn`) |
| Structural parameter | Defines intrinsic fuzzy form (e.g. `q`) |
| Procedural parameter | Controls sampling strategy (e.g. `md_dist`) |
| Backend | SoA storage of component arrays for batch instances |
| Fuzznum | Single fuzzy number façade object |
| Fuzzarray | High-dimensional container over a backend |

---
## 4. Quick Start
```python
import fuzzlab.random as fr

fr.set_seed(42)
num = fr.rand('qrofn', q=3)
arr = fr.rand('qrofn', shape=(512,), q=4, md_dist='beta', a=2, b=5)
mat = fr.rand('qrofn', shape=(128, 256), q=5, nu_mode='independent')
sample = fr.choice(arr, size=16, replace=False)
```

---
## 5. API Reference
| Function | Purpose | Notes |
|----------|---------|-------|
| `set_seed(seed)` | Set global seed | Deterministic reproducibility |
| `rand(mtype, shape=None, q=..., **params)` | Generate Fuzznum / Fuzzarray | `shape=None` → scalar |
| `random_fuzz` | Alias of `rand` | Same signature |
| `choice(fuzzarray, size, replace=True, p=None)` | Sample from 1-D fuzzy array | Returns new Fuzzarray |
| `uniform / normal / beta` | Numeric helper distributions | Share global RNG rules |
| `list_mtypes()` | List registered mtypes | Introspection |
| `register(mtype, generator)` | Add custom generator | Call at module import |
| `get_generator(mtype)` | Retrieve generator instance | Advanced usage |

RNG resolution priority: `rng` arg > `seed` arg > global seed.

---
## 6. qrofn Parameter Details
| Name | Type | Default | Description |
|------|------|---------|-------------|
| q | int | (explicit / 2) | q-rung structural parameter |
| md_dist | str | `uniform` | Membership distribution: uniform / beta / normal |
| md_low / md_high | float | 0.0 / 1.0 | Membership range |
| a / b | float | 2.0 / 2.0 | Beta shape parameters (shared) |
| loc / scale | float | 0.5 / 0.15 | Normal mean & std |
| nu_mode | str | `orthopair` | `orthopair` or `independent` |
| nu_dist | str | `uniform` | Non-membership distribution |
| nu_low / nu_high | float | 0.0 / 1.0 | Non-membership range |

Constraint handling:
- `orthopair`: dynamic ceiling `(1 - md^q)^(1/q)` then scaled sampling.
- `independent`: independent sampling + mask clamp for violations.

---
## 7. Performance Techniques
| Technique | Benefit |
|-----------|---------|
| Direct backend fill | Avoid per-object instantiation overhead |
| Vectorized sampling | Single-pass generation of large batches |
| Dynamic upper bound arrays | Eliminates Python loops |
| Masked constraint correction | Fast in-place adjustment |
| Split scalar/batch paths | Prevent unnecessary reshaping |

---
## 8. Extending with a New Generator
Steps:
1. File: `fuzzlab/fuzzy/<mtype>/random.py`
2. Subclass `ParameterizedRandomGenerator`
3. Implement: `mtype`, `get_default_parameters`, `validate_parameters`, `fuzznum`, `fuzzarray`
4. Register:  
```python
from ...random import register
register('your_mtype', YourGenerator())
```
5. Use:  
```python
fr.rand('your_mtype', shape=1024, ...)
```

Batch recommendation: fill backend arrays directly (avoid interim Fuzznum list).

---
## 9. Design Principles Recap
| Principle | Realization |
|-----------|-------------|
| Separation of concerns | Seed / registry / abstraction / specialization |
| Simplicity of extension | One class + one registration |
| High performance | SoA + vectorization + direct memory layout |
| Determinism | Layered RNG resolution |
| Maintainability | Structural vs procedural clarity |

---
## 10. FAQ
| Question | Answer |
|----------|--------|
| Why is `q` not in default params dict? | It's structural; must be explicit for clarity |
| Does batch creation allocate many objects? | No, only backend arrays |
| How to reproduce results? | Call `set_seed` once; avoid mixed local seeds unless intended |
| Different distributions for md and nmd? | Currently shared; subclass to diverge |
| Parallel usage? | Create separate `np.random.default_rng(sub_seed)` and pass via `rng=` |

---
## 11. Example Gallery
Single:
```python
num = fr.rand('qrofn', q=5, md_dist='normal', loc=0.55, scale=0.08)
```
Batch:
```python
grid = fr.rand('qrofn', shape=(64, 64), q=3, nu_mode='independent')
```
Sampling:
```python
flat = grid.reshape(-1)
picked = fr.choice(flat, size=100, replace=False)
```
Distribution helper:
```python
vals = fr.beta(2, 5, shape=2000)
```

---
## 12. Future Enhancements (Roadmap)
| Area | Idea |
|------|------|
| Separate md/nmd params | Independent a,b / loc,scale sets |
| More distributions | Triangular, gamma, user-defined kernels |
| Streaming generation | Chunked memory-friendly batches |
| Diagnostics | Optional statistical summary hooks |
| Constraint DSL | Composable constraint expressions |

---
**Feedback**: File issues or submit PRs with new generators.
