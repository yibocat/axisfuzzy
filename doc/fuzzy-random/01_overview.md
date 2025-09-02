# `axisfuzzy.random` 随机化系统：概述

欢迎来到 `axisfuzzy` 的随机化系统。在模糊集理论和应用中，生成具有特定分布和约束的随机模糊数是一项基础且关键的任务。无论是用于蒙特卡洛模拟、算法初始化、性能测试还是数据增强，一个强大、灵活且可复现的随机数生成系统都至关重要。

`axisfuzzy.random` 正是为此而生。它是一个高度模块化、可扩展且性能卓越的系统，旨在为 `axisfuzzy` 支持的各种模糊数类型（`mtype`）提供统一、可控的随机生成能力。
## 1. 设计理念

`axisfuzzy.random` 的设计遵循以下四大核心理念：

- **模块化与可扩展性 (Modularity & Extensibility)** 系统采用“插件式”架构。每种新的模糊数类型（`mtype`）都可以定义并注册自己的随机生成器，而无需修改任何核心代码。这使得 `axisfuzzy` 可以轻松地支持未来出现的各种新模糊数，保持了系统的生命力和灵活性。

```python
@register_random  # 一个装饰器就完成注册
class MyRandomGenerator(ParameterizedRandomGenerator):
    mtype = "my_custom_type"
    # ... 实现具体逻辑
```

- **高性能 (High Performance)** 对于生成大规模的 `Fuzzarray`，性能至关重要。本系统通过**向量化操作**和**直接与 `Fuzzarray` 的 SoA (Struct of Arrays) 后端交互**来实现高性能。它避免了在 Python 层面逐个创建 `Fuzznum` 对象的低效循环，从而在处理大数据集时表现出色。

```python
# 生成 100,000 个模糊数，性能接近 NumPy
large_array = fr.rand('qrofn', shape=(100000,))
```

- **可复现性 (Reproducibility)** 科学计算的基石是可复现性。`axisfuzzy.random` 包含一个集中的随机种子管理系统，可以确保在相同的种子下，每次生成的随机模糊数序列完全一致。这对于调试、验证算法和重现实验结果至关重要。

```python
fr.set_seed(42)
result1 = fr.rand('qrofn', shape=(1000,), q=2)

fr.set_seed(42)  # 重置到相同种子
result2 = fr.rand('qrofn', shape=(1000,), q=2)
# result1 和 result2 完全相同
```

- **统一的 API (Unified API)** 尽管底层实现千差万别，但我们为用户提供了简单、一致的顶层 API，如 `axisfuzzy.random.rand()`。用户只需指定 `mtype` 和所需参数，系统会自动为你找到并调用最合适的生成器，大大降低了使用门槛。

```python
# 所有 mtype 都使用相同的接口
qrofn_num = fr.rand('qrofn', q=2)           # q-rung 正交模糊数
ivfn_num = fr.rand('ivfn')                  # 区间直觉模糊数  
qrohfn_num = fr.rand('qrohfn', q=3)         # q-rung 正交犹豫模糊数
```

## 2. 系统架构与核心组件

整个系统由四个核心文件组成，每个文件各司其职：

| 模块                | 作用                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------- |
| **`api.py`**      | 面向用户的统一入口（`rand`, `choice`, `uniform`, `normal`, `beta`），负责解析参数、选择生成器、返回结果。            |
| **`registry.py`** | 随机生成器注册表，管理所有已注册的 `mtype` 生成器，实现自动注册与查找。                                               |
| **`base.py`**     | 定义生成器抽象基类 `BaseRandomGenerator` 和带参数化工具的 `ParameterizedRandomGenerator`，规定了生成器必须实现的接口。 |
| **`seed.py`**     | 全局随机状态管理，提供 `set_seed`, `get_rng`, `spawn_rng`, `get_seed` 等方法，保证可复现性和线程安全。            |

系统架构图为

```text
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (User Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│  fr.rand()  │ fr.choice() │ fr.uniform() │ fr.set_seed() │ ...   │
└─────────────┬───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                       API 层 (api.py)                            │
│  • 统一入口函数                                                    │
│  • 参数解析与验证                                                  │  
│  • 类型分发                                                       │
└─────────────┬───────────────────────────────────────────────────┘
              │
    ┌─────────▼─────────┐           ┌─────────────────────┐
    │   注册表层          │           │    种子管理层        │
    │  (registry.py)    │           │    (seed.py)        │
    │                   │           │                     │
    │ • 生成器注册管理     │◄─────────►│ • 全局随机状态       │
    │ • mtype -> 生成器   │           │ • 随机数生成器       │
    │ • @register_random │           │ • 并行独立流         │
    └─────────┬─────────┘           └─────────────────────┘
              │                               │
┌─────────────▼───────────────────────────────▼───────────────────┐
│                      生成器层 (base.py)                          │
│                                                                 │
│  BaseRandomGenerator          ParameterizedRandomGenerator      │
│  ├─ get_default_parameters()  ├─ _merge_parameters()           │
│  ├─ validate_parameters()     ├─ _sample_from_distribution()   │ 
│  ├─ fuzznum()                 ├─ _validate_range()             │
│  └─ fuzzarray()               └─ ... 工具方法                   │
└─────────────┬───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                   具体实现层 (Concrete Implementations)           │
│                                                                 │
│  QROFNRandomGenerator    │  QROHFNRandomGenerator  │  ...       │
│  ├─ mtype = "qrofn"      │  ├─ mtype = "qrohfn"    │            │
│  ├─ 约束处理逻辑          │  ├─ 犹豫集生成逻辑       │            │
│  └─ 向量化优化           │  └─ 对象数组构建         │            │
└─────────────────────────────────────────────────────────────────┘

```

下面我们来逐一了解这四个核心组件：

- **1. API 层 (`api.py`)**
    - **角色**：用户与随机化系统的**主要交互入口**。
    - **职责**：提供如 `rand()`、`choice()` 等高层函数。它负责解析用户的请求（如 `mtype`、`shape`、`q` 等参数），并协调其他模块完成生成任务。
- **2. 注册表层 (`registry.py`)**
    - **角色**：系统的“**调度中心**”或“**电话簿**”。
    - **职责**：它维护一个从 `mtype` 字符串到具体生成器实例的映射。当你请求生成一个 `'qrofn'` 类型的模糊数时，正是注册表告诉系统应该使用 `QROFNRandomGenerator`。`@register_random` 装饰器使得新的生成器可以“自报家门”，实现自动注册。
- **3. 基类层 (`base.py`)**
    - **角色**：所有随机生成器的“**抽象蓝图**”。
    - **职责**：定义了所有随机生成器必须遵守的统一接口（`BaseRandomGenerator`），比如必须实现 `fuzznum()`（生成单个）和 `fuzzarray()`（批量生成）方法。它还提供了一个便利的子类 `ParameterizedRandomGenerator`，内置了参数管理和标准分布采样等实用工具，简化了新生成器的开发。
- **4. 种子管理层 (`seed.py`)**
    - **角色**：确保实验可复现的“**全局时钟**”。
    - **职责**：它管理着一个全局的 `numpy.random.Generator` 实例。所有随机生成操作都从这个实例获取随机数。通过 `set_seed()` 函数，你可以固定这个“时钟”的起点，从而保证每次运行代码都能得到完全相同的结果。它还支持通过 `spawn_rng()` 创建独立的随机流，用于并行计算等高级场景。

## 3. 数据流：一次 `rand` 调用的生命周期

为了更好地理解这些组件如何协同工作，让我们追踪一次典型的函数调用：

```python
import axisfuzzy.random as fr

# 用户调用
arr = fr.rand('qrofn', shape=(100,), q=3, md_dist='beta', a=2.0, b=3.0)
```

**Step 1: API 层处理**
用户调用 `rand()` 函数，传入参数 `mtype='qrofn'`, `shape=(100,)`, `q=23` 以及其他参数。API 层接收到这个请求。

```text
fr.rand() 接收参数:
├─ mtype='qrofn' 
├─ shape=(100,)
├─ q=3
└─ **params: md_dist='beta', a=2.0, b=3.0
```

**Step 2: 注册表查找**
API 层调用 `get_random_generator('qrofn')` 向注册表查询。注册表根据 `'qrofn'` 这个键，找到并返回早已注册好的 `QROFNRandomGenerator` 实例。

```text
RandomGeneratorRegistry.get_generator('qrofn')
└─ 返回: QROFNRandomGenerator 实例
```

**Step 3: 种子管理**
`api.py` 检查用户是否传入了局部的 `seed` 或 `rng` 对象。由于没有，它会调用 `seed.py` 中的 内部函数`_resolve_rng()` 函数，获取全局共享的随机数生成器实例。

```
_resolve_rng(seed=None, rng=None)
└─ 返回: 全局 numpy.random.Generator 实例
```

**Step 4: 生成器执行**
因为用户提供了 `shape` 参数，API 层会调用 `QROFNRandomGenerator` 实例的 `fuzzarray` 方法。这个方法是为高性能批量生成而优化的。

在 `fuzzarray` 方法内部，它会：
- 合并用户传入的参数（`q=2`）和生成器的默认参数。
- 调用从 `seed.py` 获取的全局 RNG，进行**向量化**采样。例如，一次性生成 10 个隶属度（`md`）和 10 个非隶属度（`nmd`），并确保它们满足 `md^q + nmd^q <= 1` 的约束。
- **构建 `Fuzzarray` 并返回** 
	`fuzzarray` 方法不会创建 10 个独立的 `Fuzznum` 对象，而是直接将生成的 `md` 和 `nmd` NumPy 数组填充到 `QROFNBackend` 中。最后，这个高效的后端被包装成一个 `Fuzzarray` 对象返回给用户。

```text
QROFNRandomGenerator.fuzzarray(rng, shape=(100,), q=3, md_dist='beta', a=2.0, b=3.0)
├─ 参数合并与验证
├─ 向量化采样 (md: beta 分布, nmd: 约束处理)
├─ 创建 QROFNBackend
└─ 返回: Fuzzarray 实例
```

通过这个流程，`axisfuzzy.random` 在保证易用性的同时，实现了极高的运行效率和强大的可扩展性。
