# 4. 注册一个随机生成器

本章是为 `mtype` 注册一个功能完备、高性能的随机生成器的实践指南。我们的目标是让你能够清晰地理解如何为你自己的模糊数类型实现 `axisfuzzy.random.rand('your_mtype', ...)` 功能。

我们将跳过高层理论，直接进入实现细节。核心任务是：创建一个继承自 `ParameterizedRandomGenerator` 的类，实现其接口，并使用 `@register_random` 装饰器进行注册。

## 4.1. 核心三步走

为任何 `mtype` 添加随机生成功能都遵循以下三个步骤：

1.  **创建 `random.py` 文件**: 在你的 `mtype` 包下 (例如 `axisfuzzy/fuzztype/your_mtype/`) 创建该文件。
2.  **实现生成器类**: 在文件中定义一个生成器类，该类继承自 `axisfuzzy.random.base.ParameterizedRandomGenerator`。
3.  **注册与导入**: 使用 `@register_random` 装饰器标记你的类，并确保该 `random.py` 文件被 `mtype` 包的 `__init__.py` 文件导入。

## 4.2. 生成器类的实现解剖

一个生成器类本质上是 `ParameterizedRandomGenerator` 的一个具体实现，它必须定义 `mtype` 属性并实现四个核心方法。

```python
# axisfuzzy/random/base.py
class ParameterizedRandomGenerator(BaseRandomGenerator, ABC):
    # ...
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def validate_parameters(self, **params) -> None: ...
        
    @abstractmethod
    def fuzznum(self, rng, **params) -> Fuzznum: ...
        
    @abstractmethod
    def fuzzarray(self, rng, shape, **params) -> Fuzzarray: ...
```

我们将通过两个实际例子来展示如何实现这些方法：
-   **`QROFNRandomGenerator`**: 一个**结构化数据**的简单例子，其隶属度(md)和非隶属度(nmd)都是标量。
-   **`QROHFNRandomGenerator`**: 一个**非结构化（Ragged）数据**的复杂例子，其 md 和 nmd 是长度可变的犹豫集。

### 步骤 1: 定义类与 `mtype`

这是最简单的一步。使用 `@register_random` 装饰器，并声明它服务的 `mtype`。

```python
# /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
from ...random import register_random
from ...random.base import ParameterizedRandomGenerator

@register_random
class QROFNRandomGenerator(ParameterizedRandomGenerator):
    mtype = "qrofn"
    # ... class implementation ...
```

### 步骤 2: `get_default_parameters()` - 定义可配置项

此方法返回一个字典，定义了所有用户可自定义的生成参数及其默认值。这极大地提高了生成器的灵活性。

**示例 (`qrofn`)**:
```python
# /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
def get_default_parameters(self) -> Dict[str, Any]:
    """Returns the default parameters for QROFN generation."""
    return {
        'md_dist': 'uniform',   # 隶属度分布
        'md_low': 0.0,
        'md_high': 1.0,
        'nu_mode': 'orthopair', # 非隶属度生成模式
        'nu_dist': 'uniform',   # 非隶属度分布
        'nu_low': 0.0,
        'nu_high': 1.0,
        'a': 2.0, 'b': 2.0,     # Beta 分布参数
        'loc': 0.5, 'scale': 0.15 # Normal 分布参数
    }
```
对于更复杂的类型如 `qrohfn`，你可能还需要定义犹豫集的长度控制参数：
```python
# /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrohfs/random.py
'md_count_dist': 'uniform_int',   # 'uniform_int' | 'poisson' | 'fixed'
'md_count_min': 1,
'md_count_max': 4,
```

### 步骤 3: `validate_parameters()` - 保证输入的正确性

在生成开始前，此方法用于检查所有参数（包括 `q` 等结构化参数和用户传入的 `**kwargs`）是否合法。`ParameterizedRandomGenerator` 提供了 `_validate_range` 等工具方法来简化验证。

**示例 (`qrofn`)**:
```python
# /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
def validate_parameters(self, q: int, **kwargs) -> None:
    """Validates parameters for QROFN generation."""
    if not isinstance(q, int) or q <= 0:
        raise ValueError(f"q must be a positive integer, but got {q}")

    if 'md_low' in kwargs and 'md_high' in kwargs and kwargs['md_low'] > kwargs['md_high']:
        raise ValueError("md_low cannot be greater than md_high")

    if 'nu_mode' in kwargs and kwargs['nu_mode'] not in ['orthopair', 'independent']:
        raise ValueError("nu_mode must be 'orthopair' or 'independent'")
```

### 步骤 4: `fuzzarray()` - 高性能批量生成的核心

这是最重要的部分，直接决定了生成器的性能。目标是：**尽可能向量化，并直接构建后端**。

#### 简单情况: 完全向量化 (`qrofn`)

当数据结构规整时（例如，每个模糊数由固定数量的标量组成），可以实现完全向量化。

**`QROFNRandomGenerator.fuzzarray` 的实现逻辑**:
1.  **合并与验证参数**: 调用 `self._merge_parameters(**params)` 和 `self.validate_parameters(q=q, **params)`。
2.  **批量采样 `mds`**: 使用 `_sample_from_distribution` 一次性生成所有隶属度。
    ```python
    # /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
    size = int(np.prod(shape))
    mds = self._sample_from_distribution(
        rng,
        size=size,
        dist=params['md_dist'],
        # ... other params
    )
    ```
3.  **批量生成 `nmds` 并处理约束**:
    -   **`orthopair` 模式**: `nmd` 的上限取决于对应的 `md`。这个计算也是向量化的。
        ```python
        # /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
        max_nmd = (1 - mds ** q) ** (1 / q)
        effective_high = np.minimum(params['nu_high'], max_nmd)
        # ... scale samples to the dynamic range ...
        ```
    -   **`independent` 模式**: 独立采样，然后向量化地修正所有违反约束 `md^q + nmd^q > 1` 的元素。
        ```python
        # /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
        violates_mask = (mds ** q + nmds ** q) > 1.0
        if np.any(violates_mask):
            max_nmd_violating = (1 - mds[violates_mask] ** q) ** (1 / q)
            nmds[violates_mask] = np.minimum(nmds[violates_mask], max_nmd_violating)
        ```
4.  **直接构建后端**: 这是性能的关键！不要创建 `Fuzznum` 列表，而是直接将 `mds` 和 `nmds` 数组传递给后端的 `from_arrays` 方法。
    ```python
    # /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
    backend = QROFNBackend.from_arrays(mds=mds.reshape(shape), nmds=nmds.reshape(shape), q=q)
    return Fuzzarray(backend=backend)
    ```

#### 复杂情况: 分组向量化 (`qrohfn`)

当数据结构是 ragged array 时（如犹豫集），无法用一个规整的矩阵表示。此时采用“分组向量化”策略。

**`QROHFNRandomGenerator.fuzzarray` 的实现逻辑**:
1.  **批量生成长度**: 首先，向量化地为每个元素生成其 `md` 和 `nmd` 犹豫集的长度。
    ```python
    # /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrohfs/random.py
    md_counts = self._sample_counts(rng, size, dist=p['md_count_dist'], ...)
    nmd_counts = self._sample_counts(rng, size, dist=p['nmd_count_dist'], ...)
    ```
2.  **按长度分组并采样**:
    -   找到所有唯一的长度值 `c`。
    -   对每个 `c`，找到所有长度为 `c` 的元素的索引 `idx`。
    -   为这 `len(idx)` 个元素，批量生成一个 `(len(idx), c)` 的值矩阵。
    ```python
    # /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrohfs/random.py
    unique_md_counts = np.unique(md_counts)
    for c in unique_md_counts:
        idx = np.where(md_counts == c)[0]
        # ...
        mat = self._sample_from_distribution(rng, size=(idx.size * c), ...).reshape(idx.size, c)
    ```
3.  **组装成对象数组**: 创建一个 `dtype=object` 的数组，然后将上一步生成的 `mat` 的每一行（一个犹豫集）填充进去。
    ```python
    # /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrohfs/random.py
    md_rows = [None] * size
    # ... loop to fill md_rows ...
    md_obj = self._build_object_array_from_rows(np.array(md_rows, dtype=object))
    ```
4.  **处理约束并构建后端**: `nmd` 的生成遵循类似逻辑，但约束处理更复杂，因为每个元素的 `nmd` 上限取决于其 `md` 集合的最大值。最后，将 `md_obj` 和 `nmd_obj` 传递给 `QROHFNBackend.from_arrays`。

### 步骤 5: `fuzznum()` - 生成单个实例

`fuzznum` 的实现应该尽可能复用 `fuzzarray` 的逻辑以保证一致性。最简单的方法是直接调用 `fuzzarray` 并取第一个元素。但在某些情况下，为了逻辑更清晰或避免不必要的开销，也可以单独实现。

**示例 (`qrofn`)**:
`QROFNRandomGenerator.fuzznum` 单独实现了生成逻辑，因为它相对简单，可以作为 `fuzzarray` 逻辑的一个清晰的、非向量化的版本。

```python
# /Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzztype/qrofs/random.py
def fuzznum(self, rng: np.random.Generator, q: Optional[int] = None, **kwargs) -> 'Fuzznum':
    # ...
    # Generate a single membership degree
    md = self._sample_from_distribution(rng, size=None, ...)
    # ...
    # Calculate non-membership degree based on the constraint
    max_nmd = (1 - md ** q) ** (1 / q)
    # ...
    return Fuzznum(mtype='qrofn', q=q).create(md=md, nmd=nmd)
```

## 4.3. 总结

为你的 `mtype` 注册一个随机生成器是其生态中不可或缺的一环。请记住以下关键点：

-   **继承 `ParameterizedRandomGenerator`**: 不要重新发明轮子，直接使用它提供的参数管理和分布采样工具。
-   **性能在 `fuzzarray`**: 你的性能优化重点应该放在 `fuzzarray` 方法上。
-   **向量化是王道**: 无论是完全向量化还是分组向量化，都要最大限度地利用 NumPy 的能力。
-   **直通后端**: 永远选择 `Backend.from_arrays()` 来构建 `Fuzzarray`，避免在 Python 层循环创建 `Fuzznum`。
-   **别忘了 `@register_random`**: 否则系统将无法找到你的生成器。

遵循本指南，你将能够为你的自定义模糊数类型集成一个与 `AxisFuzzy` 内置类型同样强大和高效的随机生成器。