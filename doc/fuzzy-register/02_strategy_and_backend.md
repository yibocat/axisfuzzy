# 2. 策略与后端：定义模糊数的核心

在 `AxisFuzzy` 中，每一种新的模糊数类型都由两个核心组件定义：**策略 (Strategy)** 和 **后端 (Backend)**。这两个组件共同构成了模糊数类型的完整实现，使其能够无缝集成到 `AxisFuzzy` 的生态系统中。

- **策略 (`FuzznumStrategy`)**：定义了**单个**模糊数实例的数据结构、行为和约束。它更侧重于对象层面的逻辑，例如属性验证、格式化输出和内部状态管理。
- **后端 (`FuzzarrayBackend`)**：为**模糊数数组**提供了高性能的、基于 NumPy 的数据存储和操作。它采用“数组结构”(Struct-of-Arrays, SoA) 的设计模式，将模糊数的不同分量存储在独立的 NumPy 数组中，从而实现高效的向量化计算。

本章将深入探讨如何通过继承 `FuzznumStrategy` 和 `FuzzarrayBackend` 来实现一个新的模糊数类型，并以 `qrofn`（q-rung orthopair fuzzy number）为例进行详细说明。

## 2.1. `FuzznumStrategy`：定义单个模糊数的行为

`FuzznumStrategy` 是一个抽象基类，它为所有模糊数类型提供了一个统一的接口。要实现一个新的模糊数类型，第一步就是创建一个继承自 `FuzznumStrategy` 的具体策略类。

### 2.1.1. 核心职责

一个具体的策略类需要承担以下职责：

1.  **声明数据属性**：通过类属性或类型注解，明确该模糊数包含哪些数据分量（例如，隶属度 `md` 和非隶属度 `nmd`）。
2.  **实现验证与约束**：通过注册**验证器 (Validators)** 和**变更回调 (Change Callbacks)**，确保模糊数的每个实例都满足其数学定义（例如，q-rung 约束 `md^q + nmd^q <= 1`）。
3.  **定义格式化行为**：实现 `report()`、`str()` 和 `__format__()` 等方法，控制模糊数如何以字符串形式呈现。
4.  **注册自身**：使用 `@register_strategy` 装饰器，将新策略注册到 `AxisFuzzy` 的类型系统中。

### 2.1.2. 示例：`QROFNStrategy` 的实现

让我们以 `axisfuzzy.fuzztype.qrofs.qrofn.QROFNStrategy` 为例，剖析其实现细节。

```python
# axisfuzzy/fuzztype/qrofs/qrofn.py

from ...core import FuzznumStrategy, register_strategy

@register_strategy
class QROFNStrategy(FuzznumStrategy):
    mtype = 'qrofn'
    md: Optional[float] = None
    nmd: Optional[float] = None

    def __init__(self, q: Optional[int] = None):
        super().__init__(q=q)

        # 1. 注册属性验证器
        self.add_attribute_validator(
            'md', lambda x: x is None or 0 <= x <= 1)
        self.add_attribute_validator(
            'nmd', lambda x: x is None or 0 <= x <= 1)

        # 2. 注册变更回调
        self.add_change_callback('md', self._on_membership_change)
        self.add_change_callback('nmd', self._on_membership_change)
        self.add_change_callback('q', self._on_q_change)

    def _fuzz_constraint(self):
        # 3. 实现模糊约束检查
        if self.md is not None and self.nmd is not None and self.q is not None:
            sum_of_powers = self.md ** self.q + self.nmd ** self.q
            if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                raise ValueError(...)

    def _on_membership_change(self, ...):
        # 当 md 或 nmd 变化时，触发约束检查
        self._fuzz_constraint()

    def _on_q_change(self, ...):
        # 当 q 变化时，也触发约束检查
        self._fuzz_constraint()

    # ... 格式化方法 ...
```

#### 关键实现点：

- **`@register_strategy`**：这个装饰器至关重要，它将 `QROFNStrategy` 与字符串标识符 `'qrofn'` 关联起来，并将其注册到全局的策略注册表中。
- **属性声明**：`md` 和 `nmd` 被声明为类属性，`FuzznumStrategy` 的元类机制会自动将它们识别为该策略的数据字段。
- **属性验证器**：`add_attribute_validator` 方法注册了一个 lambda 函数，用于确保 `md` 和 `nmd` 的值始终在 `[0, 1]` 区间内。当尝试为这些属性赋一个无效值时，`__setattr__` 会自动调用此验证器并抛出 `ValueError`。
- **变更回调**：`add_change_callback` 注册了回调函数 `_on_membership_change` 和 `_on_q_change`。当 `md`、`nmd` 或 `q` 的值发生变化时，这些回调会被触发，进而调用 `_fuzz_constraint` 方法来检查是否违反了 q-rung 约束。这种响应式设计确保了数据的一致性。
- **约束检查**：`_fuzz_constraint` 方法是实现模糊数数学定义的关键。它在数据发生变化时被调用，以保证对象状态的合法性。

## 2.2. `FuzzarrayBackend`：为数组提供高性能支持

虽然 `FuzznumStrategy` 定义了单个模糊数的行为，但在实际应用中，我们通常需要处理成千上万个模糊数。为了实现高性能的数值计算，`AxisFuzzy` 引入了 `FuzzarrayBackend`。

### 2.2.1. SoA 设计模式

`FuzzarrayBackend` 采用了**数组结构 (Struct-of-Arrays, SoA)** 的设计模式。与“结构数组”(Array-of-Structs, AoS) 将整个对象存储在数组的每个位置不同，SoA 将对象的每个组件（或属性）分别存储在独立的数组中。

对于 `qrofn`，这意味着所有模糊数的 `md` 值存储在一个 NumPy 数组 `mds` 中，而所有 `nmd` 值存储在另一个数组 `nmds` 中。

**SoA 的优势**：
- **内存局部性**：当对某一分量（如所有 `md`）进行计算时，数据在内存中是连续存储的，这极大地提高了缓存命中率。
- **向量化 (SIMD)**：现代 CPU 可以对连续的数据块执行单指令多数据 (SIMD) 操作。NumPy 等库能够充分利用这一特性，对整个数组执行并行计算，速度远超逐元素循环。

### 2.2.2. 示例：`QROFNBackend` 的实现

`QROFNBackend` 继承自 `FuzzarrayBackend`，并为 `qrofn` 类型提供了具体的 SoA 实现。

```python
# axisfuzzy/fuzztype/qrofs/backend.py

from ...core import FuzzarrayBackend, register_backend

@register_backend
class QROFNBackend(FuzzarrayBackend):
    mtype = 'qrofn'

    def _initialize_arrays(self):
        # 1. 初始化分量数组
        self.mds = np.zeros(self.shape, dtype=np.float64)
        self.nmds = np.zeros(self.shape, dtype=np.float64)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        # 2. 提供单个元素的视图
        md_value = float(self.mds[index])
        nmd_value = float(self.nmds[index])
        return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        # 3. 设置单个元素的数据
        self.mds[index] = fuzznum.md
        self.nmds[index] = fuzznum.nmd

    def copy(self) -> 'QROFNBackend':
        # 4. 实现深拷贝
        new_backend = QROFNBackend(self.shape, self.q)
        new_backend.mds = self.mds.copy()
        new_backend.nmds = self.nmds.copy()
        return new_backend

    def slice_view(self, key) -> 'QROFNBackend':
        # 5. 实现切片视图
        new_shape = self.mds[key].shape
        new_backend = QROFNBackend(new_shape, self.q)
        new_backend.mds = self.mds[key]  # NumPy 切片默认是视图
        new_backend.nmds = self.nmds[key]
        return new_backend
    
    @classmethod
    def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int) -> 'QROFNBackend':
        # 6. 从原始数组创建后端的工厂方法
        backend = cls(mds.shape, q)
        backend.mds = mds.copy()
        backend.nmds = nmds.copy()
        return backend

    # ... 格式化和组件访问方法 ...
```

#### 关键实现点：

- **`@register_backend`**：与策略类似，此装饰器将后端注册到全局注册表，使其可用于创建 `Fuzzarray`。
- **`_initialize_arrays`**：这是后端的核心。它根据传入的 `shape` 初始化了两个 NumPy 数组 `self.mds` 和 `self.nmds`，用于存储所有模糊数的隶属度和非隶属度。
- **`get_fuzznum_view` / `set_fuzznum_data`**：这两个方法是高层 `Fuzzarray` 对象与底层 NumPy 数组之间的桥梁，用于读取和写入单个模糊数的数据。
- **`copy` / `slice_view`**：这两个方法实现了数组的复制和切片语义。`slice_view` 的实现巧妙地利用了 NumPy 切片默认创建视图（共享内存）的特性，避免了不必要的数据复制，从而提高了性能。
- **`from_arrays`**：这个类方法作为一个工厂，允许直接从现有的 NumPy 数组创建后端实例，这在数据转换和集成的场景中非常有用。

## 2.3. 处理更复杂的数据类型：以 `qrohfn` 为例

`AxisFuzzy` 的策略-后端架构具有很好的扩展性，可以支持更复杂的数据结构，例如犹豫模糊数 (`qrohfn`)，其隶属度和非隶属度本身就是一组数值（一个集合）。

- **`QROHFNStrategy`**：
  - 其 `md` 和 `nmd` 属性是 `np.ndarray` 类型。
  - 它注册了**属性转换器 (Transformers)**，在 `__setattr__` 中自动将输入的列表或元组转换为 NumPy 数组。
  - 其约束检查 `_fuzz_constraint` 作用于犹豫集中的最大值：`max(md)^q + max(nmd)^q <= 1`。

- **`QROHFNBackend`**：
  - 由于每个元素的 `md` 和 `nmd` 都是一个可变长度的数组，因此底层的 `self.mds` 和 `self.nmds` 数组的 `dtype` 必须是 `object`。
  - `_initialize_arrays` 会创建 `dtype=object` 的空数组。
  - `copy` 方法的实现更为复杂，需要逐个元素地复制数组，以实现真正的深拷贝。

这种差异化实现展示了策略-后端模式的灵活性：简单的值类型（如 `qrofn`）可以享受极致的性能优化，而复杂的对象类型（如 `qrohfn`）也能通过 `dtype=object` 的 NumPy 数组得到支持。

## 2.4. 总结

策略和后端是 `AxisFuzzy` 扩展性的基石。通过实现这两个组件，您可以定义一个功能完备、性能卓越的新模糊数类型。

- **`FuzznumStrategy`** 关注**单个**模糊数的**逻辑和行为**。
- **`FuzzarrayBackend`** 关注**模糊数数组**的**存储和性能**。
- **注册机制** (`@register_strategy`, `@register_backend`) 将它们无缝集成到 `AxisFuzzy` 的生态系统中。

在下一章中，我们将探讨如何为新定义的模糊数类型注册核心运算，使其能够参与到 `AxisFuzzy` 丰富的运算体系中。