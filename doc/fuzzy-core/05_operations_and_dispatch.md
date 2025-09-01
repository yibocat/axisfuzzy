# 5. 运算、调度与分派：AxisFuzzy 的计算核心

在 `AxisFuzzy` 中，所有模糊数的运算，从简单的算术加法到复杂的逻辑比较，都由一个精密设计、高度解耦的运算与调度框架负责。这个框架确保了运算的类型安全、高性能和可扩展性，是整个库的计算心脏。其核心由三大支柱构成：**数学基础 (`triangular.py`)**、**运算定义与注册 (`operation.py`)** 和 **中央分派 (`dispatcher.py`)**。

## 5.1. 数学基础：`triangular.py`

模糊逻辑中的许多运算都依赖于三角范数（Triangular Norms），即 T-范数（T-Norms）和 T-余范数（T-Conorms, 或 S-Norms）。`axisfuzzy.core.triangular` 模块为整个框架提供了这些基础的数学工具，并确保了计算的稳定性和一致性。

### `TypeNormalizer`：类型标准化工具

这是一个静态工具类，用于在整个计算流程中统一数据类型，避免由 Python 内置类型、NumPy 标量类型（如 `np.float64`）和 NumPy 数组之间的差异引发的潜在问题。

-   `to_float(value)` / `to_int(value)`: 将任何数值类型（包括 NumPy 类型）安全地转换为标准的 Python `float` 或 `int`。
-   `ensure_array_output(value)`: 确保输出结果总是一个 `np.float64` 类型的 NumPy 数组。
-   `ensure_scalar_output(value)`: 确保输出结果总是一个标准的 Python `float`。

### `BaseNormOperation`：T-范数的抽象基类

这是一个抽象基类，定义了所有 T-范数实现必须遵循的基本接口。

```python
class BaseNormOperation(ABC):
    def __init__(self, **params):
        # ... 初始化 q 值等 ...
    
    @abstractmethod
    def t_norm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T-范数的具体实现"""
        pass

    @abstractmethod
    def t_conorm_impl(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """T-余范数的具体实现"""
        pass
```

任何具体的 T-范数（如代数积）都继承自 `BaseNormOperation` 并实现其核心的 `t_norm_impl` 和 `t_conorm_impl` 方法。

### T-范数与 T-余范数实现

`triangular.py` 内置了 12 种经典的 T-范数/T-余范数对，每一种都作为 `BaseNormOperation` 的子类实现：

-   **algebraic**: 代数积 (T-norm: `a*b`, S-norm: `a+b-a*b`)
-   **lukasiewicz**: Łukasiewicz (T-norm: `max(0, a+b-1)`, S-norm: `min(1, a+b)`)
-   **einstein**: 爱因斯坦积 (T-norm: `(a*b)/(1+(1-a)*(1-b))`, S-norm: `(a+b)/(1+a*b)`)
-   **hamacher**: Hamacher 族（带参数 `gamma`）
-   **yager**: Yager 族（带参数 `p`）
-   **schweizer_sklar**: Schweizer-Sklar 族（带参数 `p`）
-   **dombi**: Dombi 族（带参数 `p`）
-   **aczel_alsina**: Aczel-Alsina 族（带参数 `p`）
-   **frank**: Frank 族（带参数 `p`）
-   **minimum**: 最小/最大范数 (T-norm: `min(a,b)`, S-norm: `max(a,b)`)
-   **drastic**: 激烈积 (Drastic Product)
-   **nilpotent**: 幂零最小/最大范数

### `OperationTNorm`：T-范数的工厂与管理器

`OperationTNorm` 是 `triangular.py` 中最核心的类，它扮演着 T-范数操作的工厂和管理者的角色。当你需要进行任何基于 T-范数的计算时，你都会与这个类的实例打交道。

#### 主要职责：

1.  **工厂模式**：根据用户指定的 `norm_type` (如 `'algebraic'`) 和相关参数（如 `q` 值），从内部的 `_NORM_REGISTRY` 注册表中查找对应的 `BaseNormOperation` 子类，并实例化一个具体的范数操作器。

2.  **q-阶扩展**：这是 `OperationTNorm` 的一个关键特性。对于支持 q-阶扩展的范数，它能自动应用 q-rung 同构映射。
    -   **q-阶 T-范数**: \( T_q(a,b) = (T_{base}(a^q, b^q))^{1/q} \)
    -   **q-阶 T-余范数**: \( S_q(a,b) = (S_{base}(a^q, b^q))^{1/q} \)
    它会根据初始化时传入的 `q` 值，动态地创建出经过 q-阶扩展的 `t_norm` 和 `t_conorm` 函数。如果 `q=1`，则使用原始的基础实现。

3.  **生成器函数**：对于阿基米德 T-范数（如 algebraic, einstein），`OperationTNorm` 会自动提供其加法生成器 `g_func` 及其伪逆 `g_inv_func`。这些函数对于实现某些高级运算（如乘幂）至关重要。同时，它还会生成对偶生成器 `f_func` 和 `f_inv_func`，用于对偶运算。

4.  **统一接口**：它对外暴露 `t_norm`, `t_conorm`, `g_func`, `f_func` 等一系列标准化的方法。无论底层是哪种具体的 T-范数实现，上层调用代码都使用相同的接口，实现了高度的抽象和解耦。

5.  **高性能 Reduce 操作**：提供了 `t_norm_reduce` 和 `t_conorm_reduce` 方法，能够高效地对 `Fuzzarray` 的整个轴执行 T-范数或 T-余范数规约操作，其内部使用树形规约算法以优化性能。

```python
# 示例：创建一个 q=3 的爱因斯坦范数操作器
tnorm_op = OperationTNorm(norm_type='einstein', q=3)

# t_norm 和 t_conorm 方法已经被 q-阶扩展
md = tnorm_op.t_norm(0.5, 0.6)  # 内部计算 ( (0.5^3 * 0.6^3) / (1+(1-0.5^3)*(1-0.6^3)) )^(1/3)

# 获取生成器函数
g = tnorm_op.g_func
g_inv = tnorm_op.g_inv_func
```

## 5.2. 运算的定义与注册：`operation.py`

这个模块定义了运算的抽象和管理机制，确保了任何一种模糊数的任何一种运算都能被框架识别和调用。

### `OperationMixin`：运算的抽象基类

`OperationMixin` 是所有具体运算实现（如 `QROFNAddition`）必须继承的基类。它定义了一个运算实现所需遵循的契约，这与文档中提到的旧设计 (`__op__`, `__types__` 属性) 不同，新设计更加清晰和健壮。

一个典型的运算实现类如下所示：

```python
# axisfuzzy/fuzztype/qrofs/op.py

@register_operation
class QROFNAddition(OperationMixin):
    def get_operation_name(self) -> str:
        """返回操作的唯一名称"""
        return 'add'

    def get_supported_mtypes(self) -> List[str]:
        """返回此实现支持的 mtype 列表"""
        return ['qrofn']

    def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm):
        """Fuzznum 的元素级运算实现"""
        # ... 具体逻辑 ...
        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self, fuzzarray_1, other, tnorm):
        """Fuzzarray 的向量化运算实现"""
        # ... 具体逻辑，与后端交互 ...
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        md_res = tnorm.t_conorm(mds1, mds2)
        nmd_res = tnorm.t_norm(nmds1, nmds2)
        
        backend_cls = get_registry_fuzztype().get_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)
```

#### 核心方法：

-   `get_operation_name()`: **必须实现**。返回一个字符串，唯一标识这个运算（如 `'add'`, `'mul'`, `'gt'`）。
-   `get_supported_mtypes()`: **必须实现**。返回一个字符串列表，指明该实现支持哪些 `mtype`。
-   `_execute_binary_op_impl()`: **选择性实现**。处理两个 `Fuzznum` 之间的运算。
-   `_execute_unary_op_operand_impl()`: **选择性实现**。处理 `Fuzznum` 和一个标量之间的运算（如乘幂）。
-   `_execute_fuzzarray_op_impl()`: **选择性实现**。处理 `Fuzzarray` 的高性能向量化运算。它直接与 `FuzzarrayBackend` 交互，操作底层的 NumPy 数组。

### `OperationScheduler`：运算的注册与调度中心

`OperationScheduler` 是 `AxisFuzzy` 运算框架的“中央神经系统”。正如其在 `axisfuzzy/core/operation.py` 中的文档字符串所详述的，它是一个全局单例（通过 `get_registry_operation()` 访问），为所有数学和逻辑运算的定义、配置和执行提供了唯一的真实来源。它的重要性体现在四大核心职责上：

1.  **运算注册 (Registration)**：它维护一个核心注册表，将运算名称（如 `'add'`）和模糊数类型（`mtype`，如 `'qrofn'`）映射到一个具体的 `OperationMixin` 实现。这种解耦的设计允许开发者模块化地添加新运算或对新 `mtype` 的支持，而无需改动框架核心。

2.  **运算调度 (Dispatch)**：当 `operate` 函数需要执行运算时，它会使用 `op_name` 和 `mtype` 查询调度器（`scheduler.get_operation(op_name, mtype)`），以找到正确的 `OperationMixin` 实例。这是实现运算多态性的关键环节。

3.  **全局 T-范数配置 (Global T-Norm Configuration)**：调度器持有全局默认的 T-范数配置（例如，默认为 `'algebraic'`）。用户可以在运行时通过 `set_t_norm()` 动态更改此配置，从而在整个框架范围内调整底层逻辑以进行实验。

4.  **性能监控 (Performance Monitoring)**：调度器内置了一个线程安全的性能监视器。对于通过系统执行的每个操作，它都会记录执行时间并更新统计数据，如总调用次数和每种操作的平均执行时间。这对于调试、性能调优和理解计算成本是无价的。

#### 代码示例：与 `OperationScheduler` 交互

尽管最终用户很少直接与 `OperationScheduler` 交互，但了解其使用方式对于开发者扩展和调试 `AxisFuzzy` 至关重要。

```python
# 1. 获取全局唯一的调度器实例
from axisfuzzy.core.operation import get_registry_operation
scheduler = get_registry_operation()

# 2. 查看 'qrofn' 类型支持哪些运算
available_ops = scheduler.get_available_ops('qrofn')
print(f"QROFN supports: {available_ops}")
# 可能的输出: QROFN supports: ['add', 'sub', 'mul', 'div', 'complement', 'equals', ...]

# 3. 更改全局默认的 T-范数，用于所有后续运算
# 将默认的 'algebraic' 范数更改为带参数 p=2 的 Hamacher 范数
print(f"Default T-Norm before: {scheduler.get_default_t_norm_config()}")
scheduler.set_t_norm('hamacher', p=2)
print(f"Default T-Norm after: {scheduler.get_default_t_norm_config()}")

# 此时，任何未显式指定 t_norm 的新运算都将使用 Hamacher 范数

# 4. 模拟 FuzznumStrategy，获取一个具体的运算实现
# 这是框架内部的常见操作
add_op_for_qrofn = scheduler.get_operation('add', 'qrofn')
print(f"Found operation for 'add' on 'qrofn': {add_op_for_qrofn}")

# 5. 运行一些运算后，获取性能统计数据 (假设已经执行了一些模糊运算)
# stats = scheduler.get_performance_stats(time_unit='us')
# print(stats['average_times_by_operation_type'])
```

## 5.3. 运算的总指挥：`dispatcher.py`

`operate` 函数是 `AxisFuzzy` 所有运算的中央入口点和总指挥。当你执行 `fuzzarray1 + fuzznum1` 这样的代码时，Python 的运算符重载机制（如 `Fuzzarray.__add__`）最终会调用 `operate` 函数。

`operate` 函数的智能分派机制是 `AxisFuzzy` 实现多态运算的关键。它通过一系列规则，按优先级检查操作数的类型，并选择最高效的执行路径。

#### `operate` 的分派规则（按优先级）：

1.  **`Fuzznum` vs `Fuzznum`**:
    -   直接调用左操作数 `Fuzznum` 的策略实例 (`FuzznumStrategy`) 的 `execute_operation` 方法。
    -   `execute_operation` 内部会向 `OperationScheduler` 查询对应的 `OperationMixin` 实例，并调用其 `_execute_binary_op_impl` 方法执行元素级计算。
    -   返回一个新的 `Fuzznum`。

2.  **`Fuzznum` vs `Fuzzarray`**:
    -   这是一个混合类型操作。`operate` 会将左侧的 `Fuzznum` **广播**成一个与右侧 `Fuzzarray` 形状相同的新 `Fuzzarray`。
    -   然后，问题转化为 **`Fuzzarray` vs `Fuzzarray`** 的情况，递归调用 `operate`。

3.  **`Fuzznum` vs 标量/`ndarray`**:
    -   如果操作是 `*` 或 `/`，`op_name` 会被映射到 `'tim'` (times)。
    -   对于 `ndarray`，`Fuzznum` 会被广播成 `Fuzzarray`，问题转化为 **`Fuzzarray` vs `ndarray`**。
    -   对于标量，直接调用 `FuzznumStrategy` 的 `execute_operation`，并将标量作为操作数。

4.  **`Fuzzarray` vs `Fuzzarray`/`Fuzznum`**:
    -   这是最高效的路径之一。直接调用左操作数 `Fuzzarray` 的 `execute_vectorized_op` 方法。
    -   该方法内部会向 `OperationScheduler` 查询 `OperationMixin` 实例，并调用其 `_execute_fuzzarray_op_impl` 方法。
    -   `_execute_fuzzarray_op_impl` 与 `FuzzarrayBackend` 紧密协作，直接在底层的 NumPy 数组上执行向量化计算。

5.  **`Fuzzarray` vs 标量/`ndarray`**:
    -   同样直接调用 `Fuzzarray` 的 `execute_vectorized_op` 方法，由 `_execute_fuzzarray_op_impl` 内部处理与标量或 `ndarray` 的广播和计算。

6.  **反向操作 (如 `2 * fuzzarray`)**:
    -   `operate` 会检查操作数类型，如果发现是 `标量/ndarray` vs `Fuzznum/Fuzzarray`，并且操作是可交换的（如 `add`, `mul`），它会智能地**交换操作数的位置**，然后递归调用 `operate`，使其匹配前面的规则。

7.  **一元操作 (如 `~fuzznum`)**:
    -   当 `operand2` 为 `None` 时，`operate` 会将其识别为一元操作（如 `'complement'`），并分派到相应的 `Fuzznum` 或 `Fuzzarray` 实现。

如果没有任何规则匹配，`operate` 会抛出 `TypeError`，清晰地告知用户该操作不受支持。

## 5.4. 总结

`AxisFuzzy` 的运算与调度框架是一个优雅而强大的系统。通过将**数学基础** (`OperationTNorm`)、**运算定义** (`OperationMixin`)、**注册与发现** (`OperationScheduler`) 和 **智能分派** (`operate` 函数) 这几个关注点清晰地分离，它实现了：

-   **高度可扩展**：添加对新模糊数类型或新运算的支持，只需定义新的 `OperationMixin` 子类并注册，无需触及核心调度逻辑。
-   **类型安全与智能**：分派器在执行前会严格检查操作数类型，并能智能地处理广播和反向操作，确保只有兼容的类型才能进行运算。
-   **高性能**：当操作涉及 `Fuzzarray` 时，框架总是优先选择基于 SoA 后端的向量化路径，确保了大规模数据计算的性能。
-   **灵活性**：支持在运行时动态选择 T-范数，为科学研究和应用提供了极大的便利。

理解这个框架，是深入掌握 `AxisFuzzy` 如何执行计算以及如何为其扩展新功能的核心。