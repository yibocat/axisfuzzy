# 3. 核心运算：`OperationMixin` 与可扩展的计算引擎

`AxisFuzzy` 的核心运算系统是其强大功能和高度可扩展性的基石。该系统围绕 `OperationMixin` 抽象基类和 `OperationScheduler` 调度器构建，允许开发者为任意模糊数类型轻松定义和注册新的运算。本章将深入探讨其设计理念、不同计算类别的实现方式，并通过 `qrofn` 和 `qrohfn` 的实例，展示如何实现从单个模糊数到高性能向量化数组的运算。

## 3.1. 核心设计理念：`OperationMixin` 与 `OperationScheduler`

### `OperationMixin`：运算的“蓝图”

`OperationMixin` 是一个抽象基类，它为所有模糊数运算提供了一个统一的接口“蓝图”。任何希望在 `AxisFuzzy` 中注册的运算都必须继承自此类。其核心职责包括：

- **定义运算身份**：通过 `get_operation_name()` 和 `get_supported_mtypes()` 方法，明确运算的名称（如 `'add'`）及其支持的模糊数类型（如 `['qrofn']`）。
- **提供统一的执行入口**：定义了四种核心的运算执行接口，分别对应不同的计算场景：
  - `_execute_binary_op_impl`：处理两个 `Fuzznum` 对象之间的二元运算（如加、减、乘、除）。
  - `_execute_unary_op_impl`：处理单个 `Fuzznum` 对象的一元运算（如求逆）。
  - `_execute_comparison_op_impl`：处理两个 `Fuzznum` 对象之间的比较运算（如大于、小于、等于）。
  - `_execute_fuzzarray_op_impl`：处理 `Fuzzarray` 数组的高性能向量化运算。
- **封装通用逻辑**：`OperationMixin` 的公共 `execute_*` 方法（非 `_impl` 版本）封装了操作数预处理、类型检查、性能计时和错误处理等通用逻辑，使得具体实现可以专注于核心算法。

### `OperationScheduler` 与 `@register_operation`：注册与调度中心

`OperationScheduler` 是一个全局单例，充当所有已注册运算的“调度中心”。当用户在模糊数上执行一个操作（例如 `fuzznum1 + fuzznum2`）时，`AxisFuzzy` 会查询 `OperationScheduler`，根据操作名称（`'add'`）和模糊数类型（`mtype`）找到对应的 `OperationMixin` 实现并执行它。

`@register_operation` 装饰器则极大地简化了注册过程。只需将此装饰器应用于任何 `OperationMixin` 的子类，该类就会被自动实例化并注册到 `OperationScheduler` 中，立即可用。

## 3.2. 不同计算类别的实现剖析

下面，我们将通过具体的代码示例，深入了解如何为新的模糊数类型实现不同类别的运算。

### 类别一：二元算术运算 (`_execute_binary_op_impl`)

二元运算是两个模糊数之间最常见的交互。以 `qrofn` 的加法 `QROFNAddition` 为例，它展示了实现一个标准二元运算的典型模式。

**示例：`axisfuzzy/fuzztype/qrofs/op.py` 中的 `QROFNAddition`**

```python
@register_operation
class QROFNAddition(OperationMixin):
    def get_operation_name(self) -> str:
        return 'add'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        # 隶属度使用 t-conorm (S-norm) 计算
        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        # 非隶属度使用 t-norm (T-norm) 计算
        nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)

        # 返回一个字典，包含新模糊数的属性
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}
```

**实现要点**：

1.  **继承与注册**：类继承自 `OperationMixin` 并使用 `@register_operation` 注册。
2.  **身份声明**：`get_operation_name` 返回 `'add'`，`get_supported_mtypes` 返回 `['qrofn']`。
3.  **核心逻辑**：`_execute_binary_op_impl` 接收两个模糊数的策略对象 (`strategy_1`, `strategy_2`) 和一个 `OperationTNorm` 实例。
4.  **利用 `OperationTNorm`**：模糊加法通常依赖于三角范数。这里，新的隶属度由 `t_conorm` 计算，非隶属度由 `t_norm` 计算。`OperationTNorm` 封装了多种 t-norm/t-conorm 的实现（如 'algebraic', 'lukasiewicz'），使得运算逻辑与具体的范数选择解耦。
5.  **返回结果**：方法返回一个字典，其中包含了构建新模糊数所需的所有属性。`AxisFuzzy` 的核心框架会利用这个字典自动创建并返回一个新的 `Fuzznum` 实例。

### 类别二：比较运算 (`_execute_comparison_op_impl`)

比较运算定义了模糊数之间的序关系，这对于排序、决策等场景至关重要。其实现通常比算术运算更复杂，可能需要一个明确的得分函数（Score Function）。

**示例：`axisfuzzy/fuzztype/qrofs/op.py` 中的 `QROFNGreaterThan`**

```python
@register_operation
class QROFNGreaterThan(OperationMixin):
    def get_operation_name(self) -> str:
        return 'gt'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> bool:
        # 定义得分函数
        def score(strategy):
            return (strategy.md**strategy.q - strategy.nmd**strategy.q)

        score1 = score(strategy_1)
        score2 = score(strategy_2)

        # 根据得分和准确度函数比较
        if score1 > score2:
            return True
        elif score1 < score2:
            return False
        else:
            # 如果得分相同，则比较准确度函数
            acc1 = strategy_1.md**strategy_1.q + strategy_1.nmd**strategy_1.q
            acc2 = strategy_2.md**strategy_2.q + strategy_2.nmd**strategy_2.q
            return acc1 > acc2
```

**实现要点**：

1.  **得分函数**：比较的核心是定义一个合理的得分函数 `score()`，它将一个模糊数映射到一个实数。
2.  **多级比较**：当得分相同时，为了提供更精确的比较，可以引入一个“准确度函数”（Accuracy Function）作为第二比较标准。
3.  **返回布尔值**：比较运算直接返回一个布尔值 `True` 或 `False`。

### 类别三：高性能数组运算 (`_execute_fuzzarray_op_impl`)

`_execute_fuzzarray_op_impl` 是 `AxisFuzzy` 性能的关键。它负责处理 `Fuzzarray` 对象上的向量化运算。由于不同模糊数类型的后端存储结构不同，其实现策略也各异。

#### 策略 A：直接向量化 (适用于 `qrofn`)

当模糊数的后端（如 `QROFNBackend`）使用标准的 `NumPy` 数组（`SoA` 结构）存储数据时，可以直接利用 `NumPy` 的广播（broadcasting）和向量化能力实现高性能运算。

**示例：`axisfuzzy/fuzztype/qrofs/op.py` 中的 `QROFNAddition._execute_fuzzarray_op_impl`**

```python
def _execute_fuzzarray_op_impl(self,
                               fuzzarray_1: Fuzzarray,
                               other: Optional[Any],
                               tnorm: OperationTNorm) -> Fuzzarray:
    # 准备操作数，处理 Fuzzarray 和 Fuzznum 之间的广播
    mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

    # 直接在整个数组上执行向量化计算
    md_res = tnorm.t_conorm(mds1, mds2)
    nmd_res = tnorm.t_norm(nmds1, nmds2)

    # 从结果数组创建新的 Fuzzarray
    backend_cls = get_registry_fuzztype().get_backend('qrofn')
    new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
    return Fuzzarray(backend=new_backend)
```

**实现要点**：

1.  **数据对齐**：`_prepare_operands` 辅助函数负责从操作数（可以是另一个 `Fuzzarray` 或一个 `Fuzznum`）中提取 `md` 和 `nmd` 数组，并利用 `np.broadcast_arrays` 确保它们的形状兼容。
2.  **向量化计算**：加法逻辑 `tnorm.t_conorm` 和 `tnorm.t_norm` 直接应用于整个 `NumPy` 数组，计算在 `C` 语言层面完成，速度极快。
3.  **构建新后端**：计算结果（`md_res`, `nmd_res`）被用来创建一个新的后端实例，并最终包装成一个新的 `Fuzzarray` 对象返回。

#### 策略 B：通用函数 `np.frompyfunc` (适用于 `qrohfn`)

当模糊数的后端存储的是 `object` 类型的 `NumPy` 数组（例如，`QROHFNBackend` 的每个元素都是一个可变长度的犹豫模糊集）时，直接的向量化不再适用。此时，`np.frompyfunc` 成为了兼顾灵活性和性能的利器。

**示例：`axisfuzzy/fuzztype/qrohfs/op.py` 中的 `QROHFNAddition._execute_fuzzarray_op_impl`**

```python
def _execute_fuzzarray_op_impl(self,
                               operand1: Any,
                               other: Optional[Any],
                               tnorm: OperationTNorm) -> Any:
    mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

    # 定义一个处理单个元素（犹豫集）的 Python 函数
    # _pairwise_combinations 会计算两个犹豫集之间的笛卡尔积组合
    add_md_func = lambda md1, md2: _pairwise_combinations(md1, md2, tnorm.t_conorm)
    add_nmd_func = lambda nmd1, nmd2: _pairwise_combinations(nmd1, nmd2, tnorm.t_norm)

    # 使用 np.frompyfunc 将 Python 函数转换为 NumPy 通用函数 (ufunc)
    # 参数：(函数, 输入参数数量, 输出参数数量)
    add_md_ufunc = np.frompyfunc(add_md_func, 2, 1)
    add_nmd_ufunc = np.frompyfunc(add_nmd_func, 2, 1)

    # 将 ufunc 应用于 object 数组
    md_res = add_md_ufunc(mds1, mds2)
    nmd_res = add_nmd_ufunc(nmds1, nmds2)

    # 创建新的 Fuzzarray
    backend_cls = get_registry_fuzztype().get_backend('qrohfn')
    new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
    return Fuzzarray(backend=new_backend)
```

**实现要点**：

1.  **定义元素级操作**：首先定义一个纯 `Python` 函数（`add_md_func`），它知道如何处理两个独立的犹豫模糊集（即 `object` 数组中的单个元素）。这里的核心是 `_pairwise_combinations`，它负责计算笛卡尔积并应用 `t_conorm`。
2.  **创建 `ufunc`**：`np.frompyfunc` 接收这个 `Python` 函数，并将其“包装”成一个 `NumPy` 通用函数 (`ufunc`)。这个 `ufunc` 可以在 `NumPy` 的迭代机制下高效运行，虽然其核心依然是 `Python` 代码，但免去了在 `Python` 层面写循环的巨大开销。
3.  **应用 `ufunc`**：创建好的 `ufunc` 可以像标准的 `NumPy` 函数（如 `np.add`）一样，直接应用于 `object` 数组 `mds1` 和 `mds2` 上。`NumPy` 会自动处理元素对的迭代。
4.  **结果**：`md_res` 和 `nmd_res` 仍然是 `object` 数组，每个元素都是运算后的新犹豫模糊集。

## 3.3. 总结

`AxisFuzzy` 的运算系统通过 `OperationMixin` 和 `OperationScheduler` 提供了一个清晰、模块化且高度可扩展的框架。开发者可以通过继承 `OperationMixin` 并实现相应的 `_execute_*_impl` 方法，为自定义的模糊数类型添加丰富的运算能力。

在实现高性能的 `Fuzzarray` 运算时，应根据后端的数据结构选择合适的策略：

-   对于基于标准 `NumPy` 数据类型（如 `float64`）的 `SoA` 后端，应优先使用**直接向量化**，以获得最佳性能。
-   对于基于 `object` 类型的后端（用于存储复杂或可变长度的元素），**`np.frompyfunc`** 提供了一个在 `Python` 灵活性和 `NumPy` 高性能迭代之间取得平衡的强大工具。

理解并掌握这一运算系统，是深入使用和扩展 `AxisFuzzy` 的关键一步。