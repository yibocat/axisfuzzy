# 6. 扩展系统：为模糊数注入灵魂

至此，我们已经为新的模糊数类型构建了坚实的基础：定义了其核心行为（策略与后端）、赋予了其计算能力（核心运算）、实现了随机生成和模糊化。然而，一个功能完备的模糊数类型远不止于此。它还需要丰富的辅助功能，如便捷的构造器、与外部世界的交互（I/O）、多样的度量方法（距离、得分）以及强大的聚合运算。

`AxisFuzzy` 的 **扩展系统 (Extension System)** 正是为此而生。它是一个强大而灵活的框架，允许您将任意功能“注入”到 `Fuzznum`、`Fuzzarray` 或 `axisfuzzy` 的顶层命名空间中，而无需修改核心库代码。本章将深入探讨这一系统的架构，并指导您如何为您的模糊数类型注册一整套完备的扩展功能。

## 6.1. 扩展系统的“三位一体”架构

`AxisFuzzy` 的扩展系统由三个核心组件协同工作，共同实现功能的动态注册、解析和调用：

1.  **注册表 (Registry)**: 位于 `axisfuzzy/extension/registry.py`，是所有扩展功能的中央数据库。它像一个详细的账本，记录了每个扩展函数的元数据：它的逻辑名称（如 `'distance'`）、针对的模糊数类型（`mtype`）、应该如何被注入（`injection_type`）以及注册的优先级等。

2.  **装饰器 (Decorator)**: 位于 `axisfuzzy/extension/decorator.py`，是您与注册表交互的主要工具。`@extension` 装饰器提供了一种声明式、极其便捷的方式来将一个普通的 Python 函数注册到系统中。您只需在函数上标记其身份，剩下的繁重工作都由框架完成。

3.  **调度器 (Dispatcher)**: 位于 `axisfuzzy/extension/dispatcher.py`，是扩展系统在运行时的“魔术师”。当您调用一个扩展方法（如 `my_array.to_csv()`）时，实际上调用的是一个由调度器创建的“代理” (Proxy)。这个代理会实时检查 `my_array` 的 `mtype`，然后去注册表中查找并执行与之匹配的具体实现函数。

这个“注册-调度-注入”的架构实现了关注点分离，使得扩展的开发者只需专注于功能的具体实现，而无需关心其在复杂系统中的调用细节。

## 6.2. 编写与注册扩展：一个完整实例

让我们以 `qrofn` 的 `to_csv` 功能为例，走一遍从实现到注册的全过程。

### 步骤 1：在 `ext/` 目录下实现功能逻辑

最佳实践是将所有扩展功能的具体实现代码放在对应 `mtype` 包下的 `ext/` 目录中。这样做可以保持代码组织的清晰。

**示例：`axisfuzzy/fuzztype/qrofs/ext/io.py`**
```python
# axisfuzzy/fuzztype/qrofs/ext/io.py

import csv
import numpy as np
from ....core import Fuzzarray

def _qrofn_to_csv(arr: Fuzzarray, path: str, **kwargs) -> None:
    """High-performance CSV export using backend arrays directly."""
    if arr.mtype != 'qrofn':
        raise TypeError(...)

    # 直接从后端获取分量数组，实现高性能
    mds, nmds = arr.backend.get_component_arrays()

    # 利用 NumPy 的向量化操作高效拼接字符串
    str_data = np.char.add(
        np.char.add('<', mds.astype(str)),
        np.char.add(',', np.char.add(nmds.astype(str), '>'))
    )

    # 写入 CSV 文件
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, **kwargs)
        if str_data.ndim == 1:
            writer.writerow(str_data)
        else:
            writer.writerows(str_data)
```
这个 `_qrofn_to_csv` 函数本身是一个独立的、可测试的单元。它接收一个 `Fuzzarray` 对象和文件路径，并执行具体的导出逻辑。

### 步骤 2：在 `extension.py` 中使用 `@extension` 注册

实现了功能后，我们需要在 `mtype` 包下的 `extension.py` 文件中将其注册到系统中。这个文件是该模糊数所有扩展的注册中心。

**示例：`axisfuzzy/fuzztype/qrofs/extension.py`**
```python
# axisfuzzy/fuzztype/qrofs/extension.py

from . import ext  # 导入包含所有实现的 ext 包
from ...extension import extension

# ...

# ========================= IO Operation =========================

@extension(name='to_csv', mtype='qrofn', target_classes=['Fuzzarray'])
def qrofn_to_csv_ext(fuzz, *args, **kwargs):
    """Save a QROFN Fuzzarray to a CSV file."""
    return ext._qrofn_to_csv(fuzz, *args, **kwargs)

@extension(
    name='read_csv',
    mtype='qrofn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def qrofn_from_csv_ext(*args, **kwargs) -> Fuzzarray:
    """Load a QROFN Fuzzarray from a CSV file."""
    return ext._qrofn_from_csv(*args, **kwargs)
```

这里我们看到了两种不同的注册方式，由 `injection_type` 参数控制：

-   **`to_csv` (实例方法)**:
    -   `name='to_csv'`: 定义了逻辑上的扩展名。
    -   `mtype='qrofn'`: 指明这是为 `qrofn` 类型定制的实现。
    -   `target_classes=['Fuzzarray']`: 指定这个方法将被注入到 `Fuzzarray` 类中。
    -   `injection_type` 未指定，默认为 `'both'`，意味着它既可以作为实例方法 (`my_array.to_csv(...)`) 调用，也可以作为顶层函数 (`axisfuzzy.to_csv(my_array, ...)` ) 调用。

-   **`read_csv` (顶层函数)**:
    -   `injection_type='top_level_function'`: 明确指定这只应作为一个顶层函数存在。这对于工厂方法（如从文件创建对象）非常合适。调用方式为 `axisfuzzy.read_csv(path, mtype='qrofn', q=2)`。

### 步骤 3：探索更多注入类型

`@extension` 装饰器还支持其他注入类型，提供了极大的灵活性。

-   **`instance_property` (实例属性)**: 用于计算那些看起来像属性的值。

**示例：为 `qrofn` 注册 `score` 属性**
```python
# axisfuzzy/fuzztype/qrofs/extension.py

@extension(
    name='score',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def qrofn_score_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the score of a QROFN Fuzzarray or Fuzznum."""
    return ext._qrofn_score(fuzz)
```
注册后，您就可以像访问普通属性一样获取得分值：`my_fuzznum.score` 或 `my_array.score`。调度器会自动调用 `qrofn_score_ext(fuzz)` 函数并返回结果。

## 6.3. `qrofn` 扩展功能全景

一个成熟的模糊数类型需要一整套完备的扩展。以下是 `axisfuzzy` 为 `qrofn` 类型注册的全部扩展功能，它们共同构成了 `qrofn` 强大易用的功能集。

### 构造器 (Constructors)
这些是顶层函数，用于创建特定状态的 `Fuzzarray` 或 `Fuzznum`。
-   `empty(shape, q)`: 创建一个未初始化的数组。
-   `positive(shape, q)`: 创建一个全为“正”的数组 (`md=1, nmd=0`)。
-   `negative(shape, q)`: 创建一个全为“负”的数组 (`md=0, nmd=1`)。
-   `full(shape, fill_value)`: 用指定的 `Fuzznum` 值填充数组。
-   `empty_like(obj)`: 创建一个与给定对象形状和 `q` 值相同的空数组。
-   `positive_like(obj)`: 创建一个与给定对象形状和 `q` 值相同的“正”数组。
-   `negative_like(obj)`: 创建一个与给定对象形状和 `q` 值相同的“负”数组。
-   `full_like(obj, fill_value)`: 创建一个与给定对象形状和 `q` 值相同并用 `fill_value` 填充的数组。

### I/O 操作
用于数据的导入和导出，作为实例方法和顶层函数存在。
-   `to_csv(path, **kwargs)`: 将 `Fuzzarray` 保存到 CSV 文件。
-   `read_csv(path, q, **kwargs)`: 从 CSV 文件加载 `Fuzzarray`。
-   `to_json(path, **kwargs)`: 保存到 JSON 文件。
-   `read_json(path, **kwargs)`: 从 JSON 文件加载。
-   `to_npy(path, **kwargs)`: 保存为 NumPy 的 `.npy` 二进制格式。
-   `read_npy(path, **kwargs)`: 从 `.npy` 文件加载。

### 度量方法 (Measurement)
用于计算模糊数之间的关系或其自身特性。
-   `distance(other, p_l, indeterminacy)`: 计算与另一个模糊数或数组的距离。
-   `score`: (属性) 计算得分值 `md^q - nmd^q`。
-   `acc`: (属性) 计算准确度 `md^q + nmd^q`。
-   `ind`: (属性) 计算不确定度 `1 - (md^q + nmd^q)`。

### 字符串转换
-   `str2fuzznum(fuzznum_str, q)`: (顶层函数) 将形如 `<md, nmd>` 的字符串转换为 `Fuzznum` 对象。

### 聚合运算 (Aggregation)
这些是 `Fuzzarray` 的实例方法，用于对数组进行降维计算。
-   `sum(axis)`: 沿指定轴计算模糊和。
-   `mean(axis)`: 沿指定轴计算模糊均值。
-   `max(axis)`: 沿指定轴根据得分函数找出最大值。
-   `min(axis)`: 沿指定轴根据得分函数找出最小值。
-   `prod(axis)`: 沿指定轴计算模糊积。
-   `var(axis)`: 沿指定轴计算模糊方差。
-   `std(axis)`: 沿指定轴计算模糊标准差。

## 6.4. 总结

`AxisFuzzy` 的扩展系统是其可扩展性的点睛之笔。通过遵循“实现逻辑 -> 注册扩展”的简单模式，您可以为您的模糊数类型添加任何可以想象到的功能。

1.  **分离关注点**: 在 `ext/` 目录中编写纯粹的功能实现。
2.  **集中注册**: 在 `extension.py` 中使用 `@extension` 装饰器进行声明式注册。
3.  **选择注入类型**: 通过 `injection_type` 参数精确控制功能如何呈现给最终用户（实例方法、顶层函数或属性）。

完成了这一步，您的模糊数类型就不仅拥有了核心的数学定义和运算能力，更拥有了一套丰富、实用、符合直觉的API，真正成为了 `AxisFuzzy` 生态中一个成熟、完备的成员。在下一章，也是最后一章，我们将通过一个完整的端到端示例，串联起所有步骤，展示如何从零开始集成一个全新的模糊数类型。