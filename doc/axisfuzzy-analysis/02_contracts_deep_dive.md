# 深入解析数据契约 (Data Contracts)

数据契约是 `axisfuzzy.analysis` 系统的核心与灵魂。它是一种形式化的规范，用于定义在分析管道中流动的数据的结构、类型和约束。通过引入数据契约，我们得以在管道构建时（Graph-Time）而非运行时（Runtime）捕捉到潜在的数据不匹配错误，从而极大地提升了分析流程的健壮性和可靠性。

本章将深入探讨数据契約系统的三个关键部分：
1.  **`Contract` 类**：数据契约的基石。
2.  **`@contract` 装饰器**：将普通函数无缝转换为契约驱动组件的魔法。
3.  **内置契约库**：开箱即用的常用数据规范。
4.  **契约在系统中的应用**：贯穿管道与组件的设计哲学。

---

## 1. `Contract` 类：契约的基石

所有数据契约的核心是 `axisfuzzy.analysis.contracts.base.Contract` 类。它是一个简洁而强大的对象，封装了定义一份数据契约所需的所有信息。

### 1.1 核心构成：`name`, `validator`, 与 `parent`

每个 `Contract` 实例都由三个核心属性定义，它们共同赋予了数据契约强大的表达力和验证能力。

-   **`name` (str): 契约的唯一身份**
    -   **唯一性与注册**: `name` 是契约在全局注册表中的唯一键。系统在初始化时会检查名称冲突，确保每个契约都有一个明确无误的身份。这使得我们可以通过 `Contract.get('ContractName')` 在任何地方引用一个已定义的契约。
    -   **可读性与调试**: 在 `FuzzyPipeline` 构建失败或打印调试信息时，`name` 提供了人类可读的标识。错误信息如“期望 `ContractCrispTable`，但收到了 `ContractFuzzyTable`”清晰地指明了问题所在，极大地简化了调试过程。

-   **`validator` (Callable[[Any], bool]): 契约的验证逻辑**
    -   **规则的执行者**: `validator` 是一个函数，它封装了判断一个数据对象是否符合契约规范的全部逻辑。这个函数是契约的“灵魂”，它定义了“是什么”的问题。例如，`ContractCrispTable` 的 `validator` 会检查对象是否为 `pd.DataFrame` 并且所有列都是数值类型。
    -   **运行时保障**: 虽然 `FuzzyPipeline` 的主要优势在于构建时检查，但 `validator` 仍然可以在需要时被调用，为系统提供最终的运行时安全保障。

-   **`parent` (Optional[Contract]): 契约的继承与兼容性**
    -   **建立“is-a”关系**: `parent` 属性是数据契约系统实现多态性和灵活性的关键。它在契约之间建立了一种“is-a”（是一种）的继承关系。例如，`ContractNormalizedWeights` 的 `parent` 是 `ContractWeightVector`，这在语义上表示“一个归一化的权重向量 **是一种** 权重向量”。
    -   **驱动兼容性检查**: 这个继承关系是 `is_compatible_with()` 方法的核心。当管道连接两个组件时，它会检查下游组件要求的输入契约（`required_contract`）与上游组件承诺的输出契约（`provided_contract`）。检查逻辑如下：
        
        ```python
        provided_contract.is_compatible_with(required_contract)
        ```

        如果 `provided_contract` 就是 `required_contract`，或者 `required_contract` 出现在 `provided_contract` 的任何祖先链（parent, parent's parent, ...）中，则检查通过。
    -   **实现多态管道**: 这意味着，一个设计为接受通用数据类型（如 `ContractWeightVector`）的组件，可以无缝地处理更具体的数据类型（如 `ContractNormalizedWeights` 或 `ContractScoreVector`），只要后者将前者声明为其父契约。这大大增强了组件的重用性和管道的灵活性，允许开发者构建更加通用和强大的分析工作流。

### 1.2 关键方法

-   **`validate(obj: Any) -> bool`**: 执行验证逻辑。
-   **`is_compatible_with(required_contract: Contract) -> bool`**: 判断当前契约是否与另一个“被要求的”契约兼容。兼容性规则是：
    -   如果两个契约是同一个对象，则兼容。
    -   如果当前契约的父链中的任何一个父契约是被要求的契约，则兼容。
    这个方法是管道在连接组件时进行静态检查的基础。

### 1.3 示例：创建一个自定义契约

```python
from axisfuzzy.analysis.contracts import Contract
import numpy as np

# 定义一个契约，要求数据是一个归一化的权重向量（和为1）
ContractNormalizedWeights = Contract(
    name='NormalizedWeights',
    validator=lambda obj: (
        isinstance(obj, np.ndarray) and
        obj.ndim == 1 and
        np.isclose(np.sum(obj), 1.0)
    ),
    parent=Contract.get('WeightVector') # 假设 WeightVector 已定义
)
```

---

## 2. `@contract` 装饰器：从类型提示到契约

`@contract` 装饰器是连接 Python 原生类型提示与 `analysis` 系统契约机制的桥梁。它允许开发者使用标准的 Python 语法来声明组件的输入和输出契约，极大地简化了组件的开发。

### 2.1 工作原理

`@contract` 装饰器在函数定义时被调用，它会：
1.  **检查函数的类型注解 (Type Hints)**：使用 `get_type_hints` 来解析函数签名中的参数和返回值类型。
2.  **提取输入契约**：遍历所有参数，如果某个参数的类型注解是一个 `Contract` 实例，就将其记录为输入契约。
3.  **提取输出契约**：
    -   如果返回值的类型注解是一个 `Contract` 实例，则认为这是一个单输出组件，输出名称默认为 `output`。
    -   如果返回值的类型注解是 `Dict[str, Contract]` 的形式，则认为这是一个多输出组件。
4.  **附加元数据**：将解析出的输入和输出契约信息，作为内部属性（`_contract_inputs`, `_contract_outputs`）附加到被装饰的函数上。

`FuzzyPipeline` 在 `add()` 一个新步骤时，会检查并使用这些元数据来构建和验证计算图。

### 2.2 使用示例

```python
from axisfuzzy.analysis.contracts import contract
from .build_in import ContractCrispTable, ContractWeightVector, ContractScoreVector
from typing import Dict

# 单输入，单输出
@contract
def calculate_scores(data: ContractCrispTable, weights: ContractWeightVector) -> ContractScoreVector:
    # ... 实现 ...
    pass

# 单输入，多输出
@contract
def analyze_data(data: ContractCrispTable) -> Dict[str, ContractScoreVector]:
    # ... 实现 ...
    # 返回一个字典，例如 {'scores': score_vector, 'confidence': confidence_vector}
    pass
```

---

## 3. 内置契约库 (`build_in.py`)

为了方便开发者，`analysis` 模块提供了一个丰富的内置契约库，这些契约覆盖了从基础数据类型到模糊数据结构的各种常见场景。

### 3.1 基础契约

-   `ContractAny`: 接受任何类型的“万能”契约。
-   `ContractCrispTable`: 要求是纯数值的 `pandas.DataFrame`。
-   `ContractFuzzyTable`: 要求是 `FuzzyDataFrame`。
-   `ContractWeightVector`: 要求是 `np.ndarray` 或 `pd.Series` 的一维向量。
-   `ContractMatrix`: 要求是 `np.ndarray` 或 `pd.DataFrame` 的二维矩阵。
-   `ContractFuzzyNumber`: 要求是 `Fuzznum`。
-   `ContractFuzzyArray`: 要求是 `Fuzzarray`。

### 3.2 派生契约

许多契约利用 `parent` 属性构建了继承关系，例如：

-   `ContractScoreVector` 继承自 `ContractWeightVector`。
-   `ContractNormalizedWeights` 继承自 `ContractWeightVector`，并增加了和为 1 的约束。
-   `ContractPairwiseMatrix` 继承自 `ContractMatrix`，并增加了行列数相等的约束。

这种设计使得契约检查更加灵活。一个需要 `ContractWeightVector` 的组件，同样可以接受 `ContractScoreVector` 或 `ContractNormalizedWeights` 作为输入。

---

## 4. 契约在系统中的应用

数据契约的思想贯穿了整个 `analysis` 模块，主要体现在 `FuzzyPipeline` 和 `AnalysisComponent` 中。

### 4.1 在 `FuzzyPipeline` 中的应用

`FuzzyPipeline` 在构建计算图（调用 `add()` 方法）时，会执行严格的契约检查。

1.  **获取契约**：从待添加组件的 `_contract_inputs` 和 `_contract_outputs` 属性中获取其输入输出契约。
2.  **检查依赖**：对于组件的每一个输入参数，找到其连接的上游步骤的输出 (`StepOutput`)。
3.  **验证兼容性**：调用 `is_compatible_with()` 方法，检查上游步骤**承诺 (Promised)** 的输出契约是否与当前组件**要求 (Required)** 的输入契约兼容。
4.  **失败则报错**：如果不兼容，`add()` 方法会立即抛出 `TypeError`，并提供详细的错误信息，明确指出哪个组件的哪个参数期望什么契约，但收到了什么契约。

这种**构建时验证 (Build-Time Validation)** 机制，确保了只有在所有数据流都满足契约要求的情况下，管道才能被成功创建。


### 4.2 工作流程：从注解到元数据

让我们以 `basic.py` 中的 `ToolNormalization` 组件为例，深入剖析 `@contract` 的工作流程。

```python
    # In axisfuzzy/analysis/component/basic.py

    from ..contracts import contract
    from ..build_in import ContractCrispTable
    from .base import AnalysisComponent

    class ToolNormalization(AnalysisComponent):
        # ... __init__ and get_config methods ...

        @contract
        def run(self, data: ContractCrispTable) -> ContractCrispTable:
            # ... normalization logic ...
            return normalized_data
```

当 Python 解释器加载这段代码时，`@contract` 装饰器会立即执行以下步骤：

1.  **检查函数签名**: 装饰器接收到 `run` 方法作为参数，并使用 `inspect` 和 `get_type_hints` 来解析其完整的函数签名，包括参数和返回值的类型注解。

2.  **提取输入契约**: 它遍历所有参数（除了 `self`），发现 `data` 参数的类型注解是 `ContractCrispTable`。由于 `ContractCrispTable` 是一个 `Contract` 实例，装饰器将其识别为一个输入契约。

3.  **提取输出契约**: 接着，它检查返回值的类型注解，发现是 `ContractCrispTable`。这被识别为输出契约。对于单个返回值，输出的逻辑名称默认为 `'output'`。

4.  **附加元数据**: 最关键的一步，装饰器将解析到的契约信息处理后，作为内部属性附加到 `run` 函数对象上：
    -   `setattr(run, '_contract_inputs', {'data': 'ContractCrispTable'})`
    -   `setattr(run, '_contract_outputs', {'output': 'ContractCrispTable'})`
    -   `setattr(run, '_is_contract_method', True)`

    注意，存储的是契约的 `name` (字符串)，而不是 `Contract` 对象本身。这使得系统可以在任何地方通过 `Contract.get(name)` 来获取契约实例，避免了循环引用和序列化问题。

这使得 `ToolNormalization` 成为了一个自包含、可验证、易于集成到任何 `FuzzyPipeline` 中的标准组件。

### 4.3 与 `FuzzyPipeline` 的协作

当您构建一个 `FuzzyPipeline` 时，管道引擎会执行以下操作：

1.  **遍历组件**: 管道会检查您传入的每一个 `AnalysisComponent` 实例。
2.  **查找契约方法**: 它会寻找组件中被 `@contract` 标记过的方法（即拥有 `_is_contract_method` 属性的方法，通常是 `run`）。
3.  **读取元数据**: 管道引擎读取 `_contract_inputs` 和 `_contract_outputs` 属性，从而确切地知道每个组件在数据流中的“输入插槽”和“输出插槽”分别需要和提供什么类型的契约。
4.  **执行兼容性检查**: 在连接两个组件时（例如 `pipe.add(component_A).add(component_B)`），管道会使用 `is_compatible_with` 方法，检查 `component_A` 的输出契约是否与 `component_B` 的输入契约兼容。

通过这种方式，`@contract` 装饰器、`AnalysisComponent` 和 `FuzzyPipeline` 形成了一个无缝协作的体系。组件开发者可以专注于实现 `run` 方法的逻辑，并通过简单的类型注解来声明其数据接口，而将所有复杂的验证和连接逻辑完全委托给管道引擎。这种设计极大地提高了代码的模块化程度和可维护性。

---

总结而言，数据契约系统是 `axisfuzzy.analysis` 模块实现声明式、健壮管道分析的核心。它通过 `Contract` 类定义规范，通过 `@contract` 装饰器简化组件开发，并通过 `FuzzyPipeline` 在图构建时强制执行这些规范，最终构建出一个既灵活又可靠的模糊数据分析框架。
