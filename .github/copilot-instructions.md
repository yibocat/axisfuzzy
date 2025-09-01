# AxisFuzzy 指南

本指南旨在帮助 AI 编码代理快速理解 AxisFuzzy 代码库的核心架构、关键模式和开发工作流程。

## 1. 核心架构概览

AxisFuzzy 是一个用于模糊逻辑和模糊计算的 Python 库。其核心设计理念是**模块化、可扩展性与高性能**。整个系统围绕着 `Fuzznum` (模糊数) 和 `Fuzzarray` (模糊数组) 这两个核心数据结构构建，并通过一系列可插拔的子系统进行功能扩展。

*   **核心数据结构**:
    *   **`Fuzznum` (`axisfuzzy/core/fuzznums.py`)**: 这是一个面向用户的**门面 (Facade)** 和**代理 (Proxy)**。它为单个模糊数提供了统一的接口，但其内部的计算逻辑则委托给特定 `mtype` 的 `FuzznumStrategy`。
    *   **`Fuzzarray` (`axisfuzzy/core/fuzzarray.py`)**: 作为高维 `Fuzznum` 的容器，其角色类似于 `numpy.ndarray`。为了实现高性能的批量计算，它由一个高效的 **SoA (Struct of Arrays) 后端 (`axisfuzzy/core/backend.py`)** 支持，避免了在 Python 层面进行对象迭代。

*   **核心抽象与注册**:
    *   **`FuzznumStrategy` (`axisfuzzy/core/base.py`)**: 这是实现新模糊数类型的关键抽象。
        *   `FuzznumStrategy`: 负责核心的数学计算和逻辑运算和约束条件。
    *   **`FuzznumRegistry` (`axisfuzzy/core/registry.py`)**: 这是一个中央注册表，用于注册和管理系统中所有可用的模糊数类型 (`mtype`) 及其对应的 `Strategy` 和 `backend`。
    *   **`FuzzarrayBackend` (`axisfuzzy/core/backend.py`)**: 这是 `Fuzzarray` 的后端实现，负责高效的批量计算。
    *   **`OperationMixin`(`axisfuzzy/core/operation`)**: 这是数学计算的核心基类, 为所有数学运算提供统一接口。
    *   **`OperationScheduler`(`axisfuzzy/core/operation.py`)**: 负责调度和管理所有的模糊数的数学运算任务，确保高效执行。

*   **核心功能**:
    *   **操作分发 (`axisfuzzy/core/dispatcher.py`)**: 负责处理 `Fuzznum` 和 `Fuzzarray` 之间的二元运算。它能够根据操作数的类型（`Fuzznum`, `Fuzzarray`, `scalar`）智能地分发到最高效的实现路径。
    *   **t-范数 (`axisfuzzy/core/triangular.py`)**: 提供了一系列标准的 t-范数和 t-协范数（如 `algebraic`, `hamacher`, `lukasiewicz`）。这些是所有模糊逻辑运算的数学基础。

*   **特定模糊数实现 (`axisfuzzy/fuzzy`)**:
    *   这是各种具体模糊数类型 (`mtype`) 的实现所在地。每个子目录（如 `qrofs`）都包含了该类型模糊数所需的所有组件：后端、操作、随机生成器、扩展功能等。
    *   例如，`axisfuzzy/fuzzy/qrofs/` 包含了 q-rung 直觉模糊数 (`qrofn`) 的所有实现。

## 2. 分析系统 (Analysis System)

**目的**: 提供一个类似 PyTorch `nn.Module` 的高级 API，用于构建、组合和重用复杂的数据分析工作流。

*   **核心理念**: **"组件化"** 和 **"图构建"**。用户通过继承 `Model` 类来定义一个分析模型，在 `__init__` 中将可重用的 `AnalysisComponent`（或其他 `Model`）实例化为属性，然后在 `forward` 方法中像调用普通函数一样将它们连接起来，从而定义数据流图。
*   **架构分层**:
    *   **模型 (`axisfuzzy/analysis/app/model.py`)**:
        *   **`Model` 类**: 所有分析模型的基类。它会自动追踪 `forward` 方法中的调用，并将其编译成一个底层的 `FuzzyPipeline`。
        *   **序列化**: `Model` 支持通过 `save()` 和 `load()` 方法进行持久化。该机制遵循 **“代码与配置分离”** 的原则，将模型架构（类路径和构造参数）保存为人类可读的 JSON 文件，而不是使用不安全、不稳定的 `pickle`。这要求所有子组件都必须是可序列化的。
    *   **组件 (`axisfuzzy/analysis/component/`)**:
        *   **`AnalysisComponent` (`base.py`)**: 所有分析组件（“层”）的抽象基类。
        *   **`basic.py`**: 提供了一系列内置的基础组件，如 `ToolNormalization`, `ToolSimpleAggregation`, `ToolFuzzification` 等。
        *   **可序列化契约**: 每个组件都必须实现 `get_config()` 方法，返回一个可用于重建该组件的、JSON 兼容的字典。
    *   **数据契约 (`axisfuzzy/analysis/contracts/`)**: 定义了数据在 `forward` 图中流动时的类型规范，例如 `ContractCrispTable`。
    *   **管道引擎 (`axisfuzzy/analysis/pipeline.py`)**:
        *   **`FuzzyPipeline`**: `Model` 的底层执行引擎，负责管理由 `forward` 方法定义的有向无环图 (DAG)。

## 3. 模糊化系统 (Fuzzification System)

**目的**: 提供一个统一、可扩展、高性能的入口，将精确的标量或数组值转换为 `Fuzznum` 或 `Fuzzarray`。这是从经典数据域到模糊数据域的关键桥梁。

*   **核心理念**: **"配置与执行分离"** 和 **"策略模式"**。用户首先配置并实例化一个 `Fuzzifier` 对象，这个对象就像一个预设好参数的“模糊化引擎”。配置内容包括隶属函数类型、目标模糊数类型 (`mtype`)、具体的模糊化方法 (`method`) 以及所有相关参数。一旦创建，这个 `Fuzzifier` 实例就可以被重复调用，以相同的配置来处理不同的数据输入，极大地提高了代码的复用性和可读性。
*   **可序列化**: `Fuzzifier` 自身支持 `get_config()` 和 `from_config()`，使其可以被轻松地集成到 `AnalysisComponent` (如 `ToolFuzzification`) 和 `Model` 的序列化流程中。

*   **架构分层**:
    *   **API (`axisfuzzy/fuzzifier/fuzzifier.py`)**:
        *   **`Fuzzifier` 类**: 这是面向用户的核心 API 和可复用的模糊化引擎。
        *   **初始化**: 在构造时，`Fuzzifier` 接收隶属函数 (`mf`)、目标 `mtype`、策略 `method` 以及一个关键的 `mf_params` 参数（一个字典或字典列表，用于配置隶属函数）。它会根据 `mtype` 和 `method` 从注册表中查找并实例化对应的模糊化策略。
        *   **执行**: `Fuzzifier` 对象本身是可调用的 (`__call__`)。当传入一个精确值或数组 `x` 时，它会调用内部策略实例的 `fuzzify` 方法，并将 `x`、隶属函数类和 `mf_params` 传递给策略，最终完成模糊化过程。

    *   **隶属函数 (`axisfuzzy/membership/`)**: 提供一个可扩展的隶属函数库。这部分是模糊化系统的基础依赖，其架构保持不变。
        *   `MembershipFunction` (`base.py`): 所有隶属函数的抽象基类。
        *   `function.py`: 包含标准隶属函数（如 `TriangularMF`, `GaussianMF`）的实现。
        *   `factory.py`: 提供 `get_mf_class` 等工厂函数，支持通过字符串名称获取隶属函数类。

    *   **注册表 (`axisfuzzy/fuzzifier/registry.py`)**:
        *   **`FuzzificationStrategyRegistry`**: 维护一个从 `(mtype, method)` 元组到具体 `FuzzificationStrategy` 类的映射。这使得系统可以为同一个 `mtype` 支持多种不同的模糊化方法（例如，一个用于专家打分，一个用于数据驱动）。
        *   **`@register_fuzzifier`**: 这是一个装饰器，用于向注册表注册新的策略类。装饰器会自动从被装饰的策略类中读取 `mtype` 和 `method` 属性来作为注册的键。它还支持将某个方法设置为特定 `mtype` 的默认方法。

    *   **策略基类 (`axisfuzzy/fuzzifier/strategy.py`)**:
        *   **`FuzzificationStrategy`**: 定义了所有模糊化策略必须遵循的接口。
        *   **属性**: 任何具体的策略实现都必须定义 `mtype: str` 和 `method: str` 这两个类属性，以便注册系统能够识别它们。
        *   **核心接口**: 定义了抽象方法 `fuzzify(self, x, mf_cls, mf_params_list)`。这个方法是实际执行模糊化逻辑的地方，它接收原始数据 `x`、隶属函数类 `mf_cls` 和一个隶属函数参数列表 `mf_params_list`，并返回一个 `Fuzznum` 或 `Fuzzarray`。

    *   **具体实现**: 每种 `mtype` 的具体模糊化策略位于其自己的模块内（例如 `axisfuzzy/fuzzy/qrofs/fuzzification.py`）。策略类继承自 `FuzzificationStrategy`，实现 `fuzzify` 方法，并使用 `@register_fuzzifier` 装饰器向系统注册自己。

## 4. 扩展系统 (Extension System)

**目的**: 为**不同 `mtype`** 的模糊数定义和注入特化的外部功能（如距离计算、相似度度量、得分函数等）。

*   **核心理念**: "注册-分发-注入"。功能被定义和注册，然后根据上下文分发，最后在初始化时动态注入到核心类中。
*   **组件**:
    *   **注册表 (`axisfuzzy/extension/decorator.py`, `registry.py`)**:
        *   使用 `@extension` 装饰器来声明一个函数是扩展功能。
        *   装饰器捕获元数据（功能名称、目标 `mtype`、注入目标类等）并将函数注册到 `ExtensionRegistry` 中。
        *   支持 `mtype` 特化（函数只对特定 `mtype` 生效）、默认实现回退（当没有特化实现时使用）和优先级排序。
        *   示例: `@extension('distance', mtype='qrofn', target_classes=['Fuzznum'])`
    *   **分发 (`axisfuzzy/extension/dispatcher.py`)**:
        *   当一个扩展功能被调用时（例如 `my_fuzznum.distance(...)`），分发器会介入。
        *   它检查 `my_fuzznum` 的 `mtype`，并在注册表中查找最匹配的实现（`mtype`='qrofn' 的 'distance' 函数）。
        *   提供 `call_extension()` 辅助函数，用于在扩展函数内部安全地调用其他扩展函数，从而避免循环导入和注入时序问题。
    *   **注入 (`axisfuzzy/extension/injector.py`)**:
        *   在库初始化时（通过 `axisfuzzy.init()`），注入器会遍历注册表。
        *   它将所有注册的函数动态地附加到它们的目标类（如 `Fuzznum`, `Fuzzarray`）或 `axisfuzzy` 顶级命名空间上，使其看起来就像是原生方法或函数。
    *   **扩展定义**: 
        *   内部扩展通常定义在特定 `mtype` 的模块中，例如 `axisfuzzy/fuzzy/qrofs/factory/`。再通过 `axisfuzzy/fuzzy/qrofs/extension.py` 注册。
        *   用户可以在外部定义自己的扩展，只需导入 `@extension` 装饰器并确保其模块被导入即可。

## 5. 混合功能系统 (Mixin System)

**目的**: 为 `Fuzznum` 和 `Fuzzarray` 提供**与 `mtype` 无关**的、类似 `NumPy` 的结构化操作方法。

*   **核心理念**: 关注数据的形状和结构，而非其模糊语义。这些操作对于任何 `mtype` 的 `Fuzzarray` 都是通用的。
*   **与扩展系统的区别**: 扩展系统是 `mtype` 敏感的（一个距离公式对 `qrofn` 和 `ivfn` 可能完全不同），而 Mixin 系统是 `mtype` 无关的（`reshape` 操作对任何类型的数组都一样）。
*   **组件 (`axisfuzzy/mixin/`)**:
    *   同样采用 "注册-注入" 的模式。
    *   功能定义在 `_function.py` 和 `_ops.py` 中。
    *   使用 `@register_mixin_function` 或 `@register_mixin_op` 等装饰器进行注册。
    *   **功能示例**: `reshape`, `flatten`, `transpose`, `concatenate`, `squeeze` 等。

## 6. 随机系统 (Random System)

**目的**: 为不同 `mtype` 的模糊数提供统一、可扩展、可复现、高性能的随机数生成能力。

*   **核心理念**: 采用 SoA 后端进行批量生成，避免在 Python 循环中逐个创建 `Fuzznum` 对象带来的性能瓶颈。
*   **架构分层 (`axisfuzzy/random/`)**:
    *   **API (`api.py`)**: 提供 `random_fuzz` 等面向用户的统一入口。用户只需指定形状和 `mtype`。
    *   **注册表 (`registry.py`)**: 维护一个从 `mtype` 字符串到其对应随机生成器类的映射。
    *   **基类 (`base.py`)**: 定义了 `RandomGenerator` 的抽象基类，规定了所有生成器必须实现的统一接口（例如 `_generate` 方法）。
    *   **种子管理 (`seed.py`)**: 控制全局和局部的随机状态，确保实验的可复现性。
    *   **具体实现**: 每种 `mtype` 的具体生成逻辑位于其自己的模块内（例如 `axisfuzzy/fuzzy/qrofs/random.py`），并向随机系统的注册表注册自己。

## 7. 关键开发模式

*   **定义新的模糊数类型**:
    1.  在 `axisfuzzy/fuzzy/` 下创建新目录，例如 `your_mtype`。
    2.  在 `your_mtype` 目录中，创建 `strategy.py` 和 `backend.py`，并实现 `FuzznumStrategy` 和 `FuzzarrayBackend` 的子类。
    3.  在 `axisfuzzy/core/registry.py` 中导入并注册你的新类型。
    4.  在 `axisfuzzy/fuzzy/your_mtype/op.py` 中定义针对该 `mtype` 的基于 `OperationMixin` 的核心数学运算逻辑子类,并通过 `@register_operation` 注册以及在 `OperationScheduler` 进行调度和管理。
    5.  在 `your_mtype` 目录中实现 `random.py` 并注册随机生成器。
    6.  在 `your_mtype` 目录中实现 `extension.py`，使用 `@extension` 装饰器定义该类型特有的功能。
    7.  确保 `axisfuzzy/fuzzy/your_mtype/__init__.py` 导入了需要注册的模块。

*   **添加新的 `mtype` 特化功能 (使用 Extension 系统)**:
    1.  找到对应 `mtype` 的 `extension.py` 文件（例如 `axisfuzzy/fuzzy/qrofs/extension.py`）。
    2.  编写功能函数，并使用 `@extension` 装饰器注册它。清晰地指定 `name`, `mtype`, `target_classes` 和 `injection_type`。
    3.  如果需要调用其他扩展，请使用 `axisfuzzy.extension.dispatcher.call_extension()`。

*   **添加新的通用数组方法 (使用 Mixin 系统)**:
    1.  在 `axisfuzzy/mixin/_function.py` 或 `_ops.py` 中添加你的函数。
    2.  使用 `@register_mixin_function` 或 `@register_mixin_op` 装饰器注册它。

*   **构建分析模型 (使用 Analysis 系统)**:
    1.  创建一个继承自 `axisfuzzy.analysis.app.model.Model` 的新类。
    2.  在 `__init__` 方法中，将所需的 `AnalysisComponent`（如 `ToolNormalization`）或其他 `Model` 实例化为类属性。
    3.  实现 `get_config()` 方法，返回重建该模型实例所需的构造参数（如果 `__init__` 有参数的话）。
    4.  在 `forward()` 方法中，使用 `self` 的属性来定义数据流图。
    5.  使用 `model.save('path/to/model.json')` 保存模型架构，使用 `Model.load()` 加载。

## 8. 注意事项

*   **注册与注入分离**: 装饰器 (`@extension` 等) 只负责将功能信息注册到相应的注册表。实际的方法注入发生在库初始化调用 `axisfuzzy.init()` 时。
*   **循环导入**: 严禁在扩展函数内部直接 `import axisfuzzy` 或其核心类。这会破坏注入流程并导致循环依赖。请始终使用内部通用函数或在函数内部进行延迟导入。
*   **序列化**: 坚持使用 JSON 进行配置序列化，避免 `pickle`。所有需要被 `Model` 保存的组件都必须实现 `get_config()` 方法并返回一个 JSON 兼容的字典。
*   **选择正确的系统**:
    *   **构建复杂工作流** -> **Analysis 系统** (e.g., `Model`, `AnalysisComponent`)。
    *   **依赖 `mtype` 的语义功能** -> **Extension 系统** (e.g., `distance`, `score`)。
    *   **不依赖 `mtype` 的结构化功能** -> **Mixin 系统** (e.g., `reshape`, `transpose`)。
    *   **`mtype` 相关的随机数生成** -> **Random 系统**。
*   **测试**: 遵循 `pytest` 约定，测试文件应位于与源代码平行的 `tests/` 目录中，并保持相似的目录结构。
