# AxisFuzzy 核心模块 (`axisfuzzy.core`) 架构总览

`axisfuzzy.core` 是 `AxisFuzzy` 库的引擎与灵魂，它不仅定义了模糊数（`Fuzznum`）和模糊数组（`Fuzzarray`）的核心数据结构，更构建了一套完整、高性能且高度可扩展的运算与类型管理框架。本篇文档旨在提供一个自包含的、详尽的架构蓝图，将所有核心概念融会贯通，阐明各组件的设计哲学及其协同工作的方式。

## 核心设计哲学

`core` 模块的设计严格遵循四大原则，这些原则贯穿于每一个组件的实现之中：

1.  **分离关注点 (Separation of Concerns)**：这是最重要的顶层设计。用户接口（`Fuzznum`, `Fuzzarray`）与具体实现（`FuzznumStrategy`, `FuzzarrayBackend`）彻底解耦。这使得用户可以享受简洁的 API，而开发者则能专注于特定模糊类型的内部逻辑，互不干扰。
2.  **性能优先 (Performance First)**：模糊计算的性能瓶颈在于数组操作。我们通过基于 NumPy 的 **Struct of Arrays (SoA)** 后端设计，将模糊数组的不同参数（如隶属度、非隶属度）存储在独立的 `ndarray` 中，从而实现了高效的向量化运算，确保了性能闭环。
3.  **高度可扩展 (High Extensibility)**：`AxisFuzzy` 的生命力在于其适应性。通过两大注册表（`FuzznumRegistry` 和 `OperationScheduler`），用户可以像插入插件一样，轻松定义和集成新的模糊数类型及其运算，而无需触及任何核心代码。
4.  **优雅的用户体验 (Elegant User Experience)**：通过运算符重载、`operate` 智能分派器和统一的工厂函数（`fuzznum`, `fuzzarray`），我们为用户提供了与 Python 内置数值类型（`int`, `float`）几乎无异的编程体验，实现了“约定优于配置”。

## 核心架构图

```
+--------------------------------------------------+
|              User-Facing Layer                   |
|               (用户接口层)                       |
|--------------------------------------------------|
| - Fuzznum (门面/代理)                            |
| - Fuzzarray (高性能容器)                         |
+--------------------------------------------------+
                  |
                  v
+--------------------------------------------------+
|            Implementation Layer                  |
|               (具体实现层)                       |
|--------------------------------------------------|
| - FuzznumStrategy (单个模糊数的策略与数据契约)   |
| - FuzzarrayBackend (模糊数组的SoA后端)           |
+--------------------------------------------------+
                  |
                  v
+--------------------------------------------------+
|         Registry & Discovery Layer               |
|              (注册与发现层)                      |
|--------------------------------------------------|
| - FuzznumRegistry (类型注册中心)                 |
| - OperationScheduler (运算调度中心)              |
+--------------------------------------------------+
                  |
                  v
+--------------------------------------------------+
|          Operation & Dispatch Layer              |
|               (运算与调度层)                     |
|--------------------------------------------------|
| - operate() (智能分派器)                         |
| - OperationMixin (运算原子单元)                  |
+--------------------------------------------------+
                  |
                  v
+--------------------------------------------------+
|           Mathematical Foundation                |
|                (数学基础层)                      |
|--------------------------------------------------|
| - OperationTNorm (T-范数工厂)                    |
+--------------------------------------------------+
```

---

## 架构组件深度解析

### **第一层：用户接口层 (User-Facing Layer)**

这一层是用户直接交互的门面，提供了简洁、统一的 API。

-   **`Fuzznum`**: 它是单个模糊数的统一接口，扮演着**门面 (Facade)** 和**动态代理 (Proxy)** 的双重角色。
    -   **门面**：用户只需通过 `fuzznum(mtype=...)` 工厂函数创建实例，无需关心背后复杂的实现。
    -   **代理**：`Fuzznum` 对象本身是轻量级的，不存储任何模糊数参数（如隶属度）。它内部持有一个对应的 `FuzznumStrategy` 实例，并将所有属性访问和方法调用（如 `.membership` 或 `__add__`）无缝地转发给该策略实例处理。这种设计将接口与实现彻底分离。

-   **`Fuzzarray`**: 它是模糊数的高性能容器，其设计思想和接口行为与 NumPy 的 `ndarray` 高度相似。
    -   它为同质化的模糊数集合（所有元素的 `mtype` 必须相同）提供了向量化操作的能力。
    -   它内部管理着一个 `FuzzarrayBackend` 实例，所有的数据存储和计算任务都委托给后端处理，从而实现了高性能。

### **第二层：具体实现层 (Implementation Layer)**

这一层是特定模糊类型“业务逻辑”的所在地，定义了数据如何存储和操作。

-   **`FuzznumStrategy`**: 定义了单个模糊数**如何存储数据、如何验证参数、如何初始化**的“策略”或“契约”。
    -   它是一个抽象基类，通过 `__init_subclass__` 钩子，从子类的类属性和类型注解中收集一个 `_declared_attributes` 列表。
    -   这个列表定义了该模糊数类型应有的数据字段。`__setattr__` 方法会使用这个列表来约束属性的赋值，防止意外添加新属性。
    -   具体的策略子类（如 `QROFNStrategy`、`QROHFNStrategy`）直接在类体中声明类属性（如 `md`、`nmd` 等）。基类的 `_declared_attributes` 机制已经提供了属性约束功能。

-   **`FuzzarrayBackend`**: `Fuzzarray` 的性能核心，是其 SoA（Struct of Arrays）设计模式的体现。
    -   它不将多个模糊数的参数打包在一起（Array of Structs, AoS），而是将所有模糊数的同一参数存储在单个连续的 NumPy 数组中。例如，一个包含 1000 个 `qrofn` 的 `Fuzzarray`，其后端会包含一个 `(1000,)` 的 `memberships` 数组和一个 `(1000,)` 的 `nonmemberships` 数组。
    -   这种 SoA 结构是实现向量化计算的关键，使得 NumPy 的底层 SIMD 优化能够被充分利用。
    -   它定义了两条核心的构造路径：
        1.  **用户友好路径** (`_build_backend_from_data`)：从原始的 Python 列表或迭代器创建，内部会进行数据校验和转换，方便用户使用。
        2.  **快速路径** (`create_from_backend_data`)：从已经准备好的 NumPy 数组直接创建，几乎没有性能开销。这条路径在运算过程中被频繁使用，以保证性能闭环。

### **第三层：注册与发现层 (Registry & Discovery Layer)**

这一层是框架的“大脑”，负责类型的动态注册、发现和运算的调度，是 `AxisFuzzy` 高度可扩展性的基石。

-   **`FuzznumRegistry`**: **类型注册中心**。它是一个线程安全的**单例 (Singleton)**，负责维护 `mtype`（字符串）到其对应的 `FuzznumStrategy` 和 `FuzzarrayBackend` 类的映射。
    -   当用户调用 `fuzznum('qrofn', ...)` 时，该工厂函数会查询 `FuzznumRegistry`，找到 `qrofn` 对应的 `QROFNStrategy` 类并实例化。
    -   注册通常通过 `@register_fuzznum` 装饰器在定义新类型时自动完成。

-   **`OperationScheduler`**: **运算调度中心**。它同样是一个单例，负责管理系统中所有可用的模糊运算。
    -   它维护着一个 `(operation_name, mtype)` 元组到具体 `OperationMixin` 实现类的映射。例如，`('add', 'qrofn')` -> `QROFNAddition`。
    -   当运算发生时，系统会向它查询正确的运算处理器。
    -   它还负责管理全局配置，例如可以统一设置项目中默认使用的 T-范数类型。

### **第四层：运算与调度层 (Operation & Dispatch Layer)**

这一层是执行计算的“双手”，负责将用户的操作意图转化为具体的计算任务。

-   **`OperationMixin`**: **运算的原子单元**。每一个具体的模糊运算（如加法、乘法、求逆）都必须由一个继承自 `OperationMixin` 的类来实现。
    -   它定义了运算的核心逻辑，例如，`QROFNAddition` 会定义如何根据两个 `qrofn` 的隶属度和非隶属度计算新的隶属度和非隶属度。
    -   它会与数学基础层（`OperationTNorm`）交互，以执行底层的模糊逻辑运算。

-   **`operate` 函数**: **智能运算分派器**。这是整个运算体系的“总指挥”。
    -   当用户执行 `a + b` 时，Python 的运算符重载机制会调用 `a.__add__(b)`，而这个方法内部最终会调用 `operate('add', a, b)`。
    -   `operate` 函数会智能地分析 `a` 和 `b` 的类型，并根据一套预设的**九大分派规则**选择最高效的执行路径。这些规则覆盖了 `(Fuzzarray, Fuzzarray)`、`(Fuzzarray, Fuzznum)`、`(Fuzznum, Fuzzarray)`、`(Fuzzarray, scalar)` 等所有可能的组合，并能正确处理广播机制。

### **第五层：数学基础层 (Mathematical Foundation Layer)**

这一层为所有模糊运算提供了可插拔的、经过验证的数学引擎。

-   **`OperationTNorm`**: **T-范数与T-余范数的工厂**。
    -   模糊逻辑中的“与”（交集）和“或”（并集）操作分别由 T-范数 (t-norm) 和 T-余范数 (t-conorm) 定义。
    -   `OperationTNorm` 以工厂模式实现了多种标准的 T-范数族（如 `algebraic`, `einstein`, `lukasiewicz` 等），并提供了统一的接口。
    -   它还为 q-rung 阶模糊集提供了专门的 q-阶扩展，确保了数学上的严谨性。
    -   `OperationMixin` 的实现会依赖一个 `OperationTNorm` 实例来执行底层的数学计算，这使得更换模糊逻辑的“引擎”变得非常简单。

---

## 融会贯通：一次 `fuzzarray + fuzznum` 的生命周期

让我们以 `qrofs` 为例，追踪一次 `Fuzzarray` 和 `Fuzznum` 相加的完整旅程，看看这些组件是如何无缝协作的。

1.  **启动与注册 (System Bootstrapping)**
    -   当 `AxisFuzzy` 初始化时，`axisfuzzy.fuzztype.qrofs` 模块被加载。
    -   `@register_fuzznum('qrofn', ...)` 装饰器执行，将 `QROFNStrategy` 和 `QROFNBackend` 注册到 `FuzznumRegistry`。
    -   `@register_operation('add', 'qrofn')` 装饰器执行，将 `QROFNAddition` 类注册到 `OperationScheduler`。

2.  **对象创建 (Object Creation)**
    -   用户执行 `arr = fuzzarray('qrofn', ...)` 和 `num = fuzznum('qrofn', ...)`。
    -   `fuzzarray` 工厂函数查询 `FuzznumRegistry`，找到 `QROFNBackend`，并通过其**用户友好路径** (`_build_backend_from_data`) 创建后端实例，最终封装成 `Fuzzarray` 对象 `arr`。
    -   `fuzznum` 工厂函数同样查询注册表，找到并实例化 `QROFNStrategy`，然后将其包裹在 `Fuzznum` 门面 `num` 中。

3.  **运算触发与分派 (Operation & Dispatch)**
    -   用户执行 `result = arr + num`。
    -   `Fuzzarray` 的 `__add__` 方法被调用，它立即将任务转交给 `operate('add', arr, num)`。
    -   `operate` 分派器启动，分析操作数类型。它匹配到 `(Fuzzarray, Fuzznum)` 的分派规则，并识别出需要进行广播运算。

4.  **执行与计算 (Execution & Calculation)**
    -   `operate` 函数最终会调用 `arr` 的 `execute_vectorized_op` 方法。
    -   该方法向 `OperationScheduler` 请求 `('add', 'qrofn')` 的运算器，获得了 `QROFNAddition` 的实例。
    -   `QROFNAddition` 的 `execute_binary_op_on_array` 方法被调用。它接收 `arr` 的 SoA 后端（NumPy 数组）和 `num` 的策略对象（从中提取标量值）。
    -   在 `QROFNAddition` 内部，它使用一个 `OperationTNorm` 实例（例如，默认的代数 T-范数）对 NumPy 数组和标量进行向量化的 T-范数和 T-余范数计算。

5.  **结果封装与返回 (Result Wrapping & Return)**
    -   计算产生了一组新的 NumPy 数组（结果的隶属度和非隶属度）。
    -   `arr` 的后端 `QROFNBackend` 调用其 `create_from_backend_data` 方法（**快速路径**），直接用这组新的 NumPy 数组创建一个新的后端实例，这几乎是零成本的。
    -   这个新的后端实例被封装在一个新的 `Fuzzarray` 对象中返回给用户。

这个流程完美地展示了 `AxisFuzzy` 的设计精髓：**从用户优雅的接口出发，通过智能分派和类型注册，最终落脚于高性能的向量化计算，并将结果通过高效路径再次封装成用户熟悉的对象，形成了一个无缝、高效且可扩展的闭环。**