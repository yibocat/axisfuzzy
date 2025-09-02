# `AxisFuzzy` 模糊数注册指南：总览

欢迎来到 `AxisFuzzy` 的世界！本系列文档旨在为您提供一份清晰、详尽的指南，指导您如何将自定义的模糊数类型无缝集成到 `AxisFuzzy` 强大的模糊计算生态系统中。`AxisFuzzy` 的核心设计哲学之一便是高度的可扩展性，我们希望每一位研究者和开发者都能轻松地将前沿的模糊集理论转化为可计算、可应用的实践。

尽管我们已经极大地简化了新类型的集成过程，但它仍然涉及多个环环相扣的步骤。本指南将以现有的 `q-Rung Orthopair Fuzzy Number (qrofn)` 和 `q-Rung Orthopair Hesitant Fuzzy Number (qrohfn)` 为例，分步拆解整个注册流程。我们的目标不是让您复制粘贴代码，而是引导您理解每个部分背后的设计思想，从而让您能够举一反三，自如地添加任何新的模糊数类型。

---

## 核心理念：策略、后端与扩展

在深入细节之前，理解 `AxisFuzzy` 的几个核心概念至关重要：

- **策略 (Strategy)**: 每个模糊数类型都被抽象为一个“策略”。`FuzznumStrategy` 是一个基类，您的新类型需要继承它。这个策略类是模糊数的“大脑”，它定义了该类型的基本属性（如参数 `q`），验证规则（如隶属度和非隶属度的约束），以及属性变更时的回调逻辑。它是确保模糊数在数学上保持有效性的第一道防线。

- **后端 (Backend)**: 如果说策略是“大脑”，那么后端 `FuzzarrayBackend` 则是模糊数的“骨架”。它定义了模糊数在内存中的数据表示和存储方式，特别是当它们以 `Fuzzarray`（模糊数组）的形式存在时。一个高效的后端是保证 `AxisFuzzy` 在处理大规模数据时依然性能卓越的关键。

- **运算 (Operation)**: 这是模糊数“行动”的方式。`AxisFuzzy` 提供了一个强大的运算注册系统，允许您为新类型定义各种算术运算（加、减、乘、除）和逻辑运算。得益于框架的精心设计，您只需关注单一运算的数学定义，而无需关心它如何适配到 `Fuzznum`（单个模糊数）或 `Fuzzarray`（模糊数组）上。

- **扩展 (Extension)**: 除了核心的运算，一个完备的模糊数类型还需要许多辅助功能，例如：
    - **随机数生成**: 用于模拟和测试。
    - **模糊化**: 将清晰数或区间转化为模糊数。
    - **构造器**: 提供更便捷的实例创建方式。
    - **I/O**: 实现与其他格式（如 JSON、CSV）的交互。
    - **度量方法**: 如距离、相似度、得分函数等。
    - **字符串表示**: 控制模糊数的打印输出格式。

所有这些功能都通过 `axisfuzzy.extension` 系统进行模块化管理和注册，保持了核心代码的整洁。

---

## 注册工作流：一个完整流程的鸟瞰

为了更直观地展示各个组件如何协同工作，下图描绘了新模糊数类型的注册框架：

```text
+--------------------------------------------------------------------------------------+
|                                 AxisFuzzy Core Registry                              |
|                                                                                      |
|  [Strategy Registry] <---- @register_strategy ----- [ NewFuzzyTypeStrategy ]         |
|                                                     (Inherits FuzznumStrategy)       |
|                                                                                      |
|  [Backend Registry] <----- @register_backend ------ [ NewFuzzyTypeBackend ]          |
|                                                     (Inherits FuzzarrayBackend)      |
|                                                                                      |
|  [Operation Registry] <--- @register_operation ---- [ Core Operations (op.py) ]      |
|                                                     (e.g., add, subtract, multiply)  |
|                                                                                      |
+--------------------------------------------------------------------------------------+
                                           ^
                                           |
                                    (Core System)
                                           |
                                           v
+--------------------------------------------------------------------------------------+
|                               AxisFuzzy Extension System                             |
|                                                                                      |
|  [Random Registry] <------ @register_random ------- [ Random Generation (random.py) ]|
|                                                                                      |
|  [Fuzzifier Registry] <--- @register_fuzzifier ---- [ Fuzzification (fuzzifier.py) ] |
|                                                                                      |
|  [Extension Injector] <--- (Registers via API) ---- [ Other Extensions (ext/) ]      |
|                                                     (Constructors, Measures, I/O)    |
|                                                                                      |
+--------------------------------------------------------------------------------------+
```

将一个新的模糊数类型（我们称之为 `NewFuzzyType`）集成到 `AxisFuzzy` 中，通常遵循以下步骤。本系列文档的后续章节将对每一步进行详细阐述。

1.  **定义策略与后端**:
    - 创建 `newfuzzytype.py`，在其中定义 `NewFuzzyTypeStrategy` 类，继承自 `FuzznumStrategy`。
    - 创建 `backend.py`，定义 `NewFuzzyTypeBackend` 类，继承自 `FuzzarrayBackend`。
    - 使用 `@register_strategy` 和 `@register_backend` 装饰器将它们注册到 `AxisFuzzy` 的核心注册表 `core.registry` 中。

2.  **实现核心运算**:
    - 创建 `op.py`，在这里为 `NewFuzzyType` 实现加、减、乘、除等基本运算。
    - 使用 `@register_operation` 装饰器为每个运算函数进行注册，指明它属于哪个模糊数类型以及对应的运算符。

3.  **开发扩展功能**:
    - **随机生成**: 在 `random.py` 中编写函数，用于生成随机的 `NewFuzzyType` 实例，并使用 `@register_random` 注册。
    - **模糊化**: 在 `fuzzification.py` 中实现模糊化逻辑，将其他数据类型转化为 `NewFuzzyType`，并使用 `@register_fuzzifier` 注册。
    - **通用扩展**: 在 `ext/` 目录下创建多个文件（如 `constructor.py`, `measure.py`, `io.py` 等），分别实现各种扩展功能。

4.  **注册所有扩展**:
    - 创建 `extension.py`，导入所有在 `ext/` 目录下编写的扩展函数。
    - 调用 `axisfuzzy.extension` 系统的注册函数，将这些扩展与 `NewFuzzyType` 关联起来。

完成以上所有步骤后，`NewFuzzyType` 就正式成为了 `AxisFuzzy` 生态系统的一员。您可以像使用任何内置类型一样，通过字符串名称 (`'newfuzzytype'`) 来创建、计算和使用它。

---

## 接下来...

本篇总览为您描绘了 `AxisFuzzy` 模糊数注册的全景图。从下一篇文档开始，我们将深入第一个技术细节：**策略与后端**。我们将一起探索如何编写 `FuzznumStrategy` 来定义模糊数的数学约束，以及如何通过 `FuzzarrayBackend` 为其设计高效的数据结构。

准备好进入 `AxisFuzzy` 的核心，开始构建您自己的模糊数类型了吗？让我们继续前进！