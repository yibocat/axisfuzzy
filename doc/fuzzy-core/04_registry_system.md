# 4. 注册系统：`FuzznumRegistry`

`AxisFuzzy` 的核心设计理念之一是其高度的可扩展性，而实现这一点的基石便是 `FuzznumRegistry`——一个强大而灵活的注册系统。该系统位于 `axisfuzzy.core.registry` 模块，它允许开发者将自定义的模糊数实现（包括其策略和后端）无缝集成到框架中。

## 4.1. `FuzznumRegistry` 的核心职责

`FuzznumRegistry` 是一个**线程安全的单例（Singleton）类**，这意味着在整个应用程序的生命周期中，只有一个注册表实例存在。这保证了所有关于模糊数类型定义的信息都有一个单一、权威的来源。

其核心职责可以概括为以下几点：

1.  **类型映射**：维护一个从成员类型字符串（`mtype`）到具体实现类的映射。每个 `mtype` 都与两个关键组件关联：
    *   **`FuzznumStrategy`**：定义模糊数规则、约束和基本操作的策略类。
    *   **`FuzzarrayBackend`**：管理 `Fuzzarray` 的高性能、结构化数组（SoA）数据存储的后端类。

2.  **注册与管理**：提供一个中心化的接口，用于注册、查询、更新和注销模糊数类型。

3.  **保证一致性**：通过事务（Transaction）机制，确保批量注册操作的原子性，从而维护注册表状态的一致性。

4.  **动态通知**：采用观察者模式（Observer Pattern），允许系统的其他部分订阅注册表的变化事件（如新类型注册），以实现动态响应。

## 4.2. 注册一个新的模糊数类型

向 `AxisFuzzy` 添加一个新的模糊数类型是一个清晰、直接的过程，主要涉及定义 `FuzznumStrategy` 和 `FuzzarrayBackend` 的子类，然后将它们注册到 `FuzznumRegistry`。

### 步骤 1: 定义策略（Strategy）和后端（Backend）

首先，你需要为你的新模糊数类型创建具体的策略和后端实现。以 `qrofs` 为例：

**策略定义 (`qrofn.py`)**:
```python
# axisfuzzy/fuzztype/qrofs/qrofn.py

from axisfuzzy.core import FuzznumStrategy, register_strategy

@register_strategy
class QrofnStrategy(FuzznumStrategy):
    """
    q-Rung Orthopair Fuzzy Number (q-ROFN) Strategy.
    """
    mtype = 'qrofs'  # 关键：定义 mtype
    
    # ... 实现数据验证、属性等 ...
```

**后端定义 (`backend.py`)**:
```python
# axisfuzzy/fuzztype/qrofs/backend.py

from axisfuzzy.core import FuzzarrayBackend, register_backend

@register_backend
class QrofsBackend(FuzzarrayBackend):
    """
    Backend for q-Rung Orthopair Fuzzy Array.
    """
    mtype = 'qrofs'  # 关键：mtype 必须与策略中的一致
    
    # ... 实现基于 NumPy 的 SoA 存储和操作 ...
```

### 步骤 2: 使用注册装饰器

`AxisFuzzy` 提供了便捷的装饰器 `register_strategy` 和 `register_backend` 来简化注册流程。当 Python 解释器加载模块时，这些装饰器会自动获取被装饰的类，并调用 `FuzznumRegistry` 的 `register` 方法将其注册到全局注册表中。

*   `@register_strategy`: 自动注册 `FuzznumStrategy` 子类。
*   `@register_backend`: 自动注册 `FuzzarrayBackend` 子类。

这种设计极大地降低了模块间的耦合度。你只需要定义好你的类并应用装饰器，`AxisFuzzy` 的核心系统就能自动发现并集成它们。

## 4.3. 注册流程详解

无论是通过装饰器还是手动调用，注册流程都遵循以下逻辑：

1.  **获取全局注册表实例**：通过 `get_registry_fuzztype()` 函数获取 `FuzznumRegistry` 的单例实例。
2.  **调用 `register` 方法**：`register` 方法接收 `strategy` 和/或 `backend` 类作为参数。
3.  **提取 `mtype`**：注册表会从传入的类中读取 `mtype` 属性，以确定要注册的类型名称。
4.  **验证与存储**：
    *   它会检查 `mtype` 是否有效（非空字符串）。
    *   如果 `mtype` 已存在，它会发出警告，表明正在覆盖旧的实现。
    *   最后，它将 `mtype` 与对应的类存储在内部的 `strategies` 和 `backends` 字典中。
5.  **记录与通知**：
    *   记录本次注册事件的历史。
    *   更新注册统计信息。
    *   如果存在已注册的观察者，则通知它们发生了注册事件。

## 4.4. 高级功能

### 事务支持 (`transaction`)

当你需要注册多个相互依赖的组件，或者进行一系列复杂的注册操作时，保持注册表状态的原子性至关重要。`FuzznumRegistry` 的 `transaction` 上下文管理器为此提供了保障。

```python
from axisfuzzy.core import get_registry_fuzztype

registry = get_registry_fuzztype()

try:
    with registry.transaction():
        # 在这个块内的所有注册操作要么全部成功，要么全部回滚
        registry.register(strategy=MyStrategy1, backend=MyBackend1)
        registry.register(strategy=MyStrategy2, backend=MyBackend2)
        # 如果这里发生异常，上面的两次注册都会被撤销
        raise ValueError("Something went wrong")
except ValueError:
    print("Transaction failed and was rolled back.")

# 检查 'mtype1' 是否存在，结果应为 False
assert 'mtype1' not in registry.get_registered_mtypes()
```
在 `with` 块开始时，注册表会创建一个当前状态的快照。如果块内代码成功执行完毕，则更改被提交。如果发生任何异常，注册表会从快照中恢复，确保其状态未被破坏。

### 观察者模式 (`add_observer`)

观察者模式允许外部代码对注册表的内部变化做出反应。这对于构建需要动态适应新模糊数类型的工具或库非常有用。

你可以注册一个回调函数（观察者），当特定事件发生时，该函数将被调用。

```python
def my_observer(event_type, event_data):
    """一个简单的观察者函数"""
    print(f"Event '{event_type}' occurred!")
    print(f"Data: {event_data}")

# 注册观察者
registry.add_observer(my_observer)

# 当新的策略被注册时，my_observer 将被自动调用
registry.register(strategy=MyNewStrategy)

# 输出:
# Event 'register_strategy' occurred!
# Data: {'mtype': 'mynewtype', 'strategy': <class 'MyNewStrategy'>, ...}
```

## 总结

`FuzznumRegistry` 是 `AxisFuzzy` 动态性和可扩展性的核心。它不仅仅是一个简单的字典，而是一个功能完备、线程安全的组件管理中心。通过清晰的注册接口、便捷的装饰器以及强大的事务和观察者功能，它为开发者提供了一个健壮的平台，可以轻松地扩展 `AxisFuzzy` 以支持新的模糊数理论和应用，而无需修改核心代码。理解注册系统是掌握 `AxisFuzzy` 扩展开发的关键第一步。