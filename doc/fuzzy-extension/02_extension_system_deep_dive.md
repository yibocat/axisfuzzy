# `extension` 系统深度剖析：API、架构与工作流

`extension` 系统是 `AxisFuzzy` 框架的动态功能扩展核心，专为实现与特定模糊数类型 (`mtype`) 相关的“语义”功能而设计。其精妙之处在于一套完整且解耦的“注册-分发-注入” (Register-Dispatch-Inject) 三层架构。这套架构确保了扩展函数可以被灵活定义、在运行时被精确调用，并无缝集成到框架的核心数据结构中。

与主要关注 `mtype` 无关的“结构化”操作的 `mixin` 系统不同，`extension` 系统是实现多态行为和特定领域逻辑的关键。

本篇文档将从开发者最关心的 API 出发，深入讲解 `@extension` 装饰器的全部功能，然后剖析其背后的三大核心组件——`Registry`, `Dispatcher`, `Injector`——是如何协同工作的。

---

## 1. 核心 API：`@extension` 装饰器详解

`@extension` 装饰器是您与 `extension` 系统交互的主要入口。它是一个功能强大的工具，允许您通过声明式的方式，将一个普通 Python 函数注册为框架的一部分。

### 1.1. 基本语法

```python
from axisfuzzy.extension import extension

@extension(
    name: str,
    mtype: str = None,
    target_classes: list[str] = ['Fuzznum', 'Fuzzarray'],
    injection_type: str = 'both',
    is_default: bool = False,
    priority: int = 0,
    **kwargs
)
def my_extension_function(*args, **kwargs):
    # Your implementation here
    ...
```

### 1.2. 参数详解

#### `name: str` (必需)

**作用**：定义扩展的逻辑名称。这个名称将成为注入后方法或函数的名字。例如，`name='distance'` 将会创建 `Fuzznum.distance()` 或 `axisfuzzy.distance()`。

**示例**：
```python
from axisfuzzy.extension import extension
from axisfuzzy.core import Fuzznum

@extension(name='custom_distance', mtype='qrofs')
def calc_dist(f1: Fuzznum, f2: Fuzznum):
    # ... implementation ...
    return 0.5

# 使用:
# my_qrofs_num.custom_distance(another_num)
# axisfuzzy.custom_distance(my_qrofs_num, another_num)
```

#### `mtype: str` (可选)

**作用**：指定此实现所服务的特定模糊数类型（如 `'qrofs'`, `'ivfs'`）。这是实现动态分发的核心。只有当操作对象的 `mtype` 与此处指定的值匹配时，这个函数才会被调用。

**示例**：
```python
@extension(name='score', mtype='qrofs')
def score_qrofs(f: Fuzznum):
    return f.md**2 - f.nmd**2

@extension(name='score', mtype='ivfs')
def score_ivfs(f: Fuzznum):
    return (f.a + f.b) / 2

# qrofs_num.score() -> 调用 score_qrofs
# ivfs_num.score()  -> 调用 score_ivfs
```

#### `is_default: bool` (可选, 默认为 `False`)

**作用**：当设置为 `True` 时，此实现将作为该 `name` 的“默认”或“后备”版本。如果在调用时，没有找到与对象 `mtype` 精确匹配的实现，`Dispatcher` 将会调用这个默认版本。一个 `name` 只能有一个默认实现。

**示例**：
```python
# 为 'score' 提供一个通用后备实现
@extension(name='score', is_default=True)
def score_default(f: Fuzznum):
    # 假设对于未知类型，返回一个中性值
    return 0.0

# other_type_num.score() -> 调用 score_default
```

#### `injection_type: str` (可选, 默认为 `'both'`)

**作用**：控制扩展最终以何种形式提供给用户。可选值包括：
-   `'instance_method'`: 只作为实例方法注入 (e.g., `my_fuzznum.score()`)。
-   `'top_level_function'`: 只作为顶层函数注入 (e.g., `axisfuzzy.score(my_fuzznum)`)。
-   `'both'`: 同时注入为实例方法和顶层函数（默认行为）。
-   `'instance_property'`: 注入为一个只读的实例属性。被装饰的函数应只接受一个参数（实例本身 `self`），并返回一个值。

**示例**：

```python
# 只作为顶层函数
@extension(name='calculate_entropy', mtype='qrofs', injection_type='top_level_function')
def entropy(f: Fuzznum): ...
# 正确: axisfuzzy.calculate_entropy(f)
# 错误: f.calculate_entropy()

# 作为只读属性
@extension(name='hesitancy', mtype='qrofs', injection_type='instance_property')
def hesitancy(f: Fuzznum):
    return 1 - f.md**2 - f.nmd**2
# 正确: h = f.hesitancy
# 错误: h = f.hesitancy()
```

#### `target_classes: list[str]` (可选, 默认为 `['Fuzznum', 'Fuzzarray']`)

**作用**：当 `injection_type` 包含实例方法或属性时，此参数指定要注入的目标类。默认情况下，会同时注入到 `Fuzznum` 和 `Fuzzarray` 中。

**示例**：
```python
# 假设一个函数只对单个模糊数有意义
@extension(name='to_latex', mtype='qrofs', target_classes=['Fuzznum'])
def to_latex(f: Fuzznum): ...

# my_fuzznum.to_latex() -> 正确
# my_fuzzarray.to_latex() -> AttributeError
```

#### `priority: int` (可选, 默认为 `0`)

**作用**：解决注册冲突。当两个不同的代码库尝试为同一个 `(name, mtype)` 组合注册实现时，`priority` 值更高的实现会覆盖值更低的。如果优先级相同，则会抛出 `ValueError`，防止意外覆盖。

**示例**：
```python
# 在核心库中定义一个基础版本
@extension(name='score', mtype='qrofs', priority=0)
def core_score(f: Fuzznum): ...

# 在用户插件中定义一个更高优先级的优化版本
@extension(name='score', mtype='qrofs', priority=10)
def plugin_optimized_score(f: Fuzznum): ...

# 最终，f.score() 会调用 plugin_optimized_score
```

---

## 2. 内部架构：注册-分发-注入

现在，让我们深入幕后，看看当您使用 `@extension` 时，框架内部发生了什么。

![Extension Architecture](https://your-image-host.com/extension_architecture_v2.png) <!-- 你可以替换成一个真实的架构图链接 -->

### 2.1. `ExtensionRegistry` - 中央注册表

**角色**：所有扩展的“户籍管理中心”。它是一个全局单例，维护着所有已注册函数的信息。

**工作机制**：
-   **内部结构**：`ExtensionRegistry` 内部主要有两个字典：
    -   `_functions`: 一个嵌套字典，结构为 `{name: {mtype: (function, metadata)}}`。它存储了所有 `mtype` 特定的实现。
    -   `_defaults`: 一个字典，结构为 `{name: (function, metadata)}`。它存储了所有默认实现。
-   **注册过程**：当 `@extension(...)` 被调用时，它实际上是调用了 `registry.register()`。此方法会：
    1.  创建一个 `FunctionMetadata` 对象，包含所有传入的参数（`name`, `mtype`, `priority` 等）。
    2.  根据 `is_default` 的值，决定是将 `(函数, metadata)` 元组存入 `_functions` 还是 `_defaults`。
    3.  在存入前，它会检查 `priority`，如果已存在一个同名同 `mtype` 的实现且优先级更高或相等，则会拒绝注册并抛出异常。

### 2.2. `ExtensionDispatcher` - 动态代理工厂

**角色**：为每一个逻辑扩展（如 `score`）创建“动态代理”或“分发器”。这个代理是一个可调用对象，它本身没有计算逻辑，但知道如何根据输入找到并执行正确的逻辑。

**工作机制**：
`Dispatcher` 会根据 `injection_type` 创建三种不同类型的代理：
1.  **实例方法代理 (`create_instance_method`)**:
    -   当 `my_num.score()` 被调用时，代理函数被触发。
    -   它从 `my_num` (第一个参数 `obj`) 中获取 `mtype`。
    -   它使用 `('score', my_num.mtype)` 去 `ExtensionRegistry` 中查找实现。
    -   如果找到，则调用该实现；否则，查找并调用默认实现；如果都没有，则抛出 `NotImplementedError`。
2.  **顶层函数代理 (`create_top_level_function`)**:
    -   当 `axisfuzzy.score(my_num)` 被调用时，代理函数被触发。
    -   它会按以下顺序解析 `mtype`：
        1.  检查调用中是否有名为 `mtype` 的关键字参数，如 `axisfuzzy.score(..., mtype='qrofs')`。
        2.  如果第一个位置参数是 `Fuzznum` 或 `Fuzzarray` 实例，则使用其 `.mtype` 属性。
        3.  如果以上都没有，则使用 `axisfuzzy` 的全局默认 `mtype` 配置。
    -   找到 `mtype` 后，后续的查找和调用逻辑与实例方法代理相同。
3.  **实例属性代理 (`create_instance_property`)**:
    -   它创建一个 Python 的 `property` 对象。
    -   其 `getter` 函数的逻辑与实例方法代理非常相似：获取实例的 `mtype`，查找并调用对应的实现。

### 2.3. `ExtensionInjector` - 最终注入器

**角色**：在库加载时，将 `Dispatcher` 创建的代理“安装”到最终用户可见的位置。

**工作机制**：
-   **触发时机**：`Injector` 的 `inject_all()` 方法通常在 `axisfuzzy` 库初始化时被调用一次。
-   **注入流程**：
    1.  `Injector` 遍历 `ExtensionRegistry` 中所有注册过的 `name`。
    2.  对于每个 `name`（如 `'score'`），它会检查所有相关的注册（包括特定 `mtype` 和默认的），以确定最终的注入策略（例如，是否需要注入为实例方法、顶层函数或属性）。
    3.  它请求 `ExtensionDispatcher` 为 `'score'` 创建相应类型的代理（`method_dispatcher`, `property_dispatcher` 等）。
    4.  最后，它使用 `setattr()` 将这些代理动态地添加到 `target_classes`（如 `Fuzznum`）和 `axisfuzzy` 模块的命名空间中。
    5.  注入过程是安全的，它不会覆盖目标类或模块上任何已存在的同名属性。

---

## 3. 完整流程回顾与手动注入

让我们以 `my_qrofs_num.score` 的调用为例，串联起整个流程：

1.  **开发阶段**：开发者在代码中使用 `@extension(name='score', mtype='qrofs', injection_type='instance_property')` 装饰了 `qrofs_score` 函数。
2.  **库加载时 (`axisfuzzy` import)**：
    -   `ExtensionRegistry` 记录下：`_functions['score']['qrofs'] = (qrofs_score, metadata)`。
    -   `ExtensionInjector.inject_all()` 被调用。
    -   `Injector` 发现 `'score'` 需要被注入为 `instance_property`。
    -   它调用 `ExtensionDispatcher.create_instance_property('score')`，得到一个 `property` 对象。
    -   它执行 `setattr(Fuzznum, 'score', property_object)`。
3.  **运行时**：
    -   代码访问 `my_qrofs_num.score`。
    -   这触发了 `property` 对象的 `getter` 方法。
    -   `getter` 方法（即代理）查看 `my_qrofs_num.mtype`，得到 `'qrofs'`。
    -   代理在 `ExtensionRegistry` 中查找 `('score', 'qrofs')`，并成功找到了 `qrofs_score` 函数。
    -   代理调用 `qrofs_score(my_qrofs_num)`，计算并返回结果。

### 关于手动注入

通常，开发者不需要关心注入过程。但是，如果您在运行时动态地加载了新的扩展模块，并希望它们立即生效，可以手动触发注入流程：

```python
import axisfuzzy
from axisfuzzy.core import Fuzznum, Fuzzarray
from axisfuzzy.extension import get_extension_injector

# 假设您刚刚 import a_new_plugin_with_extensions

# 获取注入器单例
injector = get_extension_injector()

# 定义目标
class_map = {'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}
module_namespace = axisfuzzy.__dict__

# 重新运行注入流程
injector.inject_all(class_map, module_namespace)
```

这个操作会扫描注册表中所有“新”的扩展，并将它们注入到相应的位置，而不会影响已存在的扩展。