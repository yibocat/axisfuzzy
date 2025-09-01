AxisFuzzy 的扩展系统是一个设计精巧、高度灵活的机制，它允许开发者为不同类型的模糊数（`mtype`）动态地添加和管理功能。其核心思想是基于 `mtype` 的可插拔架构，使得 AxisFuzzy 能够轻松地扩展以支持新的模糊数类型或为现有类型提供特化的操作，而无需修改核心代码。

## 1. 扩展整体架构和运行机制
AxisFuzzy 的扩展系统主要由以下几个核心组件构成：

### 1. 注册表 (`ExtensionRegistry`)
- 文件：`registry.py`
- 作用：它是整个扩展系统的“大脑”，负责存储所有注册的功能及其元数据。
- 核心功能：
    - `register()`：通过 `@extension` 装饰器（见下文）注册函数。它支持为特定 `mtype` 注册特化实现，也支持注册通用默认实现。
    - `get_function()`：根据功能名称和 `mtype` 获取对应的函数实现。如果存在特化实现，则返回特化实现；否则，回退到默认实现。
    - l`ist_functions()`：列出所有已注册的功能及其详细信息。
- 特点：线程安全，支持优先级排序（当有多个实现时，优先级高的会被优先选择）。
- 装饰器 (`extension`, `batch_extension`)：

### 2. 装饰器(`extension`, `batch_extension`):
- 文件：`decorators.py`
- 作用：提供简洁的语法糖，用于将函数注册到 `ExtensionRegistry` 中。
- 核心功能：
  - `@extension`：这是最常用的装饰器，用于注册单个功能。您可以指定功能名称 (name)、适用的 `mtype`、目标类 (`target_classes`，例如 `Fuzznum` 或 `Fuzzarray`)、注入类型 (`injection_type`，可以是实例方法、顶级函数或两者) 以及是否为默认实现 (`is_default`) 和优先级 (`priority`)。
  - `@batch_extension`：用于批量注册多个功能，方便管理。
- `injection_type` 现支持：
  - instance_method：注入为实例方法
  - top_level_function：注入为顶级函数
  - both：以上两者
  - instance_property（新增）：注入为只读分发属性（@property 形式），适合轻量快速计算的指标 (如 score / acc / ind)

示例（注册一个分发属性）：
```python
@extension(
    name='score',
    mtype='qrofn',
    target_classes=['Fuzznum', 'Fuzzarray'],
    injection_type='instance_property'
)
def qrofn_score_ext(obj):
    return obj.md ** obj.q - obj.nmd ** obj.q
```
注入后：obj.score

### 3. 分发器 (`ExtensionDispatcher`)
- 文件：`dispatcher.py`
- 作用：负责在运行时根据对象的 mtype 将方法调用分发到正确的具体实现。
- 核心功能：
  - `create_instance_method()`：创建一个“代理”实例方法。当这个方法被调用时，它会检查调用对象的 `mtype`，然后从注册表中查找并调用对应的特化或默认实现。
  - `create_top_level_function()`：类似地，创建一个“代理”顶级函数，用于处理顶级函数调用时的 `mtype` 分发。
- 运行机制：它不会直接执行功能，而是生成一个包装器函数。这个包装器函数在被调用时，会动态地从注册表中查找并执行最匹配的实现。
- 说明：
  - create_instance_property(func_name)：返回属性描述符，访问时进行 mtype 分发。
  - property 对象本身没有 __name__，通过 doc 传入文档字符串。

### 4. 注入器 (`ExtensionInjector`)
- 文件：`injector.py`
- 作用：在程序启动时，将注册表中定义的功能动态地“注入”到 `Fuzznum` 和 `Fuzzarray` 类中，或者作为 `axisfuzzy` 模块的顶级函数。
- 核心功能：
  - `inject_all()`：遍历注册表中所有已注册的功能，并根据其 `injection_type` 将它们注入到指定的类（`Fuzznum`, `Fuzzarray`）或模块命名空间中。
- 运行机制：它通过 `setattr()` 等方式，将分发器创建的代理函数绑定到目标类或模块上，使得这些扩展功能可以像原生方法或函数一样被调用。
- 说明:
    - 当发现某功能的任一实现声明 injection_type 包含 instance_property 时，会为目标类注入一个分发 property（若该名称尚不存在）。

### 5. 工具函数 (`call_extension`)
- 文件：`utils.py`
- 作用：提供一个辅助函数，允许在扩展函数内部调用其他扩展函数，而无需担心循环依赖或注入时序问题。
- 核心功能：`call_extension(func_name, obj, *args, **kwargs)`：直接通过名称和对象 `mtype` 调用注册表中的扩展函数。

### 6. 初始化 (`apply_extensions`)
- 文件：`__init__.py` 和 `__init__.py`
- 作用：这是整个扩展系统启动的入口点。
- 运行机制：在 `AxisFuzzy` 库被导入时（通过 `__init__.py` 中的 `apply_extensions()` 调用），`apply_extensions()` 函数会被执行。它会获取 `ExtensionInjector` 实例，并调用其 `inject_all()` 方法，从而完成所有已注册功能的动态注入。

## 2. 整体思想
AxisFuzzy 扩展系统的整体思想可以概括为：“注册-分发-注入”, “注册-分发-注入” 现在同时覆盖 方法调用 与 属性访问两种交互方式。

- 注册 (Registration)：开发者通过简单的 `@extension` 装饰器，声明一个函数是 AxisFuzzy 的一个扩展功能，并指定其名称、适用的 `mtype` 和注入方式。这些信息被存储在 `ExtensionRegistry` 中。
- 分发 (Dispatching)：当用户调用一个扩展功能时（无论是作为实例方法还是顶级函数），实际执行的不是原始函数，而是 `ExtensionDispatcher` 创建的一个代理函数。这个代理函数会根据调用对象的 `mtype`，智能地从 `ExtensionRegistry` 中查找并调用最合适的具体实现。
- 注入 (Injection)：在 AxisFuzzy 库加载时，`ExtensionInjector` 会将这些代理函数动态地绑定到 `Fuzznum` 和 `Fuzzarray` 类上，或者作为 `axisfuzzy` 模块的顶级函数。这样，用户就可以像调用普通方法或函数一样使用这些扩展功能，而无需关心底层的 `mtype` 分发逻辑。

这种设计带来了以下显著优势：

- 可扩展性：轻松添加新的模糊数类型和对应的特化功能，无需修改核心代码。
- 模块化：将不同 `mtype` 的功能实现分离，提高代码的可维护性。
- 灵活性：支持默认实现和特化实现，以及优先级机制，满足不同场景的需求。
- 解耦：核心 `Fuzznum` 和 `Fuzzarray` 类与具体的功能实现解耦，它们只知道如何调用分发器。
- 用户友好：用户可以像调用普通方法一样使用扩展功能，底层复杂性被隐藏。
- 
以 `_func.py` 为例
让我们看看您在 `_func.py` 中编写的 `qrofn_distance` 函数：

```python
# ...existing code...
@extension(
    name='distance',
    mtype='qrofn',
    target_classes=['Fuzznum', 'Fuzzarray'],
    injection_type='both'
)
def qrofn_distance(fuzz1: Fuzznum, fuzz2: Fuzznum, p: int = 2) -> float:
    q = fuzz1.q

    md_diff = abs(fuzz1.md ** q - fuzz2.md ** q) ** p
    nmd_diff = abs(fuzz1.nmd ** q - fuzz2.nmd ** q) ** p
    return ((md_diff + nmd_diff) / 2) ** (1 / p)
```

这里发生了什么：

1. `@extension(...)` 装饰器：
   - `name='distance'`：这表明我们正在注册一个名为 `distance` 的功能。
   - `mtype='qrofn'`：这个 `distance` 功能是专门为 `qrofn` 类型的模糊数实现的。
   - `target_classes=['Fuzznum', 'Fuzzarray']`：这意味着 `distance` 功能将被注入到 `Fuzznum` 和 `Fuzzarray` 类中，作为它们的实例方法。
   - `injection_type='both'`：这意味着 `distance` 功能不仅会作为 Fuzznum 和 `Fuzzarray` 的实例方法，还会作为 `axisfuzzy` 模块的顶级函数被注入。

2. 注册过程：当 `_func.py` 模块被导入时（通常在 `axisfuzzy` 初始化时），`@extension` 装饰器会执行。它会调用 `get_registry_extension().register(...)`，将 `qrofn_distance` 函数及其元数据注册到全局的 `ExtensionRegistry` 中。

3. 注入过程：当 `__init__.py` 中的 `apply_extensions()` 被调用时：
   - `ExtensionInjector` 会从 `ExtensionRegistry` 中获取 `distance` 功能的信息。
   - 由于 `injection_type='both'`，`ExtensionInjector` 会：
     - 为 `Fuzznum` 和 `Fuzzarray` 类创建 `distance` 的实例方法分发器（通过 `ExtensionDispatcher.create_instance_method()`）。
     - 为 `axisfuzzy` 模块创建 `distance` 的顶级函数分发器（通过 `ExtensionDispatcher.create_top_level_function()`）。
     - 将这些分发器动态地绑定到 `Fuzznum`、`Fuzzarray` 类和 `axisfuzzy` 模块的命名空间中。

4. 调用过程：
   - 作为实例方法：当您调用 `my_qrofn_fuzznum.distance(another_fuzznum)` 时，实际调用的是 `ExtensionDispatcher` 创建的代理方法。这个代理方法会检测 `my_qrofn_fuzznum` 的 `mtype`（例如 `qrofn`），然后从注册表中找到 `qrofn_distance` 函数并执行它。
   - 作为顶级函数：当您调用 `axisfuzzy.distance(my_qrofn_fuzznum, another_fuzznum)` 时，实际调用的是 `ExtensionDispatcher` 创建的代理顶级函数。这个代理函数会检测传入的第一个参数 `my_qrofn_fuzznum` 的 `mtype`，然后同样从注册表中找到 `qrofn_distance` 函数并执行它。

通过这种机制，`AxisFuzzy` 实现了高度的模块化和可扩展性。您可以为任何 `mtype` 定义 `distance` 函数，甚至可以定义一个通用的 `distance` 默认实现，系统会根据对象的实际 `mtype` 自动选择最合适的实现。

## 3. 新示例：分发属性 (score / acc / ind)

QROFN 指标定义：
- score = md^q - nmd^q
- acc = md^q + nmd^q
- ind = 1 - acc

注册：
```python
@extension(name='acc', mtype='qrofn',
           target_classes=['Fuzznum','Fuzzarray'],
           injection_type='instance_property')
def qrofn_acc_ext(x): return x.md ** x.q + x.nmd ** x.q
```

使用：
```python
x.score
x.acc
x.ind
```

高性能说明：
- 对 Fuzzarray：底层直接取后端 SoA 分量数组 (mds, nmds) 做向量化 (幂、加减)。
- 对 Fuzznum：标量直接计算。

## 4. injection_type 汇总（更新）

| 类型                | 作用                               |
|---------------------|------------------------------------ |
| instance_method     | 分发实例方法                       |
| top_level_function  | 注入顶级函数                       |
| both                | 方法 + 顶级函数                     |
| instance_property   | 分发属性（只读，访问即计算）        |