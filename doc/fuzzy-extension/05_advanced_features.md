# 高级特性：注入类型、优先级与批量注册

`AxisFuzzy` 的扩展系统不仅功能强大，还提供了一系列高级特性，允许开发者对扩展的行为进行精细化控制。理解并善用这些特性，可以让您的扩展代码更加灵活、健壮和高效。

本篇文档将深入探讨三个高级主题，不仅解释“是什么”，更阐述“为什么”和“如何更好地使用”：
1.  **注入类型 (`injection_type`)**：精确控制扩展功能的 API 形态。
2.  **优先级 (`priority`)**：优雅地管理实现冲突与覆盖。
3.  **批量注册 (`@batch_extension`)**：一种更具组织性的高效注册方式。

---

## 1. 注入类型 (`injection_type`) 精解

`injection_type` 参数是 `@extension` 和 `@register_mixin` 装饰器的核心配置之一，它决定了扩展功能最终以何种 API 形式呈现给用户。选择合适的注入类型是 API 设计的关键一步。

### 1.1. `'instance_method'`

-   **作用**：将扩展注入为 `Fuzznum` 和 `Fuzzarray` 的**实例方法**。这是最常用、最符合面向对象编程直觉的类型。
-   **调用方式**：`my_object.my_function(...)`
-   **设计哲学**：当一个功能是作用于单个对象的核心操作时，应将其设计为实例方法。这增强了对象的行为能力，使得 API 调用流畅自然。
-   **示例**：`my_array.reshape()` 或 `my_fuzznum.score()`。

### 1.2. `'top_level_function'`

-   **作用**：将扩展注入为 `axisfuzzy` 包的**顶层函数**。
-   **调用方式**：`axisfuzzy.my_function(my_object, ...)`
-   **设计哲学**：
    1.  **类 NumPy/SciPy 的函数式 API**：为习惯了函数式编程风格的用户提供一致的体验。
    2.  **多对象操作**：当一个函数需要操作多个 `Fuzznum`/`Fuzzarray` 对象时（例如计算两个模糊数之间的距离），顶层函数是更自然的选择。`axisfuzzy.distance(num1, num2)` 显然比 `num1.distance(num2)` 更具对称性。
    3.  **避免命名空间污染**：对于一些不那么常用或具有通用名称的工具函数，作为顶层函数可以避免污染 `Fuzznum`/`Fuzzarray` 的实例命名空间。

### 1.3. `'both'`

-   **作用**：同时执行 `INSTANCE_METHOD` 和 `TOP_LEVEL` 两种注入。
-   **调用方式**：`my_object.my_function(...)` 和 `axisfuzzy.my_function(my_object, ...)` 均可。
-   **设计哲学**：为核心或常用功能提供最大的灵活性，让用户可以选择自己喜欢的调用风格，无需在“面向对象”和“函数式”之间做艰难选择。`AxisFuzzy` 的许多核心功能都采用了这种方式。

### 1.4. `'instance_property'` (高级)

-   **作用**：将一个**无参数的函数**注入为 `Fuzznum` 和 `Fuzzarray` 的**只读属性**。
-   **调用方式**：`my_object.my_property` (注意：没有调用括号)。
-   **设计哲学**：当一个计算结果感觉像是对象的内在属性（例如 `shape`, `ndim`, `is_valid`）时，使用此类型可以提供极为优雅的 API。它向用户隐藏了背后的计算过程，使得调用代码更具可读性。
-   **与 `@cached_property` 结合**：为了性能，通常建议将注入为属性的函数与 `functools.cached_property`（或 `AxisFuzzy` 内部的类似实现）结合使用，这样可以确保计算只在首次访问时执行一次。

**示例**：将一个计算“纯度”的函数作为缓存属性注入。

```python
from functools import cached_property

# 假设 purity 的计算比较耗时
@extension(mtype='qrofs', injection_type='instance_property')
def purity(fuzznum: Fuzznum) -> float:
    @cached_property
    def _cached_purity():
        # 复杂的计算过程
        q = fuzznum.mtype.q
        return fuzznum.md ** q + fuzznum.nmd ** q
    return _cached_purity

# 调用
my_num = Fuzznum(mtype='qrofs', md=0.8, nmd=0.1)
purity_value = my_num.purity  # 第一次访问，执行计算
purity_value_2 = my_num.purity # 第二次访问，直接返回缓存结果
```

---

## 2. 优先级 (`priority`) 控制

在复杂的应用或插件化系统中，可能会出现多个库为同一个 `mtype` 的同一个扩展函数提供了不同的实现。`priority` 参数就是为了解决这种**实现冲突**而设计的。

-   **核心规则**：当 `ExtensionDispatcher` 查找到多个可用的实现时，它会选择 `priority` 值**最高**（即数字最大）的那一个。
-   **默认值**：所有扩展的默认 `priority` 均为 `0`。

**适用场景与最佳实践**：

1.  **良性覆盖**：`AxisFuzzy` 可能为某个函数提供了一个 `is_default=True` 的通用实现。如果您想为某个特定的 `mtype` 提供一个性能更高或算法更优的实现，只需在注册时赋予其一个正的 `priority` 值。
    -   **建议**：使用适度的优先级，例如 `10` 或 `100`，为未来的其他扩展留下空间。

2.  **插件系统中的竞争**：如果您在构建一个基于 `AxisFuzzy` 的插件化系统，不同的插件可能会竞争同一个功能的实现。通过 `priority` 机制，您可以建立一套清晰的覆盖规则。
    -   **建议**：为您的插件生态系统定义一套优先级规范。例如，官方插件的 `priority` 在 100-200 之间，社区高质量插件在 50-99 之间，个人插件在 1-49 之间。

3.  **强制覆盖（不推荐）**：您可以使用一个极高的 `priority`（如 `9999`）来确保您的实现几乎总是被选中。
    -   **警告**：这是一种“代码异味”，可能表明您的设计存在问题。滥用高优先级会破坏系统的可扩展性，使其他开发者难以覆盖您的实现。请仅在绝对必要时使用。

**示例**：一个三层覆盖的场景。

```python
# 1. AxisFuzzy 核心库 (priority=0, is_default=True)
@extension(name='distance', is_default=True, priority=0)
def slow_generic_distance(a, b):
    # ... 通用但较慢的实现 ...

# 2. 一个优化的科学计算库 (priority=100)
@extension(name='distance', mtype='qrofs', priority=100)
def fast_qrofs_distance(a, b):
    # ... 针对 qrofs 的高度优化实现 ...

# 3. 用户自己的项目，有特殊需求 (priority=200)
@extension(name='distance', mtype='qrofs', priority=200)
def special_project_distance(a, b):
    # ... 满足项目特殊需求的实现 ...
```
当调用 `distance(qrofs_obj1, qrofs_obj2)` 时，`special_project_distance` 将被调用，因为它具有最高的优先级。

---

## 3. 批量注册 (`@batch_extension`)

当您需要为一个 `mtype` 实现一系列逻辑上相关的扩展函数时，`@batch_extension` 提供了一种更简洁、更具组织性的方式。

-   **作用**：这是一个**类装饰器**。它会遍历类中的所有公共方法，并将它们作为 `extension` 函数进行批量注册。
-   **命名约定**：被注册的函数名就是类中的方法名。
-   **参数传递**：`@batch_extension` 接受所有 `@extension` 支持的参数（如 `mtype`, `priority`, `injection_type`），这些参数将统一应用于该类中的所有方法。

**示例**：为 `qrofs` 类型批量注册一系列统计指标。

```python
from axisfuzzy.extension import batch_extension
from axisfuzzy.core import Fuzznum

@batch_extension(
    mtype='qrofs', 
    injection_type='both',  # 所有方法都将作为实例方法和顶层函数注入
    priority=50
)
class QROFSMetrics:

    def score(self, fuzznum: Fuzznum) -> float:
        # 注意：第一个参数总是 self，代表类实例
        q = fuzznum.mtype.q
        return fuzznum.md ** q - fuzznum.nmd ** q

    def accuracy(self, fuzznum: Fuzznum) -> float:
        q = fuzznum.mtype.q
        return fuzznum.md ** q + fuzznum.nmd ** q

    def hesitancy(self, fuzznum: Fuzznum) -> float:
        q = fuzznum.mtype.q
        # 我们可以通过 self 调用同一个批次中的其他方法
        return 1 - self.accuracy(fuzznum)
```

**工作原理与优势**：

-   **代码组织性**：将所有与特定 `mtype` 相关的扩展逻辑聚合在一个类中，极大地提高了代码的可读性和内聚性。
-   **减少样板代码**：避免了为每个函数重复书写 `@extension(...)`。
-   **内部调用**：如 `hesitancy` 所示，可以在一个方法内部通过 `self` 调用同一个类中的其他方法，便于逻辑复用。
-   **易于维护**：添加或删除一个扩展，只需在类中增删一个方法即可。

---

## 4. 总结与最佳实践

掌握这些高级特性，将使您在 `AxisFuzzy` 上的开发能力提升到一个新的水平：

-   **精心设计您的 API**：使用**注入类型**来设计更优雅、更符合用户直觉的 API。考虑您的函数是“行为”还是“属性”。
-   **负责任地使用优先级**：使用**优先级**来管理复杂的依赖关系和实现冲突，但要避免滥用高优先级。为您的生态系统建立清晰的规则。
-   **拥抱代码组织性**：当您为一个类型实现多个相关功能时，优先使用**批量注册**来编写更整洁、更易于维护的扩展代码。

结合这些工具，您可以构建出功能强大、设计精良且高度可定制的模糊逻辑应用程序。