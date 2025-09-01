# `Fuzznum` 与 `FuzznumStrategy`：门面与策略的艺术

在 `axisfuzzy.core` 模块中，`Fuzznum` 和 `FuzznumStrategy` 的设计是理解 `AxisFuzzy` 框架如何实现用户友好性与可扩展性相统一的关键。它们共同构成了一种经典的设计模式组合：**门面模式 (Facade Pattern)** 与 **策略模式 (Strategy Pattern)**，并通过动态代理（Dynamic Proxy）技术无缝地结合在一起。

本篇文档将深入剖析这两者之间的关系，阐明它们各自的职责以及它们如何协同工作。

## 1. 核心关系：门面与策略

-   **`Fuzznum` (位于 `fuzznums.py`)：优雅的门面**
    -   `Fuzznum` 是用户直接交互的类。当你创建一个模糊数时，你实际上是在与一个 `Fuzznum` 实例打交道。
    -   它本身**不包含任何具体的模糊数逻辑**（如隶属度如何存储、如何验证、如何运算）。它的核心职责是为用户提供一个简单、统一的接口，隐藏内部复杂的子系统。

-   **`FuzznumStrategy` (位于 `base.py`)：具体的策略**
    -   `FuzznumStrategy` 是一个抽象基类（ABC），它定义了所有“具体模糊数实现”必须遵循的契约。
    -   对于每一种特定类型的模糊数（如 `qrofn`、`ivfn`），都必须有一个对应的策略类（如 `QROFNStrategy`）继承自 `FuzznumStrategy`。
    -   这个策略类才是真正负责**数据存储、合法性校验、逻辑处理和运算分派**的地方。

### `fuzznum` 工厂函数：便捷的入口

`AxisFuzzy` 提供了一个名为 `fuzznum` 的工厂函数，它是创建模糊数的推荐方式。它支持多种初始化方法，非常灵活。

**方式一：使用关键字参数（最清晰）**

```python
from axisfuzzy import fuzznum

# 通过关键字参数创建 q-rung orthopair 模糊数
# mtype 和 q 都有默认值，可以省略
my_fuzznum = fuzznum(mtype='qrofn', md=0.8, nmd=0.1, q=3)
```

**方式二：使用元组（最简洁）**

`fuzznum` 函数可以接收一个 `values` 元组，并将其按顺序映射到策略类的核心属性上。

```python
# 使用元组创建，(0.8, 0.1) 会被自动映射到 (md, nmd)
# mtype='qrofn' 也可以省略，此时会使用配置中的默认 mtype
a = fuzznum((0.8, 0.1), mtype='qrofn', q=3)
```

无论使用哪种方式，`AxisFuzzy` 内部的流程都是相似的：

1.  `fuzznum` 函数接收到参数，确定 `mtype`（如果未提供，则使用默认值）。
2.  它查询 `FuzznumRegistry` 注册表，找到与 `mtype` 关联的策略类（例如 `QrofnStrategy`）。
3.  它使用用户提供的参数实例化该策略类。
4.  最后，它创建一个 `Fuzznum` 实例，并将刚刚创建的策略实例作为其内部的 `_strategy` 成员。

从此，`my_fuzznum` 这个门面就与一个具体的策略实现绑定了。

## 2. `Fuzznum` 作为动态代理

`Fuzznum` 最巧妙的设计在于它如何将用户的操作无缝地传递给其内部的策略实例。这是通过重写 Python 的特殊方法 `__getattr__` 和 `__setattr__` 实现的，这使得 `Fuzznum` 成为一个**动态代理**。

-   **属性获取 (`__getattr__`)**：当你试图访问 `my_fuzznum.md` 时，`Fuzznum` 的 `__getattr__` 方法会拦截这个请求，并将其转发给 `self._strategy_instance.md`。
-   **属性设置 (`__setattr__`)**：当你试图执行 `my_fuzznum.md = 0.9` 时，`Fuzznum` 的 `__setattr__` 方法同样会拦截并转发给 `self._strategy_instance.md`。

这意味着，从用户的角度来看，`Fuzznum` 仿佛拥有 `md` 和 `nmd` 这些属性，但实际上这些状态都由其背后的策略对象管理。这种代理机制带来了几个好处：

-   **接口统一**：无论底层策略多么不同，用户始终面对的是同一个 `Fuzznum` 类。
-   **逻辑解耦**：`Fuzznum` 只关心如何转发请求，而 `FuzznumStrategy` 只关心如何处理业务逻辑。
-   **动态替换**：理论上，可以在运行时更换一个 `Fuzznum` 实例的策略（尽管在当前框架中不常用）。

## 3. `FuzznumStrategy` 的核心职责

作为具体实现的承载者，`FuzznumStrategy` 的子类有三大核心职责：

### a. 属性声明与自动收集

与一些传统观念不同，`AxisFuzzy` 的策略类**不使用 `__slots__`** 来限制属性。相反，它采用了一种更灵活、更具声明性的方式：

-   **通过类属性或类型注解来声明**：你只需要在策略类中像普通类变量一样定义你的模糊数组件即可。

```python
# axisfuzzy/fuzztype/qrofs/qrofn.py
class QROFNStrategy(FuzznumStrategy):
    mtype = 'qrofn'
    md: Optional[float] = None
    nmd: Optional[float] = None
    # ...
```

-   **`__init_subclass__` 自动收集**：`FuzznumStrategy` 的元编程魔法在于它的 `__init_subclass__` 钩子。当 `QROFNStrategy` 类被定义时，这个钩子会自动运行，检查所有的类属性和类型注解，将非私有的、非方法的属性名（如 `'md'`, `'nmd'`, `'q'`）收集到一个内部列表 `_declared_attributes` 中。

这种设计不仅清晰地定义了每种模糊数的数据结构，还为后续的自动验证和代理提供了元数据。

### b. 验证生命周期：验证器、转换器与回调

这是 `FuzznumStrategy` 设计中最为关键和强大的部分。它提供了一套精密的生命周期钩子，以确保数据的完整性和一致性。当用户尝试设置一个属性时（如 `my_fuzznum.md = 0.9`），会依次触发以下流程：

1.  **验证器 (Validator)**：最先执行。它是一个简单的函数，接收新值并返回 `True` 或 `False`。它用于进行**无状态的、原子化的检查**，例如确保值在 `[0, 1]` 区间内。如果验证失败，赋值操作会立即中断并抛出 `ValueError`。

2.  **转换器 (Transformer)**：在验证器通过后执行。它接收新值并返回一个可能被修改过的新值。它用于**数据的归一化和类型转换**。一个典型的例子是将输入的列表转换为 `numpy.ndarray`，以便进行高效的数学运算。

3.  **赋值操作**：将转换后的值赋给策略实例的内部属性。

4.  **变更回调 (Change Callback)**：最后执行。它是一个函数，可以访问属性名、旧值和新值。它用于处理**有状态的、涉及多个属性的复杂约束**。例如，在 `QROFNStrategy` 中，正是回调函数在 `md` 或 `nmd` 改变后，检查 `md^q + nmd^q <= 1` 这一核心约束是否依然满足。

这三个钩子可以通过 `add_attribute_validator`, `add_attribute_transformer`, 和 `add_change_callback` 方法在策略的 `__init__` 中注册。

### c. 运算执行的入口

`FuzznumStrategy` 定义了 `execute_operation` 方法。当一个运算（如加法）在 `Fuzznum` 层面被触发时，它最终会调用到策略的这个方法。该方法负责与 `OperationScheduler`（运算调度器）交互，找到并执行正确的运算逻辑（定义在 `OperationMixin` 中）。

## 4. 示例演练：`qrofn` 的创建与验证

让我们完整地看一下创建一个 `qrofn` 模糊数并修改其属性时，`Fuzznum` 和 `QrofnStrategy` 是如何协作的。

**场景：`my_fuzznum.md = 1.1`**

1.  **用户代码**:
    ```python
    from axisfuzzy import fuzznum
    my_fuzznum = fuzznum(mtype='qrofn', md=0.8, nmd=0.1, q=3)
    my_fuzznum.md = 1.1  # 尝试进行一次非法的赋值
    ```

2.  **`Fuzznum.__setattr__`**:
    -   `Fuzznum` 门面接收到请求，将其代理给内部的 `_strategy_instance`。
    -   调用 `setattr(self._strategy_instance, 'md', 1.1)`。

3.  **`QROFNStrategy.__setattr__` (继承自 `FuzznumStrategy`)**:
    -   **触发验证器**：`QROFNStrategy` 为 `md` 注册了一个验证器，检查值是否在 `[0, 1]` 之间。
      ```python
      # QROFNStrategy.__init__
      self.add_attribute_validator(
          'md', lambda x: x is None or isinstance(x, (int, float, ...)) and 0 <= x <= 1
      )
      ```
    -   验证器接收到 `1.1`，返回 `False`。
    -   `__setattr__` 方法立即抛出 `ValueError`，赋值流程中断。

**场景：`my_fuzznum.md = 0.9`**

1.  **用户代码**:
    `my_fuzznum.md = 0.9`  # 合法但可能破坏约束的赋值

2.  **`QROFNStrategy.__setattr__`**:
    -   **触发验证器**：检查 `0.9` 是否在 `[0, 1]` 之间。验证通过。
    -   **触发转换器**：`qrofn` 没有为 `md` 注册转换器，跳过此步。
    -   **执行赋值**：`self._strategy_instance` 内部的 `md` 值变为 `0.9`。
    -   **触发变更回调**：`QROFNStrategy` 为 `md` 注册了一个回调 `_on_membership_change`。
      ```python
      # QROFNStrategy.__init__
      self.add_change_callback('md', self._on_membership_change)

      # QROFNStrategy._on_membership_change
      def _on_membership_change(self, ...):
          self._fuzz_constraint() # 调用约束检查

      # QROFNStrategy._fuzz_constraint
      def _fuzz_constraint(self):
          # 检查 0.9^3 + 0.1^3 <= 1
          # 0.729 + 0.001 = 0.73 <= 1. 检查通过。
          ...
      ```
    -   回调函数执行完毕，没有抛出异常。

3.  **结果**:
    -   赋值成功，`my_fuzznum.md` 的值更新为 `0.9`。

通过这种方式，`AxisFuzzy` 确保了数据模型的完整性和正确性，同时将复杂的实现细节优雅地封装在策略类中，为用户保留了一个简洁、强大的门面接口。