# `mixin` 系统深度剖析：实现-注册-注入

与 `extension` 系统专注于“语义”扩展不同，`mixin` 系统的核心使命是为 `Fuzzarray` 和 `Fuzznum` 提供通用的、`mtype` 无关的**结构化操作**。它借鉴了 NumPy 的设计哲学，旨在赋予 `Fuzzarray` 强大的、类似 `ndarray` 的数据操作能力。其架构比 `extension` 更为直接，遵循“实现-注册-注入” (Implement-Register-Inject) 的流程，实现了极高的性能和无缝的集成。

本篇文档将深入 `mixin` 系统的内部机制，详细剖析其实现、注册和注入的各个环节，并重点解读其核心注册机制。

---

## 1. 核心架构概览

`mixin` 系统的生命周期可以划分为三个清晰的阶段，其重点在于“静态”的预处理和注入，从而实现零运行时开销。

1.  **实现 (Implementation)**：在 `axisfuzzy/mixin/factory.py` 中，开发者预先实现一系列通用的数组操作函数。这些函数直接作用于 `Fuzzarray` 的底层 Struct-of-Arrays (SoA) 后端，因此与具体的 `mtype` 无关。

2.  **注册 (Registration)**：通过 `axisfuzzy/mixin/registry.py` 中定义的 `@register_mixin` 装饰器，在 `axisfuzzy/mixin/register.py` 中，将 `factory` 中实现的函数“包装”并注册到全局的 `MixinFunctionRegistry` 注册表中。此过程将函数实现与目标注入类型（如实例方法、顶层函数）和目标类关联起来。

3.  **注入 (Injection)**：在库初始化时（当 `axisfuzzy/mixin/__init__.py` 被导入时），`MixinFunctionRegistry` 会遍历所有注册的 `mixin` 函数，并通过 `setattr` 将它们直接“混入”到 `Fuzzarray`、`Fuzznum` 类或 `axisfuzzy` 顶层命名空间中。

这种设计的最大优势在于**性能**。由于功能在加载时就已经被直接注入，调用一个 `mixin` 方法（如 `my_array.reshape(...)`）就如同调用一个普通的类方法一样，没有任何间接层或运行时分发开销。

让我们逐一深入每个环节。

---

## 2. 第一阶段：实现 (Implementation)

实现是 `mixin` 系统的基础。所有功能都在 `axisfuzzy/mixin/factory.py` 文件中以普通函数的形式预先定义好。这些函数被称为“工厂函数”。

### `factory.py` 的设计哲学

`factory.py` 中的函数被设计为直接操作 `Fuzzarray` 的 `_backend` 属性，这是一个 `FuzzarrayBackend` 实例，它以 SoA (Struct-of-Arrays) 的形式存储着模糊数的所有组件数据（如隶属度、非隶属度等）。

**源码解读 (`axisfuzzy/mixin/factory.py`)**

以 `_reshape_factory` 函数为例，这是 `reshape` 功能的真正实现者：

```python
# axisfuzzy/mixin/factory.py

def _reshape_factory(obj: Union[Fuzznum, Fuzzarray], *shape: int) -> Fuzzarray:
    """
    Gives a new shape to an array without changing its data.
    Works with the SoA backend infrastructure.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    if isinstance(obj, Fuzznum):
        # Create a scalar Fuzzarray first, then reshape
        arr = Fuzzarray(data=obj, shape=())
        return _reshape_factory(arr, *shape)

    # 1. 从后端获取所有组件数组
    components = obj.backend.get_component_arrays()

    # 2. 对每个组件数组执行 NumPy 的 reshape 操作
    try:
        reshaped_components = [comp.reshape(shape) for comp in components]
    except ValueError as e:
        raise ValueError(f"Cannot reshape array of size {obj.size} into shape {shape}") from e

    # 3. 使用变形后的组件数组创建新的后端实例
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*reshaped_components, q=obj.q)

    # 4. 用新的后端创建新的 Fuzzarray 实例并返回
    return Fuzzarray(backend=new_backend)
```

这个实现清晰地体现了 `mtype` 无关的特性：
1.  它不关心 `obj` 的 `mtype` 是什么，只关心其 `backend`。
2.  它调用 `backend.get_component_arrays()` 获取一个 NumPy 数组列表。
3.  它对列表中的每个 NumPy 数组统一执行 `reshape` 操作。
4.  最后，它使用更新后的组件数组，创建一个与原始数组类型相同的新 `Fuzzarray` 实例并返回。

所有 `factory` 中的函数，如 `_flatten_factory`, `_transpose_factory`, `_concat_factory` 等，都遵循这一模式，确保了其通用性。

---

## 3. 第二阶段：注册 (Registration)

实现了功能之后，需要通过注册机制告知 `AxisFuzzy` 框架它们的存在以及如何使用它们。这一阶段的核心是 **`@register_mixin` 装饰器** 和 **`MixinFunctionRegistry` 注册表**。

### 3.1. `MixinFunctionRegistry` 注册表

`MixinFunctionRegistry` (位于 `axisfuzzy/mixin/registry.py`) 是 `mixin` 系统的中心枢纽。它是一个全局单例，负责存储所有已注册的 `mixin` 函数及其注入配置。其内部数据结构大致如下：

```python
# axisfuzzy/mixin/registry.py (Conceptual)

class MixinFunctionRegistry:
    def __init__(self):
        # {'reshape': <function _reshape_impl at 0x...>}
        self._functions: Dict[str, Callable] = {}
        
        # {'reshape': {'target_classes': ['Fuzzarray', 'Fuzznum'], 'injection_type': 'both'}}
        self._metadata: Dict[str, Dict[str, Any]] = {}
```

-   `_functions`: 存储注册的函数名到其“包装实现”的映射。
-   `_metadata`: 存储每个函数的元数据，包括注入目标类和注入类型。

### 3.2. `@register_mixin` 装饰器

为了简化注册流程，`axisfuzzy/mixin/registry.py` 提供了一个便捷的装饰器 `@register_mixin`。它是在 `MixinFunctionRegistry.register` 方法基础上封装的。

**源码解读 (`axisfuzzy/mixin/registry.py`)**

```python
# axisfuzzy/mixin/registry.py

def register_mixin(name: str,
                   target_classes: Optional[List[str]] = None,
                   injection_type: Literal['instance_function', 'top_level_function', 'both'] = 'both') -> Callable:
    """
    Convenience decorator for registering mixin functions.
    """
    return get_registry_mixin().register(name, target_classes, injection_type)
```

这个装饰器接收三个关键参数：
-   `name`: 注入后函数或方法的名称。
-   `target_classes`: 一个字符串列表，指定该函数将作为实例方法注入到哪些类中（如 `['Fuzzarray', 'Fuzznum']`）。
-   `injection_type`: 注入类型，决定了函数如何暴露给用户。

### 3.3. 注册过程 (`register.py`)

与 `extension` 系统不同，`@register_mixin` 装饰器**不直接**用于 `factory.py` 中的工厂函数。而是在 `axisfuzzy/mixin/register.py` 文件中，为每个工厂函数创建一个简单的包装函数，并对该包装函数使用装饰器。

这种方式实现了**实现**与**注册**的完美分离。

**源码解读 (`axisfuzzy/mixin/register.py`)**

```python
# axisfuzzy/mixin/register.py

from .registry import register_mixin
from .factory import (
    _reshape_factory, _flatten_factory, _transpose_factory, 
    _copy_factory, _T_impl
)

# 示例 1: 注册为实例方法和顶层函数 ('both')
@register_mixin(name='reshape', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _reshape_impl(self, *shape: int):
    """Gives a new shape to a fuzzy array without changing its data."""
    return _reshape_factory(self, *shape)

# 示例 2: 只注册为顶层函数 ('top_level_function')
@register_mixin(name='copy', injection_type='top_level_function')
def _copy_impl(obj):
    """Return a deep copy of the fuzzy object."""
    return _copy_factory(obj)

# 示例 3: 只注册为实例属性/方法 ('instance_function')
@register_mixin(name='T', target_classes=["Fuzzarray", "Fuzznum"], injection_type='instance_function')
@property
def _T_impl(self):
    """View of the fuzzy array with axes transposed."""
    return _transpose_factory(self)

```

这个过程非常清晰：
1.  从 `factory` 导入真正的实现函数（如 `_reshape_factory`）。
2.  定义一个包装函数（如 `_reshape_impl`），其签名符合最终用户调用的方式。
3.  该包装函数内部只做一件事：调用对应的工厂函数。
4.  使用 `@register_mixin` 装饰这个包装函数，并提供所有必要的元数据。

---

## 4. 第三阶段：注入 (Injection)

注入是 `mixin` 系统的最后一步，也是使其功能对用户可用的关键。这个过程由 `axisfuzzy/mixin/__init__.py` 中的 `apply_mixins` 函数驱动，在库加载时自动执行。

### `apply_mixins` 和 `build_and_inject`

**源码解读 (`axisfuzzy/mixin/__init__.py`)**

```python
# axisfuzzy/mixin/__init__.py

# ...
from .registry import get_registry_mixin
from ..core import Fuzznum, Fuzzarray
from . import register  # <-- 关键：导入 register.py 以执行所有装饰器

_applied = False

def _apply_functions(target_module_globals: Dict[str, Any] | None = None) -> bool:
    global _applied
    if _applied:
        return True

    class_map = {'Fuzznum': Fuzznum, 'Fuzzarray': Fuzzarray}
    
    # ... 获取顶层模块的命名空间 ...

    try:
        # 调用注册表执行注入
        get_registry_mixin().build_and_inject(class_map, target_module_globals)
        _applied = True
        return True
    except Exception as e:
        warnings.warn(f"Failed to injection mixin functions: {e}")
        return False

apply_mixins = _apply_functions
```

**源码解读 (`axisfuzzy/mixin/registry.py`)**

`build_and_inject` 方法是注入的核心逻辑：

```python
# axisfuzzy/mixin/registry.py (MixinFunctionRegistry 类内)

def build_and_inject(self, class_map: Dict[str, type], module_namespace: Dict[str, Any]):
    for name, func in self._functions.items():
        meta = self._metadata[name]
        injection_type = meta['injection_type']
        target_classes = meta['target_classes']

        # 注入实例方法
        if injection_type in ['instance_function', 'both']:
            for class_name in target_classes:
                if class_name in class_map:
                    target_class = class_map[class_name]
                    setattr(target_class, name, func)

        # 注入顶层函数
        if injection_type in ['top_level_function', 'both']:
            if injection_type == 'both':
                # 创建一个包装器，将顶层调用委托给实例方法
                @functools.wraps(func)
                def top_level_wrapper(obj: Any, *args, current_name=name, **kwargs):
                    # ...
                    return getattr(obj, current_name)(*args, **kwargs)
                module_namespace[name] = top_level_wrapper
            else:
                # 直接注入
                module_namespace[name] = func
```

注入过程总结如下：
1.  `apply_mixins` 准备好目标类 (`Fuzznum`, `Fuzzarray`) 和顶层命名空间。
2.  它调用 `MixinFunctionRegistry.build_and_inject`。
3.  `build_and_inject` 遍历注册表中的每一项。
4.  根据元数据中的 `injection_type` 和 `target_classes`，使用 `setattr` 将函数（如 `_reshape_impl`）动态地添加到目标类上，或直接添加到模块的全局命名空间中。

执行完毕后，`Fuzzarray` 类就拥有了 `reshape` 方法，就像它是在类定义中直接写的一样。任何 `Fuzzarray` 实例都可以立即调用 `my_array.reshape(...)`，同时用户也可以使用 `axisfuzzy.reshape(my_array, ...)`。

---

## 5. 总结：一个高效的静态扩展系统

`mixin` 系统的“实现-注册-注入”架构，共同构成了一个高效、类型安全且易于维护的静态扩展机制。

-   **实现 (`factory.py`)**：提供 `mtype`-无关的通用算法，直接操作底层 SoA 数据结构，是功能的核心。
-   **注册 (`register.py` & `registry.py`)**：通过 `@register_mixin` 装饰器将实现与注入元数据关联，并存入中心化的 `MixinFunctionRegistry`，是连接实现与注入的桥梁。
-   **注入 (`__init__.py` & `registry.py`)**：在库启动时将功能静态地“混入”到核心类和命名空间中，实现对用户透明的、零运行时开销的调用。

这个系统是 `AxisFuzzy` 能够提供丰富、高性能的 NumPy-like 接口的关键。它与 `extension` 系统互为补充，一个负责结构，一个负责语义，共同为 `AxisFuzzy` 打造了一个全面而强大的扩展生态系统。