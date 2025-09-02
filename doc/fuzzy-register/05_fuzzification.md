# 第五步：实现模糊化策略

在定义了模糊数的后端、核心操作和随机生成方法之后，关键的一步是将其与 `axisfuzzy` 的模糊化引擎（`Fuzzifier`）集成。这一集成是通过实现一个自定义的 **模糊化策略（`FuzzificationStrategy`）** 来完成的。

本章将详细阐述如何为你的模糊数类型编写一个专属的模糊化策略，使其能够被 `Fuzzifier` 调用，从而将精确数值转换为你的模糊数。

## 1. 理解 `FuzzificationStrategy` 的角色

`FuzzificationStrategy` 是一个定义 **“如何将隶属度转化为特定模糊数”** 的蓝图。它是一个位于 `/Users/yibow/Documents/Fuzzy/AxisFuzzy/axisfuzzy/fuzzifier/strategy.py` 的抽象基类，充当了 `Fuzzifier` 引擎和具体模糊数类型之间的桥梁。

`Fuzzifier` 引擎本身不关心如何计算非隶属度或处理犹豫度，它只负责：
1.  解析用户配置（隶属函数、目标模糊类型等）。
2.  根据用户指定的 `mtype`（模糊数类型）和 `method`（方法），从注册表中查找对应的策略。
3.  调用该策略，完成模糊化任务。

因此，要让你的模糊数被 `Fuzzifier` 支持，就必须提供一个实现了特定转换逻辑的策略。

## 2. `FuzzificationStrategy` 基类结构

所有策略都必须继承自 `FuzzificationStrategy` 基类，其核心结构如下：

```python
# axisfuzzy/fuzzifier/strategy.py

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union
# ...

class FuzzificationStrategy(ABC):
    # 策略的身份标识
    mtype: Optional[str] = None
    method: Optional[str] = None

    def __init__(self, q: Optional[int] = None, **kwargs: Any):
        # ...

    @abstractmethod
    def fuzzify(self,
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
        """模糊化输入数据"""
        pass
```

### 关键组件：
-   `mtype: str`: **必须定义**。这是你的模糊数类型的唯一字符串标识（例如 `'qrofn'`, `'my_fuzzy_type'`）。它必须与你在后端和操作注册表中使用的 `mtype` 完全一致。
-   `method: str`: **必须定义**。这是该策略的方法名（例如 `'default'`, `'pi_based'`）。一个 `mtype` 可以有多种策略，用 `method` 区分。
-   `__init__(self, **kwargs)`: 构造函数，用于接收该策略运行所需的特定参数。例如，`qrofn` 的策略需要 `pi` 参数来计算非隶属度。`Fuzzifier` 会将用户传入的、非隶属函数参数的所有其他关键字参数都传递给这个构造函数。
-   `fuzzify(...)`: **必须实现的抽象方法**。这是策略的核心，包含了将精确值 `x` 转换为模糊数的完整逻辑。

## 3. 实现 `fuzzify` 核心方法

`fuzzify` 方法是实现模糊化逻辑的地方。它的签名是固定的，你需要在这个框架内实现你的算法。

```python
def fuzzify(self,
            x: Union[float, int, list, np.ndarray],
            mf_cls: type,
            mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
    # ... 你的实现 ...
```

### 参数解析：
-   `x`: 用户传入的精确输入，可以是单个数值（`float`, `int`）或数组（`list`, `np.ndarray`）。**最佳实践**是首先使用 `np.asarray(x)` 将其统一转换为 NumPy 数组进行处理。
-   `mf_cls`: 用户指定的隶属函数**类**（例如 `GaussianMF`），而不是实例。
-   `mf_params_list`: 一个包含一个或多个字典的列表。每个字典都定义了一套用于实例化 `mf_cls` 的参数。
    -   `len(mf_params_list) == 1`: 标准模糊化场景。
    -   `len(mf_params_list) > 1`: 多参数场景，通常用于生成犹豫模糊数（如 `qrohfn`）或对多个模型进行聚合。

### 返回值：
-   如果输入 `x` 是标量，应返回一个 `Fuzznum` 实例。
-   如果输入 `x` 是数组，应返回一个 `Fuzzarray` 实例。

### 实现步骤与模式

#### 模式A：单参数与多参数分离处理（如 `qrofn`）

`qrofn` 的策略为我们展示了一种常见的模式：它将多组 `mf_params` 视为独立的模糊化任务，并最终将结果堆叠成一个更高维的 `Fuzzarray`。

```python
# axisfuzzy/fuzztype/qrofs/fuzzification.py 逻辑简化
def fuzzify(self, x, mf_cls, mf_params_list):
    x = np.asarray(x, dtype=float)
    results = []

    # 1. 遍历每一组隶属函数参数
    for params in mf_params_list:
        # 2. 创建隶属函数实例
        mf = mf_cls(**params)

        # 3. 矢量化计算隶属度
        mds = np.clip(mf.compute(x), 0, 1)

        # 4. 根据策略逻辑计算非隶属度
        #    这是你的模糊数类型的核心转换算法
        nmds = np.maximum(1 - mds**self.q - self.pi**self.q, 0.0) ** (1/self.q)

        # 5. 使用你的模糊数后端创建 Fuzzarray
        backend_cls = get_registry_fuzztype().get_backend(self.mtype)
        backend = backend_cls.from_arrays(mds=mds, nmds=nmds, q=self.q)
        results.append(Fuzzarray(backend=backend, mtype=self.mtype, q=self.q))

    # 6. 根据参数组数量决定返回结果
    if len(results) == 1:
        # 如果只有一个参数集，直接返回该 Fuzzarray
        # 如果输入 x 是标量，Fuzzarray 内部只有一个元素，可以由调用者处理
        return results[0]
    else:
        # 如果有多个参数集，将多个 Fuzzarray 堆叠起来
        from ...mixin.factory import _stack_factory
        return _stack_factory(results[0], *results[1:], axis=0)
```

#### 模式B：多参数融合处理（如 `qrohfn`）

`qrohfn` 的策略则代表了另一种模式：它将多组 `mf_params` 的计算结果**融合**到单个模糊数（犹豫模糊数）中。

```python
# axisfuzzy/fuzztype/qrohfs/fuzzification.py 逻辑简化
def fuzzify(self, x, mf_cls, mf_params_list):
    x = np.asarray(x, dtype=float)
    shape = x.shape or ()

    # 准备用于存储 list of mds/nmds 的 object 数组
    mds_obj = np.empty(shape, dtype=object)
    nmds_obj = np.empty(shape, dtype=object)

    # 1. 遍历输入数据的每个元素
    for idx in np.ndindex(shape or (1,)):
        xi = x[idx] if shape else x.item()

        # 2. 对当前元素 xi，计算所有隶属函数下的隶属度
        mds_list = [float(np.clip(mf_cls(**p).compute(xi), 0, 1)) for p in mf_params_list]

        # 3. 根据这些隶属度计算非隶属度集合
        nmds_list = self._compute_nmds(np.asarray(mds_list))

        # 4. 将隶属度和非隶属度列表存入 object 数组
        if shape:
            mds_obj[idx] = np.asarray(mds_list, dtype=float)
            nmds_obj[idx] = np.asarray(nmds_list, dtype=float)
        else:
            mds_obj = np.asarray(mds_list, dtype=object)
            nmds_obj = np.asarray(nmds_list, dtype=object)

    # 5. 使用后端从 object 数组创建 Fuzzarray
    backend_cls = get_registry_fuzztype().get_backend(self.mtype)
    backend = backend_cls.from_arrays(mds=mds_obj, nmds=nmds_obj, q=self.q)
    arr = Fuzzarray(backend=backend, mtype=self.mtype, q=self.q)

    # 6. 如果输入是标量，返回 Fuzznum
    if x.ndim == 0:
        return arr[()]
    return arr
```

## 4. 注册模糊化策略

写好策略类后，最后一步是使用 `@register_fuzzifier` 装饰器将其注册到系统中，这样 `Fuzzifier` 才能找到它。

```python
from ...fuzzifier import register_fuzzifier

@register_fuzzifier(is_default=True)  # 注册并设为默认
class MyFuzzyTypeStrategy(FuzzificationStrategy):
    mtype = 'my_fuzzy_type'
    method = 'default'

    def __init__(self, ...):
        # ...

    def fuzzify(self, ...):
        # ...
```

-   `@register_fuzzifier()`: 装饰器会自动读取类的 `mtype` 和 `method` 属性，并将其注册。
-   `is_default=True`: 这是一个非常重要的参数。如果设为 `True`，当用户创建 `Fuzzifier` 时只提供了 `mtype` 而没有提供 `method`，系统将自动使用这个策略。**每个 `mtype` 都应该至少有一个默认策略。**

## 5. 总结

为你的模糊数类型实现模糊化策略是将其完全集成到 `axisfuzzy` 生态的最后一步。核心步骤如下：

1.  **创建策略类**: 创建一个继承自 `FuzzificationStrategy` 的新类。
2.  **定义身份**: 在类中明确定义 `mtype` 和 `method` 字符串。
3.  **实现构造函数**: 在 `__init__` 中接收并处理该策略独有的参数。
4.  **实现 `fuzzify` 方法**: 这是核心。在此方法中，实现从精确值 `x` 和隶属度计算到生成你的模糊数（`Fuzznum`/`Fuzzarray`）的完整转换逻辑。
    -   决定如何处理单/多隶属函数参数。
    -   调用你之前注册的模糊数后端来创建实例。
5.  **注册策略**: 使用 `@register_fuzzifier` 装饰器将你的策略类注册到系统中，并酌情将其设为默认策略。

完成这些步骤后，用户就可以像使用任何内置模糊数类型一样，通过 `axisfuzzy.Fuzzifier` 来使用你的自定义模糊数了。

```python
from axisfuzzy import Fuzzifier

# 假设你的策略已正确实现和注册
fuzz_engine = Fuzzifier(
    mf='gaussmf',
    mtype='my_fuzzy_type',  # 使用你的 mtype
    mf_params={'sigma': 0.5, 'c': 0.5},
    # ... 传入你的策略需要的其他参数 ...
)

# 执行模糊化
result = fuzz_engine(0.4)
print(result) # 输出应为你自定义的 Fuzznum
```