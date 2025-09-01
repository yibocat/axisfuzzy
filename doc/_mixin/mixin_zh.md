# AxisFuzzy Mixin 系统

AxisFuzzy Mixin 系统是一个强大的机制，用于动态扩展核心类（如 `Fuzznum` 和 `Fuzzarray`）的功能。它允许在不直接修改类定义的情况下，为这些类添加新方法，并在 `axisfuzzy` 命名空间中创建相应的顶层函数。这是通过一个在库初始化期间管理和注入这些函数的中央注册表来实现的。

## 核心概念: `MixinFunctionRegistry`

该系统的核心是 `MixinFunctionRegistry`。它提供了一种基于装饰器的方法来注册函数，并指定它们应如何集成到 AxisFuzzy 生态系统中。

函数可以通过三种方式注入：
- **`instance_function`**: 函数成为目标类的实例方法（例如，`my_fuzzarray.my_func()`）。
- **`top_level_function`**: 函数可直接从 `axisfuzzy` 模块获得（例如，`axisfuzzy.my_func(...)`）。
- **`both`**: 函数既可作为实例方法，也可作为顶层函数使用。

这种设计保持了核心类定义的整洁，同时允许轻松、有组织地扩展其功能。

## 可用的 Mixin 函数

以下函数通过 Mixin 系统提供。它们按其源模块分组。

### 数组操作 (`axisfuzzy.mixin.function`)

这些函数为 `Fuzzarray` 和 `Fuzznum` 对象提供了类似 `numpy` 的数组操作功能。

| 函数/方法 | 注入类型 | 描述 |
|---|---|---|
| `reshape` | both | 在不改变数据的情况下为数组赋予新的形状。 |
| `flatten` | both | 返回一个折叠成一维的数组副本。 |
| `squeeze` | both | 从数组的形状中移除单维条目。 |
| `copy` | top_level_function | 返回模糊对象的深拷贝。 |
| `ravel` | both | 返回一个连续的扁平化数组。如果可能，返回视图。 |
| `transpose` | top_level_function | 返回轴转置后的模糊对象的视图。 |
| `broadcast_to` | both | 将模糊对象广播到新的形状。 |
| `item` | both | 返回模糊对象的标量项。 |
| `sort` | both | 返回模糊数组的排序副本。 |
| `argsort` | both | 返回对模糊数组进行排序的索引。 |
| `argmax` | both | 返回沿轴的最大值的索引。 |
| `argmin` | both | 返回沿轴的最小值的索引。 |
| `concat` | both | 沿指定轴拼接一个或多个 `Fuzzarray`。 |
| `stack` | both | 沿新轴堆叠多个 `Fuzzarray`。 |
| `append` | both | 向对象追加元素。 |
| `pop` | both | 从一维数组中移除并返回一个元素。 |

### 数学与聚合操作 (`axisfuzzy.mixin.ops`)

这些函数提供了类似 `numpy` 的数学和聚合功能。

| 函数/方法 | 注入类型 | 描述 |
|---|---|---|
| `sum` | both | 计算所有元素的总和。 |
| `mean` | both | 计算所有元素的平均值。 |
| `max` | both | 找到最大元素。 |
| `min` | both | 找到最小元素。 |
| `prod` | both | 计算所有元素的乘积。 |
| `var` | both | 计算方差。 |
| `std` | both | 计算标准差。 |
