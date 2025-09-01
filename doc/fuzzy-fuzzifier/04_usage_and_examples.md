# 用法与示例

本文档通过丰富的代码示例，展示如何在实际应用中使用 `axisfuzzy` 的模糊化功能。我们将从基本用法开始，逐步深入到高级特性，包括多隶属函数参数的模糊化、`qrohfn` 的特殊用法以及 `Fuzzarray` 的高维模糊化等。

## 1. 基本用法

### 1.1 使用预实例化的隶属函数

最直观的使用方式是先创建一个隶属函数实例，然后将其传递给 `Fuzzifier`。当传入隶属函数实例时，`Fuzzifier` 会自动从实例中提取参数，无需重复设置 `mf_params`：

```python
from axisfuzzy.fuzzifier import Fuzzifier
from axisfuzzy.membership import GaussianMF

# 1. 创建隶属函数实例
mf = GaussianMF(sigma=4.0, c=20.0)

# 2. 配置模糊化器
fuzz_engine = Fuzzifier(
    mf=mf,                               # 传入已实例化的隶属函数，参数自动提取
    mtype='qrofn',                       # 使用 q-rung Orthopair Fuzzy Number
    q=3,                                 # q-rung 参数
    pi=0.15                              # 其他策略参数
)

# 3. 执行模糊化
result = fuzz_engine(18.0)
print(result)  # 输出一个 Fuzznum
```

### 1.1.1 使用隶属函数类

另一种方式是直接传入隶属函数类，让 `Fuzzifier` 自动实例化：

```python
from axisfuzzy.fuzzifier import Fuzzifier
from axisfuzzy.membership import GaussianMF

# 直接传入隶属函数类
fuzz_engine = Fuzzifier(
    mf=GaussianMF,                       # 传入隶属函数类
    mtype='qrofn',                       # 使用 q-rung Orthopair Fuzzy Number
    mf_params={"sigma": 4.0, "c": 20.0}, # 参数用于实例化和策略
    q=3,                                 # q-rung 参数
    pi=0.15                              # 其他策略参数
)

# 执行模糊化
result = fuzz_engine(18.0)
print(result)  # 输出一个 Fuzznum
```

**注意**：如果您仍然希望显式提供 `mf_params`（例如，使用与实例不同的参数），您可以这样做：

```python
# 显式提供 mf_params 会覆盖实例中的参数
fuzz_engine = Fuzzifier(
    mf=mf,                               # 传入已实例化的隶属函数
    mtype='qrofn',                       # 使用 q-rung Orthopair Fuzzy Number
    mf_params={"sigma": 2.0, "c": 15.0}, # 显式参数会覆盖实例参数
    q=3,                                 # q-rung 参数
    pi=0.15                              # 其他策略参数
)
```

### 1.2 使用隶属函数名称和参数

另一种更灵活的方式是通过隶属函数的名称和参数来配置：

```python
from axisfuzzy.fuzzifier import Fuzzifier

# 直接使用名称和参数配置
fuzz_engine = Fuzzifier(
    "gaussmf",                           # 隶属函数名称
    mtype="qrofn",                       # 模糊数类型
    mf_params={"sigma": 0.5, "c": 0.0},  # 隶属函数参数
    q=2                                  # 策略参数
)

result = fuzz_engine(0.2)
print(result)
```

## 2. 高级用法

### 2.1 多隶属函数参数模糊化

`axisfuzzy` 的一个强大特性是支持为单个输入值应用多个隶属函数参数集。这在某些类型的模糊数（如 `qrohfn`）中特别有用：

```python
from axisfuzzy.fuzzifier import Fuzzifier

# 使用多个隶属函数参数集
fz = Fuzzifier(
    "gaussmf",                    # 使用高斯隶属函数
    mtype="qrohfn",              # q-rung Orthopair Hesitant Fuzzy Number
    q=2,                         # q-rung 参数
    nmd_generation_mode='proportional',  # 非隶属度生成模式
    mf_params=[                  # 多个参数集
        {"sigma": 0.1, "c": 0.3},
        {"sigma": 0.05, "c": 0.6},
        {"sigma": 0.1, "c": 0.4}
    ]
)

# 对单个值进行模糊化，将生成一个具有多个隶属度的 qrohfn
result = fz(0.5)
print(result)
```

### 2.2 不同类型的多参数模糊化

不同的模糊数类型可能会以不同的方式处理多参数：

```python
# qrofn 的多参数模糊化
fzz = Fuzzifier(
    "gaussmf", 
    mtype="qrofn",
    q=2,
    mf_params=[
        {"sigma": 0.5, "c": 0.0},
        {"sigma": 0.3, "c": 0.5},
        {"sigma": 0.7, "c": 0.4}
    ]
)

# qrohfn 的多参数模糊化
fzh = Fuzzifier(
    "gaussmf",
    mtype="qrohfn",
    q=2,
    nmd_generation_mode='proportional',
    mf_params=[
        {"sigma": 0.1, "c": 0.3},
        {"sigma": 0.05, "c": 0.6}
    ]
)

# 比较两种类型的输出
result_qrofn = fzz(0.4)
result_qrohfn = fzh(0.4)
print("QROFN result:", result_qrofn)
print("QROHFN result:", result_qrohfn)
```

### 2.3 Fuzzarray 高维模糊化

`Fuzzifier` 支持对 NumPy 数组进行高效的批量模糊化：

```python
import numpy as np
from axisfuzzy.fuzzifier import Fuzzifier

# 创建模糊化器
fuzz_engine = Fuzzifier(
    "gaussmf",
    mtype="qrofn",
    q=2,
    mf_params={"sigma": 0.5, "c": 0.5}
)

# 创建一个测试数组
x = np.array([0.2, 0.4, 0.6, 0.8])

# 批量模糊化
result = fuzz_engine(x)  # 返回 Fuzzarray
print(result)

# 对二维数组进行模糊化
x_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
result_2d = fuzz_engine(x_2d)
print(result_2d)
```

## 3. 实际应用示例

### 3.1 评估系统

假设我们要构建一个评估系统，使用多个指标来评估一个对象：

```python
from axisfuzzy.fuzzifier import Fuzzifier
import numpy as np

# 创建评估用的模糊化器
evaluator = Fuzzifier(
    "gaussmf",
    mtype="qrohfn",
    q=2,
    nmd_generation_mode='proportional',
    mf_params=[
        {"sigma": 0.1, "c": 0.3},  # 保守评估
        {"sigma": 0.1, "c": 0.5},  # 中性评估
        {"sigma": 0.1, "c": 0.7}   # 乐观评估
    ]
)

# 评估数据
scores = np.array([0.45, 0.62, 0.78])  # 多个指标的得分

# 进行模糊评估
evaluations = evaluator(scores)
print("Fuzzy evaluations:", evaluations)
```

### 3.2 动态参数配置

展示如何根据运行时的需求动态配置模糊化器：

```python
def create_dynamic_fuzzifier(data_range, sensitivity):
    """根据数据范围和敏感度创建合适的模糊化器"""
    
    # 根据数据范围计算合适的中心点
    centers = np.linspace(data_range[0], data_range[1], 3)
    
    # 根据敏感度调整 sigma
    sigma = (data_range[1] - data_range[0]) * sensitivity
    
    # 创建模糊化器
    return Fuzzifier(
        "gaussmf",
        mtype="qrohfn",
        q=2,
        mf_params=[
            {"sigma": sigma, "c": c} for c in centers
        ]
    )

# 使用示例
data_range = (0, 100)
sensitivity = 0.1

fuzzifier = create_dynamic_fuzzifier(data_range, sensitivity)
result = fuzzifier(75)
print("Dynamic fuzzification result:", result)
```

### 3.3 混合使用多种隶属函数

展示如何在同一个应用中使用不同类型的隶属函数：

```python
from axisfuzzy.membership import GaussianMF, TriangularMF
from axisfuzzy.fuzzifier import Fuzzifier

# 创建不同的模糊化器
gaussian_fuzzifier = Fuzzifier(
    GaussianMF(sigma=0.1, c=0.5),  # 传入实例，参数自动提取
    mtype="qrofn",
    q=2
)

triangular_fuzzifier = Fuzzifier(
    TriangularMF(a=0.2, b=0.5, c=0.8),  # 传入实例，参数自动提取
    mtype="qrofn",
    q=2
)

# 对同一个值使用不同的模糊化器
value = 0.6
gaussian_result = gaussian_fuzzifier(value)
triangular_result = triangular_fuzzifier(value)

print("Gaussian fuzzification:", gaussian_result)
print("Triangular fuzzification:", triangular_result)
```

这些示例展示了 `axisfuzzy` 的灵活性和强大功能。您可以根据具体需求，选择合适的配置方式，并充分利用多参数模糊化和高维数组处理的特性。