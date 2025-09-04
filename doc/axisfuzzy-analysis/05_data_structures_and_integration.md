# 5. 核心数据结构与 Pandas 集成

`axisfuzzy-analysis` 系统的设计哲学之一是与现有数据科学生态系统（尤其是 Pandas）的无缝集成。我们深知，用户的工作流通常始于一个标准的 `pandas.DataFrame`。因此，我们引入了 `FuzzyDataFrame` 这一核心数据结构，并通过一个直观的 `.fuzzy` 访问器，在 Pandas 和模糊分析世界之间架起了一座坚实的桥梁。

本文档将详细阐述 `FuzzyDataFrame` 的设计理念、如何通过 `.fuzzy` 访问器与 Pandas DataFrame 交互，以及支撑这一切的底层数据结构。

## `FuzzyDataFrame`: 模糊数据的容器

`FuzzyDataFrame` 是 `axisfuzzy` 中用于处理二维、带标签的模糊数据的核心容器。你可以将其类比为 `pandas.DataFrame`，但它专为存储和操作模糊数而设计，并进行了性能优化。

### 设计理念

- **列式存储**：`FuzzyDataFrame` 内部采用列式存储，每一列都是一个高性能的 `axisfuzzy.core.Fuzzarray` 对象。这种设计不仅提升了按列操作的效率，也使得数据结构更加规整。
- **Pandas 兼容的 API**：它拥有与 `pandas.DataFrame` 类似的数据访问接口，例如通过 `[]` 访问列、拥有 `.index` 和 `.columns` 属性等，这大大降低了用户的学习成本。
- **类型一致性**：一个 `FuzzyDataFrame` 中的所有列（`Fuzzarray`）必须具有相同的模糊类型（`mtype`），以确保后续分析的一致性和有效性。

### 核心构成

- **`_data`**: 一个字典，键是列名（`str`），值是 `Fuzzarray` 对象。这是数据的实际存储位置。
- **`_index`**: 一个 `pd.Index` 对象，存储行标签。
- **`_columns`**: 一个 `pd.Index` 对象，存储列标签。
- **`_mtype`**: 一个字符串，表示该 `FuzzyDataFrame` 中模糊数的类型（例如 `'triangular'`）。

## `.fuzzy` 访问器：连接 Pandas 与 `axisfuzzy` 的桥梁

为了让用户能够以最自然的方式从原始数据过渡到模糊分析，我们开发了 `.fuzzy` 访问器。任何一个 `pandas.DataFrame` 实例都可以通过 `.fuzzy` 属性，调用 `axisfuzzy` 提供的强大功能。

`.fuzzy` 访问器是整个分析系统的主要用户入口，它提供了两大核心功能：

1.  **数据转换**：将清晰的 `pandas.DataFrame` 转换为 `FuzzyDataFrame`。
2.  **模型执行**：直接在 `pandas.DataFrame` 上运行 `FuzzyPipeline` 或 `Model`。

### 将 Pandas DataFrame 转换为 `FuzzyDataFrame`

这是进入模糊分析世界的第一步。通过 `.fuzzy.to_fuzz_dataframe()` 方法，你可以将一个包含清晰数值的 DataFrame 模糊化。

这个过程需要一个 `Fuzzifier`（模糊化器）实例，它定义了如何将清晰数转换为模糊数。`Fuzzifier` 的创建非常灵活，您可以指定隶属函数、目标模糊类型 (`mtype`) 以及相关参数。

```python
import pandas as pd
import numpy as np
from axisfuzzy.fuzzifier import Fuzzifier

# 1. 准备一个标准的 pandas.DataFrame
#    为了使输出与预期的模糊格式一致，我们创建一些随机数据
np.random.seed(42)
crisp_data = pd.DataFrame(np.random.rand(5, 3), columns=['Attr1', 'Attr2', 'Attr3'])

# 2. 创建并配置一个 Fuzzifier
#    这里我们使用高斯隶属函数 ('gaussmf') 将清晰值转换为 q-rung Orthopair Fuzzy Numbers ('qrofn')
#    这是正确的 Fuzzifier 创建方式，它通过隶属函数、目标类型和相关参数进行配置
fuzzifier = Fuzzifier(
    mf="gaussmf",                        # 隶属函数名称
    mtype="qrofn",                       # 目标模糊数类型
    mf_params={"sigma": 0.5, "c": 0.5},  # 隶属函数参数
    q=2                                  # 策略参数 (q-rung)
)

# 3. 使用 .fuzzy 访问器进行转换
fuzzy_df = crisp_data.fuzzy.to_fuzz_dataframe(fuzzifier)

print(fuzzy_df)
# Output (输出格式经过修正，以 <隶属度, 非隶属度> 格式显示):
#              Attr1            Attr2            Attr3
# 0  <0.7092,0.0908>  <0.0053,0.7947>  <0.0039,0.7961>
# 1  <0.1499,0.6501>  <0.0039,0.7961>       <0.9767,0>
# 2       <0.8417,0>  <0.0128,0.7872>  <0.0039,0.7961>
# 3  <0.0039,0.7961>  <0.2409,0.5591>    <0.612,0.188>
# 4  <0.0039,0.7961>  <0.0039,0.7961>  <0.0149,0.7851>

print(type(fuzzy_df))
# Output:
# <class 'axisfuzzy.analysis.dataframe.frame.FuzzyDataFrame'>
```

在幕后，`.fuzzy.to_fuzz_dataframe()` 调用了 `FuzzyDataFrame.from_pandas(df, fuzzifier)` 类方法，遍历 DataFrame 的每一列，并使用指定的 `fuzzifier` 将其转换为 `Fuzzarray`。

### 直接在 Pandas DataFrame 上运行分析模型

`.fuzzy` 访问器最强大的功能是 `run()` 方法。它允许你直接在一个清晰的 `pandas.DataFrame` 上执行一个复杂的 `Model` 或 `FuzzyPipeline`，而无需手动进行数据转换。

`axisfuzzy` 会智能地将 DataFrame 注入到模型或流水线的输入节点。

```python
import pandas as pd
from axisfuzzy.analysis import Model, AnalysisComponent
from axisfuzzy.fuzzifier import Fuzzifier

# 假设我们有一个定义好的 Model
class MyAnalysisModel(Model):
    def __init__(self):
        super().__init__()
        # 正确地创建 Fuzzifier
        self.fuzzifier = Fuzzifier("gaussmf", mtype="qrofn", mf_params={"sigma": 0.5, "c": 0.5}, q=2)
        self.normalizer = MyNormalizerComponent() # 假设的组件
        self.aggregator = MyAggregationComponent() # 假设的组件

    def forward(self, data, weights):
        # Fuzzifier 本身不是直接调用的，它在 to_fuzz_dataframe 中使用
        # 或者在组件内部使用。这里为了示例简化，我们假设模型第一步是模糊化
        # 在实际场景中，模糊化通常是流水线的一个组件
        # 此处我们直接在 run 方法中处理
        pass # 实际逻辑会更复杂

# 1. 准备输入数据
crisp_df = pd.DataFrame(...)
weights_df = pd.DataFrame(...)

# 2. 实例化模型
my_model = MyAnalysisModel()

# 3. 使用 .fuzzy.run() 执行模型
#    axisfuzzy 会自动将 crisp_df 注入到 'data' 输入
#    我们只需要通过关键字参数提供其他输入，如 'weights'
#    模型内部的模糊化逻辑会被触发
final_results = crisp_df.fuzzy.run(my_model, weights=weights_df)
```

**输入注入规则**：

-   如果模型/流水线只有一个输入，DataFrame 会被自动传递给该输入。
-   如果模型/流水线有多个输入，`axisfuzzy` 会优先将 DataFrame 注入到名为 `init_data` 或 `data` 的输入。
-   所有其他必需的输入必须通过关键字参数（`**kwargs`）在 `run()` 方法中提供。

这种设计极大地简化了工作流，让用户可以将注意力集中在模型构建上，而不是繁琐的数据传递和转换上。

## 底层数据结构：`Fuzzarray` 和 `Fuzznum`

`FuzzyDataFrame` 的高效运作离不开其底层的核心数据结构：`Fuzzarray` 和 `Fuzznum`。

-   **`Fuzznum`**: 这是 `axisfuzzy` 中表示**单个模糊数**的基本对象。它封装了定义模糊数的核心参数。例如，一个 q-rung Orthopair Fuzzy Number (`qrofn`) 被表示为一个隶属度 (`membership`) 和非隶属度 (`non-membership`) 的元组，如 `<0.7092, 0.0908>`。`Fuzznum` 提供了与模糊数相关的基本计算和操作。

-   **`Fuzzarray`**: 这是一个专门用于存储 `Fuzznum` 对象的**一维数组**结构。它被设计为高性能的容器，类似于 NumPy 的 `ndarray`，但专门为模糊数操作进行了优化。`FuzzyDataFrame` 的每一列都是一个 `Fuzzarray` 实例。它负责批量执行向量化的模糊数运算，这是 `axisfuzzy` 高性能的关键。

### 数据结构层级关系

这三者形成了清晰的层级关系：

```
+-----------------------+
|    FuzzyDataFrame     |  (二维模糊数据表)
|-----------------------|
| +-------------------+ |
| |    Fuzzarray 1    | |  (列1, 例如 'feature1')
| |-------------------| |
| |     Fuzznum a     | |
| |     Fuzznum b     | |
| |       ...         | |
| +-------------------+ |
+-----------------------+
| +-------------------+ |
| |    Fuzzarray 2    | |  (列2, 例如 'feature2')
| |-------------------| |
| |     Fuzznum x     | |
| |     Fuzznum y     | |
| |       ...         | |
| +-------------------+ |
+-----------------------+
```

理解这个层级关系有助于你更好地组织数据和设计分析流程。

## 总结与展望

`FuzzyDataFrame` 和 `.fuzzy` 访问器是 `axisfuzzy-analysis` 模块的核心，它们共同实现了与 Pandas 生态的无缝集成，提供了一个既强大又符合直觉的用户接口。

虽然目前 `FuzzyDataFrame` 的方法还相对基础，但其可扩展的架构为未来功能的丰富奠定了坚实的基础。随着 `axisfuzzy` 的发展，我们将继续为 `FuzzyDataFrame` 增加更多数据操作和分析方法，使其成为一个功能更加完备的模糊数据分析工具。