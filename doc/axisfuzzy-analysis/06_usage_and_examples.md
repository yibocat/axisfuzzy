# 6. 综合使用与端到端示例

本文档旨在通过一个或多个端到端的综合示例，展示如何运用 `axisfuzzy-analysis` 系统的各项功能来解决一个具体的数据分析问题。我们将从问题定义开始，一步步完成数据准备、模型构建、执行与结果解读的全过程。

## 核心理念回顾

在开始之前，让我们简要回顾一下 `axisfuzzy-analysis` 的核心设计理念：

- **声明式与命令式的结合**：通过高级 `Model` API，用户可以像定义 PyTorch 模型一样，以声明式的方式构建复杂的分析流程，而底层则由 `FuzzyPipeline` 负责命令式的执行。
- **契约驱动开发**：利用 `@contract` 装饰器和 `Contract` 类，确保数据在管道中的类型安全和一致性。
- **与 Pandas 无缝集成**：通过 `.fuzzy` 访问器，可以直接在 `pandas.DataFrame` 上执行模糊分析模型，极大地简化了工作流。
- **模块化与可扩展性**：所有分析步骤都被封装在 `AnalysisComponent` 中，易于复用和扩展。

---

## 示例一：基础多属性决策分析

假设我们需要对一组候选方案进行评估。每个方案有多个评价指标（属性），我们需要根据这些指标给出一个综合评分。

### 1. 问题定义

- **输入**：一个包含多个候选方案和其对应指标值的 `pandas.DataFrame`。
- **流程**：
    1. 对原始数据进行归一化处理。
    2. 将归一化的数据进行模糊化，转换成模糊数。
    3. 对每个方案的模糊化指标进行聚合，得到一个最终的模糊评分。
- **输出**：每个方案的综合评分。

### 2. 准备数据

首先，我们创建一个 `pandas.DataFrame` 作为输入。

```python
import pandas as pd
import numpy as np

# 创建一个 5x3 的数据框，代表 5 个方案和 3 个评价指标
df = pd.DataFrame(np.random.rand(5, 3), 
                  columns=['Performance', 'Cost', 'Reliability'],
                  index=[f'Option_{i+1}' for i in range(5)])

print("--- 原始数据 ---")
print(df)
```

### 3. 定义分析模型

我们使用 `Model` API 来定义整个分析流程。

```python
from axisfuzzy.analysis.app.model import Model
from axisfuzzy.analysis.component.basic import (
    ToolNormalization,
    ToolFuzzification,
    ToolSimpleAggregation
)
from axisfuzzy.analysis.build_in import ContractCrispTable, ContractFuzzyTable
from axisfuzzy.fuzzifier import Fuzzifier

# 1. 定义一个模糊化器
# 使用高斯隶属函数，输出 q-rung orthopair 模糊数
my_fuzzifier = Fuzzifier(
    mf='gaussmf',
    mtype='qrofn',
    pi=0.2,
    mf_params=[{'sigma': 0.15, 'c': 0.5}]
)

# 2. 定义分析模型
class BasicDecisionModel(Model):
    def __init__(self):
        super().__init__()
        # 定义模型需要的组件
        self.normalizer = ToolNormalization(method='min_max', axis=0)
        self.fuzzifier = ToolFuzzification(fuzzifier=my_fuzzifier)
        self.aggregator = ToolSimpleAggregation(operation='mean', axis=1)

    def forward(self, data: ContractCrispTable) -> ContractFuzzyTable:
        # 定义数据流
        norm_data = self.normalizer(data)
        fuzzy_data = self.fuzzifier(norm_data)
        scores = self.aggregator(fuzzy_data)
        return scores
```

### 4. 执行模型并解读结果

通过 `.fuzzy` 访问器，我们可以直接在 DataFrame 上运行模型。

```python
# 实例化模型
decision_model = BasicDecisionModel()

# 在 DataFrame 上运行模型
# .fuzzy 访问器会自动将 df 作为第一个参数传递给模型的 forward 方法
final_scores = df.fuzzy.run(decision_model)

print("\n--- 分析结果 ---")
print(final_scores)
```

`final_scores` 将是一个 `pandas.Series`，其索引是方案名称，值是每个方案的综合模糊评分（`Fuzznum` 对象）。

---

## 示例二：带权重的复杂分析模型

现在，我们考虑一个更复杂的场景，其中不仅有多个指标，还有对应的权重，并且我们希望在分析过程中能看到中间结果。

### 1. 问题定义

- **输入**：
    - `raw_data`: 一个包含指标值的 `pandas.DataFrame`。
    - `raw_weights`: 一个包含各指标权重的 `numpy.ndarray`。
- **流程**：
    1. 数据和权重分别进行归一化。
    2. 归一化后的数据进行模糊化。
    3. 使用归一化的权重对模糊数据进行加权聚合。
- **输出**：
    - `scores`: 最终的加权综合评分。
    - `fuzzy_table`: 中间的模糊化数据，用于调试或进一步分析。

### 2. 定义分析模型

这次，我们定义一个接受多个输入的 `Model`。

```python
from axisfuzzy.analysis.component.basic import ToolWeightNormalization
from axisfuzzy.analysis.build_in import ContractWeightVector, ContractNormalizedWeights

class WeightedDecisionModel(Model):
    def __init__(self):
        super().__init__()
        self.data_normalizer = ToolNormalization(method='min_max', axis=0)
        self.weight_normalizer = ToolWeightNormalization()
        self.fuzzifier = ToolFuzzification(fuzzifier=my_fuzzifier) # 复用之前的模糊化器
        # 假设我们有一个加权聚合组件
        self.weighted_aggregator = ToolSimpleAggregation(operation='mean', axis=1) # 简化演示

    def forward(self, data: ContractCrispTable, weights: ContractWeightVector):
        # 分支 1: 处理数据
        norm_data = self.data_normalizer(data)
        fuzzy_table = self.fuzzifier(norm_data)

        # 分支 2: 处理权重
        norm_weights = self.weight_normalizer(weights)

        # 聚合 (简化演示，实际应使用加权聚合)
        # 注意：此处的 norm_weights 未被使用，仅为演示多输入
        scores = self.weighted_aggregator(fuzzy_table)

        # 返回一个包含多个输出的字典
        return {'scores': scores, 'fuzzy_table': fuzzy_table}
```

### 3. 执行模型

当模型需要多个输入时，我们可以在 `.run()` 方法中通过关键字参数传入额外的数据。

```python
# 准备权重数据
weights = np.array([0.5, 0.3, 0.2])

# 实例化模型
weighted_model = WeightedDecisionModel()

# 执行模型，传入额外的 'weights' 参数
# return_intermediate=True 会返回所有中间步骤的结果
results, intermediate_results = df.fuzzy.run(
    weighted_model, 
    weights=weights, 
    return_intermediate=True
)

print("\n--- 最终输出 ---")
print(results)

print("\n--- 中间结果 ---")
# intermediate_results 是一个字典，包含了每一步的输出
for step_name, step_output in intermediate_results.items():
    print(f"\nStep: {step_name}")
    print(step_output)
```

### 4. 模型探索工具

`Model` API 还提供了便捷的工具来理解和调试模型。

- **`.summary()`**: 打印模型的层次结构、输入/输出契约。
- **`.visualize()`**: 生成模型的计算图。
- **`.step_by_step()`**: 返回一个迭代器，让你逐一执行模型中的每个组件。

```python
# 构建模型以生成计算图
weighted_model.build()

# 打印模型摘要
print("\n--- 模型摘要 ---")
weighted_model.summary()

# 可视化模型
# weighted_model.visualize() # 这将生成并显示一张图

# 逐步执行
print("\n--- 逐步执行 ---")
step_iterator = weighted_model.step_by_step(data=df, weights=weights)
for step_result in step_iterator:
    print(f"Executing step: {step_result['step_name']}")
    # print(step_result['result'])
```

## 总结与展望

通过以上示例，我们展示了 `axisfuzzy-analysis` 系统如何将数据处理、模糊化和决策分析等步骤流畅地整合到一个统一的框架中。其核心优势在于：

- **代码即模型**：分析逻辑以清晰、可读的 Python 代码形式存在，易于理解和维护。
- **高度的灵活性**：无论是简单的线性流程，还是复杂的多分支、多输入/输出模型，都能轻松应对。
- **强大的调试能力**：通过返回中间结果、模型摘要、可视化和逐步执行等功能，极大地简化了复杂模型的开发和调试过程。

`axisfuzzy-analysis` 为模糊数据科学提供了一个强大而灵活的工具集，未来将继续扩展内置组件库，并进一步优化性能，以支持更大规模的数据分析任务。