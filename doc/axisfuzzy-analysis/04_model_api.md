# Model API：像搭积木一样构建分析流

## 1. 为什么需要 `Model` API？

想象一下，您正在处理一个复杂的数据分析任务。数据需要先清洗，然后进行两种不同的预处理，接着将预处理的结果合并，再进行模糊化，最后根据一组动态权重进行聚合。

如果用 `FuzzyPipeline` 手动构建，您需要仔细地管理每一步的输入和输出，确保依赖关系正确无误。当流程变得复杂时，代码很快就会变得难以阅读和维护。

`Model` API 的诞生就是为了解决这个问题。它的核心思想是：**让用户像写一个普通的 Python 类一样，以最直观的方式描述数据“流动”的过程，而将构建计算图（DAG）的复杂工作完全自动化。**

其设计灵感源自深度学习框架 PyTorch 的 `nn.Module`，如果您熟悉 PyTorch，您会感到非常亲切。

## 2. 您的第一个模型：从线性流程开始

让我们从一个最简单的例子开始：将输入数据进行归一化，然后进行模糊化。

### 2.1. 定义模型

创建一个 `Model` 非常简单，只需两步：
1.  继承 `axisfuzzy.analysis.Model`。
2.  在 `__init__` 方法中，定义您需要的分析组件（`AnalysisComponent`）。
3.  在 `forward` 方法中，用一种看起来像是“顺序调用”的方式，描述数据如何流经这些组件。

```python
from axisfuzzy.analysis import Model
from axisfuzzy.analysis.build_in import ToolNormalization, ToolFuzzification
from axisfuzzy.analysis.contracts import ContractCrispTable, ContractFuzzyTable

# 假设我们已经有了一个模糊化器实例
# from axisfuzzy.fuzzifier import Fuzzifier
# my_fuzzifier = Fuzzifier(...)

class SimpleAnalysisModel(Model):
    """一个简单的数据预处理模型。"""
    def __init__(self, fuzzifier):
        super().__init__()
        # 1. 定义需要的“积木块”（分析组件）
        self.normalizer = ToolNormalization(method='min_max')
        self.fuzzifier = ToolFuzzification(fuzzifier=fuzzifier)

    def get_config(self):
        # 2. 告诉系统如何保存和加载这个模型
        # 这里我们只需要保存 fuzzifier 的配置
        return {"fuzzifier": self.fuzzifier.get_config()}

    def forward(self, data: ContractCrispTable) -> ContractFuzzyTable:
        """3. 描述数据如何“流动”"""
        # 数据首先流经 normalizer
        normalized_data = self.normalizer(data)
        # 然后，归一化后的数据流经 fuzzifier
        fuzzy_data = self.fuzzifier(normalized_data)
        # 返回最终结果
        return fuzzy_data
```

**解读 `forward` 方法：**
`forward` 方法是 `Model` 的灵魂。您在这里写的代码，例如 `normalized_data = self.normalizer(data)`，**并不会立即执行计算**。相反，`axisfuzzy` 在背后进行了一种“魔法”——**符号追踪（Symbolic Tracing）**。

当您调用 `build()` 方法时，系统会“模拟”执行一次 `forward`，并记录下所有组件的调用顺序和依赖关系，然后将这个逻辑自动编译成一个高效的 `FuzzyPipeline` 计算图。

### 2.2. 构建并运行模型

模型定义好后，使用它就像调用一个普通函数一样简单。

```python
# 准备输入数据（例如一个 Pandas DataFrame）
import pandas as pd
input_data = pd.DataFrame(...) 

# 准备模糊化器
# my_fuzzifier = ...

# 1. 实例化模型
model = SimpleAnalysisModel(fuzzifier=my_fuzzifier)

# 2. 构建计算图（这是关键一步！）
model.build()

# 3. 运行模型
# 您可以直接像函数一样调用它
fuzzy_result = model(input_data) 

# 或者使用 run 方法，这完全等价
# fuzzy_result = model.run(input_data)
```

至此，您已经成功创建并运行了一个 `Model`。`Model` 将声明式的编程体验（您只需关心“做什么”）和命令式编程的灵活性（可以像普通 Python 代码一样编写逻辑）完美地结合在了一起。

## 3. 更进一步：处理非线性数据流

`Model` 真正的威力在于处理非线性流程，例如数据分支和合并。

假设我们的任务是：
1.  对原始数据 `data` 进行归一化。
2.  对另一组输入 `weights` 进行归一化。
3.  用归一化后的 `data` 生成模糊表。
4.  用模糊表和归一化后的 `weights` 进行加权平均。

这个流程包含一个明显的分支（`data` 和 `weights` 分别处理）和一个合并点（在聚合时合并）。

### 3.1. 定义非线性模型

在 `forward` 方法中，这种非线性逻辑可以被非常自然地表达出来：

```python
from axisfuzzy.analysis.build_in import ToolWeightNormalization, ToolSimpleAggregation

class WeightedAnalysisModel(Model):
    def __init__(self, fuzzifier):
        super().__init__()
        self.data_normalizer = ToolNormalization(method='min_max')
        self.weight_normalizer = ToolWeightNormalization()
        self.fuzzifier = ToolFuzzification(fuzzifier=fuzzifier)
        self.aggregator = ToolSimpleAggregation(operation='weighted_mean')

    def get_config(self):
        return {"fuzzifier": self.fuzzifier.get_config()}

    def forward(self, data: ContractCrispTable, weights: ContractWeightVector):
        # 第一个分支：处理数据
        norm_data = self.data_normalizer(data)
        fuzzy_data = self.fuzzifier(norm_data)

        # 第二个分支：处理权重
        norm_weights = self.weight_normalizer(weights)

        # 合并点：两个分支的结果在这里汇合
        final_score = self.aggregator(fuzzy_data, norm_weights)

        # Model 支持多输出，只需返回一个字典
        return {
            "final_score": final_score,
            "fuzzy_table": fuzzy_data,
        }
```

### 3.2. 可视化计算图

当您构建这个模型后，可以通过 `visualize()` 方法清晰地看到其内部的 DAG 结构，这对于理解和调试复杂模型非常有帮助。

```python
# 实例化并构建
weighted_model = WeightedAnalysisModel(fuzzifier=my_fuzzifier)
weighted_model.build()

# 可视化计算图（需要安装 graphviz）
# 这会生成一个图形文件，清晰地展示了数据分支和合并的流程
weighted_model.visualize(filename="weighted_model_dag", format="png")
```

![非线性计算图示例](https://.../nonlinear_dag.png)  <!-- 占位符：这里可以放一张图 -->

## 4. 终极武器：模型嵌套

随着分析逻辑越来越复杂，您可能会希望将一些通用的、可复用的步骤封装起来。`Model` 支持任意深度的嵌套，让您可以像搭乐高一样，将小模型组合成大模型。

### 4.1. 定义子模型和主模型

假设我们可以将“数据预处理（归一化+模糊化）”封装成一个独立的子模型。

```python
# 首先，定义一个可复用的预处理子模型
# 这和我们第一个例子中的 SimpleAnalysisModel 完全一样
class PreprocessingModel(Model):
    def __init__(self, fuzzifier):
        super().__init__()
        self.normalizer = ToolNormalization(method='min_max')
        self.fuzzifier = ToolFuzzification(fuzzifier=fuzzifier)

    def get_config(self):
        return {"fuzzifier": self.fuzzifier.get_config()}

    def forward(self, data: ContractCrispTable):
        norm_data = self.normalizer(data)
        fuzzy_data = self.fuzzifier(norm_data)
        return fuzzy_data

# 然后，在主模型中使用这个子模型
class MainModel(Model):
    def __init__(self, fuzzifier):
        super().__init__()
        # 将 PreprocessingModel 像一个普通组件一样实例化
        self.preprocessor = PreprocessingModel(fuzzifier)
        self.aggregator = ToolSimpleAggregation(operation='mean')

    def get_config(self):
        # 主模型也需要能被保存和加载
        return {"fuzzifier": self.preprocessor.fuzzifier.get_config()}

    def forward(self, data: ContractCrispTable):
        # 直接调用子模型
        processed_data = self.preprocessor(data)
        # 将子模型的结果送入下一个组件
        final_score = self.aggregator(processed_data)
        return final_score
```

### 4.2. 构建和运行嵌套模型

使用方式与之前完全相同。当您调用主模型的 `build()` 方法时，系统会自动地、递归地构建所有嵌套的子模型。

```python
main_model = MainModel(fuzzifier=my_fuzzifier)
main_model.build()  # 这一步会自动构建内部的 preprocessor
result = main_model(input_data)
```

通过模型嵌套，您可以构建出层次清晰、高度模块化且易于维护的复杂分析系统。

## 5. 调试与洞察：深入理解你的模型

`Model` 不仅仅是一个执行工具，它更是一个透明的分析框架。它提供了一套强大的内省（Introspection）工具，帮助您理解、调试和优化您的分析流。

### 5.1. `summary()`：一览无余的模型结构

在调用 `model.build()` 之后，模型内部的计算图就已经确定。`summary()` 方法可以将这个复杂的结构以清晰、分层的表格形式打印出来，让您对模型的全貌一目了然。

调用 `model.summary()` 会输出类似下方的内容：

```
Model: "MainAnalysisModel"
================================================================================
Layer (type)                    Input Contracts             Output Contracts
--------------------------------------------------------------------------------
Input: raw_input_data           -                           ContractCrispTable
DataPreprocessingModel          ContractCrispTable          ContractCrispTable
  └─ ToolNormalization          ContractCrispTable          ContractCrispTable
ToolSimpleAggregation           ContractCrispTable          ContractWeightVector
--------------------------------------------------------------------------------
Total layers: 2 (including 1 nested model(s) with 1 sub-layers)
================================================================================
```

**如何解读 `summary` 输出？**

-   **Layer (type)**: 显示了计算图中的一个节点。
    -   `Input: ...`: 代表模型的输入节点。
    -   `DataPreprocessingModel`: 这是一个嵌套的子模型。`summary` 会自动识别并将其作为一个独立的层级。
    -   `└─ ToolNormalization`: `└─` 符号表示这是 `DataPreprocessingModel` 内部的一个组件，清晰地展示了模型的层次结构。
    -   `ToolSimpleAggregation`: 这是一个顶层的普通分析组件。
-   **Input Contracts / Output Contracts**: 分别列出了每个节点期望的输入数据类型和产生的输出数据类型。这对于检查数据流中各环节的类型匹配至关重要，是 `axisfuzzy` 契约系统的核心体现。
-   **Total layers**: 对模型的复杂度进行总结，告诉您总共有多少个计算层，以及其中包含了多少嵌套模型和子层。

通过 `summary()`，您可以在运行实际数据之前，快速验证模型的结构是否符合预期，检查组件之间的连接是否正确，以及数据契约是否匹配。

### 5.2. `step_by_step()`：深入细节的单步调试

`summary()` 提供了宏观的结构视图，而 `step_by_step()` 则允许您深入微观的执行细节。

当您调用 `model.step_by_step(...)` 时，它并不会立即执行整个模型，而是返回一个 `FuzzyPipelineIterator` 迭代器。这个迭代器的设计与 `axisfuzzy.analysis.pipeline.py` 中的 `FuzzyPipeline` 紧密相关，它将底层的 `FuzzyPipeline` 的执行过程暴露出来，允许您逐个节点地运行计算图。

每次遍历这个迭代器，它会执行图中的一个步骤，并返回一个包含详细信息的字典。

```python
# 假设 main_model 已经 build
iterator = main_model.step_by_step(input_data)

print("--- Starting Step-by-Step Execution ---")
for step_info in iterator:
    print(f"✅ Executed Step: '{step_info['step_name']}'")
    print(f"   - Index: {step_info['step_index']} / {step_info['total_steps']}")
    print(f"   - Time: {step_info['execution_time(ms)']} ms")
    
    # 'result' 键包含了这一步的实际计算输出
    current_result = step_info['result']
    print(f"   - Result Type: {type(current_result)}")
    
    # 在这里，您可以进行任何调试操作：
    # 1. 检查中间结果的值
    # 2. 验证数据维度或类型
    # 3. 将中间结果保存到文件
    # 4. 在调试器中设置断点
    
    print("-" * 20)

# 迭代结束后，可以通过迭代器的 .result 属性获取最终结果
final_result = iterator.result
print("--- Execution Complete ---")
print(f"Final result: {final_result}")
```

**`step_info` 字典包含哪些内容？**

-   `step_name`: 当前执行步骤的名称，与 `summary()` 中的层名对应。
-   `step_index` / `total_steps`: 当前进度，方便您了解执行到了哪一步。
-   `result`: **最重要的部分**，这是当前步骤计算产生的实际输出数据。
-   `execution_time(ms)`: 执行该步骤所花费的时间，可用于简单的性能分析。

`step_by_step()` 是一个无价的调试工具。当模型行为不符合预期时，您不再需要盲目猜测，而是可以像侦探一样，沿着数据流动的路径，一步步检查每个环节的输入和输出，直到找到问题的根源。

### 5.3. `visualize()`：图形化的逻辑流

对于非常复杂的非线性模型，`summary()` 的表格可能不足以完全展现其逻辑。`visualize()` 方法可以将计算图渲染成一张图片，用节点和有向边直观地展示数据的完整流动路径，包括分支、合并和嵌套。这对于理解和向他人解释模型架构非常有帮助。

## 6. 保存与加载：让模型持久化

当您精心设计并验证了一个模型后，自然希望能够保存它以便将来复用。`save()` 和 `load()` 方法为此而生。

```python
# 保存模型架构
main_model.save("my_main_model.json")

# ... 在另一个脚本或未来的某个时间 ...

# 加载模型架构
from axisfuzzy.analysis import Model
loaded_model = Model.load("my_main_model.json")

# 重要：加载后的模型处于“未构建”状态
# 需要重新构建才能运行
loaded_model.build()
result = loaded_model(input_data)
```

**注意**：`save()` 方法只保存模型的**架构和配置**（即重建模型所需的所有信息），它**不保存**任何数据或计算状态。

## 7. 与 `FuzzyDataFrame` 的无缝集成

`Model` 的设计与 `FuzzyDataFrame` (通过 `.fuzzy` 访问器) 紧密集成，提供了极其流畅的使用体验。

```python
import pandas as pd
from axisfuzzy.analysis import FuzzyDataFrame

df = pd.DataFrame(...)
fz_df = FuzzyDataFrame(df)

# 直接在 FuzzyDataFrame 上运行模型
# `run` 方法会自动处理 build() 和执行
result = fz_df.fuzzy.run(main_model)

# 也可以进行单步调试
iterator = fz_df.fuzzy.step_by_step(main_model)
for step in iterator:
    print(step.output)
```

## 8. 总结

`Model` API 是 `axisfuzzy` 中用于构建复杂分析流程的终极抽象。它通过引入符号追踪和类 PyTorch 的设计模式，将声明式编程的简洁性与命令式编程的灵活性融为一体。

通过 `Model`，您可以：
- **直观地**定义从简单到复杂的任意数据流。
- 构建**模块化、可复用、可嵌套**的分析组件。
- 轻松地**调试、可视化和理解**您的分析逻辑。
- **持久化**您的模型架构，并在不同项目中复用。

掌握 `Model`，您就掌握了在 `axisfuzzy` 中构建工业级模糊分析系统的钥匙。