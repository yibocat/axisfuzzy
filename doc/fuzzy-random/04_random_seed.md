# 随机数种子 (`seed.py`)

## 1 为什么可复现性如此重要

在科学计算和机器学习中，**可复现性（Reproducibility）** 是一个基本要求。想象以下场景：
- **算法验证**：你发现了一个很好的模糊数处理算法，需要与同事分享结果，或者在论文中报告性能指标。
- **调试与测试**：你的程序出现了 bug，但每次运行结果都不同，很难定位问题。
- **实验对比**：你想比较两种不同的方法，但如果输入数据每次都变化，比较就失去了意义。
如果随机数生成是**不可控**的，上述场景都会变得非常困难。`seed.py` 模块就是为了解决这个问题而设计的。

## 2 核心思想

`seed.py` 采用了 **全局单例模式** 的 `GlobalRandomState` 类，来管理整个 `AxisFuzzy` 的随机状态：
- **单一数据源**：全库所有随机生成器都使用同一个全局 RNG（`numpy.random.Generator`）。
- **线程安全**：内部使用锁（`threading.Lock`）保证多线程访问时状态一致。
- **可控性**：可以随时设置种子、获取当前 RNG、生成独立 RNG。

## 3 `GlobalRandomState`：随机状态的守护者

`GlobalRandomState` 是一个**单例类**，它在整个程序生命周期中只有一个实例，负责管理全局的随机数生成器（RNG）。
### 它的职责包括：
- **种子管理**：记住当前设置的种子值，支持查询和重设。
- **生成器提供**：向其他模块提供 NumPy 的 `Generator` 实例。
- **独立流创建**：能够生成统计独立的子生成器，用于并行计算。
- **线程安全**：使用锁机制保护内部状态，防止多线程访问冲突。
### 为什么使用单例模式？
如果允许多个 `GlobalRandomState` 实例存在，就可能出现以下问题：
- 不同实例使用不同的种子，导致"全局"一致性被破坏。
- 状态管理变得复杂，难以追踪当前的随机状态。

## 4 四个核心函数

`seed.py` 对外提供了四个简洁的函数，它们是用户与随机状态管理器交互的主要接口：

### 1. `set_seed(seed)` - 设置全局种子
这是最常用的函数，用于设置全局随机种子，确保后续所有随机操作的可复现性。

```python
import axisfuzzy.random as fr

# 设置全局种子
fr.set_seed(42)

# 现在所有的随机生成都是可复现的
num1 = fr.rand('qrofn', q=2)
arr1 = fr.rand('qrofn', shape=(100,), q=3)

# 重新设置相同种子，会得到相同结果
fr.set_seed(42)
num2 = fr.rand('qrofn', q=2)  # num2 和 num1 完全相同
arr2 = fr.rand('qrofn', shape=(100,), q=3)  # arr2 和 arr1 完全相同
```

### 2. `get_rng()` - 获取全局生成器
当你需要进行一些自定义的随机操作时，可以直接获取全局的 NumPy `Generator` 实例。

```python
import axisfuzzy.random as fr
import numpy as np

fr.set_seed(123)

# 获取全局生成器
rng = fr.get_rng()

# 用它进行自定义随机操作
custom_values = rng.uniform(0, 1, size=50)
noise = rng.normal(0, 0.1, size=100)

# 注意：这些操作会推进全局随机状态
# 后续的 fr.rand() 调用会使用推进后的状态
fuzzy_num = fr.rand('qrofn', q=2)  # 使用推进后的随机状态
```

### 3. `spawn_rng()` - 创建独立生成器
这是一个非常强大的功能，特别适用于**并行计算**或需要**隔离随机流**的场景。

```python
import axisfuzzy.random as fr
from concurrent.futures import ThreadPoolExecutor

fr.set_seed(456)

def worker_function(worker_id):
    # 每个工作线程获得独立的随机生成器
    independent_rng = fr.spawn_rng()
  
    # 使用独立生成器，不会影响全局状态
    worker_data = independent_rng.uniform(0, 1, size=100)
  
    return f"Worker {worker_id} generated {len(worker_data)} samples"

# 并行执行，每个线程的随机性互相独立
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(worker_function, range(4)))

# 全局状态保持未受影响
global_num = fr.rand('qrofn', q=2)  # 依然基于种子 456
```

### 4. `get_seed()` - 查询当前种子
用于调试、日志记录或实验追踪。

```python
import axisfuzzy.random as fr

fr.set_seed(789)
current_seed = fr.get_seed()
print(f"当前种子: {current_seed}")  # 输出: 当前种子: 789

# 在实验日志中记录种子，便于后续复现
print(f"实验开始，种子={fr.get_seed()}")
results = fr.rand('qrofn', shape=(1000,), q=2)
print(f"实验完成，使用种子 {fr.get_seed()} 生成了 {len(results)} 个样本")
```

## 5 实际应用场景

### 场景 1：科学实验的可复现性
```python
import axisfuzzy.random as fr

# 实验配置
EXPERIMENT_SEED = 2023
SAMPLE_SIZE = 10000

def run_experiment():
    # 设置实验种子
    fr.set_seed(EXPERIMENT_SEED)
    print(f"实验开始，种子: {fr.get_seed()}")
  
    # 生成测试数据
    data = fr.rand('qrofn', shape=(SAMPLE_SIZE,), q=3, md_dist='beta', a=2.0, b=5.0)
  
    # 进行某种分析...
    # result = analyze(data)
  
    print(f"实验完成，数据规模: {data.shape}")
    return data

# 任何人运行这个函数都会得到相同的结果
reproducible_data = run_experiment()
```

### 场景 2：多次实验的独立性
```python
import axisfuzzy.random as fr

fr.set_seed(100)  # 设置主种子

def monte_carlo_trial():
    # 每次试验使用独立的随机流
    trial_rng = fr.spawn_rng()
  
    # 生成本次试验的随机场景
    scenario = trial_rng.uniform(0, 1, size=50)
    # 模拟某个过程...
    result = scenario.mean()  # 简化的示例
    return result

# 运行 100 次独立的蒙特卡洛试验
trials = [monte_carlo_trial() for _ in range(100)]

import numpy as np
print(f"蒙特卡洛估计: {np.mean(trials):.4f} ± {np.std(trials):.4f}")
```

### 场景 3：调试中的状态固定
```python
import axisfuzzy.random as fr

def debug_function():
    # 在调试时固定随机状态
    fr.set_seed(999)
  
    # 现在每次调用此函数都会产生相同的"随机"数据
    problematic_data = fr.rand('qrofn', shape=(10,), q=2)
    print("调试数据:", problematic_data[0])
  
    # 你可以专注于算法逻辑，而不用担心数据变化
    return problematic_data

# 每次调用都会得到相同结果，便于调试
debug_function()
debug_function()  # 完全相同的输出
```

## 6 线程安全性保障

`GlobalRandomState` 内部使用了 `threading.Lock()` 来保护其状态，这意味着：

- **多线程读取**：多个线程可以安全地调用 `get_rng()` 而不会产生竞态条件。
- **状态修改保护**：`set_seed()` 的调用会被适当地同步，确保状态的一致性。
- **独立流创建**：`spawn_rng()` 可以安全地在多线程环境中调用，每个线程都会得到真正独立的随机流。
