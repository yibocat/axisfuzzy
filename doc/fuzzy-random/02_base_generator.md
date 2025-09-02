# 基本生成器 (`base.py`)

## 1. 生成器在随机化系统中的角色

在随机化系统中，**生成器**是核心组件，负责将随机数转化为具体的模糊数。但是不同类型的模糊数（如 QROFN、QROHFN）有不同的约束条件和参数结构，如何统一管理这些差异？
`axisfuzzy` 的解决方案是定义一个**抽象基类** `BaseRandomGenerator`，它规定了所有生成器必须遵循的"合约". 每种模糊数类型（`mtype`）都有一个对应的生成器类，它实现了该类型的随机生成逻辑. 此外, 还有一个生成器 `ParameterizedRandomGenerator` 继承自 `BaseRandomGenerator`，提供了一系列实用工具，简化了基于统计分布的生成器开发。
## 2. `BaseRandomGenerator` —— 基础契约

`BaseRandomGenerator` 是一个抽象基类（ABC），它本身不能被实例化。它的作用是规定一个随机生成器“应该长什么样”。

```python
class BaseRandomGenerator(ABC):
    mtype: str = 'unknown'  # 每个生成器必须声明自己处理的模糊数类型
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        # 返回默认参数配置
        
    @abstractmethod
    def validate_parameters(self, **params) -> None:
        # 验证参数的有效性
        
    @abstractmethod
    def fuzznum(self, 
			    rng: np.random.Generator, 
			    **params) -> Fuzznum:
        # 生成单个模糊数
        
    @abstractmethod
    def fuzzarray(self, 
			    rng: np.random.Generator,
			    shape: Tuple[int, ...], 
			    **params) -> Fuzzarray:
        # 生成模糊数数组（高性能批量生成）
```

为什么要分 `fuzznum` 和 `fuzzarray`？
- **`fuzznum`**：生成单个模糊数，逻辑简单，便于理解和调试。
- **`fuzzarray`**：批量生成，使用向量化操作，避免 Python 循环，性能提升数百倍。
这种设计让每个生成器可以针对**单元素**和**批量**场景分别优化，兼顾易用性和性能。

任何想要接入 `AxisFuzzy` 随机化系统的生成器，都必须继承它并实现以下四个核心部分：

1. **`mtype` 属性**
	 **作用**：一个字符串，用于声明该生成器服务于哪种模糊数类型（如 `'qrofn'`）。注册表会根据这个值来索引生成器。 **示例**：`mtype = "qrofn"`
2. **`get_default_parameters()` 方法**
	 **作用**：返回一个字典，包含生成随机数时的所有默认参数。当用户不指定某些参数时，系统会使用这些默认值。
3. `validate_parameters(**params)` 方法
	 **作用**：在生成随机数之前，验证用户传入的参数是否合法。例如，`q` 必须是正整数，`md_low` 不能大于 `md_high` 等。如果参数无效，应抛出 `ValueError` 或 `TypeError`。
4. `fuzznum(rng, **params)` 和 `fuzzarray(rng, shape, **params)` 方法
	 **`fuzznum`**：生成**单个** `Fuzznum` 对象。
	 **`fuzzarray`**：生成一个指定 `shape` 的 `Fuzzarray` **数组**。
	 **关键点**：`fuzzarray` 的实现应尽可能**向量化**，直接操作 NumPy 数组来填充后端（Backend），以获得高性能，避免在 Python 层面使用循环来逐个创建 `Fuzznum`。

## 3. `ParameterizedRandomGenerator` —— 参数化工具类

虽然 `BaseRandomGenerator` 定义了基本接口，但实际实现时会遇到一些**共性问题**：

- 参数合并：用户提供的参数 + 默认参数
- 分布采样：从 uniform、beta、normal 等分布中采样
- 参数验证：范围检查、类型检查

为了避免每个生成器都重复实现这些逻辑，`axisfuzzy` 提供了 `ParameterizedRandomGenerator`，它继承自 `BaseRandomGenerator` 并增加了**实用工具**：

```python
class ParameterizedRandomGenerator(BaseRandomGenerator, ABC):
    def __init__(self):
        self._default_params = self.get_default_parameters()  # 缓存默认参数
    
    def _merge_parameters(self, **params) -> Dict[str, Any]:
        # 合并用户参数和默认参数
        
    def _validate_range(self, 
					    name: str,
					    value: float, 
						min_val: float,
					    max_val: float):
        # 范围验证工具
        
    def _sample_from_distribution(self, 
		rng, 
		size=None, 
		dist='uniform', 
		low=0.0, 
		high=1.0, 
		**dist_params):
        # 统一的分布采样接口
```

它继承自 `BaseRandomGenerator`，并额外提供了三个非常有用的工具方法：

1. `_merge_parameters(**params)`
    - **作用**：自动将用户传入的参数与 `get_default_parameters()` 中的默认参数合并。用户传入的参数会覆盖默认值。
    - **用法**：在 `fuzznum` 或 `fuzzarray` 的开头调用它，可以轻松获取完整的参数配置。
2. **`_validate_range(name, value, min_val, max_val)`**
    - **作用**：一个便捷的验证函数，用于检查某个参数值是否在指定的闭区间 `[min_val, max_val]` 内。
3. **`_sample_from_distribution(...)`**
    - **作用**：这是**最核心的工具**。它提供了一个统一的接口，可以从以下三种分布中采样，并自动将结果缩放到指定的 `[low, high]` 范围内：
        - `'uniform'`：均匀分布。
        - `'beta'`：Beta 分布。需要额外的 `a` 和 `b` 参数。
        - `'normal'`：正态分布。采样后会裁剪（clip）到 `[low, high]` 范围内，以防止越界。需要额外的 `loc` (均值) 和 `scale` (标准差) 参数。
    - **优势**：开发者无需关心具体分布的实现细节，只需调用此方法即可获得符合要求的随机样本。

## 4. 如何选择继承哪个基类？

|特性|BaseRandomGenerator|ParameterizedRandomGenerator|
|---|---|---|
|是否抽象|是|是（继承自 Base）|
|是否提供采样工具|否|是（内置分布采样、参数合并等）|
|适用场景|需要完全自定义生成逻辑|逻辑符合“参数化分布采样”模式|
|开发效率|低（需自己实现所有逻辑）|高（可直接调用工具方法）|
**建议**：
- 如果你的生成逻辑是**特殊的、不依赖常规分布**，用 `BaseRandomGenerator`。
- 如果只是调整分布类型、范围等参数，直接继承 `ParameterizedRandomGenerator`。

## 5. 一个简单的生成器示例

让我们通过一个具体例子来理解如何实现生成器。假设我们要为一个假想的 "SimpleFS"（简单模糊集）实现随机生成器：

```python
from axisfuzzy.random.base import ParameterizedRandomGenerator
from axisfuzzy.random.registry import register_random

@register_random  # 自动注册到全局注册表
class SimpleFSRandomGenerator(ParameterizedRandomGenerator):
    mtype = "simplefs"  # 声明处理的模糊数类型
    
    def get_default_parameters(self):
        """返回默认参数配置"""
        return {
            'md_dist': 'uniform',    # 隶属度分布类型
            'md_low': 0.0,          # 最大隶属度
            'md_high': 1.0,         # 最小隶属度
            'a': 2.0,               # Beta分布参数
            'b': 2.0                # Beta分布参数
        }
    
    def validate_parameters(self, **params):
        """验证参数有效性"""
        # 使用继承的工具方法进行范围验证
        if 'md_low' in params:
            self._validate_range('md_low', params['md_low'], 0.0, 1.0)
        if 'md_high' in params:
            self._validate_range('md_high', params['md_high'], 0.0, 1.0)
        
        # 自定义逻辑验证
        if 'md_low' in params and 'md_high' in params:
            if params['md_low'] > params['md_high']:
                raise ValueError("md_low 不能大于 md_high")
    
    def fuzznum(self, rng, **params):
        """生成单个模糊数"""
        # 1. 合并参数（用户参数 + 默认参数）
        merged_params = self._merge_parameters(**params)
        
        # 2. 验证参数
        self.validate_parameters(**merged_params)
        
        # 3. 使用工具方法采样
        md = self._sample_from_distribution(
            rng,
            dist=merged_params['md_dist'],
            low=merged_params['md_low'],
            high=merged_params['md_high'],
            a=merged_params['a'],
            b=merged_params['b']
        )
        
        # 4. 构造并返回 Fuzznum
        return Fuzznum(mtype='simplefs').create(md=md)
    
    def fuzzarray(self, rng, shape, **params):
        """高性能批量生成"""
        merged_params = self._merge_parameters(**params)
        self.validate_parameters(**merged_params)
        
        # 计算总元素数
        size = int(np.prod(shape))
        
        # 向量化采样（关键：一次生成所有元素）
        mds = self._sample_from_distribution(
            rng,
            size=size,  # 批量生成
            dist=merged_params['md_dist'],
            low=merged_params['md_low'],
            high=merged_params['md_high'],
            a=merged_params['a'],
            b=merged_params['b']
        )
        
        # 重塑为目标形状
        mds = mds.reshape(shape)
        
        # 构造后端并返回 Fuzzarray
        backend = SimpleFSBackend.from_arrays(mds=mds)
        return Fuzzarray(backend=backend)
```

这样，`"mytype"` 就可以直接通过：
```python
import axisfuzzy.random as fr
fr.rand('mytype', shape=(10,), md_low=0.2, md_high=0.8)
```
来生成随机模糊数了。

## 5 性能优化的关键思想

在 `fuzzarray` 实现中，有一个关键的性能优化思想：**"批量采样 + 后端直构"** [4]：

1. **批量采样**：一次调用 `_sample_from_distribution(size=1000)` 比调用 1000 次 `_sample_from_distribution()` 快几十倍。
2. **后端直构**：直接用 NumPy 数组构造 `FuzzarrayBackend`，而不是先创建 1000 个 `Fuzznum` 再合并。

这种设计让 AxisFuzzy 在生成大规模模糊数数组时具有出色的性能表现。

## 6 工具方法深入解析

`ParameterizedRandomGenerator` 提供的 `_sample_from_distribution` 方法特别值得关注：

```python
# 支持多种分布类型
uniform_vals = self._sample_from_distribution(rng, size=100, dist='uniform', low=0, high=1)
beta_vals = self._sample_from_distribution(rng, size=100, dist='beta', low=0, high=1, a=2.0, b=5.0)
normal_vals = self._sample_from_distribution(rng, size=100, dist='normal', low=0, high=1, loc=0.5, scale=0.15)
```

这个方法的巧妙之处在于：
- **统一接口**：无论什么分布，调用方式一致。
- **自动范围映射**：Beta 和 Normal 分布会自动映射到 `[low, high]` 区间。
- **向量化友好**：天然支持批量采样。
