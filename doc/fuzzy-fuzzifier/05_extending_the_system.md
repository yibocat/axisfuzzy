# 扩展系统：自定义隶属函数与模糊化策略

`axisfuzzy.fuzzifier` 模块的核心设计理念之一就是**可扩展性**。无论你是研究人员需要实现最新的模糊逻辑算法，还是工程师希望为特定应用场景定制模糊化行为，该系统都为你提供了清晰、简洁的扩展路径。

本章将详细介绍如何扩展系统的两个核心组件：
1. **自定义隶属函数** - 实现新的数学形状和特性
2. **自定义模糊化策略** - 定义新的模糊数类型和转换算法

通过这些扩展机制，你可以将 AxisFuzzy 打造成完全符合你需求的专业工具。

---

## 1. 扩展隶属函数：构建新的数学形状

隶属函数是模糊逻辑的基础构建块。虽然 `axisfuzzy.membership` 已经提供了丰富的内置函数（如高斯、三角形、梯形等），但在某些专业领域或研究场景中，你可能需要实现全新的数学形状。

### 1.1 隶属函数的基本结构

所有隶属函数都必须继承自 `MembershipFunction` 基类，并实现以下核心方法：

```python
from axisfuzzy.membership import MembershipFunction
import numpy as np
from typing import Union, Dict, Any

class CustomMembershipFunction(MembershipFunction):
    """自定义隶属函数示例"""
    
    def __init__(self, param1: float, param2: float, **kwargs):
        """
        初始化隶属函数参数
        
        Parameters
        ----------
        param1 : float
            第一个参数的描述
        param2 : float
            第二个参数的描述
        **kwargs : dict
            其他可选参数
        """
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
        
        # 参数验证
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """验证参数的有效性"""
        if self.param1 <= 0:
            raise ValueError("param1 必须大于 0")
        if self.param2 <= 0:
            raise ValueError("param2 必须大于 0")
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        计算隶属度值
        
        Parameters
        ----------
        x : Union[float, np.ndarray]
            输入值或值数组
            
        Returns
        -------
        Union[float, np.ndarray]
            对应的隶属度值，范围 [0, 1]
        """
        # 确保输入是 numpy 数组
        x = np.asarray(x)
        
        # 实现你的数学公式
        # 这里是一个示例：修改的高斯函数
        result = np.exp(-((x - self.param1) / self.param2) ** 4)
        
        # 确保结果在 [0, 1] 范围内
        return np.clip(result, 0, 1)
    
    def get_parameters(self) -> Dict[str, Any]:
        """返回函数的所有参数"""
        return {
            'param1': self.param1,
            'param2': self.param2
        }
    
    def set_parameters(self, **kwargs) -> None:
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._validate_parameters()
    
    def __repr__(self) -> str:
        return f"CustomMembershipFunction(param1={self.param1}, param2={self.param2})"
```

### 1.2 高级隶属函数示例：双峰高斯函数

让我们实现一个更复杂的例子 - 双峰高斯函数，它在某些模式识别和信号处理应用中很有用：

```python
class BimodalGaussianMF(MembershipFunction):
    """双峰高斯隶属函数
    
    该函数由两个高斯峰组成，可以建模具有两个主要特征值的模糊概念。
    """
    
    def __init__(self, c1: float, c2: float, sigma1: float, sigma2: float, 
                 weight1: float = 0.5, **kwargs):
        """
        Parameters
        ----------
        c1, c2 : float
            两个峰的中心位置
        sigma1, sigma2 : float
            两个峰的标准差
        weight1 : float, default=0.5
            第一个峰的权重，第二个峰权重为 (1 - weight1)
        """
        super().__init__(**kwargs)
        self.c1 = c1
        self.c2 = c2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.weight1 = weight1
        self.weight2 = 1 - weight1
        
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        if self.sigma1 <= 0 or self.sigma2 <= 0:
            raise ValueError("标准差必须大于 0")
        if not 0 <= self.weight1 <= 1:
            raise ValueError("权重必须在 [0, 1] 范围内")
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x)
        
        # 计算两个高斯分量
        gauss1 = np.exp(-0.5 * ((x - self.c1) / self.sigma1) ** 2)
        gauss2 = np.exp(-0.5 * ((x - self.c2) / self.sigma2) ** 2)
        
        # 加权组合
        result = self.weight1 * gauss1 + self.weight2 * gauss2
        
        return np.clip(result, 0, 1)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'c1': self.c1, 'c2': self.c2,
            'sigma1': self.sigma1, 'sigma2': self.sigma2,
            'weight1': self.weight1
        }
    
    def set_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if 'weight1' in kwargs:
            self.weight2 = 1 - self.weight1
        self._validate_parameters()
```

### 1.3 使用自定义隶属函数

一旦定义了自定义隶属函数，你就可以在 `Fuzzifier` 中直接使用它：

```python
from axisfuzzy.fuzzifier import Fuzzifier

# 使用自定义隶属函数
fuzzifier = Fuzzifier(
    mf=BimodalGaussianMF,  # 直接传递类
    mtype='qrofn',
    mf_params={
        'c1': 0.3, 'c2': 0.7,
        'sigma1': 0.1, 'sigma2': 0.15,
        'weight1': 0.6
    },
    q=2
)

# 测试模糊化
test_values = [0.2, 0.3, 0.5, 0.7, 0.8]
for val in test_values:
    fuzzy_result = fuzzifier(val)
    print(f"输入 {val} -> {fuzzy_result}")

# 可视化隶属函数
fuzzifier.plot(x_range=(0, 1), num_points=1000)
```

---

## 2. 扩展模糊化策略：定义新的转换算法

模糊化策略定义了如何将隶属度值转换为特定类型的模糊数。虽然系统已经支持多种模糊数类型（如 q-rung 正交模糊数、直觉模糊数等），但你可能需要实现全新的模糊数类型或改进现有的转换算法。

### 2.1 策略的基本结构

所有模糊化策略都必须继承自 `FuzzificationStrategy` 基类：

```python
from axisfuzzy.fuzzifier.strategy import FuzzificationStrategy
from axisfuzzy.fuzzifier.registry import register_fuzzifier
from axisfuzzy.core import Fuzznum, Fuzzarray
import numpy as np
from typing import Union, List, Dict, Any

@register_fuzzifier(is_default=False)  # 注册到系统中
class CustomStrategy(FuzzificationStrategy):
    """自定义模糊化策略示例"""
    
    # 必须定义的类属性
    mtype = 'custom_fuzzy_type'  # 目标模糊数类型
    method = 'advanced'          # 策略方法名
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, **kwargs):
        """
        初始化策略参数
        
        Parameters
        ----------
        alpha : float, default=0.5
            控制参数 α
        beta : float, default=0.3
            控制参数 β
        **kwargs : dict
            其他参数（如 q 值等）
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """验证参数有效性"""
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha 必须在 [0, 1] 范围内")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta 必须在 [0, 1] 范围内")
    
    def fuzzify(self, 
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
        """
        执行模糊化转换
        
        Parameters
        ----------
        x : Union[float, int, list, np.ndarray]
            输入的精确值
        mf_cls : type
            隶属函数类
        mf_params_list : List[Dict]
            隶属函数参数列表
            
        Returns
        -------
        Union[Fuzznum, Fuzzarray]
            模糊化结果
        """
        # 转换输入为 numpy 数组
        x = np.asarray(x)
        is_scalar = x.ndim == 0
        if is_scalar:
            x = x.reshape(1)
        
        # 创建隶属函数实例
        membership_functions = []
        for params in mf_params_list:
            mf_instance = mf_cls(**params)
            membership_functions.append(mf_instance)
        
        # 计算隶属度
        memberships = []
        for mf in membership_functions:
            mu = mf(x)
            memberships.append(mu)
        
        # 实现你的转换算法
        results = []
        for i in range(len(x)):
            # 获取当前点的所有隶属度
            mu_values = [mem[i] for mem in memberships]
            
            # 实现自定义的模糊数生成逻辑
            # 这里是一个示例：基于多个隶属度生成复合模糊数
            primary_mu = np.mean(mu_values)  # 主隶属度
            
            # 使用自定义公式计算非隶属度
            nu = self._calculate_non_membership(primary_mu, mu_values)
            
            # 创建模糊数
            fuzzy_num = Fuzznum(
                membership=primary_mu,
                non_membership=nu,
                mtype=self.mtype
            )
            results.append(fuzzy_num)
        
        # 返回适当的结果类型
        if is_scalar:
            return results[0]
        else:
            return Fuzzarray(results)
    
    def _calculate_non_membership(self, primary_mu: float, 
                                 all_mu: List[float]) -> float:
        """
        自定义的非隶属度计算方法
        
        Parameters
        ----------
        primary_mu : float
            主隶属度值
        all_mu : List[float]
            所有隶属度值
            
        Returns
        -------
        float
            计算得到的非隶属度
        """
        # 示例算法：基于隶属度的变异性计算非隶属度
        variance = np.var(all_mu) if len(all_mu) > 1 else 0
        
        # 结合 alpha, beta 参数的自定义公式
        nu = self.alpha * (1 - primary_mu) + self.beta * variance
        
        # 确保结果在有效范围内
        return np.clip(nu, 0, 1 - primary_mu)
```

### 2.2 高级策略示例：区间值模糊数策略

让我们实现一个更复杂的策略，用于生成区间值模糊数：

```python
@register_fuzzifier(is_default=True)  # 设为默认策略
class IntervalValuedFuzzyStrategy(FuzzificationStrategy):
    """区间值模糊数策略
    
    该策略生成具有区间隶属度的模糊数，适用于处理不确定性更高的场景。
    """
    
    mtype = 'interval_fuzzy'
    method = 'default'
    
    def __init__(self, uncertainty_factor: float = 0.1, 
                 aggregation_method: str = 'mean', **kwargs):
        """
        Parameters
        ----------
        uncertainty_factor : float, default=0.1
            不确定性因子，控制区间的宽度
        aggregation_method : str, default='mean'
            多隶属函数的聚合方法：'mean', 'max', 'min'
        """
        super().__init__(**kwargs)
        self.uncertainty_factor = uncertainty_factor
        self.aggregation_method = aggregation_method
        
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        if not 0 <= self.uncertainty_factor <= 0.5:
            raise ValueError("uncertainty_factor 必须在 [0, 0.5] 范围内")
        if self.aggregation_method not in ['mean', 'max', 'min']:
            raise ValueError("aggregation_method 必须是 'mean', 'max', 或 'min'")
    
    def fuzzify(self, x, mf_cls, mf_params_list):
        x = np.asarray(x)
        is_scalar = x.ndim == 0
        if is_scalar:
            x = x.reshape(1)
        
        # 创建隶属函数
        membership_functions = [mf_cls(**params) for params in mf_params_list]
        
        results = []
        for i in range(len(x)):
            # 计算所有隶属度
            mu_values = [mf(x[i]) for mf in membership_functions]
            
            # 聚合隶属度
            if self.aggregation_method == 'mean':
                central_mu = np.mean(mu_values)
            elif self.aggregation_method == 'max':
                central_mu = np.max(mu_values)
            else:  # 'min'
                central_mu = np.min(mu_values)
            
            # 计算区间边界
            interval_width = self.uncertainty_factor * central_mu
            mu_lower = max(0, central_mu - interval_width)
            mu_upper = min(1, central_mu + interval_width)
            
            # 创建区间值模糊数
            # 注意：这里假设 Fuzznum 支持区间值，实际实现可能需要扩展
            fuzzy_num = Fuzznum(
                membership=(mu_lower, mu_upper),  # 区间隶属度
                mtype=self.mtype
            )
            results.append(fuzzy_num)
        
        return results[0] if is_scalar else Fuzzarray(results)
```

### 2.3 策略的注册与使用

使用 `@register_fuzzifier` 装饰器注册的策略会自动添加到系统中：

```python
# 查看可用的策略
from axisfuzzy.fuzzifier.registry import get_registry_fuzzify

registry = get_registry_fuzzify()
print("可用的模糊数类型:", registry.get_available_mtypes())
print("区间模糊数的方法:", registry.get_available_methods('interval_fuzzy'))

# 使用自定义策略
fuzzifier = Fuzzifier(
    mf='GaussianMF',
    mtype='interval_fuzzy',  # 使用我们定义的新类型
    method='default',        # 使用默认方法
    mf_params={'c': 0.5, 'sigma': 0.2},
    uncertainty_factor=0.15,
    aggregation_method='mean'
)

# 测试
result = fuzzifier(0.6)
print(f"区间值模糊化结果: {result}")
```

---

## 3. 最佳实践与设计指南

### 3.1 隶属函数设计原则

1. **数学严谨性**: 确保函数在定义域内连续且值域在 [0, 1]
2. **参数验证**: 在 `__init__` 和 `set_parameters` 中进行严格的参数检查
3. **性能优化**: 使用 NumPy 向量化操作，避免 Python 循环
4. **文档完整**: 提供清晰的数学描述和使用示例

### 3.2 策略设计原则

1. **单一职责**: 每个策略专注于一种特定的模糊数类型和转换方法
2. **参数灵活性**: 通过构造函数参数提供算法的可调节性
3. **错误处理**: 对无效输入提供清晰的错误信息
4. **向后兼容**: 新策略不应破坏现有的 API

### 3.3 测试与验证

```python
# 为自定义组件编写测试
import unittest
import numpy as np

class TestCustomMembershipFunction(unittest.TestCase):
    
    def setUp(self):
        self.mf = BimodalGaussianMF(c1=0.3, c2=0.7, sigma1=0.1, sigma2=0.1)
    
    def test_output_range(self):
        """测试输出值在 [0, 1] 范围内"""
        x = np.linspace(0, 1, 100)
        y = self.mf(x)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y <= 1))
    
    def test_peak_positions(self):
        """测试峰值位置"""
        # 在峰值位置应该有较高的隶属度
        self.assertGreater(self.mf(0.3), 0.8)
        self.assertGreater(self.mf(0.7), 0.8)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        with self.assertRaises(ValueError):
            BimodalGaussianMF(c1=0.5, c2=0.5, sigma1=-0.1, sigma2=0.1)

if __name__ == '__main__':
    unittest.main()
```

### 3.4 性能优化建议

1. **向量化计算**: 优先使用 NumPy 的向量化操作
2. **内存管理**: 对于大数据集，考虑分批处理
3. **缓存机制**: 对于计算密集的操作，实现适当的缓存
4. **并行处理**: 利用 `joblib` 或 `multiprocessing` 进行并行计算

---

## 4. 贡献到 AxisFuzzy 项目

如果你开发了有价值的隶属函数或策略，欢迎贡献到 AxisFuzzy 项目：

### 4.1 贡献流程

1. **Fork 项目**: 在 GitHub 上 fork AxisFuzzy 仓库
2. **创建分支**: 为你的功能创建一个新分支
3. **实现功能**: 按照本章的指南实现你的扩展
4. **编写测试**: 为新功能编写完整的单元测试
5. **更新文档**: 更新相关的文档和示例
6. **提交 PR**: 创建 Pull Request 并描述你的贡献

### 4.2 代码规范

- 遵循 PEP 8 Python 代码风格
- 使用 NumPy 风格的文档字符串
- 提供类型注解
- 包含完整的错误处理

### 4.3 文档要求

- 数学公式的 LaTeX 表示
- 详细的参数说明
- 实际应用示例
- 性能特征说明

---

## 总结

通过本章的学习，你已经掌握了扩展 AxisFuzzy 系统的核心技能：

1. **自定义隶属函数**: 实现新的数学形状和特性
2. **自定义模糊化策略**: 定义新的模糊数类型和转换算法
3. **最佳实践**: 遵循设计原则，确保代码质量
4. **测试验证**: 编写可靠的测试用例

这些扩展能力使 AxisFuzzy 成为一个真正开放和灵活的平台，能够适应各种研究和应用需求。无论是学术研究还是工业应用，你都可以通过这些机制将系统定制为完全符合你需求的专业工具。

记住，好的扩展不仅要功能强大，还要易于使用、文档完善、测试充分。这样才能真正为 AxisFuzzy 生态系统增加价值，并帮助其他用户解决他们的问题。