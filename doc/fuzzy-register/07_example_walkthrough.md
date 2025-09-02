# 示例演练：端到端集成新模糊类型

本章将通过一个完整的示例，引导您完成将一个全新的模糊类型集成到 `AxisFuzzy` 框架中的全过程。我们将以一个简化的 `q-rung orthopair fuzzy number (qrofn)` 为例，逐步展示如何定义其策略、后端、运算、随机生成器、模糊化方法，并最终通过扩展系统使其功能可被外部调用。

这个过程将全面展示注册表系统（Registry System）的强大之处，它如何将各个独立的组件解耦，并通过统一的接口进行管理和调度。

## 步骤 1: 定义 FuzznumStrategy

第一步是为新类型创建一个 `FuzznumStrategy`。这是模糊数的核心逻辑层，负责定义其属性、验证约束以及格式化输出。

对于 `qrofn`，它包含两个核心属性：隶属度 `md` 和非隶属度 `nmd`，并受约束 `md^q + nmd^q <= 1` 的限制。

```python
# axisfuzzy/fuzztype/qrofs/qrofn.py

from typing import Optional, Any
import numpy as np
from ...config import get_config
from ...core import FuzznumStrategy, register_strategy

@register_strategy
class QROFNStrategy(FuzznumStrategy):
    mtype = 'qrofn'
    md: Optional[float] = None
    nmd: Optional[float] = None

    def __init__(self, q: Optional[int] = None):
        super().__init__(q=q)
        # 添加属性验证器
        self.add_attribute_validator('md', lambda x: 0 <= x <= 1)
        self.add_attribute_validator('nmd', lambda x: 0 <= x <= 1)
        # 添加回调以在属性变化时检查约束
        self.add_change_callback('md', self._on_membership_change)
        self.add_change_callback('nmd', self._on_membership_change)
        self.add_change_callback('q', self._on_q_change)

    def _fuzz_constraint(self):
        # 核心约束检查
        if self.md is not None and self.nmd is not None and self.q is not None:
            sum_of_powers = self.md ** self.q + self.nmd ** self.q
            if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                raise ValueError("violates fuzzy number constraints")

    # ... 其他回调和格式化方法 ...
```

**关键点**:

-   `@register_strategy`: 这个装饰器将 `QROFNStrategy` 注册到模糊类型注册表。
-   `mtype = 'qrofn'`: 定义了该策略对应的模糊类型标识符。
-   `add_attribute_validator`: 用于确保 `md` 和 `nmd` 的值始终在 `[0, 1]` 区间内。
-   `add_change_callback` 和 `_fuzz_constraint`: 实现 `qrofn` 的核心数学约束。当 `md`, `nmd`, 或 `q` 发生变化时，自动触发约束检查。

## 步骤 2: 实现 FuzzarrayBackend

接下来，我们需要一个 `FuzzarrayBackend` 来高效地存储和管理 `Fuzzarray` 中的大量 `qrofn` 数据。我们采用 `Struct of Arrays (SoA)` 模式，将所有 `md` 和 `nmd` 分别存储在两个独立的 NumPy 数组中，以实现向量化操作。

```python
# axisfuzzy/fuzztype/qrofs/backend.py

from typing import Any, Tuple
import numpy as np
from ...core import FuzzarrayBackend, register_backend

@register_backend
class QROFNBackend(FuzzarrayBackend):
    mtype = 'qrofn'

    def _initialize_arrays(self):
        # 为 md 和 nmd 初始化两个独立的 NumPy 数组
        self.mds = np.zeros(self.shape, dtype=np.float64)
        self.nmds = np.zeros(self.shape, dtype=np.float64)

    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        # 从数组中提取单个元素以创建 Fuzznum
        md_value = float(self.mds[index])
        nmd_value = float(self.nmds[index])
        return Fuzznum(mtype=self.mtype, q=self.q).create(md=md_value, nmd=nmd_value)

    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        # 将 Fuzznum 的数据设置回数组
        self.mds[index] = fuzznum.md
        self.nmds[index] = fuzznum.nmd

    @classmethod
    def from_arrays(cls, mds: np.ndarray, nmds: np.ndarray, q: int) -> 'QROFNBackend':
        # 高性能地从现有数组创建后端
        backend = cls(mds.shape, q)
        backend.mds = mds.copy()
        backend.nmds = nmds.copy()
        return backend

    def get_component_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        # 暴露底层数组，用于向量化运算
        return self.mds, self.nmds

    # ... 其他方法如 copy, slice_view, 格式化等 ...
```

**关键点**:

-   `@register_backend`: 将 `QROFNBackend` 注册到模糊类型注册表，并与 `'qrofn'` 类型关联。
-   `_initialize_arrays`: 定义了 `Fuzzarray` 创建时如何初始化其内部数据结构。
-   `get_component_arrays`: 这是实现向量化操作的核心。它允许运算模块直接访问底层的 `mds` 和 `nmds` 数组。

## 步骤 3: 注册核心运算

现在，我们可以为 `qrofn` 定义各种数学运算，例如加法、乘法等。每个运算都是一个继承自 `OperationMixin` 的类。

以加法为例：`A + B = (S(md_A, md_B), T(nmd_A, nmd_B))`

```python
# axisfuzzy/fuzztype/qrofs/op.py

from ...core import OperationTNorm, OperationMixin, register_operation

@register_operation
class QROFNAddition(OperationMixin):
    def get_operation_name(self) -> str:
        return 'add'  # 对应 '+' 运算符

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']  # 支持 'qrofn' 类型

    def _execute_binary_op_impl(self, strategy_1, strategy_2, tnorm: OperationTNorm):
        # 单个 Fuzznum 的运算逻辑
        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self, fuzzarray_1, other, tnorm: OperationTNorm):
        # Fuzzarray 的向量化运算逻辑
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        md_res = tnorm.t_conorm(mds1, mds2)
        nmd_res = tnorm.t_norm(nmds1, nmds2)

        # 直接使用 Backend 创建结果，避免逐元素操作
        backend_cls = get_registry_fuzztype().get_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)
```

**关键点**:

-   `@register_operation`: 将该运算注册到运算注册表。
-   `get_operation_name`: 返回的 `'add'` 会被分发器映射到 `__add__` 方法。
-   `get_supported_mtypes`: 声明此运算仅用于 `'qrofn'` 类型。
-   `_execute_fuzzarray_op_impl`: 这是高性能的关键。它通过 `_prepare_operands` 获取底层的 NumPy 数组，执行向量化计算，然后直接用结果数组创建新的 `Fuzzarray`。

## 步骤 4: 注册随机生成器

为了方便地创建测试数据，我们实现一个随机生成器。

```python
# axisfuzzy/fuzztype/qrofs/random.py

from ...random import register_random, ParameterizedRandomGenerator

@register_random
class QROFNRandomGenerator(ParameterizedRandomGenerator):
    mtype = "qrofn"

    def get_default_parameters(self) -> Dict[str, Any]:
        # 定义生成 qrofn 的默认参数
        return { 'md_dist': 'uniform', 'md_low': 0.0, ... }

    def fuzzarray(self, rng, shape, q, **params) -> 'Fuzzarray':
        # 向量化生成隶属度 mds
        mds = self._sample_from_distribution(rng, size=size, ...)

        # 向量化生成非隶属度 nmds，同时处理约束
        if params['nu_mode'] == 'orthopair':
            max_nmd = (1 - mds ** q) ** (1 / q)
            # ... (省略缩放和裁剪逻辑)
        else:
            # ...

        # 直接从数组创建 Backend
        backend = QROFNBackend.from_arrays(mds=mds, nmds=nmds, q=q)
        return Fuzzarray(backend=backend)

    # ... 其他方法如 fuzznum, validate_parameters ...
```

**关键点**:

-   `@register_random`: 将 `QROFNRandomGenerator` 注册到随机生成器注册表。
-   `mtype = "qrofn"`: 将其与 `'qrofn'` 类型关联。
-   `fuzzarray`: 实现了完全向量化的生成逻辑，可以高效地创建大型 `Fuzzarray`。

## 步骤 5: 注册模糊化策略

模糊化是将清晰值（crisp values）转换为模糊数的过程。我们需要定义一个策略来处理这个转换。

```python
# axisfuzzy/fuzztype/qrofs/fuzzification.py

from ...fuzzifier import FuzzificationStrategy, register_fuzzifier

@register_fuzzifier(is_default=True)
class QROFNFuzzificationStrategy(FuzzificationStrategy):
    mtype = "qrofn"
    method = "default"

    def fuzzify(self, x, mf_cls, mf_params_list):
        x = np.asarray(x, dtype=float)
        results = []
        for params in mf_params_list:
            mf = mf_cls(**params)
            # 向量化计算 md 和 nmd
            mds = np.clip(mf.compute(x), 0, 1)
            nmds = np.maximum(1 - mds**self.q - self.pi**self.q, 0.0) ** (1/self.q)

            backend_cls = get_registry_fuzztype().get_backend(self.mtype)
            backend = backend_cls.from_arrays(mds=mds, nmds=nmds, q=self.q)
            results.append(Fuzzarray(backend=backend, ...))

        # ... (处理多组参数的情况) ...
```

**关键点**:

-   `@register_fuzzifier(is_default=True)`: 将其注册为 `'qrofn'` 类型的默认模糊化策略。
-   `fuzzify`: 接收清晰值 `x` 和隶属函数参数，通过向量化计算生成 `mds` 和 `nmds`，并最终创建 `Fuzzarray`。

## 步骤 6: 使用扩展系统

最后，我们可以通过扩展系统（Extension System）为 `Fuzznum` 和 `Fuzzarray` 添加特定于 `qrofn` 的便捷方法或顶层函数，例如 `score`（得分函数）、`acc`（精确度函数）等。

```python
# axisfuzzy/fuzztype/qrofs/extension.py

from ...extension import extension
from . import ext  # 导入包含具体实现的模块

@extension(
    name='score',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property'
)
def qrofn_score_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the score of a QROFN Fuzzarray or Fuzznum."""
    return ext._qrofn_score(fuzz) # 委托给具体实现
```

`ext._qrofn_score` 的实现通常也是向量化的：

```python
# axisfuzzy/fuzztype/qrofs/ext/_score.py

def _qrofn_score(fuzz):
    if isinstance(fuzz, Fuzznum):
        return fuzz.md ** fuzz.q - fuzz.nmd ** fuzz.q
    elif isinstance(fuzz, Fuzzarray):
        mds, nmds = fuzz.backend.get_component_arrays()
        return mds ** fuzz.q - nmds ** fuzz.q
```

**关键点**:

-   `@extension`: 声明一个扩展。
-   `name='score'`: 注入的属性/方法名。
-   `mtype='qrofn'`: 此扩展仅对 `mtype` 为 `'qrofn'` 的对象生效。
-   `injection_type='instance_property'`: 将其作为实例的属性注入。
-   实现逻辑 (`_qrofn_score`) 再次利用了 `backend.get_component_arrays()` 来实现对 `Fuzzarray` 的高效向量化计算。

## 结论

通过以上六个步骤，我们成功地将 `qrofn` 类型无缝集成到了 `AxisFuzzy` 框架中。每个组件（策略、后端、运算、随机生成器、模糊化器、扩展）都被独立定义和注册，但通过注册表系统协同工作，形成了一个完整、高效且可扩展的模糊类型支持体系。

这个流程同样适用于 `qrohfn`（q-rung orthopair hesitant fuzzy number），其 `Strategy` 和 `Backend` 处理的是 `object` 类型的数组（因为犹豫模糊集是变长的），但核心的注册和解耦思想是完全一致的。

这种基于注册表的设计模式是 `AxisFuzzy` 框架保持灵活性和可扩展性的基石。