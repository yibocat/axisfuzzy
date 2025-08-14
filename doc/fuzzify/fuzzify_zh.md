好的，我将先提交中文文档草案，落在文档目录 doc/fuzzify 下的 fuzzify_zh.md。

# FuzzLab 模糊化（Fuzzification）系统说明

本文档全面介绍 FuzzLab 的模糊化系统（fuzzify）与隶属函数系统（membership）：架构设计、核心 API、扩展方法、性能注意事项以及示例代码。

适用版本：2025-08-14 之后的主干版本

---

## 1. 模块概览

FuzzLab 从高性能模糊数计算出发，提供了“将精确值转为模糊数（或模糊数组）”的统一入口，即模糊化（Fuzzification）系统。核心能力包括：

- 通过“隶属函数”将输入 x → 隶属度 `md`；
- 基于目标 `mtype` 的模糊化策略，将 `md`（以及策略参数）→ 模糊数组件；
- 支持标量与批量（数组）输入，采用 SoA 后端构建 `Fuzzarray`，避免 Python 循环。

当前内置 `mtype`：q-rofn（q-rung orthopair fuzzy number）

目录结构（与本说明相关的主要文件）：
- fuzzlab/fuzzify/
  - `base.py`：模糊化策略抽象基类 `FuzzificationStrategy`
  - `fuzzifier.py`：调度器 `Fuzzifier` 与便捷函数 `fuzzify`
  - `registry.py`：策略注册表与装饰器
- fuzzlab/membership/
  - `base.py`：隶属函数基类 `MembershipFunction`
  - `function.py`：内置隶属函数实现（`TriangularMF`, `GaussianMF`, ...）
  - `factory.py`：隶属函数工厂与别名解析
- fuzzlab/fuzztype/qrofs/fuzzify.py：`QROFNFuzzificationStrategy`（`qrofn` 的默认策略）

---

## 2. 设计与职责

- `Fuzzifier`（调度器）
  - 将“配置阶段”和“执行阶段”分离：
    - 配置：选择 `mtype` / `method`、隶属函数与策略参数；
    - 执行：通过 `__call__` 接受数据，输出 `Fuzznum` 或 `Fuzzarray`。
  - 通过自省（`inspect`）自动分拣参数：把 `kwargs` 分配给“策略构造函数”与“隶属函数构造函数”。

- `FuzzificationStrategy`（策略抽象基类）
  - 统一接口：`fuzzify_scalar(x, mf)` 与 `fuzzify_array(x, mf)`
  - 保存策略参数：在 `__init__(q: Optional[int] = None, **kwargs`) 中接收；`self.q` 与 `self.kwargs`
  - `get_strategy_info()` 提供策略元信息（便于调试与日志）

- `Registry`（策略注册表）
  - 维护 `(mtype, method)` → `StrategyClass` 的映射
  - 支持默认方法：每个 `mtype` 可设置默认 `method`
  - 提供装饰器 `@register_fuzzification_strategy(mtype, method, is_default=False)`

- `Membership`（隶属函数）
  - 抽象基类 `MembershipFunction`:`compute(x)` → `[0,1]`
  - 内置函数：`TriangularMF`、`TrapezoidalMF`、`GaussianMF`、`SMF`、`ZMF`、`GeneralizedBellMF`、`PiMF`、`DoubleGaussianMF`、`SigmoidMF`
  - 工厂：`create_mf(name, **kwargs)`、`get_mf_class(name)`
  - 别名：`trimf`、`trapmf`、`gaussmf`、`smf`、`zmf`、`gbellmf`、`pimf`、`gauss2mf`、`sigmoid`

- `QROFNFuzzificationStrategy`（`qrofn` 默认策略）
  - `method='default'`，需要参数：`q`（q阶序对值，默认1）、`pi`（犹豫因子，必需）
  - 约束：$md^q + nmd^q + pi^q ≤ 1$
  - 公式：$nmd = (1 - md^q - pi^q)^{(1/q)}$
  - 标量与向量版本均采用向量化实现；数组路径直接构建 SoA 后端以高性能返回 `Fuzzarray`

---

## 3. 快速开始

若只需一次性模糊化，可使用便捷函数 `fuzzify`；若需同配置多次调用，推荐 `Fuzzifier`。

- 便捷函数 fuzzify（内部创建临时 `Fuzzifier`，再调用）
- 可复用引擎 `Fuzzifier`（在 `__init__` 完成参数配置，`__call__` 仅接收数据）

示例 1：单值模糊化为 `qrofn`（默认策略，三角形隶属函数）

````python
# 单值 -> Fuzznum
from fuzzlab.fuzzify import fuzzify

x = 0.7
fz = fuzzify(
    x=x,
    mf='trimf',     # 三角形隶属函数别名
    mtype='qrofn',  # 目标模糊数类型
    q=2,            # 策略参数：q
    pi=0.2,         # 策略参数：犹豫因子（必填）
    a=0.0, b=0.8, c=1.0  # 隶属函数参数
)
print(fz.get_info())
````

示例 2：批量模糊化为 `Fuzzarray`（高斯隶属函数）

````python
import numpy as np
from fuzzlab.fuzzify import fuzzify

X = np.array([10, 25, 40], dtype=float)
fa = fuzzify(
    x=X,
    mf='gaussmf',
    mtype='qrofn',
    q=3,
    pi=0.1,
    sigma=5.0, c=25.0
)
print(fa.shape, fa.mtype)
````

示例 3：使用可复用引擎 `Fuzzifier`

````python
from fuzzlab.fuzzify import Fuzzifier
from fuzzlab.membership import GaussianMF

# 预构建隶属函数实例
mf = GaussianMF(sigma=4.0, c=20.0)

# 配置阶段（只做一次）
fuzz_engine = Fuzzifier(
    mf=mf,          # 已实例化的隶属函数
    mtype='qrofn',  # 省略 method 使用默认
    q=3,            # 策略参数
    pi=0.15         # 策略参数
)

# 执行阶段（可重复）
print(fuzz_engine(18.0))           # -> Fuzznum
print(fuzz_engine([15.0, 20.0]))   # -> Fuzzarray
````

注意：
- 当传入的是“已实例化的隶属函数”时，不允许再同时传隶属函数的构造参数；否则会抛出明确错误（参数分拣校验）。
- 未知参数（既不属于策略 `__init__`，也不属于隶属函数 `__init__`）会抛出错误，帮助你尽早发现问题。

---

## 4. API 说明

### 4.1 `Fuzzifier`（调度器）

构造函数：
- `Fuzzifier(mf, mtype: Optional[str] = None, method: Optional[str] = None, **kwargs)`
  - `mf`: `MembershipFunction` 实例或其名称字符串（如 'trimf'）
  - `mtype`: 目标模糊数类型（如 `'qrofn'`）；若省略，取全局默认（`get_config().DEFAULT_MTYPE`）
  - `method`: 策略名称；若省略，取该 `mtype` 的默认策略（由注册表管理）
  - `kwargs`: 将通过自省分拣到“策略构造参数”与“隶属函数构造参数”

调用：
- `__call__(x)` -> `Fuzznum | Fuzzarray`
  - x: 标量（`int`/`float`）或数组（`list`/`np.ndarray`）

行为：
- 标量走 `fuzzify_scalar`，数组走 `fuzzify_array`
- 全流程避免 Python 层构造大量对象；数组路径采用向量化 + SoA 后端

### 4.2 `fuzzify`（便捷函数）

- `fuzzify(x, mf, mtype=None, method=None, **kwargs) -> Fuzznum | Fuzzarray`
  - 内部等价：`return Fuzzifier(mf, mtype, method, **kwargs)(x)`

### 4.3 `FuzzificationStrategy`（抽象类）

签名：
- `__init__(q: Optional[int] = None, **kwargs)`
  - `self.q`：常用于 q-rung 系列策略
  - `self.kwargs`：存放策略特定参数（如 `pi`）
- `fuzzify_scalar(x, mf)` -> `Fuzznum`（抽象）
- `fuzzify_array(x, mf)` -> `Fuzzarray`（抽象）
- `get_strategy_info()` -> `dict`

实现建议：
- 在数组路径中避免 Python for 循环，直接向量化计算 `md`、`nmd` 等组件
- 通过 `get_fuzznum_registry()` 获取 `backend_cls` 并调用 `from_arrays` 构建后端

### 4.4 `Registry`（注册表）

- `get_fuzzification_registry()` -> `FuzzificationRegistry`
- `register_fuzzification_strategy(mtype, method, is_default=False)`：装饰器
-` FuzzificationRegistry.register(...)`
- `FuzzificationRegistry.get_strategy(mtype, method=None)`
- `FuzzificationRegistry.get_default_method(mtype)`
- `FuzzificationRegistry.get_available_mtypes()`
- `FuzzificationRegistry.get_available_methods(mtype)`
- `FuzzificationRegistry.get_registry_info()`

### 4.5 `Membership`（隶属函数）

- 抽象基类 `MembershipFunction`
  - `compute(x)` -> `[0,1]`
  - `set_parameters(...)`
  - `get_parameters()`
- 工厂：
  - `get_mf_class(name)` -> `Type[MembershipFunction]`
  - `create_mf(name, **kwargs)` -> `(instance, remaining_kwargs)`
- 内置函数（部分别名）：
  - `TriangularMF`（trimf）、`TrapezoidalMF`（trapmf）、`GaussianMF`（gaussmf）
  - `SMF`（smf）、`ZMF`（zmf）、`GeneralizedBellMF`（gbellmf）
  - `PiMF`（pimf）、`DoubleGaussianMF`（gauss2mf）、`SigmoidMF`（sigmoid）

---

## 5. `qrofn` 默认策略详解：`QROFNFuzzificationStrategy`

位置：`fuzzlab/fuzztype/qrofs/fuzzify.py`  
注册：`@register_fuzzification_strategy('qrofn', 'default')`

参数：
- `q: int = 1`（可选）
- `pi: float`（必需，犹豫因子）

数学约束与公式：
- 约束：$md^q + nmd^q + pi^q ≤ 1$
- 转换：$nmd = (1 - md^q - pi^q)^{(1/q)}$
- `md = mf.compute(x)`；`mf` 应保证返回 `[0,1]`

实现要点：
- 标量路径：返回 `Fuzznum`
- 数组路径：向量化计算 `md/nmd`，使用 `get_fuzznum_registry()` -> `backend_cls.from_arrays(md, nmd, q)` 构建 SoA 后端，再返回 `Fuzzarray`

异常与数值稳定：
- 若未提供 `pi`：抛出 ValueError
- 对 `md`、`pi` 做 `clip` 到 `[0,1]` 范围
- 对 $1 - md^q - pi^q$ 做 `np.maximum(..., 0.0)` 防止微负值

---

## 6. 扩展指南

### 6.1 新增策略（同一 `mtype` 下的多策略）

示例：为 'qrofn' 新增一套不同的“pi 生成规则”的策略

````python
# 示例：新增 qrofn 的另一策略（示意）
from fuzzlab.fuzzify import FuzzificationStrategy, register_fuzzification_strategy
from fuzzlab.membership import MembershipFunction
from fuzzlab.core import Fuzznum, Fuzzarray, get_fuzznum_registry
import numpy as np

@register_fuzzification_strategy('qrofn', 'my_method')
class QROFNMyStrategy(FuzzificationStrategy):
    def __init__(self, q: int = 2, alpha: float = 0.1):
        super().__init__(q=q, alpha=alpha)

    def fuzzify_scalar(self, x: float, mf: MembershipFunction) -> Fuzznum:
        md = mf.compute(x)
        # 示例：用 alpha 导出一个“自定义犹豫度”，再计算 nmd
        pi = np.clip(self.kwargs['alpha'] * (1.0 - md), 0.0, 1.0)
        nmd = np.maximum(1.0 - md**self.q - pi**self.q, 0.0)**(1.0/self.q)
        return Fuzznum(mtype='qrofn', q=self.q).create(md=float(md), nmd=float(nmd))

    def fuzzify_array(self, x: np.ndarray, mf: MembershipFunction) -> Fuzzarray:
        md = mf.compute(x)
        pi = np.clip(self.kwargs['alpha'] * (1.0 - md), 0.0, 1.0)
        nmd = np.maximum(1.0 - md**self.q - pi**self.q, 0.0)**(1.0/self.q)

        backend_cls = get_fuzznum_registry().get_backend('qrofn')
        backend = backend_cls.from_arrays(md=md, nmd=nmd, q=self.q)
        return Fuzzarray(backend=backend, mtype='qrofn', q=self.q)
````

使用：
````python
from fuzzlab.fuzzify import Fuzzifier
fzr = Fuzzifier(mf='gaussmf', mtype='qrofn', method='my_method', q=2, alpha=0.2, sigma=3.0, c=10.0)
y = fzr([9.0, 10.0, 11.0])
````

### 6.2 新增 mtype

- 在 fuzzlab/fuzzy/<your_mtype>/ 下实现：
  - `backend.py`（SoA 后端）
  - `qrofn` 类似的策略实现并注册
- 在 core/registry 中注册 `mtype` 对应的 strategy/backend
- 策略实现尽量复用 `Fuzzifier` 的“参数分拣 + 调度”机制

### 6.3 新增隶属函数

````python
from fuzzlab.membership import MembershipFunction

class MyMF(MembershipFunction):
    def __init__(self, p: float = 2.0, name: str = None):
        super().__init__(name)
        self.p = p
        self.parameters = {'p': p}

    def compute(self, x):
        # 示例：折线型或自定义函数；保证返回 [0,1]
        # 这里只是演示
        import numpy as np
        x = np.asarray(x, dtype=float)
        y = np.clip(1.0 - np.abs(x)/self.p, 0.0, 1.0)
        return y

    def set_parameters(self, **kwargs):
        if 'p' in kwargs:
            self.p = kwargs['p']
            self.parameters['p'] = self.p
````

- 若希望通过别名工厂创建，需在 membership/factory.py 中加入别名映射。

---

## 7. 性能与最佳实践

- 尽量使用数组路径（向量化 + SoA 后端），避免 Python 循环
- 隶属函数 `compute` 应能接收 `np.ndarray` 并返回同形状数组
- 策略中避免重复创建临时对象；必要时使用 `in-place` 或向量表达式
- 合理设置 `q`、`pi`，避免极端数值导致的溢出或下溢
- 在高频路径中避免使用 Python 层判断分支

---

## 8. 常见问题与排查

- Q：为什么传入了 `pi` 仍报缺失？
  - A：检查是否把 `pi` 作为“隶属函数参数”写错了。`Fuzzifier` 会用自省区分策略参数与隶属函数参数。策略参数必须是策略 `__init__` 的命名参数（如 `QROFNFuzzificationStrategy.__init__(..., pi: float)）`。

- Q：传入了已实例化的隶属函数，还能传 a/b/c 等参数吗？
  - A：不行。要么传实例（就不再传构造参数），要么传字符串（通过工厂构造并传参数）。

- Q：报 `Unknown parameter 'xxx'`？
  - A：'xxx' 既不属于策略构造参数，也不属于隶属函数构造参数。修正参数名或其归属。

- Q：`Fuzzifier.__call__` 能否再改参数？
  - A：当前设计中不支持在 `__call__` 时改变策略或隶属函数参数。这样可确保“配置”与“执行”完全分离，提升可预测性与可复用性。

---

## 9. 与核心数据结构的关系

- `Fuzznum`：单个模糊数的门面；由策略在标量路径创建
- `Fuzzarray`：批量模糊数组的容器；由策略在数组路径用 SoA 后端构造
- `FuzznumRegistry` / `Backend`：策略在数组路径通过 `registry` 获取具体后端类，调用 `from_arrays(md, nmd, q)` 构建

---

## 10. 版本规划与后续工作

- 语言术语系统（Linguistic Variable & Terms）：作为 `fuzzify` 的上层用户，复用 `Fuzzifier` 对一组术语进行批量模糊化
- 更多 `mtype` 与策略：如 `ivfn`、`pfs` 等
- 更丰富的隶属函数库与参数校验工具
- 基准测试与数值稳定性工具（如对极值/边界输入的测试集）

---

## 11. 参考示例（完整流程）

````python
import numpy as np
from fuzzlab.fuzzify import Fuzzifier, fuzzify

# 1) 便捷函数：一次性调用
fz = fuzzify(x=0.4, mf='trimf', mtype='qrofn', q=2, pi=0.25, a=0.0, b=0.5, c=1.0)
print("Single:", fz.get_info())

# 2) 调度器：多次复用
fzr = Fuzzifier(mf='gaussmf', mtype='qrofn', q=3, pi=0.1, sigma=5.0, c=25.0)
arr = np.array([10.0, 25.0, 40.0], dtype=float)
fa = fzr(arr)
print("Array:", fa.shape, fa.mtype)

# 3) 传入自定义隶属函数实例
from fuzzlab.membership import TriangularMF
mf_ins = TriangularMF(a=0.0, b=1.0, c=2.0)
fzr2 = Fuzzifier(mf=mf_ins, mtype='qrofn', q=2, pi=0.2)
print(fzr2(1.3))
````
