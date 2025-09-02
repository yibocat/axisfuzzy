# 扩展该系统

## 1 扩展的目标
我们的目标是为一个假设的、新的模糊数类型 `pfn`（Pytagorean Fuzzy Number，勾股模糊数）添加完整的随机生成功能。`pfn` 的定义与 `qrofn` 类似，但其约束条件是 `md^2 + nmd^2 <= 1`（即 `q=2` 的特殊情况）。

完成扩展后，我们应该能够像使用内置类型一样调用它：
```python
import axisfuzzy.random as fr

# 能够生成单个 pfn
pfn_num = fr.rand('pfn')

# 能够批量生成 pfn 数组
pfn_array = fr.rand('pfn', shape=(100, 50))
```

## 2 扩展步骤
扩展一个新 `mtype` 的随机生成器，通常遵循以下**三步**：

1.  **创建 `random.py` 文件**：在对应 `mtype` 的模块目录下（例如 `axisfuzzy/fuzztype/pfn/`）创建一个 `random.py` 文件。
2.  **实现生成器类**：在该文件中，创建一个继承自 `ParameterizedRandomGenerator` 的新类，并实现其所有抽象方法。
3.  **注册生成器**：使用 `@register_random` 装饰器将你的新类注册到系统中。

## 3 详细实现
让我们一步步来实现 `PFNRandomGenerator`。

### 步骤 1 & 2：创建文件并实现生成器类

我们将在 `axisfuzzy/fuzztype/pfn/random.py` 文件中编写以下代码。

```python
# axisfuzzy/fuzztype/pfn/random.py

import numpy as np
from typing import Any, Dict, Tuple

from ....core import Fuzznum, Fuzzarray
from ....random.base import ParameterizedRandomGenerator
from ....random.registry import register_random
from .backend import PFNBackend # 假设 PFNBackend 已经实现

# 步骤 3：使用装饰器注册
@register_random
class PFNRandomGenerator(ParameterizedRandomGenerator):
    """
    pfn (Pythagorean Fuzzy Number) 的随机生成器。
    约束: md^2 + nmd^2 <= 1
    """
    # 声明 mtype，这是注册的 key
    mtype = "pfn"

    def get_default_parameters(self) -> Dict[str, Any]:
        """定义 pfn 生成的默认参数。"""
        return {
            'md_dist': 'uniform',
            'md_low': 0.0,
            'md_high': 1.0,
            'nu_mode': 'orthopair',  # 'orthopair' 或 'independent'
            'nu_dist': 'uniform',
            'nu_low': 0.0,
            'nu_high': 1.0,
            # 为 beta 和 normal 分布提供默认形状参数
            'a': 2.0, 'b': 2.0,
            'loc': 0.5, 'scale': 0.15
        }

    def validate_parameters(self, **params) -> None:
        """验证参数的有效性。"""
        # pfn 的 q 固定为 2，无需验证 q
        # 验证隶属度范围
        self._validate_range('md_low', params.get('md_low', 0.0), 0.0, 1.0)
        self._validate_range('md_high', params.get('md_high', 1.0), 0.0, 1.0)
        if params.get('md_low', 0.0) > params.get('md_high', 1.0):
            raise ValueError("md_low 不能大于 md_high")

        # 验证非隶属度模式
        if 'nu_mode' in params and params['nu_mode'] not in ['orthopair', 'independent']:
            raise ValueError("nu_mode 必须是 'orthopair' 或 'independent'")

    def fuzznum(self, rng: np.random.Generator, **params) -> 'Fuzznum':
        """生成单个 pfn。"""
        # 1. 合并用户参数和默认参数
        p = self._merge_parameters(**params)
        self.validate_parameters(**p)

        # 2. 生成隶属度 (md)
        md = self._sample_from_distribution(
            rng, dist=p['md_dist'], low=p['md_low'], high=p['md_high'],
            a=p['a'], b=p['b'], loc=p['loc'], scale=p['scale']
        )

        # 3. 根据约束条件生成非隶属度 (nmd)
        max_nmd = (1 - md**2)**0.5  # pfn 的核心约束
        if p['nu_mode'] == 'orthopair':
            # 在 [0, max_nmd] 范围内生成 nmd
            nmd = rng.uniform(0, max_nmd)
        else: # 'independent'
            # 独立生成，然后裁剪
            nmd = self._sample_from_distribution(
                rng, dist=p['nu_dist'], low=p['nu_low'], high=p['nu_high'],
                a=p['a'], b=p['b'], loc=p['loc'], scale=p['scale']
            )
            if md**2 + nmd**2 > 1:
                nmd = min(nmd, max_nmd)

        # 4. 创建 Fuzznum 实例
        return Fuzznum(mtype=self.mtype).create(md=md, nmd=nmd)

    def fuzzarray(self, rng: np.random.Generator, shape: Tuple[int, ...], **params) -> 'Fuzzarray':
        """高性能地批量生成 pfn 数组。"""
        # 1. 合并与验证参数
        p = self._merge_parameters(**params)
        self.validate_parameters(**p)
        size = int(np.prod(shape))

        # 2. 向量化生成隶属度数组
        mds = self._sample_from_distribution(
            rng, size=size, dist=p['md_dist'], low=p['md_low'], high=p['md_high'],
            a=p['a'], b=p['b'], loc=p['loc'], scale=p['scale']
        )

        # 3. 向量化生成非隶属度数组
        max_nmds = (1 - mds**2)**0.5
        if p['nu_mode'] == 'orthopair':
            # 乘以一个 [0,1] 的随机数，保证 nmds 在 [0, max_nmds] 之间
            nmds = rng.uniform(0, 1, size=size) * max_nmds
        else: # 'independent'
            nmds = self._sample_from_distribution(
                rng, size=size, dist=p['nu_dist'], low=p['nu_low'], high=p['nu_high'],
                a=p['a'], b=p['b'], loc=p['loc'], scale=p['scale']
            )
            # 找到超限的元素并修正
            violates_mask = (mds**2 + nmds**2) > 1
            nmds[violates_mask] = np.minimum(nmds[violates_mask], max_nmds[violates_mask])

        # 4. 直接创建后端（高性能的关键）
        backend = PFNBackend.from_arrays(
            mds=mds.reshape(shape),
            nmds=nmds.reshape(shape)
        )
        return Fuzzarray(backend=backend)
```

### 步骤 3：确保模块被导入
最后，你需要确保这个新的 `random.py` 文件被 `axisfuzzy/fuzztype/pfn/__init__.py` 导入，这样 Python 在加载 `pfn` 模块时，`@register_random` 装饰器才能被执行。

```python
# axisfuzzy/fuzztype/pfn/__init__.py

# ... 其他导入 ...
from . import random # <-- 确保这行存在
```

## 4 总结
通过以上三个简单的步骤，我们就成功地为 `pfn` 这个新的模糊数类型添加了完整的、高性能的、可参数化的随机生成功能。整个过程没有修改任何 `axisfuzzy.random` 的核心代码，充分体现了其插件化设计的优雅和强大。

**关键要点回顾**：
-   **继承** `ParameterizedRandomGenerator` 以复用工具。
-   **实现** `get_default_parameters`, `validate_parameters`, `fuzznum`, `fuzzarray` 四个核心方法。
-   在 `fuzzarray` 中**向量化**操作，并直接与**后端**交互，以保证性能。
-   使用 `@register_random` **装饰器**将生成器注入系统。
