#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/6 21:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
第3步：支持用户在外部注册和激活
这正是这个新架构的威力所在。

假设一个用户创建了一个名为 `my_fuzzy_types` 的库，其中定义了 `ivfn`。

1. 用户在他们的库中创建 `extend.py` 文件，并注册他们的函数：

```python
# In user's library: my_fuzzy_types/extend.py
from fuzzlab.extend.registry import get_extend_registry
from fuzzlab import Fuzznum
from .ivfn import IVFNStrategy # 用户的策略

registry = get_extend_registry()

@registry.register(name='distance', mtype='ivfn', ...)
def ivfn_distance(fuzz1, fuzz2):
    # ... ivfn 的距离计算逻辑 ...
    pass
```

2. 用户在使用时，需要先导入自己的库，然后手动调用 apply_extensions 来激活他们自己的扩展。

```python
import fuzzlab
import my_fuzzy_types # 1. 导入用户的库，这会触发 ivfn_distance 的注册

# 2. 再次调用 apply_extensions 来注入新注册的函数
#    这个函数现在是 FuzzLab 公共 API 的一部分
fuzzlab.extend.apply_extensions()

# 3. 现在可以正常使用了
ivfn1 = fuzzlab.Fuzznum(mtype='ivfn', ...)
ivfn2 = fuzzlab.Fuzznum(mtype='ivfn', ...)

# 分发器会自动找到并调用 ivfn_distance
dist = fuzzlab.distance(ivfn1, ivfn2)
```

"""