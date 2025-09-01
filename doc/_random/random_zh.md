# AxisFuzzy 随机模糊数生成系统 (Random Module)

> 版本: 初稿 2025-08-12  
> 适用范围: 当前主分支 SoA Backend 架构  
> 语言: 中文 (参见 `random_en.md` 获取英文版)

---
## 目录
1. 设计目标与理念  
2. 架构分层概览  
3. 核心概念术语  
4. 快速开始示例  
5. API 说明  
6. qrofn 类型参数详解  
7. 性能优化策略  
8. 扩展：新增自定义随机生成器  
9. 设计原则回顾  
10. 常见问题 (FAQ)  
11. 示例合集  
12. 未来可能的增强 (Roadmap)

---
## 1. 设计目标与理念
AxisFuzzy 随机系统旨在为 **不同 mtype 的模糊数** 提供：
- **高性能**：采用 SoA (Struct of Arrays) 后端，批量生成直接填充组件数组 (Backend)，避免构造临时 Fuzznum 列表。  
- **可扩展**：通过 mtype 注册机制，每种模糊数类型自定义其生成策略。  
- **可复现**：全局随机种子 + 局部调用覆盖。  
- **参数化控制**：分布、约束模式、结构参数拆分。  
- **统一入口**：`random_fuzz` / `rand` 自动区分标量与数组。  
- **简洁实现**：新增类型仅需 1 个类 + 1 行注册。

---
## 2. 架构分层概览
| 层 | 文件 | 作用 |
|----|------|------|
| 种子管理 | `axisfuzzy/random/seed.py` | 全局 RNG 状态；设定 / 获取 / 派生 |
| 注册表 | `axisfuzzy/random/registry.py` | mtype → 生成器实例映射 |
| 抽象基类 | `axisfuzzy/random/base.py` | 统一接口：`fuzznum` / `fuzzarray` |
| 用户 API | `axisfuzzy/random/api.py` + `__init__.py` | 工厂函数与辅助方法 |
| 类型实现 | `axisfuzzy/fuzzy/<mtype>/random.py` | 特定 mtype 生成逻辑，如 `qrofn` |
| 后端结构 | `axisfuzzy/fuzzy/<mtype>/backend.py` | SoA 存储组件数组 |

批量生成流程：  
`rand()` → 解析 RNG → 取注册生成器 → `generator.fuzzarray()` → 构造 Backend → 返回 Fuzzarray。

---
## 3. 核心概念术语
| 名称 | 含义 |
|------|------|
| mtype | 模糊数类型标识 (如 `qrofn`) |
| 结构性参数 | 定义模糊数形态的核心，例如 `q` (q-rung) |
| 生成参数 | 控制随机过程行为的参数 (如 `md_dist`, `nu_mode`) |
| Backend | SoA 后端结构，保存各组件 NumPy 数组 |
| Fuzznum | 单个模糊数面向使用者的对象接口 |
| Fuzzarray | 高维模糊数容器，封装 Backend |

---
## 4. 快速开始示例

```python
import axisfuzzy.random as fr

# 设定全局种子r.set_seed(42)

# 生成单个 q-rung 正交对模糊数
a = fr.rand('qrofn', q=3)

# 生成 1000 个一维 QROFN 数组
fa1 = fr.rand('qrofn', shape=1000, q=4, md_dist='beta', a=2, b=5)

# 生成二维 128 x 256 数组 (高性能向量化)
fa2 = fr.rand('qrofn', shape=(128, 256), q=5, nu_mode='independent')

# 从一维数组无放回抽样 10 个元素
sample = fr.choice(fa1, size=10, replace=False)
```

---
## 5. API 说明
| 函数 | 功能 | 说明 |
|------|------|------|
| `set_seed(seed)` | 设置全局随机种子 | 影响后续所有未显式传入 `seed`/`rng` 的调用 |
| `rand(mtype, shape=None, q=..., **params)` | 生成单个或批量模糊数 | `shape=None` 返回 Fuzznum；否则 Fuzzarray |
| `random_fuzz` | `rand` 别名 | 与 `rand` 完全一致 |
| `choice(fuzzarray, size, replace=True, p=None)` | 抽样 | 仅支持一维 Fuzzarray |
| `uniform/normal/beta` | 数值分布采样 | 与全局种子一致，可辅助调试或自定义策略 |
| `list_mtypes()` | 查看已注册 mtype | 便于发现可用类型 |
| `register(mtype, generator)` | 注册生成器 | 在自定义类型模块导入时执行 |
| `get_generator(mtype)` | 获取生成器实例 | 调试或高级扩展 |

随机数生成器 (RNG) 解析优先级：  
`rng` > `seed` > 全局 `set_seed`。

---
## 6. qrofn 类型参数详解
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| q | int | (调用时显式传入 / 默认 2) | q-rung 结构参数 |
| md_dist | str | `uniform` | 会员度分布：`uniform` / `beta` / `normal` |
| md_low / md_high | float | 0.0 / 1.0 | 会员度采样区间 |
| a / b | float | 2.0 / 2.0 | beta 分布形状参数 (当前 md / nmd 共用) |
| loc / scale | float | 0.5 / 0.15 | normal 分布参数 |
| nu_mode | str | `orthopair` | `orthopair` / `independent` |
| nu_dist | str | `uniform` | 非会员度分布名称 |
| nu_low / nu_high | float | 0.0 / 1.0 | 非会员度采样区间 |

约束实现：
- `orthopair`：根据生成的 md 计算动态上界 `max_nmd = (1 - md^q)^(1/q)`，再缩放 nmd。  
- `independent`：独立采样 nmd，若 `md^q + nmd^q > 1`，掩码裁剪 nmd 到允许上界。  

---
## 7. 性能优化策略
| 策略 | 作用 |
|------|------|
| SoA Backend | 减少 Python 对象，提升缓存友好性 |
| 向量化采样 | 一次性批量生成 md / nmd |
| 动态上界数组 | 避免循环内逐元素判断 |
| 掩码裁剪 | 约束修正 O(1) 向量操作 |
| 标量/批量路径拆分 | 单个生成避免多余 reshape/复制 |

---
## 8. 扩展：新增自定义随机生成器
**步骤：**
1. 创建文件：`axisfuzzy/fuzzy/<your_mtype>/random.py`
2. 编写类：继承 `ParameterizedRandomGenerator`
3. 实现方法：`mtype` / `get_default_parameters` / `validate_parameters` / `fuzznum` / `fuzzarray`
4. 在文件底部注册：
```python
from ...random import register
register('your_mtype', YourMtypeRandomGenerator())
```
5. 使用：

```python
import axisfuzzy.random as fr

fr.rand('your_mtype', shape=512, ...)
```

**建议：** 批量路径中直接构造对应 Backend（避免循环创建 Fuzznum）。

---
## 9. 设计原则回顾
| 原则 | 实现方式 |
|------|----------|
| 分离关注点 | 种子 / 注册 / 抽象协议 / mtype 特化独立 |
| 扩展简单 | 1 个文件 + 1 行注册 |
| 高性能 | SoA + 向量化 + 后端直填 |
| 可复现 | 分层 RNG 解析策略 |
| 可维护 | 结构参数与生成参数显式拆分 |

---
## 10. 常见问题 (FAQ)
| 问题 | 解答 |
|------|------|
| 为什么 `q` 不放进默认参数字典？ | 它是结构属性，语义高于生成策略，应显式传入 |
| 批量生成是否会创建海量 Fuzznum？ | 否，直接填充 Backend 数组 |
| 如何确保复现？ | 启动时调用 `set_seed(固定值)`，避免中途局部 seed 干扰 |
| 可以为 md 与 nmd 设置不同 beta 参数吗？ | 当前共用，可在自定义生成器中分离 |
| 如何并行？ | 为每个线程/进程创建独立 `np.random.default_rng(sub_seed)` 并传入 `rng=` |

---
## 11. 示例合集
**单个：**
```python
fn = fr.rand('qrofn', q=6, md_dist='normal', loc=0.6, scale=0.1)
```
**批量：**
```python
fa = fr.rand('qrofn', shape=(256, 256), q=4, md_dist='beta', a=3, b=5, nu_mode='independent')
```
**抽样：**
```python
subset = fr.choice(fa.reshape(-1), size=50, replace=False)
```
**分布工具：**
```python
values = fr.beta(2, 5, shape=1000)
```

---
## 12. 未来可能的增强 (Roadmap)
| 方向 | 说明 |
|------|------|
| 双独立分布参数 | md / nmd 拆分 a,b / loc,scale |
| 更多分布 | triangular / gamma / 自定义 PDF 插件 |
| 流式分块生成 | 超大数组低峰值内存支持 |
| 统计诊断 | 调试模式下输出采样分布统计 |
| 约束可组合化 | 通用约束表达 DSL |

---
**附录**：问题 / 建议可提交 Issue，或扩展自定义随机生成器后 PR。
