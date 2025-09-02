# `axisfuzzy.config` 配置系统：深入 `Config` 数据类

在 `axisfuzzy.config` 的核心，`Config` 数据类扮演着“配置蓝图”的角色。它不仅定义了所有可用的配置项及其默认值，还通过元数据（metadata）嵌入了丰富的验证逻辑和描述信息。本文将深入剖析 `Config` 类的结构，解释每个配置项的含义，并阐述其背后的验证机制。

## 1. `Config` 数据类概览

`Config` 是一个标准的 Python `dataclass`，位于 `config_file.py` 模块中。它的主要职责是作为一个纯粹的数据容器，集中管理库的所有全局设置。每个配置项都被定义为该类的一个属性，并使用 `dataclasses.field` 函数进行详细配置。

一个典型的字段定义如下所示：

```python
from dataclasses import dataclass, field

@dataclass
class Config:
    DEFAULT_PRECISION: int = field(
        default=4,
        metadata={
            'category': 'basic',
            'description': 'Default calculation precision (number of decimal places), affects all numeric calculations and display',
            'validator': lambda x: isinstance(x, int) and x >= 0,
            'error_msg': "Must be a non-negative integer."
        }
    )
    # ... 其他字段
```

每个字段都包含两个主要部分：
- **默认值 (`default`)**: 当用户未指定时，该配置项的初始值。
- **元数据 (`metadata`)**: 一个字典，包含了关于该字段的额外信息，是实现验证和文档化的关键。它通常包括：
  - `category`: 用于对配置项进行逻辑分组（如 `basic`, `performance`, `display`）。
  - `description`: 对该配置项用途的详细文字说明。
  - `validator`: 一个 `lambda` 函数或可调用对象，用于验证用户输入值的有效性。
  - `error_msg`: 当验证失败时，向用户显示的错误信息。

## 2. 配置项详解

下面，我们将按照 `category` 对 `Config` 类中的所有配置项进行分类介绍。

### 基础配置 (`basic`)

这类配置项定义了 `axisfuzzy` 库最核心、最基本的行为。

- **`DEFAULT_MTYPE: str`**
  - **描述**: 默认的模糊数类型。当用户在不指定具体类型的情况下创建 `Fuzznum` 对象时，系统将使用此类型。
  - **默认值**: `'qrofn'` (Q-Rung Orthopair Fuzzy Number)
  - **验证规则**: 必须是一个非空字符串。

- **`DEFAULT_Q: int`**
  - **描述**: 默认的 Q-Rung 值。此参数仅对 Q-Rung Orthopair 模糊数有效，用于定义其“rung”的数量。
  - **默认值**: `1`
  - **验证规则**: 必须是一个正整数。

- **`DEFAULT_PRECISION: int`**
  - **描述**: 默认的计算精度，即小数点后的位数。它会影响库中所有的数值计算和结果显示。
  - **默认值**: `4`
  - **验证规则**: 必须是一个非负整数。

- **`DEFAULT_EPSILON: float`**
  - **描述**: 默认的数值容差。这是一个极小的值，用于浮点数的相等性比较和判断一个值是否接近于零。
  - **默认值**: `1e-12`
  - **验证规则**: 必须是一个正数。

### 性能配置 (`performance`)

这类配置项用于调整与计算性能和内存使用相关的行为。

- **`CACHE_SIZE: int`**
  - **描述**: 运算缓存的最大条目数。`axisfuzzy` 会缓存一些计算结果以提高重复运算的速度。此值控制了缓存所占用的内存大小。
  - **默认值**: `256`
  - **验证规则**: 必须是一个非负整数。设置为 `0` 可以禁用缓存。

### 调试配置 (`debug`)

这类配置项主要用于开发和调试阶段，以帮助发现潜在问题。

- **`TNORM_VERIFY: bool`**
  - **描述**: T-Norm 验证开关。如果设置为 `True`，系统会在 T-Norm 初始化后运行额外的检查，以验证其是否满足数学公理（如交换律、结合律等）。
  - **默认值**: `False`
  - **注意**: 开启此选项会显著影响性能，建议仅在调试时使用。

### 显示配置 (`display`)

这类配置项控制了大型模糊数数组在控制台中的显示方式，以避免信息刷屏。

- **`DISPLAY_THRESHOLD_SMALL: int`**
- **`DISPLAY_THRESHOLD_MEDIUM: int`**
- **`DISPLAY_THRESHOLD_LARGE: int`**
- **`DISPLAY_THRESHOLD_HUGE: int`**
  - **描述**: 定义了不同尺寸数组（小、中、大、巨大）的元素数量阈值。当数组的元素总数超过某个阈值时，将采用折叠方式显示。
  - **验证规则**: 必须是正整数。

- **`DISPLAY_EDGE_ITEMS_MEDIUM: int`**
- **`DISPLAY_EDGE_ITEMS_LARGE: int`**
- **`DISPLAY_EDGE_ITEMS_HUGE: int`**
  - **描述**: 对于中、大、巨大尺寸的数组，定义了在折叠显示时，数组的每个维度两端各显示多少个元素。
  - **验证规则**: 必须是正整数。

## 3. 验证机制：`ConfigManager` 如何工作

虽然验证逻辑定义在 `Config` 类的元数据中，但实际的验证操作是由 `ConfigManager` 在幕后执行的。当用户调用 `axisfuzzy.config.set_config(key=value)` 时，`ConfigManager` 会执行以下步骤：

1. **查找字段**: 管理器首先检查 `key` 是否是 `Config` dataclass 中一个合法的字段名。
2. **提取验证器**: 如果字段存在，管理器会从该字段的 `metadata` 中提取出 `'validator'` 对应的 `lambda` 函数。
3. **执行验证**: 管理器调用该 `lambda` 函数，并将用户提供的 `value` 作为参数传入。
4. **处理结果**:
   - 如果 `lambda` 函数返回 `True`，表示验证通过，`ConfigManager` 会安全地更新其内部 `Config` 实例的状态。
   - 如果 `lambda` 函数返回 `False` 或抛出异常，表示验证失败。`ConfigManager` 会捕获这个失败，并从元数据中提取 `'error_msg'`，然后构造一个清晰的 `ValueError` 异常，将其抛出给用户。

例如，当用户尝试执行 `config.set_config(DEFAULT_PRECISION=-1)` 时：

- **Manager**: 查找到 `DEFAULT_PRECISION` 字段。
- **Manager**: 提取其验证器 `lambda x: isinstance(x, int) and x >= 0`。
- **Manager**: 调用 `validator(-1)`，返回 `False`。
- **Manager**: 提取错误信息 `"Must be a non-negative integer."`。
- **Manager**: 抛出 `ValueError: Invalid value for 'DEFAULT_PRECISION': -1. Must be a non-negative integer.`。

通过这种方式，`axisfuzzy.config` 系统将配置的定义与验证逻辑紧密地绑定在一起，同时将执行细节封装在管理器中，从而实现了一个既灵活又高度可靠的配置框架。