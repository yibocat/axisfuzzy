# `axisfuzzy.config` 配置系统：使用指南与示例

`axisfuzzy.config` 系统提供了一套简洁、直观的 API，让用户可以轻松地与全局配置进行交互。本文将通过一系列实际的代码示例，详细介绍如何读取、修改、临时设置以及持久化配置。

## 1. 导入配置模块

要使用配置系统，首先需要导入 `axisfuzzy.config` 模块。通常，我们会给它一个简短的别名，如 `config`。

```python
import axisfuzzy.config as config
```

## 2. 读取配置

读取配置项是配置系统最基本的操作。`axisfuzzy` 提供了两种主要方式来获取配置信息。

### 2.1. 获取所有配置 (`get_config`)

如果你想一次性查看所有当前的配置项及其值，可以调用 `get_config()` 函数。它会返回一个 `Config` 类的实例，其中包含了所有当前的设置。

```python
# 获取并打印所有配置
current_config = config.get_config()
print(current_config)
```

**输出示例**:
```text
Config(DEFAULT_MTYPE='qrofn', DEFAULT_Q=1, DEFAULT_PRECISION=4, ...)
```

### 2.2. 获取特定配置项

如果你只关心某个或某几个特定的配置项，可以直接从 `get_config()` 返回的对象中访问它们。

```python
# 通过 get_config() 返回的对象访问
current_config = config.get_config()
precision = current_config.DEFAULT_PRECISION
cache_size = current_config.CACHE_SIZE
print(f"Current precision: {precision}")
print(f"Cache size: {cache_size}")
```

## 3. 修改配置 (`set_config`)

`set_config()` 函数是修改全局配置的唯一入口。你可以一次性修改一个或多个配置项。

```python
# 同时修改计算精度和默认模糊数类型
config.set_config(
    DEFAULT_PRECISION=6,
    DEFAULT_MTYPE='ivfn'  # Interval-valued Fuzzy Number
)

# 验证修改是否生效
updated_config = config.get_config()
print(f"New precision: {updated_config.DEFAULT_PRECISION}")
print(f"New fuzzy type: {updated_config.DEFAULT_MTYPE}")
```

**输出示例**:
```text
New precision: 6
New fuzzy type: ivfn
```

**错误处理**：如果你尝试设置一个无效的值，`set_config` 会抛出一个 `ValueError`，并提供清晰的错误提示。

```python
try:
    config.set_config(DEFAULT_PRECISION=-2)
except ValueError as e:
    print(e)
```

**输出**:
```text
Invalid value for 'DEFAULT_PRECISION': -2. Must be a non-negative integer.
```

## 4. 临时修改配置的替代方案

虽然 `axisfuzzy.config` 目前没有提供内置的上下文管理器，但你可以通过手动保存和恢复配置来实现临时修改的效果：

```python
# 保存当前配置
original_config = config.get_config()
original_precision = original_config.DEFAULT_PRECISION
original_cache_size = original_config.CACHE_SIZE

print(f"Original: Precision = {original_precision}")

# 临时修改配置
config.set_config(DEFAULT_PRECISION=10, CACHE_SIZE=0)
print(f"Modified: Precision = {config.get_config().DEFAULT_PRECISION}")
print(f"Modified: Cache size = {config.get_config().CACHE_SIZE}")

# 执行需要特殊配置的计算...

# 恢复原始配置
config.set_config(DEFAULT_PRECISION=original_precision, CACHE_SIZE=original_cache_size)
print(f"Restored: Precision = {config.get_config().DEFAULT_PRECISION}")
```

**输出示例**:
```text
Original: Precision = 6
Modified: Precision = 10
Modified: Cache size = 0
Restored: Precision = 6
```

## 5. 持久化配置：加载与保存

`axisfuzzy.config` 允许你将当前的配置状态保存到 JSON 文件中，并在需要时重新加载。这对于在不同项目或会话间保持一致的配置非常有用。

### 5.1. 保存配置 (`save_config_file`)

`save_config_file()` 函数会将当前的**所有**配置项及其值写入一个指定的 JSON 文件。

```python
# 将当前配置保存到文件
config.save_config_file('my_settings.json')

print("Configuration saved to my_settings.json")
```

`my_settings.json` 文件内容示例：
```json
{
    "DEFAULT_MTYPE": "ivfn",
    "DEFAULT_Q": 1,
    "DEFAULT_PRECISION": 6,
    "DEFAULT_EPSILON": 1e-12,
    "CACHE_SIZE": 256,
    "TNORM_VERIFY": false,
    ...
}
```

### 5.2. 加载配置 (`load_config_file`)

`load_config_file()` 函数会从一个 JSON 文件中读取配置，并将其应用到全局状态。加载时，系统同样会执行验证。

```python
# 首先，重置配置以模拟新的会话
config.reset_config()
print(f"After reset: Precision = {config.get_config().DEFAULT_PRECISION}")

# 从文件加载配置
config.load_config_file('my_settings.json')
print(f"After loading: Precision = {config.get_config().DEFAULT_PRECISION}")
```

**输出**:
```text
After reset: Precision = 4
After loading: Precision = 6
```

- **注意**: 加载只会覆盖文件中存在的键。如果你的 JSON 文件只包含部分配置项，那么只有这些项会被更新，其余的将保持当前值。

## 6. 重置配置 (`reset_config`)

如果你想将所有配置项恢复到它们的初始默认值，可以调用 `reset_config()` 函数。

```python
# 修改配置
config.set_config(DEFAULT_PRECISION=10)
print(f"Before reset: Precision = {config.get_config().DEFAULT_PRECISION}")

# 重置所有配置
config.reset_config()
print(f"After reset: Precision = {config.get_config().DEFAULT_PRECISION}")
```

**输出**:
```text
Before reset: Precision = 10
After reset: Precision = 4
```

通过这些简单而强大的工具，`axisfuzzy` 的用户可以完全掌控库的行为，使其完美适应各种不同的计算需求和环境。