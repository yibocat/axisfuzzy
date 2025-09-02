# 注册表系统 (`registry.py`)

## 1 为什么需要注册表

想象一下，如果没有注册表，当用户调用 `fr.rand('qrofn')` 时，系统如何知道去哪里找 QROFN 的随机生成器呢？当开发者想要添加一个新的 `mtype` 时，又如何让系统自动识别这个新的生成器呢？

**注册表（Registry）** 就像一个“电话簿”，它解决了这个问题：

- **自动发现**：开发者只需用 `@register_random` 装饰器标记自己的生成器类，系统就能自动发现并注册它。
- **生成器（Generator）** 在加载时，会主动向这个“电话簿”**注册**自己，告诉系统：“我是处理 `qrofn` 的，找我请用这个名字”。
- **统一管理**：在需要时，只需向“电话簿”**查询**：“请给我处理 `qrofn` 的生成器实例”，而无需关心这个生成器具体是什么类，在哪里定义。
- **插件化扩展**：新增 `mtype` 不需要修改核心代码，只需添加生成器类并注册即可

## 2 注册表的核心：`RandomGeneratorRegistry`

`RandomGeneratorRegistry` 是一个**单例类**，这意味着整个应用程序中只有一个注册表实例，确保了全局的一致性。它的主要职责包括：

### 2.1 线程安全的存储

注册表内部维护一个字典 `_generators: Dict[str, BaseRandomGenerator]`，键是 `mtype` 字符串，值是对应的生成器**实例**（不是类！）。所有的操作都通过锁机制保护，确保多线程环境下的安全性。

### 2.2 生成器管理功能

注册表提供了完整的 CRUD（创建、读取、更新、删除）操作：
1. **注册生成器** `register(mtype, generator)`
    - 将一个生成器实例与 `mtype` 关联。
    - 会验证生成器是否继承自 `BaseRandomGenerator`。
    - 检查生成器的 `mtype` 属性是否与注册的 `mtype` 一致。
2. **查询生成器** `get_generator(mtype)`
    - 根据 `mtype` 返回对应的生成器实例。
    - 如果未找到，返回 `None`。
3. **检查注册状态** `is_registered(mtype)` 和 `__contains__(mtype)`
    - 快速检查某个 `mtype` 是否已注册。
    - 支持 Python 的 `in` 操作符：`'qrofn' in registry`。
4. **列举所有类型** `list_mtypes()`
    - 返回所有已注册的 `mtype` 列表（按字母排序）。
5. **注销生成器** `unregister(mtype)`
    - 从注册表中移除指定的 `mtype` 及其生成器。
## 3. 装饰器：`@register_random`

为了让生成器注册更方便，`registry.py` 提供了一个装饰器：

```python
@register_random  # <-- 关键！
class MyMtypeRandomGenerator(ParameterizedRandomGenerator):
    mtype = "my_mtype"  # <-- 必须提供 mtype

    # ... 实现 get_default_parameters, validate_parameters, fuzznum, fuzzarray...
```

当 Python 解释器执行到这段代码时，`@register_random` 装饰器会：
1. **读取类的 `mtype` 属性**：确认这个生成器服务于哪种模糊数类型。
2. **实例化生成器类**：调用 `cls()` 创建一个生成器实例。
3. **注册到全局注册表**：调用 `registry.register(cls.mtype, generator_instance)`。
4. **返回原始类**：装饰器不修改类本身，只是完成注册过程。

采用装饰器的优势是:
- **零侵入性**：不需要修改生成器类的实现，只需添加一行装饰器。
- **自动化**：注册过程在模块导入时自动发生，无需手动调用。
- **声明式**：通过装饰器，可以一眼看出这个类是一个随机生成器。

## 4 全局访问函数

为了方便使用，`registry.py` 还提供了一系列全局函数，它们都是对单例注册表的封装：

```python
# 获取全局注册表实例
registry = get_registry_random()

# 获取指定 mtype 的生成器
generator = get_random_generator('qrofn')

# 检查 mtype 是否已注册
if is_registered_random('qrofn'):
    print("QROFN 生成器可用")

# 列举所有可用类型
available_types = list_registered_random()
print(f"可用的 mtype: {available_types}")
```

## 5. 示例：注册一个新生成器

假设我们有一个新的模糊数类型 `simple_fn`，我们实现了它的生成器：

```python
from axisfuzzy.random.base import ParameterizedRandomGenerator
from axisfuzzy.random.registry import register_random

@register_random
class SimpleFNRandomGenerator(ParameterizedRandomGenerator):
    mtype = "simple_fn"
    def get_default_parameters(self):
        return {'low': 0.0, 'high': 1.0, 'dist': 'uniform'}
    def validate_parameters(self, **params):
        pass
    def fuzznum(self, rng, **params):
        ...
    def fuzzarray(self, rng, shape, **params):
        ...
```

只要这个类所在的模块被导入，它就会自动出现在：

```python
from axisfuzzy.random.registry import list_registered_random
print(list_registered_random())  # ['qrofn', 'qrohfn', 'simple_fn']
```

## 6. 总结

注册表是 `axisfuzzy` 随机化系统实现**插件化**和**可扩展性**的核心。通过 `@register_random` 装饰器，开发者可以轻松地为新的模糊数类型添加随机生成能力，而无需修改任何现有核心代码。这种“约定优于配置”的设计大大简化了扩展流程。