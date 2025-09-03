# Mixin 系统测试套件

本目录包含对 AxisFuzzy mixin 系统的全面测试，涵盖实现层（factory）、注册层（register）和注入层（registry）的功能验证。

## 测试架构概览

Mixin 系统遵循 "实现-注册-注入" (Implement-Register-Inject) 的三层架构：

1. **实现层** (`axisfuzzy/mixin/factory.py`): 包含所有 mtype 无关的结构化操作的具体实现
2. **注册层** (`axisfuzzy/mixin/register.py`): 使用 `@register_mixin` 装饰器将实现函数注册到全局注册表
3. **注入层** (`axisfuzzy/mixin/registry.py`): 在库初始化时动态将注册的函数注入到目标类和模块命名空间

## 测试文件结构

### 核心测试文件

- **`test_factory.py`**: 测试 `factory.py` 中的所有工厂函数实现
  - 形状操作函数：`_reshape_factory`, `_flatten_factory`, `_squeeze_factory`, `_ravel_factory`
  - 变换函数：`_transpose_factory`, `_broadcast_to_factory`
  - 数据访问函数：`_copy_factory`, `_item_factory`
  - 容器操作函数：`_concat_factory`, `_stack_factory`, `_append_factory`, `_pop_factory`
  - 逻辑函数：`_any_factory`, `_all_factory`

- **`test_register.py`**: 测试 `register.py` 中的注册包装函数
  - 验证所有 `@register_mixin` 装饰的函数正确注册
  - 测试不同注入类型（`instance_function`, `top_level_function`, `both`）
  - 验证函数签名和参数传递的正确性

- **`test_registry.py`**: 测试 `registry.py` 中的注册表和注入机制
  - `MixinFunctionRegistry` 类的功能测试
  - 注册、存储和检索机制
  - 动态注入到类和模块命名空间的验证

### 集成测试文件

- **`test_integration.py`**: 端到端集成测试
  - 验证完整的 "实现-注册-注入" 流程
  - 测试注入后的方法在 `Fuzzarray` 和 `Fuzznum` 上的实际调用
  - 验证不同 mtype 下的一致性行为

### 性能和边界测试

- **`test_performance.py`**: 性能基准测试
  - 对比 mixin 方法与原生 NumPy 操作的性能
  - 验证零运行时开销的设计目标
  - 大规模数据的性能测试

- **`test_edge_cases.py`**: 边界情况和异常处理测试
  - 无效输入的异常处理
  - 边界形状和维度的处理
  - 内存限制和错误恢复

## 测试策略

### 1. 单元测试策略

每个工厂函数都将进行以下维度的测试：

- **功能正确性**: 验证函数输出与预期结果一致
- **类型兼容性**: 测试 `Fuzznum` 和 `Fuzzarray` 输入的处理
- **Mtype 无关性**: 验证函数在不同 mtype（qrofs, ivfs, tfn 等）下的一致行为
- **形状处理**: 测试各种数组形状和维度的处理
- **异常处理**: 验证无效输入时的错误处理

### 2. 注册机制测试策略

- **装饰器功能**: 验证 `@register_mixin` 正确收集函数元数据
- **注册表状态**: 检查函数是否正确存储在全局注册表中
- **注入类型**: 验证不同注入类型的正确处理
- **目标类匹配**: 确保函数只注入到指定的目标类

### 3. 集成测试策略

- **端到端流程**: 从函数定义到最终调用的完整流程测试
- **方法可用性**: 验证注入后的方法在目标类上可正常访问
- **参数传递**: 确保方法调用时参数正确传递给底层实现
- **返回值一致性**: 验证通过不同路径调用时返回值的一致性

## 测试数据和夹具

### 测试用例数据

- **基础模糊数**: 不同 mtype 的标准模糊数实例
- **多维数组**: 1D, 2D, 3D 及更高维度的 `Fuzzarray`
- **边界情况**: 空数组、单元素数组、大型数组
- **特殊形状**: 包含 1 的维度、不规则形状

### 共享夹具

```python
@pytest.fixture
def sample_fuzznums():
    """提供不同 mtype 的标准模糊数样本"""
    
@pytest.fixture
def sample_fuzzarrays():
    """提供不同形状和 mtype 的 Fuzzarray 样本"""
    
@pytest.fixture
def registry_instance():
    """提供干净的注册表实例用于测试"""
```

## 测试覆盖率目标

- **代码覆盖率**: 目标 95% 以上
- **分支覆盖率**: 目标 90% 以上
- **功能覆盖率**: 100% 的公开 API 覆盖

## 运行测试

### 运行所有 mixin 测试
```bash
pytest test/test_mixin/ -v
```

### 运行特定测试文件
```bash
pytest test/test_mixin/test_factory.py -v
pytest test/test_mixin/test_register.py -v
pytest test/test_mixin/test_registry.py -v
```

### 运行性能测试
```bash
# 运行所有性能测试
pytest test/test_mixin/test_performance.py -v --benchmark-only

# 运行标记为 performance 的测试
pytest test/test_mixin/ -m "performance" -v
```

### 生成覆盖率报告
```bash
pytest test/test_mixin/ --cov=axisfuzzy.mixin --cov-report=html
```



## 依赖项

测试需要以下依赖：

- `pytest`: 测试框架
- `pytest-cov`: 覆盖率测试
- `pytest-benchmark`: 性能基准测试
- `numpy`: 数值计算和比较
- `hypothesis`: 属性基础测试（用于边界情况生成）

## 配置文件



## 注意事项

1. **Mtype 无关性**: 所有测试都应验证功能在不同 mtype 下的一致性
2. **性能要求**: Mixin 方法应具有与原生方法相当的性能
3. **内存安全**: 测试应验证没有内存泄漏或不当的对象引用
4. **线程安全**: 注册表操作应是线程安全的
5. **测试隔离**: 确保测试之间相互独立，没有副作用

## 贡献指南

添加新的 mixin 功能时，请确保：

1. 在 `test_factory.py` 中添加对应的工厂函数测试
2. 在 `test_register.py` 中添加注册包装函数测试
3. 在 `test_integration.py` 中添加端到端测试
4. 更新本 README 文档
5. 确保所有测试通过且覆盖率达标