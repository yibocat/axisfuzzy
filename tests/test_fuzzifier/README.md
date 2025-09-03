# AxisFuzzy 模糊化系统测试套件

本目录包含 `axisfuzzy.fuzzifier` 模块的全面测试套件，涵盖模糊化系统的核心功能、策略注册、集成测试和性能验证。

## 测试文件结构

### 核心测试文件

#### `conftest.py`
测试环境配置和共享 fixtures，提供：
- 样本清晰值和隶属函数参数
- 模糊化器和注册表实例
- 模拟策略类和测试数据
- 性能测试数据和配置备份
- 警告抑制和环境设置

#### `test_registry.py`
模糊化策略注册表功能测试，包括：
- **注册表单例模式**：验证 `get_registry_fuzzify()` 返回同一实例
- **策略注册和检索**：测试策略的注册、获取和列表功能
- **默认方法管理**：验证默认策略方法的设置和获取
- **注册表信息**：测试注册表状态信息的获取
- **装饰器注册**：验证 `@register_fuzzifier` 装饰器功能
- **注册验证**：测试重复注册和无效策略的错误处理
- **边界情况**：空注册表和不存在策略的处理

#### `test_strategy.py`
模糊化策略基类和具体策略测试，涵盖：
- **抽象基类验证**：确保 `FuzzificationStrategy` 为抽象类
- **QROFN 策略测试**：
  - 初始化参数验证（q 值范围检查）
  - 标量和数组模糊化功能
  - 不同隶属函数的兼容性
  - q 参数对约束条件的影响
- **QROHFN 策略测试**：
  - 初始化参数验证（q, pi, nmd_generation_mode）
  - 三种非隶属度生成模式（pi_based, proportional, uniform）
  - 犹豫度约束验证
  - 批量处理和矢量计算
- **策略注册验证**：确保策略正确注册到注册表
- **策略比较**：不同策略输出结构和信息的对比

#### `test_fuzzifier.py` ✅ **已重写完成**
Fuzzifier 核心类功能测试，包括：
- **初始化测试**（TestFuzzifierInitialization）：
  - 隶属函数实例、类、字符串名称初始化
  - 单/多参数字典和列表初始化
  - 默认参数、缺失参数和无效参数错误处理
- **调用功能测试**（TestFuzzifierCall）：
  - 单值、列表、NumPy数组、多维数组模糊化
  - 运行时参数覆盖和验证
  - Fuzznum和Fuzzarray结果类型验证
- **配置管理测试**（TestFuzzifierConfiguration）：
  - 配置获取和序列化（修正隶属函数类名序列化）
  - 从配置恢复Fuzzifier实例
  - 配置往返一致性验证
- **绘图功能测试**（TestFuzzifierPlotting）：
  - 绘图方法调用验证
  - 模拟matplotlib库调用
- **边界情况测试**（TestFuzzifierEdgeCases）：
  - 空数组和大数组处理（修正shape断言）
  - 极值输入处理
- **策略集成测试**（TestFuzzifierStrategies）：
  - 不同模糊化策略的参数传递
  - 策略切换和兼容性验证

#### `test_integration.py` ❌ **已删除**
原集成测试文件已删除，原因：
- **API过时**：测试代码使用了已废弃的Fuzzifier API
- **功能重复**：测试内容与重写的test_fuzzifier.py大量重复
- **维护成本**：需要大量修改才能适配当前API
- **测试覆盖**：核心功能已在其他测试文件中充分覆盖

**注意**：集成测试功能已整合到test_fuzzifier.py的TestFuzzifierStrategies类中

## 测试策略架构

### 分层测试方法
1. **单元测试**：独立测试各个组件功能
2. **集成测试**：验证组件间协作和数据流
3. **系统测试**：端到端功能和性能验证
4. **边界测试**：极值和异常情况处理

### 测试覆盖范围
- **功能覆盖**：所有公共 API 和核心算法
- **参数覆盖**：各种参数组合和边界值
- **错误覆盖**：异常情况和错误恢复
- **性能覆盖**：大规模数据和内存效率

## 运行测试

### 运行所有模糊化测试
```bash
pytest test/test_fuzzifier/ -v
```

### 运行特定测试文件
```bash
# 注册表测试
pytest test/test_fuzzifier/test_registry.py -v

# 策略测试
pytest test/test_fuzzifier/test_strategy.py -v

# Fuzzifier 核心测试（包含集成测试功能）
pytest test/test_fuzzifier/test_fuzzifier.py -v
```

### 运行特定测试类
```bash
# 注册表功能测试
pytest test/test_fuzzifier/test_registry.py::TestFuzzificationStrategyRegistry -v

# QROFN 策略测试
pytest test/test_fuzzifier/test_strategy.py::TestQROFNFuzzificationStrategy -v

# Fuzzifier 初始化测试
pytest test/test_fuzzifier/test_fuzzifier.py::TestFuzzifierInitialization -v
```

### 性能和压力测试
```bash
# 运行性能相关测试
pytest test/test_fuzzifier/ -k "performance or scalability" -v

# 运行集成测试
pytest test/test_fuzzifier/test_integration.py -v
```

## 测试数据和 Fixtures

### 共享测试数据
- `sample_crisp_values`：标准清晰值样本
- `sample_mf_params`：隶属函数参数样本
- `performance_test_data`：大规模性能测试数据
- `test_data_range`：连续数据范围

### 模糊化器实例
- `fuzzifier_instance`：默认 QROFN 模糊化器
- `qrohfn_fuzzifier_instance`：QROHFN 模糊化器
- `registry_instance`：策略注册表实例

### 模拟对象
- `mock_strategy_class`：模拟策略类
- 配置备份和警告抑制

## 调试和故障排除

### 详细输出
```bash
# 显示详细测试输出
pytest test/test_fuzzifier/ -v -s

# 显示失败的详细信息
pytest test/test_fuzzifier/ --tb=long
```

### 特定测试调试
```bash
# 调试特定测试方法
pytest test/test_fuzzifier/test_fuzzifier.py::TestFuzzifierCall::test_call_with_scalar_and_mf_params -v -s
```

### 性能分析
```bash
# 运行性能测试并显示时间
pytest test/test_fuzzifier/ -k "performance" --durations=10
```

## 测试覆盖率

### 生成覆盖率报告
```bash
# 生成模糊化模块覆盖率报告
pytest test/test_fuzzifier/ --cov=axisfuzzy.fuzzifier --cov-report=html

# 查看覆盖率摘要
pytest test/test_fuzzifier/ --cov=axisfuzzy.fuzzifier --cov-report=term-missing
```

### 覆盖率目标
- **行覆盖率**：> 95%
- **分支覆盖率**：> 90%
- **函数覆盖率**：100%

## 性能基准

### 性能指标
- **小规模数据**（< 1000 元素）：< 100ms
- **中等规模数据**（1000-10000 元素）：< 1s
- **大规模数据**（> 10000 元素）：< 5s
- **内存增长**：< 100MB（重复操作）

### 可扩展性测试
- 支持多达 50 个隶属函数
- 支持数组大小达 100,000 元素
- 并发访问安全性验证

## 扩展测试

### 添加新策略测试
1. 在 `test_strategy.py` 中添加新的测试类
2. 实现策略特定的测试方法
3. 在 `test_integration.py` 中添加集成测试
4. 更新 `conftest.py` 中的相关 fixtures

### 添加新隶属函数测试
1. 在各测试文件中添加新的隶属函数参数
2. 验证与现有策略的兼容性
3. 添加特定的边界情况测试

### 性能测试扩展
1. 在 `conftest.py` 中添加新的性能数据 fixtures
2. 在 `test_integration.py` 中添加相应的性能测试
3. 设置合理的性能基准和超时限制

## 贡献指南

### 测试编写规范
1. **命名约定**：使用描述性的测试方法名
2. **文档要求**：为复杂测试添加简洁的注释
3. **断言清晰**：使用明确的断言消息
4. **数据隔离**：确保测试间数据独立

### 代码质量
1. **遵循 PEP 8**：代码格式规范
2. **类型提示**：为复杂函数添加类型注解
3. **错误处理**：适当的异常测试和验证
4. **性能考虑**：避免不必要的重复计算

### 提交要求
1. **测试通过**：所有测试必须通过
2. **覆盖率维持**：不降低现有覆盖率
3. **性能验证**：新功能不影响现有性能
4. **文档更新**：更新相关测试文档

## 环境要求

### Python 版本
- Python 3.8+

### 核心依赖
- pytest >= 6.0
- numpy >= 1.19
- matplotlib >= 3.3（绘图测试）

### 可选依赖
- pytest-cov（覆盖率报告）
- pytest-xdist（并行测试）
- psutil（内存监控）

### 安装测试依赖
```bash
pip install pytest pytest-cov matplotlib psutil
```

## 常见问题

### Q: 测试运行缓慢
A: 使用 `-x` 参数在首次失败时停止，或使用 `-k` 参数运行特定测试

### Q: 内存不足错误
A: 减少 `performance_test_data` 的大小，或跳过性能测试

### Q: 绘图测试失败
A: 确保安装了 matplotlib，或使用 `--ignore-glob="*plot*"` 跳过绘图测试

### Q: 并发测试不稳定
A: 并发测试可能受系统负载影响，可以单独运行或调整线程数量

---

本测试套件确保 AxisFuzzy 模糊化系统的可靠性、性能和可扩展性。通过全面的测试覆盖，我们能够及时发现问题并保证系统质量。