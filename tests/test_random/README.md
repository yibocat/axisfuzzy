# AxisFuzzy Random Module Tests

本目录包含 `axisfuzzy.random` 模块的完整测试套件。该模块负责为不同类型的模糊数提供统一、高性能且可复现的随机生成功能。

## 测试架构概述

`axisfuzzy.random` 采用插件式架构，由四个核心模块组成：

- **`api.py`** - 统一的用户接口（`rand`, `choice`, `uniform`, `normal`, `beta`）
- **`base.py`** - 抽象基类定义（`BaseRandomGenerator`, `ParameterizedRandomGenerator`）
- **`registry.py`** - 生成器注册表系统（单例模式，线程安全）
- **`seed.py`** - 全局随机状态管理（可复现性保证）

## 测试文件结构

```
test_random/
├── conftest.py                 # pytest配置和共享fixture
├── README.md                   # 本文档
├── test_seed.py               # 种子管理系统测试
├── test_base.py               # 抽象基类测试
├── test_registry.py           # 注册表系统测试
└── test_api.py                # 高级API测试
```

## 测试文件详细说明

### 1. `test_seed.py` - 种子管理系统测试

**测试目标**: `axisfuzzy.random.seed` 模块

**核心测试内容**:
- `GlobalRandomState` 类的实例管理和状态控制
- `set_seed()` / `get_seed()` / `get_rng()` / `spawn_rng()` 函数功能
- 线程安全性验证（多线程并发访问）
- 可复现性验证（相同种子产生相同结果）
- 不同种子类型支持（int、SeedSequence、BitGenerator）
- 独立生成器流的统计独立性验证
- 性能测试和错误处理

**关键测试类**:
- `TestGlobalRandomState` - 全局随机状态管理（6个测试方法）
- `TestSeedFunctions` - 种子相关函数（4个测试方法）
- `TestThreadSafety` - 线程安全性（3个测试方法）
- `TestReproducibility` - 可复现性验证（5个测试方法）
- `TestErrorHandling` - 错误处理（3个测试方法）
- `TestEdgeCases` - 边界条件测试（7个测试方法）
- `TestPerformance` - 性能测试（3个测试方法）

### 2. `test_base.py` - 抽象基类测试

**测试目标**: `axisfuzzy.random.base` 模块

**核心测试内容**:
- `BaseRandomGenerator` 抽象接口验证
- `ParameterizedRandomGenerator` 工具方法测试
- 参数合并逻辑（`_merge_parameters`）
- 分布采样功能（`_sample_from_distribution`）
- 参数验证工具（`_validate_range`）
- 抽象方法强制实现检查
- 完整的生成器工作流程集成测试

**关键测试类**:
- `TestBaseRandomGenerator` - 基础生成器接口（4个测试方法）
- `TestParameterizedRandomGenerator` - 参数化生成器工具（6个测试方法）
- `TestParameterMerging` - 参数合并逻辑（4个测试方法）
- `TestDistributionSampling` - 分布采样功能（5个测试方法）
- `TestRangeValidation` - 范围验证功能（5个测试方法）
- `TestEdgeCases` - 边界条件测试（10个测试方法）
- `TestIntegrationWithMocks` - 模拟对象集成测试（1个测试方法）

### 3. `test_registry.py` - 注册表系统测试

**测试目标**: `axisfuzzy.random.registry` 模块

**核心测试内容**:
- `RandomGeneratorRegistry` 单例模式验证
- 生成器注册/注销功能
- `@register_random` 装饰器功能
- 线程安全性（并发注册/查询）
- mtype一致性验证
- 查询和列举功能
- 错误处理（重复注册、无效生成器）
- 性能测试和并发错误处理

**关键测试类**:
- `TestRandomGeneratorRegistry` - 注册表核心功能（10个测试方法）
- `TestRegistrationDecorator` - 装饰器功能（5个测试方法）
- `TestGlobalFunctions` - 全局函数接口（4个测试方法）
- `TestThreadSafety` - 注册表线程安全（4个测试方法）
- `TestRegistryValidation` - 注册验证逻辑（4个测试方法）
- `TestEdgeCases` - 边界条件测试（10个测试方法）
- `TestRegistryPerformance` - 性能测试（3个测试方法）
- `TestErrorHandling` - 错误处理（2个测试方法）

### 4. `test_api.py` - 高级API测试

**测试目标**: `axisfuzzy.random.api` 模块

**核心测试内容**:
- `_resolve_rng()` 内部函数的优先级逻辑
- `rand()` 函数的多种重载形式
- `choice()` 函数的随机采样
- `uniform()` / `normal()` / `beta()` 工具函数
- 参数解析和分发逻辑
- 错误处理（无效mtype、参数错误）
- 种子和RNG参数传递
- 返回类型验证（Fuzznum vs Fuzzarray）
- 集成场景和边界条件测试

**关键测试类**:
- `TestResolveRng` - RNG解析优先级逻辑（5个测试方法）
- `TestRandFunction` - rand函数核心功能（8个测试方法）
- `TestChoiceFunction` - choice函数功能（6个测试方法）
- `TestUtilityFunctions` - 工具函数（uniform/normal/beta）（4个测试方法）
- `TestEdgeCases` - 边界条件测试（10个测试方法）
- `TestIntegrationScenarios` - 集成场景测试（4个测试方法）



## 共享测试工具 (conftest.py)

### 主要Fixtures

- `clean_registry` - 提供干净的注册表环境，自动清理和恢复
- `clean_global_state` - 提供干净的全局随机状态，自动恢复原始状态
- `mock_fuzznum` - 模拟 Fuzznum 实例
- `mock_fuzzarray` - 模拟 Fuzzarray 实例
- `sample_fuzznum` - 示例 Fuzznum 实例
- `sample_rng` - 示例 NumPy 随机生成器
- `thread_barrier` - 线程同步屏障
- `mock_generator` - 模拟随机生成器实例
- `mock_parameterized_generator` - 模拟参数化生成器
- `concurrent_test_helper` - 并发测试工具类
- `sample_test_data` - 示例测试数据集合
- `performance_timer` - 性能测试计时器

### 模拟类

- `MockRandomGenerator` - 基础模拟生成器（继承自 BaseRandomGenerator）
- `MockParameterizedGenerator` - 参数化模拟生成器（继承自 ParameterizedRandomGenerator）
- `ConcurrentTestHelper` - 并发测试辅助类
- `PerformanceTimer` - 性能计时器类

## 运行测试

### 运行所有随机模块测试

```bash
# 运行所有测试
pytest test/test_random/ -v

# 运行测试并显示覆盖率
pytest test/test_random/ --cov=axisfuzzy.random --cov-report=html

# 运行测试并生成详细报告
pytest test/test_random/ -v --tb=long --durations=10
```

### 运行特定测试文件

```bash
# 种子管理测试
pytest test/test_random/test_seed.py -v

# 抽象基类测试
pytest test/test_random/test_base.py -v

# 注册表系统测试
pytest test/test_random/test_registry.py -v

# API测试
pytest test/test_random/test_api.py -v
```

### 运行特定测试类或方法

```bash
# 运行特定测试类
pytest test/test_random/test_seed.py::TestGlobalRandomState -v

# 运行特定测试方法
pytest test/test_random/test_seed.py::TestGlobalRandomState::test_multiple_instances_allowed -v

# 运行包含特定关键词的测试
pytest test/test_random/ -k "thread_safety" -v

# 运行边界测试
pytest test/test_random/ -k "edge" -v
```

### 性能和并发测试

```bash
# 运行性能测试
pytest test/test_random/test_seed.py::TestPerformance -v
pytest test/test_random/test_registry.py::TestRegistryPerformance -v

# 运行并发测试
pytest test/test_random/ -k "concurrent or thread" -v

# 运行线程安全测试
pytest test/test_random/test_seed.py::TestThreadSafety -v
pytest test/test_random/test_registry.py::TestThreadSafety -v

# 运行边界和压力测试
pytest test/test_random/ -k "edge or stress or memory" -v
```

## 测试策略

### 1. 单元测试
- 每个模块独立测试
- 使用模拟对象隔离依赖
- 覆盖所有公共接口
- 验证边界条件和错误情况

### 2. 集成测试
- 验证模块间协作
- 测试端到端流程
- 验证数据流和状态管理
- 确保API一致性

### 3. 性能测试
- 大规模数组生成性能
- 内存使用效率
- 并发访问性能
- 缓存和优化效果

### 4. 并发测试
- 多线程安全性
- 竞态条件检测
- 死锁预防验证
- 数据一致性保证

### 5. 可复现性测试
- 种子机制验证
- 跨平台一致性
- 版本兼容性
- 确定性行为验证

### 6. 边界测试
- 极值参数处理
- 内存压力测试
- 特殊字符和Unicode处理
- 大规模数据生成
- 异常输入验证
- 资源限制测试

## 测试数据和Mock对象

### 测试数据类别
- **有效数据**: 正常的mtype、参数、形状等
- **边界数据**: 最小值、最大值、空值等
- **无效数据**: 错误类型、超出范围、格式错误等
- **特殊数据**: NaN、无穷大、极大数值等

### Mock对象设计
- **功能完整**: 实现所有必需接口
- **行为可控**: 可配置返回值和异常
- **状态可查**: 记录调用历史和参数
- **性能友好**: 避免不必要的计算开销

### 边界测试数据增强
本次更新新增了40个边界测试用例，覆盖以下关键场景：

**API模块边界测试**:
- 极大shape值处理（10^9级别）
- 无效参数类型验证
- 内存限制测试
- 特殊字符mtype处理

**基础模块边界测试**:
- 参数范围极值测试
- 分布参数边界验证
- 数值精度处理
- 特殊浮点值（NaN, Inf, -0.0）

**注册表模块边界测试**:
- 大量注册压力测试
- 极长mtype名称处理
- 并发注册/注销压力测试
- 内存泄漏检测

**种子模块边界测试**:
- 极值种子处理
- 大量spawn操作
- 内存限制测试
- 线程安全边界验证

## 测试统计信息

### 测试数量统计

- **test_seed.py**: 35个测试方法（8个测试类）
- **test_base.py**: 35个测试方法（8个测试类）
- **test_registry.py**: 49个测试方法（9个测试类）
- **test_api.py**: 37个测试方法（7个测试类）
- **总计**: 156个测试方法，32个测试类

### 测试覆盖率目标

- **语句覆盖率**: ≥ 95%
- **分支覆盖率**: ≥ 90%
- **函数覆盖率**: 100%
- **类覆盖率**: 100%
- **边界条件覆盖**: ≥ 90%
- **异常路径覆盖**: ≥ 85%

## 持续集成

测试套件设计为在CI/CD环境中自动运行：

- **快速测试**: 基础功能验证（< 30秒）
- **完整测试**: 包含性能和并发测试（< 5分钟）
- **线程安全测试**: 并发和竞态条件检测

## 故障排除

### 常见问题

1. **测试失败**: 检查依赖模块是否正确安装，确保 `axisfuzzy.core` 模块可用
2. **性能测试超时**: 调整超时设置或检查系统负载
3. **并发测试不稳定**: 增加同步等待时间或检查线程安全实现
4. **Mock对象错误**: 验证Mock配置和期望设置，确保 `spec` 参数正确
5. **Fixture冲突**: 确保使用 `clean_registry` 和 `clean_global_state` fixture 隔离测试环境
6. **边界测试失败**: 检查系统内存限制和数值精度设置
7. **压力测试异常**: 确认系统资源充足，调整测试规模参数

### 测试性能优化建议

- 使用 `pytest -x` 在首次失败时停止
- 使用 `pytest -n auto` 启用并行测试
- 设置合适的超时时间避免长时间等待
- 定期清理测试产生的临时数据
- 对于边界测试，可根据系统配置调整测试参数

### 调试技巧

```bash
# 详细输出模式
pytest test/test_random/ -v -s

# 在第一个失败处停止
pytest test/test_random/ -x

# 显示局部变量
pytest test/test_random/ --tb=long

# 进入调试模式
pytest test/test_random/ --pdb
```

## 贡献指南

### 添加新测试

1. 确定测试类别和所属文件
2. 使用适当的fixture和mock对象
3. 遵循命名约定（`test_功能描述`）
4. 添加详细的docstring说明
5. 确保测试独立性和可重复性

### 测试命名约定

- **测试类**: `Test[功能模块名]`（如 `TestGlobalRandomState`, `TestRandFunction`）
- **测试方法**: `test_[具体功能描述]`（如 `test_multiple_instances_allowed`, `test_single_fuzznum_generation`）
- **Fixture**: `[功能描述]_[类型]` (如 `clean_registry`, `mock_generator`, `sample_rng`)

### 代码质量要求

- 遵循PEP 8代码风格
- 添加类型注解
- 编写清晰的docstring
- 保持测试简洁和专注
- 避免测试间的依赖关系

## 测试架构特点

### 模块化设计
- 每个测试文件专注于单一模块的测试
- 使用共享fixture避免代码重复
- 清晰的测试类和方法组织结构

### 隔离性保证
- `clean_registry` fixture确保注册表状态隔离
- `clean_global_state` fixture确保全局随机状态隔离
- Mock对象提供可控的测试环境

### 并发测试支持
- 专门的线程安全测试类
- `ConcurrentTestHelper` 提供并发测试工具
- 线程屏障确保同步测试执行

### 性能监控
- `PerformanceTimer` 提供精确的性能测量
- 性能基准测试验证系统效率
- 内存使用和执行时间监控

---

**注意**: 本测试套件是AxisFuzzy随机生成系统质量保证的重要组成部分。所有修改都应确保测试通过，并在必要时更新相应的测试用例。当前测试套件包含156个测试方法，覆盖了系统的核心功能、边界条件、错误处理、性能和并发安全性，包括全面的边界测试和性能基准测试。