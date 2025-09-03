# AxisFuzzy Core 模块测试说明

本目录包含 `axisfuzzy.core` 模块的测试文件，测试核心功能包括 T-范数、模糊数策略、模糊数组、注册表系统、操作调度器等关键组件。

## 测试文件结构

### 1. `conftest.py` - 测试配置和夹具
- **功能**: 提供测试环境配置和共享夹具
- **内容**:
  - 项目路径配置，确保 `axisfuzzy` 模块可被导入
  - 全局配置状态重置夹具，确保测试间相互独立
  - `qrofn` 类型注册确保夹具，支持需要真实模糊数类型的测试
  - 示例 `qrofn` 实例创建夹具

### 2. `test_triangular.py` - T-范数框架测试
- **功能**: 测试 `triangular.py` 中的 T-范数实现
- **覆盖范围**: 
  - 12种 T-范数类型的数学正确性验证
  - 参数验证和边界条件测试
  - 代数性质测试（交换律、结合律、单调性）
  - Archimedean 性质和对偶性测试
  - 标量和数组输入的行为测试
  - 注册表功能和自定义范数注册
  - 性能测试和并发安全性（可选）

### 3. `test_fuzznums_basic.py` - 基础概念测试
- **功能**: 测试基本的设计模式和概念，无复杂依赖
- **覆盖范围**:
  - `FuzznumStrategy` 基础概念（属性验证、转换、回调）
  - 工厂模式、门面模式、代理模式概念测试
  - 注册表模式和验证链概念测试
  - 适合快速迭代开发的轻量级测试

### 4. `test_fuzznums.py` - 模糊数完整功能测试
- **功能**: 测试 `base.py` 和 `fuzznums.py` 的完整功能
- **依赖**: 需要 `qrofn` 类型已在注册表中注册
- **覆盖范围**:
  - `FuzznumStrategy` 的属性声明、验证、转换、回调系统
  - `Fuzznum` 的门面模式、动态代理、序列化功能
  - `fuzznum` 工厂函数的各种创建方式
  - 与注册表系统的集成和边界情况处理

### 5. `test_fuzzarray_backend.py` - 模糊数组和后端测试
- **功能**: 测试 `fuzzarray.py` 和 `backend.py` 的功能
- **覆盖范围**:
  - `FuzzarrayBackend` 抽象基类的接口和实现测试
  - `Fuzzarray` 的创建、索引、切片、迭代功能
  - 数组属性管理和元素访问测试
  - 运算符重载结构测试（算术、比较、逻辑）
  - 后端委托机制和与真实模糊数类型的集成

### 6. `test_registry.py` - 注册表系统测试
- **功能**: 测试 `registry.py` 的注册表系统
- **覆盖范围**:
  - `FuzznumRegistry` 单例模式和线程安全性
  - 策略和后端的注册、注销、检索功能
  - 事务支持和回滚机制
  - 观察者模式和批量操作
  - 装饰器功能和系统内省
  - 健康状态检查和统计信息

### 7. `test_operation_dispatcher.py` - 操作调度器测试
- **功能**: 测试 `operation.py` 和 `dispatcher.py` 的操作系统
- **覆盖范围**:
  - `OperationMixin` 抽象接口和操作执行
  - `OperationScheduler` 注册和调度机制
  - 操作调度器路由逻辑和类型分发
  - 性能监控和 T-范数配置
  - 错误处理、验证和线程安全性
  - 装饰器注册和集成测试

## 测试运行方法

### 环境要求
- Python 3.8+
- pytest
- numpy
- axisfuzzy 包已安装

### 运行所有测试
```bash
# 在项目根目录下运行
pytest test/test_core/ -v
```

### 运行特定测试文件
```bash
# 运行 T-范数测试
pytest test/test_core/test_triangular.py -v

# 运行基础概念测试（轻量级，快速迭代）
pytest test/test_core/test_fuzznums_basic.py -v

# 运行模糊数完整功能测试
pytest test/test_core/test_fuzznums.py -v

# 运行模糊数组和后端测试
pytest test/test_core/test_fuzzarray_backend.py -v

# 运行注册表系统测试
pytest test/test_core/test_registry.py -v

# 运行操作调度器测试
pytest test/test_core/test_operation_dispatcher.py -v
```

### 运行特定测试类或方法
```bash
# 运行特定测试类
pytest test/test_core/test_triangular.py::TestTNormMath -v

# 运行特定测试方法
pytest test/test_core/test_triangular.py::TestTNormMath::test_commutativity -v

# 运行性能测试（可选）
pytest test/test_core/test_triangular.py -m performance -v
```

## 测试策略

### 1. 分层测试架构
- **配置层**: `conftest.py` - 测试环境配置和共享夹具
- **基础层**: `test_fuzznums_basic.py` - 验证基本设计模式和概念
- **功能层**: `test_fuzznums.py` - 验证完整的模糊数功能
- **数学层**: `test_triangular.py` - 验证 T-范数数学正确性
- **集成层**: `test_fuzzarray_backend.py`, `test_registry.py`, `test_operation_dispatcher.py` - 验证组件集成

### 2. 依赖管理策略
- 使用 `conftest.py` 统一管理测试依赖和夹具
- 通过 `reset_global_config` 确保测试间相互独立
- 通过 `ensure_qrofn_registered` 确保真实模糊数类型可用
- 避免测试间状态污染和依赖顺序问题

### 3. 测试标记和分类
- `@pytest.mark.performance` - 性能测试，默认跳过
- `@pytest.mark.slow` - 慢速测试，可选择性运行
- `@pytest.mark.integration` - 集成测试标记

## 开发建议

### 1. 开发阶段
- 使用 `test_fuzznums_basic.py` 进行快速迭代
- 基础测试可以快速验证逻辑正确性

### 2. 集成阶段
- 使用 `test_fuzznums.py` 验证真实组件的协作
- 确保注册表系统正常工作

### 3. 发布前
- 运行所有测试确保完整性
- 检查测试覆盖率

## 故障排除

### 常见问题

1. **导入错误**: 
   - 确保在项目根目录运行测试
   - 检查 `conftest.py` 中的路径配置

2. **注册表状态污染**: 
   - 检查 `reset_global_config` 夹具是否正常工作
   - 确认测试间注册表状态被正确重置

3. **数值精度问题**: 
   - T-范数测试中使用 `np.allclose` 进行浮点比较
   - 注意 `RuntimeWarning` 可能是正常的数值边界行为

4. **并发测试失败**: 
   - 检查线程安全性实现
   - 验证注册表的线程安全机制

5. **模糊数类型未注册**: 
   - 确认 `ensure_qrofn_registered` 夹具正常工作
   - 检查 `qrofn` 类型是否正确导入和注册

### 调试技巧

```bash
# 详细输出和实时打印
pytest test/test_core/ -v -s

# 只运行失败的测试
pytest test/test_core/ --lf

# 在第一个失败时停止
pytest test/test_core/ -x

# 显示局部变量
pytest test/test_core/ -l

# 跳过性能测试
pytest test/test_core/ -m "not performance"

# 只运行特定标记的测试
pytest test/test_core/ -m "integration"
```

## 测试覆盖率

```bash
# 生成覆盖率报告
pytest test/test_core/ --cov=axisfuzzy.core --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html

# 生成终端覆盖率报告
pytest test/test_core/ --cov=axisfuzzy.core --cov-report=term-missing
```

## 扩展测试

### 添加新的 T-范数
1. 在 `triangular.py` 中实现新的 T-范数类
2. 在 `test_triangular.py` 中添加对应的测试
3. 确保测试覆盖数学性质验证（交换律、结合律、单调性等）
4. 添加边界条件和数值稳定性测试

### 添加新的模糊数类型
1. 实现新的 `FuzznumStrategy` 子类
2. 在注册表中注册新类型
3. 添加相应的测试用例到 `test_fuzznums.py`
4. 更新 `conftest.py` 中的相关夹具

### 添加新的后端实现
1. 实现新的 `FuzzarrayBackend` 子类
2. 在 `test_fuzzarray_backend.py` 中添加测试
3. 确保与现有组件的兼容性

### 添加新的测试文件
1. 创建新的测试文件，遵循命名约定
2. 在 `conftest.py` 中添加必要的夹具
3. 更新此 README 文档

## 性能测试

性能测试默认被跳过，可通过以下方式运行：

```bash
# 运行所有性能测试
pytest test/test_core/ -m performance -v

# 运行特定文件的性能测试
pytest test/test_core/test_triangular.py -m performance -v
```

### 性能测试内容
- T-范数计算性能（大规模数组操作）
- 模糊数创建和转换性能
- 注册表操作性能
- 并发安全性和性能测试
- 内存使用效率测试

## 测试环境要求

### 依赖包
- `pytest` >= 6.0
- `numpy` >= 1.20
- `pytest-cov` (用于覆盖率测试)
- `pytest-xdist` (用于并行测试)

### Python 版本
- Python >= 3.8

### 可选依赖
- `pytest-benchmark` (用于性能基准测试)
- `pytest-mock` (用于模拟测试)

## 贡献指南

### 添加新测试
1. 遵循现有的测试结构和命名约定
2. 使用适当的夹具和标记
3. 确保测试独立性和可重复性
4. 添加适当的文档字符串和注释
5. 考虑测试的分层归属（基础/功能/集成/数学）

### 测试质量要求
- 测试应该快速、可靠、独立
- 使用清晰的断言消息和错误描述
- 覆盖正常路径、边界情况和错误处理
- 包含适当的数值稳定性测试
- 遵循 DRY 原则，合理使用夹具和辅助函数

### 测试文档要求
- 每个测试文件应有清晰的模块文档字符串
- 复杂测试方法应包含详细的文档字符串
- 使用 Numpy 风格的文档字符串格式
- 在 README.md 中更新相应的测试说明

## 总结

`test_core` 目录提供了 AxisFuzzy 核心模块的全面测试覆盖，包括：

- **数学正确性**: T-范数的严格数学验证
- **功能完整性**: 模糊数和模糊数组的完整功能测试
- **系统集成**: 注册表、调度器等系统组件的集成测试
- **性能保证**: 可选的性能测试确保系统效率
- **开发支持**: 分层测试架构支持快速迭代开发

通过合理的测试策略和完善的测试工具，确保 AxisFuzzy 核心模块的质量和可靠性。
