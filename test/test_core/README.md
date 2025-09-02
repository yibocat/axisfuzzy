# AxisFuzzy Core 模块测试说明

本目录包含 `axisfuzzy.core` 模块的测试文件，测试核心功能包括 `FuzznumStrategy`、`Fuzznum`、`Fuzzarray` 等关键组件。

## 测试文件结构

### 1. `test_triangular.py` - T-范数测试
- **功能**: 测试 `triangular.py` 中的 T-范数实现
- **覆盖范围**: 
  - 12种 T-范数类型的数学正确性
  - q-阶扩展功能
  - 边界条件和数值稳定性
  - 性能测试（可选）
  - 并发安全性

### 2. `test_fuzznums.py` - 完整功能测试
- **功能**: 测试 `base.py` 和 `fuzznums.py` 的完整功能
- **依赖**: 需要 `qrofn` 类型已在注册表中注册
- **覆盖范围**:
  - `FuzznumStrategy` 的属性声明、验证、转换、回调
  - `Fuzznum` 的门面模式、动态代理、序列化
  - 与注册表系统的集成

### 3. `test_fuzznums_comprehensive.py` - 模拟测试
- **功能**: 使用模拟对象测试核心概念，避免复杂依赖
- **特点**: 不依赖真实的注册表或模糊数类型
- **覆盖范围**:
  - 策略模式的核心概念
  - 门面模式的核心概念
  - 验证链和转换链
  - 集成测试概念

### 4. `test_fuzznums_basic.py` - 基础概念测试
- **功能**: 测试基本的设计模式和概念
- **特点**: 最简单的测试，验证基本逻辑

## 测试运行方法

### 环境要求
- Python 3.8+
- pytest
- numpy
- axisfuzzy 包已安装

### 运行所有测试
```bash
# 在项目根目录运行
pytest test/test_core/ -v
```

### 运行特定测试文件
```bash
# 运行 T-范数测试
pytest test/test_core/test_triangular.py -v

# 运行 Fuzznum 完整功能测试（需要注册表）
pytest test/test_core/test_fuzznums.py -v

# 运行模拟测试（推荐用于开发阶段）
pytest test/test_core/test_fuzznums_comprehensive.py -v

# 运行基础概念测试
pytest test/test_core/test_fuzznums_basic.py -v
```

### 运行特定测试类
```bash
# 运行 FuzznumStrategy 测试
pytest test/test_core/test_fuzznums_comprehensive.py::TestMockFuzznumStrategy -v

# 运行 Fuzznum 测试
pytest test/test_core/test_fuzznums_comprehensive.py::TestMockFuzznum -v
```

### 运行特定测试方法
```bash
# 运行特定测试方法
pytest test/test_core/test_fuzznums_comprehensive.py::TestMockFuzznumStrategy::test_attribute_validation -v
```

## 测试策略说明

### 1. 分层测试策略
- **概念层**: 使用模拟对象测试设计模式和核心概念
- **集成层**: 测试组件间的协作
- **功能层**: 测试完整的用户功能

### 2. 依赖管理
- **模拟测试**: 不依赖外部系统，适合单元测试
- **集成测试**: 依赖注册表系统，适合系统测试
- **跳过策略**: 当依赖不可用时优雅地跳过测试

### 3. 测试数据
- **边界值**: 测试 0、1、-1 等边界情况
- **异常值**: 测试超出范围的值
- **正常值**: 测试典型的有效输入

## 开发建议

### 1. 开发阶段
- 使用 `test_fuzznums_comprehensive.py` 进行快速迭代
- 模拟对象可以快速验证逻辑正确性

### 2. 集成阶段
- 使用 `test_fuzznums.py` 验证真实组件的协作
- 确保注册表系统正常工作

### 3. 发布前
- 运行所有测试确保完整性
- 检查测试覆盖率

## 故障排除

### 常见问题

#### 1. 导入错误
```
ModuleNotFoundError: No module named 'axisfuzzy'
```
**解决方案**: 确保项目根目录在 Python 路径中，或安装 axisfuzzy 包

#### 2. 注册表错误
```
ValueError: Unsupported strategy mtype: 'qrofn'
```
**解决方案**: 确保 qrofn 类型已在注册表中注册，或使用模拟测试

#### 3. 配置错误
```
AttributeError: 'NoneType' object has no attribute 'DEFAULT_MTYPE'
```
**解决方案**: 检查配置文件是否正确加载

### 调试技巧

1. **使用 -s 标志**: `pytest -s` 显示 print 输出
2. **使用 -x 标志**: `pytest -x` 在第一个失败时停止
3. **使用 --tb=short**: 显示简短的错误回溯
4. **使用 --lf**: 只运行上次失败的测试

## 扩展测试

### 添加新的测试
1. 在相应的测试文件中添加新的测试方法
2. 遵循命名约定：`test_<功能名>`
3. 添加适当的文档字符串
4. 确保测试是独立的和可重复的

### 添加新的测试文件
1. 创建新的测试文件，遵循命名约定
2. 在 `conftest.py` 中添加必要的夹具
3. 更新此 README 文档

## 性能测试

### 启用性能测试
```bash
# 设置环境变量启用性能测试
export AXISFUZZY_RUN_PERF=1
export AXISFUZZY_PERF_SIZE=100000

# 运行性能测试
pytest test/test_core/test_triangular.py -k Performance -v
```

### 性能基准
- 大型数组的吞吐量测试
- 内存使用分析
- 并发性能测试

## 总结

本测试套件提供了全面的测试覆盖，从基本概念到完整功能，确保 `axisfuzzy.core` 模块的可靠性和正确性。建议开发者在不同阶段使用相应的测试策略，以平衡开发效率和测试质量。
