# AxisFuzzy 依赖测试套件

本目录包含 AxisFuzzy 项目的依赖安装和功能测试，确保所有必需和可选依赖都正确安装并能正常工作。

## 📁 测试文件结构

```
test_dependencies/
├── README.md                      # 本文档
├── __init__.py                    # 测试套件初始化
├── test_core_dependencies.py      # 核心依赖测试
├── test_optional_dependencies.py  # 可选依赖测试
└── test_lazy_import.py            # 延迟导入功能测试
```

## 🎯 测试目标

### 核心依赖测试 (`test_core_dependencies.py`)

测试项目运行所必需的核心依赖：

- **numpy** (≥2.2.6): 数值计算基础库
- **numba** (≥0.61.2): JIT 编译加速库

**测试内容：**
- 包的可用性检查
- 版本要求验证
- 基本功能测试
- numpy-numba 集成测试

### 可选依赖测试 (`test_optional_dependencies.py`)

测试项目的可选功能依赖，分为三个组：

#### Analysis 组
- **pandas** (≥2.3.2): 数据分析
- **matplotlib** (≥3.10.5): 数据可视化
- **networkx** (≥3.0): 网络分析
- **pydot** (≥1.4.0): 图形可视化

#### Dev 组
- **pytest** (≥8.4.1): 单元测试框架
- **notebook** (≥7.4.5): 交互式开发环境

#### Docs 组
- **sphinx** 及相关扩展: 文档生成工具链

### 延迟导入功能测试 (`test_lazy_import.py`)

测试 AxisFuzzy 的延迟导入机制，确保：

- **延迟导入机制**: 验证组件按需加载的正确性
- **错误处理**: 测试依赖缺失时的优雅降级
- **缓存机制**: 验证导入缓存的有效性
- **IDE 支持**: 确保类型检查和代码补全正常工作
- **集成测试**: 验证各模块间的协同工作

**测试内容：**
- analysis 模块的延迟导入验证
- 各个组件（Model, FuzzyDataFrame, FuzzyPipeline 等）的延迟加载
- pandas 访问器的自动注册机制
- 契约系统和组件基类的集成测试
- 依赖检查功能的完整性验证
- 缓存机制和性能优化验证

## 🚀 运行测试

### 运行所有依赖测试

```bash
# 在项目根目录下运行
pytest tests/test_dependencies/ -v
```

### 运行特定测试

```bash
# 只测试核心依赖
pytest tests/test_dependencies/test_core_dependencies.py -v

# 只测试可选依赖
pytest tests/test_dependencies/test_optional_dependencies.py -v

# 只测试延迟导入功能
pytest tests/test_dependencies/test_lazy_import.py -v
```

### 运行特定依赖组测试

```bash
# 只测试分析依赖
pytest tests/test_dependencies/test_optional_dependencies.py::TestAnalysisDependencies -v

# 只测试开发依赖
pytest tests/test_dependencies/test_optional_dependencies.py::TestDevDependencies -v

# 只测试文档依赖
pytest tests/test_dependencies/test_optional_dependencies.py::TestDocsDependencies -v

# 只测试延迟导入机制
pytest tests/test_dependencies/test_lazy_import.py::TestLazyImportMechanism -v

# 只测试延迟导入错误处理
pytest tests/test_dependencies/test_lazy_import.py::TestLazyImportErrorHandling -v

# 只测试延迟导入缓存机制
pytest tests/test_dependencies/test_lazy_import.py::TestLazyImportCaching -v

# 只测试延迟导入集成功能
pytest tests/test_dependencies/test_lazy_import.py::TestLazyImportIntegration -v
```

## 📋 测试策略

### 核心依赖
- **严格要求**: 所有核心依赖必须安装且版本符合要求
- **功能验证**: 测试基本数值计算和 JIT 编译功能
- **集成测试**: 验证 numpy 和 numba 的协同工作

### 可选依赖
- **宽松策略**: 未安装的可选依赖会跳过测试，不会导致失败
- **功能验证**: 对已安装的依赖进行基本功能测试
- **总结报告**: 生成依赖安装状态的详细报告

### 延迟导入
- **机制验证**: 确保延迟导入按预期工作，组件按需加载
- **错误处理**: 验证依赖缺失时的优雅降级和错误提示
- **性能优化**: 测试缓存机制，确保重复访问的性能
- **IDE 兼容**: 验证类型检查和代码补全功能正常
- **集成测试**: 确保各模块间的延迟导入协同工作

## 📊 测试输出说明

### 成功输出示例

```
✅ numpy 版本: 2.2.6
✅ numpy 版本检查通过: 2.2.6 >= 2.2.6
✅ numpy 基本功能测试通过
✅ numba 版本: 0.61.2
✅ numba 版本检查通过: 0.61.2 >= 0.61.2
✅ numba JIT 功能测试通过
✅ numpy-numba 集成测试通过

=== 核心依赖测试总结 ===
numpy: 2.2.6 ✅
numba: 0.61.2 ✅
所有核心依赖测试通过！
```

### 跳过测试示例

```
SKIPPED [1] pandas 未安装，跳过测试
SKIPPED [1] matplotlib 未安装，跳过测试
```

### 延迟导入测试输出示例

```
=== 延迟导入功能测试总结 ===
✅ analysis 模块延迟导入成功
✅ analysis 模块导出列表验证通过
✅ Model 延迟导入成功: <class 'axisfuzzy.analysis.app.model.Model'>
✅ FuzzyDataFrame 延迟导入成功: <class 'axisfuzzy.analysis.dataframe.frame.FuzzyDataFrame'>
✅ FuzzyPipeline 延迟导入成功: <class 'axisfuzzy.analysis.pipeline.FuzzyPipeline'>
✅ Contract 延迟导入成功: <class 'axisfuzzy.analysis.contracts.base.Contract'>
✅ AnalysisComponent 延迟导入成功: <class 'axisfuzzy.analysis.component.base.AnalysisComponent'>
✅ contract 装饰器延迟导入成功: <function contract at 0x...>
✅ 依赖检查功能正常
✅ pandas 访问器自动注册成功
✅ FuzzyAccessor 正确地未被导出
✅ 延迟导入缓存机制正常
✅ 契约系统集成测试通过

🎉 延迟导入功能测试完成！
```

## 🔧 故障排除

### 核心依赖问题

如果核心依赖测试失败：

1. **安装核心依赖**:
   ```bash
   pip install -r requirements/core_requirements.txt
   ```

2. **检查版本兼容性**:
   ```bash
   pip list | grep -E "numpy|numba"
   ```

3. **升级到最新版本**:
   ```bash
   pip install --upgrade numpy numba
   ```

### 延迟导入问题

如果延迟导入测试失败：

1. **检查模块结构**:
   ```bash
   # 确认关键文件存在
   ls -la axisfuzzy/analysis/__init__.py
   ls -la axisfuzzy/analysis/__init__.pyi
   ls -la axisfuzzy/analysis/pipeline.py
   ```

2. **验证类型存根**:
   ```bash
   # 检查类型存根文件的语法
   python -c "import axisfuzzy.analysis; print('类型存根正常')"
   ```

3. **测试基本导入**:
   ```bash
   # 测试基本延迟导入功能
   python -c "import axisfuzzy; print(axisfuzzy.analysis.Model)"
   ```

4. **清理缓存**:
   ```bash
   # 清理 Python 缓存
   find . -name "__pycache__" -type d -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```

### 可选依赖问题

如果需要安装可选依赖：

1. **安装分析依赖**:
   ```bash
   pip install -r requirements/analysis_requirements.txt
   # 或者
   pip install axisfuzzy[analysis]
   ```

2. **安装开发依赖**:
   ```bash
   pip install -r requirements/dev_requirements.txt
   # 或者
   pip install axisfuzzy[dev]
   ```

3. **安装文档依赖**:
   ```bash
   pip install -r requirements/docs_requirements.txt
   # 或者
   pip install axisfuzzy[docs]
   ```

4. **安装所有依赖**:
   ```bash
   pip install -r requirements/all_requirements.txt
   # 或者
   pip install axisfuzzy[all]
   ```

## 📝 依赖配置文件

项目的依赖配置分布在以下文件中：

- `pyproject.toml`: 主要依赖配置
- `requirements/core_requirements.txt`: 核心依赖列表
- `requirements/analysis_requirements.txt`: 分析依赖列表
- `requirements/dev_requirements.txt`: 开发依赖列表
- `requirements/docs_requirements.txt`: 文档依赖列表
- `requirements/all_requirements.txt`: 所有依赖列表

## 🔄 持续集成

在 CI/CD 流水线中，建议：

1. **核心依赖测试**: 必须通过，否则构建失败
2. **可选依赖测试**: 可以部分跳过，但需要记录状态
3. **延迟导入测试**: 必须通过，确保架构完整性
4. **定期更新**: 定期检查依赖版本更新
5. **缓存清理**: 在测试前清理 Python 缓存以避免干扰

## 📈 扩展测试

如需添加新的依赖测试：

1. 在相应的测试类中添加新的测试方法
2. 遵循现有的测试模式（可用性 → 版本 → 功能）
3. 为可选依赖使用 `pytest.skip()` 处理缺失情况
4. 更新本 README 文档

---

**注意**: 这些测试专注于验证依赖的安装和基本功能，不涉及 AxisFuzzy 自身的业务逻辑测试。