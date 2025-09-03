# AxisFuzzy 依赖测试套件

本目录包含 AxisFuzzy 项目的依赖安装和功能测试，确保所有必需和可选依赖都正确安装并能正常工作。

## 📁 测试文件结构

```
test_dependencies/
├── README.md                      # 本文档
├── test_core_dependencies.py      # 核心依赖测试
└── test_optional_dependencies.py  # 可选依赖测试
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
```

### 运行特定依赖组测试

```bash
# 只测试分析依赖
pytest tests/test_dependencies/test_optional_dependencies.py::TestAnalysisDependencies -v

# 只测试开发依赖
pytest tests/test_dependencies/test_optional_dependencies.py::TestDevDependencies -v

# 只测试文档依赖
pytest tests/test_dependencies/test_optional_dependencies.py::TestDocsDependencies -v
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
3. **定期更新**: 定期检查依赖版本更新

## 📈 扩展测试

如需添加新的依赖测试：

1. 在相应的测试类中添加新的测试方法
2. 遵循现有的测试模式（可用性 → 版本 → 功能）
3. 为可选依赖使用 `pytest.skip()` 处理缺失情况
4. 更新本 README 文档

---

**注意**: 这些测试专注于验证依赖的安装和基本功能，不涉及 AxisFuzzy 自身的业务逻辑测试。