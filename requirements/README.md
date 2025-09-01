# Requirements Files

本目录包含了 AxisFuzzy 项目的各种依赖配置文件，根据不同的使用场景进行分类。

## 文件说明

### `core_requirements.txt`
**核心依赖**：包含 AxisFuzzy 运行所必需的最小依赖集合
- `numpy`: 核心数值计算库
- `numba`: JIT 编译器，用于性能优化

### `analysis_requirements.txt`
**分析功能依赖**：用于数据分析、可视化和网络分析
- `pandas`: 数据处理和分析
- `matplotlib`: 图形绘制
- `networkx`: 网络分析
- `pydot`: 图形可视化

### `dev_requirements.txt`
**开发环境依赖**：用于开发、测试和交互式编程
- `pytest`: 单元测试框架
- `notebook`: Jupyter notebook 支持

### `docs_requirements.txt`
**文档生成依赖**：用于构建项目文档
- `sphinx`: 文档生成工具
- 各种 Sphinx 扩展和主题

### `all_requirements.txt`
**完整依赖**：包含所有上述依赖的完整集合

## 使用方法

### 安装核心依赖（最小安装）
```bash
pip install -r requirements/core_requirements.txt
```

### 安装开发环境
```bash
pip install -r requirements/core_requirements.txt
pip install -r requirements/dev_requirements.txt
```

### 安装分析功能
```bash
pip install -r requirements/core_requirements.txt
pip install -r requirements/analysis_requirements.txt
```

### 安装完整功能
```bash
pip install -r requirements/all_requirements.txt
```

### 构建文档
```bash
pip install -r requirements/core_requirements.txt
pip install -r requirements/docs_requirements.txt
```

## 与 pyproject.toml 的对应关系

这些 requirements 文件与 `pyproject.toml` 中的可选依赖组对应：

- `core_requirements.txt` ↔ `[project.dependencies]`
- `analysis_requirements.txt` ↔ `[project.optional-dependencies.analysis]`
- `dev_requirements.txt` ↔ `[project.optional-dependencies.dev]`
- `docs_requirements.txt` ↔ `[project.optional-dependencies.docs]`
- `all_requirements.txt` ↔ `[project.optional-dependencies.all]`

## 注意事项

1. **版本兼容性**：所有版本号与 `pyproject.toml` 保持一致
2. **依赖顺序**：建议先安装核心依赖，再安装其他功能依赖
3. **虚拟环境**：强烈建议在虚拟环境中安装依赖
4. **更新维护**：当 `pyproject.toml` 更新时，需要同步更新对应的 requirements 文件