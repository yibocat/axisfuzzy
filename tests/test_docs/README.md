# 文档测试套件

这个目录包含了 AxisFuzzy 项目文档系统的完整测试套件。测试套件验证基于 Sphinx 的文档系统的各个方面，确保文档的质量、完整性和可用性。

## 测试文件概述

### 🔧 `test_extensions.py`
**Sphinx 扩展测试**
- 验证 Sphinx 核心功能的可用性
- 检查 `docs/conf.py` 中配置的扩展是否正确安装
- 测试内置扩展和第三方扩展的导入
- 验证主题扩展的可用性
- 生成扩展安装状态报告

### 🏗️ `test_build.py`
**文档构建测试**
- 验证文档目录结构和配置文件
- 执行 Sphinx HTML 构建过程
- 检查构建输出文件的完整性
- 验证 HTML 结构和内容质量
- 测试静态文件生成
- 监控构建警告和错误
- 评估构建性能
- 测试 API 文档生成和搜索功能

### ⚙️ `test_config.py`
**配置验证测试**
- 验证 `docs/conf.py` 配置文件的正确性
- 检查项目信息、版本、扩展配置
- 验证 HTML 主题和路径设置
- 测试 autodoc、源文件、语言配置
- 检查与 `pyproject.toml` 的一致性
- 验证扩展依赖关系

### 📄 `test_content.py`
**内容质量测试**
- 验证主页和重要页面的内容
- 检查导航结构和链接有效性
- 测试内部链接的完整性
- 抽样检查外部链接的可用性
- 验证代码块的语法正确性
- 检查图片和媒体文件
- 基本的可访问性测试
- 分析文档结构和完整性

## 运行测试

### 运行所有文档测试
```bash
# 在项目根目录下运行
pytest test/test_docs/ -v
```

### 运行特定测试文件
```bash
# 测试扩展
pytest test/test_docs/test_extensions.py -v

# 测试构建过程
pytest test/test_docs/test_build.py -v

# 测试配置
pytest test/test_docs/test_config.py -v

# 测试内容质量
pytest test/test_docs/test_content.py -v
```

### 运行特定测试类或方法
```bash
# 测试特定类
pytest test/test_docs/test_extensions.py::TestSphinxExtensions -v

# 测试特定方法
pytest test/test_docs/test_build.py::TestDocumentationBuild::test_sphinx_html_build -v
```

### 生成详细报告
```bash
# 运行测试并生成详细输出
pytest test/test_docs/ -v -s

# 只运行总结测试（快速检查）
pytest test/test_docs/ -k "summary" -v
```

## 测试依赖

### 必需依赖
- `pytest`: 测试框架
- `sphinx`: 文档生成工具
- `beautifulsoup4`: HTML 解析（用于内容测试）

### 可选依赖
这些依赖在 `docs/conf.py` 中配置，测试会检查它们的可用性：
- `myst-parser`: Markdown 支持
- `sphinx-copybutton`: 代码复制按钮
- `pydata-sphinx-theme`: 主题
- `sphinx-design`: 设计组件
- `sphinx.ext.autodoc`: API 文档生成
- `sphinx.ext.viewcode`: 源代码链接
- `sphinx.ext.intersphinx`: 交叉引用

## 测试策略

### 🚀 快速检查
如果你只想快速验证文档系统的基本功能：
```bash
pytest test/test_docs/test_extensions.py::test_extensions_summary -v
pytest test/test_docs/test_config.py::test_config_summary -v
```

### 🔍 完整验证
如果你想进行完整的文档质量检查：
```bash
pytest test/test_docs/ -v
```

### 🏗️ 构建测试
如果你修改了文档内容或配置，重点测试构建过程：
```bash
pytest test/test_docs/test_build.py -v
pytest test/test_docs/test_content.py -v
```

## 测试输出说明

### ✅ 成功标志
- 扩展正确安装和配置
- 文档构建无错误
- 链接和内容完整
- 配置文件正确

### ⚠️ 警告标志
- 可选扩展未安装（不影响基本功能）
- 外部链接暂时不可用
- 非关键配置缺失

### ❌ 错误标志
- 必需扩展缺失
- 构建失败
- 配置文件错误
- 重要内容缺失

## 故障排除

### 扩展相关问题
```bash
# 安装缺失的扩展
pip install sphinx myst-parser sphinx-copybutton pydata-sphinx-theme sphinx-design
```

### 构建相关问题
```bash
# 清理构建缓存
rm -rf docs/_build/

# 手动构建文档
cd docs/
sphinx-build -b html . _build/html
```

### 配置相关问题
- 检查 `docs/conf.py` 语法
- 验证 `pyproject.toml` 中的项目信息
- 确保路径配置正确

## 自定义测试

你可以根据项目需要扩展这些测试：

1. **添加新的扩展测试**: 在 `test_extensions.py` 中添加新的扩展检查
2. **增强内容验证**: 在 `test_content.py` 中添加特定的内容检查
3. **自定义构建测试**: 在 `test_build.py` 中添加特定的构建验证
4. **扩展配置测试**: 在 `test_config.py` 中添加新的配置验证

## 持续集成

这些测试适合集成到 CI/CD 流水线中：

```yaml
# GitHub Actions 示例
- name: Test Documentation
  run: |
    pip install -e .[docs]
    pytest test/test_docs/ -v
```

## 性能考虑

- **扩展测试**: 快速（< 10秒）
- **配置测试**: 快速（< 5秒）
- **构建测试**: 中等（30-60秒，取决于文档大小）
- **内容测试**: 中等（20-40秒，取决于页面数量）

对于快速反馈，建议先运行扩展和配置测试，然后在需要时运行构建和内容测试。

---

**注意**: 这个测试套件替换了之前的 `check_extensions.py` 和 `test_docs_installation.py` 文件，提供了更全面、更结构化的文档质量保证。