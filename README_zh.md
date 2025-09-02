# AxisFuzzy 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%3E=2.26-blue?logo=numpy)](https://numpy.org/)
[![构建状态](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![文档](https://img.shields.io/badge/docs-available-informational)](docs/)

> **现代化、模块化、可扩展的 Python 模糊计算库。**

AxisFuzzy 专为模糊逻辑、模糊集合与模糊控制的科研与工程应用而设计，注重性能、可扩展性与用户体验。

---

## ✨ 主要特性
- 🧠 **全面的模糊逻辑运算**（与、或、非等）
- 📈 **多样的隶属函数**（高斯型、三角型、梯形型、自定义）
- ⚡ **高性能的模糊数与模糊数组结构**
- 🧩 **极易扩展的架构**：便捷添加新模糊类型、策略与运算
- 🛠️ **统一的配置系统**，集中管理全局参数与行为
- 🌫️ **灵活的模糊化引擎**，将精确值转换为模糊数
- 🎲 **可复现的随机模糊数生成**，基于 NumPy 向量化加速
- 🔌 **扩展与混入系统**，支持语义和结构特性的扩展
- 🚀 **完全支持基于 NumPy 的批量运算**

---

## 🏗️ 系统架构
AxisFuzzy 由多个核心子系统构建：
- **核心模块**：主数据结构（`Fuzznum`, `Fuzzarray`）、类型注册、运算调度，基于 NumPy 实现高性能计算。
- **配置模块**：集中式配置管理，具备类型安全、参数校验与 JSON 持久化能力。
- **模糊化与隶属函数模块**：灵活的模糊化引擎与丰富的隶属函数体系。
- **扩展与混入模块**：双轨制扩展系统，分别支持类型相关的语义扩展与类型无关的结构扩展。
- **随机模块**：统一、可扩展的随机模糊数生成系统，具备全局种子管理，实现结果可复现。
- **注册模块**：指导用户集成自定义模糊类型、策略与扩展。

---

## 🚀 快速开始

AxisFuzzy 目前处于开发阶段，尚未发布至 PyPI。安装方法：

```bash
git clone https://github.com/YourName/AxisFuzzy.git
cd AxisFuzzy
```

以下是 AxisFuzzy 的简单用法示例：

### 导入 `axisfuzzy`
```python
import axisfuzzy as fuzz
```

### 创建模糊数
```python
a = fuzz.fuzznum((0.5, 0.3))
b = fuzz.fuzznum((0.3, 0.4))

result = a + b
print(result)  # <0.65, 0.12>
```

### 随机生成一个模糊数（类似 NumPy 的 API）
```python
rand_fuzz = fuzz.random.rand()
print(rand_fuzz)  # <0.1247, 0.5132>
```

---

## 📚 文档资源
- `doc/fuzzy-config/01_overview.md`：配置系统
- `doc/fuzzy-core/01_overview.md`：核心架构
- `doc/fuzzy-extension/01_overview.md`：扩展与混入系统
- `doc/fuzzy-fuzzifier/01_overview.md`：模糊化与隶属函数
- `doc/fuzzy-random/01_overview.md`：随机模糊数生成
- `doc/fuzzy-register/01_overview.md`：模糊类型注册指南

整体介绍请参阅 `.github/copilot-instructions.md`。

---

## 📦 依赖与安装

AxisFuzzy 采用模块化依赖体系，用户可根据需求选择仅安装核心依赖，或额外添加分析、开发、文档等功能模块。所有依赖均通过 requirements 文件和 pyproject.toml 统一管理。

### 核心依赖
- `numpy` (>=2.2.6)：核心数值计算库
- `numba` (>=0.61.2)：JIT 编译器，提升性能

### 分析与可视化（可选）
- `pandas` (>=2.3.2)：数据分析
- `matplotlib` (>=3.10.5)：绘图
- `networkx` (>=3.0)：网络分析
- `pydot` (>=1.4.0)：图形可视化

### 完整安装
包含上述所有功能。详见 `requirements/all_requirements.txt`，或使用 `pyproject.toml` 的 `all` 额外依赖组。

### 推荐安装方式
```bash
# 仅核心依赖
pip install -r requirements/core_requirements.txt

# 分析功能
pip install -r requirements/analysis_requirements.txt
```

> **建议**：强烈推荐在虚拟环境（如 `venv` 或 `conda`）中进行依赖管理。

---

## 📝 许可证
AxisFuzzy 采用 MIT 许可证开源。

---

## 🤝 贡献与联系
如有问题、建议或希望贡献代码，请在 GitHub 提交 issue 或 pull request！