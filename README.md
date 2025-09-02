# AxisFuzzy 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%3E=2.26-blue?logo=numpy)](https://numpy.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Docs](https://img.shields.io/badge/docs-available-informational)](docs/)

> **A modern, modular, and extensible fuzzy computation library for Python.**

AxisFuzzy is designed for research and engineering in fuzzy logic, fuzzy sets, and fuzzy control, with a focus on performance, extensibility, and user experience.

---

## ✨ Key Features
- 🧠 **Comprehensive fuzzy logic operations** (AND, OR, NOT, etc.)
- 📈 **Multiple membership functions** (Gaussian, triangular, trapezoidal, custom)
- ⚡ **High-performance fuzzy number & array structures**
- 🧩 **Highly extensible architecture**: add new fuzzy types, strategies, and operations easily
- 🛠️ **Unified configuration system** for global settings and behaviors
- 🌫️ **Flexible fuzzification engine** for converting crisp values to fuzzy numbers
- 🎲 **Random fuzzy number generation** with reproducibility and NumPy-based vectorization
- 🔌 **Extension & mixin systems** for semantic and structural feature expansion
- 🚀 **Full support for batch operations via NumPy backend**

---

## 🏗️ System Architecture
AxisFuzzy is built around several core subsystems:
- **Core**: Main data structures (`Fuzznum`, `Fuzzarray`), type registration, operation scheduling, and high-performance computation using NumPy.
- **Config**: Centralized configuration management with type safety, validation, and JSON-based persistence.
- **Fuzzifier & Membership**: Flexible engine for fuzzification and a rich set of membership functions.
- **Extension & Mixin**: Dual-track system for semantic (type-dependent) and structural (type-agnostic) feature expansion.
- **Random**: Unified, extensible random fuzzy number generation with seed management for reproducibility.
- **Register**: Guides users to integrate new fuzzy types, strategies, and extensions into the AxisFuzzy ecosystem.

---

## 🚀 Quick Start

AxisFuzzy is currently in development and not yet released on PyPI. To install:

```bash
git clone https://github.com/YourName/AxisFuzzy.git
cd AxisFuzzy
```

Here are some simple examples to get started with AxisFuzzy:

### Import `axisfuzzy`
```python
import axisfuzzy as fuzz
```

### Create fuzzy numbers
```python
a = fuzz.fuzznum((0.5, 0.3))
b = fuzz.fuzznum((0.3, 0.4))

result = a + b
print(result)  # <0.65, 0.12>
```

### Random fuzzy number generation (similar to NumPy API)
```python
rand_fuzz = fuzz.random.rand()
print(rand_fuzz)  # <0.1247, 0.5132>
```

---

## 📚 Documentation
- `doc/fuzzy-config/01_overview.md`: Configuration system
- `doc/fuzzy-core/01_overview.md`: Core architecture
- `doc/fuzzy-extension/01_overview.md`: Extension & mixin systems
- `doc/fuzzy-fuzzifier/01_overview.md`: Fuzzification & membership functions
- `doc/fuzzy-random/01_overview.md`: Random fuzzy number generation
- `doc/fuzzy-register/01_overview.md`: Fuzzy type registration guide

For an overall introduction, see `.github/copilot-instructions.md`.

---

## 📦 Dependencies & Installation

AxisFuzzy uses a modular dependency system. You can install only the core, or add analysis, development, or documentation features as needed. All dependencies are managed via requirements files and pyproject.toml.

### Core Dependencies
- `numpy` (>=2.2.6): Core numerical computation
- `numba` (>=0.61.2): JIT compilation for performance

### Analysis & Visualization (optional)
- `pandas` (>=2.3.2): Data analysis
- `matplotlib` (>=3.10.5): Plotting
- `networkx` (>=3.0): Network analysis
- `pydot` (>=1.4.0): Graph visualization

### Recommended Installation
```bash
# Core only
pip install -r requirements/core_requirements.txt

# Analysis features
pip install -r requirements/analysis_requirements.txt
```

> **Note:** It is strongly recommended to use a virtual environment (e.g., `venv` or `conda`) for dependency management.

---

## 📝 License
AxisFuzzy is released under the MIT License.

---

## 🤝 Contributing & Contact
Questions, suggestions, or want to contribute? Open an issue or pull request on GitHub!