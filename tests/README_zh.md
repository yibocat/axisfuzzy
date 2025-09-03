# AxisFuzzy 测试套件

本目录包含 AxisFuzzy 项目的完整测试套件，采用模块化设计，支持编程式测试执行和灵活的测试配置。

## 📁 目录结构

```
tests/
├── __init__.py                 # 主测试入口和运行函数
├── README.md                   # 本文档
├── test_config/                # 配置系统测试
│   ├── __init__.py
│   └── test_*.py
├── test_core/                  # 核心功能测试
│   ├── __init__.py
│   └── test_*.py
├── test_dependencies/          # 依赖验证测试
│   ├── __init__.py
│   ├── README.md
│   ├── test_core_dependencies.py
│   └── test_optional_dependencies.py
├── test_docs/                  # 文档测试
│   ├── __init__.py
│   └── test_*.py
├── test_fuzzifier/             # 模糊化器测试
│   ├── __init__.py
│   └── test_*.py
├── test_membership/            # 隶属函数测试
│   ├── __init__.py
│   └── test_*.py
├── test_mixin/                 # Mixin 功能测试
│   ├── __init__.py
│   └── test_*.py
└── test_random/                # 随机数生成测试
    ├── __init__.py
    └── test_*.py
```

## 🚀 快速开始

### 方法一：编程式执行（推荐）

```python
# 在项目根目录下的 Python 环境中
from tests import run_all_tests, run_core_tests, run_dependency_tests

# 运行所有核心测试
run_core_tests()

# 运行依赖验证测试
run_dependency_tests()

# 运行完整测试套件
run_all_tests(include_docs=True)

# 运行快速测试（依赖 + 核心）
from tests import run_quick_tests
run_quick_tests()
```

### 方法二：命令行执行

```bash
# 在项目根目录下

# 运行快速测试
python -m tests --quick

# 运行完整测试（包含文档）
python -m tests --docs

# 仅运行依赖测试
python -m tests --deps-only

# 仅运行核心测试
python -m tests --core-only

# 静默模式
python -m tests --quiet
```

### 方法三：传统 pytest 执行

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_core/ -v
pytest tests/test_dependencies/ -v

# 运行特定测试文件
pytest tests/test_core/test_fuzznum.py -v
```

## 📊 测试分类

### 1. 依赖测试 (`test_dependencies/`)

**优先级：最高** 🔴

验证项目依赖是否正确安装和配置：

- **核心依赖测试**：验证 `numpy`、`scipy`、`matplotlib` 等核心库
- **可选依赖测试**：检查分析、开发、文档构建相关的可选依赖
- **集成测试**：验证依赖间的兼容性和集成

```python
from tests import run_dependency_tests
result = run_dependency_tests()
```

### 2. 核心功能测试 (`test_core/`, `test_config/`, `test_fuzzifier/`, `test_membership/`, `test_mixin/`, `test_random/`)

**优先级：高** 🟡

验证 AxisFuzzy 的核心功能：

- **配置系统**：测试配置加载、验证和管理
- **核心数据结构**：测试 `Fuzznum` 和 `Fuzzarray`
- **模糊化器**：测试各种模糊化算法
- **隶属函数**：测试隶属函数的实现和计算
- **Mixin 功能**：测试混入类的功能
- **随机数生成**：测试模糊随机数生成

```python
from tests import run_core_tests
result = run_core_tests()
```

### 3. 文档测试 (`test_docs/`)

**优先级：中** 🟢

验证项目文档的完整性和正确性：

- **文档构建**：验证 Sphinx 文档能否正确构建
- **API 文档**：检查 API 文档的完整性
- **示例代码**：验证文档中的代码示例

```python
from tests import run_docs_tests
result = run_docs_tests()
```

### 4. 分析模块测试 (`test_analysis/`) - 未来扩展

**优先级：低** 🔵

为未来的分析功能预留的测试模块。

## 🎯 测试执行策略

### 推荐的测试流程

1. **开发阶段**：使用 `run_quick_tests()` 进行快速验证
2. **提交前**：使用 `run_all_tests()` 进行完整验证
3. **CI/CD**：使用 `run_all_tests(include_docs=True)` 进行全面测试
4. **依赖更新后**：优先运行 `run_dependency_tests()`

### 测试优先级

```
依赖测试 (Priority 1) → 核心测试 (Priority 2) → 文档测试 (Priority 3)
```

如果依赖测试失败，后续测试将自动跳过，确保测试效率。

## 🔧 高级用法

### 测试参数说明

所有测试函数都支持以下参数：

- `verbose` (bool): 是否显示详细输出，默认 `True`
- `extra_args` (list): 传递给 pytest 的额外参数
- `include_docs` (bool): 是否包含文档测试（仅适用于 `run_all_tests`）

### pytest 参数示例

```python
from tests import run_core_tests

# 显示详细输出，在第一个失败时停止
run_core_tests(verbose=True, extra_args=['-x', '--tb=short'])

# 并行运行测试（需要安装 pytest-xdist）
run_core_tests(extra_args=['-n', '4'])

# 只运行特定标记的测试
run_core_tests(extra_args=['-m', 'slow'])

# 生成覆盖率报告（需要安装 pytest-cov）
run_core_tests(extra_args=['--cov=axisfuzzy', '--cov-report=html'])
```

### 高级配置模式

```python
# 开发模式：快速反馈
dev_config = {
    'verbose': True,
    'extra_args': ['--tb=short', '-x']  # 简短回溯，快速失败
}

# 集成模式：全面测试
integration_config = {
    'verbose': True,
    'extra_args': ['--tb=short', '--maxfail=5']  # 最多允许 5 个失败
}

# CI 模式：严格测试
ci_config = {
    'verbose': False,
    'extra_args': ['--tb=line', '--strict-markers']  # 简洁输出，严格标记
}

# 使用配置
from tests import run_all_tests
result = run_all_tests(include_docs=True, **ci_config)
```

### 编程式测试结果处理

#### 基本处理

```python
from tests import run_core_tests, run_dependency_tests

# 运行测试并处理结果
core_success = run_core_tests()
deps_success = run_dependency_tests()

if core_success and deps_success:
    print("✅ 所有关键测试通过")
    # 执行部署或其他操作
else:
    print("❌ 测试失败，停止流程")
    # 发送通知或记录日志
```

#### 高级结果分析

```python
import time
from tests import run_all_tests, run_quick_tests, run_core_tests

def comprehensive_test_analysis():
    """综合测试分析"""
    results = {}
    
    # 1. 快速测试
    start_time = time.time()
    results['quick'] = {
        'success': run_quick_tests(),
        'duration': time.time() - start_time
    }
    
    # 2. 核心测试（详细模式）
    start_time = time.time()
    results['core_detailed'] = {
        'success': run_core_tests(verbose=True, extra_args=['--tb=short']),
        'duration': time.time() - start_time
    }
    
    # 3. 完整测试
    start_time = time.time()
    results['full'] = {
        'success': run_all_tests(include_docs=True),
        'duration': time.time() - start_time
    }
    
    # 分析结果
    total_duration = sum(r['duration'] for r in results.values())
    success_count = sum(1 for r in results.values() if r['success'])
    
    print(f"\n📊 测试分析报告:")
    print(f"总耗时: {total_duration:.2f}s")
    print(f"成功率: {success_count}/{len(results)}")
    
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{test_name}: {status} ({result['duration']:.2f}s)")
    
    return results

# 使用示例
analysis = comprehensive_test_analysis()
```

### 环境变量配置

通过环境变量可以灵活控制测试行为，无需修改代码：

```python
import os
from tests import run_all_tests

# 通过环境变量控制测试行为
test_config = {
    'include_docs': os.getenv('AXISFUZZY_TEST_DOCS', 'false').lower() == 'true',
    'verbose': os.getenv('AXISFUZZY_TEST_VERBOSE', 'true').lower() == 'true',
    'extra_args': os.getenv('AXISFUZZY_TEST_ARGS', '').split() if os.getenv('AXISFUZZY_TEST_ARGS') else None
}

# 运行测试
result = run_all_tests(**test_config)
```

#### 环境变量说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `AXISFUZZY_TEST_DOCS` | `false` | 是否包含文档测试 |
| `AXISFUZZY_TEST_VERBOSE` | `true` | 是否显示详细输出 |
| `AXISFUZZY_TEST_ARGS` | `` | 额外的 pytest 参数（空格分隔） |
| `AXISFUZZY_TEST_MODE` | `dev` | 测试模式（dev/integration/ci） |

#### 使用示例

```bash
# 设置环境变量
export AXISFUZZY_TEST_DOCS=true
export AXISFUZZY_TEST_VERBOSE=false
export AXISFUZZY_TEST_ARGS="--tb=short --maxfail=1"

# 运行测试
python -c "from tests import run_all_tests; run_all_tests()"

# 或者在一行中设置
AXISFUZZY_TEST_DOCS=true AXISFUZZY_TEST_VERBOSE=false python -m tests --docs
```

### 自动化测试脚本

对于复杂的自动化测试需求，我们提供了专门的自动化测试指南。请参考：

- **[自动化测试指南](automation_guide.md)** - 详细的自动化测试脚本和配置
- **[CI/CD 集成](automation_guide.md#cicd-集成)** - 持续集成配置示例
- **[Docker 测试环境](automation_guide.md#docker-环境)** - 容器化测试环境

## 📈 测试报告

每次测试执行都会提供详细的报告，包括：

- **执行时间**：每个测试类别的耗时
- **通过率**：测试通过的数量和比例
- **失败详情**：失败测试的具体信息
- **依赖状态**：依赖库的安装和版本信息

示例输出：

```
🚀 执行命令: python -m pytest tests/test_dependencies/ -v -s
📁 工作目录: /path/to/AxisFuzzy
============================================================

✅ 依赖测试通过 (耗时: 2.34s)

============================================================
📊 测试执行摘要
============================================================
dependencies   | ✅ 通过   |   2.34s
core          | ✅ 通过   |  15.67s
docs          | ✅ 通过   |   8.92s
------------------------------------------------------------
总计: 3/3 通过 | 总耗时: 26.93s

🎉 所有测试都通过了！
```

## 🛠️ 故障排除

### 常见问题

1. **依赖测试失败**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **导入错误**
   ```bash
   # 确保在项目根目录下运行
   cd /path/to/AxisFuzzy
   python -c "from tests import run_core_tests; run_core_tests()"
   ```

3. **文档测试失败**
   ```bash
   # 安装文档构建依赖
   pip install -r docs/requirements.txt
   ```

### 调试模式

```python
# 启用详细输出
from tests import run_core_tests
run_core_tests(verbose=True, extra_args=['--tb=long', '-s'])
```

## 🔄 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        python -m tests --docs
```

## 📚 相关文档

- [AxisFuzzy 主文档](../docs/)
- [依赖测试详细说明](test_dependencies/README.md)
- [开发指南](../CONTRIBUTING.md)
- [API 参考](../docs/api/)

## 🤝 贡献

如需添加新的测试或改进现有测试，请参考：

1. **添加新测试模块**：在相应的 `test_*` 目录下创建测试文件
2. **更新测试入口**：在对应的 `__init__.py` 中添加导入
3. **扩展测试框架**：修改 `tests/__init__.py` 中的测试运行函数
4. **更新文档**：相应更新本 README 文件

---

**AxisFuzzy 测试套件** - 确保代码质量，提升开发效率 ✨