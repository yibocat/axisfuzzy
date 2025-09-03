# AxisFuzzy 测试自动化指南

本文档提供了 AxisFuzzy 项目测试自动化的完整指南，包括自动化脚本、环境配置和 CI/CD 集成。

## 📋 目录

- [集成到自动化脚本](#集成到自动化脚本)
- [测试环境配置](#测试环境配置)
- [持续集成配置](#持续集成配置)
- [Docker 环境测试](#docker-环境测试)
- [故障排除](#故障排除)

## 🤖 集成到自动化脚本

### 基础自动化脚本

```python
#!/usr/bin/env python3
"""AxisFuzzy 自动化测试脚本

支持多种运行模式：
- 开发模式：快速测试
- 集成模式：完整测试
- CI 模式：严格测试
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from tests import (
    run_dependency_tests, 
    run_core_tests, 
    run_docs_tests,
    run_all_tests,
    run_quick_tests
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoTestRunner:
    """自动化测试运行器"""
    
    def __init__(self, mode='dev'):
        self.mode = mode
        self.config = self._get_config()
    
    def _get_config(self):
        """根据模式获取配置"""
        configs = {
            'dev': {
                'include_docs': False,
                'verbose': True,
                'extra_args': ['--tb=short', '-x'],
                'fail_fast': True
            },
            'integration': {
                'include_docs': True,
                'verbose': True,
                'extra_args': ['--tb=short', '--maxfail=5'],
                'fail_fast': False
            },
            'ci': {
                'include_docs': True,
                'verbose': False,
                'extra_args': ['--tb=line', '--strict-markers'],
                'fail_fast': True
            }
        }
        return configs.get(self.mode, configs['dev'])
    
    def check_environment(self):
        """检查测试环境"""
        logger.info(f"检查测试环境 (模式: {self.mode})")
        
        # 检查 Python 版本
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"Python 版本过低: {python_version}")
            return False
        
        # 检查项目根目录
        project_root = Path.cwd()
        if not (project_root / 'axisfuzzy').exists():
            logger.error("未在 AxisFuzzy 项目根目录下运行")
            return False
        
        logger.info("环境检查通过")
        return True
    
    def run_dependency_check(self):
        """运行依赖检查"""
        logger.info("🔍 开始依赖验证...")
        
        success = run_dependency_tests(
            verbose=self.config['verbose'],
            extra_args=self.config['extra_args']
        )
        
        if success:
            logger.info("✅ 依赖验证通过")
        else:
            logger.error("❌ 依赖验证失败")
            if self.config['fail_fast']:
                logger.error("快速失败模式：停止测试")
                return False
        
        return success
    
    def run_core_check(self):
        """运行核心功能检查"""
        logger.info("⚙️  开始核心功能测试...")
        
        success = run_core_tests(
            verbose=self.config['verbose'],
            extra_args=self.config['extra_args']
        )
        
        if success:
            logger.info("✅ 核心功能测试通过")
        else:
            logger.error("❌ 核心功能测试失败")
        
        return success
    
    def run_docs_check(self):
        """运行文档检查"""
        if not self.config['include_docs']:
            logger.info("跳过文档测试")
            return True
        
        logger.info("📚 开始文档测试...")
        
        success = run_docs_tests(
            verbose=self.config['verbose'],
            extra_args=self.config['extra_args']
        )
        
        if success:
            logger.info("✅ 文档测试通过")
        else:
            logger.warning("⚠️  文档测试失败")
        
        return success
    
    def run_full_suite(self):
        """运行完整测试套件"""
        logger.info(f"🎯 开始完整测试套件 (模式: {self.mode})")
        
        # 环境检查
        if not self.check_environment():
            return False
        
        # 依赖检查
        if not self.run_dependency_check():
            return False
        
        # 核心功能检查
        core_success = self.run_core_check()
        if not core_success and self.config['fail_fast']:
            return False
        
        # 文档检查
        docs_success = self.run_docs_check()
        
        # 最终结果
        overall_success = core_success and (docs_success or not self.config['include_docs'])
        
        if overall_success:
            logger.info("🎉 所有测试通过！")
        else:
            logger.error("❌ 部分测试失败")
        
        return overall_success

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='AxisFuzzy 自动化测试')
    parser.add_argument(
        '--mode', 
        choices=['dev', 'integration', 'ci'],
        default='dev',
        help='测试模式'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='运行快速测试'
    )
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            logger.info("⚡ 运行快速测试...")
            success = run_quick_tests()
        else:
            runner = AutoTestRunner(mode=args.mode)
            success = runner.run_full_suite()
        
        if success:
            logger.info("✅ 测试执行成功")
            sys.exit(0)
        else:
            logger.error("❌ 测试执行失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 测试执行异常: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

### 使用自动化脚本

将上述脚本保存为 `autotest.py`，然后可以通过以下方式使用：

```bash
# 开发模式（默认）
python autotest.py

# 集成测试模式
python autotest.py --mode integration

# CI 模式
python autotest.py --mode ci

# 快速测试
python autotest.py --quick
```

### 高级自动化示例

#### 带报告生成的自动化脚本

```python
#!/usr/bin/env python3
"""带报告生成的高级自动化测试脚本"""

import time
import json
from datetime import datetime
from pathlib import Path
from tests import run_all_tests, run_quick_tests, run_core_tests, run_dependency_tests

class AdvancedTestRunner:
    """高级测试运行器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.results = []
        self.start_time = None
    
    def run_with_timing(self, test_func, test_name, **kwargs):
        """运行测试并记录时间"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        print(f"🚀 开始执行: {test_name}")
        
        try:
            success = test_func(**kwargs)
            duration = time.time() - start_time
            
            result = {
                'test_name': test_name,
                'success': success,
                'duration': duration,
                'timestamp': timestamp,
                'config': kwargs
            }
            
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{status} {test_name} (耗时: {duration:.2f}s)")
            
            self.results.append(result)
            return success, result
            
        except Exception as e:
            duration = time.time() - start_time
            result = {
                'test_name': test_name,
                'success': False,
                'duration': duration,
                'timestamp': timestamp,
                'error': str(e),
                'config': kwargs
            }
            
            print(f"💥 异常 {test_name}: {e}")
            self.results.append(result)
            return False, result
    
    def run_comprehensive_suite(self):
        """运行综合测试套件"""
        self.start_time = time.time()
        print("🎯 开始综合测试套件执行...")
        print("=" * 60)
        
        # 1. 依赖测试
        deps_success, _ = self.run_with_timing(
            run_dependency_tests, 
            'dependency_tests',
            verbose=False
        )
        
        if not deps_success:
            print("❌ 依赖测试失败，停止执行")
            return False
        
        # 2. 快速测试
        quick_success, _ = self.run_with_timing(
            run_quick_tests, 
            'quick_tests',
            verbose=False
        )
        
        # 3. 核心测试（详细模式）
        core_success, _ = self.run_with_timing(
            run_core_tests,
            'core_tests_detailed',
            verbose=True,
            extra_args=['--tb=short']
        )
        
        # 4. 完整测试（包含文档）
        full_success, _ = self.run_with_timing(
            run_all_tests,
            'full_tests_with_docs',
            include_docs=True,
            verbose=False,
            extra_args=['--tb=short']
        )
        
        return all([deps_success, quick_success, core_success, full_success])
    
    def generate_report(self, output_file='test_report.json'):
        """生成详细的测试报告"""
        total_duration = time.time() - self.start_time if self.start_time else 0
        success_count = sum(1 for r in self.results if r['success'])
        
        print("\n" + "=" * 60)
        print("📊 测试执行报告")
        print("=" * 60)
        
        for result in self.results:
            status = "✅ 通过" if result['success'] else "❌ 失败"
            print(f"{result['test_name']:25} | {status:8} | {result['duration']:6.2f}s")
        
        print("-" * 60)
        print(f"总计: {success_count}/{len(self.results)} 通过 | 总耗时: {total_duration:.2f}s")
        
        # 生成 JSON 报告
        report = {
            'summary': {
                'total_tests': len(self.results),
                'passed': success_count,
                'failed': len(self.results) - success_count,
                'success_rate': success_count / len(self.results) if self.results else 0,
                'total_duration': total_duration,
                'timestamp': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        # 保存报告
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细报告已保存到: {output_path.absolute()}")
        return report

# 使用示例
if __name__ == '__main__':
    runner = AdvancedTestRunner()
    success = runner.run_comprehensive_suite()
    report = runner.generate_report()
    
    # 根据结果设置退出码
    import sys
    sys.exit(0 if success else 1)
```

## ⚙️ 测试环境配置

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

### 配置文件支持

创建 `test_config.json` 文件来管理测试配置：

```json
{
  "development": {
    "include_docs": false,
    "verbose": true,
    "extra_args": ["--tb=short", "-x"]
  },
  "integration": {
    "include_docs": true,
    "verbose": true,
    "extra_args": ["--tb=short", "--maxfail=5"]
  },
  "production": {
    "include_docs": true,
    "verbose": false,
    "extra_args": ["--tb=line", "--strict-markers"]
  }
}
```

配置加载脚本：

```python
import json
from pathlib import Path
from tests import run_all_tests

def load_test_config(mode='development'):
    """从配置文件加载测试配置"""
    config_file = Path('test_config.json')
    
    if config_file.exists():
        with open(config_file) as f:
            configs = json.load(f)
        return configs.get(mode, configs.get('development', {}))
    
    # 默认配置
    return {
        'include_docs': False,
        'verbose': True,
        'extra_args': ['--tb=short']
    }

# 使用配置
config = load_test_config('integration')
result = run_all_tests(**config)
```

## 🐳 Docker 环境测试

### 基础 Docker 配置

创建 `Dockerfile.test`：

```dockerfile
# Dockerfile.test
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements/ requirements/

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements/dev_requirements.txt

# 复制项目文件
COPY . .

# 设置测试环境变量
ENV AXISFUZZY_TEST_DOCS=true
ENV AXISFUZZY_TEST_VERBOSE=false
ENV AXISFUZZY_TEST_ARGS="--tb=short --maxfail=1"
ENV PYTHONPATH=/app

# 默认命令
CMD ["python", "-m", "tests", "--docs"]
```

### Docker Compose 配置

创建 `docker-compose.test.yml`：

```yaml
version: '3.8'

services:
  test-python38:
    build:
      context: .
      dockerfile: Dockerfile.test
      args:
        PYTHON_VERSION: 3.8
    environment:
      - AXISFUZZY_TEST_DOCS=true
      - AXISFUZZY_TEST_VERBOSE=false
    volumes:
      - ./test_reports:/app/test_reports
    command: python -m tests --docs
  
  test-python39:
    build:
      context: .
      dockerfile: Dockerfile.test
      args:
        PYTHON_VERSION: 3.9
    environment:
      - AXISFUZZY_TEST_DOCS=true
      - AXISFUZZY_TEST_VERBOSE=false
    volumes:
      - ./test_reports:/app/test_reports
    command: python -m tests --docs
  
  test-python310:
    build:
      context: .
      dockerfile: Dockerfile.test
      args:
        PYTHON_VERSION: "3.10"
    environment:
      - AXISFUZZY_TEST_DOCS=true
      - AXISFUZZY_TEST_VERBOSE=false
    volumes:
      - ./test_reports:/app/test_reports
    command: python -m tests --docs
  
  test-python311:
    build:
      context: .
      dockerfile: Dockerfile.test
      args:
        PYTHON_VERSION: 3.11
    environment:
      - AXISFUZZY_TEST_DOCS=true
      - AXISFUZZY_TEST_VERBOSE=false
    volumes:
      - ./test_reports:/app/test_reports
    command: python -m tests --docs
```

### 使用 Docker 进行测试

```bash
# 构建测试镜像
docker build -f Dockerfile.test -t axisfuzzy-test .

# 运行单个测试容器
docker run --rm axisfuzzy-test

# 运行所有 Python 版本的测试
docker-compose -f docker-compose.test.yml up --build

# 运行特定 Python 版本
docker-compose -f docker-compose.test.yml up test-python311

# 后台运行并查看日志
docker-compose -f docker-compose.test.yml up -d
docker-compose -f docker-compose.test.yml logs -f
```

### 多阶段 Docker 构建

```dockerfile
# Dockerfile.multistage
# 第一阶段：构建环境
FROM python:3.11-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装依赖
COPY requirements/ requirements/
RUN pip install --user --no-cache-dir -r requirements/dev_requirements.txt

# 第二阶段：运行环境
FROM python:3.11-slim as runner

WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from=builder /root/.local /root/.local

# 确保脚本在 PATH 中
ENV PATH=/root/.local/bin:$PATH

# 复制项目文件
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV AXISFUZZY_TEST_DOCS=true
ENV AXISFUZZY_TEST_VERBOSE=false

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import axisfuzzy; print('OK')" || exit 1

CMD ["python", "-m", "tests", "--docs"]
```

## 🔄 持续集成配置

### GitHub Actions

创建 `.github/workflows/tests.yml`：

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # 每天 UTC 2:00 运行
    - cron: '0 2 * * *'

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
        exclude:
          # 排除一些组合以节省资源
          - os: windows-latest
            python-version: '3.8'
          - os: macos-latest
            python-version: '3.8'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements/*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/dev_requirements.txt
    
    - name: Run dependency tests
      run: python -m tests --deps-only
    
    - name: Run core tests
      run: python -m tests --core-only
    
    - name: Run documentation tests
      if: matrix.python-version == '3.11' && matrix.os == 'ubuntu-latest'
      run: python -m tests --docs
    
    - name: Run full test suite
      if: matrix.python-version == '3.11'
      run: python -m tests --docs
      env:
        AXISFUZZY_TEST_VERBOSE: false
        AXISFUZZY_TEST_ARGS: "--tb=short --maxfail=3"
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
        path: |
          test_report.json
          .pytest_cache/
        retention-days: 7

  coverage:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/dev_requirements.txt
        pip install coverage pytest-cov
    
    - name: Run tests with coverage
      run: |
        python -m pytest tests/ --cov=axisfuzzy --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### 1. 依赖相关问题

**问题**: 依赖测试失败
```bash
ModuleNotFoundError: No module named 'numpy'
```

**解决方案**:
```bash
# 重新安装依赖
pip install -r requirements/core_requirements.txt
pip install -r requirements/dev_requirements.txt

# 检查虚拟环境
which python
pip list
```

#### 2. 路径相关问题

**问题**: 导入错误
```bash
ModuleNotFoundError: No module named 'axisfuzzy'
```

**解决方案**:
```bash
# 确保在项目根目录
cd /path/to/AxisFuzzy
pwd  # 应该显示 AxisFuzzy 项目根目录

# 检查项目结构
ls -la  # 应该看到 axisfuzzy/ 目录

# 设置 PYTHONPATH（如果需要）
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### 3. 权限问题

**问题**: 文件权限错误
```bash
PermissionError: [Errno 13] Permission denied
```

**解决方案**:
```bash
# 检查文件权限
ls -la tests/

# 修复权限（如果需要）
chmod +x tests/*.py

# 对于 Docker 环境
sudo chown -R $USER:$USER .
```

#### 4. 内存问题

**问题**: 测试过程中内存不足
```bash
MemoryError: Unable to allocate array
```

**解决方案**:
```bash
# 限制并行测试数量
python -m pytest tests/ -n 2  # 使用 2 个进程

# 或者逐个运行测试模块
python -m tests --core-only
python -m tests --deps-only
```

#### 5. 网络问题

**问题**: 文档构建时网络超时
```bash
ConnectionError: HTTPSConnectionPool
```

**解决方案**:
```bash
# 跳过文档测试
python -m tests --core-only

# 或设置代理
export https_proxy=http://proxy.example.com:8080
export http_proxy=http://proxy.example.com:8080
```

### 调试技巧

#### 1. 详细输出模式

```python
# 启用最详细的输出
from tests import run_core_tests
run_core_tests(
    verbose=True, 
    extra_args=['--tb=long', '-s', '--capture=no', '-vv']
)
```

#### 2. 单独运行失败的测试

```bash
# 只运行上次失败的测试
python -m pytest --lf

# 先运行失败的测试
python -m pytest --ff

# 在第一个失败时停止
python -m pytest -x
```

#### 3. 性能分析

```bash
# 显示最慢的测试
python -m pytest --durations=10

# 显示所有测试的耗时
python -m pytest --durations=0
```

#### 4. 内存使用监控

```python
import psutil
import os
from tests import run_core_tests

def monitor_memory():
    process = psutil.Process(os.getpid())
    print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

monitor_memory()
run_core_tests()
monitor_memory()
```

### 日志配置

创建 `logging_config.py` 用于调试：

```python
import logging
import sys
from pathlib import Path

def setup_debug_logging(log_file='test_debug.log'):
    """设置调试日志"""
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# 使用示例
if __name__ == '__main__':
    logger = setup_debug_logging()
    
    from tests import run_core_tests
    logger.info("开始运行核心测试")
    
    try:
        result = run_core_tests(verbose=True)
        logger.info(f"测试结果: {result}")
    except Exception as e:
        logger.exception(f"测试异常: {e}")
```

## 📚 相关文档

- [主测试文档](README.md) - 基础测试使用指南
- [英文测试文档](README_en.md) - English version of test documentation
- [项目主文档](../README.md) - AxisFuzzy 项目总览
- [开发指南](../CONTRIBUTING.md) - 贡献和开发指南

## 🤝 贡献

如果您在使用自动化测试过程中遇到问题或有改进建议，欢迎：

1. 提交 Issue 报告问题
2. 提交 Pull Request 改进文档或脚本
3. 分享您的自动化测试经验

---

**注意**: 本文档中的脚本和配置仅供参考，请根据您的具体环境和需求进行调整。