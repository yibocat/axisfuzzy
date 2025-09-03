# AxisFuzzy Test Suite

A comprehensive and modular test framework for the AxisFuzzy fuzzy logic library.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Categories](#test-categories)
- [Advanced Usage](#advanced-usage)
- [Test Reports](#test-reports)
- [Troubleshooting](#troubleshooting)
- [Continuous Integration](#continuous-integration)
- [Contributing](#contributing)

## 🎯 Overview

The AxisFuzzy test suite is designed with modularity and flexibility in mind. It provides multiple ways to run tests, from quick development checks to comprehensive CI/CD validation.

### Key Features

- **Modular Design**: Each test category can be run independently
- **Multiple Interfaces**: Programmatic, command-line, and traditional pytest support
- **Flexible Configuration**: Environment variables and parameter customization
- **Comprehensive Coverage**: Dependencies, core functionality, documentation, and analysis modules
- **Performance Monitoring**: Built-in timing and reporting capabilities

### Directory Structure

```
tests/
├── __init__.py              # Main test entry point
├── __main__.py              # Command-line interface
├── README.md                # This documentation (Chinese)
├── README_en.md             # This documentation (English)
├── automation_guide.md      # Automation and CI/CD guide
├── test_core/               # Core functionality tests
│   ├── __init__.py
│   ├── test_fuzznum.py
│   ├── test_fuzzarray.py
│   └── test_operations.py
├── test_dependencies/       # Dependency validation tests
│   ├── __init__.py
│   └── test_imports.py
├── test_docs/               # Documentation tests
│   ├── __init__.py
│   └── test_docstrings.py
├── test_fuzzifier/          # Fuzzification tests
│   ├── __init__.py
│   └── test_fuzzifier.py
├── test_membership/         # Membership function tests
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_standard.py
│   ├── test_factory.py
│   └── test_visualization.py
├── test_mixin/              # Mixin functionality tests
│   ├── __init__.py
│   └── test_mixin.py
└── test_random/             # Random generation tests
    ├── __init__.py
    └── test_random.py
```

## 🚀 Quick Start

### 1. Programmatic Interface

The most flexible way to run tests programmatically:

```python
from tests import (
    run_all_tests,
    run_quick_tests,
    run_core_tests,
    run_dependency_tests,
    run_docs_tests
)

# Quick development check
success = run_quick_tests()
print(f"Quick tests: {'✅ PASSED' if success else '❌ FAILED'}")

# Core functionality tests
success = run_core_tests(verbose=True)
print(f"Core tests: {'✅ PASSED' if success else '❌ FAILED'}")

# Complete test suite with documentation
success = run_all_tests(include_docs=True, verbose=False)
print(f"All tests: {'✅ PASSED' if success else '❌ FAILED'}")
```

### 2. Command-Line Interface

Convenient command-line access for scripts and CI/CD:

```bash
# Quick tests (dependencies + core, no docs)
python -m tests --quick

# Core functionality only
python -m tests --core-only

# Dependencies only
python -m tests --deps-only

# Documentation tests only
python -m tests --docs-only

# Complete test suite with documentation
python -m tests --docs

# All tests (same as --docs)
python -m tests
```

### 3. Traditional pytest

Direct pytest usage for maximum control:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_core/
pytest tests/test_dependencies/
pytest tests/test_docs/

# With custom options
pytest tests/ -v --tb=short
pytest tests/ --maxfail=3 --tb=line
```

## 📊 Test Categories

### 1. Dependency Tests (`test_dependencies/`)

**Purpose**: Validate that all required dependencies are properly installed and importable.

**What it tests**:
- Core Python packages (numpy, scipy, matplotlib)
- Optional dependencies availability
- Version compatibility
- Import functionality

**When to run**:
- After fresh installation
- Before running other tests
- In CI/CD pipelines
- When debugging import issues

```python
# Run dependency tests
from tests import run_dependency_tests
success = run_dependency_tests(verbose=True)
```

### 2. Core Tests (`test_core/`)

**Purpose**: Validate the fundamental functionality of AxisFuzzy.

**What it tests**:
- `Fuzznum` class operations
- `Fuzzarray` class operations
- Mathematical operations
- Data structure integrity
- Core algorithms

**When to run**:
- During active development
- Before committing changes
- For regression testing
- Performance validation

```python
# Run core tests
from tests import run_core_tests
success = run_core_tests(verbose=True, extra_args=['--tb=short'])
```

### 3. Documentation Tests (`test_docs/`)

**Purpose**: Ensure documentation quality and consistency.

**What it tests**:
- Docstring presence and format
- Code examples in documentation
- API documentation completeness
- Documentation build process

**When to run**:
- Before releases
- When updating documentation
- In CI/CD for documentation validation
- For API consistency checks

```python
# Run documentation tests
from tests import run_docs_tests
success = run_docs_tests(verbose=False)
```

### 4. Analysis Tests (`test_fuzzifier/`, `test_membership/`, etc.)

**Purpose**: Test specialized analysis and processing modules.

**What it tests**:
- Fuzzification algorithms
- Membership function implementations
- Visualization capabilities
- Advanced analysis features

**When to run**:
- When working on analysis features
- For comprehensive validation
- Before feature releases
- Performance benchmarking

## 🔧 Advanced Usage

### Test Parameter Configuration

All test functions support the following parameters:

- `verbose` (bool): Whether to show detailed output, default `True`
- `extra_args` (list): Additional arguments passed to pytest
- `include_docs` (bool): Whether to include documentation tests (only for `run_all_tests`)

### pytest Parameter Examples

```python
from tests import run_core_tests

# Show detailed output, stop on first failure
run_core_tests(verbose=True, extra_args=['-x', '--tb=short'])

# Run tests in parallel (requires pytest-xdist)
run_core_tests(extra_args=['-n', '4'])

# Run only tests with specific markers
run_core_tests(extra_args=['-m', 'slow'])

# Generate coverage report (requires pytest-cov)
run_core_tests(extra_args=['--cov=axisfuzzy', '--cov-report=html'])
```

### Advanced Configuration Modes

```python
# Development mode: Quick feedback
dev_config = {
    'verbose': True,
    'extra_args': ['--tb=short', '-x']  # Short traceback, fail fast
}

# Integration mode: Comprehensive testing
integration_config = {
    'verbose': True,
    'extra_args': ['--tb=short', '--maxfail=5']  # Allow up to 5 failures
}

# CI mode: Strict testing
ci_config = {
    'verbose': False,
    'extra_args': ['--tb=line', '--strict-markers']  # Concise output, strict markers
}

# Use configuration
from tests import run_all_tests
result = run_all_tests(include_docs=True, **ci_config)
```

### Programmatic Test Result Handling

#### Basic Handling

```python
from tests import run_core_tests, run_dependency_tests

# Run tests and handle results
core_success = run_core_tests()
deps_success = run_dependency_tests()

if core_success and deps_success:
    print("✅ All critical tests passed")
    # Execute deployment or other operations
else:
    print("❌ Tests failed, stopping process")
    # Send notifications or log errors
```

#### Advanced Result Analysis

```python
import time
from tests import run_all_tests, run_quick_tests, run_core_tests

def comprehensive_test_analysis():
    """Comprehensive test analysis"""
    results = {}
    
    # 1. Quick tests
    start_time = time.time()
    results['quick'] = {
        'success': run_quick_tests(),
        'duration': time.time() - start_time
    }
    
    # 2. Core tests (detailed mode)
    start_time = time.time()
    results['core_detailed'] = {
        'success': run_core_tests(verbose=True, extra_args=['--tb=short']),
        'duration': time.time() - start_time
    }
    
    # 3. Full tests
    start_time = time.time()
    results['full'] = {
        'success': run_all_tests(include_docs=True),
        'duration': time.time() - start_time
    }
    
    # Analyze results
    total_duration = sum(r['duration'] for r in results.values())
    success_count = sum(1 for r in results.values() if r['success'])
    
    print(f"\n📊 Test Analysis Report:")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Success Rate: {success_count}/{len(results)}")
    
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{test_name}: {status} ({result['duration']:.2f}s)")
    
    return results

# Usage example
analysis = comprehensive_test_analysis()
```

### Environment Variable Configuration

Control test behavior flexibly through environment variables without modifying code:

```python
import os
from tests import run_all_tests

# Control test behavior through environment variables
test_config = {
    'include_docs': os.getenv('AXISFUZZY_TEST_DOCS', 'false').lower() == 'true',
    'verbose': os.getenv('AXISFUZZY_TEST_VERBOSE', 'true').lower() == 'true',
    'extra_args': os.getenv('AXISFUZZY_TEST_ARGS', '').split() if os.getenv('AXISFUZZY_TEST_ARGS') else None
}

# Run tests
result = run_all_tests(**test_config)
```

#### Environment Variables Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AXISFUZZY_TEST_DOCS` | `false` | Whether to include documentation tests |
| `AXISFUZZY_TEST_VERBOSE` | `true` | Whether to show detailed output |
| `AXISFUZZY_TEST_ARGS` | `` | Additional pytest arguments (space-separated) |
| `AXISFUZZY_TEST_MODE` | `dev` | Test mode (dev/integration/ci) |

#### Usage Examples

```bash
# Set environment variables
export AXISFUZZY_TEST_DOCS=true
export AXISFUZZY_TEST_VERBOSE=false
export AXISFUZZY_TEST_ARGS="--tb=short --maxfail=1"

# Run tests
python -c "from tests import run_all_tests; run_all_tests()"

# Or set inline
AXISFUZZY_TEST_DOCS=true AXISFUZZY_TEST_VERBOSE=false python -m tests --docs
```

### Automated Testing Scripts

For complex automation testing needs, we provide a dedicated automation guide. Please refer to:

- **[Automation Guide](automation_guide.md)** - Detailed automation scripts and configurations
- **[CI/CD Integration](automation_guide.md#continuous-integration)** - Continuous integration configuration examples
- **[Docker Test Environment](automation_guide.md#docker-environment-testing)** - Containerized testing environments

## 📈 Test Reports

### Built-in Reporting

The test framework provides built-in reporting capabilities:

```python
from tests import run_all_tests

# Run tests with detailed reporting
success = run_all_tests(
    include_docs=True,
    verbose=True,
    extra_args=['--tb=short', '--durations=10']
)

if success:
    print("📊 All tests completed successfully")
    print("📄 Check console output for detailed timing information")
else:
    print("⚠️  Some tests failed - check output for details")
```

### Custom Reporting

For advanced reporting needs, you can integrate with pytest plugins:

```bash
# Install reporting plugins
pip install pytest-html pytest-cov pytest-json-report

# Generate HTML report
python -m pytest tests/ --html=report.html --self-contained-html

# Generate coverage report
python -m pytest tests/ --cov=axisfuzzy --cov-report=html --cov-report=term

# Generate JSON report
python -m pytest tests/ --json-report --json-report-file=report.json
```

### Integration with CI/CD

The test framework is designed to integrate seamlessly with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/dev_requirements.txt
    
    - name: Run dependency tests
      run: python -m tests --deps-only
    
    - name: Run core tests
      run: python -m tests --core-only
    
    - name: Run full test suite
      if: matrix.python-version == '3.11'
      run: python -m tests --docs
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'axisfuzzy'`

**Solution**:
```bash
# Ensure you're in the project root directory
cd /path/to/AxisFuzzy
pwd  # Should show AxisFuzzy project root

# Check project structure
ls -la  # Should see axisfuzzy/ directory

# Set PYTHONPATH if needed
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### 2. Dependency Issues

**Problem**: Missing dependencies

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements/core_requirements.txt
pip install -r requirements/dev_requirements.txt

# Check virtual environment
which python
pip list
```

#### 3. Test Failures

**Problem**: Unexpected test failures

**Solution**:
```bash
# Run with verbose output
python -m tests --core-only -v

# Run specific failing tests
pytest tests/test_core/test_specific.py -v

# Check for environment issues
python -c "import axisfuzzy; print('Import successful')"
```

### Debug Mode

For detailed debugging, use verbose mode with extended output:

```python
from tests import run_core_tests

# Maximum verbosity
result = run_core_tests(
    verbose=True,
    extra_args=['--tb=long', '-s', '--capture=no', '-vv']
)
```

### Performance Issues

If tests are running slowly:

```bash
# Show test durations
python -m pytest tests/ --durations=10

# Run tests in parallel
pip install pytest-xdist
python -m pytest tests/ -n auto

# Run only fast tests
python -m pytest tests/ -m "not slow"
```

## 🔄 Continuous Integration

The test framework is designed for seamless CI/CD integration. See our [Automation Guide](automation_guide.md) for detailed examples of:

- GitHub Actions workflows
- GitLab CI configurations
- Jenkins pipelines
- Docker-based testing
- Multi-platform testing strategies

### Quick CI Setup

For a basic CI setup, use these commands:

```bash
# Install dependencies
pip install -r requirements/dev_requirements.txt

# Run dependency validation
python -m tests --deps-only

# Run core functionality tests
python -m tests --core-only

# Run full test suite (for main branch)
python -m tests --docs
```

## 🤝 Contributing

When contributing to AxisFuzzy, please ensure:

1. **All tests pass**: Run `python -m tests --docs` before submitting
2. **Add tests for new features**: Include appropriate test coverage
3. **Update documentation**: Ensure docstrings and examples are current
4. **Follow conventions**: Use existing test patterns and naming

### Running Tests During Development

```bash
# Quick check during development
python -m tests --quick

# Before committing
python -m tests --core-only

# Before submitting PR
python -m tests --docs
```

### Writing New Tests

When adding new tests:

1. Place them in the appropriate test category directory
2. Follow existing naming conventions (`test_*.py`)
3. Use descriptive test names and docstrings
4. Include both positive and negative test cases
5. Add performance tests for critical functionality

## 📚 Related Documentation

- **[Chinese Documentation](README.md)** - 中文版测试文档
- **[Automation Guide](automation_guide.md)** - Detailed automation and CI/CD guide
- **[Project Documentation](../README.md)** - Main AxisFuzzy project overview
- **[Contributing Guide](../CONTRIBUTING.md)** - Development and contribution guidelines

---

**Note**: This test framework is designed to be robust and flexible. If you encounter issues or have suggestions for improvements, please open an issue or submit a pull request.

**Happy Testing! 🎉**