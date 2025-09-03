"""AxisFuzzy 模糊化器测试套件

本模块包含对 AxisFuzzy 模糊化器系统的全面测试，包括：
- 模糊化器核心功能测试
- 模糊化器注册表测试
- 模糊化策略测试

测试文件：
- test_fuzzifier.py: 模糊化器核心功能测试
- test_registry.py: 模糊化器注册表测试
- test_strategy.py: 模糊化策略测试

模糊化器是将精确值转换为模糊值的核心组件，
这些测试确保模糊化过程的正确性和可靠性。
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_fuzzifier
from . import test_registry
from . import test_strategy

# 定义模块的公共接口
__all__ = [
    'test_fuzzifier',
    'test_registry',
    'test_strategy'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 模糊化器测试套件'

# 模糊化器测试的特殊标记
__test_category__ = 'fuzzifier'
__execution_priority__ = 'medium'