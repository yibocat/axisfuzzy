"""AxisFuzzy 混入模块测试套件

本模块包含对 AxisFuzzy 混入系统的全面测试，包括：
- 边界情况测试
- 工厂模式测试
- 集成测试
- 性能测试
- 注册表测试

测试文件：
- test_edge_cases.py: 边界情况和异常处理测试
- test_factory.py: 混入工厂模式测试
- test_integration.py: 混入集成测试
- test_performance.py: 混入性能测试
- test_registry.py: 混入注册表测试

混入系统提供了模块化的功能扩展机制，
这些测试确保混入的正确性、性能和可靠性。
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_edge_cases
from . import test_factory
from . import test_integration
from . import test_performance
from . import test_registry

# 定义模块的公共接口
__all__ = [
    'test_edge_cases',
    'test_factory',
    'test_integration',
    'test_performance',
    'test_registry'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 混入模块测试套件'

# 混入测试的特殊标记
__test_category__ = 'mixin'
__execution_priority__ = 'medium'
__includes_performance_tests__ = True  # 包含性能测试