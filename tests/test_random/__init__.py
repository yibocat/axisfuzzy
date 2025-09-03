"""AxisFuzzy 随机数模块测试套件

本模块包含对 AxisFuzzy 随机数生成系统的全面测试，包括：
- 随机数 API 测试
- 基础生成器测试
- 注册表系统测试
- 随机种子管理测试

测试文件：
- test_api.py: 随机数 API 测试
- test_base.py: 基础随机数生成器测试
- test_registry.py: 随机数生成器注册表测试
- test_seed.py: 随机种子管理测试

随机数生成是模糊计算中的重要组件，
这些测试确保随机数生成的正确性、可重现性和统计特性。
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_api
from . import test_base
from . import test_registry
from . import test_seed

# 定义模块的公共接口
__all__ = [
    'test_api',
    'test_base',
    'test_registry',
    'test_seed'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 随机数模块测试套件'

# 随机数测试的特殊标记
__test_category__ = 'random'
__execution_priority__ = 'medium'
__requires_statistical_validation__ = True  # 需要统计验证