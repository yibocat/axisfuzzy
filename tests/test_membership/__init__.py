#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
AxisFuzzy 隶属函数测试套件

本模块包含对 AxisFuzzy 隶属函数系统的全面测试，包括：
- 基类接口测试
- 标准隶属函数实现测试
- 工厂函数测试
- 可视化功能测试

测试文件：
- test_base.py: 隶属函数基类测试
- test_factory.py: 隶属函数工厂测试
- test_function.py: 隶属函数实现测试
- test_visualization.py: 隶属函数可视化测试
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_base
from . import test_factory
from . import test_function
from . import test_visualization

# 定义模块的公共接口
__all__ = [
    'test_base',
    'test_factory',
    'test_function',
    'test_visualization'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 隶属函数测试套件'

# 隶属函数测试的特殊标记
__test_category__ = 'membership'
__execution_priority__ = 'medium'