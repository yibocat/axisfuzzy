"""AxisFuzzy 核心模块测试套件

本模块包含对 AxisFuzzy 核心功能的全面测试，包括：
- 模糊数组后端测试
- 模糊数测试
- 模糊数基础功能测试
- 操作分发器测试
- 注册表系统测试
- 三角模糊数测试

测试文件：
- test_fuzzarray_backend.py: 模糊数组后端测试
- test_fuzznums.py: 模糊数测试
- test_fuzznums_basic.py: 模糊数基础功能测试
- test_operation_dispatcher.py: 操作分发器测试
- test_registry.py: 注册表系统测试
- test_triangular.py: 三角模糊数测试
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_fuzzarray_backend
from . import test_fuzznums
from . import test_fuzznums_basic
from . import test_operation_dispatcher
from . import test_registry
from . import test_triangular

# 定义模块的公共接口
__all__ = [
    'test_fuzzarray_backend',
    'test_fuzznums',
    'test_fuzznums_basic',
    'test_operation_dispatcher',
    'test_registry',
    'test_triangular'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 核心模块测试套件'