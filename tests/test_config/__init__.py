"""AxisFuzzy 配置模块测试套件

本模块包含对 AxisFuzzy 配置系统的全面测试，包括：
- API 配置测试
- 配置文件处理测试
- 配置管理器测试
- 配置摘要和模板测试

测试文件：
- test_api.py: 配置 API 测试
- test_config_file.py: 配置文件处理测试
- test_manager.py: 配置管理器测试
- test_summary_and_template.py: 配置摘要和模板测试
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_api
from . import test_config_file
from . import test_manager
from . import test_summary_and_template

# 定义模块的公共接口
__all__ = [
    'test_api',
    'test_config_file', 
    'test_manager',
    'test_summary_and_template'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 配置模块测试套件'