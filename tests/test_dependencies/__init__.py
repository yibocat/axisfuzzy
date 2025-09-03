"""AxisFuzzy 依赖测试套件

本模块包含对 AxisFuzzy 项目依赖的全面测试，包括：
- 核心依赖测试（numpy, numba）
- 可选依赖测试（analysis, dev, docs 组件）
- 依赖版本兼容性测试
- 依赖功能验证测试

测试文件：
- test_core_dependencies.py: 核心依赖测试
- test_optional_dependencies.py: 可选依赖测试

注意：
本测试套件通常在环境设置或 CI 的早期阶段运行，
用于快速验证项目依赖是否正确安装和配置。
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_core_dependencies
from . import test_optional_dependencies

# 定义模块的公共接口
__all__ = [
    'test_core_dependencies',
    'test_optional_dependencies'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 依赖测试套件'

# 依赖测试的特殊标记
__test_category__ = 'dependencies'
__execution_priority__ = 'high'  # 依赖测试应该优先执行