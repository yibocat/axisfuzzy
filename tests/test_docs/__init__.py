"""AxisFuzzy 文档测试套件

本模块包含对 AxisFuzzy 项目文档的全面测试，包括：
- 文档构建测试
- 文档配置测试
- 文档内容完整性测试
- 文档扩展功能测试

测试文件：
- test_build.py: 文档构建测试
- test_config.py: 文档配置测试
- test_content.py: 文档内容测试
- test_extensions.py: 文档扩展测试

注意：
文档测试通常耗时较长，因为需要构建完整的文档。
建议在发布前或专门的文档验证阶段运行。
"""

# 导入所有测试模块，使其可以被测试发现机制找到
from . import test_build
from . import test_config
from . import test_content
from . import test_extensions

# 定义模块的公共接口
__all__ = [
    'test_build',
    'test_config',
    'test_content',
    'test_extensions'
]

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 文档测试套件'

# 文档测试的特殊标记
__test_category__ = 'documentation'
__execution_priority__ = 'low'  # 文档测试可以在最后执行
__requires_build_env__ = True  # 需要文档构建环境