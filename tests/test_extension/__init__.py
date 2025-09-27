"""AxisFuzzy 扩展系统测试套件

本模块包含对 AxisFuzzy 扩展系统的全面测试，包括：
- 扩展注册表测试
- 装饰器功能测试
- 方法分发测试
- 动态注入测试
- 外部扩展测试
- 边界情况测试

测试文件：
- test_registry.py: 扩展注册表核心功能测试
- test_decorator.py: 扩展装饰器功能测试
- test_dispatch.py: 方法分发机制测试
- test_injection.py: 动态注入机制测试
- test_external_extension.py: 外部扩展装饰器测试
- test_edge_cases.py: 边界情况和异常处理测试
- test_integration.py: 扩展系统集成测试
- test_performance.py: 扩展系统性能测试

扩展系统提供了灵活的功能扩展机制，支持：
- 传统扩展装饰器 (@extension)
- 外部扩展装饰器 (@external_extension)
- 自动和手动注入控制
- 类型安全的方法分发
- 高性能的运行时扩展
"""

# 导入已存在的测试模块，使其可以被测试发现机制找到
from . import test_registry
from . import test_decorator
from . import test_external_extension
from . import test_dispatch
from . import test_injection

# 定义模块的公共接口
__all__ = [
    'test_registry',
    'test_decorator',
    'test_external_extension',
    'test_dispatch',
    'test_injection',
]

# 注意：其他测试模块将在创建后逐步添加到这里
# 'test_dispatch',
# 'test_injection',
# 'test_external_extension',
# 'test_edge_cases',
# 'test_integration',
# 'test_performance'

# 测试模块元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 扩展系统测试套件'

# 扩展测试的特殊标记
__test_category__ = 'extension'
__execution_priority__ = 'high'  # 扩展系统是核心功能
__includes_performance_tests__ = True  # 包含性能测试
__includes_external_tests__ = True  # 包含外部扩展测试

# 测试配置
TEST_CONFIG = {
    'timeout': 30,  # 单个测试超时时间（秒）
    'retry_count': 3,  # 失败重试次数
    'parallel_safe': True,  # 是否支持并行测试
    'requires_cleanup': True,  # 是否需要测试后清理
}

# 扩展测试的依赖检查
REQUIRED_MODULES = [
    'axisfuzzy.extension',
    'axisfuzzy.extension.registry',
    'axisfuzzy.core.fuzznum',
    'axisfuzzy.core.fuzzarray'
]

def check_test_dependencies():
    """检查扩展测试的依赖是否满足
    
    Returns:
        bool: 如果所有依赖都满足返回 True，否则返回 False
    """
    import importlib
    
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(f"缺少必需的模块: {module_name} - {e}")
            return False
    
    return True

def get_test_summary():
    """获取扩展测试套件的摘要信息
    
    Returns:
        dict: 包含测试套件信息的字典
    """
    return {
        'name': '扩展系统测试套件',
        'version': __version__,
        'category': __test_category__,
        'priority': __execution_priority__,
        'test_count': len(__all__),
        'test_modules': __all__,
        'config': TEST_CONFIG,
        'dependencies': REQUIRED_MODULES
    }