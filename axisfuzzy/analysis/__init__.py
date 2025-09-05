#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 00:13
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

# 延迟导入策略：避免在模块级别直接导入可选依赖
# 这样可以确保核心包安装时不会因为缺少可选依赖而导致导入错误

# 分析模块所需的核心依赖
_ANALYSIS_DEPENDENCIES = {
    'pandas': 'DataFrame operations and data analysis',
    'matplotlib': 'plotting and visualization',
    'networkx': 'graph analysis and network operations',
    'pydot': 'graph visualization with Graphviz (requires system Graphviz installation)'
}


def _check_graphviz_installation():
    """
    检查系统是否安装了 Graphviz。
    
    Returns
    -------
    bool
        如果 Graphviz 可用返回 True，否则返回 False
    """
    import subprocess
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_analysis_dependencies():
    """
    检查所有分析模块依赖的安装状态。
    
    Returns
    -------
    dict
        包含每个依赖包安装状态的字典
        格式: {dep_key: {'installed': bool, 'version': str or None, 'error': str or None}}
    """
    status = {}

    # 检查Python包依赖
    import_map = {
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'networkx': 'networkx',
        'pydot': 'pydot'
    }

    for dep_key, import_name in import_map.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            status[dep_key] = {
                'installed': True,
                'version': version,
                'error': None
            }
        except ImportError:
            status[dep_key] = {
                'installed': False,
                'version': None,
                'error': f"Package '{dep_key}' not installed. Install with: pip install 'axisfuzzy[analysis]'"
            }

    # 检查系统级Graphviz安装
    graphviz_available = _check_graphviz_installation()
    status['graphviz'] = {
        'installed': graphviz_available,
        'version': 'system' if graphviz_available else None,
        'error': None if graphviz_available else "Graphviz not found. Install with: brew install graphviz (macOS) or apt-get install graphviz (Ubuntu)"
    }

    return status


# 我们可以选择性地暴露一些核心组件，但暂时保持为空
# 这样用户必须显式地从子模块导入，例如:
# from axisfuzzy.analysis.pipeline import FuzzyPipeline

# 延迟导入函数，只有在实际使用时才导入
def _lazy_import_model():
    """延迟导入 Model 类，避免在模块级别导入 pandas 等依赖。"""
    try:
        from .app import Model
        return Model
    except ImportError as e:
        raise ImportError(
            f"Cannot import Model: {e}. "
            "Please install analysis dependencies with: pip install 'axisfuzzy[analysis]'"
        ) from e


def _lazy_import_fuzzy_dataframe():
    """延迟导入 FuzzyDataFrame 类。"""
    try:
        from .dataframe.frame import FuzzyDataFrame
        return FuzzyDataFrame
    except ImportError as e:
        raise ImportError(
            f"Cannot import FuzzyDataFrame: {e}. "
            "Please install analysis dependencies with: pip install 'axisfuzzy[analysis]'"
        ) from e


def _lazy_import_fuzzy_pipeline():
    """延迟导入 FuzzyPipeline 类。"""
    try:
        from .pipeline import FuzzyPipeline
        return FuzzyPipeline
    except ImportError as e:
        raise ImportError(
            f"Cannot import FuzzyPipeline: {e}. "
            "Please install analysis dependencies with: pip install 'axisfuzzy[analysis]'"
        ) from e


def _lazy_import_contracts():
    """延迟导入 contracts 模块的核心组件。"""
    try:
        from .contracts import Contract, contract
        return Contract, contract
    except ImportError as e:
        raise ImportError(
            f"Cannot import contracts: {e}. "
            "Please install analysis dependencies with: pip install 'axisfuzzy[analysis]'"
        ) from e


def _lazy_import_component():
    """延迟导入 component 模块的核心组件。"""
    try:
        from .component import AnalysisComponent
        return AnalysisComponent
    except ImportError as e:
        raise ImportError(
            f"Cannot import AnalysisComponent: {e}. "
            "Please install analysis dependencies with: pip install 'axisfuzzy[analysis]'"
        ) from e


# 使用 __getattr__ 实现延迟导入
def __getattr__(name):
    if name == "Model":
        return _lazy_import_model()
    elif name == "FuzzyDataFrame":
        return _lazy_import_fuzzy_dataframe()
    elif name == "FuzzyPipeline":
        return _lazy_import_fuzzy_pipeline()
    elif name == "Contract":
        Contract, _ = _lazy_import_contracts()
        return Contract
    elif name == "contract":
        _, contract = _lazy_import_contracts()
        return contract
    elif name == "AnalysisComponent":
        return _lazy_import_component()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 为了解决IDE类型检查问题，我们在模块级别提供类型声明
# 但实际导入仍然是延迟的
try:
    # 尝试导入以提供类型信息，但如果失败则忽略
    from .app.model import Model as _Model
    from .dataframe.frame import FuzzyDataFrame as _FuzzyDataFrame
    from .pipeline import FuzzyPipeline as _FuzzyPipeline
    from .contracts import Contract as _Contract, contract as _contract
    from .component import AnalysisComponent as _AnalysisComponent

    # 导入accessor模块以启用pandas扩展，但不导出FuzzyAccessor类
    try:
        from . import accessor  # 这会注册pandas扩展
    except ImportError:
        pass  # 如果pandas不可用，忽略accessor导入

    # 为了解决 IDE 类型检查问题，我们在成功导入后将这些组件赋值给模块级变量
    # 这样 IDE 就能识别 __all__ 中的名称了
    Model = _Model
    FuzzyDataFrame = _FuzzyDataFrame
    FuzzyPipeline = _FuzzyPipeline
    Contract = _Contract
    contract = _contract
    AnalysisComponent = _AnalysisComponent

    # 定义 __all__ 列表（在成功导入后）
    __all__ = [
        "check_analysis_dependencies",
        # 核心分析组件
        "Model",  # 高级模型抽象类
        "FuzzyDataFrame",  # 模糊数据框架
        "FuzzyPipeline",  # 分析管道
        # 契约系统 - 开发者自定义组件时需要
        "Contract",  # 数据契约类
        "contract",  # 契约装饰器
        # 组件基类 - 开发者自定义组件时需要
        "AnalysisComponent",  # 分析组件基类
    ]

except ImportError:
    # 如果依赖不可用，定义占位符类型以支持类型检查
    _Model = None
    _FuzzyDataFrame = None
    _FuzzyPipeline = None
    _Contract = None
    _contract = None
    _AnalysisComponent = None

    # 在依赖不可用时，只导出依赖检查函数
    __all__ = [
        "check_analysis_dependencies",
    ]
