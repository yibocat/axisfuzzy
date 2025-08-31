#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 00:13
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from . import app

# 分析模块所需的核心依赖
# _ANALYSIS_DEPENDENCIES = {
#     'pandas': 'DataFrame operations and data analysis',
#     'matplotlib': 'plotting and visualization',
#     'networkx': 'graph analysis and network operations',
#     'pydot': 'graph visualization with Graphviz (requires system Graphviz installation)'
# }


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

from .app import Model

__all__ = [
    "app",
    "Model",
    "check_analysis_dependencies",
]
