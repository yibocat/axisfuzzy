#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 00:13
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Type stubs for the axisfuzzy.analysis module.

This module provides type hints for the analysis functionality,
including dependency management and lazy loading mechanisms.
"""

from typing import Dict, Any, List, Optional, Union
from types import ModuleType

# 导入子模块
from . import app as app

# 依赖检查函数的类型声明
def check_analysis_dependencies() -> Dict[str, Dict[str, Union[bool, str, None]]]:
    """
    检查所有分析模块依赖的安装状态。
    
    Returns
    -------
    Dict[str, Dict[str, Union[bool, str, None]]]
        包含每个依赖包安装状态的字典
        格式: {dep_key: {'installed': bool, 'version': str or None, 'error': str or None}}
    """
    ...

# 内部依赖导入函数（不在 __all__ 中，但提供类型声明以支持内部使用）
def _import_dependency(dep_key: str) -> ModuleType:
    """
    延迟导入指定的依赖包并缓存结果。
    
    Parameters
    ----------
    dep_key : str
        依赖包的键名
        
    Returns
    -------
    ModuleType
        导入的模块对象
        
    Raises
    ------
    ImportError
        当依赖包未安装时抛出
    ValueError
        当请求的依赖包未在支持列表中时抛出
    """
    ...

def _check_graphviz_installation() -> bool:
    """
    检查系统是否安装了 Graphviz。
    
    Returns
    -------
    bool
        如果 Graphviz 可用返回 True，否则返回 False
    """
    ...

# 声明 __all__ 列表的类型
__all__: List[str]