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

# 核心分析组件的类型声明
from .app.model import Model as Model
from .dataframe.frame import FuzzyDataFrame as FuzzyDataFrame
from .pipeline import FuzzyPipeline as FuzzyPipeline

# 契约系统的类型声明
from .contracts.base import Contract as Contract
from .contracts.decorator import contract as contract

# 组件基类的类型声明
from .component.base import AnalysisComponent as AnalysisComponent

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

# 内部函数的类型声明（不在 __all__ 中，但提供类型声明以支持内部使用）
def _check_graphviz_installation() -> bool:
    """
    检查系统是否安装了 Graphviz。
    
    Returns
    -------
    bool
        如果 Graphviz 可用返回 True，否则返回 False
    """
    ...

# 延迟导入函数的类型声明
def _lazy_import_model() -> type[Model]: ...
def _lazy_import_fuzzy_dataframe() -> type[FuzzyDataFrame]: ...
def _lazy_import_fuzzy_pipeline() -> type[FuzzyPipeline]: ...
def _lazy_import_contracts() -> tuple[type[Contract], type[contract]]: ...
def _lazy_import_component() -> type[AnalysisComponent]: ...

# 声明 __all__ 列表的类型
__all__: List[str]