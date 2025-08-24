#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 00:13
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


_PANDAS_INSTALLED = None
_PANDAS_ERROR_MSG = (
    "Pandas is required for the axisfuzzy.analysis module but is not installed. "
    "Please install it via `pip install pandas`."
)


def _import_pandas():
    """
    Lazily imports pandas and caches the result.

    Returns:
        The pandas module object.

    Raises:
        ImportError: If pandas is not installed.
    """
    global _PANDAS_INSTALLED
    if _PANDAS_INSTALLED is None:
        try:
            import pandas as pd
            _PANDAS_INSTALLED = pd
        except ImportError:
            raise ImportError(_PANDAS_ERROR_MSG)
    return _PANDAS_INSTALLED


# 我们可以选择性地暴露一些核心组件，但暂时保持为空
# 这样用户必须显式地从子模块导入，例如:
# from axisfuzzy.analysis.pipeline import FuzzyPipeline

# TODO: 怎么设置一下为可选依赖？ 当开发者导入 axisfuzzy 的时候不导入 pandas,
#  只有当 from axisfuzzy import analysis 的时候才导入 pandas
