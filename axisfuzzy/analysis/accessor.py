#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 15:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the Pandas Accessor for fuzzy analysis.

This module provides the `.fuzzy` accessor on pandas DataFrames, serving as the
primary user-facing entry point for the entire fuzzy analysis toolkit.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Union, Tuple, Optional

from axisfuzzy.fuzzifier import Fuzzifier

# Lazy import for optional dependency
try:
    import pandas as pd
except ImportError:
    pd = None

if TYPE_CHECKING:
    from .pipeline import FuzzyPipeline
    from .dataframe import FuzzyDataFrame


@pd.api.extensions.register_dataframe_accessor("fuzzy")
class FuzzyAccessor:
    """
    A pandas DataFrame accessor for fuzzy analysis tools.

    This accessor provides a ``.fuzzy`` namespace on any pandas DataFrame,
    allowing users to seamlessly integrate fuzzy logic and analysis capabilities
    into their standard data analysis workflows.

    It serves as a facade, providing convenient entry points to the more
    complex underlying components like `FuzzyDataFrame` and `FuzzyPipeline`.

    Parameters
    ----------
    pandas_obj : pd.DataFrame
        The pandas DataFrame instance this accessor is bound to.
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._df = pandas_obj
        self._validate_obj()

    def _validate_obj(self):
        """Internal validation to ensure the accessor is bound to a DataFrame."""
        if pd is None:
            raise ImportError(
                "pandas is not installed. The '.fuzzy' accessor requires "
                "pandas to be installed. Please run 'pip install pandas'."
            )
        if not isinstance(self._df, pd.DataFrame):
            # This case is unlikely due to how accessors are registered,
            # but it's good practice for robustness.
            raise AttributeError(
                "FuzzyAccessor can only be used with a pandas DataFrame."
            )

    def info(self) -> str:
        """
        Provides a brief summary of the accessor's state.

        Returns
        -------
        str
            A string containing information about the bound DataFrame.
        """
        return (
            f"<FuzzyAccessor>\n"
            f"Bound to: pandas.DataFrame\n"
            f"Shape: {self._df.shape}\n"
            f"Index: {self._df.index.__class__.__name__}\n"
            f"Columns: {list(self._df.columns)}"
        )

    def to_fuzz_dataframe(self,
                          fuzzifier: Optional[Fuzzifier] = None) -> 'FuzzyDataFrame':
        """
        Converts the pandas DataFrame into a high-performance FuzzyDataFrame.

        This method serves as a bridge from the crisp data world of pandas to
        the fuzzy data world of AxisFuzzy. It uses the `FuzzyDataFrame.from_pandas`
        class method for the conversion.

        Parameters
        ----------
        fuzzifier: Fuzzifier
            A fuzzifier with pre-set parameters, used to perform fuzzification
            on data in a pd.DataFrame

        Returns
        -------
        FuzzyDataFrame
            A new FuzzyDataFrame instance containing the fuzzy representation
            of the original data.
        """

        from .dataframe import FuzzyDataFrame  # Lazy import
        # We need to add a `from_pandas` class method to FuzzyDataFrame
        return FuzzyDataFrame.from_pandas(self._df, fuzzifier)

    def run(self,
            pipeline: 'FuzzyPipeline',
            return_intermediate: bool = False
            ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Executes a fuzzy analysis pipeline using the bound DataFrame as initial data.

        This is the main entry point for running a pre-defined `FuzzyPipeline`
        graph. The accessor automatically injects the bound DataFrame as the
        initial data handle named 'init_data'.

        Parameters
        ----------
        pipeline : FuzzyPipeline
            A `FuzzyPipeline` instance that defines the analysis workflow.
        return_intermediate : bool, default False
            If True, the pipeline will return the final results and a dictionary
            of all intermediate states. Otherwise, it returns only the final results.

        Returns
        -------
        dict[str, Any] or tuple[dict, dict]
            The result of the pipeline execution.
        """
        # Lazy import to avoid circular dependencies
        from .pipeline import FuzzyPipeline

        if not isinstance(pipeline, FuzzyPipeline):
            raise TypeError(
                f"Expected a FuzzyPipeline object, but got {type(pipeline).__name__}."
            )

        initial_data: Dict[str, Any]

        num_inputs = len(pipeline._input_nodes)

        if num_inputs == 0:
            # 如果 Pipeline 没有定义任何输入，但用户尝试用 df 运行它，这是个错误
            raise ValueError("The pipeline has no defined inputs, "
                             "but is being run with a DataFrame.")

        elif num_inputs == 1:
            # 如果只有一个输入节点，自动将 df 注入，无需关心其名称
            # 获取那个唯一的输入节点的名称
            input_name = list(pipeline._input_nodes.keys())[0]
            initial_data = {input_name: self._df}

        else:   # num_inputs > 1
            # 如果有多个输入节点，则遵循约定：必须有一个名为 'init_data' 的节点
            if 'init_data' not in pipeline._input_nodes:
                raise ValueError(
                    f"The pipeline has multiple inputs ({list(pipeline._input_nodes.keys())}), "
                    "but no default 'init_data' input was found to inject the DataFrame into. "
                    "Please name one of your inputs 'init_data'."
                )
            initial_data = {'init_data': self._df}

        return pipeline.run(initial_data, return_intermediate)

