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

import inspect
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
    from .app.model import Model


@pd.api.extensions.register_dataframe_accessor("fuzzy")
class FuzzyAccessor:
    """
    A pandas DataFrame accessor for fuzzy analysis tools.

    This accessor provides a ``.fuzzy`` namespace on any pandas DataFrame,
    allowing users to seamlessly integrate fuzzy logic and analysis capabilities
    into their standard data analysis workflows.

    It serves as a facade, providing convenient entry points to the more
    complex underlying components like `FuzzyDataFrame`, `FuzzyPipeline`, and `Model`.

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
            executable: Union['FuzzyPipeline', 'Model'],
            return_intermediate: bool = False,
            **kwargs
            ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Executes a fuzzy analysis pipeline or model using the bound DataFrame as initial data.

        This is the main entry point for running a pre-defined `FuzzyPipeline`
        graph or a high-level `Model`. The accessor automatically injects the bound
        DataFrame as input data, and other inputs can be provided as keyword arguments.

        For a `FuzzyPipeline` or `Model`:
        - If there is one input, the DataFrame is passed to it.
        - If there are multiple inputs, the DataFrame is passed to the input named
          'init_data' (preferred) or 'data'.
        - Other required inputs must be provided as keyword arguments (`**kwargs`).
        - For a `Model`, if it's not built, it will be built automatically.

        Parameters
        ----------
        executable : FuzzyPipeline or Model
            A `FuzzyPipeline` or `Model` instance to execute.
        return_intermediate : bool, default False
            If True, returns both final and intermediate results.
        **kwargs :
            Additional keyword arguments to be passed as inputs to the model or pipeline.

        Returns
        -------
        dict[str, Any] or tuple[dict, dict]
            The result of the execution.

        Raises
        ------
        ValueError
            If input injection is ambiguous or a required input is missing.
        """
        # Lazy import to avoid circular dependencies
        from .pipeline import FuzzyPipeline
        from .app.model import Model

        if isinstance(executable, Model):
            model = executable
            if not model.built:
                model.build()

            sig = inspect.signature(model.forward)
            input_names = [p.name for p in sig.parameters.values() if p.name != 'self']

            run_kwargs = kwargs.copy()

            if len(input_names) == 1:
                df_input_name = input_names[0]
                if df_input_name in run_kwargs:
                    raise ValueError(
                        f"Ambiguous input: '{df_input_name}' was provided via both the "
                        f"DataFrame accessor and keyword arguments."
                    )
                run_kwargs[df_input_name] = self._df
            else:  # multiple inputs
                # By convention, inject DataFrame into 'init_data' or 'data'
                if 'init_data' in input_names:
                    df_input_name = 'init_data'
                elif 'data' in input_names:
                    df_input_name = 'data'
                else:
                    raise ValueError(
                        f"The model's 'forward' method has multiple inputs ({input_names}), "
                        f"but no 'init_data' or 'data' input was found to inject the DataFrame. "
                        f"Please name one of your inputs 'init_data' or 'data'."
                    )

                if df_input_name in run_kwargs:
                    raise ValueError(
                        f"Ambiguous input: '{df_input_name}' was provided via both the "
                        f"DataFrame accessor and keyword arguments."
                    )

                run_kwargs[df_input_name] = self._df

            return model.run(return_intermediate=return_intermediate, **run_kwargs)

        elif isinstance(executable, FuzzyPipeline):
            pipeline = executable
            input_names = list(pipeline._input_nodes.keys())
            num_inputs = len(input_names)

            initial_data = kwargs.copy()

            if num_inputs == 0:
                if kwargs:
                    raise ValueError("The pipeline has no defined inputs, but keyword arguments were provided.")
            elif num_inputs == 1:
                df_input_name = input_names[0]
                if df_input_name in initial_data:
                    raise ValueError(
                        f"Ambiguous input: '{df_input_name}' was provided via both the "
                        f"DataFrame accessor and keyword arguments."
                    )
                initial_data[df_input_name] = self._df
            else:  # num_inputs > 1
                if 'init_data' in input_names:
                    df_input_name = 'init_data'
                elif 'data' in input_names:
                    df_input_name = 'data'
                else:
                    raise ValueError(
                        f"The pipeline has multiple inputs ({input_names}), but no 'init_data' "
                        f"or 'data' input was found to inject the DataFrame. "
                        f"Please name one of your inputs 'init_data' or 'data'."
                    )

                if df_input_name in initial_data:
                    raise ValueError(
                        f"Ambiguous input: '{df_input_name}' was provided via both the "
                        f"DataFrame accessor and keyword arguments."
                    )

                initial_data[df_input_name] = self._df

            return pipeline.run(initial_data, return_intermediate)

        else:
            raise TypeError(
                f"Expected a FuzzyPipeline or Model object, but got {type(executable).__name__}."
            )