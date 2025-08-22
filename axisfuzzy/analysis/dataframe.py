#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 14:41
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the FuzzyDataFrame, a high-performance, labeled, two-dimensional
fuzzy data structure.
"""

from typing import Dict, Any, List, Union, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from axisfuzzy.core import Fuzzarray
    from axisfuzzy.fuzzifier import Fuzzifier


class FuzzyDataFrame:
    """
    A two-dimensional, size-mutable, tabular data structure with labeled
    axes (rows and columns) for fuzzy data.

    ``FuzzyDataFrame`` is conceptually similar to a ``pandas.DataFrame``, but is
    specifically designed to hold and operate on fuzzy numbers efficiently.
    Internally, it stores data in a column-oriented fashion where each column
    is a high-performance ``axisfuzzy.core.Fuzzarray``.

    Parameters
    -----------
    data : dict[str, Fuzzarray] or FuzzyDataFrame, optional The data to populate the
        FuzzyDataFrame. Can be a dictionary mapping column names to Fuzzarray objects,
        or another FuzzyDataFrame.
    index : pd.Index or list-like, optional Index to use for resulting frame. Will default to
        `RangeIndex` if no indexing information part of input data and no index provided.
    columns : pd.Index or list-like, optional Column labels to use for resulting frame. Will
        default to keys of the data dictionary if not provided.

    Attributes
    ----------
    index : pd.Index
        The row labels of the FuzzyDataFrame.
    columns : pd.Index
        The column labels of the FuzzyDataFrame.
    shape : tuple[int, int]
        A tuple representing the dimensionality of the FuzzyDataFrame.
    """

    def __init__(self,
                 data: Optional[Dict[str, 'Fuzzarray']] = None,
                 index: Optional[Union[pd.Index, List, np.ndarray]] = None,
                 columns: Optional[Union[pd.Index, List, str]] = None):

        from axisfuzzy.core import Fuzzarray
        if data is None:
            data = {}

        if isinstance(data, FuzzyDataFrame):
            # Handle copy-from-another-FuzzyDataFrame case
            if index is None:
                index = data.index.copy()
            if columns is None:
                columns = data.columns.copy()
            data = {col: data[col].copy() for col in columns}

        self._data: Dict[str, Fuzzarray] = data
        self._validate_data()

        # Establish columns
        if columns is None:
            self._columns = pd.Index(data.keys())
        else:
            self._columns = pd.Index(columns)

        # Establish index
        if index is None:
            if not data:
                self._index = pd.Index([])
            else:
                # Infer index from the first Fuzzarray's length
                first_col_len = len(next(iter(data.values())))
                self._index = pd.RangeIndex(stop=first_col_len)
        else:
            self._index = pd.Index(index)

        # Final validation of dimensions
        if len(self._data) > 0:
            if any(len(arr) != len(self._index) for arr in self._data.values()):
                raise ValueError("All Fuzzarray columns must be of the same length.")

    def _validate_data(self):
        """Internal validation of the data dictionary."""
        from axisfuzzy.core import Fuzzarray
        for name, col in self._data.items():
            if not isinstance(col, Fuzzarray):
                raise TypeError(f"Column '{name}' must be a Fuzzarray, "
                                f"not {type(col).__name__}.")

    @classmethod
    def from_pandas(cls,
                    df: pd.DataFrame,
                    fuzzifier: 'Fuzzifier') -> 'FuzzyDataFrame':
        """
        Creates a FuzzyDataFrame from a crisp pandas DataFrame using a provided fuzzifier.

        Each column of the input DataFrame is converted into a Fuzzarray using
        the specified fuzzifier.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with crisp, numerical data.
        fuzzifier : Fuzzifier
            An already configured `axisfuzzy.fuzzifier.Fuzzifier` instance.

        Returns
        -------
        FuzzyDataFrame
            A new instance populated with Fuzzarray columns.
        """
        from axisfuzzy.fuzzifier import Fuzzifier
        # Already imported or use TYPE_CHECKING guard
        if not isinstance(fuzzifier, Fuzzifier):
            raise TypeError("fuzzifier must be an instance of axisfuzzy.fuzzifier.Fuzzifier.")

        fuzzy_data = {}
        for col_name in df.columns:
            crisp_column_data = df[col_name].values
            fuzzy_data[col_name] = fuzzifier(crisp_column_data)

        return cls(data=fuzzy_data, index=df.index, columns=df.columns)

    @property
    def shape(self) -> tuple[int, int]:
        """Return a tuple representing the dimensionality of the FuzzyDataFrame."""
        return len(self._index), len(self._columns)

    @property
    def index(self) -> pd.Index:
        """The row labels of the FuzzyDataFrame."""
        return self._index

    @property
    def columns(self) -> pd.Index:
        """The column labels of the FuzzyDataFrame."""
        return self._columns

    def __len__(self) -> int:
        """Return the number of rows."""
        return self.shape[0]

    def __getitem__(self, key: str) -> 'Fuzzarray':
        """
        Retrieve a column as a Fuzzarray.

        Parameters
        ----------
        key : str
            The name of the column to retrieve.

        Returns
        -------
        Fuzzarray
            The fuzzy array corresponding to the column.
        """
        if key not in self._columns:
            raise KeyError(f"Column '{key}' not found.")
        return self._data[key]

    def __setitem__(self, key: str, value: 'Fuzzarray'):
        """
        Set or add a column.

        The new column's length must match the existing index length.

        Parameters
        ----------
        key : str
            The name of the column.
        value : Fuzzarray
            The Fuzzarray to be set as the column.
        """
        if not isinstance(value, Fuzzarray):
            raise TypeError("Value must be a Fuzzarray.")
        if len(value) != len(self):
            raise ValueError(f"Length of values ({len(value)}) does not match "
                             f"length of index ({len(self)}).")

        self._data[key] = value
        if key not in self._columns:
            self._columns = self._columns.append(pd.Index([key]))

    # --- Display Logic (for debugging/inspection only) ---
    def _create_repr_df(self) -> pd.DataFrame:
        """
        Internal helper to create a DataFrame for string representation.
        This materializes Fuzzarrays for display and is not a public API.
        """
        # This is an expensive operation, used only for display.
        df_data = {}
        for col_name in self.columns:
            fuzz_arr = self._data[col_name]
            # Materialize the Fuzzarray into a list of Fuzznum objects for display
            df_data[col_name] = [fuzz_arr[i] for i in range(len(fuzz_arr))]

        return pd.DataFrame(df_data, index=self.index, columns=self.columns)

    def __repr__(self) -> str:
        """Return a string representation for the FuzzyDataFrame."""
        if self.shape[0] == 0:
            return f"FuzzyDataFrame(columns={list(self.columns)}, index={list(self.index)})"
        # Use the internal helper for display
        return self._create_repr_df().__repr__()
