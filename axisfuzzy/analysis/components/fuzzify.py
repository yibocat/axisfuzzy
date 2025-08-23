#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 11:27
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import Optional, TYPE_CHECKING
import pandas as pd

from .base import AnalysisComponent
from ..contracts import contract, CrispTable, FuzzyTable
from ...fuzzifier import Fuzzifier

if TYPE_CHECKING:
    from ..dataframe import FuzzyDataFrame


class FuzzifyComponent(AnalysisComponent):
    """
    An analysis component for converting crisp data into fuzzy data.

    This component wraps the core `axisfuzzy.fuzzifier.Fuzzifier` engine,
    allowing it to be seamlessly integrated into an analysis pipeline. It is
    configured during instantiation with all necessary fuzzification parameters.

    Parameters
    ----------
    fuzzifier : Fuzzifier
        A pre-configured instance of the `Fuzzifier` class. This approach
        promotes separation of concerns, where the fuzzification logic is
        defined once and then passed to this pipeline component.
    """
    def __init__(self, fuzzifier: Fuzzifier):
        if not isinstance(fuzzifier, Fuzzifier):
            raise TypeError("fuzzifier must be an instance of axisfuzzy.fuzzifier.Fuzzifier.")
        self.fuzzifier = fuzzifier

    @contract(inputs={'data': 'CrispTable'}, outputs={'result': 'FuzzyTable'})
    def run(self, data: CrispTable) -> FuzzyTable:
        """
        Executes the fuzzification process on the input data.

        Parameters
        ----------
        data : CrispTable
            A pandas DataFrame with crisp, numerical data.

        Returns
        -------
        FuzzyTable
            A FuzzyDataFrame containing the fuzzy representation of the data.
        """
        from ..dataframe import FuzzyDataFrame  # Lazy import for performance
        return FuzzyDataFrame.from_pandas(data, self.fuzzifier)











































