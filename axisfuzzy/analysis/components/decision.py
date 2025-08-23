#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 11:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pandas as pd
import numpy as np

from .base import AnalysisComponent
from ..contracts import contract, ScoreVector, RankingResult


class Ranker(AnalysisComponent):
    """
    Ranks alternatives based on their scores.
    """

    def __init__(self, ascending: bool = False):
        """
        Initializes the ranker.

        Parameters
        ----------
        ascending : bool, default False
            If True, ranks in ascending order (lower scores are better).
            If False, ranks in descending order (higher scores are better).
        """
        self.ascending = ascending

    @contract(
        inputs={'scores': 'ScoreVector', 'alternative_names': 'RankingResult'},
        outputs={'ranking': 'RankingResult'}
    )
    def run(self, scores: ScoreVector, alternative_names: list) -> RankingResult:
        """
        Ranks the alternatives.

        Parameters
        ----------
        scores : ScoreVector
            The scores for each alternative.
        alternative_names : list
            The names of the alternatives, in the same order as the scores.

        Returns
        -------
        RankingResult
            A list of alternative names, sorted by rank.
        """
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        sort_indices = np.argsort(scores)
        if not self.ascending:
            sort_indices = sort_indices[::-1]  # Reverse for descending order

        ranked_names = [alternative_names[i] for i in sort_indices]
        return ranked_names
