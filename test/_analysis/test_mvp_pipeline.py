#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 11:40
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

# file: tests/_analysis/test_mvp_pipeline.py
import pytest
import pandas as pd
import numpy as np

from axisfuzzy.analysis._pipeline import FuzzyPipeline
from axisfuzzy.analysis._components.fuzzify import FuzzifyComponent
from axisfuzzy.analysis._components.weighting import EntropyWeightTool
from axisfuzzy.analysis._components.aggregation import WeightedAggregationTool
from axisfuzzy.analysis._components.decision import Ranker
from axisfuzzy.fuzzifier import Fuzzifier

# Ensure accessor is registered
import axisfuzzy.analysis.accessor


@pytest.fixture
def crisp_data() -> pd.DataFrame:
    """Provides a standard crisp DataFrame for tests."""
    return pd.DataFrame({
        'cost': [10, 8, 5],  # Lower is better
        'safety': [6, 7, 9],  # Higher is better
        'comfort': [7, 6, 8]  # Higher is better
    }, index=['CarA', 'CarB', 'CarC'])


def test_end_to_end_mvp_pipeline(crisp_data):
    """
    Tests a full, end-to-end decision-making pipeline using the new
    component-based architecture.
    """
    # 1. Instantiate all necessary _components with their configurations
    # Note: We create a Fuzzifier instance to inject into our FuzzifyComponent
    fuzzifier_engine = Fuzzifier(mtype='qrofn', mf='TriangularMF', mf_params={'a': 0, 'b': 5, 'c': 10})

    fuzzifier = FuzzifyComponent(fuzzifier=fuzzifier_engine)
    weighter = EntropyWeightTool()
    aggregator = WeightedAggregationTool(scoring_method='mean')
    ranker = Ranker(ascending=False)  # Higher score is better

    # 2. Build the pipeline by chaining component methods
    p = FuzzyPipeline()

    # Define inputs. Note: We need two inputs, the matrix and the list of names for the final ranker.
    init_matrix = p.input("matrix_data", contract="CrispTable")
    alt_names = p.input("alternative_names", contract="RankingResult")

    # Build the graph
    fuzz_table = p.add(fuzzifier.run, data=init_matrix)
    weights = p.add(weighter.run, matrix=fuzz_table)
    scores = p.add(aggregator.run, matrix=fuzz_table, weights=weights)
    final_ranking = p.add(ranker.run, scores=scores, alternative_names=alt_names)

    # 3. Prepare initial data and execute the pipeline
    initial_pipeline_data = {
        "matrix_data": crisp_data,
        "alternative_names": list(crisp_data.index)
    }

    # The pipeline's run method will execute the graph
    result = p.run(initial_pipeline_data)

    # 4. Assertions
    # Based on the data, 'CarC' should be the best option (low cost, high safety/comfort)
    # The exact scores depend on the entropy and aggregation logic, but we can
    # assert the type and the top choice.
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == 'CarC', f"Expected 'CarC' to be top-ranked, but got {result[0]}"
    assert set(result) == {'CarA', 'CarB', 'CarC'}

    print("\n✅ Phase 2 End-to-End MVP Pipeline Test: PASSED")
