#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/25 13:10
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Tests for the Sequential application layer API.
"""

import pytest
import pandas as pd
import numpy as np

from axisfuzzy.analysis.app import Sequential
from axisfuzzy.analysis.component.basic import (
    ToolNormalization,
    ToolStatistics
)


class TestSequential:
    """Test suite for the Sequential class."""

    def test_sequential_creation(self):
        """Test basic Sequential model creation."""
        normalizer = ToolNormalization(method='min_max')
        stats = ToolStatistics()

        model = Sequential([normalizer, stats])

        assert len(model.steps) == 2
        assert isinstance(model.steps[0], ToolNormalization)
        assert isinstance(model.steps[1], ToolStatistics)
        assert model.name.startswith("Sequential_")

    def test_sequential_with_custom_name(self):
        """Test Sequential model with custom name."""
        normalizer = ToolNormalization()
        model = Sequential([normalizer], name="TestModel")

        assert model.name == "TestModel"

    def test_empty_steps_raises_error(self):
        """Test that empty steps list raises ValueError."""
        with pytest.raises(ValueError, match="Steps list cannot be empty"):
            Sequential([])

    def test_non_component_step_raises_error(self):
        """Test that non-AnalysisComponent steps raise TypeError."""
        def dummy_function():
            pass

        with pytest.raises(TypeError, match="must be an instance of AnalysisComponent"):
            Sequential([dummy_function])

    def test_lazy_pipeline_building(self):
        """Test that pipeline is built lazily."""
        normalizer = ToolNormalization()
        model = Sequential([normalizer])

        # Pipeline should not be built yet
        assert model._pipeline is None

        # Accessing .pipeline should trigger building
        pipeline = model.pipeline
        assert pipeline is not None
        assert model._pipeline is pipeline  # Should cache the result

    def test_basic_execution(self):
        """Test basic model execution with real data."""
        # Create test data
        test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        # Create model
        normalizer = ToolNormalization(method='min_max')
        stats = ToolStatistics()
        model = Sequential([normalizer, stats])

        # Run model
        result = model.run(test_df)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result
        assert 'min' in result
        assert 'max' in result

    def test_repr_and_str(self):
        """Test string representations."""
        normalizer = ToolNormalization()
        model = Sequential([normalizer], name="TestRepr")

        repr_str = repr(model)
        assert "TestRepr" in repr_str
        assert "ToolNormalization" in repr_str
        assert str(model) == repr_str
