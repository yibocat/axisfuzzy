#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 20:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Pytest configuration and shared fixtures for random module tests.

This module provides common test fixtures, utilities, and configuration
for testing the AxisFuzzy random generation system. It includes fixtures
for mock generators, test data, and shared test utilities.
"""

import pytest
import numpy as np
import threading
import time
from typing import Dict, Any, Tuple, Union, Optional
from unittest.mock import Mock, patch

# Import the modules we're testing
from axisfuzzy.random.base import BaseRandomGenerator, ParameterizedRandomGenerator
from axisfuzzy.random.registry import RandomGeneratorRegistry
from axisfuzzy.random.seed import GlobalRandomState
from axisfuzzy.core import Fuzznum, Fuzzarray


@pytest.fixture
def clean_registry():
    """
    Fixture that provides a clean registry for each test.
    
    This fixture ensures that each test starts with a fresh registry
    state and cleans up after the test completes.
    """
    # Get the registry instance
    registry = RandomGeneratorRegistry()
    
    # Store original state
    original_generators = registry._generators.copy()
    
    # Clear registry for test
    registry.clear()
    
    yield registry
    
    # Restore original state
    registry.clear()
    registry._generators.update(original_generators)


@pytest.fixture
def clean_global_state():
    """
    Fixture that provides a clean global random state for each test.
    
    This fixture ensures that each test starts with a fresh random state
    and restores the original state after the test completes.
    """
    from axisfuzzy.random.seed import _global_random_state
    
    # Store original state
    original_seed = _global_random_state._seed
    original_rng = _global_random_state._rng
    
    # Reset to default state
    _global_random_state.set_seed(None)
    
    yield _global_random_state
    
    # Restore original state
    _global_random_state._seed = original_seed
    _global_random_state._rng = original_rng


@pytest.fixture
def mock_fuzznum():
    """
    Fixture that provides a mock Fuzznum instance for testing.
    """
    mock_num = Mock(spec=Fuzznum)
    mock_num.mtype = 'test'
    mock_num.q = 1
    return mock_num


@pytest.fixture
def mock_fuzzarray():
    """
    Fixture that provides a mock Fuzzarray instance for testing.
    """
    mock_array = Mock(spec=Fuzzarray)
    mock_array.mtype = 'test'
    mock_array.q = 1
    mock_array.shape = (10,)
    mock_array.size = 10
    return mock_array


@pytest.fixture
def sample_fuzznum():
    """
    Fixture that provides a sample Fuzznum instance for testing.
    
    This is an alias for mock_fuzznum to maintain compatibility
    with existing test code.
    """
    mock_num = Mock(spec=Fuzznum)
    mock_num.mtype = 'test'
    mock_num.q = 1
    return mock_num


@pytest.fixture
def sample_rng():
    """
    Fixture that provides a sample NumPy random generator for testing.
    """
    return np.random.default_rng(42)


@pytest.fixture
def thread_barrier():
    """
    Fixture that provides a threading barrier for concurrent testing.
    
    This is useful for testing thread safety by synchronizing multiple
    threads to start operations simultaneously.
    """
    return threading.Barrier(2)  # Default to 2 threads


class MockRandomGenerator(BaseRandomGenerator):
    """
    Mock random generator for testing purposes.
    
    This class provides a simple implementation of BaseRandomGenerator
    that can be used in tests without requiring actual fuzzy number
    generation logic.
    """
    
    mtype = 'mock'
    
    def __init__(self, **default_params):
        self._default_params = default_params or {'param1': 1.0, 'param2': 2.0}
        self.call_count = 0
        self.last_params = None
        self.last_rng = None
        self.last_shape = None
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return self._default_params.copy()
    
    def validate_parameters(self, **params) -> None:
        # Simple validation - just check for negative values
        for key, value in params.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"Parameter {key} cannot be negative")
    
    def fuzznum(self, rng: np.random.Generator, **params) -> Mock:
        self.call_count += 1
        self.last_params = params
        self.last_rng = rng
        
        # Return a mock Fuzznum
        mock_num = Mock(spec=Fuzznum)
        mock_num.mtype = self.mtype
        mock_num.q = params.get('q', 1)
        return mock_num
    
    def fuzzarray(self, rng: np.random.Generator, shape: Tuple[int, ...], **params) -> Mock:
        self.call_count += 1
        self.last_params = params
        self.last_rng = rng
        self.last_shape = shape
        
        # Return a mock Fuzzarray
        mock_array = Mock(spec=Fuzzarray)
        mock_array.mtype = self.mtype
        mock_array.q = params.get('q', 1)
        mock_array.shape = shape
        mock_array.size = np.prod(shape)
        return mock_array


class MockParameterizedGenerator(ParameterizedRandomGenerator):
    """
    Mock parameterized generator for testing ParameterizedRandomGenerator utilities.
    """
    
    mtype = 'mock_param'
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'q': 1,
            'low': 0.0,
            'high': 1.0,
            'distribution': 'uniform'
        }
    
    def validate_parameters(self, **params) -> None:
        q = params.get('q', 1)
        if not isinstance(q, int) or q < 1:
            raise ValueError("q must be a positive integer")
        
        low = params.get('low', 0.0)
        high = params.get('high', 1.0)
        if low >= high:
            raise ValueError("low must be less than high")
    
    def fuzznum(self, rng: np.random.Generator, **params) -> Mock:
        # Use the distribution sampling utility
        value = self._sample_from_distribution(
            rng, 
            dist=params.get('distribution', 'uniform'),
            low=params.get('low', 0.0),
            high=params.get('high', 1.0)
        )
        
        mock_num = Mock(spec=Fuzznum)
        mock_num.mtype = self.mtype
        mock_num.q = params.get('q', 1)
        mock_num.value = value
        return mock_num
    
    def fuzzarray(self, rng: np.random.Generator, shape: Tuple[int, ...], **params) -> Mock:
        # Use the distribution sampling utility for array generation
        values = self._sample_from_distribution(
            rng,
            size=np.prod(shape),
            dist=params.get('distribution', 'uniform'),
            low=params.get('low', 0.0),
            high=params.get('high', 1.0)
        )
        
        mock_array = Mock(spec=Fuzzarray)
        mock_array.mtype = self.mtype
        mock_array.q = params.get('q', 1)
        mock_array.shape = shape
        mock_array.size = np.prod(shape)
        mock_array.values = values.reshape(shape) if hasattr(values, 'reshape') else values
        return mock_array


@pytest.fixture
def mock_generator():
    """
    Fixture that provides a Mock generator for testing.
    """
    mock_gen = Mock(spec=BaseRandomGenerator)
    mock_gen.mtype = 'mock'
    mock_gen.get_default_parameters.return_value = {'param1': 1.0, 'param2': 2.0}
    mock_gen.validate_parameters = Mock()
    return mock_gen


@pytest.fixture
def mock_parameterized_generator():
    """
    Fixture that provides a Mock parameterized generator for testing.
    """
    mock_gen = Mock(spec=ParameterizedRandomGenerator)
    mock_gen.mtype = 'mock_param'
    mock_gen.get_default_parameters.return_value = {
        'alpha': 1.0,
        'beta': 2.0,
        'gamma': 0.5
    }
    mock_gen.validate_parameters = Mock()
    mock_gen.merge_parameters = Mock(side_effect=lambda **kwargs: kwargs)
    return mock_gen


@pytest.fixture
def concurrent_test_helper():
    """
    Fixture that provides utilities for concurrent testing.
    """
    class ConcurrentTestHelper:
        def __init__(self):
            self.results = []
            self.exceptions = []
            self.lock = threading.Lock()
        
        def run_concurrent(self, func, args_list, num_threads=None):
            """
            Run a function concurrently with different arguments.
            
            Parameters
            ----------
            func : callable
                Function to run concurrently
            args_list : list
                List of argument tuples for each thread
            num_threads : int, optional
                Number of threads to use (defaults to len(args_list))
            
            Returns
            -------
            tuple
                (results, exceptions) - lists of results and exceptions
            """
            if num_threads is None:
                num_threads = len(args_list)
            
            threads = []
            barrier = threading.Barrier(num_threads)
            
            def worker(args):
                try:
                    # Wait for all threads to be ready
                    barrier.wait()
                    result = func(*args)
                    with self.lock:
                        self.results.append(result)
                except Exception as e:
                    with self.lock:
                        self.exceptions.append(e)
            
            # Start threads
            for args in args_list[:num_threads]:
                thread = threading.Thread(target=worker, args=(args,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            return self.results.copy(), self.exceptions.copy()
    
    return ConcurrentTestHelper()


# Test data fixtures
@pytest.fixture
def sample_test_data():
    """
    Fixture that provides sample test data for various test scenarios.
    """
    return {
        'valid_mtypes': ['qrofn', 'qrohfn', 'ivfn', 'fs'],
        'invalid_mtypes': ['', 'invalid', 'nonexistent', None, 123],
        'valid_shapes': [(), (5,), (3, 4), (2, 3, 4)],
        'invalid_shapes': [(-1,), (0,), (1, -2), 'invalid'],
        'valid_seeds': [42, 0, 2**32-1, np.random.SeedSequence(42)],
        'invalid_seeds': [-1, 'invalid', [], {}],
        'valid_q_values': [1, 2, 3, 5, 10],
        'invalid_q_values': [0, -1, 1.5, 'invalid', None],
    }


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """
    Fixture that provides a simple performance timer for testing.
    """
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed_time
        
        @property
        def elapsed_time(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return PerformanceTimer()