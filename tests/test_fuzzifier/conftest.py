import pytest
import numpy as np
from typing import Dict, List, Any

from axisfuzzy.membership import create_mf, TriangularMF, GaussianMF, TrapezoidalMF
from axisfuzzy.fuzzifier import Fuzzifier, FuzzificationStrategy, get_registry_fuzzify
from axisfuzzy.core import Fuzznum, Fuzzarray
from axisfuzzy.config import get_config


@pytest.fixture
def sample_crisp_values():
    return [0.1, 0.3, 0.5, 0.7, 0.9]


@pytest.fixture
def sample_crisp_array():
    return np.array([0.2, 0.4, 0.6, 0.8])


@pytest.fixture
def sample_crisp_scalar():
    return 0.5


@pytest.fixture
def triangular_mf_params():
    return {'a': 0.0, 'b': 0.5, 'c': 1.0}


@pytest.fixture
def gaussian_mf_params():
    return {'c': 0.5, 'sigma': 0.2}


@pytest.fixture
def trapezoidal_mf_params():
    return {'a': 0.0, 'b': 0.3, 'c': 0.7, 'd': 1.0}


@pytest.fixture
def multiple_mf_params(triangular_mf_params, gaussian_mf_params):
    return [triangular_mf_params, gaussian_mf_params]


@pytest.fixture
def triangular_mf_instance(triangular_mf_params):
    return TriangularMF(**triangular_mf_params)


@pytest.fixture
def gaussian_mf_instance(gaussian_mf_params):
    return GaussianMF(**gaussian_mf_params)


@pytest.fixture
def basic_fuzzifier_qrofn(triangular_mf_params):
    return Fuzzifier(
        mf=TriangularMF,
        mtype='qrofn',
        method='default',
        mf_params=triangular_mf_params,
        q=2,
        pi=0.1
    )


@pytest.fixture
def basic_fuzzifier_qrohfn(multiple_mf_params):
    return Fuzzifier(
        mf=TriangularMF,
        mtype='qrohfn',
        method='default',
        mf_params=multiple_mf_params,
        q=2,
        pi=0.1
    )


@pytest.fixture
def fuzzifier_with_string_mf(triangular_mf_params):
    return Fuzzifier(
        mf='trimf',
        mtype='qrofn',
        mf_params=triangular_mf_params,
        q=2
    )


@pytest.fixture
def fuzzifier_with_mf_instance(triangular_mf_instance):
    return Fuzzifier(
        mf=triangular_mf_instance,
        mtype='qrofn',
        q=2
    )


@pytest.fixture
def registry_instance():
    return get_registry_fuzzify()


@pytest.fixture
def mock_strategy_class():
    class MockStrategy(FuzzificationStrategy):
        mtype = 'mock'
        method = 'test'
        
        def fuzzify(self, x, mf_cls, mf_params_list):
            return None
    
    return MockStrategy


@pytest.fixture
def test_data_ranges():
    return {
        'small': np.linspace(0, 1, 5),
        'medium': np.linspace(0, 1, 20),
        'large': np.linspace(0, 1, 100),
        'edge_cases': np.array([0.0, 0.001, 0.999, 1.0]),
        'negative': np.array([-0.5, -0.1, 0.0, 0.1, 0.5]),
        'beyond_unit': np.array([-0.2, 0.5, 1.2, 1.5])
    }


@pytest.fixture
def performance_test_data():
    return {
        'small_array': np.random.rand(100),
        'medium_array': np.random.rand(1000),
        'large_array': np.random.rand(10000),
        'multidim_array': np.random.rand(50, 50)
    }


@pytest.fixture
def config_backup():
    original_config = get_config().to_dict()
    yield original_config
    from axisfuzzy.config import reset_config
    reset_config()


@pytest.fixture
def suppress_warnings():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield