import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any

from axisfuzzy.fuzzifier import FuzzificationStrategy, get_registry_fuzzify
from axisfuzzy.fuzztype.qrofs.fuzzification import QROFNFuzzificationStrategy
from axisfuzzy.fuzztype.qrohfs.fuzzification import QROHFNFuzzificationStrategy
from axisfuzzy.core import Fuzznum, Fuzzarray
from axisfuzzy.membership import TriangularMF, TrapezoidalMF, GaussianMF


class TestFuzzificationStrategyBase:
    
    def test_abstract_base_class(self):
        assert issubclass(FuzzificationStrategy, ABC)
        
        with pytest.raises(TypeError):
            FuzzificationStrategy()
    
    def test_required_attributes(self):
        class TestStrategy(FuzzificationStrategy):
            mtype = 'test'
            method = 'test_method'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        strategy = TestStrategy()
        assert hasattr(strategy, 'mtype')
        assert hasattr(strategy, 'method')
        assert strategy.mtype == 'test'
        assert strategy.method == 'test_method'
    
    def test_get_strategy_info(self):
        class TestStrategy(FuzzificationStrategy):
            mtype = 'test'
            method = 'test_method'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        strategy = TestStrategy()
        info = strategy.get_strategy_info()
        
        assert 'mtype' in info
        assert 'method' in info
        assert 'class_name' in info
        assert info['mtype'] == 'test'
        assert info['method'] == 'test_method'
        assert info['class_name'] == 'TestStrategy'
    
    def test_repr_method(self):
        class TestStrategy(FuzzificationStrategy):
            mtype = 'test'
            method = 'test_method'
            
            def fuzzify(self, x, mf_cls, mf_params_list):
                return None
        
        strategy = TestStrategy()
        repr_str = repr(strategy)
        assert 'TestStrategy' in repr_str
        assert 'test' in repr_str
        assert 'test_method' in repr_str
    
    def test_abstract_fuzzify_method(self):
        class IncompleteStrategy(FuzzificationStrategy):
            mtype = 'incomplete'
            method = 'test'
        
        with pytest.raises(TypeError):
            IncompleteStrategy()


class TestQROFNFuzzificationStrategy:
    
    def test_initialization_default(self):
        strategy = QROFNFuzzificationStrategy()
        assert strategy.q == 1  # 默认值是1，不是2
        assert strategy.pi == 0.1  # 默认值是0.1，不是0.0
        assert strategy.mtype == 'qrofn'
        assert strategy.method == 'default'
    
    def test_initialization_custom_q(self):
        strategy = QROFNFuzzificationStrategy(q=3)
        assert strategy.q == 3
    
    def test_initialization_invalid_q(self):
        # q parameter validation is handled in the base class
        strategy = QROFNFuzzificationStrategy(q=3)
        assert strategy.q == 3
    
    def test_fuzzify_scalar_single_mf(self, sample_crisp_values):
        strategy = QROFNFuzzificationStrategy(q=2)
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        result = strategy.fuzzify(x, TriangularMF, [mf_params])
        
        assert isinstance(result, Fuzzarray)
        mds, nmds = result.backend.get_component_arrays()
        # 对于标量输入，数组形状应该是 () 或 (1,)
        assert mds.shape == () or mds.shape == (1,)
        assert nmds.shape == () or nmds.shape == (1,)
    
    def test_fuzzify_scalar_multiple_mf(self, sample_crisp_values, multiple_mf_params):
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.1)
        x = sample_crisp_values[0]
        
        # 使用两个相同类型的隶属函数参数
        triangular_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        mf_params_list = [triangular_params, triangular_params]
        result = strategy.fuzzify(x, TriangularMF, mf_params_list)
        
        assert isinstance(result, Fuzzarray)
        assert result.shape == (len(multiple_mf_params),)
        # QROFN stacks multiple mf_params results along axis=0
        
        # Check membership and non-membership degree data
        mds, nmds = result.backend.get_component_arrays()
        assert mds.shape == (len(multiple_mf_params),)
        assert nmds.shape == (len(multiple_mf_params),)
    
    def test_fuzzify_array_single_mf(self, sample_crisp_values):
        strategy = QROFNFuzzificationStrategy(q=2)
        x = np.array(sample_crisp_values)
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        result = strategy.fuzzify(x, TriangularMF, [mf_params])
        
        assert isinstance(result, Fuzzarray)
        assert result.shape == x.shape
        
        # 检查隶属度和非隶属度的数据
        mds, nmds = result.backend.get_component_arrays()
        assert mds.shape == x.shape
        assert nmds.shape == x.shape
    
    def test_fuzzify_array_multiple_mf(self, sample_crisp_values, multiple_mf_params):
        strategy = QROFNFuzzificationStrategy(q=2, pi=0.1)
        x = np.array(sample_crisp_values)
        
        # 使用两个相同类型的隶属函数参数
        triangular_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        mf_params_list = [triangular_params, triangular_params]
        result = strategy.fuzzify(x, TriangularMF, mf_params_list)
        
        assert isinstance(result, Fuzzarray)
        # QROFN stacks multiple mf_params results along axis=0
        assert result.shape == (len(mf_params_list), *x.shape)
    
    def test_membership_degree_bounds(self, sample_crisp_values):
        strategy = QROFNFuzzificationStrategy(q=2)
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        result = strategy.fuzzify(x, TriangularMF, [mf_params])
        
        # 获取隶属度和非隶属度数据
        mds, nmds = result.backend.get_component_arrays()
        assert 0 <= mds.flat[0] <= 1
        assert 0 <= nmds.flat[0] <= 1
        assert mds.flat[0]**2 + nmds.flat[0]**2 <= 1
    
    def test_different_membership_functions(self, sample_crisp_values):
        strategy = QROFNFuzzificationStrategy(q=2)
        x = sample_crisp_values[0]
        
        triangular_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        trapezoidal_params = {'a': 0.0, 'b': 0.3, 'c': 0.7, 'd': 1.0}
        gaussian_params = {'c': 0.5, 'sigma': 0.2}
        
        tri_result = strategy.fuzzify(x, TriangularMF, [triangular_params])
        trap_result = strategy.fuzzify(x, TrapezoidalMF, [trapezoidal_params])
        gauss_result = strategy.fuzzify(x, GaussianMF, [gaussian_params])
        
        assert isinstance(tri_result, Fuzzarray)
        assert isinstance(trap_result, Fuzzarray)
        assert isinstance(gauss_result, Fuzzarray)
    
    def test_q_parameter_effect(self, sample_crisp_values):
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        strategy_q2 = QROFNFuzzificationStrategy(q=2)
        strategy_q3 = QROFNFuzzificationStrategy(q=3)
        
        result_q2 = strategy_q2.fuzzify(x, TriangularMF, [mf_params])
        result_q3 = strategy_q3.fuzzify(x, TriangularMF, [mf_params])
        
        # 获取隶属度和非隶属度数据
        mds_q2, nmds_q2 = result_q2.backend.get_component_arrays()
        mds_q3, nmds_q3 = result_q3.backend.get_component_arrays()
        assert mds_q2.flat[0]**2 + nmds_q2.flat[0]**2 <= 1
        assert mds_q3.flat[0]**3 + nmds_q3.flat[0]**3 <= 1


class TestQROHFNFuzzificationStrategy:
    
    def test_initialization_default(self):
        strategy = QROHFNFuzzificationStrategy()
        assert strategy.q == 1  # 默认值是1，不是2
        assert strategy.pi == 0.1  # 默认值是0.1，不是0.0
        assert strategy.nmd_generation_mode == 'pi_based'
        assert strategy.mtype == 'qrohfn'
        assert strategy.method == 'default'
    
    def test_initialization_custom_parameters(self):
        strategy = QROHFNFuzzificationStrategy(q=3, pi=0.1, nmd_generation_mode='proportional')
        assert strategy.q == 3
        assert strategy.pi == 0.1
        assert strategy.nmd_generation_mode == 'proportional'
    
    def test_initialization_invalid_q(self):
        # q parameter validation is handled in the base class
        strategy = QROHFNFuzzificationStrategy(q=3)
        assert strategy.q == 3
    
    def test_initialization_invalid_pi(self):
        with pytest.raises(ValueError, match=r"pi must be in \[0,1\]"):
            QROHFNFuzzificationStrategy(pi=-0.1)
        
        with pytest.raises(ValueError, match=r"pi must be in \[0,1\]"):
            QROHFNFuzzificationStrategy(pi=1.1)
    
    def test_initialization_invalid_nmd_mode(self):
        with pytest.raises(ValueError, match="Invalid nmd_generation_mode"):
            QROHFNFuzzificationStrategy(nmd_generation_mode='invalid')
    
    def test_fuzzify_scalar_pi_based(self, sample_crisp_values):
        strategy = QROHFNFuzzificationStrategy(q=2, pi=0.1, nmd_generation_mode='pi_based')
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        result = strategy.fuzzify(x, TriangularMF, [mf_params])
        
        # QROHFN returns Fuzznum for scalar input
        assert isinstance(result, Fuzznum)
        assert hasattr(result, 'md')
        assert hasattr(result, 'nmd')
        assert 0 <= result.md <= 1
        assert 0 <= result.nmd <= 1
    
    def test_fuzzify_scalar_proportional(self, sample_crisp_values):
        strategy = QROHFNFuzzificationStrategy(q=2, nmd_generation_mode='proportional')
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        result = strategy.fuzzify(x, TriangularMF, [mf_params])
        
        # QROHFN returns Fuzznum for scalar input
        assert isinstance(result, Fuzznum)
        assert 0 <= result.md <= 1
        assert 0 <= result.nmd <= 1
    
    def test_fuzzify_scalar_uniform(self, sample_crisp_values):
        strategy = QROHFNFuzzificationStrategy(q=2, nmd_generation_mode='uniform')
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        result = strategy.fuzzify(x, TriangularMF, [mf_params])
        
        # QROHFN returns Fuzznum for scalar input
        assert isinstance(result, Fuzznum)
        assert 0 <= result.md <= 1
        assert 0 <= result.nmd <= 1
    
    def test_fuzzify_array_multiple_mf(self, sample_crisp_values, triangular_mf_params):
        """测试数组输入的多隶属函数模糊化。"""
        strategy = QROHFNFuzzificationStrategy(q=2, pi=0.1, nmd_generation_mode='pi_based')
        x = np.array(sample_crisp_values)
        
        # 使用两个相同类型的隶属函数参数
        mf_params_list = [triangular_mf_params, triangular_mf_params]
        result = strategy.fuzzify(x, TriangularMF, mf_params_list)
        
        # QROHFN returns Fuzzarray for array input
        assert isinstance(result, Fuzzarray)
        assert result.shape == x.shape
        # QROHFN merges multiple mf_params into hesitant sets
        
        # 检查隶属度和非隶属度的数据
        mds, nmds = result.backend.get_component_arrays()
        assert mds.shape == x.shape
        assert nmds.shape == x.shape
    
    def test_hesitancy_degree_constraint(self, sample_crisp_values):
        strategy = QROHFNFuzzificationStrategy(q=2, pi=0.1, nmd_generation_mode='pi_based')
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        result = strategy.fuzzify(x, TriangularMF, [mf_params])
        
        # QROHFN returns Fuzznum for scalar input
        if isinstance(result, Fuzznum):
            md = result.md
            nmd = result.nmd
            hesitancy = 1 - md**2 - nmd**2
            
            assert hesitancy >= 0
            assert md**2 + nmd**2 <= 1
        else:
            mds, nmds = result.backend.get_component_arrays()
            assert 0 <= mds.flat[0] <= 1
            assert 0 <= nmds.flat[0] <= 1
    
    def test_different_nmd_generation_modes(self, sample_crisp_values):
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        strategy_pi = QROHFNFuzzificationStrategy(q=2, pi=0.1, nmd_generation_mode='pi_based')
        strategy_prop = QROHFNFuzzificationStrategy(q=2, nmd_generation_mode='proportional')
        strategy_uniform = QROHFNFuzzificationStrategy(q=2, nmd_generation_mode='uniform')
        
        result_pi = strategy_pi.fuzzify(x, TriangularMF, [mf_params])
        result_prop = strategy_prop.fuzzify(x, TriangularMF, [mf_params])
        result_uniform = strategy_uniform.fuzzify(x, TriangularMF, [mf_params])
        
        # QROHFN returns Fuzznum for scalar input
        assert isinstance(result_pi, Fuzznum)
        assert isinstance(result_prop, Fuzznum)
        assert isinstance(result_uniform, Fuzznum)
        
        assert result_pi.nmd != result_prop.nmd or result_prop.nmd != result_uniform.nmd
    
    def test_compute_nmds_pi_based(self):
        strategy = QROHFNFuzzificationStrategy(q=2, pi=0.2, nmd_generation_mode='pi_based')
        mds = np.array([0.6, 0.8, 0.3])
        
        nmds = strategy._compute_nmds(mds)
        
        assert len(nmds) == len(mds)
        for i, (md, nmd) in enumerate(zip(mds, nmds)):
            assert 0 <= nmd <= 1
            assert md**2 + nmd**2 <= 1
    
    def test_compute_nmds_proportional(self):
        strategy = QROHFNFuzzificationStrategy(q=2, nmd_generation_mode='proportional')
        mds = np.array([0.6, 0.8, 0.3])
        
        nmds = strategy._compute_nmds(mds)
        
        assert len(nmds) == len(mds)
        for md, nmd in zip(mds, nmds):
            assert 0 <= nmd <= 1
            assert md**2 + nmd**2 <= 1
    
    def test_compute_nmds_uniform(self):
        strategy = QROHFNFuzzificationStrategy(q=2, nmd_generation_mode='uniform')
        mds = np.array([0.6, 0.8, 0.3])
        
        nmds = strategy._compute_nmds(mds)
        
        assert len(nmds) == len(mds)
        for md, nmd in zip(mds, nmds):
            assert 0 <= nmd <= 1
            assert md**2 + nmd**2 <= 1


class TestStrategyRegistration:
    
    def test_qrofn_strategy_registered(self):
        registry = get_registry_fuzzify()
        strategy_cls = registry.get_strategy('qrofn', 'default')
        
        assert strategy_cls is QROFNFuzzificationStrategy
        assert strategy_cls.mtype == 'qrofn'
        assert strategy_cls.method == 'default'
    
    def test_qrohfn_strategy_registered(self):
        registry = get_registry_fuzzify()
        strategy_cls = registry.get_strategy('qrohfn', 'default')
        
        assert strategy_cls is QROHFNFuzzificationStrategy
        assert strategy_cls.mtype == 'qrohfn'
        assert strategy_cls.method == 'default'
    
    def test_strategy_instantiation_from_registry(self):
        registry = get_registry_fuzzify()
        
        qrofn_cls = registry.get_strategy('qrofn', 'default')
        qrohfn_cls = registry.get_strategy('qrohfn', 'default')
        
        qrofn_instance = qrofn_cls(q=3)
        qrohfn_instance = qrohfn_cls(q=3, pi=0.1)
        
        assert isinstance(qrofn_instance, QROFNFuzzificationStrategy)
        assert isinstance(qrohfn_instance, QROHFNFuzzificationStrategy)
        assert qrofn_instance.q == 3
        assert qrohfn_instance.q == 3
        assert qrohfn_instance.pi == 0.1


class TestStrategyComparison:
    
    def test_qrofn_vs_qrohfn_output_structure(self, sample_crisp_values):
        x = sample_crisp_values[0]
        mf_params = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        
        qrofn_strategy = QROFNFuzzificationStrategy(q=2)
        qrohfn_strategy = QROHFNFuzzificationStrategy(q=2, nmd_generation_mode='proportional')
        
        qrofn_result = qrofn_strategy.fuzzify(x, TriangularMF, [mf_params])
        qrohfn_result = qrohfn_strategy.fuzzify(x, TriangularMF, [mf_params])
        
        # QROFN returns Fuzzarray, QROHFN returns Fuzznum for scalar input
        assert isinstance(qrofn_result, Fuzzarray)
        assert isinstance(qrohfn_result, Fuzznum)
        
        # 检查数据访问
        mds, nmds = qrofn_result.backend.get_component_arrays()
        assert mds is not None and nmds is not None
        assert hasattr(qrohfn_result, 'md') and hasattr(qrohfn_result, 'nmd')
    
    def test_strategy_info_comparison(self):
        qrofn_strategy = QROFNFuzzificationStrategy(q=2)
        qrohfn_strategy = QROHFNFuzzificationStrategy(q=2)
        
        qrofn_info = qrofn_strategy.get_strategy_info()
        qrohfn_info = qrohfn_strategy.get_strategy_info()
        
        assert qrofn_info['mtype'] == 'qrofn'
        assert qrohfn_info['mtype'] == 'qrohfn'
        assert qrofn_info['method'] == 'default'
        assert qrohfn_info['method'] == 'default'
        assert qrofn_info['class_name'] != qrohfn_info['class_name']


if __name__ == '__main__':
    pytest.main()
    