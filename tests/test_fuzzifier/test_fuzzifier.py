#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/25 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import numpy as np
from typing import Dict, List, Any

from axisfuzzy.membership import TriangularMF, GaussianMF, TrapezoidalMF, get_mf_class
from axisfuzzy.fuzzifier import Fuzzifier
from axisfuzzy.core import Fuzznum, Fuzzarray
from axisfuzzy.config import get_config


class TestFuzzifierInitialization:
    """测试Fuzzifier的各种初始化方式"""
    
    def test_init_with_mf_instance(self):
        """测试使用隶属函数实例初始化"""
        mf_instance = GaussianMF(sigma=0.2, c=0.5)
        fuzzifier = Fuzzifier(mf=mf_instance, mtype='qrofn')
        
        assert fuzzifier.mf_cls == GaussianMF
        assert fuzzifier.provided_mf_instance == mf_instance
        assert fuzzifier.mtype == 'qrofn'
        assert len(fuzzifier.mf_params_list) == 1
        assert fuzzifier.mf_params_list[0] == {'sigma': 0.2, 'c': 0.5}
    
    def test_init_with_mf_instance_override_params(self):
        """测试使用隶属函数实例但覆盖参数"""
        mf_instance = GaussianMF(sigma=0.2, c=0.5)
        fuzzifier = Fuzzifier(
            mf=mf_instance, 
            mtype='qrofn',
            mf_params={'sigma': 0.3, 'c': 0.6}
        )
        
        assert fuzzifier.mf_cls == GaussianMF
        assert fuzzifier.provided_mf_instance == mf_instance
        assert fuzzifier.mf_params_list[0] == {'sigma': 0.3, 'c': 0.6}
    
    def test_init_with_mf_class(self):
        """测试使用隶属函数类初始化"""
        fuzzifier = Fuzzifier(
            mf=GaussianMF,
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5}
        )
        
        assert fuzzifier.mf_cls == GaussianMF
        assert fuzzifier.provided_mf_instance is None
        assert fuzzifier.mf_params_list[0] == {'sigma': 0.2, 'c': 0.5}
    
    def test_init_with_mf_string(self):
        """测试使用字符串名称初始化"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5}
        )
        
        assert fuzzifier.mf_cls == GaussianMF
        assert fuzzifier.provided_mf_instance is None
        assert fuzzifier.mf_params_list[0] == {'sigma': 0.2, 'c': 0.5}
    
    def test_init_with_multiple_mf_params_dict(self):
        """测试使用多个参数字典初始化"""
        params = [
            {'sigma': 0.1, 'c': 0.3},
            {'sigma': 0.2, 'c': 0.7}
        ]
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrohfn',
            mf_params=params
        )
        
        assert len(fuzzifier.mf_params_list) == 2
        assert fuzzifier.mf_params_list == params
    
    def test_init_with_multiple_mf_params_list(self):
        """测试使用参数列表初始化"""
        params = [
            {'a': 0.0, 'b': 0.3, 'c': 0.6},
            {'a': 0.4, 'b': 0.7, 'c': 1.0}
        ]
        fuzzifier = Fuzzifier(
            mf=TriangularMF,
            mtype='qrohfn',
            mf_params=params
        )
        
        assert len(fuzzifier.mf_params_list) == 2
        assert fuzzifier.mf_params_list == params
    
    def test_init_default_mtype(self):
        """测试默认模糊数类型"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mf_params={'sigma': 0.2, 'c': 0.5}
        )
        
        assert fuzzifier.mtype == get_config().DEFAULT_MTYPE
    
    def test_init_default_method(self):
        """测试默认方法"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5}
        )
        
        assert fuzzifier.method == 'default'
    
    def test_init_missing_mf_params_without_instance(self):
        """测试缺少mf_params参数时的错误处理"""
        with pytest.raises(ValueError, match="requires 'mf_params' argument"):
            Fuzzifier(mf='gaussmf', mtype='qrofn')
    
    def test_init_invalid_mf_params_type(self):
        """测试无效的mf_params类型"""
        with pytest.raises(TypeError, match="must be either a dict, or a list of dicts"):
            Fuzzifier(
                mf='gaussmf',
                mtype='qrofn',
                mf_params="invalid"
            )
    
    def test_init_invalid_mf_params_list(self):
        """测试无效的mf_params列表"""
        with pytest.raises(TypeError, match="must be either a dict, or a list of dicts"):
            Fuzzifier(
                mf='gaussmf',
                mtype='qrofn',
                mf_params=[{'sigma': 0.2}, "invalid"]
            )
    
    def test_init_invalid_mtype(self):
        """测试无效的模糊数类型"""
        with pytest.raises(ValueError, match="No default method for mtype"):
            Fuzzifier(
                mf='gaussmf',
                mtype='invalid_type',
                mf_params={'sigma': 0.2, 'c': 0.5}
            )
    
    def test_init_invalid_method(self):
        """测试无效的方法"""
        with pytest.raises(ValueError, match="No strategy found"):
            Fuzzifier(
                mf='gaussmf',
                mtype='qrofn',
                method='invalid_method',
                mf_params={'sigma': 0.2, 'c': 0.5}
            )
    
    def test_init_invalid_mf_name(self):
        """测试无效的隶属函数名称"""
        with pytest.raises(ValueError, match="Unknown membership function"):
            Fuzzifier(
                mf='invalid_mf',
                mtype='qrofn',
                mf_params={'sigma': 0.2, 'c': 0.5}
            )
    
    def test_init_strategy_parameters(self):
        """测试策略参数传递"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=3,
            pi=0.2
        )
        
        assert fuzzifier.strategy.q == 3
        assert fuzzifier.strategy.pi == 0.2


class TestFuzzifierCall:
    """测试Fuzzifier的调用功能"""
    
    def test_call_single_value(self):
        """测试单个值的模糊化"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        result = fuzzifier(0.5)
        
        assert isinstance(result, Fuzzarray)
        assert result.shape == ()
    
    def test_call_list_values(self):
        """测试列表值的模糊化"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        result = fuzzifier([0.3, 0.5, 0.7])
        
        assert isinstance(result, Fuzzarray)
        assert result.shape == (3,)
    
    def test_call_numpy_array(self):
        """测试NumPy数组的模糊化"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = fuzzifier(x)
        
        assert isinstance(result, Fuzzarray)
        assert result.shape == (5,)
    
    def test_call_multidimensional_array(self):
        """测试多维数组的模糊化"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        x = np.array([[0.1, 0.3], [0.5, 0.7]])
        result = fuzzifier(x)
        
        assert isinstance(result, Fuzzarray)
        assert result.shape == (2, 2)
    
    def test_call_multiple_mf_params(self):
        """测试多个隶属函数参数的模糊化"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params=[
                {'sigma': 0.1, 'c': 0.3},
                {'sigma': 0.2, 'c': 0.7}
            ],
            q=2
        )
        
        result = fuzzifier(0.5)
        
        assert isinstance(result, Fuzzarray)
        assert result.shape == (2,)
    
    def test_call_different_mf_types(self):
        """测试不同类型的隶属函数"""
        # 测试三角形隶属函数
        tri_fuzzifier = Fuzzifier(
            mf=TriangularMF,
            mtype='qrofn',
            mf_params={'a': 0.0, 'b': 0.5, 'c': 1.0},
            q=2
        )
        
        # 测试高斯隶属函数
        gauss_fuzzifier = Fuzzifier(
            mf=GaussianMF,
            mtype='qrofn',
            mf_params={'sigma': 0.1, 'c': 0.5},
            q=2
        )
        
        x = 0.3  # 使用一个会产生明显不同结果的值
        tri_result = tri_fuzzifier(x)
        gauss_result = gauss_fuzzifier(x)
        
        assert isinstance(tri_result, Fuzzarray)
        assert isinstance(gauss_result, Fuzzarray)
        # 不同隶属函数应该产生不同的结果
        tri_mds, _ = tri_result.backend.get_component_arrays()
        gauss_mds, _ = gauss_result.backend.get_component_arrays()
        assert not np.allclose(tri_mds, gauss_mds, rtol=1e-3)
    
    def test_call_qrohfn_strategy(self):
        """测试QROHFN策略的调用"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrohfn',
            mf_params=[
                {'sigma': 0.1, 'c': 0.3},
                {'sigma': 0.05, 'c': 0.6}
            ],
            q=2,
            nmd_generation_mode='proportional'
        )
        
        result = fuzzifier(0.5)
        
        assert isinstance(result, (Fuzznum, Fuzzarray))
        # QROHFN处理多个参数集时返回hesitant结构


class TestFuzzifierConfiguration:
    """测试Fuzzifier的配置序列化和反序列化"""
    
    def test_get_config_basic(self):
        """测试基本配置获取"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2,
            pi=0.1
        )
        
        config = fuzzifier.get_config()
        
        assert config['mf'] == 'gaussmf'
        assert config['mtype'] == 'qrofn'
        assert config['mf_params'] == {'sigma': 0.2, 'c': 0.5}
        assert config['q'] == 2
        assert config['pi'] == 0.1
    
    def test_get_config_with_mf_class(self):
        """测试使用隶属函数类时的配置获取"""
        fuzzifier = Fuzzifier(
            mf=GaussianMF,
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        config = fuzzifier.get_config()
        
        # 类应该被转换为字符串名称
        assert config['mf'] == 'ABCMeta'
    
    def test_from_config_basic(self):
        """测试从配置重建Fuzzifier"""
        original_config = {
            'mf': 'gaussmf',
            'mtype': 'qrofn',
            'method': None,
            'mf_params': {'sigma': 0.2, 'c': 0.5},
            'q': 2,
            'pi': 0.1
        }
        
        fuzzifier = Fuzzifier.from_config(original_config)
        
        assert fuzzifier.mf_cls == GaussianMF
        assert fuzzifier.mtype == 'qrofn'
        assert fuzzifier.mf_params_list[0] == {'sigma': 0.2, 'c': 0.5}
        assert fuzzifier.strategy.q == 2
        assert fuzzifier.strategy.pi == 0.1
    
    def test_config_roundtrip(self):
        """测试配置的完整往返"""
        original = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params=[
                {'sigma': 0.1, 'c': 0.3},
                {'sigma': 0.2, 'c': 0.7}
            ],
            q=3,
            pi=0.15
        )
        
        config = original.get_config()
        restored = Fuzzifier.from_config(config)
        
        # 验证关键属性相同
        assert restored.mf_cls == original.mf_cls
        assert restored.mtype == original.mtype
        assert restored.mf_params_list == original.mf_params_list
        assert restored.strategy.q == original.strategy.q
        assert restored.strategy.pi == original.strategy.pi
        
        # 验证功能相同
        x = 0.5
        original_result = original(x)
        restored_result = restored(x)
        
        orig_mds, orig_nmds = original_result.backend.get_component_arrays()
        rest_mds, rest_nmds = restored_result.backend.get_component_arrays()
        
        assert np.allclose(orig_mds, rest_mds)
        assert np.allclose(orig_nmds, rest_nmds)


class TestFuzzifierPlotting:
    """测试Fuzzifier的绘图功能"""
    
    def test_plot_basic(self):
        """测试基本绘图功能"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        # 测试不显示图像
        try:
            fuzzifier.plot(show=False)
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_plot_with_multiple_params(self):
        """测试多参数绘图"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params=[
                {'sigma': 0.1, 'c': 0.3},
                {'sigma': 0.2, 'c': 0.7}
            ],
            q=2
        )
        
        try:
            fuzzifier.plot(show=False)
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_plot_custom_range(self):
        """测试自定义范围绘图"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        try:
            fuzzifier.plot(x_range=(-1, 2), num_points=50, show=False)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestFuzzifierEdgeCases:
    """测试Fuzzifier的边界情况和错误处理"""
    
    def test_extreme_values(self):
        """测试极值处理"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        # 测试极小值
        result_min = fuzzifier(-1000)
        assert isinstance(result_min, Fuzzarray)
        
        # 测试极大值
        result_max = fuzzifier(1000)
        assert isinstance(result_max, Fuzzarray)
        
        # 测试零值
        result_zero = fuzzifier(0)
        assert isinstance(result_zero, Fuzzarray)
    
    def test_empty_array(self):
        """测试空数组处理"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        # 空数组应该返回相应形状的结果
        empty_array = np.array([])
        result = fuzzifier(empty_array)
        assert isinstance(result, Fuzzarray)
        assert result.shape == (0,)
    
    def test_large_array(self):
        """测试大数组处理"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        # 测试较大的数组
        large_array = np.random.rand(100)
        result = fuzzifier(large_array)
        assert isinstance(result, Fuzzarray)
        assert result.shape == (100,)
    
    def test_nan_inf_values(self):
        """测试NaN和无穷值处理"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        # 测试包含NaN的数组
        nan_array = np.array([0.1, np.nan, 0.5])
        result_nan = fuzzifier(nan_array)
        assert isinstance(result_nan, Fuzzarray)
        
        # 测试包含无穷值的数组
        inf_array = np.array([0.1, np.inf, -np.inf])
        result_inf = fuzzifier(inf_array)
        assert isinstance(result_inf, Fuzzarray)


class TestFuzzifierStrategies:
    """测试不同模糊化策略的集成"""
    
    def test_qrofn_strategy(self):
        """测试QROFN策略"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2,
            pi=0.1
        )
        
        result = fuzzifier(0.5)
        assert isinstance(result, Fuzzarray)
        assert fuzzifier.strategy.mtype == 'qrofn'
        assert fuzzifier.strategy.q == 2
        assert fuzzifier.strategy.pi == 0.1
        
        # 验证结果结构
        mds, nmds = result.backend.get_component_arrays()
        assert mds.shape == nmds.shape
    
    def test_qrohfn_strategy(self):
        """测试QROHFN策略"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrohfn',
            mf_params=[
                {'sigma': 0.1, 'c': 0.3},
                {'sigma': 0.2, 'c': 0.7}
            ],
            q=2,
            nmd_generation_mode='proportional'
        )
        
        result = fuzzifier(0.5)
        assert isinstance(result, (Fuzznum, Fuzzarray))
        assert fuzzifier.strategy.mtype == 'qrohfn'
        assert fuzzifier.strategy.q == 2
        assert fuzzifier.strategy.nmd_generation_mode == 'proportional'
    
    def test_different_q_values(self):
        """测试不同的q值"""
        for q in [1, 2, 3, 5, 10]:
            fuzzifier = Fuzzifier(
                mf='gaussmf',
                mtype='qrofn',
                mf_params={'sigma': 0.2, 'c': 0.5},
                q=q
            )
            
            result = fuzzifier(0.5)
            assert isinstance(result, Fuzzarray)
            assert fuzzifier.strategy.q == q
    
    def test_strategy_parameters(self):
        """测试策略参数传递"""
        # 测试QROFN的pi参数
        fuzzifier_qrofn = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2,
            pi=0.2
        )
        
        assert fuzzifier_qrofn.strategy.pi == 0.2
        
        # 测试QROHFN的nmd_generation_mode参数
        fuzzifier_qrohfn = Fuzzifier(
            mf='gaussmf',
            mtype='qrohfn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2,
            nmd_generation_mode='uniform'
        )
        
        assert fuzzifier_qrohfn.strategy.nmd_generation_mode == 'uniform'


class TestFuzzifierRepr:
    """测试Fuzzifier的字符串表示"""
    
    def test_repr_basic(self):
        """测试基本字符串表示"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            mf_params={'sigma': 0.2, 'c': 0.5},
            q=2
        )
        
        repr_str = repr(fuzzifier)
        assert 'Fuzzifier' in repr_str
        assert 'method=\'default\'' in repr_str
        assert 'mtype=\'qrofn\'' in repr_str
        assert 'GaussianMF' in repr_str
    
    def test_repr_with_multiple_params(self):
        """测试多参数的字符串表示"""
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrohfn',
            mf_params=[
                {'sigma': 0.1, 'c': 0.3},
                {'sigma': 0.2, 'c': 0.7}
            ],
            q=2
        )
        
        repr_str = repr(fuzzifier)
        assert 'Fuzzifier' in repr_str
        assert 'qrohfn' in repr_str
        assert len(fuzzifier.mf_params_list) == 2
    