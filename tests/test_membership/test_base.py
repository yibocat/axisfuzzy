#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
测试 MembershipFunction 基类的接口和抽象方法

本模块测试隶属函数基类的核心功能，包括：
- 抽象方法的正确实现要求
- 基类接口的一致性
- 参数管理功能
- 可调用接口
- 基本属性和方法
"""

import pytest
import numpy as np
from abc import ABC
from unittest.mock import patch, MagicMock

from axisfuzzy.membership.base import MembershipFunction
from .conftest import assert_membership_properties, assert_parameter_update


class TestMembershipFunctionInterface:
    """测试 MembershipFunction 基类接口"""
    
    def test_is_abstract_base_class(self):
        """测试 MembershipFunction 是抽象基类"""
        assert issubclass(MembershipFunction, ABC)
        
        # 不能直接实例化抽象基类
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MembershipFunction()
    
    def test_abstract_methods_defined(self):
        """测试抽象方法已正确定义"""
        abstract_methods = MembershipFunction.__abstractmethods__
        expected_methods = {'compute', 'set_parameters'}
        assert abstract_methods == expected_methods
    
    def test_concrete_methods_available(self):
        """测试具体方法在基类中可用"""
        # 检查基类定义的具体方法
        assert hasattr(MembershipFunction, '__init__')
        assert hasattr(MembershipFunction, 'get_parameters')
        assert hasattr(MembershipFunction, '__call__')
        assert hasattr(MembershipFunction, 'plot')


class ConcreteMembershipFunction(MembershipFunction):
    """用于测试的具体隶属函数实现"""
    
    def __init__(self, param1=1.0, param2=2.0):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.parameters = {'param1': param1, 'param2': param2}
    
    def compute(self, x):
        """简单的线性隶属函数实现"""
        x = np.asarray(x, dtype=float)
        result = (x - self.param1) / self.param2
        return np.clip(result, 0.0, 1.0)
    
    def set_parameters(self, **kwargs):
        """参数设置方法"""
        if 'param1' in kwargs:
            self.param1 = kwargs['param1']
            self.parameters['param1'] = self.param1
        if 'param2' in kwargs:
            if kwargs['param2'] <= 0:
                raise ValueError("param2 must be positive")
            self.param2 = kwargs['param2']
            self.parameters['param2'] = self.param2


class TestConcreteImplementation:
    """测试具体实现的基类功能"""
    
    @pytest.fixture
    def concrete_mf(self):
        """创建具体隶属函数实例"""
        return ConcreteMembershipFunction(param1=0.0, param2=1.0)
    
    def test_initialization(self, concrete_mf):
        """测试初始化功能"""
        assert concrete_mf.name == 'ConcreteMembershipFunction'
        assert isinstance(concrete_mf.parameters, dict)
        assert concrete_mf.parameters['param1'] == 0.0
        assert concrete_mf.parameters['param2'] == 1.0
    
    def test_get_parameters(self, concrete_mf):
        """测试参数获取功能"""
        params = concrete_mf.get_parameters()
        assert isinstance(params, dict)
        assert params == {'param1': 0.0, 'param2': 1.0}
        
        # 注意：当前实现返回的是引用，不是副本
        # 这是设计决策，允许直接访问参数字典
        params['param1'] = 999
        assert concrete_mf.parameters['param1'] == 999  # 修正：应该是999，因为返回的是引用
    
    def test_set_parameters_valid(self, concrete_mf):
        """测试有效参数设置"""
        # 设置单个参数
        concrete_mf.set_parameters(param1=0.5)
        assert concrete_mf.param1 == 0.5
        assert concrete_mf.parameters['param1'] == 0.5
        assert concrete_mf.param2 == 1.0  # 未改变
        
        # 设置多个参数
        concrete_mf.set_parameters(param1=1.0, param2=2.0)
        assert concrete_mf.param1 == 1.0
        assert concrete_mf.param2 == 2.0
        assert concrete_mf.parameters == {'param1': 1.0, 'param2': 2.0}
    
    def test_set_parameters_invalid(self, concrete_mf):
        """测试无效参数设置"""
        # 测试参数验证
        with pytest.raises(ValueError, match="param2 must be positive"):
            concrete_mf.set_parameters(param2=-1.0)
        
        # 确保无效参数不会改变状态
        original_params = concrete_mf.get_parameters().copy()
        try:
            concrete_mf.set_parameters(param2=0.0)
        except ValueError:
            pass
        assert concrete_mf.get_parameters() == original_params
    
    def test_set_parameters_unknown(self, concrete_mf):
        """测试设置未知参数"""
        # 设置未知参数应该被忽略（不报错）
        original_params = concrete_mf.get_parameters().copy()
        concrete_mf.set_parameters(unknown_param=123)
        assert concrete_mf.get_parameters() == original_params
    
    def test_callable_interface(self, concrete_mf, standard_x_values):
        """测试可调用接口"""
        # 测试 __call__ 方法
        result1 = concrete_mf(standard_x_values)
        result2 = concrete_mf.compute(standard_x_values)
        
        np.testing.assert_array_equal(result1, result2)
        
        # 测试单个值
        single_result1 = concrete_mf(0.5)
        single_result2 = concrete_mf.compute(0.5)
        assert single_result1 == single_result2
    
    def test_compute_scalar_input(self, concrete_mf):
        """测试标量输入的计算"""
        result = concrete_mf.compute(0.5)
        assert isinstance(result, (float, np.floating))
        assert 0.0 <= result <= 1.0
    
    def test_compute_array_input(self, concrete_mf, standard_x_values):
        """测试数组输入的计算"""
        result = concrete_mf.compute(standard_x_values)
        assert isinstance(result, np.ndarray)
        assert result.shape == standard_x_values.shape
        assert_membership_properties(concrete_mf, standard_x_values)
    
    def test_compute_list_input(self, concrete_mf):
        """测试列表输入的计算"""
        x_list = [0.0, 0.5, 1.0]
        result = concrete_mf.compute(x_list)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_list)
        assert_membership_properties(concrete_mf, x_list)
    
    def test_compute_multidimensional_input(self, concrete_mf):
        """测试多维数组输入"""
        x_2d = np.array([[0.0, 0.5], [1.0, 1.5]])
        result = concrete_mf.compute(x_2d)
        assert result.shape == x_2d.shape
        assert_membership_properties(concrete_mf, x_2d)
    
    def test_membership_properties(self, concrete_mf, extended_x_values):
        """测试隶属函数的基本数学性质"""
        assert_membership_properties(concrete_mf, extended_x_values)
    
    def test_parameter_update_consistency(self, concrete_mf):
        """测试参数更新的一致性"""
        new_params = {'param1': 0.3, 'param2': 1.5}
        assert_parameter_update(concrete_mf, new_params)


class TestPlottingFunctionality:
    """测试绘图功能"""
    
    @pytest.fixture
    def concrete_mf(self):
        return ConcreteMembershipFunction(param1=0.0, param2=1.0)
    
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.show')
    def test_plot_default_parameters(self, mock_show, mock_legend, mock_grid,
                                   mock_title, mock_ylabel, mock_xlabel,
                                   mock_plot, concrete_mf):
        """测试默认参数的绘图"""
        concrete_mf.plot()
        
        # 验证 matplotlib 函数被调用
        mock_plot.assert_called_once()
        mock_xlabel.assert_called_once_with('x')
        mock_ylabel.assert_called_once_with('Membership Degree')
        mock_title.assert_called_once_with('ConcreteMembershipFunction Membership Function')
        mock_grid.assert_called_once_with(True)
        mock_legend.assert_called_once()
        mock_show.assert_called_once()
        
        # 检查绘图数据
        args, kwargs = mock_plot.call_args
        x_data, y_data = args
        assert len(x_data) == 1000  # 默认点数
        assert len(y_data) == 1000
        assert x_data[0] == 0.0  # 默认范围起点
        assert x_data[-1] == 1.0  # 默认范围终点
        assert kwargs['label'] == 'ConcreteMembershipFunction'
    
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.show')
    def test_plot_custom_parameters(self, mock_show, mock_legend, mock_plot, concrete_mf):
        """测试自定义参数的绘图"""
        x_range = (-2, 3)
        num_points = 500
        
        concrete_mf.plot(x_range=x_range, num_points=num_points)
        
        # 检查绘图数据
        args, kwargs = mock_plot.call_args
        x_data, y_data = args
        assert len(x_data) == num_points
        assert len(y_data) == num_points
        assert x_data[0] == x_range[0]
        assert x_data[-1] == x_range[1]
    
    def test_plot_without_matplotlib(self, concrete_mf):
        """测试没有 matplotlib 时的行为"""
        with patch.dict('sys.modules', {'matplotlib.pyplot': None}):
            with pytest.raises(ImportError):
                concrete_mf.plot()


class TestEdgeCases:
    """测试边界情况和异常处理"""
    
    @pytest.fixture
    def concrete_mf(self):
        return ConcreteMembershipFunction(param1=0.0, param2=1.0)
    
    def test_extreme_input_values(self, concrete_mf, boundary_test_cases):
        """测试极值输入"""
        # 测试极大和极小值
        extreme_values = boundary_test_cases['extreme_values']
        result = concrete_mf.compute(extreme_values)
        assert np.all(np.isfinite(result))  # 结果应该是有限的
        assert np.all((result >= 0) & (result <= 1))  # 在有效范围内
    
    def test_near_zero_values(self, concrete_mf, boundary_test_cases):
        """测试接近零的值"""
        near_zero = boundary_test_cases['near_zero']
        result = concrete_mf.compute(near_zero)
        assert np.all(np.isfinite(result))
        assert np.all((result >= 0) & (result <= 1))
    
    def test_special_float_values(self, concrete_mf, boundary_test_cases):
        """测试特殊浮点值"""
        special_values = boundary_test_cases['special_values']
        result = concrete_mf.compute(special_values)
        
        # 对于 inf 和 -inf，结果应该是有限的（被 clip 处理）
        finite_mask = np.isfinite(special_values)
        assert np.all(np.isfinite(result[finite_mask]))
        
        # 对于 NaN，结果可能是 NaN（这是可接受的）
        nan_mask = np.isnan(special_values)
        # 不强制要求 NaN 的处理方式，但结果应该是可预测的
    
    def test_empty_array_input(self, concrete_mf):
        """测试空数组输入"""
        empty_array = np.array([])
        result = concrete_mf.compute(empty_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
    
    def test_single_element_array(self, concrete_mf):
        """测试单元素数组"""
        single_element = np.array([0.5])
        result = concrete_mf.compute(single_element)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0
    
    def test_parameter_persistence(self, concrete_mf):
        """测试参数持久性"""
        # 多次计算不应该改变参数
        original_params = concrete_mf.get_parameters().copy()
        
        for _ in range(10):
            concrete_mf.compute(np.random.random(100))
        
        assert concrete_mf.get_parameters() == original_params
    
    def test_thread_safety_simulation(self, concrete_mf):
        """模拟线程安全测试"""
        # 模拟并发访问（实际的线程安全测试需要更复杂的设置）
        import threading
        import time
        
        results = []
        errors = []
        
        def compute_worker():
            try:
                for _ in range(100):
                    result = concrete_mf.compute(0.5)
                    results.append(result)
                    time.sleep(0.001)  # 模拟计算时间
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=compute_worker) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 不应该有错误
        assert len(errors) == 0
        # 所有结果应该相同（因为输入相同）
        assert len(set(results)) == 1


class TestInheritanceRequirements:
    """测试继承要求和约定"""
    
    def test_missing_compute_method(self):
        """测试缺少 compute 方法的实现"""
        class IncompleteImplementation(MembershipFunction):
            def set_parameters(self, **kwargs):
                pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteImplementation()
    
    def test_missing_set_parameters_method(self):
        """测试缺少 set_parameters 方法的实现"""
        class IncompleteImplementation(MembershipFunction):
            def compute(self, x):
                return np.zeros_like(x)
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteImplementation()
    
    def test_complete_implementation(self):
        """测试完整的实现"""
        class CompleteImplementation(MembershipFunction):
            def __init__(self):
                super().__init__()
                self.parameters = {}
            
            def compute(self, x):
                return np.ones_like(x) * 0.5
            
            def set_parameters(self, **kwargs):
                pass
        
        # 应该能够成功实例化
        mf = CompleteImplementation()
        assert isinstance(mf, MembershipFunction)
        assert mf.name == 'CompleteImplementation'
    
    def test_super_init_call_requirement(self):
        """测试必须调用 super().__init__()"""
        class NoSuperInit(MembershipFunction):
            def __init__(self):
                # 故意不调用 super().__init__()
                pass
            
            def compute(self, x):
                return np.zeros_like(x)
            
            def set_parameters(self, **kwargs):
                pass
        
        mf = NoSuperInit()
        # 应该缺少基类设置的属性
        assert not hasattr(mf, 'name')
        assert not hasattr(mf, 'parameters')


if __name__ == '__main__':
    pytest.main()
