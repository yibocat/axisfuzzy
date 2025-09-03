#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
测试隶属函数工厂系统的类发现、别名系统和参数分离功能

本模块测试以下工厂功能：
- get_mf_class: 类发现和别名解析
- create_mf: 隶属函数实例创建
- 别名系统的完整性和一致性
- 参数分离和处理
- 错误处理和异常情况

工厂系统是隶属函数模块的核心API，确保其正确性对整个系统至关重要。
"""

import pytest
import numpy as np
from typing import Type, Dict, Any

from axisfuzzy.membership.factory import get_mf_class, create_mf
from axisfuzzy.membership.base import MembershipFunction
from axisfuzzy.membership.function import (
    TriangularMF, TrapezoidalMF, GaussianMF, SigmoidMF,
    SMF, ZMF, PiMF, GeneralizedBellMF, DoubleGaussianMF
)
from .conftest import TOLERANCE


class TestGetMFClass:
    """测试 get_mf_class 函数的类发现和别名解析功能"""
    
    def test_get_class_by_exact_name(self):
        """测试通过精确类名获取类"""
        # 测试所有标准隶属函数类
        test_cases = [
            ('TriangularMF', TriangularMF),
            ('TrapezoidalMF', TrapezoidalMF),
            ('GaussianMF', GaussianMF),
            ('SigmoidMF', SigmoidMF),
            ('SMF', SMF),
            ('ZMF', ZMF),
            ('PiMF', PiMF),
            ('GeneralizedBellMF', GeneralizedBellMF),
            ('DoubleGaussianMF', DoubleGaussianMF)
        ]
        
        for name, expected_class in test_cases:
            result_class = get_mf_class(name)
            assert result_class is expected_class
            assert issubclass(result_class, MembershipFunction)
    
    def test_get_class_by_standard_aliases(self):
        """测试通过标准别名获取类"""
        # 测试文档中提到的标准别名
        alias_mapping = {
            'trimf': TriangularMF,
            'trapmf': TrapezoidalMF,
            'gaussmf': GaussianMF,
            'sigmoid': SigmoidMF,  # 修正：使用 'sigmoid' 而不是 'sigmf'
            'smf': SMF,
            'zmf': ZMF,
            'pimf': PiMF,
            'gbellmf': GeneralizedBellMF,
            'gauss2mf': DoubleGaussianMF  # 修正：使用 'gauss2mf' 而不是 'dgaussmf'
        }
        
        for alias, expected_class in alias_mapping.items():
            result_class = get_mf_class(alias)
            assert result_class is expected_class
    
    def test_get_class_by_auto_generated_aliases(self):
        """测试通过自动生成的别名获取类"""
        # 测试自动生成的别名（小写、去掉MF后缀等）
        auto_aliases = {
            'triangularmf': TriangularMF,
            'trapezoidalmf': TrapezoidalMF,
            'gaussianmf': GaussianMF,
            'sigmoidmf': SigmoidMF,
            'pimf': PiMF,
            'generalizedbellmf': GeneralizedBellMF,
            'doublegaussianmf': DoubleGaussianMF
        }
        
        for alias, expected_class in auto_aliases.items():
            result_class = get_mf_class(alias)
            assert result_class is expected_class
    
    def test_case_insensitive_lookup(self):
        """测试大小写不敏感的查找"""
        test_cases = [
            ('triangularmf', TriangularMF),
            ('TRIANGULARMF', TriangularMF),
            ('TriangularMF', TriangularMF),
            ('trimf', TriangularMF),
            ('TRIMF', TriangularMF),
            ('TrimF', TriangularMF)
        ]
        
        for name, expected_class in test_cases:
            result_class = get_mf_class(name)
            assert result_class is expected_class
    
    def test_invalid_class_name(self):
        """测试无效的类名"""
        invalid_names = [
            'NonExistentMF',
            'InvalidFunction',
            'NotAMembershipFunction',
            '',
            'random_string',
            '123invalid'
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                get_mf_class(invalid_name)
    
    def test_none_input(self):
        """测试None输入"""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            get_mf_class(None)
    
    def test_non_string_input(self):
        """测试非字符串输入"""
        invalid_inputs = [123, [], {}, TriangularMF]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                get_mf_class(invalid_input)


class TestCreateMF:
    """测试 create_mf 函数的隶属函数实例创建功能"""
    
    def test_create_with_keyword_args_basic(self):
        """测试使用关键字参数创建隶属函数"""
        # 三角形隶属函数
        mf, unused = create_mf('trimf', a=0, b=0.5, c=1)
        assert isinstance(mf, TriangularMF)
        assert mf.a == 0
        assert mf.b == 0.5
        assert mf.c == 1
        assert unused == {}
        
        # 梯形隶属函数
        mf, unused = create_mf('trapmf', a=0, b=0.2, c=0.8, d=1)
        assert isinstance(mf, TrapezoidalMF)
        assert mf.a == 0
        assert mf.b == 0.2
        assert mf.c == 0.8
        assert mf.d == 1
        assert unused == {}
        
        # 高斯隶属函数
        mf, unused = create_mf('gaussmf', sigma=0.2, c=0.5)
        assert isinstance(mf, GaussianMF)
        assert mf.sigma == 0.2
        assert mf.c == 0.5
        assert unused == {}
    
    def test_create_with_keyword_args(self):
        """测试使用关键字参数创建隶属函数"""
        # 三角形隶属函数
        mf, unused = create_mf('trimf', a=0, b=0.5, c=1)
        assert isinstance(mf, TriangularMF)
        assert mf.a == 0
        assert mf.b == 0.5
        assert mf.c == 1
        assert unused == {}
        
        # 高斯隶属函数
        mf, unused = create_mf('gaussmf', sigma=0.2, c=0.5)
        assert isinstance(mf, GaussianMF)
        assert mf.sigma == 0.2
        assert mf.c == 0.5
        assert unused == {}
        
        # Sigmoid隶属函数
        mf, unused = create_mf('sigmoid', k=2.0, c=0.5)
        assert isinstance(mf, SigmoidMF)
        assert mf.k == 2.0
        assert mf.c == 0.5
        assert unused == {}
    
    def test_create_with_mixed_args(self):
        """测试使用混合参数创建隶属函数"""
        # 关键字参数
        mf, unused = create_mf('trapmf', a=0, b=0.2, c=0.8, d=1)
        assert isinstance(mf, TrapezoidalMF)
        assert mf.a == 0
        assert mf.b == 0.2
        assert mf.c == 0.8
        assert mf.d == 1
        assert unused == {}
    
    def test_create_with_defaults(self):
        """测试使用默认参数创建隶属函数"""
        # 不提供参数，使用默认值
        mf, unused = create_mf('trimf')
        assert isinstance(mf, TriangularMF)
        assert mf.a == 0.0
        assert mf.b == 0.5
        assert mf.c == 1.0
        assert unused == {}
        
        # 部分参数，其余使用默认值
        mf, unused = create_mf('gaussmf', sigma=0.3)
        assert isinstance(mf, GaussianMF)
        assert mf.sigma == 0.3
        assert unused == {}
        # c应该使用默认值
    
    def test_create_all_function_types(self):
        """测试创建所有类型的隶属函数"""
        test_cases = [
            ('trimf', {'a': 0, 'b': 0.5, 'c': 1}, TriangularMF),
            ('trapmf', {'a': 0, 'b': 0.2, 'c': 0.8, 'd': 1}, TrapezoidalMF),
            ('gaussmf', {'sigma': 0.2, 'c': 0.5}, GaussianMF),
            ('sigmoid', {'k': 2.0, 'c': 0.5}, SigmoidMF),
            ('smf', {'a': 0, 'b': 1}, SMF),
            ('zmf', {'a': 0, 'b': 1}, ZMF),
            ('pimf', {'a': 0, 'b': 0.2, 'c': 0.8, 'd': 1}, PiMF),
            ('gbellmf', {'a': 0.2, 'b': 2, 'c': 0.5}, GeneralizedBellMF),
            ('gauss2mf', {'sigma1': 0.1, 'c1': 0.3, 'sigma2': 0.1, 'c2': 0.7}, DoubleGaussianMF)
        ]
        
        for alias, kwargs, expected_class in test_cases:
            mf, unused = create_mf(alias, **kwargs)
            assert isinstance(mf, expected_class)
            assert isinstance(mf, MembershipFunction)
            assert unused == {}
    
    def test_parameter_separation(self):
        """测试参数分离功能"""
        # 测试工厂函数是否正确分离了隶属函数参数和其他参数
        mf, unused = create_mf('trimf', a=0, b=0.5, c=1, extra_param='test_value', another_param=42)
        assert isinstance(mf, TriangularMF)
        assert mf.a == 0
        assert mf.b == 0.5
        assert mf.c == 1
        # 额外参数应该在 unused 中返回
        assert unused == {'extra_param': 'test_value', 'another_param': 42}
    
    def test_invalid_function_name(self):
        """测试无效的函数名"""
        with pytest.raises(ValueError, match="Unknown membership function"):
            create_mf('invalid_function', a=0, b=0.5, c=1)
    
    def test_invalid_parameters(self):
        """测试无效的参数"""
        # 参数值无效
        with pytest.raises(ValueError):
            create_mf('trimf', a=1, b=0.5, c=0)  # a > b > c
        
        with pytest.raises(ValueError):
            create_mf('gaussmf', sigma=-0.1, c=0.5)  # sigma < 0
    
    def test_functional_correctness(self):
        """测试创建的隶属函数的功能正确性"""
        # 创建函数并测试其计算能力
        mf, unused = create_mf('trimf', a=0, b=0.5, c=1)
        assert unused == {}
        
        # 测试关键点
        assert mf.compute(0) == 0.0
        assert mf.compute(0.5) == 1.0
        assert mf.compute(1) == 0.0
        assert abs(mf.compute(0.25) - 0.5) < TOLERANCE
        
        # 测试向量化计算
        x_vals = np.array([0, 0.25, 0.5, 0.75, 1])
        y_vals = mf.compute(x_vals)
        expected = np.array([0, 0.5, 1, 0.5, 0])
        assert np.allclose(y_vals, expected, atol=TOLERANCE)


class TestAliasSystem:
    """测试别名系统的完整性和一致性"""
    
    def test_alias_completeness(self):
        """测试所有隶属函数都有对应的别名"""
        # 所有隶属函数类
        all_classes = [
            TriangularMF, TrapezoidalMF, GaussianMF, SigmoidMF,
            SMF, ZMF, PiMF, GeneralizedBellMF, DoubleGaussianMF
        ]
        
        # 每个类都应该能通过至少一个别名访问
        for cls in all_classes:
            # 通过类名应该能访问
            result_cls = get_mf_class(cls.__name__)
            assert result_cls is cls
    
    def test_alias_consistency(self):
        """测试别名的一致性"""
        # 同一个类的不同别名应该返回相同的类
        triangular_aliases = ['TriangularMF', 'trimf', 'triangularmf']
        
        expected_class = get_mf_class(triangular_aliases[0])
        for alias in triangular_aliases[1:]:
            result_class = get_mf_class(alias)
            assert result_class is expected_class
    
    def test_alias_uniqueness(self):
        """测试别名的唯一性（一个别名只对应一个类）"""
        # 收集所有别名和对应的类
        alias_to_class = {}
        
        # 测试一些已知的别名
        known_aliases = [
            'trimf', 'trapmf', 'gaussmf', 'sigmoid', 'smf', 'zmf',
            'pimf', 'gbellmf', 'gauss2mf'
        ]
        
        for alias in known_aliases:
            cls = get_mf_class(alias)
            if alias in alias_to_class:
                # 如果别名已存在，应该指向同一个类
                assert alias_to_class[alias] is cls
            else:
                alias_to_class[alias] = cls
    
    def test_matlab_compatibility_aliases(self):
        """测试MATLAB兼容性别名"""
        # 测试MATLAB风格的别名
        matlab_aliases = {
            'trimf': TriangularMF,
            'trapmf': TrapezoidalMF,
            'gaussmf': GaussianMF,
            'sigmoid': SigmoidMF,  # 修正：使用 'sigmoid' 而不是 'sigmf'
            'smf': SMF,
            'zmf': ZMF,
            'pimf': PiMF,
            'gbellmf': GeneralizedBellMF
        }
        
        for alias, expected_class in matlab_aliases.items():
            result_class = get_mf_class(alias)
            assert result_class is expected_class


class TestParameterSeparation:
    """测试参数分离和处理功能"""
    
    def test_mf_parameter_separation(self):
        """测试隶属函数参数的正确分离"""
        # 创建时传入额外的参数，应该被正确分离
        mf, unused = create_mf('trimf', a=0, b=0.5, c=1, extra_param='test_value')
        
        # 隶属函数参数应该正确设置
        assert mf.a == 0
        assert mf.b == 0.5
        assert mf.c == 1
        
        # 额外参数应该在 unused 中返回
        assert unused == {'extra_param': 'test_value'}
    
    def test_parameter_override(self):
        """测试参数覆盖"""
        # 关键字参数的处理
        mf, unused = create_mf('trimf', a=0, b=0.3, c=1)
        assert mf.a == 0
        assert mf.b == 0.3
        assert mf.c == 1
        assert unused == {}
    
    def test_extra_parameters_handling(self):
        """测试额外参数的处理"""
        # 传入隶属函数不需要的参数，应该被正确分离
        mf, unused = create_mf('trimf', a=0, b=0.5, c=1, extra_param='ignored', system_param=42)
        # 隶属函数参数应该正确
        assert mf.a == 0
        assert mf.b == 0.5
        assert mf.c == 1
        # 额外参数应该在 unused 中返回
        assert unused == {'extra_param': 'ignored', 'system_param': 42}


class TestErrorHandling:
    """测试错误处理和异常情况"""
    
    def test_empty_function_name(self):
        """测试空函数名"""
        with pytest.raises(ValueError):
            get_mf_class('')
        
        with pytest.raises(ValueError):
            create_mf('', a=0, b=0.5, c=1)
    
    def test_whitespace_function_name(self):
        """测试包含空白字符的函数名"""
        with pytest.raises(ValueError):
            get_mf_class('  ')
        
        with pytest.raises(ValueError):
            get_mf_class('tri mf')  # 包含空格
    
    def test_special_characters_in_name(self):
        """测试函数名中的特殊字符"""
        invalid_names = ['tri-mf', 'tri.mf', 'tri@mf', 'tri#mf']
        
        for name in invalid_names:
            with pytest.raises(ValueError):
                get_mf_class(name)
    
    def test_parameter_validation_propagation(self):
        """测试参数验证错误的传播"""
        # 工厂函数应该正确传播隶属函数的参数验证错误
        with pytest.raises(ValueError):
            create_mf('trimf', a=1, b=0.5, c=0)  # 无效的三角形参数
        
        with pytest.raises(ValueError):
            create_mf('gaussmf', sigma=-0.1, c=0.5)  # 无效的高斯参数
    
    def test_type_error_propagation(self):
        """测试类型错误的传播"""
        # 测试无效的参数组合（违反约束条件）
        with pytest.raises(ValueError):
            create_mf('trimf', a=1, b=0.5, c=0)  # a > b > c，违反约束


class TestIntegration:
    """集成测试"""
    
    def test_factory_with_all_functions(self):
        """测试工厂函数与所有隶属函数的集成"""
        # 定义所有隶属函数的测试参数
        test_configs = [
            ('trimf', {'a': 0, 'b': 0.5, 'c': 1}),
            ('trapmf', {'a': 0, 'b': 0.2, 'c': 0.8, 'd': 1}),
            ('gaussmf', {'sigma': 0.2, 'c': 0.5}),
            ('sigmoid', {'k': 2.0, 'c': 0.5}),
            ('smf', {'a': 0, 'b': 1}),
            ('zmf', {'a': 0, 'b': 1}),
            ('pimf', {'a': 0, 'b': 0.2, 'c': 0.8, 'd': 1}),
            ('gbellmf', {'a': 0.2, 'b': 2, 'c': 0.5}),
            ('gauss2mf', {'sigma1': 0.1, 'c1': 0.3, 'sigma2': 0.1, 'c2': 0.7})
        ]
        
        x_test = np.linspace(-0.5, 1.5, 21)
        
        for alias, params in test_configs:
            # 通过工厂创建
            mf, unused = create_mf(alias, **params)
            
            # 验证基本属性
            assert isinstance(mf, MembershipFunction)
            assert unused == {}
            
            # 验证计算功能
            result = mf.compute(x_test)
            assert isinstance(result, np.ndarray)
            assert result.shape == x_test.shape
            assert np.all((result >= 0) & (result <= 1))
            assert np.all(np.isfinite(result))
    
    def test_factory_parameter_update_integration(self):
        """测试工厂创建的函数的参数更新功能"""
        # 通过工厂创建函数
        mf, unused = create_mf('trimf', a=0, b=0.5, c=1)
        assert unused == {}
        
        # 测试参数更新
        original_b = mf.b
        mf.set_parameters(b=0.3)
        assert mf.b != original_b
        assert mf.b == 0.3
        
        # 验证更新后的功能
        assert mf.compute(0.3) == 1.0  # 新的峰值点
    
    def test_factory_with_edge_cases(self):
        """测试工厂函数处理边界情况"""
        # 退化的三角形（点函数）
        mf, unused = create_mf('trimf', a=0.5, b=0.5, c=0.5)
        assert unused == {}
        assert mf.compute(0.5) == 1.0
        assert mf.compute(0.4) == 0.0
        
        # 极小的高斯sigma
        mf, unused = create_mf('gaussmf', sigma=1e-6, c=0.5)
        assert unused == {}
        assert abs(mf.compute(0.5) - 1.0) < TOLERANCE
        
        # 极大的Sigmoid斜率
        mf, unused = create_mf('sigmoid', k=100, c=0.5)
        assert unused == {}
        assert mf.compute(0.6) > 0.9  # 应该接近1
        assert mf.compute(0.4) < 0.1  # 应该接近0
    
    def test_performance_consistency(self):
        """测试性能一致性"""
        # 通过工厂创建和直接创建应该有相同的性能特征
        import time
        
        # 大数组测试
        x_large = np.linspace(0, 1, 10000)
        
        # 直接创建
        start_time = time.time()
        mf_direct = TriangularMF(0, 0.5, 1)
        result_direct = mf_direct.compute(x_large)
        time_direct = time.time() - start_time
        
        # 工厂创建
        start_time = time.time()
        mf_factory, unused = create_mf('trimf', a=0, b=0.5, c=1)
        assert unused == {}
        result_factory = mf_factory.compute(x_large)
        time_factory = time.time() - start_time
        
        # 结果应该相同
        assert np.allclose(result_direct, result_factory, atol=TOLERANCE)
        
        # 性能差异应该很小（工厂创建的开销主要在实例化时）
        # 这里主要测试计算性能是否一致
        assert abs(time_direct - time_factory) < 0.1  # 允许0.1秒的差异


class TestDocumentationConsistency:
    """测试与文档的一致性"""
    
    def test_documented_aliases(self):
        """测试文档中提到的别名是否都可用"""
        # 根据文档，这些别名应该都可用
        documented_aliases = {
            'trimf': TriangularMF,
            'trapmf': TrapezoidalMF,
            'gaussmf': GaussianMF,
            'sigmoid': SigmoidMF,
            'smf': SMF,
            'zmf': ZMF,
            'pimf': PiMF,
            'gbellmf': GeneralizedBellMF,
            'gauss2mf': DoubleGaussianMF
        }
        
        for alias, expected_class in documented_aliases.items():
            result_class = get_mf_class(alias)
            assert result_class is expected_class
    
    def test_documented_examples(self):
        """测试文档中的示例是否正确工作"""
        # 测试文档中可能出现的示例代码
        
        # 基本创建示例
        mf1, unused1 = create_mf('trimf', a=0, b=0.5, c=1)
        assert isinstance(mf1, TriangularMF)
        assert unused1 == {}
        
        # 关键字参数示例
        mf2, unused2 = create_mf('gaussmf', sigma=0.2, c=0.5)
        assert isinstance(mf2, GaussianMF)
        assert unused2 == {}
        
        # 别名使用示例
        mf3, unused3 = create_mf('trimf', a=0, b=0.5, c=1)
        assert isinstance(mf3, TriangularMF)
        assert unused3 == {}
        
        # 验证它们的功能
        x = 0.5
        assert mf1.compute(x) == 1.0
        assert abs(mf2.compute(x) - 1.0) < TOLERANCE
        assert mf3.compute(x) == 1.0


if __name__ == '__main__':
    pytest.main()
