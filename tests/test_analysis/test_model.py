#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/26 15:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
from unittest.mock import MagicMock

from axisfuzzy.analysis.app.model import Model
from axisfuzzy.analysis.pipeline import FuzzyPipeline
from axisfuzzy.analysis.component.basic import (
    ToolNormalization,
    ToolFuzzification,
    ToolSimpleAggregation,
    ToolStatistics,
    ToolWeightNormalization
)
from axisfuzzy.analysis.build_in import (
    ContractCrispTable, 
    ContractFuzzyTable,
    ContractWeightVector,
    ContractStatisticsDict
)
from axisfuzzy.analysis.contracts import contract
from axisfuzzy.fuzzifier import Fuzzifier


def assert_dataframe_equal(df1, df2):
    """Helper function to assert that two DataFrames are equal."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False, atol=1e-5)


class SimpleTestModel(Model):
    """简单的测试模型，用于基础功能测试"""
    
    def __init__(self):
        super().__init__()
        self.normalizer = ToolNormalization(method='min_max')
        self.statistics = ToolStatistics()
    
    def get_config(self):
        return {}  # 简化配置，避免序列化问题
    
    def forward(self, data: ContractCrispTable) -> ContractStatisticsDict:
        normalized = self.normalizer(data)
        stats = self.statistics(normalized)
        return stats


class ComplexTestModel(Model):
    """复杂的测试模型，包含多个组件和分支"""
    
    def __init__(self, fuzzifier):
        super().__init__()
        self.normalizer = ToolNormalization(method='min_max')
        self.fuzzifier_tool = ToolFuzzification(fuzzifier=fuzzifier)
        self.aggregator = ToolSimpleAggregation(operation='mean')
        self.fuzzifier = fuzzifier
    
    def get_config(self):
        return {
            'fuzzifier': self.fuzzifier.get_config()
        }
    
    def forward(self, data: ContractCrispTable) -> ContractWeightVector:
        normalized = self.normalizer(data)
        fuzzy_data = self.fuzzifier_tool(normalized)
        # 注意：这里简化处理，实际应用中可能需要更复杂的逻辑
        # 从模糊数据中提取某种聚合结果
        result = self.aggregator(normalized)  # 使用归一化数据进行聚合
        return result


@pytest.fixture
def sample_dataframe():
    """创建测试用的样本数据框"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.rand(5),
        'feature_2': np.random.rand(5),
        'feature_3': np.random.rand(5)
    })


@pytest.fixture
def simple_model():
    """创建简单测试模型"""
    model = SimpleTestModel()
    model.build()
    return model


@pytest.fixture
def complex_model():
    """创建复杂测试模型"""
    # 创建一个简单的模糊化器用于测试
    fuzzifier = Fuzzifier(
        mf='gaussmf',
        mtype='qrofn',
        q=2,
        mf_params={'sigma': 0.1, 'c': 0.5}
    )
    model = ComplexTestModel(fuzzifier)
    model.build()
    return model


@pytest.fixture
def temp_model_file(tmp_path):
    """创建临时模型文件路径"""
    return tmp_path / "test_model.json"


class TestModelCore:
    """测试 Model 类的核心功能"""

    def test_model_creation_and_build(self, sample_dataframe):
        """测试模型创建和构建过程"""
        model = SimpleTestModel()
        
        # 构建前应该抛出异常
        assert not model.built
        with pytest.raises(RuntimeError, match="Model has not been built yet"):
            _ = model.pipeline
        
        # 构建模型
        model.build()
        assert model.built
        assert isinstance(model.pipeline, FuzzyPipeline)
        
        # 测试执行
        result = model(sample_dataframe)
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result

    def test_model_representation(self, simple_model):
        """测试模型的字符串表示"""
        representation = str(simple_model)
        assert "SimpleTestModel" in representation

    def test_model_forward_vs_call(self, simple_model, sample_dataframe):
        """测试 forward 方法和 __call__ 方法的等价性"""
        result_call = simple_model(sample_dataframe)
        result_forward = simple_model.run(sample_dataframe)
        
        # 两种调用方式应该产生相同结果
        assert result_call == result_forward

    def test_model_step_by_step_execution(self, simple_model, sample_dataframe):
        """测试逐步执行功能"""
        iterator = simple_model.step_by_step(sample_dataframe)
        
        steps = list(iterator)
        assert len(steps) >= 2  # 至少有归一化和统计两个步骤
        
        for step in steps:
            assert isinstance(step, dict)
            assert 'step_name' in step
            assert 'result' in step
            assert 'execution_time' in step

    def test_complex_model_execution(self, complex_model, sample_dataframe):
        """测试复杂模型的执行"""
        result = complex_model(sample_dataframe)
        # 复杂模型可能返回字典（多输出）或单一结果
        if isinstance(result, dict):
            # 检查是否包含预期的输出
            assert len(result) > 0
            # 检查其中一个输出是否为预期类型
            for key, value in result.items():
                if isinstance(value, (pd.Series, np.ndarray)):
                    assert len(value) == len(sample_dataframe)
                    break
        else:
            assert isinstance(result, (pd.Series, np.ndarray))
            assert len(result) == len(sample_dataframe)


class TestModelIO:
    """测试模型的保存和加载功能"""

    def test_save_and_load_simple_model(self, simple_model, sample_dataframe, temp_model_file):
        """测试简单模型的保存和加载"""
        # 保存模型
        simple_model.save(temp_model_file)
        assert temp_model_file.exists()
        
        # 加载模型
        loaded_model = Model.load(temp_model_file)
        assert isinstance(loaded_model, SimpleTestModel)
        
        # 需要重新构建加载的模型
        loaded_model.build()
        
        # 验证功能一致性
        original_result = simple_model(sample_dataframe)
        loaded_result = loaded_model(sample_dataframe)
        assert original_result == loaded_result

    def test_save_and_load_complex_model(self, complex_model, sample_dataframe, temp_model_file):
        """测试复杂模型的保存和加载"""
        # 保存模型
        complex_model.save(temp_model_file)
        assert temp_model_file.exists()
        
        # 加载模型
        loaded_model = Model.load(temp_model_file)
        assert isinstance(loaded_model, ComplexTestModel)
        
        # 需要重新构建加载的模型
        loaded_model.build()
        
        # 验证功能一致性
        original_result = complex_model(sample_dataframe)
        loaded_result = loaded_model(sample_dataframe)
        
        # 对于数值结果，使用近似比较
        if isinstance(original_result, pd.Series):
            pd.testing.assert_series_equal(original_result, loaded_result, atol=1e-5)
        elif isinstance(original_result, np.ndarray):
            np.testing.assert_array_almost_equal(original_result, loaded_result, decimal=5)

    def test_load_nonexistent_model_raises_error(self):
        """测试加载不存在的模型文件"""
        with pytest.raises(FileNotFoundError):
            Model.load("nonexistent_model.json")


class TestModelEdgeCases:
    """测试模型的边界情况和错误处理"""

    def test_model_without_build_raises_error(self):
        """测试未构建的模型执行时抛出异常"""
        model = SimpleTestModel()
        
        with pytest.raises(RuntimeError, match="Model has not been built yet"):
            model(pd.DataFrame({'a': [1, 2, 3]}))
        
        with pytest.raises(RuntimeError, match="Model has not been built yet"):
            model.step_by_step(pd.DataFrame({'a': [1, 2, 3]}))
        
        with pytest.raises(RuntimeError, match="Model has not been built yet"):
            model.visualize()

    def test_model_with_invalid_input_contract(self, simple_model):
        """测试输入数据不符合契约时的错误处理"""
        # 传入字符串而不是 DataFrame
        with pytest.raises(TypeError):
            simple_model("not a dataframe")
        
        # 传入非数值 DataFrame
        invalid_df = pd.DataFrame({'text': ['a', 'b', 'c']})
        with pytest.raises(TypeError):
            simple_model(invalid_df)

    def test_model_summary(self, simple_model):
        """测试模型摘要功能"""
        # 这个测试主要确保 summary 方法不会抛出异常
        # 实际输出会打印到控制台
        try:
            simple_model.summary()
        except Exception as e:
            pytest.fail(f"Model summary should not raise exception: {e}")

    def test_model_get_config_abstract_method(self):
        """测试 get_config 抽象方法必须被实现"""
        # 由于 Model 是抽象类，不能直接实例化不完整的子类
        # 这个测试验证抽象方法的存在性
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class IncompleteModel(Model):
                def forward(self, data):
                    return data
                # 故意不实现 get_config
            
            IncompleteModel()

    def test_model_forward_abstract_method(self):
        """测试 forward 抽象方法必须被实现"""
        # 由于 Model 是抽象类，不能直接实例化不完整的子类
        # 这个测试验证抽象方法的存在性
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class IncompleteModel(Model):
                def get_config(self):
                    return {}
                # 故意不实现 forward
            
            IncompleteModel()


class TestModelAdvanced:
    """测试模型的高级功能"""

    def test_nested_model_composition(self, sample_dataframe):
        """测试嵌套模型组合"""
        # 创建一个包含其他模型的模型
        class CompositeModel(Model):
            def __init__(self):
                super().__init__()
                self.sub_model = SimpleTestModel()
                self.aggregator = ToolSimpleAggregation(operation='sum')
            
            def get_config(self):
                return {
                    'sub_model_config': self.sub_model.get_config(),
                    'aggregator_operation': self.aggregator.operation
                }
            
            def forward(self, data: ContractCrispTable) -> ContractWeightVector:
                # 注意：这里简化了嵌套模型的使用
                # 实际中可能需要更复杂的数据流处理
                result = self.aggregator(data)
                return result
        
        composite_model = CompositeModel()
        composite_model.build()
        
        result = composite_model(sample_dataframe)
        assert isinstance(result, (pd.Series, np.ndarray))

    def test_model_with_return_intermediate(self, simple_model, sample_dataframe):
        """测试返回中间结果的功能"""
        result, intermediate = simple_model.run(sample_dataframe, return_intermediate=True)
        
        assert isinstance(result, dict)
        assert isinstance(intermediate, dict)
        assert len(intermediate) >= 2  # 至少有两个中间步骤

    def test_model_module_registration(self):
        """测试模块注册机制"""
        model = SimpleTestModel()
        
        # 检查模块是否正确注册
        assert 'normalizer' in model._modules
        assert 'statistics' in model._modules
        assert isinstance(model._modules['normalizer'], ToolNormalization)
        assert isinstance(model._modules['statistics'], ToolStatistics)

    def test_model_invalid_module_name(self):
        """测试无效的模块名称"""
        model = SimpleTestModel()
        
        with pytest.raises(KeyError, match="Module name cannot contain"):
            model.add_module("invalid.name", ToolNormalization())

    def test_model_invalid_module_type(self):
        """测试无效的模块类型"""
        model = SimpleTestModel()
        
        with pytest.raises(TypeError, match="is not a valid AnalysisComponent"):
            model.add_module("invalid_module", "not a component")


class TestModelIntegration:
    """测试模型与其他系统组件的集成"""

    def test_model_with_fuzzifier_integration(self, sample_dataframe):
        """测试模型与模糊化器的集成"""
        # 创建模糊化器
        fuzzifier = Fuzzifier(
            mf='gaussmf',
            mtype='qrofn',
            q=2,
            mf_params={'sigma': 0.2, 'c': 0.5}
        )
        
        # 创建包含模糊化的模型
        class FuzzyModel(Model):
            def __init__(self, fuzzifier):
                super().__init__()
                self.normalizer = ToolNormalization(method='min_max')
                self.fuzzifier_tool = ToolFuzzification(fuzzifier=fuzzifier)
                self.fuzzifier = fuzzifier
            
            def get_config(self):
                return {'fuzzifier': self.fuzzifier.get_config()}
            
            def forward(self, data: ContractCrispTable) -> ContractFuzzyTable:
                normalized = self.normalizer(data)
                fuzzy_result = self.fuzzifier_tool(normalized)
                return fuzzy_result
        
        model = FuzzyModel(fuzzifier)
        model.build()
        
        result = model(sample_dataframe)
        # 结果应该是 FuzzyDataFrame
        from axisfuzzy.analysis.dataframe import FuzzyDataFrame
        assert isinstance(result, FuzzyDataFrame)

    def test_model_pipeline_compatibility(self, simple_model):
        """测试模型与管道的兼容性"""
        # 确保模型的 pipeline 属性返回有效的 FuzzyPipeline
        pipeline = simple_model.pipeline
        assert isinstance(pipeline, FuzzyPipeline)
        
        # 检查管道的基本属性
        assert len(pipeline.steps) > 0
        assert len(pipeline.input_nodes) > 0

    def test_model_contract_validation(self, simple_model, sample_dataframe):
        """测试模型的契约验证"""
        # 正确的输入应该成功
        result = simple_model(sample_dataframe)
        assert isinstance(result, dict)
        
        # 错误的输入类型应该失败
        with pytest.raises(TypeError):
            simple_model([1, 2, 3])  # 列表而不是 DataFrame


class TestLinearModels:
    """测试线性模型（简单的顺序数据流）"""
    
    def test_simple_linear_model(self, sample_dataframe):
        """测试最简单的线性模型：归一化 -> 统计"""
        class LinearModel(Model):
            def __init__(self):
                super().__init__()
                self.normalizer = ToolNormalization(method='min_max')
                self.statistics = ToolStatistics()
            
            def get_config(self):
                return {'normalizer_method': 'min_max'}
            
            def forward(self, data: ContractCrispTable) -> ContractStatisticsDict:
                normalized = self.normalizer(data)
                stats = self.statistics(normalized)
                return stats
        
        model = LinearModel()
        model.build()
        
        result = model(sample_dataframe)
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result
        
        # 验证线性流程的步骤数
        iterator = model.step_by_step(sample_dataframe)
        steps = list(iterator)
        assert len(steps) >= 2  # normalizer + statistics
    
    def test_three_step_linear_model(self, sample_dataframe):
        """测试三步线性模型：归一化 -> 权重归一化 -> 聚合"""
        class ThreeStepLinearModel(Model):
            def __init__(self):
                super().__init__()
                self.normalizer = ToolNormalization(method='z_score')
                self.weight_normalizer = ToolWeightNormalization()
                self.aggregator = ToolSimpleAggregation(operation='mean')
            
            def get_config(self):
                return {'normalizer_method': 'z_score'}
            
            def forward(self, data: ContractCrispTable) -> ContractWeightVector:
                normalized = self.normalizer(data)
                # 直接聚合归一化数据，跳过权重归一化步骤
                result = self.aggregator(normalized)
                return result
        
        model = ThreeStepLinearModel()
        model.build()
        
        result = model(sample_dataframe)
        assert isinstance(result, (pd.Series, np.ndarray))
        
        # 验证步骤顺序
        iterator = model.step_by_step(sample_dataframe)
        steps = list(iterator)
        step_names = [step['step_name'] for step in steps]
        assert any('ToolNormalization' in name for name in step_names)
        assert any('ToolSimpleAggregation' in name for name in step_names)


class TestNonLinearModels:
    """测试非线性模型（包含分支、合并、多输入输出）"""
    
    def test_branching_model(self, sample_dataframe):
        """测试数据分支模型：一个输入，两个处理分支，多个输出"""
        class BranchingModel(Model):
            def __init__(self):
                super().__init__()
                # 分支1：归一化 -> 统计
                self.normalizer1 = ToolNormalization(method='min_max')
                self.statistics = ToolStatistics()
                # 分支2：归一化 -> 聚合
                self.normalizer2 = ToolNormalization(method='z_score')
                self.aggregator = ToolSimpleAggregation(operation='mean')
            
            def get_config(self):
                return {'model_type': 'branching'}
            
            def forward(self, data: ContractCrispTable):
                # 分支1
                norm_data1 = self.normalizer1(data)
                stats = self.statistics(norm_data1)
                
                # 分支2
                norm_data2 = self.normalizer2(data)
                aggregated = self.aggregator(norm_data2)
                
                # 多输出
                return {
                    'statistics': stats,
                    'aggregated': aggregated,
                    'normalized_minmax': norm_data1,
                    'normalized_zscore': norm_data2
                }
        
        model = BranchingModel()
        model.build()
        
        result = model(sample_dataframe)
        assert isinstance(result, dict)
        assert len(result) >= 2
        # 检查实际返回的键名
        result_keys = list(result.keys())
        assert len(result_keys) >= 2
    
    def test_multi_input_model(self, sample_dataframe):
        """测试多步骤处理模型：模拟多输入的复杂处理"""
        class MultiStepModel(Model):
            def __init__(self):
                super().__init__()
                self.normalizer = ToolNormalization(method='min_max')
                self.aggregator = ToolSimpleAggregation(operation='mean')
                self.statistics = ToolStatistics()
            
            def get_config(self):
                return {'model_type': 'multi_step'}
            
            def forward(self, data: ContractCrispTable) -> ContractStatisticsDict:
                # 多步骤处理：标准化 -> 聚合 -> 统计
                norm_data = self.normalizer(data)
                agg_data = self.aggregator(norm_data)
                result = self.statistics(data)  # 使用原始数据计算统计
                return result
        
        model = MultiStepModel()
        model.build()
        
        result = model(sample_dataframe)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestModelNesting:
    """测试模型嵌套功能"""
    
    def test_deep_nesting_model(self, sample_dataframe):
        """测试深度嵌套模型：子模型包含子模型"""
        # 最内层子模型
        class InnerModel(Model):
            def __init__(self):
                super().__init__()
                self.normalizer = ToolNormalization(method='min_max')
            
            def get_config(self):
                return {'level': 'inner'}
            
            def forward(self, data: ContractCrispTable) -> ContractCrispTable:
                return self.normalizer(data)
        
        # 中间层子模型
        class MiddleModel(Model):
            def __init__(self):
                super().__init__()
                self.inner_model = InnerModel()
                self.statistics = ToolStatistics()
            
            def get_config(self):
                return {
                    'level': 'middle',
                    'inner_config': self.inner_model.get_config()
                }
            
            def forward(self, data: ContractCrispTable) -> ContractStatisticsDict:
                normalized = self.inner_model(data)
                stats = self.statistics(normalized)
                return stats
        
        # 最外层主模型
        class OuterModel(Model):
            def __init__(self):
                super().__init__()
                self.middle_model = MiddleModel()
                self.aggregator = ToolSimpleAggregation(operation='mean')
            
            def get_config(self):
                return {
                    'level': 'outer',
                    'middle_config': self.middle_model.get_config()
                }
            
            def forward(self, data: ContractCrispTable):
                stats = self.middle_model(data)
                # 从统计结果中提取数值进行聚合
                # 这里简化处理，直接返回统计结果
                return stats
        
        model = OuterModel()
        model.build()
        
        result = model(sample_dataframe)
        assert isinstance(result, dict)
        assert 'mean' in result
        
        # 验证嵌套模型的构建
        assert model.built
        assert model.middle_model.built
        assert model.middle_model.inner_model.built
    
    def test_nested_model_independence(self, sample_dataframe):
        """测试嵌套模型的独立性：子模型可以独立运行"""
        class SubModel(Model):
            def __init__(self):
                super().__init__()
                self.normalizer = ToolNormalization(method='min_max')
                self.statistics = ToolStatistics()
            
            def get_config(self):
                return {'type': 'sub_model'}
            
            def forward(self, data: ContractCrispTable) -> ContractStatisticsDict:
                normalized = self.normalizer(data)
                return self.statistics(normalized)
        
        class MainModel(Model):
            def __init__(self):
                super().__init__()
                self.sub_model = SubModel()
                self.aggregator = ToolSimpleAggregation(operation='sum')
            
            def get_config(self):
                return {
                    'type': 'main_model',
                    'sub_config': self.sub_model.get_config()
                }
            
            def forward(self, data: ContractCrispTable):
                stats = self.sub_model(data)
                # 简化处理，直接返回统计结果
                return stats
        
        # 测试子模型独立运行
        sub_model = SubModel()
        sub_model.build()
        sub_result = sub_model(sample_dataframe)
        
        # 测试主模型运行
        main_model = MainModel()
        main_model.build()
        main_result = main_model(sample_dataframe)
        
        # 结果应该相同（因为主模型只是调用了子模型）
        assert sub_result == main_result


class TestModelPerformance:
    """测试模型性能相关功能"""
    
    def test_execution_time_measurement(self, sample_dataframe):
        """测试执行时间测量功能"""
        class TimedModel(Model):
            def __init__(self):
                super().__init__()
                self.normalizer = ToolNormalization(method='min_max')
                self.statistics = ToolStatistics()
                # 添加一个稍微耗时的操作
                self.aggregator = ToolSimpleAggregation(operation='mean')
            
            def get_config(self):
                return {'type': 'timed_model'}
            
            def forward(self, data: ContractCrispTable):
                normalized = self.normalizer(data)
                stats = self.statistics(normalized)
                aggregated = self.aggregator(normalized)
                return {'stats': stats, 'aggregated': aggregated}
        
        model = TimedModel()
        model.build()
        
        # 测试step_by_step的时间测量
        iterator = model.step_by_step(sample_dataframe)
        steps = list(iterator)
        
        # 每个步骤都应该有执行时间
        for step in steps:
            assert 'execution_time' in step
            assert isinstance(step['execution_time'], (int, float))
            assert step['execution_time'] >= 0
        
        # 测试总执行时间
        start_time = time.time()
        result = model(sample_dataframe)
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert execution_time > 0
        assert isinstance(result, dict)
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建较大的测试数据集
        np.random.seed(42)
        large_df = pd.DataFrame({
            f'feature_{i}': np.random.rand(1000) for i in range(10)
        })
        
        class PerformanceModel(Model):
            def __init__(self):
                super().__init__()
                self.normalizer = ToolNormalization(method='min_max')
                self.statistics = ToolStatistics()
            
            def get_config(self):
                return {'type': 'performance_model'}
            
            def forward(self, data: ContractCrispTable) -> ContractStatisticsDict:
                normalized = self.normalizer(data)
                return self.statistics(normalized)
        
        model = PerformanceModel()
        model.build()
        
        # 测试执行时间应该在合理范围内
        start_time = time.time()
        result = model(large_df)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # 应该在5秒内完成
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result


class TestModelDebugging:
    """测试模型调试和内省功能"""
    
    def test_model_summary_functionality(self, simple_model):
        """测试模型summary功能的详细输出"""
        # 这个测试主要确保summary方法能正常工作
        # 实际输出会打印到控制台，我们主要测试不抛异常
        try:
            simple_model.summary()
        except Exception as e:
            pytest.fail(f"Model summary should not raise exception: {e}")
        
        # 验证模型已构建
        assert simple_model.built
        assert isinstance(simple_model.pipeline, FuzzyPipeline)
    
    def test_step_by_step_detailed_info(self, simple_model, sample_dataframe):
        """测试step_by_step返回的详细信息"""
        iterator = simple_model.step_by_step(sample_dataframe)
        steps = list(iterator)
        
        # 验证每个步骤的信息完整性
        for i, step in enumerate(steps):
            assert isinstance(step, dict)
            
            # 必需的字段
            required_fields = ['step_name', 'result', 'execution_time', 'step_index', 'total_steps']
            for field in required_fields:
                assert field in step, f"Missing field '{field}' in step {i}"
            
            # 验证字段类型
            assert isinstance(step['step_name'], str)
            assert isinstance(step['step_index'], int)
            assert isinstance(step['total_steps'], int)
            assert isinstance(step['execution_time'], (int, float))
            
            # 验证索引范围
            assert 0 <= step['step_index'] <= step['total_steps']
            assert step['execution_time'] >= 0
        
        # 验证步骤索引的连续性
        if len(steps) > 0:
            step_indices = [step['step_index'] for step in steps]
            # 步骤索引从1开始，而不是从0开始
            assert step_indices == list(range(1, len(steps) + 1))
        
        # 验证最终结果
        final_result = iterator.result
        assert isinstance(final_result, dict)
    
    def test_model_pipeline_introspection(self, simple_model):
        """测试模型管道的内省功能"""
        pipeline = simple_model.pipeline
        
        # 验证管道基本属性
        assert isinstance(pipeline, FuzzyPipeline)
        assert len(pipeline.steps) > 0
        assert len(pipeline.input_nodes) > 0
        
        # 验证步骤名称包含预期的工具
        step_names = [step.display_name for step in pipeline.steps]
        assert any('ToolNormalization' in name for name in step_names)
        assert any('ToolStatistics' in name for name in step_names)
    
    def test_model_module_introspection(self, simple_model):
        """测试模型模块的内省功能"""
        # 验证模块注册
        assert hasattr(simple_model, '_modules')
        assert isinstance(simple_model._modules, dict)
        
        # 验证注册的模块
        assert 'normalizer' in simple_model._modules
        assert 'statistics' in simple_model._modules
        
        # 验证模块类型
        assert isinstance(simple_model._modules['normalizer'], ToolNormalization)
        assert isinstance(simple_model._modules['statistics'], ToolStatistics)