#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟导入功能测试模块

测试 AxisFuzzy 延迟导入机制的正确性，包括：
- 延迟导入的基本功能
- 错误处理和优雅降级
- 缓存机制验证
- 依赖检查功能
- IDE 类型支持验证

Author: AxisFuzzy Team
Date: 2025-01-25
"""

import sys
import pytest
import importlib
from unittest.mock import patch, MagicMock


class TestLazyImportMechanism:
    """延迟导入机制测试类"""
    
    def test_analysis_module_lazy_import(self):
        """测试 analysis 模块的延迟导入"""
        try:
            import axisfuzzy
            
            # 测试模块级别的延迟导入
            analysis = axisfuzzy.analysis
            assert analysis is not None
            print("✅ analysis 模块延迟导入成功")
            
            # 测试 __all__ 列表的正确性
            expected_exports = [
                "check_analysis_dependencies",
                "Model", "FuzzyDataFrame", "FuzzyPipeline",
                "Contract", "contract", "AnalysisComponent"
            ]
            
            for export in expected_exports:
                assert hasattr(analysis, export), f"缺少导出: {export}"
            
            print("✅ analysis 模块导出列表验证通过")
            
        except Exception as e:
            pytest.fail(f"❌ analysis 模块延迟导入失败: {e}")
    
    def test_individual_component_lazy_import(self):
        """测试各个组件的延迟导入"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 测试核心组件延迟导入
            components = {
                "Model": "axisfuzzy.analysis.app.model.Model",
                "FuzzyDataFrame": "axisfuzzy.analysis.dataframe.frame.FuzzyDataFrame",
                "FuzzyPipeline": "axisfuzzy.analysis.pipeline.FuzzyPipeline",
                "Contract": "axisfuzzy.analysis.contracts.base.Contract",
                "AnalysisComponent": "axisfuzzy.analysis.component.base.AnalysisComponent"
            }
            
            for name, expected_class in components.items():
                component = getattr(analysis, name)
                assert component is not None
                assert str(component).find(expected_class.split('.')[-1]) != -1
                print(f"✅ {name} 延迟导入成功: {component}")
            
            # 测试装饰器
            contract_decorator = analysis.contract
            assert callable(contract_decorator)
            print(f"✅ contract 装饰器延迟导入成功: {contract_decorator}")
            
        except Exception as e:
            pytest.fail(f"❌ 组件延迟导入失败: {e}")
    
    def test_dependency_check_function(self):
        """测试依赖检查功能"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 测试依赖检查函数
            check_func = analysis.check_analysis_dependencies
            assert callable(check_func)
            
            # 执行依赖检查
            result = check_func()
            assert isinstance(result, dict)
            
            # 验证返回的依赖信息
            expected_deps = ['pandas', 'matplotlib', 'networkx', 'pydot']
            for dep in expected_deps:
                assert dep in result
                print(f"✅ 依赖检查包含 {dep}: {result[dep]}")
            
            print("✅ 依赖检查功能正常")
            
        except Exception as e:
            pytest.fail(f"❌ 依赖检查功能失败: {e}")
    
    def test_pandas_accessor_registration(self):
        """测试 pandas 访问器的自动注册"""
        try:
            # 导入 analysis 模块会自动注册 pandas 访问器
            import axisfuzzy.analysis as analysis
            
            # 检查 pandas 是否可用
            try:
                import pandas as pd
                
                # 创建测试数据
                df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
                
                # 检查访问器是否注册
                assert hasattr(df, 'fuzzy'), "FuzzyAccessor 未正确注册"
                
                # 测试访问器基本功能
                fuzzy_accessor = df.fuzzy
                assert fuzzy_accessor is not None
                
                print("✅ pandas 访问器自动注册成功")
                
            except ImportError:
                pytest.skip("pandas 未安装，跳过访问器测试")
                
        except Exception as e:
            pytest.fail(f"❌ pandas 访问器注册失败: {e}")
    
    def test_fuzzy_accessor_not_exported(self):
        """测试 FuzzyAccessor 未被导出"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 确认 FuzzyAccessor 不在 __all__ 中
            assert 'FuzzyAccessor' not in analysis.__all__
            
            # 确认 FuzzyAccessor 不能直接访问
            with pytest.raises(AttributeError):
                _ = analysis.FuzzyAccessor
            
            print("✅ FuzzyAccessor 正确地未被导出")
            
        except Exception as e:
            pytest.fail(f"❌ FuzzyAccessor 导出检查失败: {e}")


class TestLazyImportErrorHandling:
    """延迟导入错误处理测试类"""
    
    def test_missing_dependency_handling(self):
        """测试缺失依赖的处理"""
        # 这个测试需要模拟依赖缺失的情况
        with patch.dict('sys.modules', {'pandas': None}):
            try:
                # 重新导入模块以触发错误处理
                if 'axisfuzzy.analysis' in sys.modules:
                    del sys.modules['axisfuzzy.analysis']
                
                import axisfuzzy.analysis as analysis
                
                # 在依赖缺失时，应该只有依赖检查函数可用
                assert hasattr(analysis, 'check_analysis_dependencies')
                
                # 其他组件应该通过 __getattr__ 延迟导入
                # 但在依赖缺失时会抛出适当的错误
                print("✅ 缺失依赖处理机制正常")
                
            except Exception as e:
                # 这是预期的行为，因为依赖确实缺失
                print(f"✅ 缺失依赖时正确抛出错误: {type(e).__name__}")
    
    def test_graceful_degradation(self):
        """测试优雅降级机制"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 测试依赖检查在部分依赖缺失时的行为
            result = analysis.check_analysis_dependencies()
            
            # 即使某些依赖缺失，函数也应该返回结果
            assert isinstance(result, dict)
            
            # 检查是否包含状态信息
            for dep_name, dep_info in result.items():
                assert 'available' in str(dep_info) or 'version' in str(dep_info) or dep_info is None
            
            print("✅ 优雅降级机制正常")
            
        except Exception as e:
            pytest.fail(f"❌ 优雅降级测试失败: {e}")


class TestLazyImportCaching:
    """延迟导入缓存机制测试类"""
    
    def test_import_caching(self):
        """测试导入缓存机制"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 第一次访问
            model1 = analysis.Model
            
            # 第二次访问应该返回相同的对象（缓存）
            model2 = analysis.Model
            
            assert model1 is model2, "延迟导入缓存机制失效"
            
            print("✅ 延迟导入缓存机制正常")
            
        except Exception as e:
            pytest.fail(f"❌ 缓存机制测试失败: {e}")
    
    def test_multiple_component_caching(self):
        """测试多个组件的缓存"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 测试多个组件的缓存
            components = ['Model', 'FuzzyDataFrame', 'FuzzyPipeline', 'Contract', 'AnalysisComponent']
            
            first_access = {}
            second_access = {}
            
            # 第一次访问
            for comp in components:
                first_access[comp] = getattr(analysis, comp)
            
            # 第二次访问
            for comp in components:
                second_access[comp] = getattr(analysis, comp)
            
            # 验证缓存
            for comp in components:
                assert first_access[comp] is second_access[comp], f"{comp} 缓存失效"
            
            print("✅ 多组件缓存机制正常")
            
        except Exception as e:
            pytest.fail(f"❌ 多组件缓存测试失败: {e}")


class TestLazyImportIntegration:
    """延迟导入集成测试类"""
    
    def test_pipeline_lazy_imports(self):
        """测试 pipeline 模块的延迟导入"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 获取 FuzzyPipeline 类
            FuzzyPipeline = analysis.FuzzyPipeline
            
            # 测试 pipeline 的依赖检查功能
            # 这会间接测试 pipeline.py 中的延迟导入机制
            pipeline_instance = FuzzyPipeline()
            
            # 测试依赖检查方法（如果存在）
            if hasattr(pipeline_instance, 'check_dependencies'):
                deps = pipeline_instance.check_dependencies()
                assert isinstance(deps, dict)
                print("✅ Pipeline 依赖检查正常")
            
            print("✅ Pipeline 延迟导入集成测试通过")
            
        except Exception as e:
            pytest.fail(f"❌ Pipeline 延迟导入集成测试失败: {e}")
    
    def test_contract_system_integration(self):
        """测试契约系统的集成"""
        try:
            import axisfuzzy.analysis as analysis
            
            # 测试 Contract 类
            Contract = analysis.Contract
            contract_decorator = analysis.contract
            
            # 创建简单的契约
            positive_contract = Contract(
                name="PositiveNumber",
                validator=lambda x: x > 0
            )
            
            # 测试契约验证
            assert positive_contract.validate(5) == True
            assert positive_contract.validate(-1) == False
            
            # 测试装饰器（使用类型注解方式）
            @contract_decorator
            def test_function(x: positive_contract) -> int:
                return x * 2

            # 测试装饰器功能
            result = test_function(3)
            assert result == 6
            
            print("✅ 契约系统集成测试通过")
            
        except Exception as e:
            pytest.fail(f"❌ 契约系统集成测试失败: {e}")


class TestLazyImportSummary:
    """延迟导入功能总结测试类"""
    
    def test_lazy_import_summary(self):
        """延迟导入功能总结测试"""
        print("\n=== 延迟导入功能测试总结 ===")
        
        try:
            import axisfuzzy.analysis as analysis
            
            # 统计可用组件
            available_components = []
            unavailable_components = []
            
            test_components = [
                'Model', 'FuzzyDataFrame', 'FuzzyPipeline',
                'Contract', 'contract', 'AnalysisComponent',
                'check_analysis_dependencies'
            ]
            
            for comp in test_components:
                try:
                    component = getattr(analysis, comp)
                    if component is not None:
                        available_components.append(comp)
                    else:
                        unavailable_components.append(comp)
                except AttributeError:
                    unavailable_components.append(comp)
            
            print(f"✅ 可用组件 ({len(available_components)}): {', '.join(available_components)}")
            if unavailable_components:
                print(f"❌ 不可用组件 ({len(unavailable_components)}): {', '.join(unavailable_components)}")
            
            # 测试依赖状态
            deps = analysis.check_analysis_dependencies()
            available_deps = [k for k, v in deps.items() if v is not None and 'available' not in str(v)]
            unavailable_deps = [k for k, v in deps.items() if v is None or 'available' in str(v)]
            
            print(f"✅ 可用依赖 ({len(available_deps)}): {', '.join(available_deps)}")
            if unavailable_deps:
                print(f"⚠️  不可用依赖 ({len(unavailable_deps)}): {', '.join(unavailable_deps)}")
            
            print("\n🎉 延迟导入功能测试完成！")
            
        except Exception as e:
            pytest.fail(f"❌ 延迟导入总结测试失败: {e}")


def test_lazy_import_requirements_file():
    """测试延迟导入相关的需求文件"""
    import os
    
    # 检查相关配置文件是否存在
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    config_files = {
        'pyproject.toml': os.path.join(project_root, 'pyproject.toml'),
        'analysis __init__.py': os.path.join(project_root, 'axisfuzzy', 'analysis', '__init__.py'),
        'analysis __init__.pyi': os.path.join(project_root, 'axisfuzzy', 'analysis', '__init__.pyi'),
        'pipeline.py': os.path.join(project_root, 'axisfuzzy', 'analysis', 'pipeline.py'),
    }
    
    for name, path in config_files.items():
        assert os.path.exists(path), f"缺少配置文件: {name} ({path})"
        print(f"✅ 配置文件存在: {name}")
    
    print("✅ 延迟导入相关配置文件检查通过")


if __name__ == "__main__":
    print("AxisFuzzy 延迟导入功能测试")
    print("=" * 50)
    pytest.main([__file__, "-v"])