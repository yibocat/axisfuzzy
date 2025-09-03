#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可选依赖测试模块

测试 AxisFuzzy 可选依赖包的安装和基本功能，包括：
- analysis: pandas, matplotlib, networkx, pydot
- dev: pytest, notebook
- docs: sphinx 相关包

Author: AxisFuzzy Team
Date: 2025-01-25
"""

import sys
import pytest
import importlib.util
from packaging import version


class TestAnalysisDependencies:
    """分析依赖测试类"""
    
    def test_pandas_availability(self):
        """测试 pandas 是否可用"""
        try:
            import pandas as pd
            assert pd is not None
            print(f"✅ pandas 版本: {pd.__version__}")
        except ImportError:
            pytest.skip("pandas 未安装，跳过测试")
    
    def test_pandas_basic_functionality(self):
        """测试 pandas 基本功能"""
        try:
            import pandas as pd
            import numpy as np
            
            # 测试 DataFrame 创建
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            assert df.shape == (3, 2)
            
            # 测试基本操作
            result = df['A'].sum()
            assert result == 6
            
            print("✅ pandas 基本功能测试通过")
        except ImportError:
            pytest.skip("pandas 未安装，跳过测试")
        except Exception as e:
            pytest.fail(f"❌ pandas 基本功能测试失败: {e}")
    
    def test_matplotlib_availability(self):
        """测试 matplotlib 是否可用"""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            assert matplotlib is not None
            print(f"✅ matplotlib 版本: {matplotlib.__version__}")
        except ImportError:
            pytest.skip("matplotlib 未安装，跳过测试")
    
    def test_matplotlib_basic_functionality(self):
        """测试 matplotlib 基本功能"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 测试基本绘图功能（不显示图形）
            plt.ioff()  # 关闭交互模式
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            plt.close(fig)  # 关闭图形
            
            print("✅ matplotlib 基本功能测试通过")
        except ImportError:
            pytest.skip("matplotlib 未安装，跳过测试")
        except Exception as e:
            pytest.fail(f"❌ matplotlib 基本功能测试失败: {e}")
    
    def test_networkx_availability(self):
        """测试 networkx 是否可用"""
        try:
            import networkx as nx
            assert nx is not None
            print(f"✅ networkx 版本: {nx.__version__}")
        except ImportError:
            pytest.skip("networkx 未安装，跳过测试")
    
    def test_networkx_basic_functionality(self):
        """测试 networkx 基本功能"""
        try:
            import networkx as nx
            
            # 测试图创建
            G = nx.Graph()
            G.add_edge(1, 2)
            G.add_edge(2, 3)
            
            assert G.number_of_nodes() == 3
            assert G.number_of_edges() == 2
            
            print("✅ networkx 基本功能测试通过")
        except ImportError:
            pytest.skip("networkx 未安装，跳过测试")
        except Exception as e:
            pytest.fail(f"❌ networkx 基本功能测试失败: {e}")
    
    def test_pydot_availability(self):
        """测试 pydot 是否可用"""
        try:
            import pydot
            assert pydot is not None
            print(f"✅ pydot 版本: {pydot.__version__}")
        except ImportError:
            pytest.skip("pydot 未安装，跳过测试")


class TestDevDependencies:
    """开发依赖测试类"""
    
    def test_pytest_availability(self):
        """测试 pytest 是否可用"""
        try:
            import pytest
            assert pytest is not None
            print(f"✅ pytest 版本: {pytest.__version__}")
        except ImportError:
            pytest.skip("pytest 未安装，跳过测试")
    
    def test_pytest_basic_functionality(self):
        """测试 pytest 基本功能"""
        try:
            import pytest
            
            # 测试断言功能
            assert 1 + 1 == 2
            
            # 测试 pytest 标记功能
            @pytest.mark.skip(reason="测试跳过功能")
            def dummy_test():
                pass
            
            print("✅ pytest 基本功能测试通过")
        except ImportError:
            pytest.skip("pytest 未安装，跳过测试")
        except Exception as e:
            pytest.fail(f"❌ pytest 基本功能测试失败: {e}")
    
    def test_notebook_availability(self):
        """测试 notebook 是否可用"""
        try:
            import notebook
            assert notebook is not None
            print(f"✅ notebook 版本: {notebook.__version__}")
        except ImportError:
            pytest.skip("notebook 未安装，跳过测试")


class TestDocsDependencies:
    """文档依赖测试类"""
    
    def test_sphinx_availability(self):
        """测试 sphinx 是否可用"""
        try:
            import sphinx
            assert sphinx is not None
            print(f"✅ sphinx 版本: {sphinx.__version__}")
        except ImportError:
            pytest.skip("sphinx 未安装，跳过测试")
    
    def test_sphinx_extensions_availability(self):
        """测试 sphinx 扩展是否可用"""
        extensions = [
            'sphinx_copybutton',
            'sphinx_design',
            'sphinx_autodoc_typehints',
            'sphinx_tabs',
            'myst_parser',
            'babel',
            'pydata_sphinx_theme'
        ]
        
        available_extensions = []
        missing_extensions = []
        
        for ext in extensions:
            try:
                __import__(ext)
                available_extensions.append(ext)
                print(f"✅ {ext} 可用")
            except ImportError:
                missing_extensions.append(ext)
                print(f"⚠️ {ext} 未安装")
        
        if missing_extensions:
            pytest.skip(f"部分 sphinx 扩展未安装: {missing_extensions}")
        else:
            print("✅ 所有 sphinx 扩展都可用")


class TestOptionalDependenciesSummary:
    """可选依赖总结测试类"""
    
    def test_analysis_dependencies_summary(self):
        """生成分析依赖测试总结"""
        analysis_deps = ['pandas', 'matplotlib', 'networkx', 'pydot']
        available = []
        missing = []
        
        for dep in analysis_deps:
            try:
                module = __import__(dep)
                version_attr = getattr(module, '__version__', 'unknown')
                available.append(f"{dep}: {version_attr}")
            except ImportError:
                missing.append(dep)
        
        print("\n=== 分析依赖测试总结 ===")
        if available:
            print("已安装:")
            for dep in available:
                print(f"  {dep} ✅")
        
        if missing:
            print("未安装:")
            for dep in missing:
                print(f"  {dep} ❌")
        
        # 不返回值，避免 pytest 警告
        pass
    
    def test_dev_dependencies_summary(self):
        """生成开发依赖测试总结"""
        dev_deps = ['pytest', 'notebook']
        available = []
        missing = []
        
        for dep in dev_deps:
            try:
                module = __import__(dep)
                version_attr = getattr(module, '__version__', 'unknown')
                available.append(f"{dep}: {version_attr}")
            except ImportError:
                missing.append(dep)
        
        print("\n=== 开发依赖测试总结 ===")
        if available:
            print("已安装:")
            for dep in available:
                print(f"  {dep} ✅")
        
        if missing:
            print("未安装:")
            for dep in missing:
                print(f"  {dep} ❌")
        
        # 不返回值，避免 pytest 警告
        pass
    
    def test_docs_dependencies_summary(self):
        """生成文档依赖测试总结"""
        docs_deps = ['sphinx', 'sphinx_copybutton', 'sphinx_design', 
                    'sphinx_autodoc_typehints', 'sphinx_tabs', 'myst_parser', 
                    'babel', 'pydata_sphinx_theme']
        available = []
        missing = []
        
        for dep in docs_deps:
            try:
                module = __import__(dep)
                version_attr = getattr(module, '__version__', 'unknown')
                available.append(f"{dep}: {version_attr}")
            except ImportError:
                missing.append(dep)
        
        print("\n=== 文档依赖测试总结 ===")
        if available:
            print("已安装:")
            for dep in available:
                print(f"  {dep} ✅")
        
        if missing:
            print("未安装:")
            for dep in missing:
                print(f"  {dep} ❌")
        
        # 不返回值，避免 pytest 警告
        pass


def test_optional_requirements_files():
    """测试可选依赖文件是否存在"""
    import os
    
    requirements_files = {
        'analysis': '/Users/yibow/Documents/Fuzzy/AxisFuzzy/requirements/analysis_requirements.txt',
        'dev': '/Users/yibow/Documents/Fuzzy/AxisFuzzy/requirements/dev_requirements.txt',
        'docs': '/Users/yibow/Documents/Fuzzy/AxisFuzzy/requirements/docs_requirements.txt'
    }
    
    for name, file_path in requirements_files.items():
        assert os.path.exists(file_path), f"{name} 依赖文件不存在: {file_path}"
        print(f"✅ {name} 依赖文件存在")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])