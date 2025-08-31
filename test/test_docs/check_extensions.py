#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Sphinx 扩展的安装状态

这个测试模块验证 docs/conf.py 中配置的所有 Sphinx 扩展是否正确安装。
确保文档构建系统的所有依赖都可用。
"""

import pytest
import sys
import importlib
from typing import List, Tuple

# 从 docs/conf.py 中提取的扩展列表
SPHINX_EXTENSIONS = [
    'myst_parser',            # 支持 reStructuredText 和 Markdown (.md 文件)
    'sphinx.ext.autodoc',     # 从 docstring 自动生成文档
    'sphinx.ext.napoleon',    # 支持 Google 和 NumPy 风格的 docstring
    'sphinx.ext.viewcode',    # 为文档页面添加"查看源码"链接
    'sphinx_copybutton',      # 为代码块添加"复制"按钮
    'sphinx_design',          # 提供设计组件（卡片、网格等）
    'sphinx_autodoc_typehints', # 在 autodoc 输出中漂亮地渲染 Python 3 类型提示
    'sphinx.ext.mathjax',     # 使用 MathJax 渲染数学公式
    'sphinx.ext.autosummary', # 为 API 文档生成摘要表格
    'sphinx_tabs.tabs',       # 添加标签页功能
    'pydata_sphinx_theme',    # PyData Sphinx 主题扩展
]

# 扩展名到包名的映射
PACKAGE_MAPPING = {
    'myst_parser': 'myst-parser',
    'sphinx_copybutton': 'sphinx-copybutton',
    'sphinx_design': 'sphinx-design',
    'sphinx_autodoc_typehints': 'sphinx-autodoc-typehints',
    'sphinx_tabs.tabs': 'sphinx-tabs',
    'pydata_sphinx_theme': 'pydata-sphinx-theme',
}


class TestSphinxExtensions:
    """测试 Sphinx 扩展安装状态的测试类"""
    
    def test_sphinx_available(self):
        """测试 Sphinx 本身是否可用"""
        try:
            import sphinx
            assert sphinx.__version__, "Sphinx 版本信息不可用"
        except ImportError:
            pytest.fail("Sphinx 未安装")
    
    @pytest.mark.parametrize("extension", SPHINX_EXTENSIONS)
    def test_extension_importable(self, extension: str):
        """测试每个扩展是否可以正常导入
        
        Parameters
        ----------
        extension : str
            要测试的扩展名称
        """
        try:
            if extension.startswith('sphinx.ext.'):
                # Sphinx 内置扩展，只需要确保 Sphinx 可用
                import sphinx
                # 尝试导入具体的扩展模块
                importlib.import_module(extension)
            else:
                # 第三方扩展
                importlib.import_module(extension)
        except ImportError as e:
            package_name = PACKAGE_MAPPING.get(extension, extension)
            pytest.fail(
                f"扩展 '{extension}' 导入失败: {e}\n"
                f"建议安装: uv add {package_name}"
            )
    
    def test_all_extensions_available(self):
        """测试所有扩展的整体可用性"""
        missing_extensions = []
        installed_extensions = []
        
        for ext in SPHINX_EXTENSIONS:
            try:
                if ext.startswith('sphinx.ext.'):
                    import sphinx
                    importlib.import_module(ext)
                else:
                    importlib.import_module(ext)
                installed_extensions.append(ext)
            except ImportError:
                missing_extensions.append(ext)
        
        # 断言所有扩展都已安装
        if missing_extensions:
            missing_packages = [
                PACKAGE_MAPPING.get(ext, ext) 
                for ext in missing_extensions 
                if not ext.startswith('sphinx.ext.')
            ]
            error_msg = (
                f"缺失 {len(missing_extensions)} 个扩展: {missing_extensions}\n"
                f"建议安装命令: uv add {' '.join(missing_packages)}"
            )
            pytest.fail(error_msg)
        
        # 验证安装数量
        assert len(installed_extensions) == len(SPHINX_EXTENSIONS), \
            f"预期 {len(SPHINX_EXTENSIONS)} 个扩展，实际安装 {len(installed_extensions)} 个"
    
    def test_extension_versions(self):
        """测试扩展版本信息（如果可用）"""
        version_info = {}
        
        for ext in SPHINX_EXTENSIONS:
            try:
                if ext.startswith('sphinx.ext.'):
                    import sphinx
                    version_info[ext] = f"Sphinx {sphinx.__version__}"
                else:
                    module = importlib.import_module(ext)
                    version = getattr(module, '__version__', '未知版本')
                    version_info[ext] = version
            except ImportError:
                version_info[ext] = "未安装"
        
        # 确保至少有一些版本信息可用
        available_versions = [v for v in version_info.values() if v != "未安装"]
        assert len(available_versions) > 0, "没有可用的扩展版本信息"
        
        # 打印版本信息用于调试
        print("\n扩展版本信息:")
        for ext, version in version_info.items():
            print(f"  {ext}: {version}")


def test_docs_build_dependencies():
    """测试文档构建的核心依赖"""
    core_deps = ['sphinx', 'docutils']
    
    for dep in core_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', '未知版本')
            print(f"{dep}: {version}")
        except ImportError:
            pytest.fail(f"核心依赖 '{dep}' 未安装")


if __name__ == "__main__":
    # 允许直接运行此文件进行快速检查
    pytest.main([__file__, "-v"])