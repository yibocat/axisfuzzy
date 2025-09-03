#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Sphinx 扩展的安装和配置

这个模块验证 AxisFuzzy 文档系统中所有 Sphinx 扩展的安装状态和配置正确性。
确保文档构建系统的所有依赖都可用且配置正确。
"""

import pytest
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CONF_PY = DOCS_DIR / "conf.py"


def get_extensions_from_conf() -> List[str]:
    """
    从 docs/conf.py 中动态提取扩展列表
    
    Returns
    -------
    List[str]
        配置文件中定义的扩展列表
    """
    if not CONF_PY.exists():
        return []
    
    # 读取配置文件内容
    with open(CONF_PY, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 简单的扩展提取（查找 extensions = [...] 块）
    import re
    pattern = r'extensions\s*=\s*\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return []
    
    # 提取扩展名
    extensions_block = match.group(1)
    extensions = []
    for line in extensions_block.split('\n'):
        line = line.strip()
        if line.startswith("'") or line.startswith('"'):
            # 提取引号内的扩展名
            ext_match = re.search(r'["\']([^"\',]+)["\']', line)
            if ext_match:
                extensions.append(ext_match.group(1))
    
    return extensions


# 从配置文件动态获取扩展列表
CONFIGURED_EXTENSIONS = get_extensions_from_conf()

# 扩展分类
SPHINX_BUILTIN_EXTENSIONS = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', 
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary'
]

THIRD_PARTY_EXTENSIONS = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_autodoc_typehints',
    'sphinx_tabs.tabs',
    'pydata_sphinx_theme'
]

# 扩展名到包名的映射（用于安装建议）
PACKAGE_MAPPING = {
    'myst_parser': 'myst-parser',
    'sphinx_copybutton': 'sphinx-copybutton', 
    'sphinx_design': 'sphinx-design',
    'sphinx_autodoc_typehints': 'sphinx-autodoc-typehints',
    'sphinx_tabs.tabs': 'sphinx-tabs',
    'pydata_sphinx_theme': 'pydata-sphinx-theme'
}


class TestSphinxExtensions:
    """测试 Sphinx 扩展安装和配置的测试类"""
    
    def test_sphinx_core_available(self):
        """测试 Sphinx 核心是否可用"""
        try:
            import sphinx
            assert hasattr(sphinx, '__version__'), "Sphinx 版本信息不可用"
            # 检查 Sphinx 版本是否足够新（至少 4.0+）
            version_parts = sphinx.__version__.split('.')
            major_version = int(version_parts[0])
            assert major_version >= 4, f"Sphinx 版本过低: {sphinx.__version__}，建议 4.0+"
        except ImportError:
            pytest.fail("Sphinx 未安装，请运行: uv add sphinx")
    
    def test_conf_py_exists(self):
        """测试 docs/conf.py 配置文件是否存在"""
        assert CONF_PY.exists(), f"配置文件不存在: {CONF_PY}"
        assert CONF_PY.is_file(), f"配置路径不是文件: {CONF_PY}"
    
    def test_extensions_configured(self):
        """测试扩展是否在配置文件中正确配置"""
        extensions = get_extensions_from_conf()
        assert len(extensions) > 0, "配置文件中未找到扩展列表"
        
        # 检查是否包含基本的必需扩展
        required_extensions = ['sphinx.ext.autodoc', 'myst_parser']
        for ext in required_extensions:
            assert ext in extensions, f"缺少必需扩展: {ext}"
    
    @pytest.mark.parametrize("extension", CONFIGURED_EXTENSIONS)
    def test_extension_importable(self, extension: str):
        """测试每个配置的扩展是否可以正常导入"""
        try:
            if extension.startswith('sphinx.ext.'):
                # Sphinx 内置扩展
                import sphinx
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
    
    def test_builtin_extensions_available(self):
        """测试 Sphinx 内置扩展的可用性"""
        try:
            import sphinx
        except ImportError:
            pytest.skip("Sphinx 未安装，跳过内置扩展测试")
        
        for ext in SPHINX_BUILTIN_EXTENSIONS:
            if ext in CONFIGURED_EXTENSIONS:
                try:
                    importlib.import_module(ext)
                except ImportError:
                    pytest.fail(f"Sphinx 内置扩展 '{ext}' 不可用")
    
    def test_third_party_extensions_versions(self):
        """测试第三方扩展的版本信息"""
        version_info = {}
        missing_extensions = []
        
        for ext in THIRD_PARTY_EXTENSIONS:
            if ext in CONFIGURED_EXTENSIONS:
                try:
                    module = importlib.import_module(ext)
                    version = getattr(module, '__version__', '未知版本')
                    version_info[ext] = version
                except ImportError:
                    missing_extensions.append(ext)
        
        # 如果有缺失的扩展，提供安装建议
        if missing_extensions:
            missing_packages = [
                PACKAGE_MAPPING.get(ext, ext) for ext in missing_extensions
            ]
            pytest.fail(
                f"缺失第三方扩展: {missing_extensions}\n"
                f"建议安装: uv add {' '.join(missing_packages)}"
            )
        
        # 打印版本信息（用于调试）
        if version_info:
            print("\n第三方扩展版本信息:")
            for ext, version in version_info.items():
                print(f"  {ext}: {version}")
    
    def test_theme_extensions_available(self):
        """测试主题扩展的可用性"""
        theme_extensions = [
            ext for ext in CONFIGURED_EXTENSIONS 
            if 'theme' in ext.lower()
        ]
        
        if not theme_extensions:
            pytest.skip("配置中未找到主题扩展")
        
        for theme_ext in theme_extensions:
            try:
                importlib.import_module(theme_ext)
            except ImportError:
                package_name = PACKAGE_MAPPING.get(theme_ext, theme_ext)
                pytest.fail(
                    f"主题扩展 '{theme_ext}' 不可用\n"
                    f"建议安装: uv add {package_name}"
                )
    
    def test_extension_compatibility(self):
        """测试扩展之间的兼容性"""
        # 检查可能的冲突扩展组合
        problematic_combinations = [
            # 示例：某些扩展可能不兼容
            # (['ext1', 'ext2'], "这两个扩展可能冲突")
        ]
        
        for extensions, reason in problematic_combinations:
            if all(ext in CONFIGURED_EXTENSIONS for ext in extensions):
                pytest.fail(f"检测到可能的扩展冲突: {extensions} - {reason}")
    
    def test_docs_directory_structure(self):
        """测试文档目录结构的完整性"""
        # 检查基本目录结构
        assert DOCS_DIR.exists(), f"文档目录不存在: {DOCS_DIR}"
        assert DOCS_DIR.is_dir(), f"文档路径不是目录: {DOCS_DIR}"
        
        # 检查必需的文件和目录
        required_items = [
            'conf.py',      # 配置文件
            'index.rst',    # 主页（可能是 .rst 或 .md）
            'index.md',     # 或者是 Markdown 格式
            '_static',      # 静态文件目录
            '_templates'    # 模板目录
        ]
        
        existing_items = []
        for item in required_items:
            item_path = DOCS_DIR / item
            if item_path.exists():
                existing_items.append(item)
        
        # 至少应该有 conf.py 和一个主页文件
        assert 'conf.py' in existing_items, "缺少 conf.py 配置文件"
        assert ('index.rst' in existing_items or 'index.md' in existing_items), \
            "缺少主页文件 (index.rst 或 index.md)"


def test_extension_summary():
    """生成扩展安装状态的总结报告"""
    print("\n=== Sphinx 扩展安装状态总结 ===")
    print(f"配置文件: {CONF_PY}")
    
    extensions = get_extensions_from_conf()
    print(f"配置的扩展数量: {len(extensions)}")
    
    # 分类统计
    builtin_count = sum(1 for ext in extensions if ext.startswith('sphinx.ext.'))
    third_party_count = len(extensions) - builtin_count
    
    print(f"  - Sphinx 内置扩展: {builtin_count}")
    print(f"  - 第三方扩展: {third_party_count}")
    
    # 检查安装状态
    installed = []
    missing = []
    
    for ext in extensions:
        try:
            importlib.import_module(ext)
            installed.append(ext)
        except ImportError:
            missing.append(ext)
    
    print(f"\n安装状态:")
    print(f"  ✅ 已安装: {len(installed)}/{len(extensions)}")
    if missing:
        print(f"  ❌ 缺失: {len(missing)}")
        for ext in missing:
            package_name = PACKAGE_MAPPING.get(ext, ext)
            print(f"    - {ext} (安装: uv add {package_name})")
    
    # 断言所有扩展都已安装
    assert len(missing) == 0, f"有 {len(missing)} 个扩展未安装: {missing}"


if __name__ == "__main__":
    # 允许直接运行此文件进行快速检查
    pytest.main([__file__, "-v"])