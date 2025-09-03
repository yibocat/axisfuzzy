#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Sphinx 配置文件的正确性

这个模块验证 docs/conf.py 配置文件的各项设置是否正确，
包括项目信息、扩展配置、主题设置等。
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util
import re

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CONF_PY = DOCS_DIR / "conf.py"
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"


def load_conf_py() -> Dict[str, Any]:
    """
    动态加载 conf.py 配置文件
    
    Returns
    -------
    Dict[str, Any]
        配置文件中的变量字典
    """
    if not CONF_PY.exists():
        return {}
    
    # 临时添加文档目录到 Python 路径
    original_path = sys.path.copy()
    sys.path.insert(0, str(DOCS_DIR))
    
    try:
        # 使用 importlib 加载配置
        spec = importlib.util.spec_from_file_location("conf", CONF_PY)
        conf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf_module)
        
        # 提取配置变量
        config = {}
        for attr_name in dir(conf_module):
            if not attr_name.startswith('_'):
                config[attr_name] = getattr(conf_module, attr_name)
        
        return config
        
    except Exception as e:
        pytest.fail(f"加载 conf.py 失败: {e}")
    finally:
        # 恢复原始路径
        sys.path = original_path


class TestSphinxConfiguration:
    """测试 Sphinx 配置的测试类"""
    
    @pytest.fixture(scope="class")
    def config(self):
        """加载配置文件的 fixture"""
        return load_conf_py()
    
    def test_conf_py_exists(self):
        """测试 conf.py 文件是否存在"""
        assert CONF_PY.exists(), f"配置文件不存在: {CONF_PY}"
        assert CONF_PY.is_file(), f"配置路径不是文件: {CONF_PY}"
    
    def test_basic_project_info(self, config):
        """测试基本项目信息配置"""
        # 检查必需的项目信息
        required_fields = ['project', 'author', 'copyright']
        
        for field in required_fields:
            assert field in config, f"缺少必需的配置字段: {field}"
            assert config[field], f"配置字段 '{field}' 不能为空"
            assert isinstance(config[field], str), f"配置字段 '{field}' 应该是字符串"
        
        # 检查项目名称
        project_name = config['project']
        assert 'AxisFuzzy' in project_name, f"项目名称应包含 'AxisFuzzy': {project_name}"
        
        # 检查版权信息
        copyright_info = config['copyright']
        assert '2025' in copyright_info, f"版权信息应包含年份: {copyright_info}"
    
    def test_version_configuration(self, config):
        """测试版本配置"""
        # 检查版本字段
        version_fields = ['version', 'release']
        
        for field in version_fields:
            if field in config:
                version_value = config[field]
                assert isinstance(version_value, str), f"版本字段 '{field}' 应该是字符串"
                assert len(version_value) > 0, f"版本字段 '{field}' 不能为空"
                
                # 检查版本格式（基本的语义版本检查）
                version_pattern = r'^\d+\.\d+(\.\d+)?'
                assert re.match(version_pattern, version_value), \
                    f"版本格式不正确: {version_value}"
    
    def test_extensions_configuration(self, config):
        """测试扩展配置"""
        assert 'extensions' in config, "缺少 extensions 配置"
        
        extensions = config['extensions']
        assert isinstance(extensions, list), "extensions 应该是列表"
        assert len(extensions) > 0, "extensions 列表不能为空"
        
        # 检查必需的扩展
        required_extensions = [
            'sphinx.ext.autodoc',
            'myst_parser'
        ]
        
        for ext in required_extensions:
            assert ext in extensions, f"缺少必需的扩展: {ext}"
        
        # 检查扩展名格式
        for ext in extensions:
            assert isinstance(ext, str), f"扩展名应该是字符串: {ext}"
            assert len(ext) > 0, "扩展名不能为空"
            # 扩展名应该是有效的 Python 模块名
            assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', ext), \
                f"扩展名格式不正确: {ext}"
    
    def test_html_theme_configuration(self, config):
        """测试 HTML 主题配置"""
        # 检查主题设置
        if 'html_theme' in config:
            theme = config['html_theme']
            assert isinstance(theme, str), "html_theme 应该是字符串"
            assert len(theme) > 0, "html_theme 不能为空"
            
            # 检查是否是已知的主题
            known_themes = [
                'alabaster', 'classic', 'sphinxdoc', 'scrolls',
                'agogo', 'traditional', 'nature', 'haiku',
                'pyramid', 'bizstyle', 'sphinx_rtd_theme',
                'pydata_sphinx_theme', 'furo', 'book_theme'
            ]
            
            print(f"\n使用的主题: {theme}")
        
        # 检查主题选项
        if 'html_theme_options' in config:
            theme_options = config['html_theme_options']
            assert isinstance(theme_options, dict), "html_theme_options 应该是字典"
    
    def test_path_configuration(self, config):
        """测试路径配置"""
        # 检查模板路径
        if 'templates_path' in config:
            templates_path = config['templates_path']
            assert isinstance(templates_path, list), "templates_path 应该是列表"
            
            for path in templates_path:
                template_dir = DOCS_DIR / path
                if not template_dir.exists():
                    print(f"\n警告: 模板目录不存在: {template_dir}")
        
        # 检查静态文件路径
        if 'html_static_path' in config:
            static_path = config['html_static_path']
            assert isinstance(static_path, list), "html_static_path 应该是列表"
            
            for path in static_path:
                static_dir = DOCS_DIR / path
                if not static_dir.exists():
                    print(f"\n警告: 静态文件目录不存在: {static_dir}")
        
        # 检查排除模式
        if 'exclude_patterns' in config:
            exclude_patterns = config['exclude_patterns']
            assert isinstance(exclude_patterns, list), "exclude_patterns 应该是列表"
    
    def test_autodoc_configuration(self, config):
        """测试 autodoc 扩展配置"""
        extensions = config.get('extensions', [])
        
        if 'sphinx.ext.autodoc' in extensions:
            # 检查 autodoc 默认选项
            if 'autodoc_default_options' in config:
                autodoc_options = config['autodoc_default_options']
                assert isinstance(autodoc_options, dict), \
                    "autodoc_default_options 应该是字典"
                
                # 检查常见的 autodoc 选项
                valid_options = [
                    'members', 'undoc-members', 'inherited-members',
                    'show-inheritance', 'member-order', 'special-members'
                ]
                
                for option in autodoc_options:
                    if option not in valid_options:
                        print(f"\n警告: 未知的 autodoc 选项: {option}")
            
            # 检查类型提示配置
            if 'autodoc_typehints' in config:
                typehints = config['autodoc_typehints']
                valid_values = ['signature', 'description', 'none']
                assert typehints in valid_values, \
                    f"autodoc_typehints 值无效: {typehints}"
    
    def test_source_suffix_configuration(self, config):
        """测试源文件后缀配置"""
        if 'source_suffix' in config:
            source_suffix = config['source_suffix']
            
            if isinstance(source_suffix, dict):
                # 新格式：{'.rst': 'restructuredtext', '.md': 'markdown'}
                for suffix, parser in source_suffix.items():
                    assert suffix.startswith('.'), f"文件后缀应以点开头: {suffix}"
                    assert isinstance(parser, str), f"解析器名称应该是字符串: {parser}"
            elif isinstance(source_suffix, (str, list)):
                # 旧格式：'.rst' 或 ['.rst', '.md']
                if isinstance(source_suffix, str):
                    assert source_suffix.startswith('.'), \
                        f"文件后缀应以点开头: {source_suffix}"
                else:
                    for suffix in source_suffix:
                        assert suffix.startswith('.'), \
                            f"文件后缀应以点开头: {suffix}"
    
    def test_master_doc_configuration(self, config):
        """测试主文档配置"""
        if 'master_doc' in config:
            master_doc = config['master_doc']
            assert isinstance(master_doc, str), "master_doc 应该是字符串"
            assert len(master_doc) > 0, "master_doc 不能为空"
            
            # 检查主文档文件是否存在
            possible_files = [
                DOCS_DIR / f"{master_doc}.rst",
                DOCS_DIR / f"{master_doc}.md"
            ]
            
            master_exists = any(f.exists() for f in possible_files)
            if not master_exists:
                print(f"\n警告: 主文档文件不存在: {master_doc}")
    
    def test_language_configuration(self, config):
        """测试语言配置"""
        if 'language' in config:
            language = config['language']
            if language is not None:
                assert isinstance(language, str), "language 应该是字符串或 None"
                
                # 检查是否是有效的语言代码
                valid_languages = [
                    'en', 'zh', 'zh_CN', 'zh_TW', 'ja', 'ko', 
                    'fr', 'de', 'es', 'it', 'ru', 'pt', 'ar'
                ]
                
                if language not in valid_languages:
                    print(f"\n警告: 可能的无效语言代码: {language}")
    
    def test_math_configuration(self, config):
        """测试数学公式配置"""
        extensions = config.get('extensions', [])
        
        # 检查数学扩展
        math_extensions = [
            'sphinx.ext.mathjax', 'sphinx.ext.imgmath', 
            'sphinx.ext.jsmath'
        ]
        
        math_ext_found = any(ext in extensions for ext in math_extensions)
        
        if math_ext_found:
            print("\n✅ 数学公式支持已启用")
            
            # 检查 MathJax 配置
            if 'sphinx.ext.mathjax' in extensions:
                if 'mathjax_config' in config:
                    mathjax_config = config['mathjax_config']
                    assert isinstance(mathjax_config, dict), \
                        "mathjax_config 应该是字典"
        else:
            print("\n⚠️ 未启用数学公式支持")
    
    def test_intersphinx_configuration(self, config):
        """测试 Intersphinx 配置"""
        extensions = config.get('extensions', [])
        
        if 'sphinx.ext.intersphinx' in extensions:
            if 'intersphinx_mapping' in config:
                intersphinx_mapping = config['intersphinx_mapping']
                assert isinstance(intersphinx_mapping, dict), \
                    "intersphinx_mapping 应该是字典"
                
                print(f"\n✅ Intersphinx 映射配置: {len(intersphinx_mapping)} 个项目")
                
                for name, mapping in intersphinx_mapping.items():
                    assert isinstance(mapping, (tuple, list)), \
                        f"Intersphinx 映射 '{name}' 应该是元组或列表"
                    assert len(mapping) >= 1, \
                        f"Intersphinx 映射 '{name}' 至少需要一个 URL"


class TestConfigurationConsistency:
    """测试配置一致性的测试类"""
    
    def test_pyproject_toml_consistency(self):
        """测试与 pyproject.toml 的一致性"""
        if not PYPROJECT_TOML.exists():
            pytest.skip("pyproject.toml 文件不存在")
        
        config = load_conf_py()
        
        try:
            import tomllib
            with open(PYPROJECT_TOML, 'rb') as f:
                pyproject_data = tomllib.load(f)
        except ImportError:
            pytest.skip("tomllib 不可用，跳过 pyproject.toml 检查")
        except Exception as e:
            pytest.fail(f"读取 pyproject.toml 失败: {e}")
        
        # 检查项目名称一致性
        if 'project' in pyproject_data and 'name' in pyproject_data['project']:
            pyproject_name = pyproject_data['project']['name']
            conf_project = config.get('project', '')
            
            # 名称可能不完全相同，但应该相关
            if pyproject_name.lower() != conf_project.lower():
                print(f"\n注意: 项目名称不一致")
                print(f"  pyproject.toml: {pyproject_name}")
                print(f"  conf.py: {conf_project}")
        
        # 检查版本一致性
        if 'project' in pyproject_data and 'version' in pyproject_data['project']:
            pyproject_version = pyproject_data['project']['version']
            conf_release = config.get('release', '')
            
            if pyproject_version != conf_release:
                print(f"\n注意: 版本号不一致")
                print(f"  pyproject.toml: {pyproject_version}")
                print(f"  conf.py: {conf_release}")
    
    def test_extension_dependencies(self):
        """测试扩展依赖的一致性"""
        config = load_conf_py()
        extensions = config.get('extensions', [])
        
        # 检查扩展之间的依赖关系
        dependency_rules = {
            'sphinx_autodoc_typehints': ['sphinx.ext.autodoc'],
            'sphinx.ext.napoleon': ['sphinx.ext.autodoc'],
            'sphinx.ext.autosummary': ['sphinx.ext.autodoc']
        }
        
        for ext, deps in dependency_rules.items():
            if ext in extensions:
                for dep in deps:
                    assert dep in extensions, \
                        f"扩展 '{ext}' 需要依赖 '{dep}'，但未在配置中找到"


def test_configuration_summary():
    """生成配置状态的总结报告"""
    print("\n=== Sphinx 配置状态总结 ===")
    print(f"配置文件: {CONF_PY}")
    
    if not CONF_PY.exists():
        print("❌ 配置文件不存在")
        return
    
    try:
        config = load_conf_py()
        print("✅ 配置文件加载成功")
        
        # 基本信息
        print(f"\n项目信息:")
        print(f"  项目名称: {config.get('project', '未设置')}")
        print(f"  作者: {config.get('author', '未设置')}")
        print(f"  版本: {config.get('release', '未设置')}")
        
        # 扩展信息
        extensions = config.get('extensions', [])
        print(f"\n扩展配置:")
        print(f"  扩展数量: {len(extensions)}")
        
        builtin_count = sum(1 for ext in extensions if ext.startswith('sphinx.ext.'))
        third_party_count = len(extensions) - builtin_count
        print(f"  内置扩展: {builtin_count}")
        print(f"  第三方扩展: {third_party_count}")
        
        # 主题信息
        theme = config.get('html_theme', '未设置')
        print(f"\nHTML 主题: {theme}")
        
        # 路径配置
        static_paths = config.get('html_static_path', [])
        template_paths = config.get('templates_path', [])
        print(f"\n路径配置:")
        print(f"  静态文件路径: {len(static_paths)} 个")
        print(f"  模板路径: {len(template_paths)} 个")
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")


if __name__ == "__main__":
    # 允许直接运行此文件进行快速检查
    pytest.main([__file__, "-v"])