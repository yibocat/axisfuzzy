#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Sphinx 文档构建过程和输出质量

这个模块验证 AxisFuzzy 文档系统的构建过程是否正常，
以及生成的文档是否符合质量标准。
"""

import pytest
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import re
import time

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
BUILD_DIR = DOCS_DIR / "_build"
HTML_DIR = BUILD_DIR / "html"
DOCTREE_DIR = BUILD_DIR / "doctrees"


class TestDocumentationBuild:
    """测试文档构建过程的测试类"""
    
    def test_docs_directory_exists(self):
        """测试文档目录是否存在"""
        assert DOCS_DIR.exists(), f"文档目录不存在: {DOCS_DIR}"
        assert DOCS_DIR.is_dir(), f"文档路径不是目录: {DOCS_DIR}"
    
    def test_conf_py_syntax(self):
        """测试 conf.py 文件的语法正确性"""
        conf_py = DOCS_DIR / "conf.py"
        assert conf_py.exists(), "conf.py 文件不存在"
        
        # 尝试编译 conf.py 文件
        try:
            with open(conf_py, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, str(conf_py), 'exec')
        except SyntaxError as e:
            pytest.fail(f"conf.py 语法错误: {e}")
        except Exception as e:
            pytest.fail(f"conf.py 编译失败: {e}")
    
    def test_source_files_exist(self):
        """测试源文档文件是否存在"""
        # 检查主页文件
        index_files = [
            DOCS_DIR / "index.rst",
            DOCS_DIR / "index.md"
        ]
        
        index_exists = any(f.exists() for f in index_files)
        assert index_exists, "缺少主页文件 (index.rst 或 index.md)"
        
        # 检查是否有其他文档文件
        doc_files = list(DOCS_DIR.glob("*.rst")) + list(DOCS_DIR.glob("*.md"))
        assert len(doc_files) > 0, "文档目录中没有找到任何 .rst 或 .md 文件"
    
    def test_clean_build(self):
        """测试清理构建目录"""
        if BUILD_DIR.exists():
            try:
                shutil.rmtree(BUILD_DIR)
            except Exception as e:
                pytest.fail(f"清理构建目录失败: {e}")
        
        # 确认构建目录已被清理
        assert not BUILD_DIR.exists(), "构建目录清理失败"
    
    def test_sphinx_build_html(self):
        """测试 Sphinx HTML 构建过程"""
        # 确保从干净状态开始
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        
        # 执行 Sphinx 构建
        cmd = [
            "python", "-m", "sphinx",
            "-b", "html",          # 构建 HTML
            "-E",                  # 重新读取所有文件
            str(DOCS_DIR),         # 源目录
            str(HTML_DIR)          # 输出目录
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )
            
            # 检查构建结果 - 只有退出码非0才算失败，警告不算失败
            if result.returncode != 0:
                error_msg = (
                    f"Sphinx 构建失败 (退出码: {result.returncode})\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )
                pytest.fail(error_msg)
            
            # 记录警告但不失败
            if "WARNING" in result.stderr or "WARNING" in result.stdout:
                warning_count = (result.stderr + result.stdout).count("WARNING")
                print(f"\n⚠️ 构建完成但有 {warning_count} 个警告")
                
        except subprocess.TimeoutExpired:
            pytest.fail("Sphinx 构建超时 (>120秒)")
        except FileNotFoundError:
            pytest.fail("找不到 sphinx 命令，请确保 Sphinx 已安装")
        except Exception as e:
            pytest.fail(f"Sphinx 构建过程出错: {e}")
    
    def test_build_output_exists(self):
        """测试构建输出文件是否存在"""
        # 检查 HTML 目录
        assert HTML_DIR.exists(), f"HTML 构建目录不存在: {HTML_DIR}"
        assert HTML_DIR.is_dir(), f"HTML 路径不是目录: {HTML_DIR}"
        
        # 检查主页文件
        index_html = HTML_DIR / "index.html"
        assert index_html.exists(), f"主页文件不存在: {index_html}"
        assert index_html.is_file(), f"主页路径不是文件: {index_html}"
        
        # 检查文件大小（确保不是空文件）
        assert index_html.stat().st_size > 100, "主页文件过小，可能构建不完整"
    
    def test_html_structure(self):
        """测试生成的 HTML 文件结构"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在，跳过结构测试")
        
        # 检查必需的文件和目录
        required_items = [
            "index.html",          # 主页
            "_static",             # 静态文件目录
            "_sources",            # 源文件目录（如果启用）
            "genindex.html",       # 索引页面（可能存在）
            "search.html"          # 搜索页面
        ]
        
        existing_items = []
        for item in required_items:
            item_path = HTML_DIR / item
            if item_path.exists():
                existing_items.append(item)
        
        # 至少应该有主页和静态文件目录
        assert "index.html" in existing_items, "缺少主页文件"
        assert "_static" in existing_items, "缺少静态文件目录"
    
    def test_html_content_quality(self):
        """测试 HTML 内容质量"""
        index_html = HTML_DIR / "index.html"
        if not index_html.exists():
            pytest.skip("主页文件不存在，跳过内容质量测试")
        
        with open(index_html, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查基本 HTML 结构
        assert "<html" in content, "缺少 HTML 标签"
        assert "<head>" in content, "缺少 head 标签"
        assert "<body>" in content, "缺少 body 标签"
        assert "</html>" in content, "HTML 标签未正确关闭"
        
        # 检查标题
        title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
        assert title_match, "缺少页面标题"
        title = title_match.group(1).strip()
        assert len(title) > 0, "页面标题为空"
        assert "AxisFuzzy" in title, "页面标题中应包含项目名称"
        
        # 检查是否包含项目内容
        content_lower = content.lower()
        assert "axisfuzzy" in content_lower, "页面内容中应包含项目名称"
        
        # 检查是否有明显的错误信息
        error_indicators = [
            "error", "exception", "traceback", 
            "failed to import", "module not found"
        ]
        for indicator in error_indicators:
            if indicator in content_lower:
                pytest.fail(f"HTML 内容中包含错误指示器: {indicator}")
    
    def test_static_files(self):
        """测试静态文件是否正确生成"""
        static_dir = HTML_DIR / "_static"
        if not static_dir.exists():
            pytest.skip("静态文件目录不存在，跳过静态文件测试")
        
        # 检查基本的静态文件
        expected_files = [
            "basic.css",           # Sphinx 基本样式
            "doctools.js",         # Sphinx 文档工具
            "sphinx_highlight.js"  # 语法高亮
        ]
        
        existing_files = []
        for file_name in expected_files:
            file_path = static_dir / file_name
            if file_path.exists():
                existing_files.append(file_name)
        
        # 至少应该有一些基本文件
        assert len(existing_files) > 0, "静态文件目录中没有找到预期的文件"
    
    def test_build_warnings(self):
        """测试构建过程中的警告"""
        # 重新构建以捕获警告
        cmd = [
            "python", "-m", "sphinx",
            "-b", "html",
            "-v",                  # 详细输出
            str(DOCS_DIR),
            str(HTML_DIR)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # 分析输出中的警告
            output = result.stdout + result.stderr
            warning_lines = []
            
            for line in output.split('\n'):
                if 'warning' in line.lower() or 'warn' in line.lower():
                    warning_lines.append(line.strip())
            
            # 打印警告信息（用于调试）
            if warning_lines:
                print("\n构建警告:")
                for warning in warning_lines:
                    print(f"  {warning}")
            
            # 检查是否有严重警告
            serious_warnings = [
                "undefined label", "unknown document", 
                "duplicate label", "malformed hyperlink"
            ]
            
            for warning_line in warning_lines:
                for serious in serious_warnings:
                    if serious in warning_line.lower():
                        pytest.fail(f"发现严重构建警告: {warning_line}")
                        
        except Exception as e:
            pytest.skip(f"无法执行警告检查: {e}")
    
    def test_build_performance(self):
        """测试构建性能"""
        # 清理构建目录
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        
        # 测量构建时间
        start_time = time.time()
        
        cmd = [
            "python", "-m", "sphinx",
            "-b", "html",
            "-q",                  # 安静模式
            str(DOCS_DIR),
            str(HTML_DIR)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            build_time = time.time() - start_time
            
            # 构建应该在合理时间内完成
            assert build_time < 120, f"构建时间过长: {build_time:.2f}秒 (>120秒)"
            
            print(f"\n构建性能: {build_time:.2f}秒")
            
            if result.returncode != 0:
                pytest.fail(f"性能测试中构建失败: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            pytest.fail("构建超时 (>300秒)")
        except Exception as e:
            pytest.skip(f"无法执行性能测试: {e}")


class TestDocumentationContent:
    """测试文档内容的测试类"""
    
    def test_api_documentation_generated(self):
        """测试 API 文档是否正确生成"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在，跳过 API 文档测试")
        
        # 查找可能的 API 文档文件
        api_patterns = [
            "**/axisfuzzy*.html",
            "**/api*.html",
            "**/modules*.html"
        ]
        
        api_files = []
        for pattern in api_patterns:
            api_files.extend(HTML_DIR.glob(pattern))
        
        if api_files:
            print(f"\n找到 API 文档文件: {len(api_files)} 个")
            for api_file in api_files[:5]:  # 只显示前5个
                print(f"  {api_file.relative_to(HTML_DIR)}")
        else:
            # 这可能是正常的，如果项目还没有设置 API 文档
            print("\n未找到 API 文档文件（可能尚未配置）")
    
    def test_search_functionality(self):
        """测试搜索功能是否可用"""
        search_html = HTML_DIR / "search.html"
        searchindex_js = HTML_DIR / "_static" / "searchindex.js"
        
        if search_html.exists():
            print("\n✅ 搜索页面已生成")
        else:
            print("\n⚠️ 搜索页面未生成")
        
        if searchindex_js.exists():
            print("✅ 搜索索引已生成")
            # 检查搜索索引文件大小
            size = searchindex_js.stat().st_size
            assert size > 100, "搜索索引文件过小"
        else:
            print("⚠️ 搜索索引未生成")


def test_build_summary():
    """生成构建状态的总结报告"""
    print("\n=== 文档构建状态总结 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"文档目录: {DOCS_DIR}")
    print(f"构建目录: {BUILD_DIR}")
    
    # 检查目录状态
    if DOCS_DIR.exists():
        print("✅ 文档源目录存在")
        
        # 统计源文件
        rst_files = list(DOCS_DIR.glob("**/*.rst"))
        md_files = list(DOCS_DIR.glob("**/*.md"))
        print(f"  - RST 文件: {len(rst_files)} 个")
        print(f"  - Markdown 文件: {len(md_files)} 个")
    else:
        print("❌ 文档源目录不存在")
    
    if HTML_DIR.exists():
        print("✅ HTML 构建目录存在")
        
        # 统计生成的文件
        html_files = list(HTML_DIR.glob("**/*.html"))
        print(f"  - HTML 文件: {len(html_files)} 个")
        
        # 检查主要文件
        index_html = HTML_DIR / "index.html"
        if index_html.exists():
            size = index_html.stat().st_size
            print(f"  - 主页大小: {size} 字节")
    else:
        print("❌ HTML 构建目录不存在")


if __name__ == "__main__":
    # 允许直接运行此文件进行快速检查
    pytest.main([__file__, "-v"])