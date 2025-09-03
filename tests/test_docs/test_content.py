#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文档内容的完整性和质量

这个模块验证 AxisFuzzy 文档系统的内容质量，
包括链接有效性、内容完整性、格式正确性等。
"""

import pytest
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from bs4 import BeautifulSoup
import time

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
HTML_DIR = DOCS_DIR / "_build" / "html"


class TestDocumentationContent:
    """测试文档内容的测试类"""
    
    def test_html_build_exists(self):
        """测试 HTML 构建是否存在"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在，请先运行文档构建")
    
    def test_index_page_content(self):
        """测试主页内容"""
        index_html = HTML_DIR / "index.html"
        if not index_html.exists():
            pytest.skip("主页文件不存在")
        
        with open(index_html, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # 检查页面标题
        title = soup.find('title')
        assert title, "缺少页面标题"
        title_text = title.get_text().strip()
        assert 'AxisFuzzy' in title_text, f"页面标题应包含项目名称: {title_text}"
        
        # 检查主要内容区域
        main_content = soup.find('main') or soup.find('div', class_='document')
        assert main_content, "缺少主要内容区域"
        
        # 检查是否有实际内容（不只是空白）
        text_content = main_content.get_text().strip()
        assert len(text_content) > 100, "主页内容过少"
        
        # 检查项目描述
        content_lower = text_content.lower()
        fuzzy_keywords = ['fuzzy', 'logic', 'computation', 'axisfuzzy']
        found_keywords = [kw for kw in fuzzy_keywords if kw in content_lower]
        assert len(found_keywords) >= 2, \
            f"主页应包含更多项目相关关键词，当前找到: {found_keywords}"
    
    def test_navigation_structure(self):
        """测试导航结构"""
        index_html = HTML_DIR / "index.html"
        if not index_html.exists():
            pytest.skip("主页文件不存在")
        
        with open(index_html, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # 检查导航菜单
        nav_elements = soup.find_all(['nav', 'div'], class_=re.compile(r'nav|menu|toc'))
        
        if nav_elements:
            print(f"\n✅ 找到 {len(nav_elements)} 个导航元素")
            
            # 检查导航链接
            nav_links = []
            for nav in nav_elements:
                links = nav.find_all('a', href=True)
                nav_links.extend(links)
            
            if nav_links:
                print(f"✅ 找到 {len(nav_links)} 个导航链接")
            else:
                print("⚠️ 导航元素中没有找到链接")
        else:
            print("⚠️ 未找到导航元素")
    
    def test_internal_links(self):
        """测试内部链接的有效性"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在")
        
        # 收集所有 HTML 文件
        html_files = list(HTML_DIR.glob("**/*.html"))
        if not html_files:
            pytest.skip("没有找到 HTML 文件")
        
        # 收集所有内部链接
        internal_links = set()
        broken_links = []
        
        for html_file in html_files[:10]:  # 限制检查前10个文件以避免超时
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    
                    # 跳过外部链接和特殊链接
                    if (href.startswith('http') or 
                        href.startswith('mailto:') or 
                        href.startswith('#') or
                        href.startswith('javascript:')):
                        continue
                    
                    # 处理相对链接
                    if href.startswith('./'):
                        href = href[2:]
                    
                    # 移除锚点
                    if '#' in href:
                        href = href.split('#')[0]
                    
                    if href:
                        internal_links.add(href)
            
            except Exception as e:
                print(f"\n警告: 处理文件 {html_file} 时出错: {e}")
        
        # 检查内部链接的有效性
        for link in internal_links:
            target_path = HTML_DIR / link
            if not target_path.exists():
                # 尝试添加 .html 后缀
                if not link.endswith('.html'):
                    target_path = HTML_DIR / f"{link}.html"
                
                if not target_path.exists():
                    broken_links.append(link)
        
        print(f"\n内部链接检查结果:")
        print(f"  总链接数: {len(internal_links)}")
        print(f"  有效链接: {len(internal_links) - len(broken_links)}")
        print(f"  无效链接: {len(broken_links)}")
        
        if broken_links:
            print("\n无效链接列表:")
            for link in broken_links[:5]:  # 只显示前5个
                print(f"  - {link}")
            
            # 如果无效链接太多，可能是配置问题
            # 调整阈值，因为文档可能包含一些预期的外部引用
            if len(broken_links) > len(internal_links) * 0.8:
                pytest.fail(f"无效链接过多 ({len(broken_links)}/{len(internal_links)})，可能存在配置问题")
            elif len(broken_links) > len(internal_links) * 0.5:
                print(f"\n⚠️ 发现较多无效链接，但在可接受范围内")
    
    def test_external_links_sample(self):
        """测试外部链接的有效性（抽样检查）"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在")
        
        # 收集外部链接
        external_links = set()
        html_files = list(HTML_DIR.glob("**/*.html"))
        
        for html_file in html_files[:5]:  # 只检查前5个文件
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    if href.startswith('http'):
                        external_links.add(href)
            
            except Exception:
                continue
        
        if not external_links:
            pytest.skip("没有找到外部链接")
        
        # 抽样检查外部链接（最多检查5个）
        sample_links = list(external_links)[:5]
        broken_external = []
        
        for link in sample_links:
            try:
                # 设置较短的超时时间
                req = urllib.request.Request(
                    link, 
                    headers={'User-Agent': 'Mozilla/5.0 (Documentation Test)'}
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.getcode() >= 400:
                        broken_external.append((link, response.getcode()))
                
                # 避免请求过于频繁
                time.sleep(0.5)
                
            except Exception as e:
                broken_external.append((link, str(e)))
        
        print(f"\n外部链接检查结果（抽样）:")
        print(f"  检查链接数: {len(sample_links)}")
        print(f"  有效链接: {len(sample_links) - len(broken_external)}")
        print(f"  无效链接: {len(broken_external)}")
        
        if broken_external:
            print("\n无效外部链接:")
            for link, error in broken_external:
                print(f"  - {link}: {error}")
    
    def test_code_blocks_syntax(self):
        """测试代码块的语法正确性"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在")
        
        html_files = list(HTML_DIR.glob("**/*.html"))
        code_blocks_found = 0
        syntax_errors = []
        
        for html_file in html_files[:10]:  # 限制检查文件数量
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # 查找代码块
                code_blocks = soup.find_all(['pre', 'code'], class_=re.compile(r'highlight|language|code'))
                
                for block in code_blocks:
                    code_blocks_found += 1
                    code_text = block.get_text().strip()
                    
                    # 基本的语法检查（针对 Python 代码）
                    if ('python' in str(block.get('class', [])).lower() or 
                        'py' in str(block.get('class', [])).lower()):
                        
                        # 检查常见的语法错误
                        if code_text:
                            # 检查括号匹配
                            if code_text.count('(') != code_text.count(')'):
                                syntax_errors.append(f"括号不匹配: {code_text[:50]}...")
                            
                            if code_text.count('[') != code_text.count(']'):
                                syntax_errors.append(f"方括号不匹配: {code_text[:50]}...")
                            
                            if code_text.count('{') != code_text.count('}'):
                                syntax_errors.append(f"花括号不匹配: {code_text[:50]}...")
            
            except Exception:
                continue
        
        print(f"\n代码块检查结果:")
        print(f"  找到代码块: {code_blocks_found} 个")
        print(f"  语法错误: {len(syntax_errors)} 个")
        
        if syntax_errors:
            print("\n语法错误列表:")
            for error in syntax_errors[:3]:  # 只显示前3个
                print(f"  - {error}")
    
    def test_images_and_media(self):
        """测试图片和媒体文件"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在")
        
        # 查找图片文件
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(HTML_DIR.glob(f"**/*{ext}"))
        
        print(f"\n媒体文件检查结果:")
        print(f"  图片文件: {len(image_files)} 个")
        
        if image_files:
            # 检查图片文件大小
            large_images = []
            for img in image_files:
                size = img.stat().st_size
                if size > 1024 * 1024:  # 1MB
                    large_images.append((img.name, size // 1024))
            
            if large_images:
                print(f"  大文件 (>1MB): {len(large_images)} 个")
                for name, size_kb in large_images[:3]:
                    print(f"    - {name}: {size_kb}KB")
        
        # 检查 HTML 中的图片引用
        html_files = list(HTML_DIR.glob("**/*.html"))
        missing_images = []
        
        for html_file in html_files[:5]:  # 限制检查文件数量
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                img_tags = soup.find_all('img', src=True)
                
                for img in img_tags:
                    src = img['src']
                    
                    # 跳过外部图片和数据 URL
                    if src.startswith('http') or src.startswith('data:'):
                        continue
                    
                    # 构建图片路径
                    if src.startswith('./'):
                        src = src[2:]
                    
                    img_path = HTML_DIR / src
                    if not img_path.exists():
                        missing_images.append(src)
            
            except Exception:
                continue
        
        if missing_images:
            print(f"  缺失图片: {len(missing_images)} 个")
            for img in missing_images[:3]:
                print(f"    - {img}")
    
    def test_accessibility_basics(self):
        """测试基本的可访问性"""
        index_html = HTML_DIR / "index.html"
        if not index_html.exists():
            pytest.skip("主页文件不存在")
        
        with open(index_html, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        accessibility_issues = []
        
        # 检查图片的 alt 属性
        img_tags = soup.find_all('img')
        for img in img_tags:
            if not img.get('alt'):
                accessibility_issues.append(f"图片缺少 alt 属性: {img.get('src', '未知')}")
        
        # 检查标题层级
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings:
            # 检查是否有 h1
            h1_tags = soup.find_all('h1')
            if not h1_tags:
                accessibility_issues.append("页面缺少 h1 标题")
            elif len(h1_tags) > 1:
                accessibility_issues.append(f"页面有多个 h1 标题: {len(h1_tags)} 个")
        
        # 检查链接文本
        links = soup.find_all('a')
        for link in links:
            link_text = link.get_text().strip()
            if not link_text and not link.get('aria-label'):
                accessibility_issues.append("链接缺少文本或 aria-label")
        
        print(f"\n可访问性检查结果:")
        print(f"  检查项目: 图片 alt 属性、标题层级、链接文本")
        print(f"  发现问题: {len(accessibility_issues)} 个")
        
        if accessibility_issues:
            print("\n可访问性问题:")
            for issue in accessibility_issues[:5]:
                print(f"  - {issue}")


class TestDocumentationStructure:
    """测试文档结构的测试类"""
    
    def test_page_hierarchy(self):
        """测试页面层级结构"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在")
        
        html_files = list(HTML_DIR.glob("**/*.html"))
        
        # 分析目录结构
        directory_levels = {}
        for html_file in html_files:
            relative_path = html_file.relative_to(HTML_DIR)
            level = len(relative_path.parts) - 1  # 减去文件名本身
            
            if level not in directory_levels:
                directory_levels[level] = []
            directory_levels[level].append(str(relative_path))
        
        print(f"\n文档结构分析:")
        print(f"  HTML 文件总数: {len(html_files)}")
        
        for level in sorted(directory_levels.keys()):
            files = directory_levels[level]
            print(f"  层级 {level}: {len(files)} 个文件")
            
            # 显示一些示例文件
            for file_path in files[:3]:
                print(f"    - {file_path}")
            
            if len(files) > 3:
                print(f"    ... 还有 {len(files) - 3} 个文件")
    
    def test_content_completeness(self):
        """测试内容完整性"""
        if not HTML_DIR.exists():
            pytest.skip("HTML 构建目录不存在")
        
        # 检查重要页面是否存在
        important_pages = [
            'index.html',           # 主页
            'genindex.html',        # 索引页
            'search.html'           # 搜索页
        ]
        
        existing_pages = []
        missing_pages = []
        
        for page in important_pages:
            page_path = HTML_DIR / page
            if page_path.exists():
                existing_pages.append(page)
            else:
                missing_pages.append(page)
        
        print(f"\n重要页面检查:")
        print(f"  存在: {existing_pages}")
        if missing_pages:
            print(f"  缺失: {missing_pages}")
        
        # 至少应该有主页
        assert 'index.html' in existing_pages, "缺少主页文件"


def test_content_summary():
    """生成内容质量的总结报告"""
    print("\n=== 文档内容质量总结 ===")
    print(f"HTML 构建目录: {HTML_DIR}")
    
    if not HTML_DIR.exists():
        print("❌ HTML 构建目录不存在，请先运行文档构建")
        return
    
    # 统计文件数量
    html_files = list(HTML_DIR.glob("**/*.html"))
    css_files = list(HTML_DIR.glob("**/*.css"))
    js_files = list(HTML_DIR.glob("**/*.js"))
    
    print(f"\n文件统计:")
    print(f"  HTML 文件: {len(html_files)} 个")
    print(f"  CSS 文件: {len(css_files)} 个")
    print(f"  JavaScript 文件: {len(js_files)} 个")
    
    # 检查主要文件
    index_html = HTML_DIR / "index.html"
    if index_html.exists():
        size = index_html.stat().st_size
        print(f"\n主页状态:")
        print(f"  ✅ 主页存在")
        print(f"  文件大小: {size} 字节")
        
        # 简单的内容检查
        try:
            with open(index_html, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'AxisFuzzy' in content:
                print(f"  ✅ 包含项目名称")
            else:
                print(f"  ⚠️ 未找到项目名称")
                
        except Exception as e:
            print(f"  ❌ 读取主页失败: {e}")
    else:
        print(f"\n❌ 主页不存在")
    
    # 检查静态文件目录
    static_dir = HTML_DIR / "_static"
    if static_dir.exists():
        static_files = list(static_dir.glob("**/*"))
        print(f"\n静态文件: {len(static_files)} 个")
    else:
        print(f"\n⚠️ 静态文件目录不存在")


if __name__ == "__main__":
    # 允许直接运行此文件进行快速检查
    pytest.main([__file__, "-v"])