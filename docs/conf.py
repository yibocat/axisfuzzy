import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # 让 Sphinx 能找到 axisfuzzy 包

# -- Project information -----------------------------------------------------
project = 'AxisFuzzy'
copyright = '2025, yibocat'
author = 'yibocat'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',            # 支持 Markdown
    'sphinx.ext.autodoc',     # 自动 API 文档
    'sphinx.ext.napoleon',    # 支持 Google/Numpy 风格 docstring
    'sphinx.ext.viewcode',    # 源码高亮
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# 支持 Markdown 文件
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# 主题配置
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/your-repo/axisfuzzy",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "show_nav_level": 2,
    "navigation_depth": 4,      # 增加导航深度以支持子目录
    "footer_items": ["copyright", "sphinx-version"],
    "show_source": False,       # 不显示 Show Source
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
}

extensions += [
    'sphinx.ext.autosummary',
]

# 在构建时自动生成 autosummary stub pages（第一次构建会生成 _autosummary）
autosummary_generate = True

# 全局 autodoc 默认选项（可按需调整）
autodoc_default_options = {
    'members': False,          # 默认不列出所有成员，按需用 :members: 指定
    'undoc-members': False,
    'inherited-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',  # 或 'groupwise'
}

# 代码高亮
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# 复制按钮配置
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


master_doc = 'index'
html_show_sourcelink = False  # 不显示“查看源码”链接

# 启用 MyST 扩展功能
# 这一行是关键，它开启了对 ":::" 指令语法的解析
# myst_enable_extensions = ["colon_fence"]
