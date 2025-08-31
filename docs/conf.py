# Standard Library Imports
import os
import sys
import tomllib  # 用于解析 pyproject.toml 文件，获取包的版本信息

# -- 路径设置 --------------------------------------------------------------
# 此部分允许 Sphinx 找到 'axisfuzzy' 包。
# 它将 'docs' 目录（conf.py 通常所在位置）的父目录添加到 Python 路径中，
# 假设包的根目录在上一级。
sys.path.insert(0, os.path.abspath('..'))


# -- 项目信息 --------------------------------------------------------------
# 定义项目的基础元数据，如项目名称、版权和作者。
project = 'AxisFuzzy'  # 更改为首字母大写
copyright = '2025, yibocat'
author = 'yibocat'


# -- 版本定义 --------------------------------------------------------------
# 定义项目用于文档的版本号。
# 'release' 是完整的版本字符串（例如：'1.2.3b1'）。
# 'version' 是简化的主版本.次版本（例如：'1.2'）。
# 这段逻辑尝试从包本身导入版本，如果失败，则回退到从 pyproject.toml 读取。
try:
    # 尝试直接从包中导入版本。
    # 如果包已安装或在 Python 路径中，这是首选方法。
    from axisfuzzy.version import __version__ as release
except ImportError:
    # 如果直接导入失败（例如，在初始文档构建期间或开发环境中），
    # 尝试从 pyproject.toml 文件中读取版本。
    pyproject_path = os.path.join(os.path.abspath('..'), 'pyproject.toml')
    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
            release = data['project']['version']
    except Exception:
        # 如果所有其他方法都失败，则回退到默认版本。
        release = '0.0.0'

# 'version' 变量只包含主版本和次版本号（例如 '1.2'）。
version = '.'.join(release.split('.')[:2])
# 'release' 变量包含完整的版本号，包括预发布标签（例如 '1.2.3b1'）。
# 通常用于文档页脚等位置。


# -- 通用配置 --------------------------------------------------------------
# 核心 Sphinx 设置，适用于整个文档构建过程。

# 添加任何 Sphinx 扩展模块的名称，作为字符串。
# 这些扩展启用了额外的功能，例如 Markdown 支持、API 文档、更好的 docstring 解析等。
extensions = [
    'myst_parser',            # 支持 reStructuredText 和 Markdown (.md 文件)
    'sphinx.ext.autodoc',     # 从 docstring 自动生成文档
    'sphinx.ext.napoleon',    # 支持 Google 和 NumPy 风格的 docstring，用于 autodoc
    'sphinx.ext.viewcode',    # 为文档页面添加"查看源码"链接
    'sphinx_copybutton',      # 为代码块添加"复制"按钮
    'sphinx_design',          # 提供设计组件（卡片、网格等）
    'sphinx_autodoc_typehints', # 在 autodoc 输出中漂亮地渲染 Python 3 类型提示
    'sphinx.ext.mathjax',     # 使用 MathJax 渲染数学公式
    'sphinx.ext.autosummary', # 为 API 文档生成摘要表格
    'sphinx_tabs.tabs',       # 添加标签页功能
    'pydata_sphinx_theme',    # PyData Sphinx 主题扩展
]

# 包含模板的路径，相对于此目录。
templates_path = ['_templates']

# 相对于源目录的模式列表，用于匹配在查找源文件时要忽略的文件和目录。
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# 源文件名的后缀。
# 允许 Sphinx 处理 reStructuredText (.rst) 和 Markdown (.md) 文件。
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# 根文档的名称（例如，'index.rst' 或 'index.md'）。
master_doc = 'index'


# -- 扩展特定配置 ----------------------------------------------------------
# 各个 Sphinx 扩展的详细设置。

# MyST Parser 配置:
# 启用特定的 MyST 扩展（例如，用于冒号围栏，允许 Markdown 中的嵌套指令）。
# 如果您需要在 Markdown 中使用 ::: 来创建 admonition 或其他块级指令，请取消注释。
# myst_enable_extensions = ["colon_fence"]

# Autodoc 配置:
# 'autodoc' 扩展的默认选项。
# 这些选项可以通过单个指令（例如，使用 :members:）进行覆盖。
autodoc_default_options = {
    'members': False,           # 默认不自动显示所有成员
    'undoc-members': False,     # 不显示没有 docstring 的成员
    'inherited-members': True,  # 文档化从基类继承的成员
    'show-inheritance': True,   # 显示基类列表
    'member-order': 'alphabetical', # 成员排序方式：按字母顺序
                                # 可选值有 'bysource'（按源代码顺序）、'alphabetical'（按字母顺序）、'groupwise'（按类型分组）
}

autodoc_typehints = "signature"  # 将类型提示放在签名中，而不是描述中
autodoc_typehints_format = "short"  # 简化类型提示，避免完整路径
python_use_unqualified_type_names = True # 解决由 .pyi 和动态注入引起的重复警告

# Autosummary 配置:
# 在构建过程中自动为 autosummary 指令生成存根页面。
autosummary_generate = True

# Pygments (代码高亮) 配置:
# 用于源代码 Pygments 高亮的样式名称。
pygments_style = "sphinx"
pygments_dark_style = "monokai"     # 暗色模式下的样式

# Sphinx Copybutton 配置:
# 用于从代码块复制时从提示符中移除文本的正则表达式。
# 这可以防止将 '>>>' 或 '$' 等提示符与代码一起复制。
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


# -- HTML 输出配置 ---------------------------------------------------------
# 与文档的 HTML 输出相关的设置。

# 用于 HTML 和 HTML Help 页面的主题。
# 其他主题选项（例如 "furo", "pydata_sphinx_theme", "shibuya"）可以在依赖稳定后尝试
# 使用 pydata_sphinx_theme 主题，提供现代化的文档界面
html_theme = 'pydata_sphinx_theme'

# 包含自定义静态文件（例如 CSS、JavaScript）的路径。
html_static_path = ['_static']

# 主题特定选项。
# 这些选项特定于 'pydata_sphinx_theme'。
html_theme_options = {
    "logo": {
        "text": "AxisFuzzy",  # 添加 logo 文本
        "image_light": "_static/logo-light.png",  # 可选：浅色模式 logo
        "image_dark": "_static/logo-dark.png",   # 可选：深色模式 logo
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/your-repo/axisfuzzy", # 重要：请更新此 URL 为您的实际 GitHub 仓库地址！
            "icon": "fab fa-github-square",
            "type": "fontawesome", # 指定使用 Font Awesome 图标
        },
    ],
    "navbar_end": ["version-switcher", "navbar-icon-links", "theme-switcher"],
    "show_nav_level": 1,            # 减少默认展开级别，使导航更简洁
    "navigation_depth": 3,          # 减少导航深度
    # "footer_items": ["copyright"],  # 简化页脚，移除 sphinx-version
    "primary_sidebar_end": [],      # 清空主侧边栏底部
    "show_toc_level": 2,            # 减少目录层级
    "pygments_light_style": "github-light",
    "pygments_dark_style": "github-dark",
}

# 不在页面底部显示"查看源码"链接。
# 这通常更适用于干净的公共文档，避免用户看到原始rst文件。
html_show_sourcelink = False

# 自定义页面标题
html_title = f"{project} Documentation"

# 添加自定义 CSS
html_css_files = [
    'custom.css',
]
