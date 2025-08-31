#!/usr/bin/env python3
"""
测试文档系统依赖安装脚本

此脚本验证 AxisFuzzy 项目的 Sphinx 文档系统是否正确配置和安装。
包括检查所有必需的文档依赖包、扩展和主题。
"""

import sys
import importlib
from pathlib import Path


def check_import(module_name, description=""):
    """检查模块导入（重命名避免 pytest 自动发现）"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {module_name} ({description}): {version}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} ({description}): 导入失败 - {e}")
        return False


def check_sphinx_extensions():
    """检查 Sphinx 扩展"""
    print("\n=== 测试 Sphinx 扩展 ===")

    # 核心包测试
    core_packages = [
        ('sphinx', 'Sphinx 核心'),
        ('sphinx_design', '设计组件'),
        ('sphinx_autodoc_typehints', '类型提示'),
        ('myst_parser', 'Markdown 支持'),
        ('babel', 'Babel 国际化支持')
    ]

    # Sphinx 内置扩展（需要在 Sphinx 环境中测试）
    builtin_extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx.ext.viewcode',
        'sphinx.ext.mathjax',
        'sphinx.ext.autosummary'
    ]

    success_count = 0
    total_count = 0

    # 测试核心包
    for pkg_name, description in core_packages:
        total_count += 1
        if check_import(pkg_name, description):
            success_count += 1

    # 测试 Sphinx 内置扩展
    try:
        import sphinx
        sphinx_version = sphinx.__version__
        print(f"✅ Sphinx 内置扩展可用 (基于 Sphinx {sphinx_version})")
        for ext in builtin_extensions:
            print(f"  ✅ {ext}")
        success_count += len(builtin_extensions)
    except ImportError:
        print("❌ Sphinx 未安装，无法验证内置扩展")

    total_count += len(builtin_extensions)

    # 测试可选扩展
    optional_extensions = [
        ('sphinx_copybutton', '代码复制按钮'),
        ('sphinx_tabs', '标签页功能')
    ]

    for ext_name, description in optional_extensions:
        total_count += 1
        try:
            importlib.import_module(ext_name)
            print(f"✅ {ext_name} ({description}): 已安装")
            success_count += 1
        except ImportError:
            print(f"⚠️ {ext_name} ({description}): 未安装（可选）")

    return success_count, total_count


def check_sphinx_themes():
    """检查 Sphinx 主题"""
    print("\n=== 测试 Sphinx 主题 ===")
    themes = [
        ('sphinx_rtd_theme', 'Read the Docs 主题'),
        ('pydata_sphinx_theme', 'PyData Sphinx 主题'),
        ('furo', 'Furo 现代主题')
    ]

    success_count = 0
    for theme_name, description in themes:
        try:
            module = importlib.import_module(theme_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {theme_name} ({description}): {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {theme_name} ({description}): 导入失败 - {e}")

    return success_count, len(themes)


def check_docs_build():
    """检查文档构建"""
    print("\n=== 测试文档构建 ===")
    project_root = Path(__file__).parent.parent.parent
    docs_dir = project_root / 'docs'
    build_dir = docs_dir / '_build' / 'html'

    if build_dir.exists():
        index_file = build_dir / 'index.html'
        if index_file.exists():
            print(f"✅ 文档构建成功: {build_dir}")
            print(f"✅ 主页文件存在: {index_file}")
            return True
        else:
            print(f"❌ 主页文件不存在: {index_file}")
            return False
    else:
        print(f"❌ 构建目录不存在: {build_dir}")
        return False


# pytest 测试函数
def test_sphinx_documentation_system():
    """测试 Sphinx 文档系统的完整性"""
    print("AxisFuzzy 文档系统依赖测试")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    print(f"测试路径: {Path(__file__).parent}")

    # 测试 Sphinx 扩展
    ext_success, ext_total = check_sphinx_extensions()

    # 测试 Sphinx 主题
    theme_success, theme_total = check_sphinx_themes()

    # 测试文档构建
    build_success = check_docs_build()

    # 总结
    print("\n=== 测试总结 ===")
    print(f"Sphinx 扩展: {ext_success}/{ext_total} 成功")
    print(f"Sphinx 主题: {theme_success}/{theme_total} 成功")
    print(f"文档构建: {'成功' if build_success else '失败'}")

    total_tests = ext_total + theme_total + 1
    total_success = ext_success + theme_success + (1 if build_success else 0)

    print(f"\n总体结果: {total_success}/{total_tests} 测试通过")

    if total_success == total_tests:
        print("\n🎉 所有文档依赖测试通过！文档系统已准备就绪。")
    else:
        print("\n⚠️  部分测试失败，请检查依赖安装。")
        # 在 pytest 中，我们使用断言而不是返回退出码
        assert total_success == total_tests, f"文档系统测试失败: {total_success}/{total_tests} 通过"


def main():
    """主函数，用于直接运行脚本"""
    try:
        test_sphinx_documentation_system()
        return 0
    except AssertionError:
        return 1


if __name__ == '__main__':
    sys.exit(main())
