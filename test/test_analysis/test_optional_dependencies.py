#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 AxisFuzzy 可选依赖架构的示例脚本

这个脚本演示了如何使用新的可选依赖架构：
1. 核心包的轻量级导入
2. analysis 模块的按需导入
3. 依赖检查功能
"""


def test_core_import():
    """测试核心包导入"""
    print("=== 测试核心包导入 ===")
    try:
        import axisfuzzy
        print("✓ 核心包导入成功")
        print(f"✓ 可用的顶层API: {len(axisfuzzy.__all__)} 个")
        print(f"✓ 配置管理器: {type(axisfuzzy.get_config_manager())}")
        return True
    except Exception as e:
        print(f"✗ 核心包导入失败: {e}")
        return False


def test_analysis_import():
    """测试 analysis 模块导入"""
    print("\n=== 测试 analysis 模块导入 ===")
    try:
        from axisfuzzy import analysis
        print("✓ analysis 模块导入成功")

        # 检查依赖状态
        status = analysis.check_analysis_dependencies()
        print("\n依赖检查结果:")
        for dep_name, dep_info in status.items():
            if dep_info['installed']:
                print(f"  ✓ {dep_name}: v{dep_info['version']}")
            else:
                print(f"  ✗ {dep_name}: 未安装")

        return True
    except Exception as e:
        print(f"✗ analysis 模块导入失败: {e}")
        return False


def test_dependency_error_handling():
    """测试依赖错误处理机制"""
    print("\n=== 测试依赖错误处理 ===")
    try:
        # 测试依赖检查功能
        from axisfuzzy.analysis import check_analysis_dependencies
        deps = check_analysis_dependencies()

        # 检查是否包含所有预期的依赖
        expected_deps = ['pandas', 'matplotlib', 'networkx', 'pydot', 'graphviz']
        for dep in expected_deps:
            if dep in deps:
                status = "✓" if deps[dep]['installed'] else "✗"
                print(f"{status} {dep}: {deps[dep]['version'] or 'not installed'}")
            else:
                print(f"✗ 缺少依赖检查: {dep}")
                return False

        print("✓ 依赖检查功能正常工作")
        return True
    except Exception as e:
        print(f"✗ 依赖错误处理测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("AxisFuzzy 可选依赖架构测试")
    print("=" * 50)

    results = []
    results.append(test_core_import())
    results.append(test_analysis_import())
    results.append(test_dependency_error_handling())

    print("\n=== 测试总结 ===")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total} 项测试")

    if passed == total:
        print("\n🎉 所有测试通过！可选依赖架构工作正常。")
        print("\n使用说明:")
        print("  - 基础安装: pip install axisfuzzy")
        print("  - 分析功能: pip install 'axisfuzzy[analysis]'")
        print("  - 开发环境: pip install 'axisfuzzy[dev]'")
        print("  - 文档构建: pip install 'axisfuzzy[docs]'")
        print("  - 完整安装: pip install 'axisfuzzy[all]'")
        print("\n  注意: pydot需要系统安装Graphviz")
        print("  - macOS: brew install graphviz")
        print("  - Ubuntu: sudo apt-get install graphviz")
    else:
        print(f"\n⚠️  {total - passed} 项测试失败，需要进一步检查。")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
