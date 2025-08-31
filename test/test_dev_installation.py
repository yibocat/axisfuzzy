#!/usr/bin/env python3
"""
开发模式安装测试脚本

此脚本验证 AxisFuzzy 在开发模式下的完整功能：
1. 核心包轻量级导入
2. 分析模块可选导入
3. 依赖检查机制
4. 实际功能验证
"""

import sys
from pathlib import Path


def test_core_package():
    """测试核心包导入"""
    print("=== 测试 1: 核心包导入 ===")
    try:
        import axisfuzzy
        print("✅ axisfuzzy 核心包导入成功")
        print(f"✅ 安装路径: {axisfuzzy.__file__}")
        print(f"✅ 版本: {getattr(axisfuzzy, '__version__', '0.0.1')}")

        # 验证核心组件
        core_components = ['Fuzznum', 'Fuzzarray', 'Fuzzifier']
        available = [comp for comp in core_components if hasattr(axisfuzzy, comp)]
        print(f"✅ 核心组件: {available}")
        return True
    except Exception as e:
        print(f"❌ 核心包导入失败: {e}")
        return False


def test_analysis_module():
    """测试分析模块导入"""
    print("\n=== 测试 2: 分析模块导入 ===")
    try:
        from axisfuzzy import analysis
        print("✅ analysis 模块导入成功")

        # 检查分析模块组件
        if hasattr(analysis, 'app'):
            print("✅ analysis.app 可用")
        if hasattr(analysis, 'check_analysis_dependencies'):
            print("✅ 依赖检查功能可用")
        return True
    except Exception as e:
        print(f"❌ 分析模块导入失败: {e}")
        return False


def test_dependency_check():
    """测试依赖检查功能"""
    print("\n=== 测试 3: 依赖检查 ===")
    try:
        from axisfuzzy.analysis import check_analysis_dependencies
        status = check_analysis_dependencies()

        print("依赖状态:")
        for dep, info in status.items():
            status_icon = "✅" if info['installed'] else "❌"
            version = info.get('version', 'unknown')
            print(f"  {status_icon} {dep}: {version}")
            if info.get('error'):
                print(f"    错误: {info['error']}")

        all_installed = all(info['installed'] for info in status.values())
        return all_installed
    except Exception as e:
        print(f"❌ 依赖检查失败: {e}")
        return False


def test_functionality():
    """测试实际功能"""
    print("\n=== 测试 4: 实际功能验证 ===")
    try:
        # 测试 pandas
        import pandas as pd
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        print(f"✅ pandas {pd.__version__}: DataFrame 创建成功 {df.shape}")

        # 测试 matplotlib
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__}: 导入成功")

        # 测试 networkx
        import networkx as nx
        G = nx.Graph()
        G.add_edge(1, 2)
        print(f"✅ networkx {nx.__version__}: 图创建成功，节点数 {G.number_of_nodes()}")

        # 测试 pydot
        import pydot
        print(f"✅ pydot {pydot.__version__}: 导入成功")

        return True
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False


def test_development_workflow():
    """测试开发工作流"""
    print("\n=== 测试 5: 开发工作流验证 ===")
    try:
        # 验证是否为开发模式安装
        import axisfuzzy
        install_path = Path(axisfuzzy.__file__).parent
        current_dir = Path.cwd() / 'axisfuzzy'

        if install_path.samefile(current_dir):
            print("✅ 开发模式安装确认：代码修改将立即生效")
        else:
            print(f"⚠️ 可能不是开发模式安装")
            print(f"  安装路径: {install_path}")
            print(f"  当前路径: {current_dir}")

        # 测试模块重新加载
        import importlib
        importlib.reload(axisfuzzy)
        print("✅ 模块重新加载成功")

        return True
    except Exception as e:
        print(f"❌ 开发工作流测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("AxisFuzzy 开发模式安装测试")
    print("=" * 50)

    tests = [
        test_core_package,
        test_analysis_module,
        test_dependency_check,
        test_functionality,
        test_development_workflow
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试异常: {e}")
            results.append(False)

    print("\n=== 测试总结 ===")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有测试通过！开发环境配置成功！")
        print("\n现在您可以：")
        print("1. 修改代码后立即测试（无需重新安装）")
        print("2. 使用 import axisfuzzy 进行核心功能开发")
        print("3. 使用 from axisfuzzy import analysis 进行分析功能开发")
        print("4. 运行 pytest 进行单元测试")
        return 0
    else:
        print("❌ 部分测试失败，请检查安装配置")
        return 1


if __name__ == "__main__":
    sys.exit(main())
