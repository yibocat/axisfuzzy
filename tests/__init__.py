"""AxisFuzzy 测试套件主入口

本模块提供了 AxisFuzzy 项目的统一测试入口，支持：
- 编程式测试执行
- 分类测试运行（核心、依赖、文档等）
- 灵活的测试配置
- 详细的测试报告

使用示例：
    # 运行所有核心测试
    from tests import run_core_tests
    run_core_tests()
    
    # 运行依赖测试
    from tests import run_dependency_tests
    run_dependency_tests()
    
    # 运行所有测试
    from tests import run_all_tests
    run_all_tests()
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

# 获取测试目录路径
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent

# 测试模块映射
TEST_MODULES = {
    'core': [
        'test_config',
        'test_core', 
        'test_fuzzifier',
        'test_membership',
        'test_mixin',
        'test_random'
    ],
    'dependencies': ['test_dependencies'],
    'docs': ['test_docs'],
    'analysis': ['test_analysis']  # 为未来扩展预留
}

# 测试优先级定义
TEST_PRIORITIES = {
    'dependencies': 1,  # 最高优先级，环境验证
    'core': 2,         # 核心功能测试
    'docs': 3,         # 文档测试
    'analysis': 4      # 分析模块测试（未来）
}


def _run_pytest(test_paths: List[str], 
                verbose: bool = True,
                capture: str = 'no',
                extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """运行 pytest 并返回结果
    
    Args:
        test_paths: 测试路径列表
        verbose: 是否显示详细输出
        capture: 输出捕获模式 ('no', 'sys', 'fd')
        extra_args: 额外的 pytest 参数
    
    Returns:
        包含测试结果的字典
    """
    cmd = ['python', '-m', 'pytest']
    
    # 添加基本参数
    if verbose:
        cmd.append('-v')
    
    cmd.extend(['-s' if capture == 'no' else f'--capture={capture}'])
    
    # 添加测试路径
    cmd.extend(test_paths)
    
    # 添加额外参数
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n🚀 执行命令: {' '.join(cmd)}")
    print(f"📁 工作目录: {PROJECT_ROOT}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 在项目根目录下运行测试
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,  # 直接显示输出
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'duration': duration,
            'command': ' '.join(cmd)
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ 测试执行失败: {e}")
        return {
            'success': False,
            'returncode': -1,
            'duration': duration,
            'error': str(e),
            'command': ' '.join(cmd)
        }


def _print_test_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """打印测试摘要"""
    print("\n" + "=" * 60)
    print("📊 测试执行摘要")
    print("=" * 60)
    
    total_duration = 0
    success_count = 0
    
    for category, result in results.items():
        status = "✅ 通过" if result['success'] else "❌ 失败"
        duration = result['duration']
        total_duration += duration
        
        if result['success']:
            success_count += 1
            
        print(f"{category:15} | {status:8} | {duration:6.2f}s")
    
    print("-" * 60)
    print(f"总计: {success_count}/{len(results)} 通过 | 总耗时: {total_duration:.2f}s")
    
    if success_count == len(results):
        print("\n🎉 所有测试都通过了！")
    else:
        print(f"\n⚠️  有 {len(results) - success_count} 个测试类别失败")


def run_dependency_tests(verbose: bool = True, 
                        extra_args: Optional[List[str]] = None) -> bool:
    """运行依赖测试
    
    Args:
        verbose: 是否显示详细输出
        extra_args: 额外的 pytest 参数
        
    Returns:
        测试是否全部通过
    """
    print("\n🔍 开始运行依赖测试...")
    print("这些测试验证项目依赖是否正确安装和配置")
    
    test_paths = [str(TEST_DIR / module) for module in TEST_MODULES['dependencies']]
    result = _run_pytest(test_paths, verbose=verbose, extra_args=extra_args)
    
    if result['success']:
        print(f"\n✅ 依赖测试通过 (耗时: {result['duration']:.2f}s)")
    else:
        print(f"\n❌ 依赖测试失败 (耗时: {result['duration']:.2f}s)")
        print("请检查项目依赖是否正确安装")
    
    return result['success']


def run_core_tests(verbose: bool = True,
                  extra_args: Optional[List[str]] = None) -> bool:
    """运行核心功能测试
    
    Args:
        verbose: 是否显示详细输出
        extra_args: 额外的 pytest 参数
        
    Returns:
        测试是否全部通过
    """
    print("\n⚙️  开始运行核心功能测试...")
    print("这些测试验证 AxisFuzzy 的核心功能")
    
    test_paths = [str(TEST_DIR / module) for module in TEST_MODULES['core']]
    result = _run_pytest(test_paths, verbose=verbose, extra_args=extra_args)
    
    if result['success']:
        print(f"\n✅ 核心测试通过 (耗时: {result['duration']:.2f}s)")
    else:
        print(f"\n❌ 核心测试失败 (耗时: {result['duration']:.2f}s)")
    
    return result['success']


def run_docs_tests(verbose: bool = True,
                  extra_args: Optional[List[str]] = None) -> bool:
    """运行文档测试
    
    Args:
        verbose: 是否显示详细输出
        extra_args: 额外的 pytest 参数
        
    Returns:
        测试是否全部通过
    """
    print("\n📚 开始运行文档测试...")
    print("这些测试验证项目文档的完整性和正确性")
    
    test_paths = [str(TEST_DIR / module) for module in TEST_MODULES['docs']]
    result = _run_pytest(test_paths, verbose=verbose, extra_args=extra_args)
    
    if result['success']:
        print(f"\n✅ 文档测试通过 (耗时: {result['duration']:.2f}s)")
    else:
        print(f"\n❌ 文档测试失败 (耗时: {result['duration']:.2f}s)")
        print("请检查文档构建环境和文档内容")
    
    return result['success']


def run_analysis_tests(verbose: bool = True,
                      extra_args: Optional[List[str]] = None) -> bool:
    """运行分析模块测试（未来扩展）
    
    Args:
        verbose: 是否显示详细输出
        extra_args: 额外的 pytest 参数
        
    Returns:
        测试是否全部通过
    """
    print("\n📈 开始运行分析模块测试...")
    print("这些测试验证 AxisFuzzy 的分析功能")
    
    # 检查分析测试目录是否存在
    analysis_dir = TEST_DIR / 'test_analysis'
    if not analysis_dir.exists():
        print("⚠️  分析测试目录不存在，跳过分析测试")
        return True
    
    test_paths = [str(TEST_DIR / module) for module in TEST_MODULES['analysis']]
    result = _run_pytest(test_paths, verbose=verbose, extra_args=extra_args)
    
    if result['success']:
        print(f"\n✅ 分析测试通过 (耗时: {result['duration']:.2f}s)")
    else:
        print(f"\n❌ 分析测试失败 (耗时: {result['duration']:.2f}s)")
    
    return result['success']


def run_all_tests(include_docs: bool = False,
                 include_analysis: bool = False,
                 verbose: bool = True,
                 extra_args: Optional[List[str]] = None) -> bool:
    """运行所有测试
    
    Args:
        include_docs: 是否包含文档测试
        include_analysis: 是否包含分析测试
        verbose: 是否显示详细输出
        extra_args: 额外的 pytest 参数
        
    Returns:
        所有测试是否全部通过
    """
    print("\n🎯 开始运行完整测试套件...")
    
    results = {}
    
    # 按优先级运行测试
    test_categories = ['dependencies', 'core']
    
    if include_docs:
        test_categories.append('docs')
    
    if include_analysis:
        test_categories.append('analysis')
    
    # 依赖测试
    print("\n" + "=" * 60)
    print("第 1 步: 依赖验证")
    print("=" * 60)
    results['dependencies'] = {
        'success': run_dependency_tests(verbose=verbose, extra_args=extra_args),
        'duration': 0  # 这里简化处理，实际duration在函数内部计算
    }
    
    # 如果依赖测试失败，提前退出
    if not results['dependencies']['success']:
        print("\n❌ 依赖测试失败，停止后续测试")
        print("请先解决依赖问题后再运行测试")
        return False
    
    # 核心测试
    print("\n" + "=" * 60)
    print("第 2 步: 核心功能测试")
    print("=" * 60)
    results['core'] = {
        'success': run_core_tests(verbose=verbose, extra_args=extra_args),
        'duration': 0
    }
    
    # 文档测试（可选）
    if include_docs:
        print("\n" + "=" * 60)
        print("第 3 步: 文档测试")
        print("=" * 60)
        results['docs'] = {
            'success': run_docs_tests(verbose=verbose, extra_args=extra_args),
            'duration': 0
        }
    
    # 分析测试（可选）
    if include_analysis:
        print("\n" + "=" * 60)
        print("第 4 步: 分析模块测试")
        print("=" * 60)
        results['analysis'] = {
            'success': run_analysis_tests(verbose=verbose, extra_args=extra_args),
            'duration': 0
        }
    
    # 打印总结
    all_passed = all(result['success'] for result in results.values())
    
    print("\n" + "=" * 60)
    print("🏁 测试套件执行完成")
    print("=" * 60)
    
    for category, result in results.items():
        status = "✅ 通过" if result['success'] else "❌ 失败"
        print(f"{category:15} | {status}")
    
    if all_passed:
        print("\n🎉 恭喜！所有测试都通过了！")
        print("AxisFuzzy 项目状态良好 ✨")
    else:
        failed_count = sum(1 for result in results.values() if not result['success'])
        print(f"\n⚠️  有 {failed_count} 个测试类别失败")
        print("请检查失败的测试并修复相关问题")
    
    return all_passed


def run_quick_tests(verbose: bool = True) -> bool:
    """运行快速测试（依赖 + 核心，跳过文档和分析）
    
    Args:
        verbose: 是否显示详细输出
        
    Returns:
        测试是否全部通过
    """
    print("\n⚡ 开始运行快速测试套件...")
    print("包含：依赖测试 + 核心功能测试")
    
    return run_all_tests(
        include_docs=False,
        include_analysis=False,
        verbose=verbose
    )


# 导入测试模块（使其可被发现）
try:
    from . import test_config
    from . import test_core
    from . import test_dependencies
    from . import test_docs
    from . import test_fuzzifier
    from . import test_membership
    from . import test_mixin
    from . import test_random
except ImportError as e:
    print(f"警告: 无法导入某些测试模块: {e}")

# 定义公共接口
__all__ = [
    'run_all_tests',
    'run_core_tests',
    'run_dependency_tests',
    'run_docs_tests',
    'run_analysis_tests',
    'run_quick_tests',
    'TEST_MODULES',
    'TEST_PRIORITIES'
]

# 测试套件元信息
__version__ = '1.0.0'
__author__ = 'AxisFuzzy Development Team'
__description__ = 'AxisFuzzy 统一测试套件'


if __name__ == '__main__':
    # 命令行入口
    import argparse
    
    parser = argparse.ArgumentParser(description='AxisFuzzy 测试套件')
    parser.add_argument('--docs', action='store_true', help='包含文档测试')
    parser.add_argument('--analysis', action='store_true', help='包含分析测试')
    parser.add_argument('--quick', action='store_true', help='运行快速测试')
    parser.add_argument('--deps-only', action='store_true', help='仅运行依赖测试')
    parser.add_argument('--core-only', action='store_true', help='仅运行核心测试')
    parser.add_argument('--quiet', action='store_true', help='减少输出')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.deps_only:
        success = run_dependency_tests(verbose=verbose)
    elif args.core_only:
        success = run_core_tests(verbose=verbose)
    elif args.quick:
        success = run_quick_tests(verbose=verbose)
    else:
        success = run_all_tests(
            include_docs=args.docs,
            include_analysis=args.analysis,
            verbose=verbose
        )
    
    sys.exit(0 if success else 1)