#!/usr/bin/env python3
"""AxisFuzzy 测试套件命令行入口

这个模块允许通过 `python -m tests` 命令来运行测试套件。

使用示例:
    python -m tests --quick
    python -m tests --docs
    python -m tests --deps-only
    python -m tests --core-only
"""

import sys
import argparse
from . import (
    run_all_tests,
    run_core_tests,
    run_dependency_tests,
    run_docs_tests,
    run_analysis_tests,
    run_quick_tests
)


def main():
    """命令行主入口函数"""
    parser = argparse.ArgumentParser(
        description='AxisFuzzy 测试套件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m tests                       # 运行基本测试（依赖 + 核心）
  python -m tests --quick               # 运行快速测试（同上）
  python -m tests --docs                # 包含文档测试
  python -m tests --analysis            # 包含分析测试
  python -m tests --deps-only           # 仅运行依赖测试
  python -m tests --core-only           # 仅运行核心测试
  python -m tests --quiet               # 减少输出
  python -m tests --docs --analysis     # 运行完整测试套件
"""
    )
    
    # 测试类型选项
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        '--quick', 
        action='store_true', 
        help='运行快速测试（依赖 + 核心，默认行为）'
    )
    test_group.add_argument(
        '--deps-only', 
        action='store_true', 
        help='仅运行依赖验证测试'
    )
    test_group.add_argument(
        '--core-only', 
        action='store_true', 
        help='仅运行核心功能测试'
    )
    test_group.add_argument(
        '--docs-only', 
        action='store_true', 
        help='仅运行文档测试'
    )
    test_group.add_argument(
        '--analysis-only', 
        action='store_true', 
        help='仅运行分析模块测试'
    )
    
    # 包含选项（用于完整测试）
    parser.add_argument(
        '--docs', 
        action='store_true', 
        help='包含文档测试（与核心测试一起运行）'
    )
    parser.add_argument(
        '--analysis', 
        action='store_true', 
        help='包含分析测试（与核心测试一起运行）'
    )
    
    # 输出控制
    parser.add_argument(
        '--quiet', 
        action='store_true', 
        help='减少输出详细程度'
    )
    
    # 额外的 pytest 参数
    parser.add_argument(
        '--pytest-args',
        nargs='*',
        help='传递给 pytest 的额外参数'
    )
    
    args = parser.parse_args()
    
    # 设置详细程度
    verbose = not args.quiet
    
    # 准备额外参数
    extra_args = args.pytest_args if args.pytest_args else None
    
    # 执行相应的测试
    try:
        if args.deps_only:
            print("🔍 运行依赖验证测试...")
            success = run_dependency_tests(verbose=verbose, extra_args=extra_args)
            
        elif args.core_only:
            print("⚙️  运行核心功能测试...")
            success = run_core_tests(verbose=verbose, extra_args=extra_args)
            
        elif args.docs_only:
            print("📚 运行文档测试...")
            success = run_docs_tests(verbose=verbose, extra_args=extra_args)
            
        elif args.analysis_only:
            print("📈 运行分析模块测试...")
            success = run_analysis_tests(verbose=verbose, extra_args=extra_args)
            
        elif args.quick or (not args.docs and not args.analysis):
            # 默认行为：快速测试
            print("⚡ 运行快速测试套件...")
            success = run_quick_tests(verbose=verbose)
            
        else:
            # 完整测试套件
            print("🎯 运行完整测试套件...")
            success = run_all_tests(
                include_docs=args.docs,
                include_analysis=args.analysis,
                verbose=verbose,
                extra_args=extra_args
            )
            
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(130)  # 标准的 Ctrl+C 退出码
        
    except Exception as e:
        print(f"\n❌ 测试执行过程中发生错误: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # 根据测试结果设置退出码
    if success:
        print("\n🎉 测试执行成功！")
        sys.exit(0)
    else:
        print("\n❌ 测试执行失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()