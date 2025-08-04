#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/4 00:55
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
FuzzLab 运算功能测试脚本

本脚本旨在全面测试 FuzzLab 库中 Fuzznum 和 Fuzzarray 对象的核心运算功能。
它模拟了 `notebook.ops.test.ipynb` 中的测试用例，涵盖了算术运算、
比较运算在不同类型（Fuzznum, Fuzzarray, 标量, ndarray）之间的组合。

主要功能：
1. 初始化 FuzzLab 环境并准备测试数据。
2. 逐一执行加、减、乘、除、幂和比较运算。
3. 对每次运算进行异常处理：
   - 成功时，打印成功信息及结果。
   - 失败时，打印失败信息及原因，并继续执行后续测试。

如何运行：
- 将此文件保存在 FuzzLab 项目的根目录下。
- 在终端中直接运行 `python test_operations.py`。
"""
import numpy as np

# 假设此脚本与 fuzzlab 目录位于同一父目录下，或者 fuzzlab 已安装
from fuzzlab.core.fuzzarray import fuzzarray
from fuzzlab.core.fuzznums import Fuzznum
from fuzzlab.core.ops import get_operation_registry
from fuzzlab.modules.qrofs.op import register_qrofn_operations


def run_test(description: str, operation_lambda: callable):
    """
    执行单个运算测试，并优雅地打印结果或错误信息。

    通过将实际的运算逻辑包装在 lambda 函数中，此函数可以捕获任何可能
    发生的异常，从而防止单个失败的测试中断整个程序的执行。

    Args:
        description (str): 对当前测试的文字描述，例如 "Fuzznum + Fuzzarray"。
        operation_lambda (callable): 一个无参数的 lambda 函数，其内部执行实际的运算。
    """
    print(f"--- Testing: {description} ---")
    try:
        result = operation_lambda()
        # 对 Fuzzarray 和 ndarray 使用 repr() 可以获得更结构化的输出
        result_repr = repr(result)
        print(f"[SUCCESS] Result:\n{result_repr}\n")
    except Exception as e:
        # 捕获所有可能的异常，打印失败原因，然后继续
        print(f"[FAILURE] Reason: {type(e).__name__} - {e}\n")


def main():
    """
    主函数，用于设置测试数据并按顺序执行所有运算测试。
    """
    # --- 1. 初始化和数据准备 ---
    print("="*20 + " Initializing FuzzLab Environment " + "="*20)
    try:
        # 注册 q-ROFN 相关的运算实现
        register_qrofn_operations()
        # 获取全局运算注册表实例
        get_operation_registry()
        print("Environment initialized successfully.\n")
    except Exception as e:
        print(f"Fatal: Failed to initialize environment: {e}")
        return  # 如果环境初始化失败，则无法继续

    print("="*20 + " Preparing Test Data " + "="*20)
    # 创建 Fuzznum 实例
    f1 = Fuzznum('qrofn', 3).create(md=0.7, nmd=0.4)
    f2 = Fuzznum('qrofn', 3).create(md=0.5, nmd=0.7)
    print(f"f1: {f1}")
    print(f"f2: {f2}\n")

    # 创建 Fuzzarray 实例
    fy1_data = [[Fuzznum('qrofn', qrung=3).create(md=0.7, nmd=0.4), Fuzznum('qrofn', qrung=3).create(md=0.5, nmd=0.6), Fuzznum('qrofn', qrung=3).create(md=0.8, nmd=0.3)],
                [Fuzznum('qrofn', qrung=3).create(md=0.4, nmd=0.2), Fuzznum('qrofn', qrung=3).create(md=0.5, nmd=0.7), Fuzznum('qrofn', qrung=3).create(md=0.5, nmd=0.1)],
                [Fuzznum('qrofn', qrung=3).create(md=0.8, nmd=0.6), Fuzznum('qrofn', qrung=3).create(md=0.2, nmd=0.3), Fuzznum('qrofn', qrung=3).create(md=0.6, nmd=0.3)]]
    fy2_data = [[Fuzznum('qrofn', qrung=3).create(md=0.2, nmd=0.8), Fuzznum('qrofn', qrung=3).create(md=0.8, nmd=0.5), Fuzznum('qrofn', qrung=3).create(md=0.3, nmd=0.5)],
                [Fuzznum('qrofn', qrung=3).create(md=0.4, nmd=0.4), Fuzznum('qrofn', qrung=3).create(md=0.1, nmd=0.7), Fuzznum('qrofn', qrung=3).create(md=0.9, nmd=0.3)],
                [Fuzznum('qrofn', qrung=3).create(md=0.3, nmd=0.3), Fuzznum('qrofn', qrung=3).create(md=0.4, nmd=0.7), Fuzznum('qrofn', qrung=3).create(md=0.6, nmd=0.2)]]
    s1 = fuzzarray(fy1_data)
    s2 = fuzzarray(fy2_data)
    print(f"s1 (shape {s1.shape}):\n{s1}\n")
    print(f"s2 (shape {s2.shape}):\n{s2}\n")

    # 创建一个随机的 numpy 数组用于混合运算测试
    n = np.random.rand(3, 3)
    print(f"n (numpy array, shape {n.shape}):\n{n}\n")

    # --- 2. 执行运算测试 ---

    # 加法测试
    print("\n" + "="*25 + " Addition Tests " + "="*25)
    run_test("Fuzznum + Fuzznum", lambda: f1 + f2)
    run_test("Fuzznum + Fuzzarray", lambda: f1 + s1)
    run_test("Fuzzarray + Fuzznum", lambda: s2 + f2)
    run_test("Fuzzarray + Fuzzarray", lambda: s1 + s2)

    # 减法测试
    print("\n" + "="*25 + " Subtraction Tests " + "="*25)
    run_test("Fuzznum - Fuzznum", lambda: f1 - f2)
    run_test("Fuzznum - Fuzzarray", lambda: f1 - s1)
    run_test("Fuzzarray - Fuzznum", lambda: s2 - f2)
    run_test("Fuzzarray - Fuzzarray", lambda: s1 - s2)

    # 乘法测试
    print("\n" + "="*25 + " Multiplication Tests " + "="*25)
    run_test("Fuzznum * Fuzznum", lambda: f1 * f2)
    run_test("Fuzznum * float", lambda: f1 * 0.5)
    run_test("float * Fuzznum", lambda: 0.5 * f1)
    run_test("Fuzznum * ndarray", lambda: f1 * n)
    run_test("ndarray * Fuzznum", lambda: n * f1)
    run_test("Fuzznum * Fuzzarray", lambda: f1 * s1)
    run_test("Fuzzarray * Fuzznum", lambda: s1 * f1)
    run_test("Fuzzarray * float", lambda: s1 * 0.5)
    run_test("float * Fuzzarray", lambda: 0.5 * s1)
    run_test("Fuzzarray * ndarray", lambda: s1 * n)
    run_test("ndarray * Fuzzarray", lambda: n * s1)
    run_test("Fuzzarray * Fuzzarray", lambda: s1 * s2)

    # 除法测试
    print("\n" + "="*25 + " Division Tests " + "="*25)
    run_test("Fuzznum / Fuzznum", lambda: f1 / f2)
    run_test("Fuzznum / float", lambda: f1 / 2.0)
    run_test("Fuzznum / ndarray", lambda: f1 / n)
    run_test("Fuzznum / Fuzzarray", lambda: f1 / s1)
    run_test("Fuzzarray / Fuzznum", lambda: s1 / f1)
    run_test("Fuzzarray / float", lambda: s1 / 2.0)
    run_test("Fuzzarray / ndarray", lambda: s1 / n)
    run_test("Fuzzarray / Fuzzarray", lambda: s1 / s2)

    # 幂运算测试
    print("\n" + "="*25 + " Power Tests " + "="*25)
    run_test("Fuzznum ** float", lambda: f1 ** 2.0)
    run_test("Fuzznum ** ndarray", lambda: f1 ** n)
    run_test("Fuzzarray ** float", lambda: s1 ** 2.0)
    run_test("Fuzzarray ** ndarray", lambda: s1 ** n)

    # 比较运算测试
    print("\n" + "="*25 + " Comparison Tests " + "="*25)
    # Greater Than (>)
    run_test("Fuzznum > Fuzznum", lambda: f1 > f2)
    run_test("Fuzznum > Fuzzarray", lambda: f1 > s1)
    run_test("Fuzzarray > Fuzznum", lambda: s2 > f2)
    run_test("Fuzzarray > Fuzzarray", lambda: s2 > s1)
    # Less Than (<)
    run_test("Fuzznum < Fuzznum", lambda: f1 < f2)
    run_test("Fuzznum < Fuzzarray", lambda: f1 < s1)
    run_test("Fuzzarray < Fuzznum", lambda: s1 < f1)
    run_test("Fuzzarray < Fuzzarray", lambda: s1 < s2)
    # Equals (==)
    run_test("Fuzznum == Fuzznum", lambda: f1 == f2)
    run_test("Fuzznum == Fuzzarray", lambda: f1 == s1)
    run_test("Fuzzarray == Fuzznum", lambda: s1 == f1)
    run_test("Fuzzarray == Fuzzarray", lambda: s1 == s2)
    # Not Equals (!=)
    run_test("Fuzznum != Fuzznum", lambda: f1 != f2)
    run_test("Fuzznum != Fuzzarray", lambda: f1 != s1)
    run_test("Fuzzarray != Fuzznum", lambda: s1 != f1)
    run_test("Fuzzarray != Fuzzarray", lambda: s1 != s2)
    # Greater or Equals (>=)
    run_test("Fuzznum >= Fuzznum", lambda: f1 >= f2)
    run_test("Fuzznum >= Fuzzarray", lambda: f1 >= s1)
    run_test("Fuzzarray >= Fuzznum", lambda: s1 >= f1)
    run_test("Fuzzarray >= Fuzzarray", lambda: s1 >= s2)
    # Less or Equals (<=)
    run_test("Fuzznum <= Fuzznum", lambda: f1 <= f2)
    run_test("Fuzznum <= Fuzzarray", lambda: f1 <= s1)
    run_test("Fuzzarray <= Fuzznum", lambda: s1 <= f1)
    run_test("Fuzzarray <= Fuzzarray", lambda: s1 <= s2)


if __name__ == "__main__":
    main()
