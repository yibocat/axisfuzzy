#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 20:02
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
高级特性:条件注册和动态加载

条件注册和动态加载 (conditional_extension)

核心思想:
--------
有些功能可能依赖于特定的外部库（如 scipy、tensorflow 等），或者只在特定环境下才需要注册。
条件注册允许我们根据运行时条件来决定是否注册某个扩展。

实现方式:
--------
- 定义一个 conditional_extension 装饰器。
- 这个装饰器接收一个条件函数 (condition: Callable[[], bool]) 作为参数。
- 在装饰器内部，它会执行这个条件函数。
- 如果条件函数返回 True，则它会继续执行其包裹的 @extension 装饰器，从而注册功能。
- 如果条件函数返回 False，则它会跳过 @extension 装饰器的执行，该功能就不会被注册。

优点:
------
- 减少依赖: 避免强制用户安装不必要的依赖库。只有当用户需要使用依赖于这些库的功能时，才需要安装它们。
- 环境适应性: 可以根据操作系统、Python 版本、硬件特性等进行条件注册。
- 模块化加载: 允许按需加载功能，例如，只有当检测到 GPU 时才加载 GPU 优化的模糊计算函数。
"""
import sys
from typing import Callable

from .decorators import extension


def conditional_extension(condition: Callable[[], bool]):
    """条件注册装饰器"""
    def decorator(register_func):
        if condition():
            return register_func()
        return lambda func: func
    return decorator


# 使用示例
@conditional_extension(lambda: 'scipy' in sys.modules)
@extension('advanced_distance', mtype='qrofn')
def scipy_based_distance(fuzz1, fuzz2, **kwargs):
    """基于scipy的高级距离计算"""
    # import scipy.spatial.distance
    # 使用scipy实现
    pass
