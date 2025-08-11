#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 20:05
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
高级特性:性能监控

性能监控和缓存 (PerformanceMonitor, monitored_extension)

核心思想:
--------
了解各个扩展功能的运行效率对于优化和调试至关重要。性能监控允许我们收集每个函数调用的时间、成功率等数据。
虽然这里没有直接实现缓存，但性能监控是实现智能缓存策略（例如，基于调用频率和性能决定是否缓存）的基础。

实现方式:
--------
PerformanceMonitor 类:

- 维护一个 call_stats 字典，记录每个功能（按 func_name:mtype 区分）的调用次数、总耗时、平均耗时、成功调用次数和错误信息。
- 提供 monitor 方法，它是一个高阶函数，返回一个装饰器。

monitor 装饰器:

- 包裹实际的功能函数。
- 在函数执行前后记录时间戳，计算执行时间。
- 捕获函数执行结果（成功或失败），并更新 PerformanceMonitor 中的统计数据。

monitored_extension 装饰器:

- 这是一个复合装饰器，它首先使用 PerformanceMonitor.monitor 来包裹功能函数，然后再使用 @extension 装饰器将其注册。

优点:
------

- 性能洞察: 提供关于哪些功能是性能瓶颈的宝贵数据。
- 优化依据: 帮助开发者识别需要优化的代码区域。
- 问题诊断: 记录错误信息，有助于快速定位和解决问题。
- 智能缓存基础: 收集的调用频率和性能数据可以用于实现更智能的缓存策略（例如，频繁调用且计算耗时的结果可以被缓存）。

示例:
------
>>> @monitored_extension('distance', mtype='qrofn')
>>> def qrofn_euclidean_distance(fuzz1, fuzz2, **kwargs):
...    # ... 距离计算逻辑 ...
...    pass
```

"""
import functools
import time
from typing import Dict, Any

from .decorator import extension


class ExtensionPerformanceMonitor:
    """扩展性能监控器"""

    def __init__(self):
        self.call_stats: Dict[str, Dict[str, Any]] = {}

    def monitor(self, func_name: str, mtype: str):
        """监控装饰器"""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                success: bool = True
                error = None
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    self._record_call(func_name, mtype, end_time - start_time, success, error)

                return result

            return wrapper

        return decorator

    def _record_call(self, func_name: str, mtype: str, duration: float, success: bool, error: str):
        """记录调用统计"""
        key = f"{func_name}:{mtype}"
        if key not in self.call_stats:
            self.call_stats[key] = {
                'total_calls': 0,
                'successful_calls': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'errors': []
            }

        stats = self.call_stats[key]
        stats['total_calls'] += 1
        stats['total_time'] += duration
        stats['avg_time'] = stats['total_time'] / stats['total_calls']

        if success:
            stats['successful_calls'] += 1
        else:
            stats['errors'].append(error)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取统计信息"""
        return self.call_stats.copy()


# 全局性能监控器
_performance_monitor = ExtensionPerformanceMonitor()


def get_extension_performance_monitor() -> ExtensionPerformanceMonitor:
    return _performance_monitor


def monitored_extension(name: str, mtype: str = None, **kwargs):
    """带性能监控的扩展装饰器"""

    def decorator(func):
        # 先应用性能监控
        monitored_func = _performance_monitor.monitor(name, mtype or 'default')(func)
        # 再应用扩展注册
        return extension(name, mtype, **kwargs)(monitored_func)

    return decorator
