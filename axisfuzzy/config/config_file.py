#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 21:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from dataclasses import dataclass, field


@dataclass
class Config:

    """
    模糊计算统一配置类。

    该类定义了 MohuPy 框架中所有可配置的参数及其默认值。
    每个字段都包含 `metadata`，用于指定其所属的类别和验证规则。
    """

    # ================== 基础配置 ===================
    DEFAULT_MTYPE: str = field(
        default='qrofn',
        metadata={
            'category': 'basic',
            'description': 'Default fuzzy number type, affects the type selection '
                           'when Fuzznum is constructed without parameters',
            'validator': lambda x: isinstance(x, str) and len(x) > 0,
            'error_msg': "Must be a non-empty string."
        }
    )

    DEFAULT_PRECISION: int = field(
        default=4,
        metadata={
            'category': 'basic',
            'description': 'Default calculation precision (number of decimal places), '
                           'affects all numeric calculations and display',
            'validator': lambda x: isinstance(x, int) and x >= 0,
            'error_msg': "Must be a non-negative integer."
        }
    )
    #
    DEFAULT_EPSILON: float = field(
        default=1e-12,
        metadata={
            'category': 'basic',
            'description': 'Default numerical tolerance, used for floating-point '
                           'number comparison and zero value judgment',
            'validator': lambda x: isinstance(x, (int, float)) and x > 0,
            'error_msg': "Must be a positive number."
        }
    )

    CACHE_SIZE: int = field(
        default=256,
        metadata={
            'category': 'performance',
            'description': 'Maximum number of entries in the operation cache, '
                           'which controls memory usage',
            'validator': lambda x: isinstance(x, int) and x >= 0,
            'error_msg': "Must be a non-negative integer."
        }
    )

    TNORM_VERIFY: bool = field(
        default=False,
        metadata={
            'category': 'debug',
            'description': 'T-Norm verification switch, used to verify the mathematical '
                           'properties after T-Norm initialization. It is generally set '
                           'to default off (False) to improve the computational efficiency '
                           'of fuzzy numbers.',
            'validator': lambda x: isinstance(x, bool),
            'error_msg': "Must be a boolean value (True/False)."
        }
    )

    # DEFAULT_T_NORM: str = field(
    #     default='algebraic',
    #     metadata={
    #         'category': 'basic',
    #         'description': '默认t-范数类型，影响 Fuzznum 的运算规则',
    #         'validator': lambda x: isinstance(x, str) and len(x) > 0,
    #         'error_msg': "必须是非空字符串。"
    #     }
    # )
    #

    # STRICT_ATTRIBUTE_MODE: bool = field(
    #     default=True,
    #     metadata={
    #         'category': 'basic',
    #         'description': '严格属性检查，主要用于 FuzznumStrategy 中 __setattr__ 方法的属性检查',
    #         'validator': lambda x: isinstance(x, bool),
    #         'error_msg': "必须是布尔值 (True/False)。"
    #     }
    # )

    # ENABLE_CACHE: bool = field(
    #     default=True,
    #     metadata={
    #         'category': 'performance',
    #         'description': '是否启用运算缓存，影响工厂类和执行器等计算和创建实例的缓存行为',
    #         'validator': lambda x: isinstance(x, bool),
    #         'error_msg': "必须是布尔值 (True/False)。"
    #     }
    # )

    # ENABLE_FUZZNUM_CACHE: bool = field(
    #     default=True,
    #     metadata={
    #         'category': 'performance',
    #         'description': '是否启用模糊数缓存，影响模糊数实例的缓存行为',
    #         'validator': lambda x: isinstance(x, bool),
    #         'error_msg': "必须是布尔值 (True/False)。"
    #     }
    # )

    # ENABLE_PERFORMANCE_MONITORING: bool = field(
    #     default=True,
    #     metadata={
    #         'category': 'debug',
    #         'description': '启动性能监控，用于调制监控一些计算的性能信息',
    #         'validator': lambda x: isinstance(x, bool),
    #         'error_msg': "必须是布尔值 (True/False)。"
    #     }
    # )

    # ENABLE_LOGGING: bool = field(
    #     default=True,
    #     metadata={
    #         'category': 'debug',
    #         'description': '启动日志记录',
    #         'validator': lambda x: isinstance(x, bool),
    #         'error_msg': "必须是布尔值 (True/False)。"
    #     }
    # )

    # DEBUG_MODE: bool = field(
    #     default=True,
    #     metadata={
    #         'category': 'debug',
    #         'description': '调试模式开关，启用详细的调试信息',
    #         'validator': lambda x: isinstance(x, bool),
    #         'error_msg': "必须是布尔值 (True/False)。"
    #     }
    # )

