#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 21:05
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
高级特性: 配置与验证


核心思想:
-------
确保注册的功能符合预期的规范和约定，提高系统的健壮性和可靠性。这对于防止因不正确的函数签名或缺失的实现而导致的运行时错误非常重要。

实现方式:
-------

ExtensionValidator 类:

- validate_function_signature: 检查注册函数的签名是否符合要求（例如，第一个参数是否是 self 或 obj，参数数量是否正确）。
  这可以使用 Python 的 inspect 模块来实现。
- validate_all_registrations: 遍历所有注册的功能，对每个功能及其不同 mtype 的实现进行签名验证。
- check_coverage: 检查某个功能是否为所有“必需”的 mtype 都提供了实现（或者至少有默认实现）。
  这对于确保核心功能在所有支持的模糊数类型上都可用非常有用。

优点:
------
- 早期错误检测: 在运行时之前发现潜在的问题，避免运行时崩溃。
- 提高代码质量: 强制开发者遵循一致的接口约定。
- 增强可靠性: 确保系统在各种情况下都能正常工作。
- 开发辅助: 为开发者提供清晰的错误信息，指导他们修正不符合规范的实现。

示例:
------
在 apply_extensions() 内部，会调用 validator.validate_all_registrations()。
如果发现任何问题，会发出警告，提醒开发者修正。
"""
from typing import Dict, List

from .registry import get_extension_registry


class ExtensionValidator:

    def __init__(self):
        self.registry = get_extension_registry()

    def validate_function_signature(self, func_name: str, mtype: str) -> bool:
        """验证函数签名"""
        implementation = self.registry.get_function(func_name, mtype)
        if implementation is None:
            return False

        # 这里可以添加更复杂的签名验证逻辑
        import inspect
        sig = inspect.signature(implementation)

        # 基本验证:第一个参数应该是 self 或者 obj
        params = list(sig.parameters.keys())
        if not params or params[0] not in ['self', 'obj']:
            return False

        return True

    def validate_all_registrations(self) -> Dict[str, List[str]]:
        """验证所有注册"""
        issues = {}
        functions = self.registry.list_functions()

        for func_name, func_info in functions.items():
            func_issues = []

            # 验证特化实现
            for mtype in func_info['implementations']:
                if not self.validate_function_signature(func_name, mtype):
                    func_issues.append(f"Invalid signature for {mtype} implementation")

            # 验证默认实现
            if func_info['default']:
                if not self.validate_function_signature(func_name, 'default'):
                    func_issues.append("Invalid signature for default implementation")

            if func_issues:
                issues[func_name] = func_issues

        return issues

    def check_coverage(self, required_mtypes: List[str]) -> Dict[str, List[str]]:
        """检查功能覆盖率"""
        coverage_issues = {}
        functions = self.registry.list_functions()

        for func_name, func_info in functions.items():
            missing_mtypes = []

            for mtype in required_mtypes:
                if (mtype not in func_info['implementations'] and
                        not func_info['default']):
                    missing_mtypes.append(mtype)

            if missing_mtypes:
                coverage_issues[func_name] = missing_mtypes

        return coverage_issues


# 全局验证器
_validator = ExtensionValidator()


def get_extension_validator() -> ExtensionValidator:
    """获取验证器"""
    return _validator
