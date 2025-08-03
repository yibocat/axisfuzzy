#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 21:15
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import List, Any, Dict, Optional, Union

from fuzzlab.config import get_config
from fuzzlab.core.ops import OperationMixin, get_operation_registry
from fuzzlab.core.triangular import OperationTNorm


# --- QROFN Arithmetic Operations ---

class QROFNAddition(OperationMixin):
    def get_operation_name(self) -> str:
        return 'add'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNSubtraction(OperationMixin):
    def get_operation_name(self) -> str:
        return 'sub'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_norm(
            strategy_1.md,
            tnorm.f_inv_func(tnorm.f_func(strategy_1.md) - tnorm.f_func(strategy_2.md)))
        nmd = tnorm.t_conorm(
            strategy_1.nmd,
            tnorm.g_inv_func(tnorm.g_func(strategy_2.nmd) - tnorm.g_func(strategy_1.nmd)))

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNMultiplication(OperationMixin):
    def get_operation_name(self) -> str:
        return 'mul'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_norm(strategy_1.md, strategy_2.md)
        nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNDivision(OperationMixin):
    def get_operation_name(self) -> str:
        return 'div'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_norm(
            strategy_1.md, tnorm.f_inv_func(tnorm.f_func(strategy_1.md) / tnorm.f_func(strategy_2.md)))
        nmd = tnorm.t_conorm(
            strategy_1.nmd, tnorm.g_inv_func(tnorm.g_func(strategy_2.nmd) / tnorm.g_func(strategy_1.nmd)))

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNPower(OperationMixin):
    def get_operation_name(self) -> str:
        return 'pow'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        return {'md': md, 'nmd': nmd, 'q': strategy.q}


class QROFNTimes(OperationMixin):
    def get_operation_name(self) -> str:
        return 'tim'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        return {'md': md, 'nmd': nmd, 'q': strategy.q}


class QROFNExponential(OperationMixin):
    def get_operation_name(self) -> str:
        return 'exp'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:

        # TODO: exp 计算目前还存在缺陷。此处写出来仅用于测试
        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        return {'md': md, 'nmd': nmd, 'q': strategy.q}


class QROFNLogarithmic(OperationMixin):
    def get_operation_name(self) -> str:
        return 'log'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:

        # TODO: exp 计算目前还存在缺陷。此处写出来仅用于测试
        md = tnorm.f_inv_func(tnorm.f_func(strategy.md) / operand)
        nmd = tnorm.g_inv_func(tnorm.g_func(strategy.nmd) / operand)

        return {'md': md, 'nmd': nmd, 'q': strategy.q}


# --- QROFN Comparison Operations ---

class QROFNGreaterThan(OperationMixin):
    def get_operation_name(self) -> str:
        return 'gt'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        return {'value': strategy_1.md > strategy_2.md and strategy_1.nmd < strategy_2.nmd}


class QROFNLessThan(OperationMixin):
    def get_operation_name(self) -> str:
        return 'lt'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        return {'value': strategy_1.md < strategy_2.md and strategy_1.nmd > strategy_2.nmd}


class QROFNEquals(OperationMixin):
    def get_operation_name(self) -> str:
        return 'eq'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        config = get_config()
        value = abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON and \
            abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON

        return {'value': value}


class QROFNGreaterEquals(OperationMixin):
    def get_operation_name(self) -> str:
        return 'ge'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        config = get_config()
        value = (strategy_1.md > strategy_2.md and strategy_1.nmd < strategy_2.nmd) or \
            (abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON and
             abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON)

        return {'value': value}


class QROFNLessEquals(OperationMixin):
    def get_operation_name(self) -> str:
        return 'le'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        config = get_config()
        value = (strategy_1.md < strategy_2.md and strategy_1.nmd > strategy_2.nmd) or \
            (abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON and
             abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON)

        return {'value': value}


class QROFNNotEquals(OperationMixin):
    def get_operation_name(self) -> str:
        return 'ne'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        config = get_config()
        value = not (abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON and
                     abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON)

        return {'value': value}


# --- QROFN Set Operations ---

class QROFNIntersection(OperationMixin):
    def get_operation_name(self) -> str:
        return 'intersection'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_norm(strategy_1.md, strategy_2.md)
        nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNUnion(OperationMixin):
    def get_operation_name(self) -> str:
        return 'union'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNComplement(OperationMixin):
    def get_operation_name(self) -> str:
        return 'complement'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_unary_op_pure(self,
                              strategy: Any,
                              tnorm: OperationTNorm) -> Dict[str, Any]:
        md = strategy.md
        nmd = strategy.nmd

        return {'md': md, 'nmd': nmd, 'q': strategy.q}


class QROFNImplication(OperationMixin):
    def get_operation_name(self) -> str:
        return 'implication'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_conorm(tnorm.f_inv_func(1 - tnorm.f_func(strategy_1.md)), strategy_2.md)
        nmd = tnorm.t_norm(strategy_1.nmd, tnorm.g_inv_func(1 - tnorm.g_func(strategy_2.nmd)))

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNEquivalence(OperationMixin):
    def get_operation_name(self) -> str:
        return 'equivalence'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_norm(
            tnorm.t_conorm(tnorm.f_inv_func(1 - tnorm.f_func(strategy_1.md)), strategy_2.md),
            tnorm.t_conorm(tnorm.f_inv_func(1 - tnorm.f_func(strategy_2.md)), strategy_1.md)
        )
        nmd = tnorm.t_conorm(
            tnorm.t_norm(strategy_1.nmd, tnorm.g_inv_func(1 - tnorm.g_func(strategy_2.nmd))),
            tnorm.t_norm(strategy_2.nmd, tnorm.g_inv_func(1 - tnorm.g_func(strategy_1.nmd)))
        )
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNDifference(OperationMixin):
    def get_operation_name(self) -> str:
        return 'difference'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_norm(strategy_1.md, tnorm.f_inv_func(1 - tnorm.f_func(strategy_2.md)))
        nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.md)

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


class QROFNSymmetricDifference(OperationMixin):
    def get_operation_name(self) -> str:
        return 'symdiff'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        md = tnorm.t_conorm(
            tnorm.t_norm(strategy_1.md, tnorm.f_inv_func(1 - tnorm.f_func(strategy_2.md))),
            tnorm.t_norm(strategy_2.md, tnorm.f_inv_func(1 - tnorm.f_func(strategy_1.md)))
        )
        nmd = tnorm.t_norm(
            tnorm.t_conorm(strategy_1.nmd, strategy_2.md),
            tnorm.t_conorm(strategy_2.nmd, strategy_1.md)
        )

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}


def register_qrofn_operations():
    """
    Register all QROFN-related operational methods to the global registry.
    """
    registry = get_operation_registry()
    registry.register(QROFNAddition())
    registry.register(QROFNSubtraction())
    registry.register(QROFNMultiplication())
    registry.register(QROFNDivision())
    registry.register(QROFNPower())
    registry.register(QROFNTimes())
    registry.register(QROFNExponential())
    registry.register(QROFNLogarithmic())
    registry.register(QROFNGreaterThan())
    registry.register(QROFNLessThan())
    registry.register(QROFNEquals())
    registry.register(QROFNGreaterEquals())
    registry.register(QROFNLessEquals())
    registry.register(QROFNNotEquals())
    registry.register(QROFNIntersection())
    registry.register(QROFNUnion())
    registry.register(QROFNComplement())
    registry.register(QROFNImplication())
    registry.register(QROFNEquivalence())
    registry.register(QROFNDifference())
    registry.register(QROFNSymmetricDifference())
