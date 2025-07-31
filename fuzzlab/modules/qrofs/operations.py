#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/1 00:07
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import numpy as np
from typing import Dict, Any, Union, Optional

from fuzzlab.config import get_config
from fuzzlab.core.fuzzarray import Fuzzarray
from fuzzlab.core.fuzznums import Fuzznum
from fuzzlab.core.operations import OperationMixin, get_operation_registry
from fuzzlab.core.triangular import OperationTNorm


# --- QROFN 算术运算 ---

class QROFNAddition(OperationMixin):
    def get_operation_name(self) -> str:
        return "add"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_conorm(strategy1.md, strategy2.md)
        nmd = tnorm.t_norm(strategy1.nmd, strategy2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray addition must be a Fuzzarray or Fuzznum.")

        def add_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_add = np.vectorize(add_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_add(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_add(fuzzarray1.data, other)

        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNSubtraction(OperationMixin):
    def get_operation_name(self) -> str:
        return "sub"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_norm(
            strategy1.md, tnorm.f_inv_func(tnorm.f_func(strategy1.md) - tnorm.f_func(strategy2.md)))
        nmd = tnorm.t_conorm(
            strategy1.nmd, tnorm.g_inv_func(tnorm.g_func(strategy2.nmd) - tnorm.g_func(strategy1.nmd)))

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray subtraction must be a Fuzzarray or Fuzznum.")

        def sub_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_sub = np.vectorize(sub_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_sub(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_sub(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNMultiplication(OperationMixin):
    def get_operation_name(self) -> str:
        return "mul"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_norm(strategy1.md, strategy2.md)
        nmd = tnorm.t_conorm(strategy1.nmd, strategy2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray multiplication must be a Fuzzarray or Fuzznum.")

        def mul_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_mul = np.vectorize(mul_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_mul(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_mul(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNDivision(OperationMixin):
    def get_operation_name(self) -> str:
        return "div"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        config = get_config()
        if strategy2.md < config.DEFAULT_EPSILON:
            raise ValueError("Division by fuzzy number with zero membership degree is not allowed.")

        md = tnorm.t_norm(
            strategy1.md, tnorm.f_inv_func(tnorm.f_func(strategy1.md) / tnorm.f_func(strategy2.md)))
        nmd = tnorm.t_conorm(
            strategy1.nmd, tnorm.g_inv_func(tnorm.g_func(strategy2.nmd) / tnorm.g_func(strategy1.nmd)))

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray division must be a Fuzzarray or Fuzznum.")

        def div_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_div = np.vectorize(div_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_div(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_div(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNPower(OperationMixin):
    def get_operation_name(self) -> str:
        return "pow"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_unary_with_operand(self,
                                   strategy: Any,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm,
                                   **kwargs) -> Dict[str, Any]:

        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          operand: Optional[Union[int, float, np.ndarray]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(operand, (int, float, np.ndarray)):
            raise TypeError(f"Unsupported operand type for power operation: {type(operand)}")

        def pow_op(fuzznum: Fuzznum, op_val: Union[int, float]) -> Fuzznum:
            result_dict = self.execute_unary_with_operand(
                fuzznum.get_strategy_instance(),
                op_val,
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_pow = np.vectorize(pow_op, otypes=[object])
        # np.vectorize 会自动处理 fuzzarray1.data 和 operand 之间的广播
        result_data = vectorized_pow(fuzzarray1.data, operand)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNTimes(OperationMixin):
    def get_operation_name(self) -> str:
        return "tim"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_unary_with_operand(self,
                                   strategy: Any,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm,
                                   **kwargs) -> Dict[str, Any]:

        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          operand: Optional[Union[int, float, np.ndarray]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(operand, (int, float, np.ndarray)):
            raise TypeError(f"Unsupported operand type for times operation: {type(operand)}")

        def tim_op(fuzznum: Fuzznum, op_val: Union[int, float]) -> Fuzznum:
            result_dict = self.execute_unary_with_operand(
                fuzznum.get_strategy_instance(),
                op_val,
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_tim = np.vectorize(tim_op, otypes=[object])
        result_data = vectorized_tim(fuzzarray1.data, operand)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNExponential(OperationMixin):
    def get_operation_name(self) -> str:
        return "exp"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_unary_with_operand(self,
                                   strategy: Any,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm,
                                   **kwargs) -> Dict[str, Any]:

        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          operand: Optional[Union[int, float, np.ndarray]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':
        if not isinstance(operand, (int, float, np.ndarray)):
            raise TypeError(f"Unsupported operand type for exponential operation: {type(operand)}")

        def exp_op(fuzznum: Fuzznum, op_val: Union[int, float]) -> Fuzznum:
            result_dict = self.execute_unary_with_operand(
                fuzznum.get_strategy_instance(),
                op_val,
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_exp = np.vectorize(exp_op, otypes=[object])
        result_data = vectorized_exp(fuzzarray1.data, operand)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNLogarithmic(OperationMixin):
    def get_operation_name(self) -> str:
        return "log"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_unary_with_operand(self,
                                   strategy: Any,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm,
                                   **kwargs) -> Dict[str, Any]:
        config = get_config()
        if strategy.md < config.DEFAULT_EPSILON:
            # Log of zero is undefined, return a default invalid fuzzy number
            return {'md': 0., 'nmd': 1., 'q': strategy.q}

        md = tnorm.f_inv_func(tnorm.f_func(strategy.md) / operand)
        nmd = tnorm.g_inv_func(tnorm.g_func(strategy.nmd) / operand)

        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          operand: Optional[Union[int, float, np.ndarray]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(operand, (int, float, np.ndarray)):
            raise TypeError(f"Unsupported operand type for logarithmic operation: {type(operand)}")

        def log_op(fuzznum: Fuzznum, op_val: Union[int, float]) -> Fuzznum:
            result_dict = self.execute_unary_with_operand(
                fuzznum.get_strategy_instance(),
                op_val,
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_log = np.vectorize(log_op, otypes=[object])
        result_data = vectorized_log(fuzzarray1.data, operand)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


# --- QROFN 比较运算 ---

class QROFNGreaterThan(OperationMixin):
    def get_operation_name(self) -> str:
        return "gt"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_comparison(self,
                           strategy1: Any,
                           strategy2: Any,
                           tnorm: OperationTNorm,
                           **kwargs) -> bool:

        return strategy1.md > strategy2.md and strategy1.nmd < strategy2.nmd

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> np.ndarray:

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray comparison must be a Fuzzarray or Fuzznum.")

        def gt_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> bool:
            return self.execute_comparison(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

        vectorized_gt = np.vectorize(gt_op, otypes=[bool])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_gt(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_gt(fuzzarray1.data, other)
        return result_data


class QROFNLessThan(OperationMixin):
    def get_operation_name(self) -> str:
        return "lt"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_comparison(self,
                           strategy1: Any,
                           strategy2: Any,
                           tnorm: OperationTNorm,
                           **kwargs) -> bool:

        return strategy1.md < strategy2.md and strategy1.nmd > strategy2.nmd

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> np.ndarray:

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray comparison must be a Fuzzarray or Fuzznum.")

        def lt_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> bool:
            return self.execute_comparison(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

        vectorized_lt = np.vectorize(lt_op, otypes=[bool])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_lt(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_lt(fuzzarray1.data, other)
        return result_data


class QROFNEqual(OperationMixin):
    def get_operation_name(self) -> str:
        return "eq"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_comparison(self,
                           strategy1: Any,
                           strategy2: Any,
                           tnorm: OperationTNorm,
                           **kwargs) -> bool:

        config = get_config()
        return abs(strategy1.md - strategy2.md) < config.DEFAULT_EPSILON and \
            abs(strategy1.nmd - strategy2.nmd) < config.DEFAULT_EPSILON

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> np.ndarray:

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray comparison must be a Fuzzarray or Fuzznum.")

        def eq_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> bool:
            return self.execute_comparison(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

        vectorized_eq = np.vectorize(eq_op, otypes=[bool])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_eq(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_eq(fuzzarray1.data, other)
        return result_data


class QROFNGreaterEqual(OperationMixin):
    def get_operation_name(self) -> str:
        return "ge"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_comparison(self,
                           strategy1: Any,
                           strategy2: Any,
                           tnorm: OperationTNorm,
                           **kwargs) -> bool:

        config = get_config()
        return (strategy1.md > strategy2.md and strategy1.nmd < strategy2.nmd) or \
            (abs(strategy1.md - strategy2.md) < config.DEFAULT_EPSILON and
             abs(strategy1.nmd - strategy2.nmd) < config.DEFAULT_EPSILON)

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> np.ndarray:

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray comparison must be a Fuzzarray or Fuzznum.")

        def ge_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> bool:
            return self.execute_comparison(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

        vectorized_ge = np.vectorize(ge_op, otypes=[bool])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_ge(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_ge(fuzzarray1.data, other)
        return result_data


class QROFNLessEqual(OperationMixin):
    def get_operation_name(self) -> str:
        return "le"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_comparison(self,
                           strategy1: Any,
                           strategy2: Any,
                           tnorm: OperationTNorm,
                           **kwargs) -> bool:

        config = get_config()
        return (strategy1.md < strategy2.md and strategy1.nmd > strategy2.nmd) or \
            (abs(strategy1.md - strategy2.md) < config.DEFAULT_EPSILON and
             abs(strategy1.nmd - strategy2.nmd) < config.DEFAULT_EPSILON)

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> np.ndarray:

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray comparison must be a Fuzzarray or Fuzznum.")

        def le_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> bool:
            return self.execute_comparison(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

        vectorized_le = np.vectorize(le_op, otypes=[bool])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_le(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_le(fuzzarray1.data, other)
        return result_data


class QROFNNotEqual(OperationMixin):
    def get_operation_name(self) -> str:
        return "ne"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_comparison(self,
                           strategy1: Any,
                           strategy2: Any,
                           tnorm: OperationTNorm,
                           **kwargs) -> bool:

        config = get_config()
        return not (abs(strategy1.md - strategy2.md) < config.DEFAULT_EPSILON and
                    abs(strategy1.nmd - strategy2.nmd) < config.DEFAULT_EPSILON)

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> np.ndarray:

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray comparison must be a Fuzzarray or Fuzznum.")

        def ne_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> bool:
            return self.execute_comparison(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

        vectorized_ne = np.vectorize(ne_op, otypes=[bool])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_ne(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_ne(fuzzarray1.data, other)
        return result_data


# --- QROFN 集合运算 ---

class QROFNIntersection(OperationMixin):
    def get_operation_name(self) -> str:
        return "intersection"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_norm(strategy1.md, strategy2.md)
        nmd = tnorm.t_conorm(strategy1.nmd, strategy2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray intersection must be a Fuzzarray or Fuzznum.")

        def intersection_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_intersection = np.vectorize(intersection_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_intersection(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_intersection(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNUnion(OperationMixin):
    def get_operation_name(self) -> str:
        return "union"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_conorm(strategy1.md, strategy2.md)
        nmd = tnorm.t_norm(strategy1.nmd, strategy2.nmd)

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray union must be a Fuzzarray or Fuzznum.")

        def union_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_union = np.vectorize(union_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_union(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_union(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNComplement(OperationMixin):
    def get_operation_name(self) -> str:
        return "complement"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_unary(self,
                      strategy: Any,
                      tnorm: OperationTNorm,
                      **kwargs) -> Dict[str, Any]:

        md = strategy.nmd
        nmd = strategy.md

        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Any],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        def complement_op(fuzznum: Fuzznum) -> Fuzznum:
            result_dict = self.execute_unary(fuzznum.get_strategy_instance(), tnorm, **kwargs)
            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']
            return result_fuzznum

        vectorized_complement = np.vectorize(complement_op, otypes=[object])
        result_data = vectorized_complement(fuzzarray1.data)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNImplication(OperationMixin):
    def get_operation_name(self) -> str:
        return "implication"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_conorm(tnorm.f_inv_func(1 - tnorm.f_func(strategy1.md)), strategy2.md)
        nmd = tnorm.t_norm(strategy1.nmd, tnorm.g_inv_func(1 - tnorm.g_func(strategy2.nmd)))

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray implication must be a Fuzzarray or Fuzznum.")

        def implication_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_implication = np.vectorize(implication_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_implication(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_implication(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNEquivalence(OperationMixin):
    def get_operation_name(self) -> str:
        return "equivalence"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_norm(
            tnorm.t_conorm(tnorm.f_inv_func(1 - tnorm.f_func(strategy1.md)), strategy2.md),
            tnorm.t_conorm(tnorm.f_inv_func(1 - tnorm.f_func(strategy2.md)), strategy1.md)
        )
        nmd = tnorm.t_conorm(
            tnorm.t_norm(strategy1.nmd, tnorm.g_inv_func(1 - tnorm.g_func(strategy2.nmd))),
            tnorm.t_norm(strategy2.nmd, tnorm.g_inv_func(1 - tnorm.g_func(strategy1.nmd)))
        )
        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray equivalence must be a Fuzzarray or Fuzznum.")

        def equivalence_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']
            return result_fuzznum

        vectorized_equivalence = np.vectorize(equivalence_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_equivalence(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_equivalence(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNDifference(OperationMixin):
    def get_operation_name(self) -> str:
        return "difference"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_norm(strategy1.md, tnorm.f_inv_func(1 - tnorm.f_func(strategy2.md)))
        nmd = tnorm.t_conorm(strategy1.nmd, strategy2.md)

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray difference must be a Fuzzarray or Fuzznum.")

        def difference_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']
            return result_fuzznum

        vectorized_difference = np.vectorize(difference_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_difference(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_difference(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


class QROFNSymmetricDifference(OperationMixin):
    def get_operation_name(self) -> str:
        return "symdiff"

    def get_supported_mtypes(self) -> list[str]:
        return ["qrofn"]

    def execute_binary(self,
                       strategy1: Any,
                       strategy2: Any,
                       tnorm: OperationTNorm,
                       **kwargs) -> Dict[str, Any]:

        md = tnorm.t_conorm(
            tnorm.t_norm(strategy1.md, tnorm.f_inv_func(1 - tnorm.f_func(strategy2.md))),
            tnorm.t_norm(strategy2.md, tnorm.f_inv_func(1 - tnorm.f_func(strategy1.md)))
        )
        nmd = tnorm.t_norm(
            tnorm.t_conorm(strategy1.nmd, strategy2.md),
            tnorm.t_conorm(strategy2.nmd, strategy1.md)
        )

        return {'md': md, 'nmd': nmd, 'q': strategy1.q}

    def execute_fuzzarray(self,
                          fuzzarray1: 'Fuzzarray',
                          other: Optional[Union['Fuzzarray', Fuzznum]],
                          tnorm: OperationTNorm,
                          **kwargs) -> 'Fuzzarray':

        if not isinstance(other, (Fuzzarray, Fuzznum)):
            raise TypeError("Second operand for Fuzzarray symmetric difference must be a Fuzzarray or Fuzznum.")

        def symdiff_op(fuzznum1: Fuzznum, fuzznum2: Fuzznum) -> Fuzznum:
            result_dict = self.execute_binary(
                fuzznum1.get_strategy_instance(),
                fuzznum2.get_strategy_instance(),
                tnorm,
                **kwargs)

            result_fuzznum = Fuzznum(mtype='qrofn', qrung=result_dict['q'])
            result_fuzznum.md = result_dict['md']
            result_fuzznum.nmd = result_dict['nmd']

            return result_fuzznum

        vectorized_symdiff = np.vectorize(symdiff_op, otypes=[object])
        if isinstance(other, Fuzzarray):
            result_data = vectorized_symdiff(fuzzarray1.data, other.data)
        else:
            result_data = vectorized_symdiff(fuzzarray1.data, other)
        return Fuzzarray(result_data, mtype='qrofn', copy=False)


def register_qrofn_operations():
    """
    Register all QROFN-related operational methods to the global registry.
    """
    registry = get_operation_registry()
    registry.register_operation(QROFNAddition())
    registry.register_operation(QROFNSubtraction())
    registry.register_operation(QROFNMultiplication())
    registry.register_operation(QROFNDivision())
    registry.register_operation(QROFNPower())
    registry.register_operation(QROFNTimes())
    registry.register_operation(QROFNExponential())
    registry.register_operation(QROFNLogarithmic())
    registry.register_operation(QROFNGreaterThan())
    registry.register_operation(QROFNLessThan())
    registry.register_operation(QROFNEqual())
    registry.register_operation(QROFNGreaterEqual())
    registry.register_operation(QROFNLessEqual())
    registry.register_operation(QROFNNotEqual())
    registry.register_operation(QROFNIntersection())
    registry.register_operation(QROFNUnion())
    registry.register_operation(QROFNComplement())
    registry.register_operation(QROFNImplication())
    registry.register_operation(QROFNEquivalence())
    registry.register_operation(QROFNDifference())
    registry.register_operation(QROFNSymmetricDifference())
