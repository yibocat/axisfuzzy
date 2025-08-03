#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 19:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import warnings
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Optional

from fuzzlab.core.triangular import OperationTNorm


class OperationMixin(ABC):

    @abstractmethod
    def get_operation_name(self) -> str:
        pass

    @abstractmethod
    def get_supported_mtypes(self) -> List[str]:
        pass

    def supports(self, mtype: str) -> bool:
        return mtype in self.get_supported_mtypes()

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:
        raise NotImplementedError(f"Binary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:
        raise NotImplementedError(f"Unary operation with operand '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_unary_op_pure(self,
                              strategy: Any,
                              tnorm: OperationTNorm) -> Dict[str, Any]:
        raise NotImplementedError(f"Pure unary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        raise NotImplementedError(f"Comparison operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_fuzzarray_op(self,
                             fuzzarray_1: Any,
                             other: Optional[Any],
                             tnorm: OperationTNorm) -> Any:
        raise NotImplementedError(f"Fuzzarray operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _validate_operands(self, *operands: Any): ...

    def _postprocess_operands(self, operands: Any): ...


class OperationRegistry:

    def __init__(self):
        self._operations: Dict[str, Dict[str, OperationMixin]] = {}
        self._default_t_norm_type: str = 'algebraic'
        self._default_t_norm_params: Dict[str, Any] = {}

    def switch_t_norm(self, t_norm_type: str, **params: Any):
        self._default_t_norm_type = t_norm_type
        self._default_t_norm_params = params

    def get_default_t_norm_config(self) -> tuple[str, Dict[str, Any]]:
        return self._default_t_norm_type, self._default_t_norm_params

    # def __init__(self):
    #     self._operations: Dict[str, Dict[str, OperationMixin]] = {}
    #     self._default_t_norm: Optional[OperationTNorm] = OperationTNorm()
    #
    # def switch_t_norm(self, t_norm: str):
    #     self._default_t_norm = OperationTNorm(t_norm)
    #     self.update_all_operations()
    #
    # def set_t_norm_q_value(self, q_value: int):
    #     self._default_t_norm.q = q_value
    #     self.update_all_operations()
    #
    # def set_t_norm_params(self, params: Dict[str, Any]):
    #     self._default_t_norm.params = params
    #     self.update_all_operations()
    #
    # def set_t_norm(self,
    #                t_norm: Union[OperationTNorm, str],
    #                q_value: int,
    #                **params) -> None:
    #
    #     self._default_t_norm = t_norm if isinstance(t_norm, OperationTNorm) \
    #         else OperationTNorm(t_norm, q_value, **params)
    #     self.update_all_operations()
    #
    # def update_all_operations(self):
    #     for mtype_ops in self._operations.values():
    #         for op in mtype_ops.values():
    #             if op.t_norm is None:
    #                 op.set_t_norm(self._default_t_norm)

    def register(self, operation: OperationMixin) -> None:

        op_name = operation.get_operation_name()
        if op_name not in self._operations:
            self._operations[op_name] = {}

        for mtype in operation.get_supported_mtypes():
            if mtype in self._operations[op_name]:
                warnings.warn(f"Operation '{op_name}' for mtype '{mtype}' already registered.")
            self._operations[op_name][mtype] = operation

    def get_operation(self, op_name: str, mtype: str) -> Optional[OperationMixin]:
        """Get operation instance."""
        return self._operations.get(op_name, {}).get(mtype)

    def get_available_ops(self, mtype: str) -> List[str]:
        """Get list of available operations for mtype."""
        operations = []
        for op_name, mtype_ops in self._operations.items():
            if mtype in mtype_ops:
                operations.append(op_name)
        return operations


_operation_registry = OperationRegistry()


def get_operation_registry() -> OperationRegistry:
    """Get the global operation registry instance."""
    return _operation_registry
