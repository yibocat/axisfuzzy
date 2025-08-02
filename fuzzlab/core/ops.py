#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 19:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Optional

from fuzzlab.core.triangular import OperationTNorm


class OperationMixin(ABC):

    _t_norm: OperationTNorm = OperationTNorm()

    @property
    def t_norm(self):
        return self._t_norm

    def set_t_norm(self, norm: Union[str, OperationTNorm], **t_norm_params):
        if isinstance(norm, str):
            self._t_norm = OperationTNorm(norm, **t_norm_params)
        elif isinstance(norm, OperationTNorm):
            self._t_norm = norm
        else:
            raise TypeError("t-norm must be 'str' or 'OperationTNorm'.")

    def get_t_norm(self) -> OperationTNorm:
        return self._t_norm

    @abstractmethod
    def get_operation_name(self) -> str:
        pass

    @abstractmethod
    def get_supported_mtypes(self) -> List[str]:
        pass

    def supports(self, mtype: str) -> bool:
        return mtype in self.get_supported_mtypes()

    def execute_binary_op(self, strategy_1: Any, strategy_2: Any) -> Dict[str, Any]:
        raise NotImplementedError(f"Binary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_unary_op_operand(self, strategy: Any, operand: Union[int, float]) -> Dict[str, Any]:
        raise NotImplementedError(f"Unary operation with operand '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_unary_op_pure(self, strategy: Any) -> Dict[str, Any]:
        raise NotImplementedError(f"Pure unary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_comparison_op(self, strategy_1: Any, strategy_2: Any) -> Dict[str, bool]:
        raise NotImplementedError(f"Comparison operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_fuzzarray_op(self, fuzzarray_1: Any, other: Optional[Any]) -> Any:
        raise NotImplementedError(f"Fuzzarray operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")


class OperationRegistry:

    def __init__(self):
        self._operations: Dict[str, Dict[str, OperationMixin]] = {}
        self._default_t_norm: Optional[OperationTNorm] = None

    def set_default_t_norm(self, t_norm: Union[OperationTNorm, str], **params) -> None:
        self._default_t_norm = t_norm if isinstance(t_norm, OperationTNorm) \
            else OperationTNorm(t_norm, **params)

        # Update all existing operations
        for mtype_ops in self._operations.values():
            for op in mtype_ops.values():
                if op.t_norm is None:
                    op.set_t_norm(t_norm, **params)

    def register(self, operation: OperationMixin) -> None:

        op_name = operation.get_operation_name()
        if op_name not in self._operations:
            self._operations[op_name] = {}

        for mtype in operation.get_supported_mtypes():
            if mtype in self._operations[op_name]:
                raise ValueError(f"Operation '{op_name}' for mtype '{mtype}' already registered.")
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
