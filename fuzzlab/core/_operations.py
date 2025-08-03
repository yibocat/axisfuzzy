#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/31 22:07
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional


class OperationMixin(ABC):
    """
    Abstract base class for operations.

    All specific fuzzy number operations (such as addition,
    subtraction, multiplication, etc.) should inherit from this
    base class. It defines the core interfaces required for
    operations and provides methods for registration and querying support.
    """

    @abstractmethod
    def get_operation_name(self) -> str:
        """
        Return the name of this operation

        This name will be used to look up and invoke the operation
        in the registry (for example, 'add', 'sub', 'power')..
        """
        pass

    @abstractmethod
    def get_supported_mtypes(self) -> List[str]:
        pass

    def supports_mtype(self, mtype: str) -> bool:
        return mtype in self.get_supported_mtypes()

    def execute_binary(self,
                       strategy_1: Any,
                       strategy_2: Any,
                       t_norm: Any,
                       **kwargs) -> Dict[str, Any]:
        """
        Perform binary operations

        (e.g., addition, subtraction, multiplication, division, comparison, etc.).
        This method should be overridden in subclasses to implement specific operation logic.

        Args:
            strategy_1: First fuzzy number strategy instance.
            strategy_2: Second fuzzy number strategy example.
            t_norm: T-norm instance, used for fuzzy logic operations.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the calculations
                (e.g., {'md': ..., 'nmd': ..., 'q': ...}).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"Binary operation '{self.get_operation_name()}'"
                                  f" not implemented for {self.__class__.__name__}")

    def execute_unary_with_operand(self,
                                   strategy: Any,
                                   operand: Union[int, float],
                                   tnorm: Any,
                                   **kwargs) -> Dict[str, Any]:
        """
        Perform a unary operation with operands

        (e.g., exponentiation, scalar multiplication, exponent, logarithm).

        This method should be overridden in subclasses to implement
        specific operational logic.

        Args:
            strategy: Fuzzy Number Strategy Example.
            operand: Operand of the operation (e.g., exponent, multiplier).
            tnorm: T-norm instance, used for fuzzy logic operations.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the calculations.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"Unary operation with operand '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_unary(self, strategy: Any, tnorm: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform pure unary operations

        (e.g., complement).
        This method should be overridden in subclasses to implement
        specific operational logic.

        Args:
            strategy: Fuzzy Number Strategy Example.
            tnorm: T-norm instance, used for fuzzy logic operations.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the calculations.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"Pure unary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_comparison(self, strategy1: Any, strategy2: Any, tnorm: Any, **kwargs) -> bool:
        """
        Perform comparison operations

        (e.g., greater than, less than, equal to).
        This method should be overridden in subclasses to implement specific comparison logic.

        Args:
            strategy1: First fuzzy number strategy instance.
            strategy2: Second fuzzy number strategy example.
            tnorm: T-norm instance, used for fuzzy logic operations.
            **kwargs: Additional parameters.

        Returns:
            bool: Comparison result (True or False).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"Comparison operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def execute_fuzzarray(self,
                          fuzzarray1: Any,
                          fuzzarray2: Optional[Any],
                          tnorm: Any,
                          **kwargs) -> Any:
        """
        Perform Fuzzarray-level vectorized operations.

        This method should be overridden in subclasses to implement
        efficient computational logic for Fuzzarray.

        Args:
            fuzzarray1: The first Fuzzarray instance.
            fuzzarray2: The second Fuzzarray instance (for binary operations).
            tnorm: T-norm instance, used for fuzzy logic operations.
            **kwargs: Additional parameters.

        Returns:
            Any: The result of the operation, typically a new Fuzzarray instance or a numpy array.

        Raises:
            NotImplementedError: 如果子类未实现此方法。
        """
        raise NotImplementedError(f"Fuzzarray operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")


class OperationRegistry:
    """
    Operation Method Registry.

    Used to store and manage all registered fuzzy number operation methods.

    Through this registry, the corresponding operation implementation can
    be dynamically obtained based on the operation name and fuzzy number type.
    """

    def __init__(self):
        # _operations 结构: {operation_name: {mtype: OperationMixin_instance}}
        self._operations: Dict[str, Dict[str, OperationMixin]] = {}

    def register_operation(self, operation: OperationMixin):
        """
        Register an operation method instance.

        Args:
            operation: The OperationMixin instance to be registered.

        Raises:
            ValueError: If the attempted registration of an operator
                name and type combination already exists.
        """
        op_name = operation.get_operation_name()
        if op_name not in self._operations:
            self._operations[op_name] = {}

        for mtype in operation.get_supported_mtypes():
            if mtype in self._operations[op_name]:
                raise ValueError(f"Operation '{op_name}' for mtype '{mtype}' already registered.")
            self._operations[op_name][mtype] = operation
        # print(f"Registered operation '{op_name}' for mtypes:
        #   {operation.get_supported_mtypes()}") # Debugging

    def get_operation(self, op_name: str, mtype: str) -> Optional[OperationMixin]:
        """
        Get instance of operation method

        Obtain the corresponding operation method instance based on
        the operation name and fuzzy number type.

        Args:
            op_name: Name of the operation.
            mtype: Types of fuzzy numbers.

        Returns:
            Optional[OperationMixin]: The corresponding OperationMixin instance, or None if not found.
        """
        return self._operations.get(op_name, {}).get(mtype)

    def get_available_operations(self, mtype: str) -> List[str]:
        """
        Get a list of supported operator names

        Get the names of all operations supported by the specified fuzzy number type.

        Args:
            mtype: Type of fuzzy numbers.

        Returns:
            List[str]: List of supported operation names.
        """
        operations = []
        for op_name, mtype_ops in self._operations.items():
            if mtype in mtype_ops:
                operations.append(op_name)
        return operations


# Global registry instance to ensure the entire application shares the same registry.
_operation_registry = OperationRegistry()


def get_operation_registry() -> OperationRegistry:
    """
    Get the global operator registry instance.
    """
    return _operation_registry
