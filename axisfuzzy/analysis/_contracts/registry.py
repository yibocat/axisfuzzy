#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 16:35
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Dict, Any, Union, Type, Callable

from .base import ContractValidator


class _ContractRegistry:
    """
    Manages all data contracts and their validators in a central registry.

    This is a singleton class ensuring that there is only one instance of the
    registry throughout the application's lifecycle. It maps contract names
    (strings) to their corresponding validator instances. This class is not
    intended for direct use; interact with it via the public API functions
    ``register_contract`` and ``validate``.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_ContractRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # The __init__ method is guarded to run only once.
        if self._initialized:
            return
        self._validators: Dict[str, ContractValidator] = {}
        self._initialized = True

    def register(self, name: str, validator: ContractValidator):
        if not isinstance(validator, ContractValidator):
            raise TypeError("Validator must be an instance of ContractValidator.")
        self._validators[name] = validator

    def get_validator(self, name: str) -> ContractValidator:
        if name not in self._validators:
            available = ", ".join(self._validators.keys())
            raise KeyError(
                f"Contract '{name}' is not registered. "
                f"Available contracts: [{available}]"
            )
        return self._validators[name]


_contract_registry = _ContractRegistry()


def get_registry_contracts() -> _ContractRegistry:
    """
    Get the singleton instance of the contract registry.

    This function provides access to the central contract registry, allowing
    inspection of registered contracts and their validators.

    Returns
    -------
    _ContractRegistry
        The singleton instance of the contract registry.
    """
    return _contract_registry


def register_contract(name: str) -> Callable:
    """
    A decorator to register a data contract validator.

    This decorator can be applied to a class that inherits from
    `ContractValidator` or to a simple function that returns a boolean.
    It provides a declarative and elegant way to extend the system with
    new, custom data contracts.

    Parameters
    ----------
    name : str
        The unique name for the contract being registered.

    Returns
    -------
    Callable
        A decorator that registers the decorated class or function.

    Examples
    --------
    Decorating a class:

    .. code-block:: python

        from axisfuzzy.analysis.contracts import register_contract, ContractValidator

        @register_contract("MyCustomContract")
        class MyValidator(ContractValidator):
            def validate(self, obj):
                return hasattr(obj, 'my_special_attribute')

    Decorating a function:

    .. code-block:: python

        @register_contract("IsPositiveNumber")
        def is_positive(obj):
            return isinstance(obj, (int, float)) and obj > 0
    """

    def decorator(validator_cls_or_func: Union[Type[ContractValidator], Callable[[Any], bool]]):
        if isinstance(validator_cls_or_func, type) and issubclass(validator_cls_or_func, ContractValidator):
            validator_instance = validator_cls_or_func()
        elif callable(validator_cls_or_func):
            class FunctionalValidator(ContractValidator):
                def validate(self, obj: Any) -> bool:
                    return validator_cls_or_func(obj)

            validator_instance = FunctionalValidator()
        else:
            raise TypeError(
                f"@{register_contract.__name__} can only decorate a "
                f"ContractValidator subclass or a callable function."
            )

        _contract_registry.register(name, validator_instance)
        return validator_cls_or_func

    return decorator


def validate(contract_name: str, obj: Any) -> bool:
    """
    Validate an object against a named contract at runtime.

    This is the primary function used by the pipeline engine to perform
    data contract checks before executing a tool.

    Parameters
    ----------
    contract_name : str
        The name of the contract to validate against.
    obj : Any
        The object to be validated.

    Returns
    -------
    bool
        ``True`` if the object conforms to the contract, ``False`` otherwise.
        Returns ``False`` if the contract name is not registered.
    """
    try:
        validator = _contract_registry.get_validator(contract_name)
        return validator.validate(obj)
    except KeyError:
        return False






















