#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 17:54
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the core Contract object for the v2 data contract system.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Union


class Contract:
    """
    A unified data contract object.

    This class combines a semantic name, a runtime validation function, and
    parent-child relationship metadata into a single, powerful object. It serves
    as both a type hint for static analysis and a validation mechanism for the
    pipeline engine.

    Instances of this class are the single source of truth for data contracts.

    Parameters
    ----------
    name : str
        The unique name of the contract. This name is used for registration
        and identification.
    validator : Callable[[Any], bool]
        A function that takes an object and returns ``True`` if the object
        conforms to the contract, and ``False`` otherwise.
    parent : Contract, optional
        Another ``Contract`` object that this contract inherits from. This
        establishes a compatibility relationship.

    Attributes
    ----------
    name : str
        The name of the contract.
    validator : Callable[[Any], bool]
        The validation function for the contract.
    parent : Contract or None
        The parent contract, if any.
    """
    _registry: Dict[str, Contract] = {}

    def __init__(self,
                 name: str,
                 validator: Callable[[Any], bool],
                 parent: Optional[Contract] = None):
        if name in self._registry:
            raise NameError(
                f"Contract with name '{name}' is already registered. "
                "Contract names must be unique."
            )

        if not callable(validator):
            raise TypeError(f"The validator '{validator.__name__}' must be a callable function.")

        if parent and not isinstance(parent, Contract):
            raise TypeError(f"The 'parent' must be another Contract instance.")

        self.name = name
        self.validator = validator
        self.parent = parent
        self._registry[name] = self

    def validate(self, obj: Any) -> bool:
        """
        Executes the runtime validation for this contract.

        Parameters
        ----------
        obj : Any
            The object to validate.

        Returns
        -------
        bool
            ``True`` if the object is valid, ``False`` otherwise.
        """
        return self.validator(obj)

    def is_compatible_with(self, required_contract: Contract) -> bool:
        """
        Checks if this contract is compatible with a required contract.

        Compatibility means this contract is either the same as the required
        contract or one of its descendants in the inheritance chain.

        Parameters
        ----------
        required_contract : Contract
            The contract that is required by a component's input.

        Returns
        -------
        bool
            ``True`` if this contract can be safely used where the
            `required_contract` is expected.
        """
        if self == required_contract:
            return True

        current = self
        while current.parent:
            if current.parent == required_contract:
                return True
            current = current.parent
        return False

    @classmethod
    def get(cls, name: Union[str, Contract]) -> Contract:
        """
        Retrieves a contract instance from the global registry.

        Parameters
        ----------
        name : str or Contract
            The name of the contract to retrieve. If a Contract object is
            passed, it is returned directly.

        Returns
        -------
        Contract
            The registered contract instance.

        Raises
        ------
        NameError
            If no contract with the given name is found.
        """
        if isinstance(name, Contract):
            return name
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise NameError(
                f"Contract '{name}' is not registered. "
                f"Available contracts: [{available}]"
            )
        return cls._registry[name]

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the Contract."""
        return f"Contract '{self.name}'"
