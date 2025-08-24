#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 16:17
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Base definitions for the data contract system, including the validator
interface and standard type aliases.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias, Union, Dict, Any, List, TYPE_CHECKING


class ContractValidator(ABC):
    """
    Abstract base class for all data contract validators.

    This class defines a standard interface, ``validate``, which all concrete
    validator implementations must provide. This ensures that the registry
    and pipeline engine can handle any registered contract polymorphically.
    """

    @abstractmethod
    def validate(self, obj: Any) -> bool:
        """
        Validate if an object conforms to the specific contract.

        Parameters
        ----------
        obj : Any
            The object to be validated.

        Returns
        -------
        bool
            ``True`` if the object conforms to the contract, ``False`` otherwise.
        """
        pass













































