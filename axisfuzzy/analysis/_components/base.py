#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 10:57
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Defines the base class for all analysis _components in AxisFuzzy.
"""
from abc import abstractmethod, ABC


class AnalysisComponent(ABC):
    """
    A base marker class for all analysis component tools.

    While not strictly required, it is best practice for all analysis
    _components to inherit from this class. This provides a clear, common
    ancestor for all tools and can be used for type checking, discovery,
    and future framework-level features.

    It enforces a consistent design pattern where tools are instantiated
    objects rather than standalone functions, promoting better state
    management and code organization.

    Notes
    -----
    As of Phase 3.B, this class is no longer abstract. The `run` method
    is provided with a basic implementation to allow classes with their own
    `run` signature (like `FuzzyPipeline`) to inherit without conflict.
    Subclasses are still expected to provide a meaningful `run` method.
    """

    def run(self, *args, **kwargs):
        """
        The main execution method for the analysis component.

        This method should be overridden by all subclasses to implement
        their specific functionality. It serves as the entry point for
        running the tool's analysis logic.

        Parameters
        ----------
        *args : tuple
            Positional arguments specific to the tool's requirements.
        **kwargs : dict
            Keyword arguments specific to the tool's requirements.

        Returns
        -------
        Any
            The output of the analysis, which can vary based on the tool.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement its own run method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'run' method."
        )
