#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/1 00:29
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
This module defines the `QROFNStrategy` and `QROFNTemplate` classes,
which implement the specific behavior and representation for Q-Rung Orthopair Fuzzy Numbers (QROFNs)
within the FuzzLab framework.

QROFNs are a generalization of Intuitionistic Fuzzy Sets and Pythagorean Fuzzy Sets,
allowing the sum of the q-th powers of the membership degree (md) and non-membership degree (nmd)
to be less than or equal to 1.

Classes:
    QROFNStrategy: Implements the core logic and constraints for QROFNs,
                   inheriting from `FuzznumStrategy`.
    QROFNTemplate: Defines the representation and auxiliary properties for QROFNs,
                   inheriting from `FuzznumTemplate`.

Example:
    >>> from fuzzlab.core.fuzznums import Fuzznum
    >>> # Create a QROFN instance with q=3, membership degree 0.8, and non-membership degree 0.5
    >>> qrofn_num = Fuzznum(mtype='qrofn', qrung=3).create(md=0.8, nmd=0.5)
    >>> print(qrofn_num.report())
    QROFN(md=0.8, nmd=0.5, q=3)
    >>> print(qrofn_num.score)
    0.39
    >>> print(qrofn_num.accuracy)
    0.717
    >>> print(qrofn_num.indeterminacy)
    0.656
"""

from typing import Optional, Any

from fuzzlab.config import get_config
from fuzzlab.core.base import FuzznumStrategy, FuzznumTemplate


class QROFNStrategy(FuzznumStrategy):
    """
    Implements the strategy for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    This class extends `FuzznumStrategy` to define the specific attributes,
    validation rules, and constraint checks for QROFNs. It ensures that
    the membership degree (`md`) and non-membership degree (`nmd`) adhere
    to the QROFN definition: `md^q + nmd^q <= 1`.

    Attributes:
        mtype (str): The membership type identifier for QROFNs, set to 'qrofn'.
        md (Optional[float]): The membership degree of the QROFN, a value between 0 and 1.
        nmd (Optional[float]): The non-membership degree of the QROFN, a value between 0 and 1.
    """
    mtype = 'qrofn'
    md: Optional[float] = None
    nmd: Optional[float] = None

    def __init__(self, qrung: Optional[int] = None):
        """
        Initializes a QROFNStrategy instance.

        Args:
            qrung (Optional[int]): The q-rung value for this QROFN.
                                   Inherited from `FuzznumStrategy`.
        """
        super().__init__(qrung=qrung)

        # Add attribute validators for 'md' and 'nmd'.
        # These validators ensure that 'md' and 'nmd' are floats or ints within the [0, 1] range.
        self.add_attribute_validator(
            'md', lambda x: x is None or isinstance(x, (int, float)) and 0 <= x <= 1)
        self.add_attribute_validator(
            'nmd', lambda x: x is None or isinstance(x, (int, float)) and 0 <= x <= 1)

        # Add change callbacks for 'md', 'nmd', and 'q'.
        # These callbacks trigger the fuzzy constraint check whenever these attributes change.
        self.add_change_callback('md', self._on_membership_change)
        self.add_change_callback('nmd', self._on_membership_change)
        self.add_change_callback('q', self._on_q_change)

    def _fuzz_constraint(self):
        """
        Enforces the Q-Rung Orthopair Fuzzy Number constraint.

        This method checks if the sum of the q-th powers of `md` and `nmd`
        exceeds 1. If it does, a `ValueError` is raised, indicating a violation
        of the QROFN definition. This method is called by attribute change
        callbacks and the main `_validate` method.

        Raises:
            ValueError: If `md^q + nmd^q` is greater than 1.
        """
        # Check if both membership and non-membership degrees, and the q-rung, are set.
        if self.md is not None and self.nmd is not None and self.q is not None:
            # Calculate the sum of the q-th powers.
            sum_of_powers = self.md ** self.q + self.nmd ** self.q
            # Compare with 1, allowing for a small epsilon to account for floating-point inaccuracies.
            if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                raise ValueError(
                    f"violates fuzzy number constraints: "
                    f"md^q ({self.md}^{self.q}) + nmd^q ({self.nmd}^{self.q})"
                    f"={sum_of_powers: .4f} > 1.0."
                    f"(q: {self.q}, md: {self.md}, nmd: {self.nmd})"
                )

    def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """
        Callback function triggered when `md` or `nmd` attributes change.

        This callback ensures that the fuzzy number constraints are checked
        whenever the membership or non-membership degrees are updated.
        It prevents the instance from entering an invalid state.

        Args:
            attr_name (str): The name of the attribute that changed ('md' or 'nmd').
            old_value (Any): The previous value of the attribute.
            new_value (Any): The new value of the attribute.
        """
        # Only perform the constraint check if the new value is not None,
        # and both 'md' and 'nmd' attributes are present on the instance.
        # This prevents incomplete checks during the object's initialization phase.
        if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
            self._fuzz_constraint()
        # if self.md is not None and self.nmd is not None and self.q is not None:
        #     self._fuzz_constraint()

    def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """
        Callback function triggered when the 'q' (qrung) attribute changes.

        This callback ensures that the fuzzy number constraints are re-checked
        when the q-rung value is updated, as the constraint `md^q + nmd^q <= 1`
        is dependent on `q`.

        Args:
            attr_name (str): The name of the attribute that changed ('q').
            old_value (Any): The previous value of the attribute.
            new_value (Any): The new value of the attribute.
        """
        # Similar to `_on_membership_change`, this triggers the constraint check.
        # At this point, `self.q` has already been updated to `new_value`
        # (because `super().__setattr__` has already executed), so `_fuzz_constraint`
        # can be called directly.
        if self.md is not None and self.nmd is not None and new_value is not None:
            self._fuzz_constraint()

    def _validate(self) -> None:
        """
        Performs comprehensive validation of the QROFNStrategy instance.

        This method extends the base `FuzznumStrategy._validate` method
        by adding the specific QROFN constraint check. It ensures that
        the instance's state is valid according to the QROFN definition.

        Raises:
            ValueError: If any validation constraint is violated.
        """
        # Call the parent class's _validate method to perform its default validation.
        super()._validate()
        # Perform the specific QROFN fuzzy constraint check.
        self._fuzz_constraint()


class QROFNTemplate(FuzznumTemplate):
    """
    Implements the template for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    This class extends `FuzznumTemplate` to provide specific string
    representations, detailed reports, and computational properties
    (score, accuracy, indeterminacy) for QROFNs. It interacts with
    the associated `Fuzznum` instance (via `self.instance`) to access
    its `md`, `nmd`, and `q` attributes.

    Attributes:
        mtype (str): The membership type identifier for QROFNs, set to 'qrofn'.
    """

    mtype = 'qrofn'

    def report(self) -> str:
        """
        Generates a detailed report string for the QROFN.

        This method provides a comprehensive string representation of the QROFN,
        including its membership degree (`md`), non-membership degree (`nmd`),
        and q-rung (`q`).

        Returns:
            str: A detailed string representation of the QROFN,
                 e.g., "QROFN(md=0.8, nmd=0.5, q=3)".
        """
        # If both md and nmd are None, return an empty representation.
        if self.instance.md is None and self.instance.nmd is None:
            return "<>"
        # Return a formatted string including md, nmd, and q.
        precision = get_config().DEFAULT_PRECISION
        md = round(self.instance.md, precision)
        nmd = round(self.instance.nmd, precision)
        return f"<{md},{nmd}>"

    def str(self) -> str:
        """
        Generates a concise string representation of the QROFN.

        This method provides a brief string representation of the QROFN,
        typically used for `print()` or `str()` calls.

        Returns:
            str: A concise string representation of the QROFN,
                 e.g., "<0.8,0.5>_q=3".
        """
        # If both md and nmd are None, return an empty representation.
        if self.instance.md is None and self.instance.nmd is None:
            return "<>"
        # Return a formatted string including md, nmd, and q.
        precision = get_config().DEFAULT_PRECISION
        md = round(self.instance.md, precision)
        nmd = round(self.instance.nmd, precision)
        return f"<{md},{nmd}>"

    @property
    def score(self):
        """
        Calculates the score of the QROFN.

        The score function for a QROFN is typically defined as `md^q - nmd^q`.
        This property provides a rounded result based on the default precision
        configured in FuzzLab.

        Returns:
            float: The calculated score value.
        """
        config = get_config()
        # Access md, nmd, and q from the associated Fuzznum instance.
        result = self.instance.md ** self.instance.q - self.instance.nmd ** self.instance.q
        return round(result, config.DEFAULT_PRECISION)

    @property
    def accuracy(self):
        """
        Calculates the accuracy of the QROFN.

        The accuracy function for a QROFN is typically defined as `md^q + nmd^q`.
        This property provides a rounded result based on the default precision
        configured in FuzzLab.

        Returns:
            float: The calculated accuracy value.
        """
        config = get_config()
        # Access md, nmd, and q from the associated Fuzznum instance.
        result = self.instance.md ** self.instance.q + self.instance.nmd ** self.instance.q
        return round(result, config.DEFAULT_PRECISION)

    @property
    def indeterminacy(self):
        """
        Calculates the indeterminacy (hesitation degree) of the QROFN.

        The indeterminacy for a QROFN is derived from its accuracy,
        calculated as `(1 - accuracy)^(1/q)`. This represents the degree
        to which an element neither belongs nor does not belong to the set.
        This property provides a rounded result based on the default precision
        configured in FuzzLab.

        Returns:
            float: The calculated indeterminacy value.
        """
        config = get_config()
        # Calculate indeterminacy based on the accuracy and q-rung.
        # Ensure the base of the power is non-negative to avoid complex numbers.
        base = 1 - self.accuracy
        if base < config.DEFAULT_EPSILON:
            base = 0    # Handle potential floating point inaccuracies
        result = base ** (1 / self.instance.q)
        return round(result, config.DEFAULT_PRECISION)

    def __format__(self, format_spec: str) -> str:
        """Provides custom formatting for the example fuzzy number.

        This implementation extends the default formatting behavior to support
        several custom format specifiers specific to this fuzzy number type.

        Format Specifiers:
            s: The score of the fuzzy number (md - nmd).
            a: The accuracy of the fuzzy number accuracy.
            i: The indeterminacy of the fuzzy number indeterminacy.
            r: The detailed report string.
            p: The parameters as a tuple string, e.g., '(0.8, 0.1)'.
            j: A JSON string representation of the fuzzy number's attributes.

        If a specifier other than the custom ones is provided, it falls back
        to the standard string formatting applied to the concise representation.

        Args:
            format_spec (str): The format specification string.

        Returns:
            str: The formatted string representation.

        Examples:
            fuzz = ...
            f"Score: {fuzz:s}"
            f"Accuracy: {fuzz:a}"
            f"Indeterminacy: {fuzz:i}"
            f"Report: {fuzz:r}"
            f"Params: {fuzz:p}"
            f"JSON: {fuzz:j}"
            f"Right-aligned: {fuzz:>20}"
        """
        # If no format specifier is provided, return the default string representation.
        if not format_spec:
            return self.str()

        # Handle custom format specifiers.
        if format_spec == 's':
            return str(self.score)
        if format_spec == 'a':
            return str(self.accuracy)
        if format_spec == 'i':
            return str(self.indeterminacy)
        if format_spec == 'r':
            return self.report()
        if format_spec == 'p':
            return f"({self.instance.md}, {self.instance.nmd})"
        if format_spec == 'j':
            import json
            return json.dumps({
                'mtype': self.mtype,
                'md': self.instance.md,
                'nmd': self.instance.nmd,
                'q': self.instance.q
            })

        # If it's not a special specifier, use the default behavior from the base class.
        # This allows for standard formatting like alignment and width on the default string.
        return super().__format__(format_spec)
