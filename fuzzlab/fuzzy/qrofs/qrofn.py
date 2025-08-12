#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/12 19:53
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Optional, Any

from ...config import get_config
from ...core import FuzznumStrategy


class QROFNStrategy(FuzznumStrategy):

    mtype = 'qrofn'
    md: Optional[float] = None
    nmd: Optional[float] = None

    def __init__(self, q: Optional[int] = None):
        super().__init__(q=q)

        self.add_attribute_validator(
            'md', lambda x: x is None or isinstance(x, (int, float)) and 0 <= x <= 1)
        self.add_attribute_validator(
            'nmd', lambda x: x is None or isinstance(x, (int, float)) and 0 <= x <= 1)

        self.add_change_callback('md', self._on_membership_change)
        self.add_change_callback('nmd', self._on_membership_change)
        self.add_change_callback('q', self._on_q_change)

    def _fuzz_constraint(self):
        if self.md is not None and self.nmd is not None and self.q is not None:
            sum_of_powers = self.md ** self.q + self.nmd ** self.q
            if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                raise ValueError(
                    f"violates fuzzy number constraints: "
                    f"md^q ({self.md}^{self.q}) + nmd^q ({self.nmd}^{self.q})"
                    f"={sum_of_powers: .4f} > 1.0."
                    f"(q: {self.q}, md: {self.md}, nmd: {self.nmd})")

    def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
            self._fuzz_constraint()

    def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        if self.md is not None and self.nmd is not None and new_value is not None:
            self._fuzz_constraint()

    def _validate(self) -> None:
        super()._validate()
        self._fuzz_constraint()

    @classmethod
    def format_from_components(cls, md: float, nmd: float) -> str:
        """
        A class method to format a q-rung orthopair fuzzy number from its raw components.
        """
        if md is None and nmd is None:
            return "<>"
        precision = get_config().DEFAULT_PRECISION
        md_str = f"{round(md, precision)}"
        nmd_str = f"{round(nmd, precision)}"
        return f"<{md_str},{nmd_str}>"

    def report(self) -> str:
        return self.format_from_components(self.md, self.nmd)

    def str(self) -> str:
        return self.format_from_components(self.md, self.nmd)

    def __format__(self, format_spec: str) -> str:
        """Provides custom formatting for the example fuzzy number.

        This implementation extends the default formatting behavior to support
        several custom format specifiers specific to this fuzzy number type.

        Format Specifiers:
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
            f"Report: {fuzz:r}"
            f"Params: {fuzz:p}"
            f"JSON: {fuzz:j}"
            f"Right-aligned: {fuzz:>20}"
        """
        # If no format specifier is provided, return the default string representation.
        if not format_spec:
            return self.str()

        if format_spec == 'r':
            return self.report()
        if format_spec == 'p':
            return f"({self.md}, {self.nmd})"
        if format_spec == 'j':
            import json
            return json.dumps({
                'mtype': self.mtype,
                'md': self.md,
                'nmd': self.nmd,
                'q': self.q
            })
        return super().__format__(format_spec)
