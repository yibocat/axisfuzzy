#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Optional, Any

import numpy as np

from ...config import get_config
from ...core import FuzznumStrategy, register_strategy


@register_strategy
class QROFNStrategy(FuzznumStrategy):
    mtype = 'qrofn'
    md: Optional[float] = None
    nmd: Optional[float] = None

    def __init__(self, q: Optional[int] = None):
        super().__init__(q=q)

        self.add_attribute_validator(
            'md', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)
        self.add_attribute_validator(
            'nmd', lambda x: x is None or isinstance(x, (int, float, np.floating, np.integer)) and 0 <= x <= 1)

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

    def format_from_components(self, md: float, nmd: float,
                               format_spec: str = "") -> str:
        if md is None and nmd is None:
            return "<>"
        precision = get_config().DEFAULT_PRECISION
        if format_spec == 'p':
            return f"({md}, {nmd})"
        if format_spec == 'j':
            import json
            return json.dumps({'mtype': self.mtype, 'md': md, 'nmd': nmd, 'q': self.q})
        # 'r' 当前等同默认

        def strip_trailing_zeros(x: float) -> str:
            s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
            return s if s else "0"

        md_str = strip_trailing_zeros(md)
        nmd_str = strip_trailing_zeros(nmd)
        return f"<{md_str},{nmd_str}>"

    def report(self) -> str:
        return self.format_from_components(self.md, self.nmd)

    def str(self) -> str:
        return self.format_from_components(self.md, self.nmd)

    def __format__(self, format_spec: str) -> str:
        """Provides custom formatting by delegating to the stateless class method."""
        # If a specifier other than the custom ones is provided, it falls back
        # to the standard string formatting applied to the concise representation.
        if format_spec and format_spec not in ['r', 'p', 'j']:
            return format(self.str(), format_spec)

        return self.format_from_components(self.md, self.nmd, format_spec)
