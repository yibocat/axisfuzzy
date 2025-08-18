#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Optional, Any, Union, List

import numpy as np

from ...config import get_config
from ...core import FuzznumStrategy, register_strategy


@register_strategy
class QROHFNStrategy(FuzznumStrategy):
    mtype = 'qrohfn'
    md: Optional[Union[np.ndarray, List]] = None
    nmd: Optional[Union[np.ndarray, List]] = None

    def __init__(self, q: Optional[int] = None):
        super().__init__(q=q)

        def _to_ndarray(x):
            if x is None:
                return None
            return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=np.float64)

        def _attr_validator(x):
            if x is None:
                return True
            attr = _to_ndarray(x)
            if attr.ndim == 1 and np.max(attr) <= 1 and np.min(attr) >= 0:
                return True
            return False
        
        self.add_attribute_transformer('md', _to_ndarray)
        self.add_attribute_transformer('nmd', _to_ndarray)

        self.add_attribute_validator('md', _attr_validator)
        self.add_attribute_validator('nmd', _attr_validator)

        self.add_change_callback('md', self._on_membership_change)
        self.add_change_callback('nmd', self._on_membership_change)
        self.add_change_callback('q', self._on_q_change)

    def _fuzz_constraint(self):
        if self.md is not None and self.nmd is not None and self.q is not None:
            # For hesitant sets, the constraint applies to the maximum values
            if len(self.md) > 0 and len(self.nmd) > 0:
                sum_of_powers = np.max(self.md) ** self.q + np.max(self.nmd) ** self.q
                if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                    raise ValueError(
                        f"violates fuzzy number constraints: "
                        f"max(md)^q ({np.max(self.md)}^{self.q}) + max(nmd)^q ({np.max(self.nmd)}^{self.q})"
                        f"={sum_of_powers: .4f} > 1.0."
                        f"(q: {self.q}, md: {self.md}, nmd: {self.nmd})")

    def _on_membership_change(self,
                              attr_name: str,
                              old_value: Optional[np.ndarray],
                              new_value: Optional[np.ndarray]) -> None:
        if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
            self._fuzz_constraint()

    def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        if self.md is not None and self.nmd is not None and new_value is not None:
            self._fuzz_constraint()

    def _validate(self) -> None:
        super()._validate()
        self._fuzz_constraint()

    def format_from_components(self,
                               md: Union[np.ndarray, List],
                               nmd: Union[np.ndarray, List],
                               format_spec: str = "") -> str:

        if md is None and nmd is None:
            return "<>"

        precision = get_config().DEFAULT_PRECISION

        def _process_hesitant_set(hesitant_set: Optional[Union[np.ndarray, List]]) -> List:
            """Rounds and uniques a hesitant set for display."""
            if hesitant_set is None or len(hesitant_set) == 0:
                return []
            # Ensure it's a NumPy array for processing
            arr = np.asarray(hesitant_set, dtype=np.float64)
            # 1. Round to the configured precision
            # 2. Get unique elements (which also sorts them)
            # 3. Convert to a Python list for display
            return np.unique(np.round(arr, precision)).tolist()

        md_list = _process_hesitant_set(md)
        nmd_list = _process_hesitant_set(nmd)

        if format_spec == 'p':
            return f"({md_list}, {nmd_list})"
        if format_spec == 'j':
            import json
            return json.dumps({'mtype': self.mtype, 'md': md_list, 'nmd': nmd_list, 'q': self.q})

        return f"<{md_list},{nmd_list}>"

        # def strip_trailing_zeros(x: float) -> str:
        #     s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
        #     return s if s else "0"
        #
        # md_str = strip_trailing_zeros(md)
        # nmd_str = strip_trailing_zeros(nmd)
        # return f"<{md_str},{nmd_str}>"

    def report(self) -> str:
        return self.format_from_components(self.md, self.nmd)

    def str(self) -> str:
        return self.format_from_components(self.md, self.nmd)

    def __format__(self, format_spec) -> str:
        if format and format_spec not in ['p', 'j', 'r']:
            return format(self.str(), format_spec)

        return self.format_from_components(self.md, self.nmd, format_spec)
