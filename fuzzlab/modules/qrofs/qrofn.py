#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/1 00:29
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from typing import Optional, Dict, Any, Union

from fuzzlab.config import get_config
from fuzzlab.core.base import FuzznumStrategy, FuzznumTemplate
from fuzzlab.core.triangular import OperationTNorm


class QROFNStrategy(FuzznumStrategy):
    mtype = 'qrofn'
    md: Optional[float] = None
    nmd: Optional[float] = None

    def __init__(self, qrung: Optional[int] = None):
        super().__init__(qrung=qrung)

        # 添加隶属度约束和非隶属度约束条件
        self.add_attribute_validator(
            'md', lambda x: x is None or isinstance(x, (int, float)) and 0 <= x <= 1)
        self.add_attribute_validator(
            'nmd', lambda x: x is None or isinstance(x, (int, float)) and 0 <= x <= 1)

        # 添加模糊约束条件变更回调
        self.add_change_callback('md', self._on_membership_change)
        self.add_change_callback('nmd', self._on_membership_change)
        self.add_change_callback('q', self._on_q_change)

    def _fuzz_constraint(self):
        # 模糊约束条件
        if self.md is not None and self.nmd is not None and self.q is not None:
            sum_of_powers = self.md ** self.q + self.nmd ** self.q
            if sum_of_powers > 1 + get_config().DEFAULT_EPSILON:
                raise ValueError(
                    f"violates fuzzy number constraints: "
                    f"md^q ({self.md}^{self.q}) + nmd^q ({self.nmd}^{self.q})"
                    f"={sum_of_powers: .4f} > 1.0."
                    f"(q: {self.q}, md: {self.md}, nmd: {self.nmd})"
                )

    def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """隶属度或非隶属度变更时的回调函数。

        此回调函数在 `md` 或 `nmd` 属性被设置时触发。它会检查模糊数约束条件
        `md + nmd <= 1`，如果违反则发出警告。
        """
        if new_value is not None and self.q is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
            # 只有当新值不为 None，并且实例上同时存在 'md' 和 'nmd' 属性时才执行后续检查。
            # 这确保了在对象初始化过程中，当属性可能尚未完全设置时，不会触发不完整的检查。
            self._fuzz_constraint()
        # if self.md is not None and self.nmd is not None and self.q is not None:
        #     self._fuzz_constraint()

    def _on_q_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """
        当 'q' 属性变更时触发的回调函数。
        执行模糊数约束检查。
        """
        # 类似于 _on_membership_change，触发约束检查。
        # 此时 self.q 已经更新为 new_value (因为 super().__setattr__ 已经执行)，
        # 所以直接调用 _fuzz_constraint 即可。
        if self.md is not None and self.nmd is not None and new_value is not None:
            self._fuzz_constraint()

    def _validate(self) -> None:
        # 计算结果验证，所以一般情况下一旦 attribute_validator 验证成功，计算结果就是正确的
        # 该方法用于加强验证
        super()._validate()
        self._fuzz_constraint()


class QROFNTemplate(FuzznumTemplate):
    """Q阶序对模糊数模板"""

    mtype = 'qrofn'

    def report(self) -> str:
        # 实现 FuzznumTemplate 的抽象方法 report。
        # 这里简单地返回 str() 的结果，可以根据需要扩展为更详细的报告。

        if self.instance.md is None and self.instance.nmd is None:
            return "<>"
        return f"<{self.instance.md},{self.instance.nmd}>"

    def str(self) -> str:
        # 实现 FuzznumTemplate 的抽象方法 str。
        # 返回 QROFN 的简洁字符串表示形式，包含 md, nmd 和 q。
        # 通过 self.instance 访问关联的 Fuzznum 实例的属性。

        if self.instance.md is None and self.instance.nmd is None:
            return "<>"
        return f"<{self.instance.md},{self.instance.nmd}>"

    @property
    def score(self):
        config = get_config()
        result = self.instance.md ** self.instance.q - self.instance.nmd ** self.instance.q
        return round(result, config.DEFAULT_PRECISION)

    @property
    def accuracy(self):
        config = get_config()
        result = self.instance.md ** self.instance.q + self.instance.nmd ** self.instance.q
        return round(result, config.DEFAULT_PRECISION)

    @property
    def indeterminacy(self):
        config = get_config()
        result = (1 - self.accuracy) ** (1/self.instance.q)
        return round(result, config.DEFAULT_PRECISION)
