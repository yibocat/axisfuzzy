#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np


class MembershipFunction(ABC):
    """隶属函数基类"""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.parameters = {}

    @abstractmethod
    def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """计算隶属度值

        Args:
            x: 输入值或数组

        Returns:
            隶属度值，范围[0, 1]
        """
        pass

    def get_parameters(self) -> dict:
        """获取函数参数"""
        return self.parameters

    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """设置函数参数"""
        pass

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """使函数对象可调用"""
        return self.compute(x)

    def plot(self, x_range: Tuple[float, float] = (0, 1), num_points: int = 1000):
        """绘制隶属函数图形"""
        import matplotlib.pyplot as plt
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = self.compute(x)
        plt.plot(x, y, label=self.name)
        plt.xlabel('x')
        plt.ylabel('Membership Degree')
        plt.title(f'{self.name} Membership Function')
        plt.grid(True)
        plt.legend()
        plt.show()
