#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
axisfuzzy.core.backend
======================

Abstract backend interface for Fuzzarray (Struct-of-Arrays).

This module defines the `FuzzarrayBackend` abstract base class which is the
primary contract between high-level `Fuzzarray` containers and concrete,
mtype-specific, NumPy-backed implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Callable

import numpy as np

from ..config import get_config


class FuzzarrayBackend(ABC):
    """
    Abstract base class for Fuzzarray data backends (SoA implementation).

    Each concrete backend manages the NumPy arrays that store the fuzzy-number
    components for a specific `mtype`. Subclasses must implement array
    initialization, element view construction, slicing/copy semantics and
    formatting.

    Parameters
    ----------
    shape : tuple of int
        Shape of the logical Fuzzarray.
    q : int, optional
        q-rung parameter for q-rung based mtypes. If ``None``, the default
        from configuration is used.
    **kwargs
        Additional backend-specific parameters.

    Attributes
    ----------
    shape : tuple of int
        The logical shape of the array.
    size : int
        Total number of elements (product of shape).
    q : int
        Effective q-rung used by this backend instance.
    kwargs : dict
        Backend-specific extra parameters.
    mtype : str
        Class attribute that concrete subclasses should override to indicate
        the supported mtype string.

    Notes
    -----
    - Implementations are expected to prefer views (shared memory) for slicing
      whenever possible to avoid expensive copies.
    - The backend API focuses on array-level operations; high-level semantics
      (e.g., operator dispatch) are handled by other core components.
    """

    mtype: str = "unknown"

    def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
        self.shape = shape
        self.size = int(np.prod(shape))
        self.q = q if q is not None else get_config().DEFAULT_Q

        self.kwargs = kwargs

        # 子类需要在这里初始化具体的 NumPy 数组
        self._initialize_arrays()

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the logical Fuzzarray.

        Returns
        -------
        int
            The number of dimensions (len(shape)).
        """
        return len(self.shape)

    @abstractmethod
    def _initialize_arrays(self):
        """
        Initialize the NumPy arrays that store the fuzzy-number components.

        Implementations should create and store the component arrays (for
        example: membership degrees, non-membership degrees, etc.) as
        attributes of the backend instance.

        Notes
        -----
        - This method is called during construction; subclasses should not
          assume the existence of other backend methods being callable yet.
        """
        pass

    @abstractmethod
    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """
        Create a lightweight Fuzznum view for the element at `index`.

        The view should be a thin wrapper (no deep copy) that provides
        per-element access and can be used by high-level operations that
        expect a Fuzznum-like object.

        Parameters
        ----------
        index : int or tuple or slice
            Index of the element to view. Accepts the same indexing semantics
            as NumPy arrays for single-element access.

        Returns
        -------
        Fuzznum
            A Fuzznum-like object representing the element at `index`.

        Raises
        ------
        IndexError
            If `index` is out of bounds.

        Notes
        -----
        - Implementations may return a proxy object that reads/writes directly
          into the backend arrays to avoid copying.
        """
        pass

    @abstractmethod
    def set_fuzznum_data(self, index: Any, fuzznum: 'Fuzznum'):
        """
        Set data at `index` using values from a Fuzznum object.

        Parameters
        ----------
        index : int or tuple or slice
            Target location to set.
        fuzznum : Fuzznum
            Source Fuzznum providing component values.

        Raises
        ------
        IndexError
            If `index` is out of bounds.
        TypeError
            If `fuzznum` is not compatible with this backend's mtype.
        """
        pass

    @abstractmethod
    def copy(self) -> 'FuzzarrayBackend':
        """
        Create a deep copy of this backend and its underlying arrays.

        Returns
        -------
        FuzzarrayBackend
            A new backend instance with duplicated (deep-copied) arrays.

        Notes
        -----
        - This operation must not share memory with the original instance.
        """
        pass

    @abstractmethod
    def slice_view(self, key) -> 'FuzzarrayBackend':
        """
        Return a backend representing a slice/view of this backend.

        Parameters
        ----------
        key : slice, tuple of slices or other valid indexing key
            Indexing key describing the requested view.

        Returns
        -------
        FuzzarrayBackend
            A new backend instance representing the requested slice. The
            returned backend should share data with the original when possible.

        Notes
        -----
        - Preferred behavior is to avoid copying and to provide a view with
          consistent semantics for read/write operations.
        """
        pass

    @staticmethod
    def from_arrays(*components, **kwargs) -> 'FuzzarrayBackend':
        """
        Construct a backend instance from raw component arrays.

        Parameters
        ----------
        *components : array_like
            Component arrays that represent the SoA layout for a specific mtype.
        **kwargs
            Backend-specific keyword arguments (e.g., shape, q, dtype).

        Returns
        -------
        FuzzarrayBackend
            A new backend instance initialised from the given arrays.

        Notes
        -----
        - This factory is a convenience hook and should be implemented by
          concrete subclasses to validate shapes and dtypes.
        """
        pass

    def fill_from_values(self, *values: float):
        """
        Fill every element in the backend with the provided component values.

        Parameters
        ----------
        *values : float
            Values to broadcast to every element for each component.

        Notes
        -----
        - Concrete backends should validate the number of provided values
          against the expected number of components for the mtype.
        """
        pass

    def get_component_arrays(self) -> tuple:
        """
        Return the underlying component arrays used by the backend.

        Returns
        -------
        tuple
            Tuple containing the NumPy arrays that make up the SoA backend.

        Notes
        -----
        - The order and meaning of returned arrays is backend-specific and
          should be documented by each concrete implementation.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, mtype='{self.mtype}')"

    # ================== 智能显示通用实现 ==================

    def format_elements(self, format_spec: str = "") -> np.ndarray:
        """
        智能分段格式化：对大数据集只显示部分元素，其余用省略号代替。

        显示策略：
        - 小数组（< DISPLAY_THRESHOLD_SMALL）：完整显示
        - 大数组：显示首尾部分 + 中间省略号
        - 超大数组：进一步减少显示元素数量

        Parameters
        ----------
        format_spec : str, optional
            Format specification passed from higher-level formatting calls.

        Returns
        -------
        numpy.ndarray
            Array of formatted strings with same shape as backend.
        """
        if self.size == 0:
            return np.array([], dtype=object).reshape(self.shape)

        config = get_config()

        # 根据数据大小选择显示策略
        if self.size <= config.DISPLAY_THRESHOLD_SMALL:
            # 小数据集：完整显示
            return self._format_all_elements(format_spec)
        else:
            # 大数据集：分段显示
            return self._format_partial_elements(format_spec)

    def _format_all_elements(self, format_spec: str) -> np.ndarray:
        """完整格式化所有元素（用于小数据集）"""
        # 对于小数据集，子类可以选择性地重写这个方法来优化
        # 默认实现：逐元素格式化
        result = np.empty(self.shape, dtype=object)
        formatter = self._get_element_formatter(format_spec)

        it = np.nditer(result, flags=['multi_index', 'refs_ok'], op_flags=['writeonly'])
        while not it.finished:
            idx = it.multi_index
            result[idx] = self._format_single_element(idx, formatter, format_spec)
            it.iternext()

        return result

    def _format_partial_elements(self, format_spec: str) -> np.ndarray:
        """
        分段格式化大数据集：只格式化需要显示的部分。

        策略：
        1. 根据数组维度和大小计算显示参数
        2. 只格式化需要显示的索引位置
        3. 其他位置用省略号占位
        """
        # 计算显示参数
        display_params = self._calculate_display_parameters()

        # 创建结果数组，默认填充省略号
        result = np.full(self.shape, '...', dtype=object)

        # 获取需要显示的索引
        indices_to_format = self._get_display_indices(display_params)

        # 只格式化需要显示的元素
        formatter = self._get_element_formatter(format_spec)

        for idx in indices_to_format:
            result[idx] = self._format_single_element(idx, formatter, format_spec)

        return result

    def _calculate_display_parameters(self) -> dict:
        """
        根据数组大小和维度计算显示参数。

        Returns:
            dict: 包含显示参数的字典
        """
        config = get_config()

        # 基础显示参数
        if self.size < config.DISPLAY_THRESHOLD_MEDIUM:
            # 中等大小：显示更多元素
            edge_items = config.DISPLAY_EDGE_ITEMS_MEDIUM
            threshold = config.DISPLAY_THRESHOLD_MEDIUM
        elif self.size < config.DISPLAY_THRESHOLD_LARGE:
            # 大数组：适中显示
            edge_items = config.DISPLAY_EDGE_ITEMS_LARGE
            threshold = config.DISPLAY_THRESHOLD_LARGE
        else:
            # 超大数组：最少显示
            edge_items = config.DISPLAY_EDGE_ITEMS_HUGE
            threshold = config.DISPLAY_THRESHOLD_HUGE

        return {
            'edge_items': edge_items,
            'threshold': threshold,
            'ndim': self.ndim,
            'shape': self.shape,
            'size': self.size
        }

    def _get_display_indices(self, params: dict) -> list:
        """
        根据显示参数计算需要格式化的索引位置。

        模仿 NumPy 的显示逻辑：
        - 1D: [0, 1, ..., -2, -1]
        - 2D: 四角 + 边缘
        - 高维: 递归处理
        """
        indices = []
        edge_items = params['edge_items']
        shape = params['shape']

        if self.ndim == 1:
            # 1D 数组：显示前后各 edge_items 个
            if shape[0] <= 2 * edge_items + 1:
                # 数组太小，全部显示
                indices = list(range(shape[0]))
            else:
                # 前 edge_items 个
                indices.extend(list(range(edge_items)))
                # 后 edge_items 个
                indices.extend(list(range(shape[0] - edge_items, shape[0])))

        elif self.ndim == 2:
            # 2D 数组：显示四角区域
            rows, cols = shape

            # 确定要显示的行和列
            if rows <= 2 * edge_items + 1:
                row_indices = list(range(rows))
            else:
                row_indices = (list(range(edge_items)) +
                              list(range(rows - edge_items, rows)))

            if cols <= 2 * edge_items + 1:
                col_indices = list(range(cols))
            else:
                col_indices = (list(range(edge_items)) +
                              list(range(cols - edge_items, cols)))

            # 生成所有需要的 (row, col) 组合
            for r in row_indices:
                for c in col_indices:
                    indices.append((r, c))

        else:
            # 高维数组：递归处理（简化实现）
            indices = self._get_high_dim_indices(shape, edge_items)

        return indices

    def _get_high_dim_indices(self, shape: tuple, edge_items: int) -> list:
        """处理高维数组的显示索引（简化实现）"""
        indices = []

        def generate_edge_indices(current_shape, current_idx=None):
            if current_idx is None:
                current_idx = []
            if not current_shape:
                indices.append(tuple(current_idx))
                return

            dim_size = current_shape[0]
            remaining_shape = current_shape[1:]

            if dim_size <= 2 * edge_items + 1:
                # 维度较小，全部显示
                for i in range(dim_size):
                    generate_edge_indices(remaining_shape, current_idx + [i])
            else:
                # 显示前后各 edge_items 个
                for i in range(edge_items):
                    generate_edge_indices(remaining_shape, current_idx + [i])
                for i in range(dim_size - edge_items, dim_size):
                    generate_edge_indices(remaining_shape, current_idx + [i])

        generate_edge_indices(shape)
        return indices

    @abstractmethod
    def _get_element_formatter(self, format_spec: str) -> Callable:
        """
        获取元素格式化函数。

        子类必须实现此方法，返回一个可调用对象，该对象接受必要的参数
        并返回格式化后的字符串。

        Parameters
        ----------
        format_spec : str
            格式规格说明

        Returns
        -------
        Callable
            格式化函数
        """
        pass

    @abstractmethod
    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """
        格式化单个元素。

        子类必须实现此方法来定义如何格式化位于指定索引的单个元素。

        Parameters
        ----------
        index : Any
            元素索引
        formatter : Callable
            格式化函数（由 _get_element_formatter 返回）
        format_spec : str
            格式规格说明

        Returns
        -------
        str
            格式化后的字符串
        """
        pass
