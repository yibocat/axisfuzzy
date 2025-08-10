#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/10 13:19
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Optional, Union, List, Any, Tuple
import numpy as np

from ..config import get_config
from .fuzznums import Fuzznum
from .t_backend import FuzzarrayBackend
from .registry import get_fuzznum_registry
from .operation import get_operation_registry, OperationMixin
from .triangular import OperationTNorm


class Fuzzarray:
    """
    High-performance fuzzy array using Struct of Arrays (SoA) architecture.
    """

    def __init__(self,
                 data: Optional[Union[np.ndarray, list, tuple, Fuzznum]] = None,
                 mtype: Optional[str] = None,
                 shape: Optional[Tuple[int, ...]] = None,
                 copy: bool = True,
                 backend: Optional[FuzzarrayBackend] = None,
                 **mtype_kwargs):
        """
        Initialize a Fuzzarray with SoA backend.
        """
        # 如果提供了预构建的 backend，直接使用它
        if backend is not None:
            self._backend = backend
            self._mtype = backend.mtype
            self._mtype_kwargs = backend.mtype_kwargs
            return

        # 1. 确定 mtype 和 mtype_kwargs
        self._mtype, self._mtype_kwargs = self._determine_mtype_and_kwargs(
            data, mtype, mtype_kwargs
        )

        # 2. 确定最终形状
        final_shape = self._determine_shape(data, shape)

        # 3. 创建 mtype 专属的后端
        self._backend = self._create_backend(final_shape)

        # 4. 填充数据到后端
        if data is not None:
            self._populate_backend_from_data(data, copy)

    def _determine_mtype_and_kwargs(self, data, mtype, mtype_kwargs):
        """确定 mtype 和相关参数"""
        if mtype is not None:
            return mtype, mtype_kwargs

        # 从数据中推断 mtype 和参数
        if isinstance(data, Fuzznum):
            # 从 Fuzznum 中提取 mtype 和 q 参数
            extracted_kwargs = {'q': data.q}
            extracted_kwargs.update(mtype_kwargs)  # 用户提供的参数优先
            return data.mtype, extracted_kwargs
        elif data is not None and len(data) > 0:
            # 从第一个非空元素推断
            first_elem = self._find_first_fuzznum(data)
            if first_elem is not None:
                extracted_kwargs = {'q': first_elem.q}
                extracted_kwargs.update(mtype_kwargs)  # 用户提供的参数优先
                return first_elem.mtype, extracted_kwargs

        # 使用默认值
        default_kwargs = {'q': 1}
        default_kwargs.update(mtype_kwargs)
        return get_config().DEFAULT_MTYPE, default_kwargs

    def _find_first_fuzznum(self, data):
        """递归查找第一个 Fuzznum 对象"""
        if isinstance(data, Fuzznum):
            return data
        elif hasattr(data, '__iter__'):
            try:
                for item in data:
                    result = self._find_first_fuzznum(item)
                    if result is not None:
                        return result
            except (TypeError, StopIteration):
                pass
        return None

    def _determine_shape(self, data, shape):
        """确定最终数组形状"""
        if shape is not None:
            return shape

        if isinstance(data, Fuzznum):
            return ()  # 标量
        elif isinstance(data, np.ndarray):
            return data.shape
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=object).shape
        else:
            raise ValueError("Shape must be provided when data is None or ambiguous")

    def _create_backend(self, shape):
        """创建 mtype 专属的后端"""
        registry = get_fuzznum_registry()
        backend_class = registry.get_backend(self._mtype)

        if backend_class is None:
            raise ValueError(
                f"No FuzzarrayBackend registered for mtype '{self._mtype}'. "
                f"Available mtypes: {list(registry.backends.keys())}"
            )

        return backend_class(shape=shape, **self._mtype_kwargs)

    def _populate_backend_from_data(self, data, copy):
        """从输入数据填充后端"""
        if isinstance(data, Fuzznum):
            # 单个 Fuzznum，填充到标量后端
            self._backend.set_fuzznum_data((), data)
        elif isinstance(data, (list, tuple, np.ndarray)):
            # 数组数据，需要遍历填充
            flat_data = np.array(data, dtype=object).flatten()

            for i, item in enumerate(flat_data):
                if not isinstance(item, Fuzznum):
                    raise TypeError(f"All elements must be Fuzznum objects, got {type(item)}")

                # 验证 mtype 一致性
                if item.mtype != self._mtype:
                    raise ValueError(
                        f"All elements must have the same mtype. "
                        f"Expected '{self._mtype}', got '{item.mtype}' at index {i}"
                    )

                # 验证 q 参数一致性
                expected_q = self._mtype_kwargs.get('q', 1)
                if item.q != expected_q:
                    raise ValueError(
                        f"All elements must have the same q parameter. "
                        f"Expected q={expected_q}, got q={item.q} at index {i}"
                    )

                # 计算多维索引
                multi_index = np.unravel_index(i, self._backend.shape)
                self._backend.set_fuzznum_data(multi_index, item)

    # ==================== 属性接口 ====================
    @property
    def shape(self) -> Tuple[int, ...]:
        """返回数组形状"""
        return self._backend.shape

    @property
    def ndim(self) -> int:
        """返回数组维度"""
        return len(self._backend.shape)

    @property
    def size(self) -> int:
        """返回数组大小"""
        return self._backend.size

    @property
    def T(self):
        # TODO: 实现转置功能
        return None

    @property
    def mtype(self) -> str:
        """返回模糊数类型"""
        return self._mtype

    @property
    def q(self) -> int:
        """返回 q 参数"""
        return self._mtype_kwargs.get('q', 1)

    @property
    def mtype_kwargs(self) -> dict:
        """返回 mtype 相关参数"""
        return self._mtype_kwargs.copy()

    @property
    def backend(self) -> FuzzarrayBackend:
        """返回底层后端（主要用于调试和高级操作）"""
        return self._backend

    # ==================== 索引和切片 ====================

    def __len__(self):
        """返回数组大小"""
        return self._backend.shape[0]

    def __getitem__(self, key) -> Union[Fuzznum, 'Fuzzarray']:
        """
        实现索引和切片操作。

        - 标量索引返回 Fuzznum
        - 切片索引返回新的 Fuzzarray
        """

        def _is_scalar_index(k) -> bool:
            if isinstance(key, (int, np.integer)):
                return True
            elif isinstance(key, tuple):
                # 检查是否所有元素都是整数
                return all(isinstance(k, (int, np.integer)) for k in key)
            return False

        if _is_scalar_index(key):
            # 标量索引：返回单个 Fuzznum 视图
            return self._backend.get_fuzznum_view(key)
        else:
            # 切片索引：返回新的 Fuzzarray
            sliced_backend = self._backend.slice_view(key)
            return Fuzzarray(backend=sliced_backend)

    def __setitem__(self, key, value):
        """设置数组元素"""
        if isinstance(value, Fuzznum):
            if value.mtype != self._mtype:
                raise ValueError(f"Mtype mismatch: expected '{self._mtype}', got '{value.mtype}'")
            if value.q != self.q:
                raise ValueError(f"Q parameter mismatch: expected q={self.q}, got q={value.q}")
            self._backend.set_fuzznum_data(key, value)
        elif isinstance(value, Fuzzarray):
            if value.mtype != self._mtype:
                raise ValueError(f"Mtype mismatch: expected '{self._mtype}', got '{value.mtype}'")
            if value.q != self.q:
                raise ValueError(f"Q parameter mismatch: expected q={self.q}, got q={value.q}")
            # 实现数组赋值逻辑
            self._assign_from_fuzzarray(key, value)
        else:
            raise TypeError(f"Can only assign Fuzznum or Fuzzarray objects, got {type(value)}")

    def _assign_from_fuzzarray(self, key, value_array):
        """从另一个 Fuzzarray 赋值"""
        # 这里需要实现复杂的数组赋值逻辑
        # 暂时使用简化版本
        raise NotImplementedError("Array assignment not yet implemented")

    def __delitem__(self, key):
        # TODO: 实现删除操作
        raise NotImplementedError("Fuzzarray does not support item deletion.")

    def __contains__(self, item) -> bool:
        """检查元素是否在数组中"""
        # TODO: 实现包含检查, 已实现代码是否合理?
        ...
        # if isinstance(item, Fuzznum):
        #     if item.mtype != self._mtype or item.q != self.q:
        #         return False
        #     # 检查是否存在于后端
        #     for idx in np.ndindex(self.shape):
        #         if self._backend.get_fuzznum_view(idx) == item:
        #             return True
        #     return False
        # return False

    # TODO: 实现迭代器
    def __iter__(self): ...

    # ==================== 复制操作 ====================

    def copy(self) -> 'Fuzzarray':
        """创建深拷贝"""
        copied_backend = self._backend.copy()
        return Fuzzarray(backend=copied_backend)

    # ==================== 运算符重载 ====================
    # TODO: 实现运算符重载
    def __add__(self, other): ...
    def __sub__(self, other): ...
    def __mul__(self, other): ...
    def __truediv__(self, other): ...
    def __pow__(self, other): ...
    def __gt__(self, other: Any) -> np.ndarray: ...
    def __lt__(self, other: Any) -> np.ndarray: ...
    def __ge__(self, other: Any) -> np.ndarray: ...
    def __le__(self, other: Any) -> np.ndarray: ...
    def __eq__(self, other: Any) -> np.ndarray: ...
    def __ne__(self, other: Any) -> np.ndarray: ...
    def __and__(self, other): ...
    def __or__(self, other): ...
    def __invert__(self, other=None): ...
    def __lshift__(self, other): ...
    def __rshift__(self, other): ...
    def __xor__(self, other): ...
    def equivalent(self, other): ...
    def __matmul__(self, other): ...

    # ==================== 字符串表示 ====================

    def __repr__(self) -> str:
        """字符串表示"""
        if self.size == 0:
            return f"Fuzzarray([], mtype='{self.mtype}', shape={self.shape})"

        # 对于小数组，展示部分内容
        # if self.size <= 8:
        elements = []
        for idx in np.ndindex(self.shape):
            fuzznum = self._backend.get_fuzznum_view(idx)
            elements.append(str(fuzznum))
        content = ', '.join(elements)
        # else:
        #     content = f"... {self.size} elements ..."

        return f"Fuzzarray([{content}], mtype='{self.mtype}', q={self.q}, shape={self.shape})"

    def __str__(self) -> str:
        """用户友好的字符串表示"""
        return self.__repr__()

    # TODO: 实现特殊方法
    def __bool__(self) -> bool: ...
    def __format__(self, format_spec: str) -> Any: ...
    def __getstate__(self) -> Any: ...
    def __setstate__(self, state: Any): ...


# ================================= 工厂函数 =================================

def fuzzarray(data: Union[np.ndarray, list, tuple, Fuzznum],
              mtype: Optional[str] = None,
              shape: Optional[Tuple[int, ...]] = None,
              copy: bool = True,
              **mtype_kwargs) -> Fuzzarray:
    """
    Factory function to create Fuzzarray objects.

    Args:
        data: Input data
        mtype: Fuzzy number type
        shape: Array shape
        copy: Whether to copy data
        **mtype_kwargs: Mtype-specific parameters (e.g., q=2)

    Returns:
        A new Fuzzarray instance
    """
    return Fuzzarray(data=data, mtype=mtype, shape=shape, copy=copy, **mtype_kwargs)
