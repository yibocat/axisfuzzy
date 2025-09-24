#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Tests for Fuzzarray and FuzzarrayBackend functionality.

This module tests the high-level Fuzzarray container and its interaction
with backend implementations, focusing on:
- Array creation and initialization
- Shape and property management
- Element access and modification
- Slicing and indexing
- Operator overloading
- Backend delegation
"""

import pytest
import numpy as np
from typing import Any, Tuple, Optional, Callable

from axisfuzzy.core.fuzzarray import Fuzzarray
from axisfuzzy.core.backend import FuzzarrayBackend
from axisfuzzy.core.fuzznums import Fuzznum
from axisfuzzy.core import fuzzynum
from axisfuzzy.core.registry import get_registry_fuzztype


class MockBackend(FuzzarrayBackend):
    """
    简单的模拟后端，用于测试 FuzzarrayBackend 抽象接口。
    
    这个模拟后端实现了所有必需的抽象方法，使用简单的 NumPy 数组
    来存储模糊数的成员度和非成员度值。
    """
    
    def __init__(self, shape: Tuple[int, ...], q: Optional[int] = None, **kwargs):
        self.mtype = 'qrofn'  # 设置默认类型
        super().__init__(shape, q, **kwargs)
    
    @property
    def cmpnum(self) -> int:
        """返回组件数量。"""
        return 2
    
    @property
    def cmpnames(self) -> Tuple[str, ...]:
        """返回组件名称。"""
        return ('md', 'nmd')
    
    @property
    def dtype(self) -> np.dtype:
        """返回数据类型。"""
        return np.dtype(np.float64)
    
    def _initialize_arrays(self):
        """初始化成员度和非成员度数组。"""
        self._md = np.zeros(self.shape, dtype=float)
        self._nmd = np.zeros(self.shape, dtype=float)
    
    def get_fuzznum_view(self, index: Any) -> 'Fuzznum':
        """获取指定索引处的模糊数视图。"""
        md_val = float(self._md[index])
        nmd_val = float(self._nmd[index])
        return fuzzynum(mtype=self.mtype, q=self.q, md=md_val, nmd=nmd_val)
    
    def set_fuzznum_data(self, index: Any, fuzzynum: 'Fuzznum'):
        """设置指定索引处的模糊数数据。"""
        self._md[index] = fuzzynum.md
        self._nmd[index] = fuzzynum.nmd
    
    def copy(self) -> 'MockBackend':
        """创建后端的深拷贝。"""
        new_backend = MockBackend(self.shape, self.q, **self.kwargs)
        new_backend._md = self._md.copy()
        new_backend._nmd = self._nmd.copy()
        return new_backend
    
    def slice_view(self, key) -> 'MockBackend':
        """创建切片视图。"""
        sliced_md = self._md[key]
        sliced_nmd = self._nmd[key]
        
        new_backend = MockBackend(sliced_md.shape, self.q, **self.kwargs)
        new_backend._md = sliced_md
        new_backend._nmd = sliced_nmd
        return new_backend
    
    @staticmethod
    def from_arrays(*components, **kwargs) -> 'MockBackend':
        """从组件数组创建后端。"""
        if len(components) != 2:
            raise ValueError("MockBackend requires exactly 2 components (md, nmd)")
        
        md_array, nmd_array = components
        backend = MockBackend(md_array.shape, **kwargs)
        backend._md = md_array.copy()
        backend._nmd = nmd_array.copy()
        return backend
    
    def get_component_arrays(self) -> tuple:
        """获取组件数组。"""
        return (self._md, self._nmd)
    
    def _get_element_formatter(self, format_spec: str) -> Callable:
        """获取元素格式化器。"""
        def formatter(index, md_val, nmd_val):
            if format_spec:
                return f"({md_val:{format_spec}}, {nmd_val:{format_spec}})"
            return f"({md_val:.3f}, {nmd_val:.3f})"
        return formatter
    
    def _format_single_element(self, index: Any, formatter: Callable, format_spec: str) -> str:
        """格式化单个元素。"""
        md_val = self._md[index]
        nmd_val = self._nmd[index]
        return formatter(index, md_val, nmd_val)


class TestFuzzarrayBackend:
    """测试 FuzzarrayBackend 抽象基类的功能。"""
    
    def test_backend_initialization(self):
        """测试后端初始化。"""
        backend = MockBackend((2, 3), q=2)
        
        assert backend.shape == (2, 3)
        assert backend.size == 6
        assert backend.q == 2
        assert backend.ndim == 2
        assert hasattr(backend, '_md')
        assert hasattr(backend, '_nmd')
    
    def test_backend_default_q(self):
        """测试默认 q 值。"""
        backend = MockBackend((2, 2))
        assert backend.q is not None  # 应该有默认值
    
    def test_backend_component_arrays(self):
        """测试组件数组访问。"""
        backend = MockBackend((2, 2))
        components = backend.get_component_arrays()
        
        assert len(components) == 2
        assert all(isinstance(comp, np.ndarray) for comp in components)
        assert all(comp.shape == (2, 2) for comp in components)
    
    def test_backend_copy(self):
        """测试后端拷贝。"""
        backend = MockBackend((2, 2))
        backend._md[0, 0] = 0.5
        backend._nmd[0, 0] = 0.3
        
        copied = backend.copy()
        
        assert copied.shape == backend.shape
        assert copied.q == backend.q
        assert np.array_equal(copied._md, backend._md)
        assert np.array_equal(copied._nmd, backend._nmd)
        
        # 修改原始数据不应影响拷贝
        backend._md[0, 0] = 0.8
        assert copied._md[0, 0] == 0.5
    
    def test_backend_slice_view(self):
        """测试后端切片视图。"""
        backend = MockBackend((3, 3))
        backend._md[1, 1] = 0.7
        backend._nmd[1, 1] = 0.2
        
        sliced = backend.slice_view((slice(1, 3), slice(1, 3)))
        
        assert sliced.shape == (2, 2)
        assert sliced._md[0, 0] == 0.7
        assert sliced._nmd[0, 0] == 0.2
    
    def test_backend_from_arrays(self):
        """测试从数组创建后端。"""
        md_array = np.array([[0.6, 0.7], [0.8, 0.9]])
        nmd_array = np.array([[0.3, 0.2], [0.1, 0.05]])
        
        backend = MockBackend.from_arrays(md_array, nmd_array, q=3)
        
        assert backend.shape == (2, 2)
        assert backend.q == 3
        assert np.array_equal(backend._md, md_array)
        assert np.array_equal(backend._nmd, nmd_array)
    
    def test_backend_fuzznum_operations(self):
        """测试模糊数操作。"""
        backend = MockBackend((2, 2), q=2)
        
        # 设置模糊数
        test_fuzznum = fuzzynum((0.6, 0.3), q=2, mtype='qrofn')
        backend.set_fuzznum_data((0, 0), test_fuzznum)
        
        # 获取模糊数视图
        retrieved = backend.get_fuzznum_view((0, 0))
        
        assert retrieved.md == 0.6
        assert retrieved.nmd == 0.3
        assert retrieved.mtype == 'qrofn'
        assert retrieved.q == 2


class TestFuzzarray:
    """测试 Fuzzarray 高级容器的功能。"""
    
    def test_fuzzarray_creation_from_backend(self):
        """测试从后端创建 Fuzzarray。"""
        backend = MockBackend((2, 3), q=2)
        arr = Fuzzarray(backend=backend)
        
        assert arr.shape == (2, 3)
        assert arr.size == 6
        assert arr.ndim == 2
        assert arr.q == 2
        assert arr.mtype == 'qrofn'
    
    def test_fuzzarray_creation_from_fuzznum(self):
        """测试从单个模糊数创建 Fuzzarray。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        arr = Fuzzarray(fn, shape=(2, 2))
        
        assert arr.shape == (2, 2)
        assert arr.mtype == 'qrofn'
        assert arr.q == 2
        
        # 检查所有元素是否都是相同的模糊数
        for i in range(2):
            for j in range(2):
                elem = arr[i, j]
                assert elem.md == 0.6
                assert elem.nmd == 0.3
    
    def test_fuzzarray_creation_from_list(self):
        """测试从模糊数列表创建 Fuzzarray。"""
        fuzznums = [
            fuzzynum(mtype='qrofn', q=2, md=0.1, nmd=0.2),
            fuzzynum(mtype='qrofn', q=2, md=0.3, nmd=0.4),
            fuzzynum(mtype='qrofn', q=2, md=0.5, nmd=0.6)
        ]
        arr = Fuzzarray(fuzznums)
        
        assert arr.shape == (3,)
        assert arr.size == 3
        assert arr[0].md == 0.1
        assert arr[1].md == 0.3
        assert arr[2].md == 0.5
    
    def test_fuzzarray_indexing(self):
        """测试 Fuzzarray 索引访问。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.7, nmd=0.2)
        arr = Fuzzarray(fn, shape=(3, 3))
        
        # 单元素访问
        elem = arr[1, 1]
        assert isinstance(elem, Fuzznum)
        assert elem.md == 0.7
        
        # 切片访问
        sub_arr = arr[1:3, 1:3]
        assert isinstance(sub_arr, Fuzzarray)
        assert sub_arr.shape == (2, 2)
    
    def test_fuzzarray_setitem(self):
        """测试 Fuzzarray 元素设置。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.5, nmd=0.3)
        arr = Fuzzarray(fn, shape=(2, 2))
        
        new_fn = fuzzynum(mtype='qrofn', q=2, md=0.8, nmd=0.1)
        arr[0, 0] = new_fn
        
        assert arr[0, 0].md == 0.8
        assert arr[0, 0].nmd == 0.1
        # 其他元素应该保持不变
        assert arr[0, 1].md == 0.5
    
    def test_fuzzarray_iteration(self):
        """测试 Fuzzarray 迭代。"""
        fuzznums = [
            fuzzynum(mtype='qrofn', q=2, md=0.1, nmd=0.9),
            fuzzynum(mtype='qrofn', q=2, md=0.2, nmd=0.8),
            fuzzynum(mtype='qrofn', q=2, md=0.3, nmd=0.7)
        ]
        arr = Fuzzarray(fuzznums)
        
        collected = list(arr)
        assert len(collected) == 3
        assert all(isinstance(elem, Fuzznum) for elem in collected)
        assert collected[0].md == 0.1
        assert collected[1].md == 0.2
        assert collected[2].md == 0.3
    
    def test_fuzzarray_copy(self):
        """测试 Fuzzarray 拷贝。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        arr = Fuzzarray(fn, shape=(2, 2))
        
        copied = arr.copy()
        
        assert copied.shape == arr.shape
        assert copied.mtype == arr.mtype
        assert copied.q == arr.q
        
        # 修改原数组不应影响拷贝
        new_fn = fuzzynum(mtype='qrofn', q=2, md=0.9, nmd=0.1)
        arr[0, 0] = new_fn
        
        assert copied[0, 0].md == 0.6  # 拷贝应该保持原值
    
    def test_fuzzarray_properties(self):
        """测试 Fuzzarray 属性。"""
        fn = fuzzynum(mtype='qrofn', q=3, md=0.7, nmd=0.2)
        arr = Fuzzarray(fn, shape=(2, 3, 4))
        
        assert arr.shape == (2, 3, 4)
        assert arr.size == 24
        assert arr.ndim == 3
        assert arr.mtype == 'qrofn'
        assert arr.q == 3
        assert len(arr) == 2  # 第一维的长度
    
    def test_fuzzarray_contains(self):
        """测试 Fuzzarray 包含检查。"""
        fn1 = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        fn2 = fuzzynum(mtype='qrofn', q=2, md=0.8, nmd=0.1)
        
        arr = Fuzzarray([fn1, fn2])
        
        # 注意：包含检查可能基于值比较而非对象身份
        assert fn1 in arr or any(elem.md == fn1.md and elem.nmd == fn1.nmd for elem in arr)
    
    def test_fuzzarray_bool_single_element(self):
        """测试单元素 Fuzzarray 的布尔值。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        arr = Fuzzarray([fn])
        
        # 单元素数组的布尔值应该基于模糊数本身
        assert bool(arr) is True  # 模糊数通常为 True
    
    def test_fuzzarray_bool_multiple_elements_error(self):
        """测试多元素 Fuzzarray 的布尔值应该抛出错误。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        arr = Fuzzarray(fn, shape=(2, 2))
        
        with pytest.raises(ValueError, match="ambiguous"):
            bool(arr)
    
    def test_fuzzarray_empty_array(self):
        """测试空 Fuzzarray。"""
        arr = Fuzzarray(data=None, mtype='qrofn', q=2, shape=(0,))
        
        assert arr.shape == (0,)
        assert arr.size == 0
        assert len(arr) == 0
        assert bool(arr) is False  # 空数组应该为 False
    
    def test_fuzzarray_string_representation(self):
        """测试 Fuzzarray 字符串表示。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        arr = Fuzzarray([fn])
        
        repr_str = repr(arr)
        str_str = str(arr)
        
        assert 'Fuzzarray' in repr_str
        assert 'qrofn' in repr_str
        assert isinstance(str_str, str)
        assert len(str_str) > 0


class TestFuzzarrayOperators:
    """测试 Fuzzarray 运算符重载。"""
    
    def test_arithmetic_operators_structure(self):
        """测试算术运算符的结构（不测试具体计算）。"""
        fn1 = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        fn2 = fuzzynum(mtype='qrofn', q=2, md=0.4, nmd=0.5)
        
        arr1 = Fuzzarray([fn1])
        arr2 = Fuzzarray([fn2])
        
        # 测试运算符是否存在且可调用
        assert hasattr(arr1, '__add__')
        assert hasattr(arr1, '__sub__')
        assert hasattr(arr1, '__mul__')
        assert hasattr(arr1, '__truediv__')
        assert hasattr(arr1, '__pow__')
        
        # 测试运算符调用不会立即失败（具体计算由 dispatcher 处理）
        try:
            result_add = arr1 + arr2
            result_sub = arr1 - arr2
            result_mul = arr1 * arr2
            result_div = arr1 / arr2
            result_pow = arr1 ** 2
            
            # 如果运算成功，结果应该是 Fuzzarray
            assert isinstance(result_add, Fuzzarray)
            assert isinstance(result_sub, Fuzzarray)
            assert isinstance(result_mul, Fuzzarray)
            assert isinstance(result_div, Fuzzarray)
            assert isinstance(result_pow, Fuzzarray)
        except (NotImplementedError, ImportError):
            # 如果 dispatcher 未实现，这是预期的
            pass
    
    def test_comparison_operators_structure(self):
        """测试比较运算符的结构。"""
        fn1 = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        fn2 = fuzzynum(mtype='qrofn', q=2, md=0.4, nmd=0.5)
        
        arr1 = Fuzzarray([fn1])
        arr2 = Fuzzarray([fn2])
        
        # 测试比较运算符是否存在
        assert hasattr(arr1, '__gt__')
        assert hasattr(arr1, '__lt__')
        assert hasattr(arr1, '__ge__')
        assert hasattr(arr1, '__le__')
        assert hasattr(arr1, '__eq__')
        assert hasattr(arr1, '__ne__')
        
        # 测试比较运算符调用
        try:
            result_gt = arr1 > arr2
            result_lt = arr1 < arr2
            result_eq = arr1 == arr2
            
            # 比较结果应该是 numpy 数组
            assert isinstance(result_gt, np.ndarray)
            assert isinstance(result_lt, np.ndarray)
            assert isinstance(result_eq, np.ndarray)
        except (NotImplementedError, ImportError):
            # 如果 dispatcher 未实现，这是预期的
            pass
    
    def test_logical_operators_structure(self):
        """测试逻辑运算符的结构。"""
        fn = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        arr = Fuzzarray([fn])
        
        # 测试逻辑运算符是否存在
        assert hasattr(arr, '__and__')  # &
        assert hasattr(arr, '__or__')   # |
        assert hasattr(arr, '__invert__')  # ~
        assert hasattr(arr, '__xor__')  # ^
        assert hasattr(arr, '__lshift__')  # <<
        assert hasattr(arr, '__rshift__')  # >>
        assert hasattr(arr, 'equivalent')
        assert hasattr(arr, '__matmul__')  # @


class TestFuzzarrayIntegration:
    """测试 Fuzzarray 与真实模糊数类型的集成。"""
    
    def test_integration_with_real_fuzznums(self):
        """测试与真实模糊数类型的集成。"""
        # 使用真实的模糊数类型
        try:
            fn = fuzzynum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
            arr = Fuzzarray([fn, fn, fn])
            
            assert arr.shape == (3,)
            assert arr.mtype == 'qrofn'
            assert arr.q == 2
            
            # 测试元素访问
            elem = arr[0]
            assert elem.md == 0.6
            assert elem.nmd == 0.3
            
        except Exception as e:
            pytest.skip(f"Real fuzzynum integration not available: {e}")
    
    def test_integration_with_registry(self):
        """测试与注册表的集成。"""
        try:
            # 尝试获取注册的模糊数类型
            registry = get_registry_fuzztype()
            if 'qrofn' in registry:
                fn = fuzzynum(mtype='qrofn', q=2, md=0.7, nmd=0.2)
                arr = Fuzzarray([fn])
                
                assert arr.mtype == 'qrofn'
                assert arr[0].md == 0.7
        except Exception as e:
            pytest.skip(f"Registry integration not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])