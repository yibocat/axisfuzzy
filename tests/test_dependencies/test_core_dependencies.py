#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心依赖测试模块

测试 AxisFuzzy 核心依赖包的安装和基本功能，包括：
- numpy: 数值计算基础库
- numba: JIT 编译加速库

Author: AxisFuzzy Team
Date: 2025-01-25
"""

import sys
import pytest
import importlib.util
from packaging import version


class TestCoreDependencies:
    """核心依赖测试类"""
    
    def test_numpy_availability(self):
        """测试 numpy 是否可用"""
        try:
            import numpy as np
            assert np is not None
            print(f"✅ numpy 版本: {np.__version__}")
        except ImportError as e:
            pytest.fail(f"❌ numpy 导入失败: {e}")
    
    def test_numpy_version(self):
        """测试 numpy 版本是否满足要求"""
        try:
            import numpy as np
            required_version = "2.2.6"
            current_version = np.__version__
            
            assert version.parse(current_version) >= version.parse(required_version), \
                f"numpy 版本过低: {current_version} < {required_version}"
            print(f"✅ numpy 版本检查通过: {current_version} >= {required_version}")
        except ImportError:
            pytest.fail("❌ numpy 未安装")
    
    def test_numpy_basic_functionality(self):
        """测试 numpy 基本功能"""
        try:
            import numpy as np
            
            # 测试数组创建
            arr = np.array([1, 2, 3, 4, 5])
            assert arr.shape == (5,)
            
            # 测试基本运算
            result = np.sum(arr)
            assert result == 15
            
            # 测试数据类型
            float_arr = np.array([1.0, 2.0, 3.0])
            assert float_arr.dtype == np.float64
            
            print("✅ numpy 基本功能测试通过")
        except Exception as e:
            pytest.fail(f"❌ numpy 基本功能测试失败: {e}")
    
    def test_numba_availability(self):
        """测试 numba 是否可用"""
        try:
            import numba
            assert numba is not None
            print(f"✅ numba 版本: {numba.__version__}")
        except ImportError as e:
            pytest.fail(f"❌ numba 导入失败: {e}")
    
    def test_numba_version(self):
        """测试 numba 版本是否满足要求"""
        try:
            import numba
            required_version = "0.61.2"
            current_version = numba.__version__
            
            assert version.parse(current_version) >= version.parse(required_version), \
                f"numba 版本过低: {current_version} < {required_version}"
            print(f"✅ numba 版本检查通过: {current_version} >= {required_version}")
        except ImportError:
            pytest.fail("❌ numba 未安装")
    
    def test_numba_jit_functionality(self):
        """测试 numba JIT 编译功能"""
        try:
            import numba
            import numpy as np
            
            @numba.jit
            def simple_function(x):
                return x * 2 + 1
            
            # 测试 JIT 编译函数
            result = simple_function(5)
            assert result == 11
            
            # 测试数组操作
            @numba.jit
            def array_sum(arr):
                total = 0
                for i in range(len(arr)):
                    total += arr[i]
                return total
            
            test_arr = np.array([1, 2, 3, 4, 5])
            result = array_sum(test_arr)
            assert result == 15
            
            print("✅ numba JIT 功能测试通过")
        except Exception as e:
            pytest.fail(f"❌ numba JIT 功能测试失败: {e}")
    
    def test_numpy_numba_integration(self):
        """测试 numpy 和 numba 的集成"""
        try:
            import numpy as np
            import numba
            
            @numba.jit
            def simple_matrix_operation(a, b):
                # 使用简单的元素级操作，避免依赖 scipy
                result = np.zeros_like(a)
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        result[i, j] = a[i, j] + b[i, j]
                return result
            
            # 创建测试矩阵
            a = np.array([[1, 2], [3, 4]])
            b = np.array([[5, 6], [7, 8]])
            
            # 测试矩阵操作
            result = simple_matrix_operation(a, b)
            expected = np.array([[6, 8], [10, 12]])
            
            np.testing.assert_array_equal(result, expected)
            print("✅ numpy-numba 集成测试通过")
        except Exception as e:
            pytest.fail(f"❌ numpy-numba 集成测试失败: {e}")
    
    def test_core_dependencies_summary(self):
        """生成核心依赖测试总结"""
        try:
            import numpy as np
            import numba
            
            print("\n=== 核心依赖测试总结 ===")
            print(f"numpy: {np.__version__} ✅")
            print(f"numba: {numba.__version__} ✅")
            print("所有核心依赖测试通过！")
            
        except ImportError as e:
            pytest.fail(f"❌ 核心依赖缺失: {e}")


def test_core_requirements_file():
    """测试核心依赖文件是否存在"""
    import os
    
    requirements_file = "/Users/yibow/Documents/Fuzzy/AxisFuzzy/requirements/core_requirements.txt"
    assert os.path.exists(requirements_file), "核心依赖文件不存在"
    
    with open(requirements_file, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "numpy" in content, "核心依赖文件中缺少 numpy"
        assert "numba" in content, "核心依赖文件中缺少 numba"
    
    print("✅ 核心依赖文件检查通过")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])