#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/24 下午3:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
测试隶属函数的可视化功能

本模块测试以下可视化功能：
- 基本绘图功能
- 绘图参数和自定义选项
- 多函数对比绘图
- 特殊情况下的绘图稳定性
- 绘图输出的正确性验证

可视化功能对于理解和调试隶属函数至关重要，
确保其正确性和稳定性是测试的重要组成部分。
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from unittest.mock import patch, MagicMock
import io
import sys

from axisfuzzy.membership.function import (
    TriangularMF, TrapezoidalMF, GaussianMF, SigmoidMF,
    SMF, ZMF, PiMF, GeneralizedBellMF, DoubleGaussianMF
)
from axisfuzzy.membership.factory import create_mf
from .conftest import TOLERANCE


class TestBasicPlotting:
    """测试基本绘图功能"""
    
    def test_plot_method_exists(self):
        """测试所有隶属函数都有plot方法"""
        functions = [
            TriangularMF(0, 0.5, 1),
            TrapezoidalMF(0, 0.2, 0.8, 1),
            GaussianMF(sigma=0.2, c=0.5),
            SigmoidMF(k=2, c=0.5),
            SMF(a=0, b=1),
            ZMF(a=0, b=1),
            PiMF(a=0, b=0.2, c=0.8, d=1),
            GeneralizedBellMF(a=0.2, b=2, c=0.5),
            DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.1, c2=0.7)
        ]
        
        for mf in functions:
            assert hasattr(mf, 'plot')
            assert callable(mf.plot)
    
    def test_basic_plot_execution(self):
        """测试基本绘图执行不出错"""
        mf = TriangularMF(0, 0.5, 1)
        
        # 应该能够正常执行而不抛出异常
        try:
            mf.plot()
        except Exception as e:
            pytest.fail(f"Basic plot execution failed: {e}")
        finally:
            plt.close('all')  # 清理图形
    
    def test_plot_with_custom_range(self):
        """测试自定义范围绘图"""
        mf = TriangularMF(0, 0.5, 1)
        
        try:
            # 测试自定义x范围
            mf.plot(x_range=(-0.5, 1.5))
        except Exception as e:
            pytest.fail(f"Plot with custom range failed: {e}")
        finally:
            plt.close('all')
    

    
    def test_plot_with_custom_points(self):
        """测试自定义点数绘图"""
        mf = GaussianMF(sigma=0.2, c=0.5)
        
        try:
            # 测试自定义点数
            mf.plot(num_points=50)
        except Exception as e:
            pytest.fail(f"Plot with custom points failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_data_correctness(self):
        """测试绘图数据的正确性"""
        mf = TriangularMF(0, 0.5, 1)
        
        # 手动计算一些点的值
        test_x = np.array([0, 0.25, 0.5, 0.75, 1])
        expected_y = mf.compute(test_x)
        
        try:
            # 绘图后检查数据
            mf.plot(x_range=(0, 1), num_points=101)
            
            # 获取当前图形的数据
            fig = plt.gcf()
            ax = fig.gca()
            lines = ax.get_lines()
            
            if lines:
                plot_x, plot_y = lines[0].get_data()
                
                # 检查绘图范围
                assert plot_x[0] == pytest.approx(0, abs=TOLERANCE)
                assert plot_x[-1] == pytest.approx(1, abs=TOLERANCE)
                
                # 检查峰值点
                max_idx = np.argmax(plot_y)
                assert plot_x[max_idx] == pytest.approx(0.5, abs=1e-2)
                assert plot_y[max_idx] == pytest.approx(1.0, abs=TOLERANCE)
        except Exception as e:
            pytest.fail(f"Plot data correctness test failed: {e}")
        finally:
            plt.close('all')


class TestPlotCustomization:
    """测试绘图自定义选项"""
    
    def test_plot_with_different_ranges(self):
        """测试不同的绘图范围"""
        mf = TriangularMF(0, 0.5, 1)
        
        ranges_to_test = [
            (-1, 2),
            (0, 1),
            (0.2, 0.8),
            (-0.5, 1.5)
        ]
        
        for x_range in ranges_to_test:
            try:
                mf.plot(x_range=x_range)
            except Exception as e:
                pytest.fail(f"Plot with range {x_range} failed: {e}")
            finally:
                plt.close('all')
    
    def test_plot_with_different_point_counts(self):
        """测试不同的点数设置"""
        mf = GaussianMF(sigma=0.2, c=0.5)
        
        point_counts = [10, 50, 100, 500, 1000]
        
        for num_points in point_counts:
            try:
                mf.plot(num_points=num_points)
            except Exception as e:
                pytest.fail(f"Plot with {num_points} points failed: {e}")
            finally:
                plt.close('all')


class TestMultipleFunctionPlotting:
    """测试多函数绘图"""
    
    def test_plot_multiple_functions_same_axes(self):
        """测试在同一坐标轴上绘制多个函数"""
        mf1 = TriangularMF(0, 0.3, 0.6)
        mf2 = TriangularMF(0.4, 0.7, 1.0)
        
        try:
            # 绘制第一个函数
            mf1.plot()
            
            # 在同一图上绘制第二个函数
            mf2.plot()
            
            # 检查是否有两条线
            fig = plt.gcf()
            ax = fig.gca()
            lines = ax.get_lines()
            assert len(lines) >= 2
            
        except Exception as e:
            pytest.fail(f"Multiple function plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_function_comparison(self):
        """测试函数对比绘图"""
        functions = [
            TriangularMF(0, 0.5, 1),
            GaussianMF(sigma=0.2, c=0.5),
            SigmoidMF(k=5, c=0.5)
        ]
        
        try:
            for mf in functions:
                mf.plot()
            
            # 检查图形中的线条数量
            fig = plt.gcf()
            ax = fig.gca()
            lines = ax.get_lines()
            assert len(lines) >= len(functions)
            
        except Exception as e:
            pytest.fail(f"Function comparison plotting failed: {e}")
        finally:
            plt.close('all')


class TestSpecialCasePlotting:
    """测试特殊情况下的绘图"""
    
    def test_plot_degenerate_triangle(self):
        """测试退化三角形的绘图"""
        # 退化为点的三角形
        mf = TriangularMF(0.5, 0.5, 0.5)
        
        try:
            mf.plot()
        except Exception as e:
            pytest.fail(f"Degenerate triangle plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_extreme_gaussian(self):
        """测试极端参数的高斯函数绘图"""
        # 极小的sigma
        mf = GaussianMF(sigma=0.001, c=0.5)
        
        try:
            mf.plot()
        except Exception as e:
            pytest.fail(f"Extreme Gaussian plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_steep_sigmoid(self):
        """测试陡峭的Sigmoid函数绘图"""
        # 极大的斜率
        mf = SigmoidMF(k=100, c=0.5)
        
        try:
            mf.plot()
        except Exception as e:
            pytest.fail(f"Steep Sigmoid plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_double_gaussian_peaks(self):
        """测试双峰高斯函数的绘图"""
        mf = DoubleGaussianMF(sigma1=0.1, c1=0.3, sigma2=0.1, c2=0.7)
        
        try:
            mf.plot()
            
            # 检查是否正确绘制了双峰
            fig = plt.gcf()
            ax = fig.gca()
            lines = ax.get_lines()
            
            if lines:
                plot_x, plot_y = lines[0].get_data()
                # 应该有两个局部最大值
                # 这里只检查绘图不出错
                assert len(plot_x) > 0
                assert len(plot_y) > 0
        except Exception as e:
            pytest.fail(f"Double Gaussian plotting failed: {e}")
        finally:
            plt.close('all')


class TestPlotErrorHandling:
    """测试绘图错误处理"""
    
    def test_plot_with_invalid_range(self):
        """测试无效范围的处理"""
        mf = TriangularMF(0, 0.5, 1)
        
        try:
            # 测试无效范围（最小值大于最大值）
            # plot方法应该能处理这种情况或使用默认值
            mf.plot(x_range=(1, 0))
        except Exception as e:
            # 如果抛出异常也是可接受的
            assert isinstance(e, (ValueError, TypeError))
        finally:
            plt.close('all')
    
    def test_plot_with_invalid_points(self):
        """测试无效点数的处理"""
        mf = GaussianMF(sigma=0.2, c=0.5)
        
        try:
            # 测试负数点数 - plot方法应该能处理或使用默认值
            mf.plot(num_points=-10)
        except Exception as e:
            # 如果抛出异常也是可接受的
            assert isinstance(e, (ValueError, TypeError))
        finally:
            plt.close('all')
        
        try:
            # 测试零点数
            mf.plot(num_points=0)
        except Exception as e:
            # 如果抛出异常也是可接受的
            assert isinstance(e, (ValueError, TypeError))
        finally:
            plt.close('all')
    
    def test_plot_with_edge_case_parameters(self):
        """测试边界情况参数"""
        mf = TriangularMF(0, 0.5, 1)
        
        try:
            # 测试最小点数
            mf.plot(num_points=2)
        except Exception as e:
            pytest.fail(f"Plot with edge case parameters failed: {e}")
        finally:
            plt.close('all')


class TestPlotPerformance:
    """测试绘图性能"""
    
    def test_plot_large_point_count(self):
        """测试大点数绘图性能"""
        mf = GaussianMF(sigma=0.2, c=0.5)
        
        try:
            # 测试大点数（应该在合理时间内完成）
            mf.plot(num_points=10000)
        except Exception as e:
            pytest.fail(f"Large point count plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_multiple_functions_performance(self):
        """测试多函数绘图性能"""
        functions = [
            TriangularMF(0, 0.2, 0.4),
            TriangularMF(0.1, 0.3, 0.5),
            TriangularMF(0.2, 0.4, 0.6),
            TriangularMF(0.3, 0.5, 0.7),
            TriangularMF(0.4, 0.6, 0.8),
            TriangularMF(0.5, 0.7, 0.9),
            TriangularMF(0.6, 0.8, 1.0)
        ]
        
        try:
            for mf in functions:
                mf.plot()
        except Exception as e:
            pytest.fail(f"Multiple functions performance test failed: {e}")
        finally:
            plt.close('all')


class TestPlotIntegration:
    """测试绘图集成功能"""
    
    def test_plot_with_factory_created_functions(self):
        """测试工厂创建函数的绘图"""
        try:
            # 使用工厂函数创建隶属函数
            mf, _ = create_mf('trimf', a=0, b=0.5, c=1)
            
            # 测试绘图
            mf.plot()
        except Exception as e:
            pytest.fail(f"Factory created function plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_after_parameter_update(self):
        """测试参数更新后的绘图"""
        mf = TriangularMF(0, 0.5, 1)
        
        try:
            # 初始绘图
            mf.plot()
            
            # 更新参数
            mf.set_parameters(a=0.1, b=0.6, c=0.9)
            
            # 再次绘图
            mf.plot()
        except Exception as e:
            pytest.fail(f"Plot after parameter update failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_consistency_across_calls(self):
        """测试多次调用的一致性"""
        mf = GaussianMF(sigma=0.2, c=0.5)
        
        try:
            # 多次绘图调用
            for _ in range(3):
                mf.plot()
        except Exception as e:
            pytest.fail(f"Plot consistency test failed: {e}")
        finally:
            plt.close('all')


class TestPlotDocumentation:
    """测试绘图文档一致性"""
    
    def test_documented_plot_parameters(self):
        """测试文档中提到的参数"""
        mf = TriangularMF(0, 0.5, 1)
        
        # 测试文档中提到的参数组合
        documented_params = [
            {'x_range': (0, 1), 'num_points': 1000},  # 默认参数
            {'x_range': (-2, 3), 'num_points': 2000},  # 自定义范围和高分辨率
        ]
        
        for params in documented_params:
            try:
                mf.plot(**params)
            except Exception as e:
                pytest.fail(f"Documented parameter {params} failed: {e}")
            finally:
                plt.close('all')
    
    def test_plot_examples_from_docs(self):
        """测试文档示例"""
        try:
            # 文档示例1：默认设置绘图
            mf = TriangularMF(a=0, b=0.5, c=1)
            mf.plot()
            
            # 文档示例2：自定义范围和高分辨率
            mf.plot(x_range=(-2, 3), num_points=2000)
        except Exception as e:
            pytest.fail(f"Documentation examples failed: {e}")
        finally:
            plt.close('all')


def test_plot_in_headless_environment():
    """测试无头环境下的绘图"""
    # 设置matplotlib后端为Agg（无头模式）
    import matplotlib
    original_backend = matplotlib.get_backend()
    
    try:
        matplotlib.use('Agg')
        
        mf = TriangularMF(0, 0.5, 1)
        mf.plot()
        
    except Exception as e:
        pytest.fail(f"Headless environment plotting failed: {e}")
    finally:
        # 恢复原始后端
        matplotlib.use(original_backend)
        plt.close('all')


if __name__ == '__main__':
    pytest.main()
