#  Copyright (c) yibocat 2025 All Rights Reserved
#  Author: yibocat
#  Date: 2025/1/25 下午3:45
#  Filename: test_performance.py
#  Last Modified: 2025/1/25 下午3:45

"""
Performance tests for the mixin system.

This module contains performance benchmarks for the mixin system components,
including registration, injection, and function call overhead measurements.
"""

import time
import pytest
from typing import List, Dict, Any

from axisfuzzy.core import Fuzznum, Fuzzarray
from axisfuzzy.mixin.registry import MixinFunctionRegistry
from axisfuzzy.mixin.factory import (
    _concat_factory, _stack_factory, _append_factory,
    _reshape_factory, _flatten_factory, _copy_factory
)


class TestMixinPerformance:
    """Performance tests for mixin system components."""
    
    def test_registry_registration_performance(self):
        """Test performance of function registration."""
        registry = MixinFunctionRegistry()
        
        # Measure time for registering many functions
        start_time = time.time()
        
        for i in range(1000):
            @registry.register(
                name=f'test_func_{i}',
                injection_type='top_level_function'
            )
            def test_function():
                return f"result_{i}"
        
        registration_time = time.time() - start_time
        
        # Should complete registration in reasonable time (< 1 second)
        assert registration_time < 1.0, f"Registration took too long: {registration_time:.3f}s"
        
        # Verify all functions were registered
        assert len(registry._functions) == 1000
    
    def test_registry_injection_performance(self):
        """Test performance of function injection."""
        registry = MixinFunctionRegistry()
        
        # Register multiple functions
        for i in range(100):
            @registry.register(
                name=f'test_func_{i}',
                injection_type='top_level_function'
            )
            def test_function():
                return f"result_{i}"
        
        # Measure injection time
        start_time = time.time()
        
        module_namespace = {}
        registry.build_and_inject({}, module_namespace)
        
        injection_time = time.time() - start_time
        
        # Should complete injection in reasonable time (< 0.1 seconds)
        assert injection_time < 0.1, f"Injection took too long: {injection_time:.3f}s"
        
        # Verify all functions were injected
        assert len(module_namespace) == 100
    
    def test_factory_function_performance(self):
        """Test performance of factory functions with large arrays."""
        # Create large test arrays using qrofn type
        from axisfuzzy.core.fuzznums import fuzznum
        
        # Create fuzznums for the arrays
        fuzz1 = fuzznum(mtype='qrofn', q=2).create(md=0.6, nmd=0.3)
        fuzz2 = fuzznum(mtype='qrofn', q=2).create(md=0.7, nmd=0.2)
        
        large_array1 = Fuzzarray([fuzz1] * 1000)  # Reduced size for performance
        large_array2 = Fuzzarray([fuzz2] * 1000)
        
        # Test concat performance
        start_time = time.time()
        result = _concat_factory(large_array1, large_array2)
        concat_time = time.time() - start_time
        
        assert concat_time < 0.1, f"Concat took too long: {concat_time:.3f}s"
        assert len(result) == 2000
        
        # Test stack performance
        start_time = time.time()
        result = _stack_factory(large_array1, large_array2)
        stack_time = time.time() - start_time
        
        assert stack_time < 0.1, f"Stack took too long: {stack_time:.3f}s"
        assert result.shape == (2, 1000)
    
    def test_repeated_function_calls_performance(self):
        """Test performance of repeated mixin function calls."""
        registry = MixinFunctionRegistry()
        
        # Register a simple function
        @registry.register(
            name='simple_add',
            injection_type='top_level_function'
        )
        def simple_add(a, b):
            return a + b
        
        # Inject into namespace
        module_namespace = {}
        registry.build_and_inject({}, module_namespace)
        
        # Measure repeated calls
        start_time = time.time()
        
        for i in range(10000):
            result = module_namespace['simple_add'](i, i + 1)
        
        call_time = time.time() - start_time
        
        # Should complete 10000 calls in reasonable time (< 0.1 seconds)
        assert call_time < 0.1, f"10000 calls took too long: {call_time:.3f}s"
    
    def test_memory_usage_with_large_registry(self):
        """Test memory efficiency with large number of registered functions."""
        import gc
        import sys
        
        # Force garbage collection and get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        registry = MixinFunctionRegistry()
        
        # Register many functions
        for i in range(1000):
            @registry.register(
                name=f'memory_test_{i}',
                injection_type='top_level_function'
            )
            def memory_test_function():
                return f"result_{i}"
        
        # Force garbage collection and check memory
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory increase should be reasonable (less than 10x the number of functions)
        object_increase = final_objects - initial_objects
        assert object_increase < 10000, f"Too many objects created: {object_increase}"
    
    def test_concurrent_registration_simulation(self):
        """Simulate concurrent registration patterns (single-threaded test)."""
        registry = MixinFunctionRegistry()
        
        # Simulate interleaved registration and injection
        start_time = time.time()
        
        for batch in range(10):
            # Register a batch of functions
            for i in range(10):
                func_name = f'batch_{batch}_func_{i}'
                
                @registry.register(
                    name=func_name,
                    injection_type='top_level_function'
                )
                def batch_function():
                    return f"batch_{batch}_result_{i}"
            
            # Inject current functions
            module_namespace = {}
            registry.build_and_inject({}, module_namespace)
            
            # Verify injection worked
            assert len(module_namespace) == (batch + 1) * 10
        
        total_time = time.time() - start_time
        
        # Should complete in reasonable time (< 0.5 seconds)
        assert total_time < 0.5, f"Concurrent simulation took too long: {total_time:.3f}s"
    
    def test_factory_with_many_small_arrays(self):
        """Test factory performance with many small arrays."""
        # Create many small arrays using qrofn type
        from axisfuzzy.core.fuzznums import fuzznum
        
        small_arrays = []
        for i in range(100):  # Reduced for performance
            fuzz = fuzznum(mtype='qrofn', q=2).create(md=0.1 + i*0.001, nmd=0.1)
            small_arrays.append(Fuzzarray([fuzz]))
        
        # Test concat performance with many arrays
        start_time = time.time()
        result = _concat_factory(small_arrays[0], *small_arrays[1:])
        concat_time = time.time() - start_time
        
        assert concat_time < 0.1, f"Concat many small arrays took too long: {concat_time:.3f}s"
        assert len(result) == 100
        
        # Test stack performance with many arrays
        start_time = time.time()
        result = _stack_factory(small_arrays[0], *small_arrays[1:])
        stack_time = time.time() - start_time
        
        assert stack_time < 0.1, f"Stack many small arrays took too long: {stack_time:.3f}s"
        assert result.shape == (100, 1)
    
    def test_registry_lookup_performance(self):
        """Test performance of function lookup in registry."""
        registry = MixinFunctionRegistry()
        
        # Register functions with predictable names
        for i in range(1000):
            @registry.register(
                name=f'lookup_test_{i:04d}',
                injection_type='top_level_function'
            )
            def lookup_function():
                return f"lookup_result_{i}"
        
        # Measure lookup performance
        start_time = time.time()
        
        # Perform many lookups
        for i in range(1000):
            func_name = f'lookup_test_{i:04d}'
            assert func_name in registry._functions
        
        lookup_time = time.time() - start_time
        
        # Should complete lookups quickly (< 0.01 seconds)
        assert lookup_time < 0.01, f"1000 lookups took too long: {lookup_time:.3f}s"
    
    def test_stress_test_large_scale_operations(self):
        """Stress test with large-scale operations."""
        registry = MixinFunctionRegistry()
        
        # Register a large number of functions
        num_functions = 5000
        
        start_time = time.time()
        
        for i in range(num_functions):
            @registry.register(
                name=f'stress_test_{i}',
                injection_type='top_level_function'
            )
            def stress_function():
                return f"stress_result_{i}"
        
        registration_time = time.time() - start_time
        
        # Inject all functions
        start_time = time.time()
        module_namespace = {}
        registry.build_and_inject({}, module_namespace)
        injection_time = time.time() - start_time
        
        # Call all functions
        start_time = time.time()
        results = []
        for func_name in module_namespace:
            results.append(module_namespace[func_name]())
        call_time = time.time() - start_time
        
        # Performance assertions (generous limits for stress test)
        assert registration_time < 5.0, f"Registration took too long: {registration_time:.3f}s"
        assert injection_time < 1.0, f"Injection took too long: {injection_time:.3f}s"
        assert call_time < 1.0, f"Function calls took too long: {call_time:.3f}s"
        
        # Verify correctness
        assert len(results) == num_functions
        assert len(module_namespace) == num_functions
    
    def test_factory_performance_comparison(self):
        """Compare performance of different factory functions."""
        # Create test data using qrofn type
        from axisfuzzy.core.fuzznums import fuzznum
        
        test_arrays = []
        for i in range(50):  # Reduced for performance
            fuzz = fuzznum(mtype='qrofn', q=2).create(md=0.1 + i*0.01, nmd=0.1)
            test_arrays.append(Fuzzarray([fuzz] * 50))
        
        # Measure concat performance
        start_time = time.time()
        concat_result = _concat_factory(test_arrays[0], *test_arrays[1:])
        concat_time = time.time() - start_time
        
        # Measure stack performance
        start_time = time.time()
        stack_result = _stack_factory(test_arrays[0], *test_arrays[1:])
        stack_time = time.time() - start_time
        
        # Both should be reasonably fast
        assert concat_time < 0.1, f"Concat took too long: {concat_time:.3f}s"
        assert stack_time < 0.1, f"Stack took too long: {stack_time:.3f}s"
        
        # Verify results are correct
        assert len(concat_result) == 2500  # 50 arrays * 50 elements each
        assert stack_result.shape == (50, 50)  # 50 arrays stacked
        
        # Performance comparison (stack is typically faster than concat)
        # This is informational, not a strict requirement
        print(f"\nPerformance comparison:")
        print(f"Concat time: {concat_time:.4f}s")
        print(f"Stack time: {stack_time:.4f}s")