#!/usr/bin/env python3
"""
Performance tests for AxisFuzzy extension system.

This module contains performance benchmarks for the extension system,
including method call overhead, registry lookup efficiency, and injection performance.
"""

import time
import unittest
import threading
import gc
import psutil
import os
from typing import List, Dict, Any
from unittest.mock import Mock

from axisfuzzy.extension.registry import get_registry_extension
from axisfuzzy.extension.dispatcher import get_extension_dispatcher
from axisfuzzy.extension.injector import get_extension_injector
from axisfuzzy.extension import extension


class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        gc.collect()  # Clean up before measurement
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time


class TestMethodCallOverhead(unittest.TestCase):
    """Test performance overhead of extension method calls."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = get_registry_extension()
        self.dispatcher = get_extension_dispatcher()
        
        # Create a simple test function
        def test_func(x):
            return x * 2
        
        # Register it manually for testing
        self.test_func = test_func
        self.method_proxy = self.dispatcher.create_instance_method('test_perf_func')
    
    def test_direct_function_call_baseline(self):
        """Baseline: direct function call performance."""
        iterations = 10000
        
        with PerformanceTimer() as timer:
            for _ in range(iterations):
                result = self.test_func(42)
        
        avg_time = timer.elapsed_time / iterations
        print(f"Direct call average time: {avg_time * 1e6:.2f} μs")
        
        # Performance target: < 1 μs per call
        self.assertLess(avg_time, 1e-6, "Direct function call should be < 1 μs")
    
    def test_registry_lookup_performance(self):
        """Test registry lookup performance."""
        registry = self.registry
        iterations = 1000
        
        with PerformanceTimer() as timer:
            for _ in range(iterations):
                # Test lookup of existing function
                func = registry.get_function('test_perf_func', 'default')
        
        avg_time = timer.elapsed_time / iterations
        print(f"Registry lookup average time: {avg_time * 1e6:.2f} μs")
        
        # Performance target: < 10 μs per lookup
        self.assertLess(avg_time, 10e-6, "Registry lookup should be < 10 μs")


class TestRegistrationPerformance(unittest.TestCase):
    """Test performance of function registration."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = get_registry_extension()
    
    def test_single_function_registration(self):
        """Test performance of registering a single function."""
        def test_func():
            return "test"
        
        # Use unique name with timestamp and random component
        import random
        unique_name = f'perf_test_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}'
        
        with PerformanceTimer() as timer:
            self.registry.register(
                name=unique_name,
                mtype='test_type',
                target_classes=['TestClass'],
                injection_type='instance_method'
            )(test_func)
        
        print(f"Single registration time: {timer.elapsed_time * 1000:.2f} ms")
        
        # Performance target: < 10 ms per registration
        self.assertLess(timer.elapsed_time, 0.01, "Registration should be < 10 ms")
    
    def test_batch_registration_performance(self):
        """Test performance of registering multiple functions."""
        function_counts = [10, 50, 100]
        
        for count in function_counts:
            import random
            base_timestamp = int(time.time() * 1000000)
            random_id = random.randint(10000, 99999)
            
            functions = []
            for i in range(count):
                def func():
                    return f"test_{i}"
                functions.append(func)
            
            with PerformanceTimer() as timer:
                for i, func in enumerate(functions):
                    unique_name = f'batch_test_{base_timestamp}_{random_id}_{i}'
                    self.registry.register(
                        name=unique_name,
                        mtype=f'batch_type_{random_id}',
                        target_classes=['TestClass'],
                        injection_type='instance_method'
                    )(func)
            
            avg_time = timer.elapsed_time / count
            print(f"Batch registration ({count} funcs) avg time: {avg_time * 1000:.2f} ms")
            
            # Performance target: < 5 ms per function in batch
            self.assertLess(avg_time, 0.005, f"Batch registration should be < 5 ms per function")


class TestInjectionPerformance(unittest.TestCase):
    """Test performance of extension injection."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = get_registry_extension()
        self.dispatcher = get_extension_dispatcher()
        self.injector = get_extension_injector()
    
    def test_single_class_injection_performance(self):
        """Test performance of injecting into a single class."""
        # Create a test class
        class TestClass:
            pass
        
        # Register a test function
        import random
        unique_name = f'inject_test_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}'
        
        @extension(name=unique_name, 
                  mtype='test_type',
                  target_classes=['TestClass'],
                  injection_type='instance_method')
        def test_method(self):
            return "injected"
        
        class_map = {'TestClass': TestClass}
        module_namespace = {}
        
        with PerformanceTimer() as timer:
            self.injector.inject_all(class_map, module_namespace)
        
        print(f"Single class injection time: {timer.elapsed_time * 1000:.2f} ms")
        
        # Performance target: < 50 ms for injection
        self.assertLess(timer.elapsed_time, 0.05, "Injection should be < 50 ms")
        
        # Verify injection worked
        instance = TestClass()
        self.assertTrue(hasattr(instance, unique_name), "Method should be injected")


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage of extension system."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = get_registry_extension()
        self.dispatcher = get_extension_dispatcher()
        self.injector = get_extension_injector()
    
    def test_registration_memory_usage(self):
        """Test memory usage during function registration."""
        import random
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        base_timestamp = int(time.time() * 1000000)
        random_id = random.randint(10000, 99999)
        
        # Register many functions
        function_count = 100
        for i in range(function_count):
            def test_func():
                return f"test_{i}"
            
            unique_name = f'memory_test_{base_timestamp}_{random_id}_{i}'
            self.registry.register(
                name=unique_name,
                mtype=f'memory_type_{random_id}',
                target_classes=['TestClass'],
                injection_type='instance_method'
            )(test_func)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase for {function_count} registrations: {memory_increase:.2f} MB")
        
        # Performance target: < 10 MB for 100 functions
        self.assertLess(memory_increase, 10, "Memory usage should be < 10 MB for 100 functions")


class TestConcurrentAccess(unittest.TestCase):
    """Test concurrent access performance."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = get_registry_extension()
        self.dispatcher = get_extension_dispatcher()
        self.results = []
        self.errors = []
    
    def worker_function(self, worker_id, iterations):
        """Worker function for concurrent testing."""
        try:
            import random
            base_timestamp = int(time.time() * 1000000)
            random_id = random.randint(10000, 99999)
            
            for i in range(iterations):
                # Register a function
                def test_func():
                    return f"worker_{worker_id}_iteration_{i}"
                
                unique_name = f'concurrent_test_{base_timestamp}_{worker_id}_{random_id}_{i}'
                self.registry.register(
                    name=unique_name,
                    mtype=f'concurrent_type_{worker_id}_{random_id}',
                    target_classes=['TestClass'],
                    injection_type='instance_method'
                )(test_func)
                
                # Lookup the function
                result = self.registry.get_function(unique_name, f'concurrent_type_{worker_id}_{random_id}')
                self.results.append(1 if result else 0)
        except Exception as e:
            self.errors.append(f"Worker {worker_id} error: {e}")
    
    def test_concurrent_registry_access(self):
        """Test concurrent access to registry."""
        thread_count = 5
        iterations_per_thread = 100
        threads = []
        
        with PerformanceTimer() as timer:
            # Start threads
            for i in range(thread_count):
                thread = threading.Thread(
                    target=self.worker_function,
                    args=(i, iterations_per_thread)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
        
        print(f"Concurrent access time ({thread_count} threads): {timer.elapsed_time * 1000:.2f} ms")
        print(f"Results: {len(self.results)}, Errors: {len(self.errors)}")
        
        # Verify no errors occurred
        self.assertEqual(len(self.errors), 0, f"Concurrent access errors: {self.errors}")
        
        # Performance target: < 1 second for concurrent access
        self.assertLess(timer.elapsed_time, 1.0, "Concurrent access should be < 1 second")


if __name__ == '__main__':
    unittest.main()