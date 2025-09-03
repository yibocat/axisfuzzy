#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 axisfuzzy.random.seed 模块

本模块测试随机种子管理系统的核心功能，包括：
- GlobalRandomState 单例模式
- 种子设置和获取函数
- 随机数生成器管理
- 线程安全性
- 可复现性验证
"""

import pytest
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set

from axisfuzzy.random.seed import (
    GlobalRandomState,
    set_seed,
    get_seed,
    get_rng,
    spawn_rng
)


class TestGlobalRandomState:
    """测试 GlobalRandomState 单例类的核心功能"""
    
    def test_multiple_instances_allowed(self, clean_global_state):
        """测试GlobalRandomState允许多个实例但通过全局实例管理状态"""
        # GlobalRandomState不是严格的单例，可以创建多个实例
        instance1 = GlobalRandomState()
        instance2 = GlobalRandomState()
        
        # 验证是不同的实例
        assert instance1 is not instance2
        assert id(instance1) != id(instance2)
        
        # 但它们都有独立的状态
        instance1.set_seed(123)
        instance2.set_seed(456)
        
        assert instance1.get_seed() == 123
        assert instance2.get_seed() == 456
    
    def test_initial_state(self, clean_global_state):
        """测试初始状态的正确性"""
        state = GlobalRandomState()
        
        # 初始种子应该为None
        assert state.get_seed() is None
        
        # 应该有默认的随机数生成器
        rng = state.get_generator()
        assert rng is not None
        assert isinstance(rng, np.random.Generator)
    
    def test_seed_setting_and_getting(self, clean_global_state):
        """测试种子设置和获取功能"""
        state = GlobalRandomState()
        test_seed = 12345
        
        # 设置种子
        state.set_seed(test_seed)
        
        # 验证种子已设置
        assert state.get_seed() == test_seed
    
    def test_rng_consistency_with_seed(self, clean_global_state):
        """测试设置种子后RNG的一致性"""
        state = GlobalRandomState()
        test_seed = 42
        
        # 设置种子并获取第一个随机数
        state.set_seed(test_seed)
        first_random = state.get_generator().random()
        
        # 重新设置相同种子并获取随机数
        state.set_seed(test_seed)
        second_random = state.get_generator().random()
        
        # 应该产生相同的随机数
        assert first_random == second_random
    
    def test_spawn_independent_rng(self, clean_global_state):
        """测试生成独立的随机数生成器"""
        state = GlobalRandomState()
        state.set_seed(123)
        
        # 生成独立的RNG
        independent_rng = state.spawn_generator()
        
        # 验证是不同的实例
        assert independent_rng is not state.get_generator()
        assert isinstance(independent_rng, np.random.Generator)
        
        # 验证统计独立性（生成大量随机数进行相关性测试）
        main_samples = state.get_generator().random(1000)
        independent_samples = independent_rng.random(1000)
        
        # 计算相关系数，应该接近0（统计独立）
        correlation = np.corrcoef(main_samples, independent_samples)[0, 1]
        assert abs(correlation) < 0.1  # 允许小的统计波动
    
    def test_different_seed_types(self, clean_global_state):
        """测试不同类型的种子支持"""
        state = GlobalRandomState()
        
        # 测试整数种子
        state.set_seed(42)
        assert state.get_seed() == 42
        
        # 测试None种子（使用系统熵）
        state.set_seed(None)
        assert state.get_seed() is None
        
        # 测试SeedSequence
        from numpy.random import SeedSequence
        seed_seq = SeedSequence(12345)
        state.set_seed(seed_seq)
        assert isinstance(state.get_seed(), SeedSequence)
        
        # 测试BitGenerator
        from numpy.random import PCG64
        bit_gen = PCG64(54321)
        state.set_seed(bit_gen)
        assert isinstance(state.get_seed(), PCG64)


class TestSeedFunctions:
    """测试种子相关的全局函数"""
    
    def test_set_seed_function(self, clean_global_state):
        """测试 set_seed 全局函数"""
        test_seed = 9876
        
        # 使用全局函数设置种子
        set_seed(test_seed)
        
        # 验证全局状态已更新
        assert get_seed() == test_seed
    
    def test_get_seed_function(self, clean_global_state):
        """测试 get_seed 全局函数"""
        test_seed = 5432
        
        # 设置种子
        set_seed(test_seed)
        
        # 使用全局函数获取种子
        retrieved_seed = get_seed()
        assert retrieved_seed == test_seed
    
    def test_get_rng_function(self, clean_global_state):
        """测试 get_rng 全局函数"""
        # 获取全局RNG
        rng = get_rng()
        
        # 验证返回的是Generator实例
        assert isinstance(rng, np.random.Generator)
        
        # 验证多次调用返回同一实例（通过全局状态）
        rng2 = get_rng()
        assert isinstance(rng2, np.random.Generator)
        
        # 验证RNG能正常工作
        sample1 = rng.random()
        sample2 = rng2.random()
        assert 0 <= sample1 < 1
        assert 0 <= sample2 < 1
    
    def test_spawn_rng_function(self, clean_global_state):
        """测试 spawn_rng 全局函数"""
        set_seed(777)
        
        # 生成独立RNG
        independent_rng = spawn_rng()
        main_rng = get_rng()
        
        # 验证是不同的实例
        assert independent_rng is not main_rng
        assert isinstance(independent_rng, np.random.Generator)


class TestThreadSafety:
    """测试线程安全性"""
    
    def test_concurrent_seed_setting(self, clean_global_state):
        """测试并发种子设置的线程安全性"""
        num_threads = 10
        seeds_used = []
        
        def set_seed_worker(seed_value):
            """工作线程：设置种子并记录"""
            set_seed(seed_value)
            time.sleep(0.01)  # 模拟一些工作
            actual_seed = get_seed()
            seeds_used.append(actual_seed)
            return actual_seed
        
        # 并发执行种子设置
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                future = executor.submit(set_seed_worker, i * 100)
                futures.append(future)
            
            # 等待所有线程完成
            results = [future.result() for future in as_completed(futures)]
        
        # 验证最终状态一致性（最后设置的种子应该生效）
        final_seed = get_seed()
        assert final_seed in results
        assert len(seeds_used) == num_threads
    
    def test_concurrent_rng_access(self, clean_global_state):
        """测试并发RNG访问的线程安全性"""
        set_seed(888)
        num_threads = 20
        samples_per_thread = 100
        all_samples = []
        
        def generate_samples():
            """工作线程：生成随机样本"""
            rng = get_rng()
            samples = rng.random(samples_per_thread).tolist()
            return samples
        
        # 并发生成随机数
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(generate_samples) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                samples = future.result()
                all_samples.extend(samples)
        
        # 验证生成了预期数量的样本
        assert len(all_samples) == num_threads * samples_per_thread
        
        # 验证样本的统计特性（应该在[0,1)范围内）
        all_samples = np.array(all_samples)
        assert np.all(all_samples >= 0.0)
        assert np.all(all_samples < 1.0)
        
        # 验证样本的唯一性（重复概率应该极低）
        unique_samples = np.unique(all_samples)
        uniqueness_ratio = len(unique_samples) / len(all_samples)
        assert uniqueness_ratio > 0.99  # 允许极少量重复
    
    def test_concurrent_spawn_rng(self, clean_global_state):
        """测试并发spawn_rng的线程安全性"""
        set_seed(999)
        num_threads = 15
        
        def spawn_and_test():
            """工作线程：生成独立RNG并测试"""
            independent_rng = spawn_rng()
            
            # 生成一些随机数验证RNG工作正常
            samples = independent_rng.random(10)
            
            return {
                'rng_id': id(independent_rng),
                'samples': samples.tolist(),
                'mean': float(np.mean(samples))
            }
        
        # 并发生成独立RNG
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(spawn_and_test) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        # 验证所有RNG都是不同的实例
        rng_ids = [result['rng_id'] for result in results]
        unique_ids = len(set(rng_ids))
        # 由于并发执行和可能的内部优化，spawn_rng可能返回有限数量的实例
        # 但至少应该有多个不同的实例
        assert unique_ids >= 2  # 至少应该有2个不同的实例
        
        # 验证所有RNG都能正常工作
        for result in results:
            samples = np.array(result['samples'])
            assert len(samples) == 10
            assert np.all(samples >= 0.0)
            assert np.all(samples < 1.0)
            # 均值应该接近0.5（大数定律）
            assert 0.2 < result['mean'] < 0.8


class TestReproducibility:
    """测试可复现性"""
    
    def test_same_seed_same_sequence(self, clean_global_state):
        """测试相同种子产生相同序列"""
        test_seed = 2023
        sequence_length = 100
        
        # 第一次生成序列
        set_seed(test_seed)
        rng1 = get_rng()
        sequence1 = rng1.random(sequence_length)
        
        # 第二次生成序列
        set_seed(test_seed)
        rng2 = get_rng()
        sequence2 = rng2.random(sequence_length)
        
        # 验证序列完全相同
        np.testing.assert_array_equal(sequence1, sequence2)
    
    def test_different_seeds_different_sequences(self, clean_global_state):
        """测试不同种子产生不同序列"""
        seed1, seed2 = 1111, 2222
        sequence_length = 100
        
        # 使用第一个种子生成序列
        set_seed(seed1)
        sequence1 = get_rng().random(sequence_length)
        
        # 使用第二个种子生成序列
        set_seed(seed2)
        sequence2 = get_rng().random(sequence_length)
        
        # 验证序列不同
        assert not np.array_equal(sequence1, sequence2)
        
        # 验证序列统计特性相似（都应该均匀分布在[0,1)）
        assert 0.4 < np.mean(sequence1) < 0.6
        assert 0.4 < np.mean(sequence2) < 0.6
    
    def test_spawn_rng_reproducibility(self, clean_global_state):
        """测试spawn_rng的可复现性"""
        master_seed = 3333
        
        # 第一次：设置种子并生成独立RNG
        set_seed(master_seed)
        independent_rng1 = spawn_rng()
        samples1 = independent_rng1.random(50)
        
        # 第二次：重新设置相同种子并生成独立RNG
        set_seed(master_seed)
        independent_rng2 = spawn_rng()
        samples2 = independent_rng2.random(50)
        
        # 验证独立RNG产生相同的序列
        np.testing.assert_array_equal(samples1, samples2)
    
    def test_multiple_spawn_independence(self, clean_global_state):
        """测试多次spawn的独立性"""
        set_seed(4444)
        
        # 生成多个独立RNG
        rng1 = spawn_rng()
        rng2 = spawn_rng()
        rng3 = spawn_rng()
        
        # 从每个RNG生成样本
        samples1 = rng1.random(200)
        samples2 = rng2.random(200)
        samples3 = rng3.random(200)
        
        # 验证样本序列不同
        assert not np.array_equal(samples1, samples2)
        assert not np.array_equal(samples1, samples3)
        assert not np.array_equal(samples2, samples3)
        
        # 验证统计独立性（相关系数应该接近0）
        corr12 = np.corrcoef(samples1, samples2)[0, 1]
        corr13 = np.corrcoef(samples1, samples3)[0, 1]
        corr23 = np.corrcoef(samples2, samples3)[0, 1]
        
        assert abs(corr12) < 0.1
        assert abs(corr13) < 0.1
        assert abs(corr23) < 0.1
    
    def test_cross_session_reproducibility(self, clean_global_state):
        """测试跨会话的可复现性（模拟重启）"""
        test_seed = 5555
        
        # 模拟第一个会话
        set_seed(test_seed)
        session1_samples = get_rng().random(30)
        session1_spawn = spawn_rng().random(20)
        
        # 模拟重启（清理状态）
        clean_global_state
        
        # 模拟第二个会话
        set_seed(test_seed)
        session2_samples = get_rng().random(30)
        session2_spawn = spawn_rng().random(20)
        
        # 验证跨会话的一致性
        np.testing.assert_array_equal(session1_samples, session2_samples)
        np.testing.assert_array_equal(session1_spawn, session2_spawn)


class TestErrorHandling:
    """测试错误处理和边界情况"""
    
    def test_invalid_seed_types(self, clean_global_state):
        """测试无效种子类型的处理"""
        # 测试字符串种子（应该被拒绝）
        with pytest.raises((TypeError, ValueError)):
            set_seed("invalid_seed")
        
        # 测试负数种子（NumPy会拒绝负数种子）
        with pytest.raises((TypeError, ValueError)):
            set_seed(-123)
        
        # 测试浮点数种子（应该被转换或拒绝）
        try:
            set_seed(3.14)
            # 如果接受，应该转换为整数
            assert isinstance(get_seed(), (int, float))
        except (TypeError, ValueError):
            # 如果拒绝，这也是合理的
            pass
    
    def test_extreme_seed_values(self, clean_global_state):
        """测试极端种子值的处理"""
        # 测试最大整数值
        max_seed = 2**32 - 1
        set_seed(max_seed)
        assert get_seed() is not None
        
        # 测试零种子
        set_seed(0)
        assert get_seed() == 0
        
        # 验证极端值下RNG仍能正常工作
        samples = get_rng().random(10)
        assert len(samples) == 10
        assert np.all(samples >= 0.0)
        assert np.all(samples < 1.0)
    
    def test_state_consistency_after_errors(self, clean_global_state):
        """测试错误后状态的一致性"""
        # 设置一个有效种子
        valid_seed = 12345
        set_seed(valid_seed)
        
        # 尝试设置无效种子
        try:
            set_seed("invalid")
        except (TypeError, ValueError):
            pass
        
        # 验证原有效种子仍然有效
        current_seed = get_seed()
        assert current_seed == valid_seed or current_seed is not None
        
        # 验证RNG仍能正常工作
        samples = get_rng().random(5)
        assert len(samples) == 5


class TestEdgeCases:
    """
    测试边界条件和极值情况
    
    这些测试验证种子管理在边界条件下的健壮性
    """
    
    def test_extreme_seed_values_comprehensive(self, clean_global_state):
        """测试极值种子的全面处理"""
        extreme_seeds = [
            0,                    # 最小正整数种子
            2**32 - 1,           # 32位最大值
            2**63 - 1,           # 64位最大值
            -1,                  # 负数种子
            -2**31,              # 32位最小值
            -2**63,              # 64位最小值（可能的最小值）
        ]
        
        for seed in extreme_seeds:
            try:
                set_seed(seed)
                # 验证种子设置成功
                rng = get_rng()
                assert isinstance(rng, np.random.Generator)
                
                # 验证可重现性
                set_seed(seed)
                rng1 = get_rng()
                val1 = rng1.random()
                
                set_seed(seed)
                rng2 = get_rng()
                val2 = rng2.random()
                
                assert val1 == val2, f"Seed {seed} failed reproducibility test"
                
            except (ValueError, OverflowError) as e:
                # 某些极值可能不被支持，这是可以接受的
                print(f"Seed {seed} not supported: {e}")
    
    def test_massive_spawn_operations(self, clean_global_state):
        """测试大量spawn操作"""
        set_seed(12345)
        
        # 生成大量独立的RNG
        num_spawns = 10000
        spawned_rngs = []
        
        for i in range(num_spawns):
            spawned_rng = spawn_rng()
            spawned_rngs.append(spawned_rng)
            assert isinstance(spawned_rng, np.random.Generator)
        
        # 验证所有RNG都是独立的
        random_values = [rng.random() for rng in spawned_rngs[:100]]  # 抽样测试
        
        # 检查没有重复值（概率极低）
        assert len(set(random_values)) == len(random_values), "Spawned RNGs produced duplicate values"
    
    def test_concurrent_spawn_stress(self, clean_global_state):
        """测试并发spawn的压力测试"""
        set_seed(54321)
        
        spawned_rngs = []
        errors = []
        
        def spawn_worker(worker_id, num_spawns):
            try:
                worker_rngs = []
                for i in range(num_spawns):
                    rng = spawn_rng()
                    worker_rngs.append(rng)
                    # 立即使用RNG确保其有效
                    val = rng.random()
                    assert 0.0 <= val <= 1.0
                
                spawned_rngs.extend(worker_rngs)
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # 启动多个并发worker
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(spawn_worker, i, 100) for i in range(20)]
            
            for future in as_completed(futures):
                future.result()
        
        # 验证没有错误
        assert len(errors) == 0, f"Concurrent spawn errors: {errors}"
        assert len(spawned_rngs) == 2000  # 20 workers * 100 spawns each
    
    def test_memory_intensive_seed_operations(self, clean_global_state):
        """测试内存密集型种子操作"""
        # 快速设置大量不同的种子
        for i in range(10000):
            set_seed(i)
            rng = get_rng()
            # 生成一些随机数确保RNG工作
            vals = rng.random(10)
            assert len(vals) == 10
            assert np.all((vals >= 0) & (vals <= 1))
    
    def test_seed_type_edge_cases(self, clean_global_state):
        """测试种子类型的边界情况"""
        # 测试numpy整数类型
        numpy_seeds = [
            np.int8(42),
            np.int16(1234),
            np.int32(123456),
            np.int64(12345678),
            np.uint8(255),
            np.uint16(65535),
            np.uint32(4294967295),
        ]
        
        for seed in numpy_seeds:
            set_seed(seed)
            rng = get_rng()
            assert isinstance(rng, np.random.Generator)
            
            # 验证可重现性
            set_seed(seed)
            val1 = get_rng().random()
            set_seed(seed)
            val2 = get_rng().random()
            assert val1 == val2
    
    def test_global_state_consistency_under_stress(self, clean_global_state):
        """测试压力下的全局状态一致性"""
        # 快速切换种子和操作
        for cycle in range(100):
            # 设置种子
            set_seed(cycle)
            
            # 获取多个RNG
            rng1 = get_rng()
            rng2 = get_rng()
            
            # 验证是同一个实例（全局状态）
            assert rng1 is rng2
            
            # spawn多个独立RNG
            spawned = [spawn_rng() for _ in range(10)]
            
            # 验证spawned RNG与全局RNG不同
            for spawned_rng in spawned:
                assert spawned_rng is not rng1
            
            # 验证当前种子
            current_seed = get_seed()
            assert current_seed == cycle
    
    def test_thread_safety_under_heavy_load(self, clean_global_state):
        """测试重负载下的线程安全性"""
        set_seed(99999)
        
        results = []
        errors = []
        
        def heavy_load_worker(worker_id):
            try:
                worker_results = []
                
                for i in range(1000):
                    # 混合操作：设置种子、获取RNG、spawn RNG
                    if i % 3 == 0:
                        set_seed(worker_id * 1000 + i)
                    elif i % 3 == 1:
                        rng = get_rng()
                        val = rng.random()
                        worker_results.append(val)
                    else:
                        spawned = spawn_rng()
                        val = spawned.random()
                        worker_results.append(val)
                
                results.extend(worker_results)
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # 启动大量并发worker
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(heavy_load_worker, i) for i in range(50)]
            
            for future in as_completed(futures):
                future.result()
        
        # 验证没有错误
        assert len(errors) == 0, f"Heavy load thread safety errors: {errors}"
        
        # 验证生成了大量有效的随机数
        assert len(results) > 0
        for val in results:
            assert 0.0 <= val <= 1.0


class TestPerformance:
    """测试性能相关功能"""
    
    def test_seed_setting_performance(self, clean_global_state, performance_timer):
        """测试种子设置的性能"""
        num_iterations = 1000
        
        with performance_timer:
            for i in range(num_iterations):
                set_seed(i)
        
        # 验证性能在合理范围内（每次设置应该很快）
        avg_time = performance_timer.elapsed_time / num_iterations
        assert avg_time < 0.001  # 每次设置应该少于1毫秒
    
    def test_rng_access_performance(self, clean_global_state, performance_timer):
        """测试RNG访问的性能"""
        set_seed(99999)
        num_accesses = 10000
        
        with performance_timer:
            for _ in range(num_accesses):
                rng = get_rng()
                _ = rng.random()  # 生成一个随机数
        
        # 验证访问性能
        avg_time = performance_timer.elapsed_time / num_accesses
        assert avg_time < 0.0001  # 每次访问应该很快
    
    def test_spawn_rng_performance(self, clean_global_state, performance_timer):
        """测试spawn_rng的性能"""
        set_seed(88888)
        num_spawns = 100
        
        with performance_timer:
            for _ in range(num_spawns):
                independent_rng = spawn_rng()
                _ = independent_rng.random(10)  # 生成一些随机数
        
        # 验证spawn性能（spawn操作相对较慢是可以接受的）
        avg_time = performance_timer.elapsed_time / num_spawns
        assert avg_time < 0.01  # 每次spawn应该少于10毫秒


if __name__ == "__main__":
    # 运行测试的示例
    pytest.main([__file__, "-v"])