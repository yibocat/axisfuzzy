import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import json


# 模拟必要的导入，避免复杂的依赖问题
class MockConfig:
    DEFAULT_MTYPE = 'qrofn'
    DEFAULT_Q = 1
    CACHE_SIZE = 256
    DEFAULT_EPSILON = 1e-12


class MockOperationTNorm:
    def __init__(self, norm_type='algebraic', q=1, **params):
        self.norm_type = norm_type
        self.q = q
        self.params = params

    def get_info(self):
        return {
            'norm_type': self.norm_type,
            'q': self.q,
            'parameters': self.params
        }


class MockLruCache:
    def __init__(self, maxsize=256):
        self.maxsize = maxsize
        self._cache = {}

    def get(self, key):
        return self._cache.get(key)

    def put(self, key, value):
        self._cache[key] = value


# 模拟 FuzznumStrategy 基类
class MockFuzznumStrategy:
    """模拟的 FuzznumStrategy 基类，用于测试核心概念"""

    def __init__(self, q=None):
        self.q = q or 1
        self._declared_attributes = []
        self._attribute_validators = {}
        self._attribute_transformers = {}
        self._change_callbacks = {}
        self._op_cache = MockLruCache()

        # 模拟属性声明收集
        self._collect_declared_attributes()

        # 添加 q 验证器
        self.add_attribute_validator('q', lambda x: isinstance(x, int) and 1 <= x <= 100)

        # 执行初始验证
        self._validate()

    def _collect_declared_attributes(self):
        """收集声明的属性"""
        # 模拟从类定义中收集属性
        for attr_name in dir(self.__class__):
            if not attr_name.startswith('_') and not callable(getattr(self.__class__, attr_name)):
                self._declared_attributes.append(attr_name)

    def add_attribute_validator(self, attr_name, validator):
        """添加属性验证器"""
        if not callable(validator):
            raise TypeError("Validator must be callable")
        self._attribute_validators[attr_name] = validator

    def add_attribute_transformer(self, attr_name, transformer):
        """添加属性转换器"""
        if not callable(transformer):
            raise TypeError("Transformer must be callable")
        self._attribute_transformers[attr_name] = transformer

    def add_change_callback(self, attr_name, callback):
        """添加变更回调"""
        if not callable(callback):
            raise TypeError("Callback must be callable")
        self._change_callbacks[attr_name] = callback

    def __setattr__(self, name, value):
        """重写属性设置，添加验证和转换逻辑"""
        if name.startswith('_'):
            super().__setattr__(name, value)
            return

        # 检查是否为声明的属性
        if name not in self._declared_attributes and name != 'q':
            raise AttributeError(f"Attribute '{name}' not declared")

        # 获取旧值
        old_value = getattr(self, name, None)

        # 执行验证
        if name in self._attribute_validators:
            validator = self._attribute_validators[name]
            if not validator(value):
                raise ValueError(f"Validation failed for attribute '{name}' with value '{value}'")

        # 执行转换
        if name in self._attribute_transformers:
            transformer = self._attribute_transformers[name]
            value = transformer(value)

        # 设置属性
        super().__setattr__(name, value)

        # 执行回调
        if name in self._change_callbacks:
            callback = self._change_callbacks[name]
            try:
                callback(name, old_value, value)
            except Exception as e:
                raise RuntimeError(f"Callback for attribute '{name}' failed: {e}")

    def _validate(self):
        """执行验证"""
        if hasattr(self, 'mtype') and (not isinstance(self.mtype, str) or not self.mtype.strip()):
            raise ValueError(f"mtype must be a non-empty string, got '{self.mtype}'")

    def validate_all_attributes(self):
        """验证所有属性"""
        errors = []

        try:
            self._validate()
        except Exception as e:
            errors.append(str(e))

        for attr_name in self._declared_attributes:
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                if attr_name in self._attribute_validators:
                    validator = self._attribute_validators[attr_name]
                    if not validator(value):
                        errors.append(f"Attribute '{attr_name}' validation failed with value '{value}'")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }

    def get_declared_attributes(self):
        """获取声明的属性列表"""
        return self._declared_attributes.copy()

    def execute_operation(self, op_name, operand=None):
        """执行运算（模拟）"""
        # 模拟运算执行
        return {'operation': op_name, 'result': 'mock_result'}


class TestMockFuzznumStrategy:
    """测试模拟的 FuzznumStrategy 功能"""

    def test_basic_creation(self):
        """测试基本创建"""
        strategy = MockFuzznumStrategy(q=2)
        assert strategy.q == 2

    def test_declared_attributes(self):
        """测试声明的属性收集"""

        class TestStrategy(MockFuzznumStrategy):
            mtype = 'test'
            attr1: float = 0.0
            attr2: int = 1

        strategy = TestStrategy()
        assert 'mtype' in strategy._declared_attributes
        assert 'attr1' in strategy._declared_attributes
        assert 'attr2' in strategy._declared_attributes

    def test_attribute_validation(self):
        """测试属性验证"""

        class TestStrategy(MockFuzznumStrategy):
            mtype = 'test'
            value: float = 0.5

        strategy = TestStrategy()

        # 添加验证器
        strategy.add_attribute_validator('value', lambda x: 0 <= x <= 1)

        # 有效值应该通过
        strategy.value = 0.8
        assert strategy.value == 0.8

        # 无效值应该失败
        with pytest.raises(ValueError, match="Validation failed"):
            strategy.value = 1.5

    def test_attribute_transformer(self):
        """测试属性转换"""

        class TestStrategy(MockFuzznumStrategy):
            mtype = 'test'
            data: list = None

        strategy = TestStrategy()

        # 添加转换器
        strategy.add_attribute_transformer('data', lambda x: np.array(x) if x is not None else None)

        # 测试转换
        strategy.data = [1, 2, 3]
        assert isinstance(strategy.data, np.ndarray)
        assert np.array_equal(strategy.data, [1, 2, 3])

    def test_change_callback(self):
        """测试变更回调"""

        class TestStrategy(MockFuzznumStrategy):
            mtype = 'test'
            value: float = 0.0

        strategy = TestStrategy()
        callback_called = False
        callback_data = None

        def callback(name, old_value, new_value):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = (name, old_value, new_value)

        strategy.add_change_callback('value', callback)

        # 改变属性
        strategy.value = 0.5

        # 验证回调被调用
        assert callback_called
        assert callback_data == ('value', 0.0, 0.5)

    def test_q_validation(self):
        """测试 q 参数验证"""
        # 有效值
        strategy = MockFuzznumStrategy(q=5)
        assert strategy.q == 5

        # 无效值应该失败
        with pytest.raises(ValueError):
            MockFuzznumStrategy(q=0)

        with pytest.raises(ValueError):
            MockFuzznumStrategy(q=101)

    def test_validate_all_attributes(self):
        """测试全面属性验证"""

        class TestStrategy(MockFuzznumStrategy):
            mtype = 'test'
            valid_attr: float = 0.5
            invalid_attr: float = 1.5

        strategy = TestStrategy()

        # 添加验证器
        strategy.add_attribute_validator('valid_attr', lambda x: 0 <= x <= 1)
        strategy.add_attribute_validator('invalid_attr', lambda x: 0 <= x <= 1)

        # 运行验证
        result = strategy.validate_all_attributes()

        # 应该有验证错误
        assert not result['is_valid']
        assert len(result['errors']) > 0
        assert any('invalid_attr' in error for error in result['errors'])

    def test_undeclared_attribute_error(self):
        """测试未声明属性的错误"""
        strategy = MockFuzznumStrategy()

        # 尝试设置未声明的属性应该失败
        with pytest.raises(AttributeError, match="not declared"):
            strategy.undeclared_attr = "value"


# 模拟 Fuzznum 类
class MockFuzznum:
    """模拟的 Fuzznum 类，用于测试门面模式"""

    def __init__(self, mtype=None, q=None):
        self.mtype = mtype or 'qrofn'
        self.q = q or 1
        self._strategy_instance = None
        self._bound_strategy_methods = {}
        self._bound_strategy_attributes = set()
        self._initialized = False

        # 模拟策略绑定
        self._initialize()

    def _initialize(self):
        """初始化策略绑定"""
        # 模拟策略实例创建
        self._strategy_instance = MockFuzznumStrategy(q=self.q)
        self._strategy_instance.mtype = self.mtype

        # 模拟方法绑定
        self._bound_strategy_methods = {
            'report': self._strategy_instance.report if hasattr(self._strategy_instance, 'report') else None,
            'str': self._strategy_instance.str if hasattr(self._strategy_instance, 'str') else None
        }

        # 模拟属性绑定
        self._bound_strategy_attributes = set(self._strategy_instance.get_declared_attributes())

        self._initialized = True

    def __getattr__(self, name):
        """属性访问代理"""
        if not self._initialized:
            raise AttributeError("Fuzznum not initialized")

        if name in self._bound_strategy_attributes:
            return getattr(self._strategy_instance, name)

        if name in self._bound_strategy_methods:
            return self._bound_strategy_methods[name]

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """属性设置代理"""
        if name.startswith('_') or name in ['mtype', 'q']:
            super().__setattr__(name, value)
            return

        if name == 'mtype':
            raise AttributeError("Cannot modify immutable attribute 'mtype'")

        if self._initialized and name in self._bound_strategy_attributes:
            setattr(self._strategy_instance, name, value)
        else:
            super().__setattr__(name, value)

    def get_strategy_instance(self):
        """获取策略实例"""
        if not self._initialized:
            raise RuntimeError("Strategy instance not found")
        return self._strategy_instance

    def get_strategy_attributes_dict(self):
        """获取策略属性字典"""
        if not self._initialized:
            raise RuntimeError("Cannot get strategy attributes from uninitialized Fuzznum")

        result = {}
        for attr in self._bound_strategy_attributes:
            if hasattr(self._strategy_instance, attr):
                result[attr] = getattr(self._strategy_instance, attr)
        return result

    def copy(self):
        """复制 Fuzznum"""
        if not self._initialized:
            raise RuntimeError("Cannot copy uninitialized object")

        new_fnum = MockFuzznum(self.mtype, self.q)
        attrs = self.get_strategy_attributes_dict()
        for key, value in attrs.items():
            setattr(new_fnum, key, value)
        return new_fnum

    def create(self, **kwargs):
        """创建新的 Fuzznum 实例"""
        new_fnum = MockFuzznum(self.mtype, self.q)
        for key, value in kwargs.items():
            try:
                setattr(new_fnum, key, value)
            except Exception as e:
                raise AttributeError(f"The parameter '{key}' is invalid: {e}")
        return new_fnum

    def to_dict(self):
        """序列化为字典"""
        if not self._initialized:
            raise RuntimeError("Unable to serialize uninitialized object")

        result = {
            'mtype': self.mtype,
            'attributes': {}
        }

        attrs = self.get_strategy_attributes_dict()
        for key, value in attrs.items():
            result['attributes'][key] = value

        return result

    @classmethod
    def from_dict(cls, data):
        """从字典反序列化"""
        if 'mtype' not in data:
            raise ValueError("Dictionary must contain 'mtype' key")

        instance = cls(data['mtype'])

        if 'attributes' in data:
            for key, value in data['attributes'].items():
                try:
                    setattr(instance, key, value)
                except AttributeError:
                    pass

        return instance


class TestMockFuzznum:
    """测试模拟的 Fuzznum 功能"""

    def test_basic_creation(self):
        """测试基本创建"""
        fnum = MockFuzznum(mtype='qrofn', q=2)
        assert fnum.mtype == 'qrofn'
        assert fnum.q == 2

    def test_strategy_binding(self):
        """测试策略绑定"""
        fnum = MockFuzznum(mtype='qrofn')

        # 检查策略实例
        strategy = fnum.get_strategy_instance()
        assert strategy is not None
        assert strategy.mtype == 'qrofn'

        # 检查绑定属性
        bound_attrs = fnum.get_strategy_attributes_dict()
        assert isinstance(bound_attrs, dict)

    def test_attribute_delegation(self):
        """测试属性代理"""
        fnum = MockFuzznum(mtype='qrofn')

        # 设置属性
        fnum.md = 0.8
        fnum.nmd = 0.2

        # 验证属性访问
        assert fnum.md == 0.8
        assert fnum.nmd == 0.2

        # 验证策略中的存储
        strategy = fnum.get_strategy_instance()
        assert strategy.md == 0.8
        assert strategy.nmd == 0.2

    def test_immutable_attributes(self):
        """测试不可变属性"""
        fnum = MockFuzznum(mtype='qrofn')

        # mtype 应该是不可变的
        with pytest.raises(AttributeError, match="Cannot modify immutable attribute 'mtype'"):
            fnum.mtype = 'other_type'

    def test_copy_functionality(self):
        """测试复制功能"""
        fnum = MockFuzznum(mtype='qrofn')
        fnum.md = 0.7
        fnum.nmd = 0.3

        # 创建副本
        fnum_copy = fnum.copy()

        # 验证是不同的实例
        assert fnum_copy is not fnum

        # 验证属性被复制
        assert fnum_copy.md == 0.7
        assert fnum_copy.nmd == 0.3

        # 验证修改副本不影响原实例
        fnum_copy.md = 0.9
        assert fnum.md == 0.7
        assert fnum_copy.md == 0.9

    def test_create_functionality(self):
        """测试创建功能"""
        fnum = MockFuzznum(mtype='qrofn')

        # 创建新实例
        new_fnum = fnum.create(md=0.6, nmd=0.4)

        # 验证是不同的实例
        assert new_fnum is not fnum

        # 验证参数被设置
        assert new_fnum.md == 0.6
        assert new_fnum.nmd == 0.4

        # 验证 mtype 和 q 被继承
        assert new_fnum.mtype == fnum.mtype
        assert new_fnum.q == fnum.q

    def test_serialization(self):
        """测试序列化功能"""
        fnum = MockFuzznum(mtype='qrofn')
        fnum.md = 0.8
        fnum.nmd = 0.2

        # 序列化为字典
        fnum_dict = fnum.to_dict()

        # 验证结构
        assert 'mtype' in fnum_dict
        assert 'attributes' in fnum_dict
        assert fnum_dict['mtype'] == 'qrofn'
        assert fnum_dict['attributes']['md'] == 0.8
        assert fnum_dict['attributes']['nmd'] == 0.2

        # 反序列化
        restored_fnum = MockFuzznum.from_dict(fnum_dict)

        # 验证恢复
        assert restored_fnum.mtype == fnum.mtype
        assert restored_fnum.md == fnum.md
        assert restored_fnum.nmd == fnum.nmd


class TestIntegration:
    """集成测试"""

    def test_strategy_fuzznum_integration(self):
        """测试策略和 Fuzznum 的集成"""
        # 创建策略
        strategy = MockFuzznumStrategy(q=2)
        strategy.mtype = 'qrofn'

        # 添加验证器和转换器
        strategy.add_attribute_validator('md', lambda x: 0 <= x <= 1)
        strategy.add_attribute_validator('nmd', lambda x: 0 <= x <= 1)
        strategy.add_attribute_transformer('md', lambda x: float(x))
        strategy.add_attribute_transformer('nmd', lambda x: float(x))

        # 创建 Fuzznum
        fnum = MockFuzznum(mtype='qrofn', q=2)

        # 设置属性
        fnum.md = 0.8
        fnum.nmd = 0.2

        # 验证
        assert fnum.md == 0.8
        assert fnum.nmd == 0.2

        # 验证策略中的存储
        strategy_instance = fnum.get_strategy_instance()
        assert strategy_instance.md == 0.8
        assert strategy_instance.nmd == 0.2

    def test_validation_chain(self):
        """测试验证链"""
        # 创建策略
        strategy = MockFuzznumStrategy(q=2)
        strategy.mtype = 'qrofn'

        # 添加验证器
        strategy.add_attribute_validator('md', lambda x: 0 <= x <= 1)
        strategy.add_attribute_validator('nmd', lambda x: 0 <= x <= 1)

        # 添加转换器
        strategy.add_attribute_transformer('md', lambda x: float(x))
        strategy.add_attribute_transformer('nmd', lambda x: float(x))

        # 测试有效值
        strategy.md = 0.5
        strategy.nmd = 0.3

        # 测试无效值
        with pytest.raises(ValueError):
            strategy.md = 1.5

        with pytest.raises(ValueError):
            strategy.nmd = -0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
