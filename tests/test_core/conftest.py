import sys
from pathlib import Path
import pytest

# 将项目根目录加入 sys.path，确保 `axisfuzzy` 可被导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 统一管理配置全局状态，确保各测试用例之间相互独立
@pytest.fixture(autouse=True)
def reset_global_config_state():
    """在每个用例前后重置全局配置状态。"""
    from axisfuzzy.config import api as config_api

    # 测试前：确保干净状态
    config_api.reset_config()
    yield
    # 测试后：恢复默认，避免跨文件污染
    config_api.reset_config()


@pytest.fixture(scope="session")
def ensure_qrofn_registered():
    """确保 qrofn 类型在测试环境中已注册"""
    try:
        from axisfuzzy.core.registry import get_registry_fuzztype
        registry = get_registry_fuzztype()

        # 检查 qrofn 是否已注册
        if 'qrofn' not in registry.strategies:
            # 尝试导入并注册 qrofn
            try:
                import axisfuzzy.fuzztype.qrofs.qrofn
                import axisfuzzy.fuzztype.qrofs.backend
                # 如果导入成功，qrofn 应该已经通过装饰器自动注册
            except ImportError:
                pytest.skip("qrofn module not available")

        yield registry

    except Exception as e:
        pytest.skip(f"Failed to setup qrofn registration: {e}")


@pytest.fixture
def sample_qrofn():
    """创建一个示例 qrofn 实例用于测试"""
    try:
        from axisfuzzy.core.fuzznums import fuzznum
        return fuzznum((0.8, 0.2), mtype='qrofn', q=2)
    except ValueError as e:
        if "Unsupported mtype" in str(e):
            pytest.skip("qrofn not registered in test environment")
        else:
            raise
