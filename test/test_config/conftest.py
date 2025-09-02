import pytest

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


