import pytest

import numpy as np

from axisfuzzy.config.config_file import Config


def test_config_defaults_and_types():
    cfg = Config()

    # 基本默认值是否存在（不做强耦合于具体数值，只挑关键项）
    assert isinstance(cfg.DEFAULT_MTYPE, str) and len(cfg.DEFAULT_MTYPE) > 0
    assert isinstance(cfg.DEFAULT_Q, (int, np.integer)) and cfg.DEFAULT_Q > 0
    assert isinstance(cfg.DEFAULT_PRECISION, int) and cfg.DEFAULT_PRECISION >= 0
    assert isinstance(cfg.DEFAULT_EPSILON, (int, float)) and cfg.DEFAULT_EPSILON > 0
    assert isinstance(cfg.CACHE_SIZE, int) and cfg.CACHE_SIZE >= 0
    assert isinstance(cfg.TNORM_VERIFY, bool)

    # 显示阈值相关（为正整数）
    assert cfg.DISPLAY_THRESHOLD_SMALL > 0
    assert cfg.DISPLAY_THRESHOLD_MEDIUM > 0
    assert cfg.DISPLAY_THRESHOLD_LARGE > 0
    assert cfg.DISPLAY_THRESHOLD_HUGE > 0
    assert cfg.DISPLAY_EDGE_ITEMS_MEDIUM > 0
    assert cfg.DISPLAY_EDGE_ITEMS_LARGE > 0
    assert cfg.DISPLAY_EDGE_ITEMS_HUGE > 0


if __name__ == "__main__":
    pytest.main()
