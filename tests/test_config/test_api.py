import json
from pathlib import Path

import pytest

from axisfuzzy.config.api import (
    get_config_manager,
    get_config,
    set_config,
    load_config_file,
    save_config_file,
    reset_config,
)
from axisfuzzy.config.manager import ConfigManager
from axisfuzzy.config.config_file import Config


def test_get_config_manager_singleton():
    mgr1 = get_config_manager()
    mgr2 = get_config_manager()
    assert isinstance(mgr1, ConfigManager)
    assert mgr1 is mgr2


def test_get_config_returns_config_instance():
    cfg = get_config()
    assert isinstance(cfg, Config)


def test_set_config_updates_values_and_validation():
    # 正常更新
    set_config(DEFAULT_PRECISION=6)
    assert get_config().DEFAULT_PRECISION == 6

    # 非法键
    with pytest.raises(ValueError):
        set_config(UNKNOWN_KEY=1)  # type: ignore[arg-type]

    # 非法值
    with pytest.raises(ValueError):
        set_config(DEFAULT_Q=0)


def test_save_and_load_config_file_roundtrip(tmp_path: Path):
    # 设置一些非默认值
    set_config(DEFAULT_PRECISION=8, DEFAULT_EPSILON=1e-9, DEFAULT_Q=2)

    out = tmp_path / "conf" / "axisfuzzy_config.json"
    save_config_file(out)
    assert out.exists()

    # 修改内存，再从文件恢复
    set_config(DEFAULT_PRECISION=3)
    assert get_config().DEFAULT_PRECISION == 3

    load_config_file(out)
    cfg = get_config()
    assert cfg.DEFAULT_PRECISION == 8
    assert cfg.DEFAULT_EPSILON == 1e-9
    assert cfg.DEFAULT_Q == 2


def test_load_config_file_errors(tmp_path: Path):
    # 文件不存在
    with pytest.raises(FileNotFoundError):
        load_config_file(tmp_path / "nope.json")

    # 非法 JSON
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json ")
    with pytest.raises(ValueError):
        load_config_file(bad)

    # JSON 不是对象
    not_obj = tmp_path / "not_obj.json"
    not_obj.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_config_file(not_obj)

    # JSON 对象但包含非法值
    invalid_value = tmp_path / "invalid_value.json"
    invalid_value.write_text(json.dumps({"DEFAULT_Q": 0}))
    with pytest.raises(ValueError):
        load_config_file(invalid_value)


def test_reset_config_restores_defaults():
    set_config(DEFAULT_PRECISION=7)
    assert get_config().DEFAULT_PRECISION == 7
    reset_config()
    assert get_config().DEFAULT_PRECISION == Config().DEFAULT_PRECISION


if __name__ == "__main__":
    pytest.main()