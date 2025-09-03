import json
from pathlib import Path

import pytest

from axisfuzzy.config.manager import ConfigManager
from axisfuzzy.config.config_file import Config


def test_manager_singleton_semantics():
    m1 = ConfigManager()
    m2 = ConfigManager()
    assert m1 is m2


def test_set_config_and_validation():
    m = ConfigManager()
    # 多键更新
    m.set_config(DEFAULT_PRECISION=5, CACHE_SIZE=128)
    cfg = m.get_config()
    assert cfg.DEFAULT_PRECISION == 5
    assert cfg.CACHE_SIZE == 128

    # 未知键
    with pytest.raises(ValueError):
        m.set_config(UNKNOWN=1)  # type: ignore[arg-type]

    # 非法值
    with pytest.raises(ValueError):
        m.set_config(DEFAULT_EPSILON=-1.0)


def test_load_save_config_file_and_status(tmp_path: Path):
    m = ConfigManager()
    m.reset_config()

    # 初始状态
    assert m.get_config_source() is None
    assert m.is_modified() is False

    # 修改后应为 modified
    m.set_config(DEFAULT_PRECISION=9)
    assert m.is_modified() is True

    # 保存到文件，父目录自动创建
    out = tmp_path / "nested" / "conf.json"
    m.save_config_file(out)
    assert out.exists()
    assert m.is_modified() is False

    # 人为修改文件内容后加载
    data = json.loads(out.read_text())
    data["DEFAULT_Q"] = 3
    out.write_text(json.dumps(data))

    m.load_config_file(out)
    assert m.get_config().DEFAULT_Q == 3
    assert m.get_config_source() == str(out)


def test_load_config_file_error_cases(tmp_path: Path):
    m = ConfigManager()

    with pytest.raises(FileNotFoundError):
        m.load_config_file(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{bad json}")
    with pytest.raises(ValueError):
        m.load_config_file(bad)

    not_obj = tmp_path / "not_obj.json"
    not_obj.write_text(json.dumps(42))
    with pytest.raises(ValueError):
        m.load_config_file(not_obj)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"DEFAULT_Q": 0}))
    with pytest.raises(ValueError):
        m.load_config_file(invalid)


def test_reset_config_and_summary_and_validate_all(tmp_path: Path):
    m = ConfigManager()
    m.set_config(DEFAULT_PRECISION=6)
    m.reset_config()
    assert m.get_config() == Config()
    assert m.get_config_source() is None
    assert m.is_modified() is False

    # 摘要结构
    summary = m.get_config_summary()
    assert "meta" in summary and isinstance(summary["meta"], dict)
    assert "basic" in summary and isinstance(summary["basic"], dict)

    # 默认配置应通过校验
    errors = m.validate_all_config()
    assert errors == []

    # 注入非法值，再验证应返回错误
    m.set_config(DEFAULT_PRECISION=4)  # 合法，先落一笔修改
    with pytest.raises(ValueError):
        m.set_config(DEFAULT_EPSILON=0)


def test_create_config_template(tmp_path: Path):
    m = ConfigManager()
    template_path = tmp_path / "tpl" / "cfg.json"
    m.create_config_template(template_path)
    assert template_path.exists()

    data = json.loads(template_path.read_text())
    # 模板元字段
    assert "_comment" in data and "_description" in data and "_version" in data
    # 配置字段
    cfg_keys = set(Config().__dict__.keys())
    assert cfg_keys.issubset(set(data.keys()))


if __name__ == "__main__":
    pytest.main()
    