import pytest

import json
from pathlib import Path

from axisfuzzy.config.manager import ConfigManager


def test_get_config_summary_structure_and_values():
    m = ConfigManager()
    m.reset_config()

    # 修改一部分配置，确保 summary 反映最新值
    m.set_config(DEFAULT_PRECISION=10, CACHE_SIZE=512)
    summary = m.get_config_summary()

    assert isinstance(summary, dict)
    assert 'meta' in summary and isinstance(summary['meta'], dict)

    # 类别键应该至少包含 basic/performance/display（依赖字段元数据）
    assert 'basic' in summary and isinstance(summary['basic'], dict)
    assert 'performance' in summary and isinstance(summary['performance'], dict)

    # 关键值是否同步
    assert summary['basic']['DEFAULT_PRECISION'] == 10
    assert summary['performance']['CACHE_SIZE'] == 512


def test_create_config_template_contains_defaults(tmp_path: Path):
    m = ConfigManager()
    out = tmp_path / 'dir' / 'template.json'
    m.create_config_template(out)
    assert out.exists()

    data = json.loads(out.read_text())
    # 模板元字段存在
    assert {'_comment', '_description', '_version'} <= set(data.keys())
    # 至少包含部分关键配置键
    assert 'DEFAULT_PRECISION' in data
    assert 'DEFAULT_Q' in data
    assert 'CACHE_SIZE' in data


if __name__ == "__main__":
    pytest.main()
