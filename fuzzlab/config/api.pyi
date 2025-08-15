#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:53
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from pathlib import Path
from typing import Union, Any
from .config_file import Config
from .manager import ConfigManager

_config_manager: ConfigManager

def get_config() -> Config: ...
def set_config(**kwargs: Any) -> None: ...
def load_config_file(file_path: Union[str, Path]) -> None: ...
def save_config_file(file_path: Union[str, Path]) -> None: ...
def reset_config() -> None: ...