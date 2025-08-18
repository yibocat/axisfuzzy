#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""High-level convenience API for axisfuzzy configuration subsystem.

This module exposes simple convenience functions around a single global
:class:`~axisfuzzy.config.manager.ConfigManager` instance so library users
can quickly read or mutate configuration without dealing with the manager
class directly.

The functions are intentionally thin wrappers and preserve the behaviour of
the underlying manager (validation, load/save semantics, etc.).
"""

from pathlib import Path
from typing import Union, Any, Optional

from .config_file import Config
from .manager import ConfigManager

_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """
    Return the global configuration manager instance.

    This is the main entry point for accessing and modifying configuration.
    It provides methods to get, set, load, and save configurations.

    Returns
    -------
    ConfigManager
        The global :class:`~axisfuzzy.config.manager.ConfigManager` instance.

    Examples
    --------
    >>> from axisfuzzy.config.api import get_config_manager
    >>> manager = get_config_manager()
    >>> isinstance(manager, ConfigManager)
    True
    """
    return _config_manager


def get_config() -> Config:
    """
    Return the active configuration object.

    Returns
    -------
    Config
        The current :class:`~axisfuzzy.config.config_file.Config` instance.

    Examples
    --------
    >>> from axisfuzzy.config.api import get_config
    >>> cfg = get_config()
    >>> isinstance(cfg, Config)
    True
    """
    return _config_manager.get_config()


def set_config(**kwargs: Any):
    """
    Update multiple configuration entries.

    Parameters
    ----------
    **kwargs
        Keyword arguments mapping configuration field names to new values.
        Field names must match the dataclass attributes defined in
        :class:`~axisfuzzy.config.config_file.Config`.

    Raises
    ------
    ValueError
        If an unknown configuration key is provided or validation fails.

    Examples
    --------
    >>> set_config(DEFAULT_PRECISION=6)
    """
    _config_manager.set_config(**kwargs)


def load_config_file(file_path: Union[str, Path]):
    """
    Load configuration from a JSON file and apply it.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a JSON configuration file.

    Raises
    ------
    FileNotFoundError
        If the given file does not exist.
    ValueError
        If the file is not valid JSON or contains invalid configuration values.
    RuntimeError
        For unexpected errors while reading or applying the file.
    """
    _config_manager.load_config_file(file_path)


def save_config_file(file_path: Union[str, Path]):
    """
    Save the current configuration to a JSON file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path where the configuration will be saved. Parent directories will be
        created if necessary.

    Raises
    ------
    RuntimeError
        If writing the file fails.
    """
    _config_manager.save_config_file(file_path)


def reset_config():
    """
    Reset the active configuration to defaults.

    This restores :class:`~axisfuzzy.config.config_file.Config` to its
    default state and clears any source metadata.
    """
    _config_manager.reset_config()
