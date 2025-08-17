#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
import json
import threading

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Union, Optional, Dict, List

from axisfuzzy.config.config_file import Config


class ConfigManager:
    """
    Singleton manager for application configuration.

    The manager holds a single :class:`~axisfuzzy.config.config_file.Config`
    instance, provides load/save/reset operations, and validates updates
    against field metadata.

    Notes
    -----
    The class implements a thread-safe singleton pattern: multiple imports
    will share the same manager instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        Create or return the singleton instance.

        Returns
        -------
        ConfigManager
            The unique ConfigManager instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._config = Config()
        self._config_source = None       # Configure source tracking
        self._is_modified = False        # Modify Status Tracking

        # --------------------- Reserved for future expansion ----------------------

        self._observers = []             # Observer List (Not currently used)
        self._config_history = []        # Configuration History (Not currently used)
        self._validation_rules = {}      # Validation Rules (Not Currently Used)

        self._initialized = True

    def get_config(self) -> Config:
        """
        Return the active Config instance.

        Returns
        -------
        Config
            The current configuration dataclass.
        """
        return self._config

    def set_config(self, **kwargs):
        """
        Set one or more configuration fields.

        Parameters
        ----------
        **kwargs
            Mapping of configuration field names to desired values.

        Raises
        ------
        ValueError
            If an unknown key is provided or a value fails validation.

        Examples
        --------
        >>> mgr = ConfigManager()
        >>> mgr.set_config(DEFAULT_PRECISION=6)
        """
        for key, value in kwargs.items():
            config_field = None
            for f in fields(self._config):
                if f.name == key:
                    config_field = f
                    break

            if config_field is None:
                available_params = [
                    field.name
                    for field in fields(self._config)]
                raise ValueError(
                    f"Unknown configuration parameter: '{key}'. "
                    f"Available Parameters: {', '.join(available_params)}"
                )

            self._validate_parameter(key, value)
            setattr(self._config, key, value)
            self._is_modified = True

    # ==================== Configuration file operations ====================

    def load_config_file(self, file_path: Union[str, Path]):
        """
        Load configuration from a JSON file and apply it.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the JSON configuration file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file content is not a dict or a validation error occurs.
        RuntimeError
            For unexpected IO/parse errors.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            if not isinstance(config_data, dict):
                raise ValueError("The content of the configuration file must be a JSON object (dictionary).。")

            self.set_config(**config_data)
            self._config_source = str(file_path)

        except json.JSONDecodeError as e:
            raise ValueError(f"The JSON format of the configuration file '{file_path}' is invalid: {e}")
        except ValueError as e:
            raise ValueError(f"Configuration data loaded from the file '{file_path}' failed validation: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the configuration file '{file_path}': {e}")

    def save_config_file(self, file_path: Union[str, Path]):
        """
        Save the current configuration to a JSON file.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Destination path for the JSON file. Parent directories are created
            automatically.

        Raises
        ------
        RuntimeError
            If the file cannot be written.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_data = asdict(self._config)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            self._is_modified = False

        except Exception as e:
            raise RuntimeError(f"Failed to save the configuration file '{file_path}': {e}")

    # ==================== Configuration Management ====================

    def reset_config(self):
        """
        Reset configuration to defaults.

        This replaces the internal Config instance with a new default
        instance and clears source/modified flags.
        """
        self._config = Config()
        self._config_source = None
        self._is_modified = False

    def is_modified(self) -> bool:
        """
        Check whether configuration was modified since last save/load.

        Returns
        -------
        bool
            True if modified, False otherwise.
        """
        return self._is_modified

    def get_config_source(self) -> Optional[str]:
        """
        Return the path of the source file if the configuration was loaded from a file.

        Returns
        -------
        str or None
            File path string or None when configuration was not loaded from a file.
        """
        return self._config_source

    # ==================== Configuration Verification ====================

    def _validate_parameter(self, param_name: str, value: Any):
        """
        Validate a single configuration parameter using field metadata.

        Parameters
        ----------
        param_name : str
            Name of the configuration field to validate.
        value : Any
            Value to validate.

        Raises
        ------
        ValueError
            When the parameter doesn't exist or the validator metadata rejects the value.
        """
        config_field = None
        for f in fields(self._config):
            if f.name == param_name:
                config_field = f
                break

        # In theory, this check should be completed before calling _validate_parameter,
        # but it is retained as a safety measure.
        if config_field is None:
            raise ValueError(f"Internal error: Attempting to verify unknown parameter '{param_name}'。")

        if 'validator' in config_field.metadata:
            validator = config_field.metadata['validator']
            error_msg = config_field.metadata.get('error_msg', "Value is invalid。")
            if not validator(value):
                raise ValueError(f"Validation failed for parameter '{param_name}': {error_msg} Given value: {value}")

    # ==================== 诊断和工具方法 ====================

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Produce a categorized summary of current configuration.

        Returns
        -------
        dict
            Mapping of category -> {field_name: value}, with an additional
            'meta' key that contains 'config_source' and 'is_modified'.
        """
        summary = {}
        config = self._config

        for f in fields(config):
            category = f.metadata.get('category', 'uncategorized')  # Get category, default is 'uncategorized'

            if category not in summary:
                summary[category] = {}

            summary[category][f.name] = getattr(config, f.name)

        summary['meta'] = {
            'config_source': self._config_source,
            'is_modified': self._is_modified,
        }

        return summary

    def validate_all_config(self) -> List[str]:
        """
        Validate all configuration fields and collect validation errors.

        Returns
        -------
        list of str
            List of error messages. Empty list means all fields are valid.
        """
        errors = []
        # Traverse all fields of the Config dataclass
        for f in fields(self._config):
            param_name = f.name
            value = getattr(self._config, param_name)
            try:
                self._validate_parameter(param_name, value)
            except ValueError as e:
                errors.append(str(e))  # Additional error message

        return errors

    @staticmethod
    def create_config_template(file_path: Union[str, Path]):
        """
        Create a JSON configuration template file populated with defaults.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Destination path for the template JSON. Parent directories will be created.

        Notes
        -----
        The generated file contains some top-level comment/metadata fields
        alongside the actual default configuration for easy editing.
        """
        template = {
            "_comment":
                "MohuPy Configuration File Template",
            "_description":
                "Please modify the following configuration parameters as needed:",
            "_version": "1.0",

            # Actual configuration parameters
            **asdict(Config())
        }

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

    # ==================== Reserved interface for future expansion ====================

    def add_config_observer(self, observer: Any):
        """
        Register an observer for config changes.

        Parameters
        ----------
        observer : Any
            Observer object. Observer semantics are reserved for future use.
        """
        # Not yet implemented, reserved for future expansion
        self._observers.append(observer)

    def remove_observer(self, observer: Any):
        """
        Remove a previously registered observer.

        Parameters
        ----------
        observer : Any
            Observer to remove. No-op if observer not registered.
        """
        # Not yet implemented, reserved for future expansion
        if observer in self._observers:
            self._observers.remove(observer)

    def get_config_history(self) -> List[Any]:
        """
        Return a shallow copy of the configuration change history.

        Returns
        -------
        list
            The stored configuration history entries (currently unused).
        """
        # Not yet implemented, reserved for future expansion
        return self._config_history.copy()
