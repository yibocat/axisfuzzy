#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/12 18:05
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import threading
import warnings
from contextlib import contextmanager

from typing import Optional, Dict, Type, List, Any, Callable, Tuple

from fuzzlab.core.base import FuzznumStrategy
from fuzzlab.core.backend import FuzzarrayBackend


class FuzznumRegistry:

    _instance: Optional['FuzznumRegistry'] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> 'FuzznumRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._in_transaction = False
        if not FuzznumRegistry._initialized:
            with FuzznumRegistry._lock:
                if not FuzznumRegistry._initialized:
                    self._init_registry()
                    FuzznumRegistry._initialized = True

    def _init_registry(self) -> None:
        self.strategies: Dict[str, Type[FuzznumStrategy]] = {}
        self.backends: Dict[str, Type[FuzzarrayBackend]] = {}

        self._registration_history: List[Dict[str, Any]] = []

        self._registration_stats = {
            'total_registrations': 0,
            'failed_registrations': 0,
            'overwrites': 0
        }

        self._transaction_stack: List[Dict[str, Any]] = []
        self._in_transaction = False
        self._observers: List[Callable[[str, Dict[str, Any]], None]] = []

        # Call a private method to load predefined default fuzzy number types.
        self._load_default_fuzznum_types()

    def _load_default_fuzznum_types(self) -> None:

        default_types = self._get_default_types()

        with self.transaction():

            for strategy_cls, backend_cls in default_types:
                try:
                    self.register(strategy=strategy_cls, backend=backend_cls)
                except Exception as e:
                    warnings.warn(f"Failed to load default type {strategy_cls.mtype}: {e}")

    @staticmethod
    def _get_default_types() -> List[Tuple[Type[FuzznumStrategy], Type[FuzzarrayBackend]]]:

        from ..fuzztype.qrofs import QROFNStrategy, QROFNBackend

        return [(QROFNStrategy, QROFNBackend)]

    # ======================== Transaction Support ========================

    @contextmanager
    def transaction(self):
        if self._in_transaction:
            yield
            return

        self._in_transaction = True

        snapshot = self._create_snapshot()

        try:
            yield
            self._transaction_stack.clear()
        except Exception as e:
            self._restore_snapshot(snapshot)
            raise e

        finally:
            self._in_transaction = False

    def _create_snapshot(self) -> Dict[str, Any]:
        return {
            'strategies': self.strategies.copy(),
            'backends': self.backends.copy(),
            'stats': self._registration_stats.copy()
        }

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.strategies.clear()
        self.backends.clear()

        self.strategies.update(snapshot['strategies'])
        self.backends.update(snapshot['backends'])
        self._registration_stats.update(snapshot['stats'])

    # ======================== Observer Pattern ========================
    def add_observer(self, observer: Callable[[str, Dict[str, Any]], None]) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Callable[[str, Dict[str, Any]], None]) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        for observer in self._observers:
            try:
                observer(event_type, event_data)
            except Exception as e:
                warnings.warn(f"Observer notification failed: {e}")

    # ======================== Registration Management ========================

    def register(self,
                 strategy: Optional[Type[FuzznumStrategy]] = None,
                 backend: Optional[Type[FuzzarrayBackend]] = None) -> Dict[str, Any]:

        if not strategy and not backend:
            raise ValueError("At least one of 'strategy' or 'backend' must be provided.")

        if strategy is not None:
            self._validate_strategy_class(strategy)
        if backend is not None:
            self._validate_backend_class(backend)

        if strategy is not None and backend is not None:
            if strategy.mtype != backend.mtype:
                raise ValueError(
                    f"mtype mismatch: "
                    f"strategy='{strategy.mtype}', backend='{backend.mtype}'"
                )

        with self._lock:
            # Determine mtype from any provided component
            mtype = None
            if strategy and hasattr(strategy, 'mtype'):
                mtype = strategy.mtype
            elif backend and hasattr(backend, 'mtype'):
                mtype = backend.mtype

            existing_strategy = mtype in self.strategies
            existing_backend = mtype in self.backends

            result = {
                'mtype': mtype,
                'strategy_registered': False,
                'backend_registered': False,
                'is_complete': False,
                'overwrote_existing': {
                    'strategy': existing_strategy and strategy is not None,
                    'backend': existing_backend and backend is not None
                }
            }

            try:
                if strategy is not None:
                    if existing_strategy:
                        self._registration_stats['overwrites'] += 1
                    self.strategies[mtype] = strategy
                    result['strategy_registered'] = True

                if backend is not None:
                    if existing_backend:
                        self._registration_stats['overwrites'] += 1
                    self.backends[mtype] = backend
                    result['back_registered'] = True

                result['is_complete'] = (
                        mtype in self.strategies and mtype in self.backends)

                self._registration_stats['total_registrations'] += 1
                self._notify_observers('register', result)

                return result

            except Exception as e:
                self._registration_stats['failed_registrations'] += 1
                raise e

    @staticmethod
    def _validate_strategy_class(strategy: Type[FuzznumStrategy]) -> None:
        if not isinstance(strategy, type):
            raise TypeError(f"Strategy must be a class, got {type(strategy).__name__}")

        if not issubclass(strategy, FuzznumStrategy):
            raise TypeError(f"Strategy must be a subclass of FuzznumStrategy, got {strategy.__name__}")

        if not hasattr(strategy, 'mtype'):
            raise ValueError(f"Strategy class {strategy.__name__} must define 'mtype' attribute")

    @staticmethod
    def _validate_backend_class(backend: Type[FuzzarrayBackend]) -> None:
        if not isinstance(backend, type):
            raise TypeError(f"Backend must be a class, got {type(backend).__name__}")

        if not issubclass(backend, FuzzarrayBackend):
            raise TypeError(f"Backend must be a subclass of FuzzarrayBackend, got {backend.__name__}")

        if not hasattr(backend, 'mtype'):
            raise ValueError(f"Backend class {backend.__name__} must define 'mtype' attribute")

    def batch_register(self, registrations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(registrations, list):
            raise TypeError(f"Registrations must be a list, got {type(registrations).__name__}")
        results = {}
        with self.transaction():
            for i, registration in enumerate(registrations):
                # Iterates through the list of registration requests.
                # Checks: Ensures that each element in the list is a dictionary.
                if not isinstance(registration, dict):
                    raise TypeError(f"Each registration must be a dict, got {type(registration).__name__} at index {i}")

                strategy = registration.get('strategy')
                backend = registration.get('backend')

                try:
                    result = self.register(strategy=strategy, backend=backend)
                    results[result['mtype']] = result

                except Exception as e:
                    error_info = {
                        'error': str(e),
                        'index': i,
                    }
                    results[f"error_{i}"] = error_info
                    raise

        return results

    def unregister(self, mtype: str,
                   remove_strategy: bool = True,
                   remove_backend: bool = True) -> Dict[str, Any]:

        if not isinstance(mtype, str):
            raise TypeError(f"mtype must be a string, got {type(mtype).__name__}")

        with self._lock:
            result = {
                'mtype': mtype,
                'strategy_removed': False,
                'backend_removed': False,
                'was_complete': (mtype in self.strategies and mtype in self.backends),
            }

            if remove_strategy and mtype in self.strategies:
                del self.strategies[mtype]
                result['strategy_removed'] = True

            if remove_backend and mtype in self.backends:
                del self.backends[mtype]
                result['backend_removed'] = True

            self._registration_history.append(result.copy())
            self._notify_observers('unregister', result)

            return result

    # ======================== Introspection Methods ========================

    def get_strategy(self, mtype: str) -> Type[FuzznumStrategy]:
        strategy_cls = self.strategies.get(mtype)
        if strategy_cls is None:
            raise ValueError(f"Strategy for mtype '{mtype}' not found in registry.")
        return strategy_cls

    def get_backend(self, mtype: str) -> Type[FuzzarrayBackend]:
        backend_cls = self.backends.get(mtype)
        if backend_cls is None:
            raise ValueError(f"Backend for mtype '{mtype}' not found in registry.")
        return backend_cls

    def get_registered_mtypes(self) -> Dict[str, Dict[str, Any]]:
        all_mtypes = set(self.strategies.keys()) | set(self.backends.keys())

        result = {}

        for mtype in all_mtypes:
            has_strategy = mtype in self.strategies
            has_backend = mtype in self.backends

            result[mtype] = {
                'has_strategy': has_strategy,
                'has_backend': has_backend,
                'strategy_class': self.strategies[mtype].__name__ if has_strategy else None,
                'backend_class': self.backends[mtype].__name__ if has_backend else None,
                'is_complete': has_strategy and has_backend
            }

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves registry statistics.

        This method provides quantitative data on registry operations,
        facilitating monitoring and analysis of registry activity and status.

        Returns:
            Dict[str, Any]: A dictionary containing statistics about the registry.
        """
        return {
            'total_strategies': len(self.strategies),
            'total_backends': len(self.backends),
            'complete_types': len(set(self.strategies.keys()) & set(self.backends.keys())),
            'registration_stats': self._registration_stats.copy(),
            'observer_count': len(self._observers)
        }

    def get_health_status(self) -> Dict[str, Any]:
        complete_types = set(self.strategies.keys()) & set(self.backends.keys())

        incomplete_types = (set(self.strategies.keys()) | set(self.backends.keys())) - complete_types

        return {
            'is_healthy': len(incomplete_types) == 0,
            'total_types': len(self.strategies) + len(self.backends),
            'complete_types': list(complete_types),
            'incomplete_types': list(incomplete_types),
            'missing_strategies': list(set(self.backends.keys()) - set(self.strategies.keys())),
            'missing_backends': list(set(self.strategies.keys()) - set(self.backends.keys())),
            'error_rate': (self._registration_stats['failed_registrations'] /
                           max(1, self._registration_stats['total_registrations']))
        }


# ======================== Global Singleton and Factory Method ========================

# Global registry instance
_registry_instance: Optional[FuzznumRegistry] = None
_registry_lock = threading.RLock()


def get_fuzznum_registry() -> FuzznumRegistry:
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = FuzznumRegistry()

    return _registry_instance


def register_fuzznum(strategy: Optional[Type[FuzznumStrategy]] = None,
                     backend: Optional[Type[FuzzarrayBackend]] = None) -> Dict[str, Any]:
    return get_fuzznum_registry().register(
        strategy=strategy, backend=backend)


def batch_register_fuzz(registrations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return get_fuzznum_registry().batch_register(registrations)


def unregister_fuzznum(mtype: str,
                       remove_strategy: bool = True,
                       remove_backend: bool = True) -> Dict[str, Any]:
    return get_fuzznum_registry().unregister(
        mtype=mtype,
        remove_strategy=remove_strategy,
        remove_backend=remove_backend
    )


def get_strategy(mtype: str) -> Optional[Type[FuzznumStrategy]]:
    try:
        return get_fuzznum_registry().get_strategy(mtype)
    except ValueError:
        return None


def get_backend(mtype: str) -> Optional[Type[FuzzarrayBackend]]:
    try:
        return get_fuzznum_registry().get_backend(mtype)
    except ValueError:
        return None


def get_fuzznum_registered_mtypes() -> Dict[str, Dict[str, Any]]:
    return get_fuzznum_registry().get_registered_mtypes()
