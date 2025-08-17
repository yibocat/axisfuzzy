#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
axisfuzzy.core.registry
=======================

Central registry for fuzzy number types (`mtype`).

This module provides the `FuzznumRegistry`, a thread-safe singleton class
that serves as the central directory for all fuzzy number implementations
within the AxisFuzzy ecosystem. It maps a unique string identifier, the
membership type or `mtype`, to its corresponding concrete implementation
classes:

- :class:`~.base.FuzznumStrategy`: Handles the logic and mathematics for a
  single fuzzy number.
- :class:`~.backend.FuzzarrayBackend`: Manages the high-performance,
  Struct-of-Arrays (SoA) data storage for :class:`~.fuzzarray.Fuzzarray`.

The registry supports transactional registrations, an observer pattern for
monitoring changes, and comprehensive introspection capabilities. A suite of
factory functions and decorators is also provided for convenient interaction
with the global registry instance.

This module is a cornerstone of AxisFuzzy's extensibility, allowing developers
to define and integrate new types of fuzzy numbers seamlessly.
"""

import threading
import warnings
from contextlib import contextmanager

from typing import Optional, Dict, Type, List, Any, Callable

from .base import FuzznumStrategy
from .backend import FuzzarrayBackend


class FuzznumRegistry:
    """
    A thread-safe, singleton registry for fuzzy number implementations.

    This class manages the association between a membership type string (`mtype`)
    and the corresponding `FuzznumStrategy` and `FuzzarrayBackend` classes that
    define its behavior and storage. As a singleton, it ensures that there is
    only one central source of truth for all fuzzy number types throughout the
    application's lifecycle.

    The registry is designed for robustness and extensibility, featuring:
    - **Thread Safety**: All registration and retrieval operations are protected
      by locks to prevent race conditions in multi-threaded environments.
    - **Transactional Operations**: The `transaction` context manager allows
      for atomic batch registrations, ensuring that either all registrations
      in a batch succeed or none do, maintaining a consistent state.
    - **Observer Pattern**: External components can subscribe to registry events
      (e.g., registration, unregistration) to react dynamically to changes.
    - **Introspection**: Provides methods to query the registry's state, such
      as listing all registered `mtype`s, checking their completeness, and
      retrieving performance statistics.

    Attributes
    ----------
    strategies : dict[str, type[FuzznumStrategy]]
        A dictionary mapping `mtype` strings to their registered `FuzznumStrategy` classes.
    backends : dict[str, type[FuzzarrayBackend]]
        A dictionary mapping `mtype` strings to their registered `FuzzarrayBackend` classes.

    Notes
    -----
    This class should not be instantiated directly. Instead, the global singleton
    instance should be accessed via the :func:`get_registry_fuzztype` factory function.

    Examples
    --------
    Registering a new, complete fuzzy number type (`mtype`).

    First, define a mock strategy and backend:

    >>> from axisfuzzy.core.base import FuzznumStrategy
    >>> from axisfuzzy.core.backend import FuzzarrayBackend
    >>> class MyNewStrategy(FuzznumStrategy):
    ...     mtype = 'mynewtype'
    ...     # ... implementation ...
    >>> class MyNewBackend(FuzzarrayBackend):
    ...     mtype = 'mynewtype'
    ...     # ... implementation ...

    Now, get the global registry and register the new type:

    >>> from axisfuzzy.core.registry import get_registry_fuzztype
    >>> registry = get_registry_fuzztype()
    >>> result = registry.register(strategy=MyNewStrategy, backend=MyNewBackend)
    >>> print(result['mtype'])
    mynewtype
    >>> 'mynewtype' in registry.get_registered_mtypes()
    True
    """
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
        # self._load_default_fuzznum_types()

    # ======================== Transaction Support ========================

    @contextmanager
    def transaction(self):
        """
        A context manager for performing atomic, transactional registrations.

        This method ensures that a series of registration or unregistration
        operations are treated as a single, atomic unit. If any operation
        within the `with` block raises an exception, all changes made to the
        registry during the transaction are automatically rolled back to their
        original state. This is particularly useful for batch registrations
        where a consistent state must be maintained.

        Yields
        ------
        None
            This context manager does not yield a value.

        Raises
        ------
        Exception
            Any exception raised within the `with` block will be re-raised
            after the rollback is complete.

        Examples
        --------
        >>> # Assume MyStrategy, MyBackend, BadStrategy are defined
        >>> registry = get_registry_fuzztype()
        >>> try:
        ...     with registry.transaction():
        ...         registry.register(strategy=MyStrategy, backend=MyBackend)
        ...         # This next line will raise a ValueError
        ...         registry.register(strategy=BadStrategy)
        ... except ValueError:
        ...     print("Transaction failed and rolled back.")
        ...
        >>> # The first registration was also rolled back
        >>> 'my_mtype' in registry.get_registered_mtypes()
        False
        """
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
        """
        Registers an observer to be notified of registry events.

        The observer pattern allows external components to listen for changes
        within the registry, such as the registration or unregistration of
        a fuzzy number type. The observer must be a callable that accepts
        two arguments: an event type string and a dictionary containing
        event-specific data.

        Parameters
        ----------
        observer : callable
            A callable with the signature `observer(event_type, event_data)`,
            where `event_type` is a string (e.g., 'register_strategy') and
            `event_data` is a dictionary with details about the event.
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Removes a previously registered observer.

        If the specified observer is found in the list of registered observers,
        it will be removed and will no longer receive notifications of
        registry events.

        Parameters
        ----------
        observer : callable
            The observer callable to be removed.
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        for observer in self._observers:
            try:
                observer(event_type, event_data)
            except Exception as e:
                warnings.warn(f"Observer notification failed: {e}")

    # ======================== Registration Management ========================

    def register_strategy(self, strategy: Type[FuzznumStrategy]) -> Dict[str, Any]:
        """
        Registers a single FuzznumStrategy subclass with the registry.

        This method performs validation to ensure the provided class is a valid
        subclass of `FuzznumStrategy` and has the required `mtype` attribute.
        It then adds the class to the internal `strategies` mapping.

        Parameters
        ----------
        strategy : FuzznumStrategy
            The `FuzznumStrategy` subclass to register.

        Returns
        -------
        dict
            A dictionary containing details of the registration, including
            `mtype`, `component` ('strategy'), `registered_class` name, and
            whether an existing registration was `overwrote_existing`.

        Raises
        ------
        TypeError
            If `strategy` is not a class or not a subclass of `FuzznumStrategy`.
        ValueError
            If the `strategy` class does not have an `mtype` attribute defined.
        """
        self._validate_strategy_class(strategy)
        mtype = strategy.mtype

        with self._lock:
            try:
                existing = mtype in self.strategies
                self.strategies[mtype] = strategy

                result = {
                    'mtype': mtype,
                    'component': 'strategy',
                    'registered_class': strategy.__name__,
                    'overwrote_existing': existing
                }
                self._notify_observers('register_strategy', result)
                return result
            except Exception as e:
                self._registration_stats['failed_registrations'] += 1
                raise e

    def register_backend(self, backend: Type[FuzzarrayBackend]) -> Dict[str, Any]:
        """
        Registers a single FuzzarrayBackend subclass with the registry.

        This method performs validation to ensure the provided class is a valid
        subclass of `FuzzarrayBackend` and has the required `mtype` attribute.
        It then adds the class to the internal `backends` mapping.

        Parameters
        ----------
        backend : FuzzarrayBackend
            The `FuzzarrayBackend` subclass to register.

        Returns
        -------
        dict
            A dictionary containing details of the registration, including
            `mtype`, `component` ('backend'), `registered_class` name, and
            whether an existing registration was `overwrote_existing`.

        Raises
        ------
        TypeError
            If `backend` is not a class or not a subclass of `FuzzarrayBackend`.
        ValueError
            If the `backend` class does not have an `mtype` attribute defined.
        """
        self._validate_backend_class(backend)
        mtype = backend.mtype

        with self._lock:
            try:
                existing = mtype in self.backends
                self.backends[mtype] = backend

                result = {
                    'mtype': mtype,
                    'component': 'backend',
                    'registered_class': backend.__name__,
                    'overwrote_existing': existing
                }
                self._notify_observers('register_backend', result)
                return result
            except Exception as e:
                self._registration_stats['failed_registrations'] += 1
                raise e

    def register(self,
                 strategy: Optional[Type[FuzznumStrategy]] = None,
                 backend: Optional[Type[FuzzarrayBackend]] = None) -> Dict[str, Any]:
        """
        Registers a strategy and/or a backend for a given `mtype`.

        This is the primary method for registering the components of a fuzzy
        number type. It can register a strategy, a backend, or both in a
        single, thread-safe operation. It validates the inputs and ensures
        that if both are provided, their `mtype` attributes match.

        Parameters
        ----------
        strategy : FuzznumStrategy, optional
            The `FuzznumStrategy` subclass to register.
        backend : FuzzarrayBackend, optional
            The `FuzzarrayBackend` subclass to register.

        Returns
        -------
        dict
            A dictionary summarizing the registration bundle, containing the
            `mtype` and a list of `details` for each component registered.

        Raises
        ------
        ValueError
            If neither `strategy` nor `backend` is provided, or if their
            `mtype` attributes do not match.
        TypeError
            If the provided `strategy` or `backend` are not valid classes.
        """

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

            mtype = (strategy or backend).mtype
            result = {'mtype': mtype, 'details': []}

            try:
                if strategy:
                    reg_info = self.register_strategy(strategy)
                    result['details'].append(reg_info)

                if backend:
                    reg_info = self.register_backend(backend)
                    result['details'].append(reg_info)

                self._registration_stats['total_registrations'] += 1
                self._notify_observers('register_bundle', result)
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
        """
        Registers multiple fuzzy number types from a list in a single transaction.

        This method wraps the registration process in a transaction, ensuring
        that all types in the list are registered successfully. If any single
        registration fails, the entire batch is rolled back, leaving the
        registry in its original state.

        Parameters
        ----------
        registrations : list[dict]
            A list where each item is a dictionary, typically with 'strategy'
            and/or 'backend' keys pointing to the classes to be registered.
            Example: `[{'strategy': QROFNStrategy, 'backend': QROFNBackend}, ...]`

        Returns
        -------
        dict
            A dictionary where keys are the `mtype`s of the successfully
            registered types and values are the detailed results from the
            `register` method.

        Raises
        ------
        TypeError
            If `registrations` is not a list or if an item in the list is not a dict.
        ValueError, TypeError
            If any individual registration fails its validation, the exception
            is re-raised after the transaction is rolled back.
        """
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
        """
        Removes a strategy and/or backend for a given `mtype` from the registry.

        This allows for the dynamic removal of fuzzy number types.

        Parameters
        ----------
        mtype : str
            The `mtype` identifier of the fuzzy number type to unregister.
        remove_strategy : bool, default True
            If True, removes the associated `FuzznumStrategy` class.
        remove_backend : bool, default True
            If True, removes the associated `FuzzarrayBackend` class.

        Returns
        -------
        dict
            A dictionary detailing the result of the unregistration, including
            which components were removed.
        
        Raises
        ------
        TypeError
            If `mtype` is not a string.
        """

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
        """
        Retrieves the registered `FuzznumStrategy` class for a given `mtype`.

        Parameters
        ----------
        mtype : str
            The `mtype` identifier.

        Returns
        -------
        type[FuzznumStrategy]
            The registered `FuzznumStrategy` subclass.

        Raises
        ------
        ValueError
            If no strategy is found for the specified `mtype`.
        """
        strategy_cls = self.strategies.get(mtype)
        if strategy_cls is None:
            raise ValueError(f"Strategy for mtype '{mtype}' not found in registry.")
        return strategy_cls

    def get_backend(self, mtype: str) -> Type[FuzzarrayBackend]:
        """
        Retrieves the registered `FuzzarrayBackend` class for a given `mtype`.

        Parameters
        ----------
        mtype : str
            The `mtype` identifier.

        Returns
        -------
        type[FuzzarrayBackend]
            The registered `FuzzarrayBackend` subclass.

        Raises
        ------
        ValueError
            If no backend is found for the specified `mtype`.
        """
        backend_cls = self.backends.get(mtype)
        if backend_cls is None:
            raise ValueError(f"Backend for mtype '{mtype}' not found in registry.")
        return backend_cls

    def get_registered_mtypes(self) -> Dict[str, Dict[str, Any]]:
        """
        Provides a comprehensive overview of all registered `mtype`s.

        This introspection method returns a dictionary detailing the status of
        every `mtype` known to the registry, including whether it has a
        registered strategy and/or backend, and the names of the associated classes.

        Returns
        -------
        dict
            A dictionary where keys are `mtype` strings. Each value is another
            dictionary containing boolean flags `has_strategy`, `has_backend`,
            `is_complete`, and the string names `strategy_class` and `backend_class`.
        """
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
        Retrieves quantitative statistics about the registry's state and activity.

        This method is useful for monitoring and debugging, providing insights
        into the number of registered components, registration failures, and
        active observers.

        Returns
        -------
        dict
            A dictionary containing statistics such as `total_strategies`,
            `total_backends`, `complete_types`, `registration_stats`, and
            `observer_count`.
        """
        return {
            'total_strategies': len(self.strategies),
            'total_backends': len(self.backends),
            'complete_types': len(set(self.strategies.keys()) & set(self.backends.keys())),
            'registration_stats': self._registration_stats.copy(),
            'observer_count': len(self._observers)
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Performs a health check on the registry to find incomplete registrations.

        An `mtype` is considered "incomplete" if it has a registered strategy
        but no backend, or vice versa. A healthy registry has no incomplete types.

        Returns
        -------
        dict
            A dictionary containing health status information, including an
            `is_healthy` boolean flag, and lists of `complete_types`,
            `incomplete_types`, `missing_strategies`, and `missing_backends`.
        """
        
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


def get_registry_fuzztype() -> FuzznumRegistry:
    """
    Access the global singleton instance of `FuzznumRegistry`.

    This is the standard factory function for obtaining the registry. It ensures
    that only one instance of the registry exists across the entire application,
    providing a consistent and centralized management point for all fuzzy number types.

    Returns
    -------
    FuzznumRegistry
        The global singleton registry instance.
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = FuzznumRegistry()

    return _registry_instance


def register_strategy(cls: Type[FuzznumStrategy]) -> Type[FuzznumStrategy]:
    """
    A class decorator for automatically registering a `FuzznumStrategy`.

    This decorator provides a convenient way to register a strategy class with
    the global registry at the time of its definition.

    Parameters
    ----------
    cls : FuzznumStrategy
        The `FuzznumStrategy` subclass to be decorated and registered.

    Returns
    -------
    FuzznumStrategy
        The original class, unchanged.

    Examples
    --------
    >>> from axisfuzzy.core.registry import register_strategy
    >>>
    >>> @register_strategy
    ... class MyNewStrategy(FuzznumStrategy):
    ...     mtype = 'decorated_strategy'
    ...     # ... implementation ...
    """
    get_registry_fuzztype().register_strategy(cls)
    return cls


def register_backend(cls: Type[FuzzarrayBackend]) -> Type[FuzzarrayBackend]:
    """
    A class decorator for automatically registering a `FuzzarrayBackend`.

    This decorator provides a convenient way to register a backend class with
    the global registry at the time of its definition.

    Parameters
    ----------
    cls : type[FuzzarrayBackend]
        The `FuzzarrayBackend` subclass to be decorated and registered.

    Returns
    -------
    type[FuzzarrayBackend]
        The original class, unchanged.

    Examples
    --------
    >>> from axisfuzzy.core.registry import register_backend
    >>>
    >>> @register_backend
    ... class MyNewBackend(FuzzarrayBackend):
    ...     mtype = 'decorated_backend'
    ...     # ... implementation ...
    """
    get_registry_fuzztype().register_backend(cls)
    return cls


def register_fuzztype(strategy: Optional[Type[FuzznumStrategy]] = None,
                      backend: Optional[Type[FuzzarrayBackend]] = None) -> Dict[str, Any]:
    """
    A convenience function to register a strategy and/or backend with the global registry.

    This function is a simple wrapper around the `register` method of the
    global `FuzznumRegistry` instance.

    Parameters
    ----------
    strategy : type[FuzznumStrategy], optional
        The `FuzznumStrategy` subclass to register.
    backend : type[FuzzarrayBackend], optional
        The `FuzzarrayBackend` subclass to register.

    Returns
    -------
    dict
        The result dictionary from the underlying `register` call.

    Examples
    --------
    >>> # Assuming MyStrategy and MyBackend classes are defined
    >>> from axisfuzzy.core.registry import register_fuzztype
    >>> register_fuzztype(strategy=MyStrategy, backend=MyBackend)
    """
    return get_registry_fuzztype().register(
        strategy=strategy, backend=backend)


def register_batch_fuzztypes(registrations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    A convenience function to batch-register multiple fuzzy types with the global registry.

    This function is a simple wrapper around the `batch_register` method of the
    global `FuzznumRegistry` instance. It performs the registrations within a
    single transaction, ensuring atomicity.

    Parameters
    ----------
    registrations : list[dict]
        A list where each item is a dictionary specifying the components to
        register. Each dictionary should have 'strategy' and/or 'backend' keys.
        Example: `[{'strategy': QROFNStrategy, 'backend': QROFNBackend}, ...]`

    Returns
    -------
    dict
        A dictionary where keys are the `mtype`s of the successfully
        registered types and values are the detailed results from the
        underlying `register` method.
    """
    return get_registry_fuzztype().batch_register(registrations)


def unregister_fuzztype(mtype: str,
                        remove_strategy: bool = True,
                        remove_backend: bool = True) -> Dict[str, Any]:
    """
    A convenience function to unregister a fuzzy type from the global registry.

    This function is a simple wrapper around the `unregister` method of the
    global `FuzznumRegistry` instance.

    Parameters
    ----------
    mtype : str
        The `mtype` identifier of the fuzzy number type to unregister.
    remove_strategy : bool, default True
        If True, removes the associated `FuzznumStrategy` class.
    remove_backend : bool, default True
        If True, removes the associated `FuzzarrayBackend` class.

    Returns
    -------
    dict
        A dictionary detailing the result of the unregistration.
    """
    return get_registry_fuzztype().unregister(
        mtype=mtype,
        remove_strategy=remove_strategy,
        remove_backend=remove_backend
    )


def get_fuzztype_strategy(mtype: str) -> Optional[Type[FuzznumStrategy]]:
    """
    A convenience function to retrieve a strategy class from the global registry.

    This function is a simple, error-suppressing wrapper around the `get_strategy`
    method of the global `FuzznumRegistry` instance.

    Parameters
    ----------
    mtype : str
        The `mtype` identifier for which to retrieve the strategy.

    Returns
    -------
    type[FuzznumStrategy] or None
        The registered `FuzznumStrategy` subclass if found, otherwise `None`.
    """
    try:
        return get_registry_fuzztype().get_strategy(mtype)
    except ValueError:
        return None


def get_fuzztype_backend(mtype: str) -> Optional[Type[FuzzarrayBackend]]:
    """
    A convenience function to retrieve a backend class from the global registry.

    This function is a simple, error-suppressing wrapper around the `get_backend`
    method of the global `FuzznumRegistry` instance.

    Parameters
    ----------
    mtype : str
        The `mtype` identifier for which to retrieve the backend.

    Returns
    -------
    type[FuzzarrayBackend] or None
        The registered `FuzzarrayBackend` subclass if found, otherwise `None`.
    """
    try:
        return get_registry_fuzztype().get_backend(mtype)
    except ValueError:
        return None


def get_fuzztype_mtypes() -> Dict[str, Dict[str, Any]]:
    """
    A convenience function to get an overview of all registered mtypes.

    This function is a simple wrapper around the `get_registered_mtypes` method
    of the global `FuzznumRegistry` instance.

    Returns
    -------
    dict
        A dictionary where keys are `mtype` strings and values are dictionaries
        detailing the registration status for that `mtype`.
    """
    return get_registry_fuzztype().get_registered_mtypes()
