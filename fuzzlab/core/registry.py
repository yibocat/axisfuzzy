#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 23:28
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

"""
This module implements a singleton registry (`FuzznumRegistry`) for managing
`FuzznumStrategy` and `FuzznumTemplate` classes within the FuzzLab framework.

The registry provides a centralized mechanism for:
- Registering and unregistering fuzzy number types (strategies and templates).
- Ensuring thread-safe operations for registration and retrieval.
- Supporting transactional registration for atomic batch operations.
- Providing introspection methods to query registered types and their completeness.
- Offering observer pattern support for external components to react to registry changes.
- Loading default fuzzy number types upon initialization.

Classes:
    FuzznumRegistry: A singleton class that manages the registration, retrieval,
                     and lifecycle of `FuzznumStrategy` and `FuzznumTemplate` classes.

Functions:
    get_fuzznum_registry(): Returns the global singleton instance of `FuzznumRegistry`.
    register_fuzznum(): A convenience function to register a single fuzzy number type.
    batch_register_fuzznums(): A convenience function to perform transactional batch registration.
    unregister_fuzznum(): A convenience function to unregister a fuzzy number type.
    get_strategy(): A convenience function to retrieve a registered strategy class.
    get_template(): A convenience function to retrieve a registered template class.
    get_fuzznum_registered_mtypes(): A convenience function to get information about all registered types.
"""
import datetime
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, Type, List, Callable, Tuple

from fuzzlab.config import get_config
from fuzzlab.core.base import FuzznumStrategy, FuzznumTemplate, ExampleStrategy, ExampleTemplate

logger = logging.getLogger(__name__)


class FuzznumRegistry:
    """
    A singleton registry for managing FuzznumStrategy and FuzznumTemplate classes.

    This class ensures that only one instance of the registry exists throughout
    the application, providing a centralized and thread-safe mechanism for
    registering, retrieving, and managing different fuzzy number types.

    Attributes:
        _instance (Optional[FuzznumRegistry]): The singleton instance of the registry.
        _lock (threading.RLock): A reentrant lock to ensure thread-safe access
                                 to the registry's internal state.
        _initialized (bool): A flag indicating whether the registry has been
                             fully initialized.
        strategies (Dict[str, Type[FuzznumStrategy]]): A dictionary mapping
                                                        mtype strings to
                                                        FuzznumStrategy subclasses.
        templates (Dict[str, Type[FuzznumTemplate]]): A dictionary mapping
                                                       mtype strings to
                                                       FuzznumTemplate subclasses.
        _registration_history (List[Dict[str, Any]]): A list recording the history
                                                      of registration and
                                                      unregistration operations.
        _registration_stats (Dict[str, int]): Statistics about registration
                                              operations (total, failed, overwrites).
        _transaction_stack (List[Dict[str, Any]]): A stack to store snapshots
                                                    of the registry state for
                                                    transactional support.
        _in_transaction (bool): A flag indicating if the registry is currently
                                in a transaction.
        _observers (List[Callable[[str, Dict[str, Any]], None]]): A list of
                                                                   callable
                                                                   observers
                                                                   to be notified
                                                                   of registry changes.
    """

    _instance: Optional['FuzznumRegistry'] = None

    # Purpose: This is a class-level private reentrant lock.
    # 1. In the implementation of the singleton pattern (__new__ method),
    #   it is used to protect the creation process of _instance, ensuring
    #   that the instance of FuzznumRegistry is created only once in a
    #   multithreaded environment, avoiding race conditions.
    # 2. Inside the FuzznumRegistry instance, it is also used to protect concurrent
    #   access and modification of the registry's core data structures,
    #   such as strategies and templates, as well as statistical information
    #   (_registration_stats), ensuring thread safety during operations like
    #   registration and deregistration.
    _lock: threading.RLock = threading.RLock()

    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> 'FuzznumRegistry':
        """
        Ensures that only one instance of FuzznumRegistry is created (singleton pattern).
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the FuzznumRegistry.

        This constructor is called only once for the singleton instance. It sets
        up the internal data structures, loads default fuzzy number types, and
        configures logging.
        """
        self._in_transaction = None
        if not FuzznumRegistry._initialized:

            with FuzznumRegistry._lock:
                if not FuzznumRegistry._initialized:
                    self._init_registry()
                    FuzznumRegistry._initialized = True

    def _init_registry(self) -> None:
        """
        The actual initialization logic for the registry.

        This method is called only once during the lifecycle of a FuzznumRegistry instance.
        It sets up the dictionaries for strategies and templates, initializes
        registration history and statistics, and configures transaction and
        observer support.
        """
        self.strategies: Dict[str, Type[FuzznumStrategy]] = {}
        self.templates: Dict[str, Type[FuzznumTemplate]] = {}

        # Registration history statistics
        self._registration_history: List[Dict[str, Any]] = []

        # Stores statistics for registration operations
        # - 'total_registrations': Total number of successful calls to the register method.
        # - 'failed_registrations': Total number of failed calls to the register method.
        # - 'overwrites': Number of times an existing strategy or template was overwritten during registration.
        self._registration_stats = {
            'total_registrations': 0,
            'failed_registrations': 0,
            'overwrites': 0
        }

        # Transaction support:
        self._transaction_stack: List[Dict[str, Any]] = []
        # Purpose: A list used to store snapshots of the registry state when a transaction begins.

        # Used to prevent duplicate transaction initiation or handle nested transactions.
        self._in_transaction = False

        # Observer pattern support:
        self._observers: List[Callable[[str, Dict[str, Any]], None]] = []

        config = get_config()

        if config.DEBUG_MODE:
            logger.debug(f"FuzznumRegistry initialized. ID: {id(self)}")

        # Call a private method to load predefined default fuzzy number types.
        self._load_default_fuzznum_types()

    def _load_default_fuzznum_types(self) -> None:
        """
        Loads default fuzzy number strategy and template classes.

        This method retrieves a list of predefined default fuzzy number types
        and registers them with the registry. It uses a transaction to ensure
        atomic batch registration.
        """
        config = get_config()
        if config.DEBUG_MODE:
            logger.info("Loading default fuzzy number types...")

        # Calls the _get_default_types method to get a list containing tuples
        #   of all default fuzzy number strategy and template classes.
        default_types = self._get_default_types()

        with self.transaction():
            # Purpose: Uses the transaction context manager provided by the registry.
            # This ensures that all default type registration operations are an atomic batch.
            # If an error occurs during the registration of any default type, the entire
            # batch operation will be rolled back, ensuring that the registry remains
            # in a consistent state after loading default types.

            for strategy_cls, template_cls in default_types:
                try:
                    # Calls the register method to register the strategy and template classes with the registry.
                    self.register(strategy=strategy_cls, template=template_cls)

                    if config.DEBUG_MODE:
                        logger.debug(f"Loaded default type: {strategy_cls.mtype}")
                except Exception as e:
                    # If registration of a default type fails (e.g., mtype definition is incorrect),
                    # a warning message is logged, but it does not prevent other default types from loading
                    # (because it is within a transaction, the final outcome depends on the transaction result).
                    logger.warning(f"Failed to load default type {strategy_cls.mtype}: {e}")

    def _get_default_types(self) -> List[Tuple[Type[FuzznumStrategy], Type[FuzznumTemplate]]]:
        """
        Retrieves the default fuzzy number type definitions.

        Purpose: This method defines the built-in fuzzy number strategy and template classes
        that are loaded by default by the framework. It returns a list where each element
        is a tuple of (strategy class, template class). These classes are typically defined
        as nested classes (inner classes) to maintain encapsulation and directly inherit
        from FuzznumStrategy and FuzznumTemplate.

        Returns:
            List[Tuple[Type[FuzznumStrategy], Type[FuzznumTemplate]]]: A list of tuples,
                                                                       each containing a
                                                                       strategy and template class.
        """

        from fuzzlab.modules.qrofs.qrofn import QROFNStrategy, QROFNTemplate

        return [
            (QROFNStrategy, QROFNTemplate),
        ]

    # ======================== Transaction Support ========================
    # Transaction support in FuzznumRegistry is crucial for ensuring atomicity
    #   when performing batch modifications to the registry's state.
    # This means that a series of operations either all succeed and are persisted,
    #   or all fail and are rolled back to the state before the operations,
    #   thereby maintaining the data consistency of the registry.
    # This module primarily consists of a context manager `transaction`
    #   and two helper methods `_create_snapshot` and `_restore_snapshot`.

    @contextmanager
    def transaction(self):
        """
        Context manager for registry transactions.

        Usage:
            with registry.transaction():
                registry.register(strategy1, template1)
                registry.register(strategy2, template2)
                # If an exception occurs, all registrations will be rolled back.
        """

        if self._in_transaction:
            # Checks if the current registry instance is already in a transaction.
            # This handles the case of "nested transactions." If already in a transaction,
            # no new snapshot is created; instead, control is directly yielded to the
            # inner `with` block, and inner operations become part of the outer transaction.
            # This means only the outermost transaction is responsible for snapshot creation
            # and final rollback/commit.
            yield
            # Yields control to the code inside the `with` statement block.
            return

        # Start transaction
        self._in_transaction = True

        # Create a snapshot before the transaction begins.
        snapshot = self._create_snapshot()

        try:
            yield
            # Yields control to the code inside the `with` statement block.
            # All registration, unregistration, and other operations are executed here.

            # Transaction successful, clear snapshot
            self._transaction_stack.clear()

        except Exception as e:
            # If any exception occurs inside the `with` statement block, it means the transaction failed.
            logger.warning(f"Transaction failed, rolling back: {e}")
            # Calls the _restore_snapshot method to restore the registry's state to the snapshot
            # created before the transaction began.
            self._restore_snapshot(snapshot)
            # Re-raises the caught exception.
            # This is very important as it allows the external caller to be aware of the
            # transaction failure and handle errors accordingly.
            # If not re-raised, the exception would be "swallowed" by this context manager.
            raise

        finally:
            self._in_transaction = False
            # Regardless of whether the transaction successfully commits or rolls back,
            # this block will always be executed.
            # Resets the _in_transaction flag to False, indicating the end of the transaction,
            # and allowing new transactions to begin.

    def _create_snapshot(self) -> Dict[str, Any]:
        """
        Creates a snapshot of the current registry state.

        This method captures the current state of `FuzznumRegistry` at the
        beginning of a transaction. The captured state includes registered
        strategies, templates, and current statistics.

        Returns:
            Dict[str, Any]: A dictionary representing the snapshot of the registry.
        """
        return {
            'strategies': self.strategies.copy(),
            # Copies the current `strategies` dictionary.
            # Using `.copy()` is key; it creates a shallow copy of the dictionary,
            # ensuring the snapshot is independent and modifications to `self.strategies`
            # within the transaction do not affect the data in the snapshot.

            'templates': self.templates.copy(),
            # Copies the current `templates` dictionary.
            # Similarly, `.copy()` is used to ensure independence.

            'stats': self._registration_stats.copy()
            # Copies the current `_registration_stats` dictionary.
            # Ensures that statistics are also independent in the snapshot.
        }

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Restores the registry state from a given snapshot.

        This method is called when a transaction fails, reverting the registry
        to its state at the time the snapshot was created.

        Args:
            snapshot (Dict[str, Any]): The snapshot dictionary to restore from.
        """

        # Clears all strategies currently in the registry.
        # Clears all templates currently in the registry.
        self.strategies.clear()
        self.templates.clear()

        # Updates the current `strategies` dictionary with the `strategies` dictionary
        # saved in the snapshot. This restores the strategies in the registry to their
        # state at the beginning of the transaction.
        # Restores the templates in the registry to their state at the beginning of the transaction.
        # Ensures that statistics are also rolled back to the state at the beginning of the transaction,
        # maintaining consistency.
        self.strategies.update(snapshot['strategies'])
        self.templates.update(snapshot['templates'])
        self._registration_stats.update(snapshot['stats'])

    # ======================== Observer Pattern ========================
    # The Observer Pattern is a behavioral design pattern that defines a one-to-many
    #   dependency between objects, so that when one object changes state, all its
    #   dependents are notified and updated automatically.
    # In FuzznumRegistry, it allows external components to "listen" for changes
    #   in the registry's state (e.g., new fuzzy number types being registered or unregistered)
    #   without being tightly coupled to the registry.
    # Significance of the Observer Pattern:
    # Decoupling: The registry no longer needs to know who cares about its state changes;
    #   it only needs to call `_notify_observers` when its state changes.
    #   Components that care about these changes (observers) register themselves independently.
    #   This loose coupling makes the system easier to maintain and extend.
    # Event-driven: It transforms registry operations into events, allowing other modules
    #   to respond to these changes in an event-driven manner. For example, a logging
    #   module can register as an observer to record all registration/unregistration events,
    #   and a cache management module can clear relevant caches when a type is unregistered.
    # Extensibility: Adding new response logic is easy; simply write a new observer function
    #   and register it, without modifying the core code of FuzznumRegistry.
    # Testability: Since the logic is decoupled, observers and the registry can be unit-tested independently.

    def add_observer(self, observer: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Adds an observer to the registry.

        Args:
            observer: The observer function to add. It should accept two
                      arguments: `event_type` (str) and `event_data` (Dict[str, Any]).
        """
        if observer not in self._observers:
            # Core logic: This method allows external code to register a callable object
            #   (usually a function or method) as an observer.
            # It first checks if the passed `observer` already exists in the internal `_observers` list.
            # If not, it adds it to the list.
            # This check prevents the same observer from being registered multiple times,
            # which would lead to duplicate notifications.
            self._observers.append(observer)

    def remove_observer(self, observer: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Removes an observer from the registry.

        Args:
            observer: The observer function to remove.
        """
        if observer in self._observers:
            # Core logic: This method allows external code to remove a registered callable object
            #   from the observer list.
            # It first checks if the passed `observer` exists in the `_observers` list.
            # If it exists, it removes it from the list.
            # This allows observers to "unsubscribe" when they no longer need to receive notifications,
            # avoiding unnecessary resource consumption or erroneous behavior.
            self._observers.remove(observer)

    def _notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notifies all registered observers of a registry event.

        Args:
            event_type (str): The type of event (e.g., 'register', 'unregister').
            event_data (Dict[str, Any]): A dictionary containing details about the event.
        """
        for observer in self._observers:
            # Core logic: This method is the "notification" part of the observer pattern;
            #   it iterates through all currently registered observers.
            # For each `observer` in the list, it attempts to call that observer.

            try:
                observer(event_type, event_data)
                # Core logic: Calls the observer function, passing two arguments:
                # 1. `event_type` (string): Represents the type of event that occurred,
                #    e.g., 'register' or 'unregister'.
                # 2. `event_data` (dictionary): Contains detailed information about the event,
                #    e.g., the `mtype` of the registration, the registration result, etc.
                # This allows each observer to execute its specific response logic based on
                # the event type and data.

            except Exception as e:
                # Core logic: There is a crucial `try-except` block here.
                # It catches any exceptions that might occur when calling a single observer function.
                # This is a very important design decision: if an observer function encounters an error
                # during execution, this exception should not prevent other observers from receiving notifications.
                # Without this `try-except`, a faulty observer could interrupt the entire notification process,
                # preventing other observers from receiving notifications, thereby affecting system stability and consistency.
                logger.warning(f"Observer notification failed: {e}")
                # Logs a warning message indicating which observer notification failed and the reason for the failure,
                # but allows the notification process to continue without affecting other observers.

    # ======================== Registration Management ========================

    def register(self,
                 strategy: Optional[Type[FuzznumStrategy]] = None,
                 template: Optional[Type[FuzznumTemplate]] = None) -> Dict[str, Any]:
        """
        Registers a new fuzzy number type (strategy and/or template) with the registry.

        This method performs strict type checking, mtype consistency checks,
        and supports overwriting existing registrations. The operation is thread-safe.

        Args:
            strategy (Optional[Type[FuzznumStrategy]]): The FuzznumStrategy subclass to register.
            template (Optional[Type[FuzznumTemplate]]): The FuzznumTemplate subclass to register.

        Returns:
            Dict[str, Any]: A dictionary containing the registration result,
                            including mtype, success status, and completeness.

        Raises:
            ValueError: If both strategy and template are missing, or if mtypes are inconsistent.
            TypeError: If the provided arguments are not valid class types.
        """

        if not strategy and not template:
            raise ValueError("At least one of 'strategy' or 'template' must be provided.")

        # Calls a helper method to pre-validate the passed strategy and template classes,
        # ensuring they are legitimate FuzznumStrategy/FuzznumTemplate subclasses.
        if strategy is not None:
            self._validate_strategy_class(strategy)
        if template is not None:
            self._validate_template_class(template)

        # Extracts the fuzzy number type identifier (mtype) from the provided strategy or template.
        # This is crucial for registration, as mtype is the unique key for looking up and managing these classes in the registry.
        mtype = self._extract_mtype(strategy, template)

        if strategy is not None and template is not None:
            if strategy.mtype != template.mtype:
                raise ValueError(
                    f"Strategy and template mtype mismatch: "
                    f"strategy='{strategy.mtype}', template='{template.mtype}'"
                )

        with self._lock:
            # Thread safety: Acquires a registry-level lock.
            # This ensures that modifications to the registry's internal data structures
            # (self.strategies, self.templates) are atomic in a multithreaded environment,
            # preventing race conditions and data corruption.

            # Determines if a corresponding strategy or template already exists for the mtype,
            # used for subsequent overwrite warnings and statistics.
            existing_strategy = mtype in self.strategies
            existing_template = mtype in self.templates

            # Prepares the registration result dictionary:
            # Initializes a dictionary to record detailed results of this registration operation,
            # including mtype, registration status, whether overwritten, etc.
            result = {
                'mtype': mtype,
                'strategy_registered': False,
                'template_registered': False,
                'is_complete': False,
                'overwrote_existing': {
                    'strategy': existing_strategy and strategy is not None,
                    'template': existing_template and template is not None
                },
                'timestamp': self._get_timestamp()
            }

            try:
                # Registers strategies and templates:
                # If a strategy class is provided, it is registered in the self.strategies dictionary.
                # If a template class is provided, it is registered in the self.templates dictionary.
                # If a name conflict exists, a warning will be issued and the existing entry will be overwritten.
                if strategy is not None:
                    if existing_strategy:
                        logger.warning(f"Overwriting existing strategy for mtype '{mtype}'")
                        self._registration_stats['overwrites'] += 1
                    self.strategies[mtype] = strategy
                    result['strategy_registered'] = True
                    logger.debug(f"Registered strategy: {strategy.__name__} for mtype '{mtype}'")

                if template is not None:
                    if existing_template:
                        logger.warning(f"Overwriting existing template for mtype '{mtype}'")
                        self._registration_stats['overwrites'] += 1
                    self.templates[mtype] = template
                    result['template_registered'] = True
                    logger.debug(f"Registered template: {template.__name__} for mtype '{mtype}'")

                # Checks completeness:
                result['is_complete'] = (mtype in self.strategies and mtype in self.templates)

                # Updates statistics:
                self._registration_stats['total_registrations'] += 1

                # Notifies observers:
                # Calls the _notify_observers method to notify all registered observers
                # that a new registration event has occurred.
                # Observers can perform corresponding logic (e.g., logging, cache updates) based on this notification.
                self._notify_observers('register', result)

                logger.info(f"Successfully registered mtype '{mtype}' (complete: {result['is_complete']})")

                return result

            except Exception as e:
                self._registration_stats['failed_registrations'] += 1
                logger.error(f"Registration failed for mtype '{mtype}': {e}")
                raise

    @staticmethod
    def _validate_strategy_class(strategy: Type[FuzznumStrategy]) -> None:
        """
        Validates a given strategy class.

        Args:
            strategy (Type[FuzznumStrategy]): The strategy class to validate.

        Raises:
            TypeError: If `strategy` is not a class or not a subclass of `FuzznumStrategy`.
            ValueError: If the strategy class does not define an `mtype` attribute.
        """
        # Checks: Is the passed `strategy` a class (not an instance or other type)?
        if not isinstance(strategy, type):
            raise TypeError(f"Strategy must be a class, got {type(strategy).__name__}")

        # Checks: Is the passed `strategy` a subclass of `FuzznumStrategy`?
        # This is crucial for ensuring the strategy class conforms to the interface definition.
        if not issubclass(strategy, FuzznumStrategy):
            raise TypeError(f"Strategy must be a subclass of FuzznumStrategy, got {strategy.__name__}")

        # Checks: Does the strategy class define an `mtype` attribute?
        # `mtype` is the unique identifier for the fuzzy number type, essential for registration and lookup.
        if not hasattr(strategy, 'mtype'):
            raise ValueError(f"Strategy class {strategy.__name__} must define 'mtype' attribute")

    @staticmethod
    def _validate_template_class(template: Type[FuzznumTemplate]) -> None:
        """
        Validates a given template class.

        Args:
            template (Type[FuzznumTemplate]): The template class to validate.

        Raises:
            TypeError: If `template` is not a class or not a subclass of `FuzznumTemplate`.
            ValueError: If the template class does not define an `mtype` attribute.
        """
        # Checks: Is the passed `template` a class?
        if not isinstance(template, type):
            raise TypeError(f"Template must be a class, got {type(template).__name__}")

        # Checks: Is the passed `template` a subclass of `FuzznumTemplate`?
        # This is crucial for ensuring the template class conforms to the interface definition.
        if not issubclass(template, FuzznumTemplate):
            raise TypeError(f"Template must be a subclass of FuzznumTemplate, got {template.__name__}")

        # Checks: Does the template class define an `mtype` attribute?
        # `mtype` is the unique identifier for the fuzzy number type, essential for registration and lookup.
        if not hasattr(template, 'mtype'):
            raise ValueError(f"Template class {template.__name__} must define 'mtype' attribute")

    @staticmethod
    def _extract_mtype(strategy: Optional[Type[FuzznumStrategy]],
                       template: Optional[Type[FuzznumTemplate]]) -> str:
        """
        Extracts the fuzzy number type (mtype) string from a strategy or template class.

        Core logic: This method extracts the `mtype` string from the provided strategy or template class.
        It prioritizes getting the `mtype` from `strategy`; if `strategy` is None, it gets it from `template`.

        Args:
            strategy (Optional[Type[FuzznumStrategy]]): The strategy class.
            template (Optional[Type[FuzznumTemplate]]): The template class.

        Returns:
            str: The extracted mtype string.

        Raises:
            ValueError: If both strategy and template are None.
        """
        if strategy is not None:
            return str(strategy.mtype)
        elif template is not None:
            return str(template.mtype)
        else:
            # If both strategy and template are None, mtype cannot be extracted, raise an error.
            raise ValueError("Cannot extract mtype: both strategy and template are None")

    @staticmethod
    def _get_timestamp():
        """
        Generates a formatted timestamp string.

        Returns:
            str: The current timestamp in "YYYYMMDDHHMMSS.ffffff" format.
        """
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")

    def batch_register(self, registrations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Performs transactional batch registration of multiple fuzzy number types.

        This method allows registering multiple fuzzy number strategies and/or
        templates in an atomic operation. If any registration in the batch fails,
        all changes made during that batch operation will be rolled back,
        ensuring registry consistency.

        Args:
            registrations (List[Dict[str, Any]]): A list of dictionaries, where
                each dictionary specifies the strategy and/or template to register.
                Each dictionary should contain 'strategy' (Type[FuzznumStrategy])
                and/or 'template' (Type[FuzznumTemplate]) keys, corresponding to
                the classes to be registered.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are `mtype` strings
                and values are the registration results for each type. If a specific
                registration within the batch fails, even if the entire transaction
                is rolled back, the returned dictionary might contain an entry with
                a key like "error_N" (where N is the index in the input list) and
                a value detailing the error.

        Raises:
            TypeError: If `registrations` is not a list, or any item in the list
                       is not a dictionary.
            Exception: If any single registration within the batch fails, the
                       original exception (e.g., `ValueError`, `TypeError`) will
                       be re-raised after the transaction rollback. This means
                       if `batch_register` raises an exception, the registry's
                       state will be restored to its state before the method call.
        """
        if not isinstance(registrations, list):
            raise TypeError(f"Registrations must be a list, got {type(registrations).__name__}")

        results = {}

        with self.transaction():
            # Core logic: Wraps the entire batch registration process with the `self.transaction()` context manager.
            # This ensures that all registration operations within the `with` block are atomic:
            # if any `register` call fails, the entire `batch_register` operation will be rolled back,
            # and the registry will revert to its state before the `batch_register` call.

            for i, registration in enumerate(registrations):
                # Iterates through the list of registration requests.
                # Checks: Ensures that each element in the list is a dictionary.
                if not isinstance(registration, dict):
                    raise TypeError(f"Each registration must be a dict, got {type(registration).__name__} at index {i}")

                strategy = registration.get('strategy')
                template = registration.get('template')

                try:
                    result = self.register(strategy=strategy, template=template)
                    # Calls the `register` method to handle a single registration request.
                    results[result['mtype']] = result
                    # Stores the result of the single registration in the `results` dictionary.

                except Exception as e:
                    # Error handling: Catches exceptions if a single `register` call fails.
                    # Logs the error message and re-raises the exception.
                    # Key point: Re-raising the exception will cause the `self.transaction()` context manager
                    # to catch the exception, thereby triggering the entire transaction's rollback mechanism.
                    error_info = {
                        'error': str(e),
                        'index': i,
                        'timestamp': self._get_timestamp()
                    }
                    results[f"error_{i}"] = error_info
                    raise

        return results

    def unregister(self, mtype: str,
                   remove_strategy: bool = True,
                   remove_template: bool = True) -> Dict[str, Any]:
        """
        Unregisters a fuzzy number type (strategy and/or template) from the registry.

        Args:
            mtype (str): The mtype of the fuzzy number to unregister.
            remove_strategy (bool): Whether to remove the associated strategy class (defaults to True).
            remove_template (bool): Whether to remove the associated template class (defaults to True).

        Returns:
            Dict[str, Any]: A dictionary containing the unregistration result.

        Raises:
            TypeError: If `mtype` is not a string.
        """

        # Checks: Ensures that the passed `mtype` is a string.
        if not isinstance(mtype, str):
            raise TypeError(f"mtype must be a string, got {type(mtype).__name__}")

        with self._lock:
            # Thread safety: Acquires a registry-level lock.
            # This ensures that modifications to the registry's internal data structures
            # (self.strategies, self.templates) are atomic in a multithreaded environment.

            # Prepares the unregistration result dictionary:
            # Initializes a dictionary to record detailed results of this unregistration operation,
            # including mtype, removal status, whether it was complete before unregistration, etc.
            result = {
                'mtype': mtype,
                'strategy_removed': False,
                'template_removed': False,
                'was_complete': (mtype in self.strategies and mtype in self.templates),
                'timestamp': self._get_timestamp()
            }

            # Removes strategy:
            # If `remove_strategy` is True and a strategy exists for the `mtype`, it is deleted from `self.strategies`.
            if remove_strategy and mtype in self.strategies:
                del self.strategies[mtype]
                result['strategy_removed'] = True
                logger.debug(f"Removed strategy for mtype '{mtype}'")

            # Removes template:
            # If `remove_template` is True and a template exists for the `mtype`, it is deleted from `self.templates`.
            if remove_template and mtype in self.templates:
                del self.templates[mtype]
                result['template_removed'] = True
                logger.debug(f"Removed template for mtype '{mtype}'")

            # Records history:
            # Adds a copy of the detailed result of this unregistration operation to the registration history list,
            # for tracking and auditing.
            self._registration_history.append(result.copy())

            # Notifies observers:
            # Calls the _notify_observers method to notify all registered observers that a new unregistration event has occurred.
            self._notify_observers('unregister', result)

            logger.info(
                f"Unregistered mtype '{mtype}' (strategy: {result['strategy_removed']}, template: {result['template_removed']})")

            return result

    # ======================== Introspection Methods ========================

    def get_strategy(self, mtype: str) -> Type[FuzznumStrategy]:
        """
        Retrieves the strategy class for a given `mtype`.

        Args:
            mtype (str): The fuzzy number type identifier.

        Returns:
            Type[FuzznumStrategy]: The corresponding strategy class.

        Raises:
            ValueError: If no strategy class is found for the specified `mtype`.
        """
        # This method retrieves and returns the corresponding strategy class from the
        # registered strategies dictionary based on the fuzzy number type identifier `mtype`.
        # It is a key pathway for the Fuzznum object to obtain specific strategy implementations during initialization.
        strategy_cls = self.strategies.get(mtype)
        if strategy_cls is None:
            raise ValueError(f"Strategy for mtype '{mtype}' not found in registry.")
        return strategy_cls
        # Directly uses the dictionary's `get()` method. If `mtype` exists, it returns the corresponding strategy class;
        # if not, it returns `None`, avoiding a KeyError.

    def get_template(self, mtype: str) -> Type[FuzznumTemplate]:
        """
        Retrieves the template class for a given `mtype`.

        Args:
            mtype (str): The fuzzy number type identifier.

        Returns:
            Type[FuzznumTemplate]: The corresponding template class.

        Raises:
            ValueError: If no template class is found for the specified `mtype`.
        """
        # This method retrieves and returns the corresponding template class from the
        # registered templates dictionary based on the fuzzy number type identifier `mtype`.
        # It is a key pathway for the Fuzznum object to obtain specific template implementations during initialization.
        template_cls = self.templates.get(mtype)
        if template_cls is None:
            raise ValueError(f"Template for mtype '{mtype}' not found in registry.")
        return template_cls
        # Similarly uses the dictionary's `get()` method. If `mtype` exists, it returns the corresponding template class;
        # if not, it returns `None`.

    def get_registered_mtypes(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves information about all registered fuzzy number types.

        This method provides a comprehensive overview, listing all fuzzy number
        types that have appeared in the registry, and indicating whether each
        type has a strategy, a template, and whether it is "complete" (i.e.,
        both strategy and template exist).

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing information for
                                       all registered types.
        """
        all_mtypes = set(self.strategies.keys()) | set(self.templates.keys())
        # Core logic: By performing a union operation on the key sets of the `strategies`
        #   dictionary and the `templates` dictionary, it obtains a set of all unique `mtype`s
        #   that have appeared in the registry. This ensures that types with only a strategy
        #   or only a template are also included.

        result = {}
        # Initializes an empty dictionary to store the final type information.

        for mtype in all_mtypes:
            has_strategy = mtype in self.strategies
            has_template = mtype in self.templates

            result[mtype] = {
                'has_strategy': has_strategy,
                'has_template': has_template,
                'strategy_class': self.strategies[mtype].__name__ if has_strategy else None,
                # If a strategy exists, records the name of the strategy class; otherwise, None.
                'template_class': self.templates[mtype].__name__ if has_template else None,
                # If a template exists, records the name of the template class; otherwise, None.
                'is_complete': has_strategy and has_template
                # A type is considered "complete" only if both strategy and template exist.
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
            # Total number of registered strategy classes.
            'total_templates': len(self.templates),
            # Total number of registered template classes.
            'complete_types': len(set(self.strategies.keys()) & set(self.templates.keys())),
            # Core logic: By performing an intersection operation on the key sets of `strategies`
            #   and `templates`, it obtains the number of `mtype`s that have both a strategy
            #   and a template, i.e., the number of "complete" types.
            'registration_stats': self._registration_stats.copy(),
            # Returns a copy of the internal statistics dictionary `_registration_stats`,
            # containing counts for successful registrations, failed registrations, overwrites, etc.
            # A copy is returned to prevent external direct modification of internal statistics.
            'observer_count': len(self._observers)
            # Number of currently registered observers.
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Retrieves the health status of the registry.

        This method assesses the "health" of the registry, identifying potential
        issues such as incomplete or missing type definitions.

        Returns:
            Dict[str, Any]: A dictionary indicating the health status, including
                            `is_healthy` (boolean), issues, and warnings.
        """
        complete_types = set(self.strategies.keys()) & set(self.templates.keys())
        # Core logic: Calculates the set of `mtype`s that have both a strategy and a template.

        incomplete_types = (set(self.strategies.keys()) | set(self.templates.keys())) - complete_types
        # Core logic: Calculates the set of all `mtype`s that have appeared (union)
        #   minus the set of complete types, to get the "incomplete" types that
        #   have only a strategy or only a template.

        return {
            'is_healthy': len(incomplete_types) == 0,
            # The registry is considered healthy if there are no incomplete types.
            'total_types': len(self.strategies) + len(self.templates),
            # Total number of strategies and templates (may include duplicates).
            'complete_types': list(complete_types),
            # List of `mtype`s for complete types.
            'incomplete_types': list(incomplete_types),
            # List of `mtype`s for incomplete types.
            'missing_strategies': list(set(self.templates.keys()) - set(self.strategies.keys())),
            # Core logic: Calculates the list of `mtype`s that have only a template but no strategy.
            # This indicates that a template exists but its corresponding algorithm implementation is missing.
            'missing_templates': list(set(self.strategies.keys()) - set(self.templates.keys())),
            # Core logic: Calculates the list of `mtype`s that have only a strategy but no template.
            # This indicates that an algorithm implementation exists but its corresponding representation is missing.
            'error_rate': (self._registration_stats['failed_registrations'] /
                           max(1, self._registration_stats['total_registrations']))
            # Calculates the registration failure rate. `max(1, ...)` is used to prevent division by zero
            # if `total_registrations` is 0.
        }


# ======================== Global Singleton and Factory Method ========================

# Global registry instance
_registry_instance: Optional[FuzznumRegistry] = None
_registry_lock = threading.RLock()


def get_fuzznum_registry() -> FuzznumRegistry:
    """
    Retrieves the global singleton instance of `FuzznumRegistry`.

    This is a factory function that ensures only one instance of `FuzznumRegistry`
    is returned throughout the application, regardless of how many times it is called.

    Returns:
        FuzznumRegistry: The global unique instance of the fuzzy number registry.

    Examples:
        >>> registry_instance = get_fuzznum_registry()
        >>> print(registry_instance) # Prints the registry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = FuzznumRegistry()

    return _registry_instance


# Convenient global functions
def register_fuzznum(strategy: Optional[Type[FuzznumStrategy]] = None,
                     template: Optional[Type[FuzznumTemplate]] = None) -> Dict[str, Any]:
    """
    Global registration function: Registers a single fuzzy number strategy and/or template.

    This function is a convenient wrapper around `get_registry().register()`.

    Args:
        strategy (Optional[Type[FuzznumStrategy]]): The fuzzy number strategy class to register.
        template (Optional[Type[FuzznumTemplate]]): The fuzzy number template class to register.

    Returns:
        Dict[str, Any]: A dictionary containing the registration result.

    Examples:
        # >>> from mohupy.core.base import ExampleStrategy, ExampleTemplate
        >>> # Register a new type
        >>> result = register_fuzznum(strategy=ExampleStrategy, template=ExampleTemplate)
        >>> print(result['mtype'], result['is_complete'])
        my_type True
        >>> # Verify if registered
        >>> print(get_fuzznum_registry().get_fuzznum_registered_mtypes().get('my_type', {}).get('is_complete'))
        True
    """
    return get_fuzznum_registry().register(strategy=strategy, template=template)


def batch_register_fuzznums(registrations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Global batch registration function: Performs transactional batch registration
    of multiple fuzzy number types.

    This function is a convenient wrapper around `get_registry().batch_register()`.

    Args:
        registrations (List[Dict[str, Any]]): A list of dictionaries, where each
                                               dictionary specifies the strategy
                                               and/or template to register.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where keys are `mtype` strings
                                   and values are the registration results for
                                   each type.

    Examples:
        >>> # Assuming MyStrategyA, MyTemplateA, MyStrategyB, MyTemplateB are defined as follows:
        >>> # class MyStrategyA(FuzznumStrategy): mtype = "type_a"; pass
        >>> # class MyTemplateA(FuzznumTemplate): mtype = "type_a"; def report(self): return ""; def str(self): return ""
        >>> # class MyStrategyB(FuzznumStrategy): mtype = "type_b"; pass
        >>> # class MyTemplateB(FuzznumTemplate): mtype = "type_b"; def report(self): return ""; def str(self): return ""
        >>>
        >>> # Build a list of batch registration requests
        >>> registrations_list = [
        ...     {'strategy': MyStrategyA, 'template': MyTemplateA},
        ...     {'strategy': MyStrategyB, 'template': MyTemplateB}
        ... ]
        >>>
        >>> # Execute batch registration
        >>> results = batch_register_fuzznums(registrations_list)
        >>> print(results['type_a']['is_complete'], results['type_b']['is_complete'])
        (True, True)
        >>> # Verify successful registration
        >>> print(get_fuzznum_registry().get_fuzznum_registered_mtypes().get('type_a', {}).get('is_complete'))
        True
    """
    return get_fuzznum_registry().batch_register(registrations)


def unregister_fuzznum(mtype: str,
                       remove_strategy: bool = True,
                       remove_template: bool = True) -> Dict[str, Any]:
    """
    Global unregistration function: Unregisters a fuzzy number type from the registry.

    This function is a convenient wrapper around `get_registry().unregister()`.

    Args:
        mtype (str): The fuzzy number type identifier to unregister.
        remove_strategy (bool): Whether to remove the corresponding strategy class (defaults to True).
        remove_template (bool): Whether to remove the corresponding template class (defaults to True).

    Returns:
        Dict[str, Any]: A dictionary containing the unregistration result.

    Examples:
        >>> result = unregister_fuzznum("my_type")
        >>> print(result['mtype'], result['strategy_removed'], result['template_removed'])
        my_type, True, True
        >>> # Verify if unregistered
        >>> print(get_fuzznum_registry().get_fuzznum_registered_mtypes().get('my_type'))
        None
    """
    return get_fuzznum_registry().unregister(
        mtype=mtype,
        remove_strategy=remove_strategy,
        remove_template=remove_template
    )


def get_strategy(mtype: str) -> Optional[Type[FuzznumStrategy]]:
    """
    Global get strategy function: Retrieves the strategy class for a given `mtype`.

    This function is a convenient wrapper around `get_fuzznum_registry().get_strategy()`.

    Args:
        mtype (str): The fuzzy number type identifier.

    Returns:
        Optional[Type[FuzznumStrategy]]: The corresponding strategy class, or `None` if not found.

    Examples:
        >>> strategy_cls = get_strategy("my_type")
        >>> print(strategy_cls.__name__)
        MyStrategy
        >>> # Get a non-existent strategy class
        >>> non_existent_strategy = get_strategy("non_existent_type")
        >>> print(non_existent_strategy)
        None
    """
    try:
        return get_fuzznum_registry().get_strategy(mtype)
    except ValueError:
        return None


def get_template(mtype: str) -> Optional[Type[FuzznumTemplate]]:
    """
    Global get template function: Retrieves the template class for a given `mtype`.

    This function is a convenient wrapper around `get_fuzznum_registry().get_template()`.

    Args:
        mtype (str): The fuzzy number type identifier.

    Returns:
        Optional[Type[FuzznumTemplate]]: The corresponding template class, or `None` if not found.

    Examples:
        >>> # Get template class
        >>> template_cls = get_template("my_type")
        >>> print(template_cls.__name__)
        MyTemplate
        >>> # Get a non-existent template class
        >>> non_existent_template = get_template("non_existent_type")
        >>> print(non_existent_template)
        None
    """
    try:
        return get_fuzznum_registry().get_template(mtype)
    except ValueError:
        return None


def get_fuzznum_registered_mtypes() -> Dict[str, Dict[str, Any]]:
    """
    Global get registered types function: Retrieves information about all registered fuzzy number types.

    This function is a convenient wrapper around `get_fuzznum_registry().get_registered_mtypes()`.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing information for all registered types.

    Examples:
        >>> # Get all registered types
        >>> all_types = get_fuzznum_registered_mtypes()
        >>> # Print information for 'some_type' (if registered)
        >>> print(all_types.get('some_type', {}).get('is_complete'))
        True
        >>> # Print all registered mtype keys
        >>> print(sorted(list(all_types.keys())))
        ['ivqfn', 'qrofn', 'some_type'] # May include default registered types
    """
    return get_fuzznum_registry().get_registered_mtypes()
