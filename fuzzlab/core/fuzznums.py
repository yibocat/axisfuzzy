#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 22:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import difflib
from typing import Optional, Dict, Callable, Any, Set, List

from fuzzlab.config import get_config
from fuzzlab.core.base import FuzznumTemplate, FuzznumStrategy
from fuzzlab.core.registry import get_registry


class Fuzznum:

    _INTERNAL_ATTRS = {
        'mtype',
        '_initialized',
        '_strategy_instance',
        '_strategy_instance',
        '_bound_strategy_methods',
        '_bound_strategy_attributes',
        '_bound_template_methods',
        '_bound_template_attributes',
    }

    def __init__(self,
                 mtype: Optional[str] = None,
                 qrung: Optional[int] = None):
        object.__setattr__(self, '_initialized', False)
        config = get_config()
        if mtype is None:
            mtype = config.DEFAULT_MTYPE

        if qrung is None:
            qrung = 1

        if not isinstance(mtype, str):
            raise TypeError(f"mtype must be a string type, got '{type(mtype).__name__}'")

        if not isinstance(qrung, int):
            raise TypeError(f"qrung must be an integer, got '{type(qrung).__name__}'")

        object.__setattr__(self, 'mtype', mtype)
        object.__setattr__(self, 'q', qrung)

        try:
            self._initialize()
            object.__setattr__(self, '_initialized', True)

        except Exception:
            self._cleanup_partial_initialization()
            raise

    def _initialize(self):
        self._configure_strategy()
        self._configure_template()

    def _configure_strategy(self):
        registry = get_registry()

        if self.mtype not in registry.strategies:
            available_mtypes = ', '.join(registry.strategies.keys())
            raise ValueError(
                f"Unsupported strategy mtype: '{self.mtype}'."
                f"Available mtypes: {available_mtypes}"
            )

        # Retrieve and instantiate the FuzznumStrategy class corresponding to
        #   the current mtype from the registry. For example, if the mtype is 'qrofn',
        #   an instance of QROFNStrategy will be created here. Call the
        #   `_bind_instance_members` helper method to bind the public methods and
        #   properties of the strategy instance to the current Fuzznum instance.
        #   This way, users can directly access the methods and properties of the
        #   strategy through the Fuzznum instance (e.g., `fuzznum_instance.md` or
        #   `fuzznum_instance.calculate_value()`), without having to get the strategy
        #   instance first, achieving transparent property delegation.
        strategy_instance = registry.strategies[self.mtype](self.q)
        bound_methods, bound_attributes = self._bind_instance_members(
            strategy_instance, 'strategy'
        )
        object.__setattr__(self, '_strategy_instance', strategy_instance)
        object.__setattr__(self, '_bound_strategy_methods', bound_methods)
        object.__setattr__(self, '_bound_strategy_attributes', bound_attributes)

    def _configure_template(self) -> None:
        registry = get_registry()

        if self.mtype not in registry.templates:
            available_templates = ', '.join(registry.templates.keys())
            raise ValueError(
                f"Unsupported template mtype: '{self.mtype}'."
                f"Available templates: {available_templates}"
            )

        # Retrieve and instantiate the FuzznumTemplate class corresponding to the
        #   current mtype from the registry. Key point: When instantiating the
        #   template, pass the current Fuzznum instance, `self`, as an argument
        #   to the template's constructor. This allows the template instance to
        #   hold a reference (usually a weak reference) to its associated Fuzznum
        #   instance, thereby accessing the Fuzznum instance's attributes and
        #   methods within the template (for example, `self.instance.md`). Call the
        #   `_bind_instance_members` helper method to bind the public methods and
        #   attributes of the template instance to the current Fuzznum instance.
        #   This way, users can directly access the template's methods and attributes
        #   through the Fuzznum instance (for example, `fuzznum_instance.report()` or
        #   `fuzznum_instance.str()`).
        template_instance = registry.templates[self.mtype](self)
        bound_methods, bound_attributes = self._bind_instance_members(
            template_instance, 'template'
        )

        # 使用 `object.__setattr__` 安全地存储模板实例和绑定成员的信息。
        object.__setattr__(self, '_template_instance', template_instance)
        object.__setattr__(self, '_bound_template_methods', bound_methods)
        object.__setattr__(self, '_bound_template_attributes', bound_attributes)

    def _bind_instance_members(self,
                               instance: Any,
                               instance_type: str) -> tuple[Dict[str, Callable[..., Any]], Set[str]]:
        """
        Bind instance methods and properties to the current object.

        This method is the key to the Fuzznum class implementing the "Facade Pattern."
        It is responsible for dynamically "copying" or "binding" the public methods
        and properties of the underlying FuzznumStrategy or FuzznumTemplate instance
        to the Fuzznum instance itself. This allows external users to directly access
        these methods and properties through the Fuzznum object, without needing to
        interact directly with the strategy or template instance, thereby simplifying
        the API and improving the level of abstraction.

        Args:
            instance: The strategy or template instance to bind its members.
            instance_type: The type of the instance, used to distinguish processing
                (for example, 'strategy' or 'template').

        Returns:
            tuple[Dict[str, Callable[..., Any]], Set[str]]:
                A tuple containing two elements:
                    1. Bound_methods (Dict[str, Callable[..., Any]]): A dictionary
                    of methods bound to a Fuzznum instance, where the keys are method
                    names and the values are the methods themselves.
                    2. Bound_attributes (Set[str]): The set of property names bound
                    to the Fuzznum instance.

        Raises:
            RuntimeError: If any exceptions occur during the binding process.
        """
        bound_methods: Dict[str, Callable[..., Any]] = {}
        bound_attributes: Set[str] = set()

        # For template instances, 'mtype' and 'instance' (a reference pointing to
        #   Fuzznum itself) generally should not be bound directly, because they
        #   are internally managed by the template or are special attributes
        #   associated with Fuzznum.
        exclude_attrs = {'mtype', 'instance'} if instance_type == 'template' else {'mtype'}

        try:
            for attr_name in dir(instance):
                if attr_name.startswith('_') or attr_name in exclude_attrs:
                    continue

                # Get attribute descriptor to determine if it is a property.
                # `getattr(instance.__class__, attr_name, None)` attempt to get
                #   attributes from the class perspective. This is useful for
                #   identifying `property`, because `property` is a class-level
                #   descriptor.
                attr_descriptor = getattr(instance.__class__, attr_name, None)
                if isinstance(attr_descriptor, property):
                    bound_attributes.add(attr_name)
                else:
                    attr_value = getattr(instance, attr_name)
                    if callable(attr_value):
                        object.__setattr__(self, attr_name, attr_value)
                        bound_methods[attr_name] = attr_value
                    else:
                        bound_attributes.add(attr_name)

            return bound_methods, bound_attributes

        except Exception as e:
            raise RuntimeError(f"{instance_type} '{self.mtype}' dynamic binding failed: {e}")

    def _cleanup_partial_initialization(self) -> None:
        """
        Clean up some initialized states

        When the constructor `__init__` of a Fuzznum instance encounters an exception
        during execution, this method is called. Its purpose is to clean up or reset
        internal properties that may have been partially initialized but not fully
        configured, to prevent the instance from being in an inconsistent or corrupted
        state. This helps avoid resource leaks or unexpected behavior in later
        operations after initialization fails.
        """
        cleanup_attrs = [
            '_strategy_instance',
            '_template_instance',
            '_bound_strategy_methods',
            '_bound_strategy_attributes',
            '_bound_template_methods',
            '_bound_template_attributes',
        ]

        for attr in cleanup_attrs:
            try:
                object.__delattr__(self, attr)
            except AttributeError:
                pass

    def _is_initialized(self) -> bool:
        """
        Safely check the initialization status to avoid recursive calls.

        This method provides a safe, side-effect-free way to query the initialization
        status of a Fuzznum instance. It is called by methods such as `__getattr__` and
        `__setattr__`, to determine whether the instance has completed all
        initialization steps. This is crucial for preventing recursion or errors
        caused by accessing or modifying attributes before the object is fully prepared.

        Returns:
            bool: Returns True if the instance has been fully initialized,
                otherwise returns False.
        """
        try:
            return object.__getattribute__(self, '_initialized')
        except AttributeError:
            return False

    def get_template_instance(self) -> FuzznumTemplate:
        """
        Get Template Instance

        This method provides a protected interface to securely access the
        `FuzznumTemplate` instance associated with the Fuzznum instance. It is
        called when there is a need to access template-specific functionalities
        such as `report()` or `str()`. By centralizing access management, it ensures
        that clear error messages can be provided when the instance does not exist
        or is not fully initialized.

        Returns:
            FuzznumTemplate: Associated template instance.

        Raises:
            RuntimeError: If the template instance is not found or has not been
                fully initialized.
        """
        try:
            template_instance = object.__getattribute__(self, '_template_instance')
            if template_instance is None:
                raise RuntimeError("Template instance not found.")
            return template_instance
        except AttributeError:
            raise RuntimeError("Template instance not found.")

    def get_strategy_instance(self) -> FuzznumStrategy:
        """
        Get strategy instance

        This method provides a protected interface to securely access the
        `FuzznumStrategy` instance associated with a Fuzznum instance. It is called
        when there is a need to access strategy-specific features, such as property
        values or core algorithms. Similar to `get_template_instance`, it ensures
        clear error messages are provided if the instance does not exist or has not
        been fully initialized.

        Returns:
            FuzznumStrategy: Associated strategy instance.

        Raises:
            RuntimeError: If the strategy instance is not found or has not been
                fully initialized.
        """
        try:
            strategy_instance = object.__getattribute__(self, '_strategy_instance')
            if strategy_instance is None:
                raise RuntimeError("Strategy instance not found.")
            return strategy_instance
        except AttributeError:
            raise RuntimeError("Strategy instance not found.")

    def _delegate_attribute_access(self, name: str) -> Any:
        """
        The specific implementation of delegated property access

        This method is key to how the Fuzznum class implements its core "attribute delegation"
        mechanism. When `__getattribute__` cannot find an attribute directly on a Fuzznum
        instance, it will invoke this method to attempt to retrieve the attribute value from
        the associated strategy instance (`_strategy_instance`) or template instance
        (`_template_instance`). It also integrates caching lookup and performance monitoring.

        Args:
            name: The name of the attribute to access.

        Returns:
            Any: Found attribute value.

        Raises:
            AttributeError: If the attribute is not found in the Fuzznum instance,
                strategy instance, and template instance.
            RuntimeError: If the strategy or template instance is not properly initialized.
        """

        try:
            bound_strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')
            # 如果要访问的属性名在策略属性集合中。
            if name in bound_strategy_attrs:
                # 从策略实例中获取属性值。
                # 调用 `get_strategy_instance()` 确保获取到有效的策略实例。
                value = getattr(self.get_strategy_instance(), name)
                return value

            # Gets the collection of template property names bound to the Fuzznum instance.
            bound_template_attrs = object.__getattribute__(self, '_bound_template_attributes')
            # If the attribute name to be accessed is in the template attribute set.
            #   Get property values from the template instance.
            #   Call `get_template_instance()` to ensure obtaining a valid template instance.
            if name in bound_template_attrs:
                value = getattr(self.get_template_instance(), name)
                return value

        except (AttributeError, RuntimeError):
            # Catch `AttributeError` or `RuntimeError` that may occur when attempting
            #   to get a strategy/template instance or its properties. This usually indicates
            #   that the instance is not initialized or the property does not exist in the
            #   strategy/template. Silently ignore it here to pass control to `__getattr__` for
            #   final handling.
            pass

        return self.__getattr__(name)

    def __getattribute__(self, name: str) -> Any:
        """
        Optimized attribute access method

        `__getattribute__` is the first entry point for attribute lookup in Python.
        Every time an instance's attribute is accessed, this method is called first,
        regardless of whether the attribute exists. It is responsible for intercepting
        all attribute accesses and processing them differently based on the attribute
        type (internal attributes, special methods, own attributes, delegated attributes),
         to implement complex attribute delegation and avoid recursion.

        Args:
            name: The name of the attribute to access (string).

        Returns:
            Any: The accessed attribute value.

        Raises:
            AttributeError: If the attribute is ultimately not found.
            RuntimeError: If an object attempts to access an uninitialized property during initialization.
        """
        # Internal attributes are accessed directly to avoid recursion. For internal
        #   attributes defined in `_INTERNAL_ATTRS`, they are set directly via
        #   `object.__setattr__` during the `__init__` phase. To prevent triggering
        #   `__getattribute__` again when accessing these attributes internally, which
        #   could lead to infinite recursion, we directly use `object.__getattribute__(self, name)`
        #   to retrieve them.
        if name in Fuzznum._INTERNAL_ATTRS:
            return object.__getattribute__(self, name)

        # Special methods and private attributes access For Python special methods
        #   (such as `__str__`, `__repr__`, etc.) and private attributes starting with a
        #   single underscore, they usually do not participate in Fuzznum's attribute
        #   delegation logic. Directly use `object.__getattribute__` to access them,
        #   which can improve efficiency and avoid unnecessary complex processing.
        if name.startswith('_') or name in ('__dict__', '__class__'):
            return object.__getattribute__(self, name)

        # First, attempt to find the attribute directly from the Fuzznum instance
        #   itself (i.e., `self.__dict__`). If the attribute exists directly on the
        #   Fuzznum instance, return its value directly. This approach prioritizes attributes
        #   belonging to the Fuzznum instance itself, avoiding the overhead of delegation.
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        if not self._is_initialized():
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'."
                f"The Fuzznum is still initializing or the property does not exist."
            )

        return self._delegate_attribute_access(name)

    def __getattr__(self, name: str) -> Any:
        """
        Final handling method for attribute access

        `__getattr__` is the "last line of defense" in Python's attribute lookup mechanism.
        When `__getattribute__` (and the `_delegate_attribute_access` it calls
        internally) fails to find the requested attribute, `__getattr__` is invoked.
        Its main responsibilities are handling dynamically generated attributes, or
        providing a detailed and helpful `AttributeError` message when the attribute
        truly does not exist.

        Args:
            name: Property name that was attempted to be accessed but not found.

        Returns:
            Any: (Theoretically, if dynamic properties are found) The found property value.

        Raises:
            AttributeError: The attribute was not found in any of the possible lookup paths.
        """
        if not self._is_initialized():
            raise AttributeError(
                f"'{self.__class__.__name__}' 对象没有属性 '{name}'。"
                f"The Fuzznum is still initializing."
            )

        available_info = self._get_available_members_info()

        # Construct detailed error messages
        error_msg = f"'{self.__class__.__name__}' object has no attribute '{name}'."
        all_members = available_info['attributes'] + available_info['methods']
        suggestions = difflib.get_close_matches(name, all_members, n=3, cutoff=0.6)
        if suggestions:
            error_msg += f" Did you mean: {', '.join(suggestions)}?"

        # Show all available members only in debug mode to avoid information overload
        config = get_config()
        if getattr(config, 'DEBUG_MODE', False):
            if available_info['attributes']:
                error_msg += f"\nAvailable attributes: {', '.join(sorted(available_info['attributes']))}."

            if available_info['methods']:
                error_msg += f"\nAvailable methods: {', '.join(sorted(available_info['methods']))}."

        raise AttributeError(error_msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Attribute Setting Method

        `__setattr__` is the interception point for attribute assignment in Python.
        This method is called every time an attempt is made to set an attribute of an
        instance (e.g., `obj.attr = value`). It is responsible for handling different
        types of attributes (internal attributes, immutable attributes, delegated
        attributes) appropriately, and implements attribute validation and delegation.

        Args:
            name: The name of the property to set.
            value: The value to assign to the property.

        Raises:
            AttributeError: If attempting to modify an immutable property,
                or the property is read-only.
            RuntimeError: An unexpected error occurred during the delegate setup process.
        """
        # Direct assignment during initialization or for internal attributes
        # Similar to `__getattribute__`, this directly uses `object.__setattr__` for
        #   internal attributes defined in `_INTERNAL_ATTRS`, or when the object has not
        #   been fully initialized yet. This ensures these core attributes can be
        #   reliably set without triggering custom logic or causing recursion.
        if (name in Fuzznum._INTERNAL_ATTRS or
                name.startswith('_') or not self._is_initialized()):
            object.__setattr__(self, name, value)
            return

        if name == 'mtype':
            raise AttributeError(f"Cannot modify immutable attribute '{name}' of Fuzznum instance.")

        try:
            strategy_attributes = object.__getattribute__(self, '_bound_strategy_attributes')

            if name in strategy_attributes:
                try:
                    strategy_instance = self.get_strategy_instance()
                    # Attempt to get the attribute descriptor from the perspective
                    #   of the strategy class to determine whether it is a `property`.
                    strategy_class = strategy_instance.__class__
                    attr_descriptor = getattr(strategy_class, name, None)

                    # If it is a `property` and `fset` (setter method) is defined,
                    #   then invoke its setter through `setattr`.
                    if isinstance(attr_descriptor, property):
                        if attr_descriptor.fset:
                            setattr(strategy_instance, name, value)
                            return
                        else:
                            # If it is a `property` but `fset` is not defined,
                            #   then the property is read-only.
                            raise AttributeError(f"The attribute '{name}' is read-only "
                                                 f"for the fuzzy number mtype '{self.mtype}'.")
                    else:
                        # If it is not `property`, directly set the attribute of the
                        #   strategy instance through `setattr`.
                        setattr(strategy_instance, name, value)
                        object.__setattr__(self, name, value)
                        return

                except AttributeError as e:
                    raise AttributeError(f"Cannot set property '{name}' on the policy instance "
                                         f"(fuzzy number mtype '{self.mtype}'): {e}")
                except Exception as e:
                    raise RuntimeError(f"An unexpected error occurred while setting the property '{name}' "
                                       f"on the strategy instance (fuzzy number type '{self.mtype}'): {e}")

            template_attributes = object.__getattribute__(self, '_bound_template_attributes')
            if name in template_attributes:
                try:
                    template_instance = self.get_template_instance()
                    template_class = template_instance.__class__
                    attr_descriptor = getattr(template_class, name, None)

                    if isinstance(attr_descriptor, property):
                        if attr_descriptor.fset:
                            setattr(template_instance, name, value)
                            return
                        else:
                            raise AttributeError(f"The attribute '{name}' is read-only for "
                                                 f"the fuzzy number type '{self.mtype}'.")
                    else:
                        setattr(template_instance, name, value)
                        return
                except AttributeError as e:
                    raise AttributeError(f"Cannot set attribute '{name}' on template instance "
                                         f"(fuzzy number type '{self.mtype}'): {e}")
                except Exception as e:
                    raise RuntimeError(f"An unexpected error occurred while setting the property '{name}' "
                                       f"on the template instance (fuzzy number type '{self.mtype}'): {e}")

        except AttributeError:
            pass

        object.__setattr__(self, name, value)

    def __del__(self) -> None:
        """
        Destructor, cleans up resources

        The `__del__` method is called when an object is about to be garbage collected.
        It provides an opportunity to perform necessary cleanup tasks, such as releasing resources,
        closing file handles, or clearing internal containers here to ensure that circular
        references do not cause memory leaks (although Python's garbage collector can usually
        handle circular references, explicit cleanup helps release memory in a timely manner
        and avoid potential issues).

        Notes:
            The timing of the `__del__` call is uncertain, and it should not be relied upon to perform
            critical resource release. A better approach is to use a context manager (`with` statement)
            or the explicit `close()` method. The clearing here is mainly to ensure that internal
            dictionaries and collections are emptied, releasing the references they hold.
        """
        try:
            # Define a list of internal container properties that need to be cleaned.
            cleanup_attrs = ['_bound_strategy_methods',
                             '_bound_template_methods']

            for attr in cleanup_attrs:
                try:
                    container = object.__getattribute__(self, attr)
                    if hasattr(container, 'clear'):
                        container.clear()
                except AttributeError:
                    pass
        except AttributeError:
            pass

    def _get_available_members_info(self) -> Dict[str, List[str]]:
        """
        Retrieve information for all available members

        This method aims to collect the names of all public methods and properties exposed
        by the current `Fuzznum` instance through its associated `FuzznumStrategy` and
        `FuzznumTemplate` instances. Its main purpose is to provide a detailed, user-friendly
        error message listing all available properties and methods when a user attempts to
        access a non-existent attribute within the `__getattr__` method, thereby helping
        the user quickly identify the issue or understand the available interface of the object.

        Returns:
            Dict[str, List[str]]: A dictionary containing two keys:
                 - 'attributes': A list storing all available attribute names.
                 - 'methods': A list storing all available method names.
                 If information cannot be retrieved due to initialization issues, return an empty list.
        """
        try:
            strategy_methods = object.__getattribute__(self, '_bound_strategy_methods')
            strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')
            template_methods = object.__getattribute__(self, '_bound_template_methods')
            template_attrs = object.__getattribute__(self, '_bound_template_attributes')

            return {
                'attributes': list(strategy_attrs) + list(template_attrs),
                'methods': list(strategy_methods.keys()) + list(template_methods.keys())
            }
        except AttributeError:
            # If an `AttributeError` occurs while attempting to retrieve any internal binding
            #   information, it usually means the object has not been fully initialized,
            #   or there was a problem during initialization, causing these internal attributes
            #   to not exist. In such cases, return empty lists for attributes and methods
            #   to avoid triggering a new exception while generating the error message.
            return {'attributes': [], 'methods': []}

    def create(self, **kwargs) -> 'Fuzznum':
        """
        Convenient creation method

        The `create` method is a class method (`@classmethod`) that acts as a factory function
        for `Fuzznum` instances. It simplifies the creation process of `Fuzznum` objects,
        allowing users to directly specify the fuzzy number type (`mtype`) upon instantiation,
        and optionally set properties of its associated strategy instance in bulk using keyword
        arguments (`**kwargs`). This approach improves code readability and convenience by
        hiding the underlying complex initialization details.

        Args:
            **kwargs: Optional keyword arguments used to batch set its properties after instance creation.
                      These attributes are typically properties of the underlying policy instance
                      (such as `md`, `nmd`, `q`, etc.).

        Returns:
            Fuzznum: A newly created and initialized Fuzznum instance.

        Examples:
            >>> fuzz = Fuzznum('qrofn', qrung=2).create(md=0.7, nmd=0.2)
            >>> print(fuzz.mtype)
            example_fuzznum
            >>> print(fuzz.md)
            0.7
            >>> print(fuzz.nmd)
            0.2
        """
        instance = Fuzznum(self.mtype, self.q)

        # 批量设置属性
        # If `kwargs` is not empty, iterate through each key-value pair and attempt to set
        #   it on the newly created instance. This leverages Fuzznum's `__setattr__`
        #   mechanism, which is responsible for delegating the attribute to the underlying
        #   policy instance.
        if kwargs:
            for key, value in kwargs.items():
                try:
                    setattr(instance, key, value)
                except AttributeError:
                    raise f"The parameter '{key}' is invalid for the fuzzy number mtype '{self.mtype}'"

        return instance

    def copy(self) -> 'Fuzznum':
        """
        Create a copy of the current instance

        The `copy` method is used to create an independent copy of the current `Fuzznum` instance.
        It performs a "deep copy," ensuring the new copy has all the same property values as the
        original instance, and these property values are independent, so modifying the copy will
        not affect the original instance. This is very useful when creating new variants based on
        existing fuzzy numbers, or when performing operations without modifying the original object.

        Returns:
            Fuzznum: A standalone copy of the current instance.

        Raises:
            RuntimeError: If you attempt to copy an object that has not been fully initialized.

        Examples:
            >>> fuzz1 = Fuzznum('qrofn', qrung=3).create(md=0.7, nmd=0.2)
            >>> fuzz2 = fuzz1.copy()
            >>> fuzz2.md = 0.5 # Modifying a copy does not affect the original instance.
            >>> print(fuzz1.md)
            0.7
            >>> print(fuzz2.md)
            0.5
        """
        if not self._is_initialized():
            raise RuntimeError("Cannot copy uninitialized object")

        current_params = {}
        try:
            strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')

            for attr_name in strategy_attrs:
                try:
                    current_params[attr_name] = getattr(self, attr_name)

                except AttributeError:
                    pass
        except AttributeError:
            pass

        # Use the `create` class method to create a new Fuzznum instance. Pass the original
        #   instance's `mtype` to the `create` method to ensure the new copy is a fuzzy
        #   number of the same types. Pass the collected `current_params` as keyword
        #   arguments to set the new copy's properties during creation. This
        #   approach ensures that the initialization process of the new copy is consistent
        #   with the normal creation process, and that property values are correctly copied.
        return self.create(**current_params)

    # ======================== Information and Debugging ========================
    # The Fuzznum class's information and debugging module provides a series of
    #   methods for obtaining an instance's internal state, performance metrics,
    #   and health status during runtime. These features are crucial for developers to
    #   understand object behavior, diagnose issues, monitor performance, and ensure
    #   data consistency.

    def get_strategy_attributes_dict(self) -> Dict[str, Any]:
        """
        Retrieve all public properties and their values of the
        strategy instance associated with the current Fuzznum instance.

        This method accesses the internal strategy instance (`_strategy_instance`)
        to collect all public properties declared in `_declared_attributes`,
        and returns a dictionary containing these attribute names and their current values.
        This is crucial for serialization, cache key generation, and the operation
        executor (`OperationExecutor`) that requires the complete strategy state.

        Returns:
            Dict[str, Any]: A dictionary where the keys are the names of
                strategy attributes (strings) and the values are the current
                values of those attributes.

        Raises:
            RuntimeError: If the policy instance has not been initialized or is inaccessible.
            AttributeError: If an error occurs while retrieving the attributes of a statement.

        Examples:
            >>> fuzznum = Fuzznum(mtype='qrofn', qrung=3).create(md=0.8, nmd=0.5)
            >>> attrs_dict = fuzznum.get_strategy_attributes_dict()
            >>> print(attrs_dict)
            {'q': 3, 'md': 0.8, 'nmd': 0.5}
        """
        if not self._is_initialized():
            raise RuntimeError("Cannot get strategy attributes from an uninitialized Fuzznum object.")

        strategy_instance = self.get_strategy_instance()

        try:
            declared_attrs = object.__getattribute__(self, '_bound_strategy_attributes')
        except AttributeError:
            raise RuntimeError("Fuzznum's internal strategy attribute bindings are not properly initialized.")

        return {
            attr: getattr(strategy_instance, attr)
            for attr in declared_attrs
            if hasattr(strategy_instance, attr)
        }

    def get_info(self) -> Dict[str, Any]:
        """
        Retrieve object basic information

        This method provides an overview and returns the basic status and
        configuration information of the current `Fuzznum` instance.
        It helps developers quickly understand the instance's type, lifecycle,
        binding status, and cache state.

        Returns:
            Dict[str, Any]: A dictionary containing basic information about the object.
                            Returns simplified information if the object has not been
                            fully initialized.
        """
        if not self._is_initialized():
            return {
                'mtype': getattr(self, 'mtype', 'unknown'),
                'status': 'not_initialized',
            }

        try:
            strategy_methods = object.__getattribute__(self, '_bound_strategy_methods')
            strategy_attributes = object.__getattribute__(self, '_bound_strategy_attributes')
            template_attributes = object.__getattribute__(self, '_bound_template_attributes')
            template_methods = object.__getattribute__(self, '_bound_template_methods')

            return {
                'mtype': self.mtype,
                'status': 'initialized',
                'binding_info': {
                    'bound_methods': sorted(list(strategy_methods.keys()) + list(template_methods.keys())),
                    'bound_attributes': sorted(list(strategy_attributes) + list(template_attributes)),
                }
            }
        except AttributeError as e:
            return {
                'mtype': getattr(self, 'mtype', 'unknown'),
                'status': 'partially_initialized',
                'error': str(e)
            }

    def validate_state(self) -> Dict[str, Any]:
        """
        Verify the consistency of the object state

        This method performs a comprehensive health check on the internal state of a `Fuzznum` instance.
        It verifies the existence of key attributes, checks the consistency of initialization status,
        and attempts to invoke the validation methods of the underlying strategies and templates.
        This is useful for ensuring the object is in a valid and operational state, and for providing
        diagnostic information when debugging complex issues.

        Returns:
            Dict[str, Any]: Validation result dictionary, containing `is_valid` (boolean value),
                `issues` (list of issues), and `warnings` (list of warnings)
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
        }

        try:
            if not hasattr(self, 'mtype'):
                validation_result['issues'].append("Missing mtype attribute")
                validation_result['is_valid'] = False

            if not self._is_initialized():
                validation_result['issues'].append("Object not fully initialized")
                validation_result['is_valid'] = False
                return validation_result

            required_attrs = [
                '_strategy_instance', '_template_instance',
                '_bound_strategy_methods', '_bound_strategy_attributes',
                '_bound_template_methods', '_bound_template_attributes'
            ]

            for attr in required_attrs:
                if not hasattr(self, attr):
                    validation_result['issues'].append(f"The initialized object is missing required attributes.: {attr}")
                    validation_result['is_valid'] = False

            try:
                strategy_instance = self.get_strategy_instance()
                if hasattr(strategy_instance, 'validate_all_attributes'):
                    strategy_validation = strategy_instance.validate_all_attributes()
                    if not strategy_validation['is_valid']:
                        validation_result['issues'].extend([f"Strategy Validation: "
                                                            f"{err}" for err in strategy_validation['errors']])
                        validation_result['is_valid'] = False
            except RuntimeError as e:
                validation_result['issues'].append(f"Strategy instance validation failed: {e}")
                validation_result['is_valid'] = False

            try:
                template_instance = self.get_template_instance()
                if hasattr(template_instance, 'is_valid') and not template_instance.is_valid():
                    validation_result['issues'].append("The template instance has expired.")
                    validation_result['is_valid'] = False
            except RuntimeError as e:
                validation_result['issues'].append(f"Template instance validation failed: {e}")
                validation_result['is_valid'] = False

        except Exception as e:
            validation_result['issues'].append(f"An exception occurred during the verification process.: {e}")
            validation_result['is_valid'] = False

        return validation_result

    # ======================== Serialization support ========================
    # Serialization is the process of converting the state of an object into a storable or transmittable
    #   format (such as a byte stream, string, JSON, dictionary, etc.). Deserialization,
    #   on the other hand, is the process of restoring this format back into the original
    #   object. For complex objects like Fuzznum, serialization support has the following
    #   significant importance:
    # 1. Persistence: Allows the state of a Fuzznum instance to be saved to a file or database,
    #   enabling data to be persistently stored and reloaded after the program is closed.
    # 2. Data Exchange: Allows Fuzznum instances to be transmitted and shared between different
    #   processes, machines, or systems. For example, sending the state of a fuzzy number object over a network.
    # 3. Debugging & Logging: Convert object states into a human-readable format, helpful for
    #   inspecting object contents during debugging or logging key object states in logs.
    # 4. Configuration & Initialization: The initial state of fuzzy numbers can be defined
    # through external configuration files (e.g., JSON), and then deserialized into Fuzznum instances.
    # 5. Cloning & Copying: Although the copy() method provides object copying,
    #   serialization/deserialization is also a common mechanism for achieving deep copying,
    #   especially when the object structure is complex.

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the fuzzy number instance sequence into a dictionary

        This method converts the core state of a `Fuzznum` instance into a Python dictionary.
        This dictionary contains the type of the fuzzy number (`mtype`), its creation time,
        and all attribute values of its underlying strategy instance. Optionally, it
        can also include internal attribute caches. This makes it convenient to store,
        transmit, or inspect `Fuzznum` instances.

        Returns:
            Dict[str, Any]: Dictionary containing instance status。

        Raises:
            RuntimeError: If attempting to serialize an object that has not been fully initialized.
        """
        if not self._is_initialized():
            raise RuntimeError("Unable to serialize uninitialized object")

        try:
            result = {
                'mtype': self.mtype,
                'attributes': {},
            }

            # Obtain the set of policy attribute names bound to the Fuzznum instance.
            #   Only policy attributes represent the core data of the fuzzy number
            #   and need to be serialized. Template attributes are typically computational
            #   results or presentation-level attributes and do not need to be directly serialized.
            strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')
            for attr_name in strategy_attrs:
                try:
                    # Attempt to get the current value of each strategy attribute and add
                    #   it to the `attributes` dictionary. Using `getattr(self, attr_name)`
                    #   will trigger `__getattribute__` and `_delegate_attribute_access`,
                    #   ensuring that the actual attribute value is obtained (possibly from
                    #   the cache or the strategy instance).
                    result['attributes'][attr_name] = getattr(self, attr_name)
                except AttributeError:
                    pass

            return result
        except AttributeError as e:
            raise RuntimeError(f"Failed to serialize: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fuzznum':
        """
        Deserialize fuzzy number instance from dictionary

        This method is a class method (`@classmethod`) used to reconstruct a `Fuzznum`
        instance from a dictionary. It is the inverse operation of the `to_dict()` method,
        allowing the object's state to be restored from serialized data.

        Args:
            cls: The class itself (Fuzznum).
            data: A dictionary containing the state of a fuzzy number instance,
                typically generated by the `to_dict()` method. It must contain the 'mtype' key.

        Returns:
            Fuzznum: A new `Fuzznum` instance reconstructed from the dictionary data.

        Raises:
            ValueError: If the input dictionary is missing the 'mtype' key.
        """
        if 'mtype' not in data:
            raise ValueError("The dictionary must contain the 'mtype' key.")

        instance = cls(data['mtype'])

        # Setting Attributes
        # If the dictionary contains the 'attributes' key (which stores policy attributes),
        #   iterate through and set these attributes. Using `setattr(instance, attr_name, value)`
        #   will trigger the `__setattr__` method of the `Fuzznum` instance, ensuring that the
        #   attribute value is properly delegated to the underlying policy instance and that any
        #   associated validation and callbacks are triggered.
        if 'attributes' in data:
            for attr_name, value in data['attributes'].items():
                try:
                    setattr(instance, attr_name, value)
                except AttributeError:
                    pass

        return instance

    def __repr__(self) -> str:
        """
        String representation of the object

        The `__repr__` method provides an "official" and unambiguous string representation, primarily intended
        for developers and debugging. Its goal is to generate a string that, if entered as a valid Python
        expression, can (ideally) recreate an object with the same value. For `Fuzznum` instances, it displays
        the instance's type, initialization status, lifetime, as well as the specific class names of its
        associated strategy and template, and attempts to include values of some key attributes.

        Returns:
            str: The formal string representation of the object.
        """
        if not self._is_initialized():
            return f"Fuzznum(mtype='{getattr(self, 'mtype', 'unknown')}', status='not_initialized')"

        try:
            mtype = self.mtype
            obj_id = id(self)

            # 获取策略和模板类名
            strategy_cls_name = "N/A"
            template_cls_name = "N/A"
            try:
                # 尝试获取策略实例并获取其类名
                strategy_instance = self.get_strategy_instance()
                strategy_cls_name = strategy_instance.__class__.__name__
            except RuntimeError:
                pass

            try:
                template_instance = self.get_template_instance()
                template_cls_name = template_instance.__class__.__name__
            except RuntimeError:
                pass

            key_attrs = []
            bound_strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')

            attrs_to_prioritize = ['q', 'md', 'nmd']

            for attr_name in attrs_to_prioritize:
                if attr_name in bound_strategy_attrs:
                    try:
                        attr_value = getattr(self, attr_name)
                        key_attrs.append(f"{attr_name}={attr_value!r}")
                    except AttributeError:
                        pass

            other_attrs_count = 0
            for attr_name in sorted(bound_strategy_attrs):
                if attr_name not in attrs_to_prioritize:
                    try:
                        attr_value = getattr(self, attr_name)
                        key_attrs.append(f"{attr_name}={attr_value!r}")
                        other_attrs_count += 1
                        if other_attrs_count >= 2:
                            break
                    except AttributeError:
                        pass

            attrs_str = ", ".join(key_attrs)

            return (
                f"Fuzznum("
                f"\n {attrs_str}"
                f"\n mtype='{mtype}'"
                f"\n id={obj_id}"
                f"\n is_valid={self.validate_state()['is_valid']}"
                f"\n status=initialized"
                f"\n strategy={strategy_cls_name}"
                f"\n template={template_cls_name})"
            )
        except Exception as e:
            return f"Fuzznum(mtype='{self.mtype}', status='initialized', error='Failed to generate detailed repr: {e}')"

    def __str__(self) -> str:
        """
        User-friendly string representation

        The `__str__` method provides an "informal" and user-friendly string representation,
        mainly intended for end users. It aims to generate a concise and readable string,
        typically used with the `print()` function or `str()` type conversion. For `Fuzznum` instances,
        it first attempts to invoke the `str()` method of its associated template instance, as the
        template is responsible for defining the external representation of the fuzzy number.
        If the template is unavailable or its `str()` method fails, it falls back to a default
        representation.

        Returns:
            str: A concise, user-friendly string representation of the object.
        """
        if hasattr(self, 'str') and callable(self.str):
            try:
                return self.get_template_instance().str()
            except ValueError:
                pass

        return f"Fuzznum[{getattr(self, 'mtype', 'unknown')}]"
