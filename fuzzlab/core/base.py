#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 23:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module defines the base classes for fuzzy number strategies and templates
within the FuzzLab framework.

It provides an extensible architecture for defining various types of fuzzy numbers
(e.g., intuitionistic fuzzy numbers, Pythagorean fuzzy numbers) and their
associated operations and representations.

Classes:
    FuzznumStrategy: An abstract base class defining the core behavior and
                     attributes for different fuzzy number types. It includes
                     mechanisms for attribute validation, change callbacks,
                     operation execution, and caching.
    FuzznumTemplate: An abstract base class for defining how fuzzy numbers are
                     represented (e.g., string representation, detailed reports)
                     and providing cached computational properties.

Example Usage (for subclasses):
    # See ExampleStrategy and ExampleTemplate for concrete implementations.
"""
import collections
import json
import weakref
from abc import ABC
from typing import Optional, Set, Dict, Callable, Any, Union, List

from fuzzlab.config import get_config
from fuzzlab.core.cache import LruCache
from fuzzlab.core.ops import get_operation_registry
from fuzzlab.core.triangular import OperationTNorm


class FuzznumStrategy(ABC):
    """Abstract base class for defining fuzzy number strategies.

    This class provides a foundational structure for various types of fuzzy numbers,
    managing their core attributes, validation rules, change notifications,
    and operation execution. It implements a robust attribute management system
    with strict mode enforcement, attribute-level validation, and change callbacks.

    Attributes:
        mtype (str): The membership type identifier for the fuzzy number.
                     Defaults to `get_config().DEFAULT_MTYPE`. This attribute
                     is class-level and defines the type of fuzzy number
                     (e.g., "intuitionistic", "pythagorean").
        q (Optional[int]): The q-rung value for the fuzzy number, if applicable.
                           For non-q-rung fuzzy numbers, it can be set to 1 or None.
                           This attribute is validated to be an integer between 1 and 100.
        _declared_attributes (Set[str]): A class-level set that stores the names
                                         of all attributes explicitly declared
                                         in the class and its subclasses. Used
                                         for strict attribute mode and serialization.
        _op_cache (LruCache): An LRU cache instance used to store results of
                              expensive fuzzy number operations, improving performance.
        _attribute_validators (Dict[str, Callable[[Any], bool]]): A dictionary
                                                                  mapping attribute names
                                                                  to validation functions.
                                                                  Each function takes
                                                                  an attribute value
                                                                  and returns True if valid.
        _change_callbacks (Dict[str, Callable[[str, Any, Any], None]]): A dictionary
                                                                        mapping attribute
                                                                        names to callback
                                                                        functions. Each
                                                                        function is called
                                                                        after an attribute
                                                                        changes, receiving
                                                                        the attribute name,
                                                                        old value, and new value.
    """

    mtype: str = get_config().DEFAULT_MTYPE
    q: Optional[int] = None

    _declared_attributes: Set[str] = set()
    _op_cache: LruCache = LruCache()
    _attribute_validators: Dict[str, Callable[[Any], bool]] = {}
    _change_callbacks: Dict[str, Callable[[str, Any, Any], None]] = {}

    def __init__(self, qrung: Optional[int] = None):
        """Initializes a FuzznumStrategy instance.

        Args:
            qrung (Optional[int]): The q-rung value for this specific fuzzy number instance.
                                   If None, it will be initialized based on default or
                                   subclass-specific logic.
        """
        # Set 'q' attribute directly using object.__setattr__ to bypass custom __setattr__
        # during initialization, preventing potential recursion or validation issues
        # before the object is fully set up.
        object.__setattr__(self, 'q', qrung)

        # Initialize the operation cache for this instance.
        # Using object.__setattr__ ensures that each instance gets its own cache,
        # rather than sharing a class-level cache.
        object.__setattr__(self, '_op_cache', LruCache(maxsize=get_config().CACHE_SIZE))

        # Ensure each instance has its own independent validators and callbacks dictionaries.
        # This prevents instances from inadvertently sharing or overwriting each other's
        # validation rules or callback functions, which would happen if they were
        # initialized as class attributes without this check.
        if (not hasattr(self, '_attribute_validators')
                or self._attribute_validators is FuzznumStrategy._attribute_validators):
            object.__setattr__(self, '_attribute_validators', {})

        if (not hasattr(self, '_change_callbacks')
                or self._change_callbacks is FuzznumStrategy._change_callbacks):
            object.__setattr__(self, '_change_callbacks', {})

        # Add a validator for the 'q' attribute.
        # The 'q' attribute is fundamental to many fuzzy number types (e.g., q-rung orthopair).
        # This validator ensures that 'q' is an integer and falls within a reasonable range (1 to 100).
        # This validation is applied whenever 'q' is set via __setattr__.
        self.add_attribute_validator(
            'q', lambda x: isinstance(x, int) and 1 <= x <= 100)

        # Perform initial validation of the instance's state.
        # This calls the _validate method, which can be overridden by subclasses
        # to include specific validation logic for their attributes.
        self._validate()

    def __init_subclass__(cls, **kwargs):
        """Automatically collects declared attributes for subclasses.

        This method is called automatically when a class inherits from FuzznumStrategy.
        It populates the `_declared_attributes` set for each subclass, which is then
        used by `__setattr__` for strict attribute mode and by `get_declared_attributes`
        for introspection and serialization.

        Args:
            **kwargs: Arbitrary keyword arguments passed during subclass creation.
        """
        # Call the __init_subclass__ method of the parent class (ABC) to ensure
        # proper initialization across the inheritance hierarchy.
        super().__init_subclass__(**kwargs)

        # Initialize the _declared_attributes set for the current subclass (cls).
        # This set is class-level, meaning all instances of this subclass will share
        # the same list of declared attributes.
        cls._declared_attributes = set()

        # Iterate through all attributes and methods of the current class.
        for attr_name in dir(cls):
            # Exclude private attributes (starting with '_'), as they are typically
            # internal implementation details and should not be directly accessed or modified externally.
            # Ensure the attribute actually exists on the class.
            # Exclude callable objects (methods) as we are only interested in data attributes.
            if (not attr_name.startswith('_') and
                    hasattr(cls, attr_name) and
                    not callable(getattr(cls, attr_name))):
                # Add the qualified attribute name to the _declared_attributes set.
                # This set will be used for strict mode checks in __setattr__ and for
                # attribute serialization (e.g., in to_dict/from_dict methods).
                cls._declared_attributes.add(attr_name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Custom attribute setter method.

        This method intercepts all attribute assignments to instances of FuzznumStrategy
        and its subclasses. It enforces strict attribute mode, performs attribute-level
        validation, and triggers change callbacks.

        Args:
            name (str): The name of the attribute being set.
            value (Any): The value being assigned to the attribute.

        Raises:
            AttributeError: If strict attribute mode is enabled and the attribute
                            is not declared in `_declared_attributes`.
            ValueError: If an attribute's value fails validation or if a change
                        callback rejects the change.
            RuntimeError: If a change callback encounters an unexpected error.
        """
        # Fast path for internal attributes (starting with '_') and the 'mtype' attribute.
        # These attributes bypass the custom logic (strict mode, validation, callbacks)
        # and are set directly using the parent class's __setattr__ (object's __setattr__).
        # This is crucial for internal state management and preventing infinite recursion.
        if name.startswith('_') or name == 'mtype':
            super().__setattr__(name, value)
            return

        # Strict attribute mode check.
        # If `STRICT_ATTRIBUTE_MODE` is enabled in the global configuration and
        # the attribute being set is not found in the `_declared_attributes` set
        # (which contains attributes explicitly defined in the class or its subclasses),
        # an AttributeError is raised. This prevents accidental creation of new attributes
        # and helps maintain a well-defined object schema.
        config = get_config()
        if (hasattr(config, 'STRICT_ATTRIBUTE_MODE') and
                config.STRICT_ATTRIBUTE_MODE and
                name not in self._declared_attributes):
            raise AttributeError(
                f"Attribute '{name}' not declared in {self.__class__.__name__}. "
                f"Declared attributes: {sorted(self._declared_attributes)}"
            )

        # Get the old value of the attribute before it's changed.
        # This old value is passed to change callback functions.
        # `getattr(self, name, None)` safely retrieves the current value, returning None
        # if the attribute doesn't exist yet.
        old_value = getattr(self, name, None)

        # Attribute value validation.
        # Check if a validator function has been registered for the current attribute name.
        # Validators are added via `add_attribute_validator`.
        if name in self._attribute_validators:
            validator = self._attribute_validators[name]
            # If the validator function returns False, it means the new value is invalid,
            # and a ValueError is raised, preventing the invalid value from being set.
            if not validator(value):
                raise ValueError(f"Validation failed for attribute '{name}' with value '{value}'")

        # Actually set the attribute value.
        # After all checks (strict mode, validation) have passed, the attribute's value
        # is finally set using the parent class's __setattr__ method.
        super().__setattr__(name, value)

        # Execute attribute change callbacks.
        # Check if any callback functions are registered for the current attribute name.
        # Callbacks are added via `add_change_callback`.
        if name in self._change_callbacks:
            callback = self._change_callbacks[name]
            try:
                # Call the callback function with the attribute name, old value, and new value.
                callback(name, old_value, value)
            except ValueError as e:
                # If the callback explicitly raises a ValueError, it indicates that the
                # change is rejected by the callback's logic. This error is re-raised.
                raise ValueError(f"Attribute '{name}' change rejected by callback: {e}") from e
            except Exception as e:
                # Catch any other unexpected exceptions raised by the callback.
                # This indicates a problem within the callback itself, and a RuntimeError is raised.
                raise RuntimeError(f"Callback for attribute '{name}' failed, "
                                   f"change has been rolled back.") from e

    def add_attribute_validator(self,
                                attr_name: str,
                                validator: Callable[[Any], bool]) -> None:
        """Adds an attribute validator function.

        This method allows registering a custom validation function for a specific
        attribute. The validator will be called whenever the attribute is set,
        ensuring data integrity.

        Args:
            attr_name (str): The name of the attribute to validate.
            validator (Callable[[Any], bool]): A callable (function or lambda)
                                               that takes the attribute's value
                                               as input and returns True if the
                                               value is valid, False otherwise.
        """
        self._attribute_validators[attr_name] = validator

    def add_change_callback(self,
                            attr_name: str,
                            callback: Callable[[str, Any, Any], None]) -> None:
        """Adds an attribute change callback function.

        This method allows registering a function to be called whenever a specific
        attribute's value changes. Callbacks can be used to trigger side effects,
        update dependent properties, or perform additional validation after a change.

        Args:
            attr_name (str): The name of the attribute to monitor for changes.
            callback (Callable[[str, Any, Any], None]): A callable (function or lambda)
                                                        that takes three arguments:
                                                        the attribute name, its old value,
                                                        and its new value.
        """
        self._change_callbacks[attr_name] = callback

    def _validate(self) -> None:
        """Performs internal validation of the fuzzy number strategy's state.

        This is a protected method intended to be overridden by subclasses to
        implement specific validation logic relevant to their fuzzy number type.
        It's typically used for complex constraints involving multiple attributes
        or for validation that cannot be performed at the individual attribute
        setting level.

        The default implementation performs a basic check on the `mtype` attribute.
        Subclasses should call `super()._validate()` to ensure base class validation
        is also performed.

        Raises:
            ValueError: If any validation constraint is violated.
        """
        # Check if 'mtype' exists and is a non-empty string.
        # This is a fundamental property for all fuzzy number strategies.
        if hasattr(self, 'mtype') and (not isinstance(self.mtype, str) or not self.mtype.strip()):
            raise ValueError(f"mtype must be a non-empty string, got '{self.mtype}'")

    def _generate_cache_key(self,
                            op_name: str,
                            operands: Union['FuzznumStrategy', int, float],
                            tnorm: OperationTNorm) -> Optional[str]:
        """Generates a unique cache key for an operation.

        The key is based on the operation name, the state of the operands (including
        the current FuzznumStrategy instance and the other operand), and the
        parameters of the t-norm used. This ensures that identical operations
        with identical inputs and t-norm settings can retrieve cached results.

        Args:
            op_name (str): The name of the operation (e.g., 'add', 'mul').
            operands (Union[FuzznumStrategy, int, float]): The second operand of the operation.
            tnorm (OperationTNorm): The t-norm object used for the operation.

        Returns:
            Optional[str]: A unique MD5 hash string representing the cache key,
                           or None if key generation fails (e.g., due to un-serializable objects).
        """
        def get_state_dict(obj: 'FuzznumStrategy'):
            """Helper function to get a serializable state dictionary of a FuzznumStrategy."""
            # Get all explicitly declared attributes of the FuzznumStrategy instance.
            attrs = obj.get_declared_attributes()
            # Create a dictionary mapping attribute names to their current values.
            state = {attr: getattr(obj, attr, None) for attr in attrs}
            # Include the 'q' attribute, which is crucial for many fuzzy number types.
            state['q'] = obj.q
            return state

        try:
            import hashlib
            # Serialize the state of the first operand (self) into a JSON string.
            # `sort_keys=True` ensures consistent key order for reproducible hashes.
            operand_1 = json.dumps(get_state_dict(self), sort_keys=True)

            # Serialize the second operand based on its type.
            if isinstance(operands, FuzznumStrategy):
                operand_2 = json.dumps(get_state_dict(operands), sort_keys=True)
            else:
                # For scalar operands (int, float), convert directly to string.
                operand_2 = str(operands)

            # Get serializable information about the t-norm object.
            t_norm_info = json.dumps(tnorm.get_info(), sort_keys=True)

            # Combine all parts into a list.
            key_parts = [op_name, operand_1, operand_2, t_norm_info]

            # Join the parts and encode to bytes for hashing.
            key_str = '_'.join(key_parts)
            # Compute the MD5 hash and return its hexadecimal representation.
            return hashlib.md5(key_str.encode()).hexdigest()
        except (TypeError, AttributeError):
            # Return None if any part of the key generation fails (e.g., non-serializable data).
            return None

    def execute_operation(self,
                          op_name: str,
                          operand: Optional[Union['FuzznumStrategy', int, float]]) -> Dict[str, Any]:
        """Executes a specified operation with another operand.

        This method acts as a central dispatcher for performing various fuzzy
        number operations (e.g., addition, multiplication, comparison, complement).
        It retrieves the appropriate operation handler from the global registry
        based on the operation name and the fuzzy number's `mtype`. It also
        manages caching of operation results.

        Args:
            op_name (str): The name of the operation to execute (e.g., 'add', 'mul', 'complement').
            operand (Optional[Union[FuzznumStrategy, int, float]]): The second operand for the operation.
                                                                     Can be another FuzznumStrategy instance,
                                                                     an integer, or a float, depending on the operation.

        Returns:
            Dict[str, Any]: A dictionary containing the result of the operation.
                            The structure of the dictionary depends on the specific operation.

        Raises:
            NotImplementedError: If the requested operation is not supported for the current `mtype`.
            TypeError: If the operand type is incompatible with the operation.
            ValueError: If an unknown operation type is requested.
        """
        # Get the global operation registry.
        registry = get_operation_registry()
        # Retrieve the specific operation handler for the given operation name and fuzzy number type.
        operation = registry.get_operation(op_name, self.mtype)
        if operation is None:
            # If no operation handler is found, raise an error indicating lack of support.
            raise NotImplementedError(
                f"Operation '{op_name}' is not supported for mtype '{self.mtype}'."
                f" Available operations for '{self.mtype}': {self.get_available_operations()}"
            )

        # Get the default t-norm configuration from the registry.
        norm_type, norm_params = registry.get_default_t_norm_config()
        # Create an OperationTNorm instance with the retrieved configuration and the current q-rung.
        tnorm = OperationTNorm(norm_type=norm_type, q=self.q, norm_params=norm_params)

        # Generate a unique cache key for the current operation.
        cache_key = self._generate_cache_key(op_name, operand, tnorm)

        # If a valid cache key is generated, attempt to retrieve the result from the cache.
        if cache_key:
            cached_result = self._op_cache.get(cache_key)
            if cached_result is not None:
                return cached_result  # Return cached result if found.

        # Dispatch to the appropriate operation execution method based on the operation name.
        if op_name in ['add', 'sub', 'mul', 'div',
                       'intersection', 'union', 'implication',
                       'equivalence', 'difference', 'symdiff']:
            # These are binary operations requiring another FuzznumStrategy as an operand.
            if not isinstance(operand, FuzznumStrategy):
                raise TypeError(f"Operands must be a fuzznum strategy, got '{type(operand)}'")
            result = operation.execute_binary_op(self, operand, tnorm)

        elif op_name in ['pow', 'tim', 'exp', 'log']:
            # These are operations with a scalar operand (int or float).
            if not isinstance(operand, (int, float)):
                raise TypeError(f"Operands must be a number of 'int' or 'float', got '{type(operand)}'")
            result = operation.execute_unary_op_operand(self, operand, tnorm)

        elif op_name in ['complement']:
            # This is a pure unary operation that takes no additional operand.
            if operand is not None:
                raise TypeError(f"Pure unary operation '{op_name}' takes no additional operands.")
            result = operation.execute_unary_op_pure(self, tnorm)

        elif op_name in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
            # These are comparison operations requiring another FuzznumStrategy as an operand.
            if not isinstance(operand, FuzznumStrategy):
                raise TypeError(f"Comparison operation '{op_name}' requires one FuzznumStrategy operand, "
                                f"got '{type(operand)}'")
            result = operation.execute_comparison_op(self, operand, tnorm)
        else:
            # If the operation name is not recognized, raise a ValueError.
            raise ValueError(f"Unknown operation type: {op_name}")

        # If a valid cache key was generated and the operation was successful, cache the result.
        if cache_key:
            self._op_cache.put(cache_key, result)

        return result

    def get_declared_attributes(self) -> Set[str]:
        """Retrieves a copy of the set of declared attribute names for this strategy.

        This method provides introspection capabilities, allowing external code
        to discover which attributes are explicitly defined as data members
        of a `FuzznumStrategy` instance or its subclasses.

        Returns:
            Set[str]: A copy of the set containing the names of all declared attributes.
        """
        return self._declared_attributes.copy()

    def get_available_operations(self) -> List[str]:
        """Gets the names of all operations supported by the current fuzzy number type.

        This method queries the global operation registry to determine which
        operations are registered and available for the `mtype` of this
        `FuzznumStrategy` instance.

        Returns:
            List[str]: A list of strings, where each string is the name of an
                       operation supported by this fuzzy number type.
        """
        operation = get_operation_registry()
        return operation.get_available_ops(self.mtype)

    def validate_all_attributes(self) -> Dict[str, Any]:
        """Validates all attributes of the FuzznumStrategy instance.

        This method provides a unified interface to trigger a comprehensive
        health check of the instance's state. It performs both the general
        `_validate()` checks and individual attribute validations.

        Returns:
            Dict[str, Any]: A dictionary containing the validation results.
                            It includes:
                            - 'is_valid' (bool): True if all validations pass, False otherwise.
                            - 'errors' (List[str]): A list of error messages for failed validations.
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
        }
        # Initialize a dictionary to store the validation results.
        # 'is_valid' flag indicates overall validation success, 'errors' list collects all error messages.

        try:
            # First, call the `_validate()` method.
            # `_validate()` is a protected method, typically overridden by subclasses,
            # to perform core or complex validation logic specific to that strategy type.
            # For example, checking relationships between multiple attributes (e.g., `md + nmd <= 1`).
            # If `_validate()` finds a serious issue, it will directly raise an exception (e.g., `ValueError`),
            # which will be caught by the outer `try-except` block and recorded as an error.
            self._validate()

            # Next, iterate through all attributes explicitly declared in the class.
            # These attributes are collected and stored in the `_declared_attributes` set
            # during `__init_subclass__`.
            # This ensures that only data attributes belonging to this strategy model are validated,
            # not internal private attributes or methods.
            for attr_name in self._declared_attributes:
                if hasattr(self, attr_name):
                    value = getattr(self, attr_name)

                    # Check if a specific validator has been registered for this attribute.
                    # These validators are added via the `add_attribute_validator()` method,
                    # and they are typically used to validate the legality of individual attributes
                    # (e.g., whether a value is within a certain range, or if it's of a specific type).
                    if attr_name in self._attribute_validators:
                        validator = self._attribute_validators[attr_name]

                        # Call the validator function with the attribute value.
                        # If the validator function returns `False`, it means the attribute's value
                        # does not meet the requirements.
                        if not validator(value):
                            validation_result['errors'].append(
                                f"Attribute '{attr_name}' validation failed with value '{value}'"
                            )
                            # Mark the overall validation result as False.
                            validation_result['is_valid'] = False

        # Catch any exception that might be raised during the `_validate()` method
        # or during individual attribute validation.
        # This ensures that the `validate_all_attributes` method itself does not crash
        # due to internal validation failures, but instead gracefully returns error information.
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {e}")
            # Add the captured exception message to the `errors` list.

            validation_result['is_valid'] = False
            # Mark the overall validation status as `False`.

        return validation_result


class FuzznumTemplate(ABC):
    """Abstract base class for fuzzy number templates.

    This class defines the interface for how fuzzy numbers are represented
    and provides auxiliary functionalities, such as cached computational properties.
    Subclasses are responsible for implementing specific representations
    (e.g., string, report) and any derived properties.

    Attributes:
        mtype (str): The membership type identifier for the fuzzy number
                     this template is associated with. Defaults to
                     `get_config().DEFAULT_MTYPE`.
        _instance_ref (weakref.ref): A weak reference to the associated
                                     `Fuzznum` instance. This prevents
                                     circular references and allows the
                                     `Fuzznum` instance to be garbage
                                     collected when no other strong
                                     references exist.
        _instance_id (int): The unique ID (memory address) of the associated
                            `Fuzznum` instance at the time of template creation.
                            Used for debugging and identifying the original instance.
        _is_valid (bool): A flag indicating whether the template is still valid.
                          It becomes False if the associated `Fuzznum` instance
                          is garbage collected.
        _max_cache_size (int): The maximum number of items to store in the
                               internal template cache. Configurable via
                               `TEMPLATE_CACHE_SIZE` in the global config.
        _template_cache (collections.OrderedDict): An ordered dictionary used
                                                   as an LRU cache for storing
                                                   computed template values
                                                   (e.g., report strings, scores).
        _cache_enabled (bool): A flag indicating whether caching is currently enabled.
    """

    mtype: str = get_config().DEFAULT_MTYPE

    def __init__(self, instance: Any):
        """Initializes a FuzznumTemplate instance.

        Args:
            instance (Any): The `Fuzznum` instance that this template will represent.
                            This should be a concrete instance of a fuzzy number.

        Raises:
            ValueError: If the provided `instance` is None.
        """
        if instance is None:
            raise ValueError("Template instance cannot be None.")

        # Create a weak reference to the provided `instance` (typically a Fuzznum instance).
        # This prevents the template from keeping the Fuzznum instance alive if it's
        # no longer referenced elsewhere, avoiding memory leaks due to circular references.
        # The `_on_instance_cleanup` method will be called when the instance is garbage collected.
        self._instance_ref = weakref.ref(instance, self._on_instance_cleanup)

        # Store the unique identifier (memory address) of the associated instance.
        # This can be useful for debugging and tracking.
        self._instance_id = id(instance)

        # Initialize a boolean flag indicating whether the current template instance is still valid.
        # This flag will be set to False if the associated Fuzznum instance is garbage collected.
        self._is_valid = True

        # Get configuration settings for caching.
        config = get_config()
        # Set the maximum cache size, defaulting to 256 if not specified in config.
        self._max_cache_size = getattr(config, 'TEMPLATE_CACHE_SIZE', 256)
        # Initialize an OrderedDict to serve as an LRU cache for template computation results.
        # OrderedDict maintains insertion order, allowing easy removal of the least recently used item.
        self._template_cache: collections.OrderedDict = collections.OrderedDict()

        # Set the initial cache enabled status based on the global configuration.
        self._cache_enabled = config.ENABLE_CACHE

    def _on_instance_cleanup(self, ref: weakref.ref) -> None:
        """Callback method executed when the associated Fuzznum instance is garbage collected.

        This method is registered with the weak reference and is automatically called
        when the object it points to is no longer strongly referenced. It marks the
        template as invalid and clears its cache.

        Args:
            ref (weakref.ref): The weak reference object that triggered the callback.
        """
        self._is_valid = False  # Mark the template as invalid.
        self._template_cache.clear()  # Clear any cached results, as they are now stale.

    @property
    def instance(self) -> Any:
        """Provides access to the associated Fuzznum instance.

        This property ensures that the associated Fuzznum instance is still valid
        and accessible before returning it. It raises a RuntimeError if the instance
        has been garbage collected.

        Returns:
            Any: The associated Fuzznum instance.

        Raises:
            RuntimeError: If the associated Fuzznum instance has been garbage collected
                          and the template is no longer valid.
        """
        # First, check the `_is_valid` flag. If it's already False, it means the
        # associated instance has already been collected, so raise an error immediately.
        if not self._is_valid:
            raise RuntimeError(
                f"Template for mtype '{self.mtype}' is no longer valid. "
                f"Associated Fuzznum instance (id: {self._instance_id}) has been garbage collected."
            )

        # Attempt to retrieve the strong reference from the weak reference.
        instance = self._instance_ref()

        # If the weak reference returns None, it means the Fuzznum instance was
        # garbage collected since the last check.
        if instance is None:
            self._is_valid = False  # Mark the template as invalid.
            raise RuntimeError(
                f"Template for mtype '{self.mtype}' has lost its Fuzznum instance "
                f"(id: {self._instance_id}). Instance has been garbage collected."
            )

        return instance

    def is_template_valid(self) -> bool:
        """Checks if the template is still valid and its associated Fuzznum instance exists.

        Returns:
            bool: True if the template is valid and its associated instance is still alive,
                  False otherwise.
        """
        # If the internal `_is_valid` flag is already False, the template is definitely invalid.
        if not self._is_valid:
            return False

        # Attempt to get the strong reference from the weak reference.
        instance = self._instance_ref()

        # If the instance is None, it means the associated object has been garbage collected.
        if instance is None:
            self._is_valid = False  # Update the internal flag.
            return False

        return True

    def get_cached_value(self, key: str, compute_func: Optional[Callable[[], Any]] = None) -> Any:
        """Retrieves a cached value, computing and caching it if it doesn't exist.

        This method is designed to cache results of expensive computations related
        to the fuzzy number's representation or properties (e.g., score functions,
        report strings). If caching is disabled, it directly computes the value.

        Args:
            key (str): The unique key for the cached value.
            compute_func (Optional[Callable[[], Any]]): A callable (function or lambda)
                                                        that computes the value if it's
                                                        not found in the cache. This
                                                        function should take no arguments.

        Returns:
            Any: The cached value or the newly computed value. Returns None if the
                 key is not found and no `compute_func` is provided.
        """
        # If caching is disabled, directly execute the compute function (if provided)
        # and do not perform any caching operations.
        if not self._cache_enabled:
            return compute_func() if compute_func else None

        # Check if the value for the given `key` already exists in the cache.
        if key in self._template_cache:
            # If found, move the item to the end of the OrderedDict to mark it as recently used (LRU logic).
            self._template_cache.move_to_end(key)
            return self._template_cache[key]

        # If the value is not in the cache and a `compute_func` is provided:
        if compute_func:
            # Call the `compute_func` to calculate the value.
            value = compute_func()
            # Store the computed value in the cache.
            self._template_cache[key] = value
            # Check if the cache size exceeds the maximum allowed size.
            if len(self._template_cache) > self._max_cache_size:
                # If it does, remove the least recently used item (the first item in OrderedDict).
                self._template_cache.popitem(last=False)
            return value

        # If the key is not found and no compute function is provided, return None.
        return None

    def clear_cache(self) -> None:
        """Clears all entries from the template's internal cache."""
        self._template_cache.clear()

    def enable_cache(self) -> None:
        """Enables caching for this template."""
        self._cache_enabled = True

    def disable_cache(self) -> None:
        """Disables caching for this template and clears any existing cached values."""
        self._cache_enabled = False
        self.clear_cache()

    def report(self) -> str:
        """Generates a detailed report string for the fuzzy number.

        This method must be implemented by subclasses to provide a specific
        formatted string representation of the fuzzy number's properties.

        Raises:
            NotImplementedError: This method is abstract and must be implemented by subclasses.
        """
        raise NotImplementedError("report method must be implemented by subclasses")

    def str(self) -> str:
        """Generates a concise string representation of the fuzzy number.

        This method must be implemented by subclasses to provide a specific
        short string representation, typically used for `print()` or `str()` calls.

        Raises:
            NotImplementedError: This method is abstract and must be implemented by subclasses.
        """
        raise NotImplementedError("str method must be implemented by subclasses")

    def get_template_info(self) -> Dict[str, Any]:
        """Retrieves information about the template's current state and cache.

        Returns:
            Dict[str, Any]: A dictionary containing various pieces of information, including:
                            - 'mtype' (str): The fuzzy number type associated with this template.
                            - 'is_valid' (bool): Whether the template is still valid (associated instance exists).
                            - 'instance_id' (int): The unique ID of the associated Fuzznum instance.
                            - 'cache_enabled' (bool): Whether caching is enabled.
                            - 'cache_size' (int): The number of items currently in the cache.
                            - 'cache_keys' (List[str] or str): A list of cache keys, or a summary string
                                                               if there are too many keys.
        """
        return {
            'mtype': self.mtype,  # The fuzzy number type associated with this template.
            'is_valid': self._is_valid,  # Whether the template is still valid (associated instance exists).
            'instance_id': self._instance_id,  # The unique ID of the associated Fuzznum instance.
            'cache_enabled': self._cache_enabled,  # Whether caching is enabled.
            'cache_size': len(self._template_cache),  # The number of items currently in the cache.
            'cache_keys': list(self._template_cache.keys())
            # Display cache keys as a list if there are fewer than 10, otherwise show a summary string.
            if len(self._template_cache) < 10 else f"{len(self._template_cache)} items"
        }


# ======================== Improved Example Implementations ========================
# These example implementations are for demonstration purposes and are not
# directly involved in the core initialization or operation processes.

class ExampleStrategy(FuzznumStrategy):
    """An example fuzzy number strategy class demonstrating attribute definition, validation, and callbacks.

    This class inherits from `FuzznumStrategy` and specifically implements the core
    attributes `md` (membership degree) and `nmd` (non-membership degree) for a
    hypothetical fuzzy number type. It showcases how to leverage the base class's
    mechanisms to:
        - Set value range validators for `md` and `nmd`.
        - Trigger callbacks when `md` or `nmd` change, checking basic fuzzy number
          constraints (e.g., `md + nmd <= 1`).
        - Extend the base validation logic in the `_validate` method to enforce
          the sum of `md` and `nmd`.

    Attributes:
        mtype (str): The fuzzy number type identifier for this strategy, fixed as "example_fuzznum".
        md (Optional[float]): The membership degree, an optional float representing
                              the extent to which an element belongs to the fuzzy set.
                              Its value should be in the range [0, 1].
        nmd (Optional[float]): The non-membership degree, an optional float representing
                               the extent to which an element does not belong to the fuzzy set.
                               Its value should be in the range [0, 1].
    """
    mtype = "example_fuzznum"
    md: Optional[float] = None
    nmd: Optional[float] = None

    def __init__(self):
        """Initializes an ExampleStrategy instance.

        In this constructor, the parent class's initialization method is called,
        and validators and change callbacks are registered for the `md` and `nmd` attributes.
        """
        super().__init__()
        # Call the parent class (FuzznumStrategy)'s initialization method.
        # This will initialize instance-specific attributes and add the validator for the 'q' attribute.

        # Add attribute validators for 'md' and 'nmd'.
        # The validator ensures that 'md' is either None or a float/int within the [0, 1] range.
        self.add_attribute_validator('md', lambda x: x is None or (isinstance(x, (int, float)) and 0 <= x <= 1))
        # The validator ensures that 'nmd' is either None or a float/int within the [0, 1] range.
        self.add_attribute_validator('nmd', lambda x: x is None or (isinstance(x, (int, float)) and 0 <= x <= 1))

        # Add change callbacks for 'md' and 'nmd'.
        # When the 'md' attribute's value changes, the `_on_membership_change` method is registered as a callback.
        self.add_change_callback('md', self._on_membership_change)
        # When the 'nmd' attribute's value changes, the `_on_membership_change` method is registered as a callback.
        self.add_change_callback('nmd', self._on_membership_change)

    def _on_membership_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """Callback function triggered when membership or non-membership degrees change.

        This callback is invoked when the `md` or `nmd` attribute is set. It checks
        the fuzzy number constraint `md + nmd <= 1`. If the constraint is violated,
        it raises a `ValueError`, which triggers the rollback logic in
        `FuzznumStrategy.__setattr__`.

        Args:
            attr_name (str): The name of the attribute that changed ('md' or 'nmd').
            old_value (Any): The old value of the attribute.
            new_value (Any): The new value of the attribute.
        """
        # Only proceed with the check if the new value is not None and both 'md' and 'nmd'
        # attributes exist on the instance. This prevents incomplete checks during object initialization.
        if new_value is not None and hasattr(self, 'md') and hasattr(self, 'nmd'):
            # Only perform the fuzzy number constraint check if both 'md' and 'nmd' have been assigned (are not None).
            if self.md is not None and self.nmd is not None:
                # Check the fuzzy number constraint: typically, the sum of membership (md)
                # and non-membership (nmd) degrees should not exceed 1.
                if self.md + self.nmd > 1:
                    # **Critical change: Changed from warnings.warn to raise ValueError.**
                    # Raising a ValueError here will cause `FuzznumStrategy.__setattr__`
                    # to catch it and potentially roll back the attribute change,
                    # ensuring that invalid states are not set.
                    raise ValueError(f"md + nmd = {self.md + self.nmd} > 1, violates fuzzy number constraints")

    def _validate(self) -> None:
        """Extends the strategy's validation method to include a check on the sum of membership and non-membership degrees.

        This method builds upon the parent class's `_validate` method by adding
        a specific check for the combined constraint of `md` and `nmd`. If the
        sum `md + nmd` is greater than 1, a `ValueError` is raised.
        """
        super()._validate()
        # Call the `_validate` method of the parent class (FuzznumStrategy)
        # to perform its default validation logic (e.g., for `mtype`).

        # Check the constraint for membership and non-membership degrees: `md + nmd` must not exceed 1.
        # This combined constraint check is only performed if both 'md' and 'nmd' have been assigned (are not None).
        if (self.md is not None and self.nmd is not None and
                self.md + self.nmd > 1):
            # If the sum of md and nmd is greater than 1, raise a ValueError,
            # indicating that the fuzzy number's state is invalid.
            raise ValueError(f"md + nmd = {self.md + self.nmd} must not exceed 1")


class ExampleTemplate(FuzznumTemplate):
    """An improved example fuzzy number template implementation.

    This class demonstrates how to generate string representations, detailed reports,
    and add custom computational methods for a fuzzy number. It inherits from
    `FuzznumTemplate` and provides specific representation logic for fuzzy numbers
    of type "example_fuzznum". It leverages the base class's caching mechanism to:
        - Generate a concise string representation (`str`).
        - Generate a detailed report string (`report`).
        - Compute a custom score function (`score`).
    All these computed results are cached to improve performance on repeated access.

    Attributes:
        mtype (str): The fuzzy number type identifier for this template, fixed as "example_fuzznum".
    """

    mtype = "example_fuzznum"

    def report(self) -> str:
        """Generates a detailed report string for the fuzzy number.

        This method utilizes the caching mechanism of `FuzznumTemplate` to generate
        a detailed report containing all key parameters of the fuzzy number.
        The report includes membership degree (`md`), non-membership degree (`nmd`),
        and q-rung (`q`).

        Returns:
            str: A detailed fuzzy number report string, e.g., "ExampleFuzz(md=0.8, nmd=0.1, q=2)".
        """
        return self.get_cached_value(
            'report',  # Cache key for the report string.
            # Lambda function to compute the report string if not cached.
            # It safely accesses the associated Fuzznum instance's attributes via `self.instance`.
            lambda: f"ExampleFuzz(md={self.instance.md}, nmd={self.instance.nmd}, q={self.instance.q})"
        )

    def str(self) -> str:
        """Generates a concise string representation of the fuzzy number.

        This method utilizes the caching mechanism of `FuzznumTemplate` to generate
        a brief string representation of the fuzzy number. It is typically used
        for `print()` or `str()` calls.

        Returns:
            str: A concise fuzzy number string representation, e.g., "<0.8,0.1>_q=2".
        """
        return self.get_cached_value(
            'str',  # Cache key for the concise string representation.
            # Lambda function to compute the concise string representation if not cached.
            # It safely accesses the associated Fuzznum instance's attributes via `self.instance`.
            lambda: f"<{self.instance.md},{self.instance.nmd}>_q={self.instance.q}"
        )

    def score(self) -> float:
        """Computes the score function of the fuzzy number.

        This method calculates the score of the fuzzy number, defined as the
        difference between the membership degree and the non-membership degree (`md - nmd`).
        It utilizes the caching mechanism of `FuzznumTemplate` to avoid redundant computations.
        If `md` or `nmd` are None, they are treated as 0 in the calculation.

        Returns:
            float: The computed score of the fuzzy number.
        """
        return self.get_cached_value(
            'score',  # Cache key for the score.
            # Lambda function to compute the score if not cached.
            # `(self.instance.md or 0)` ensures that if `md` is None, it defaults to 0,
            # preventing a TypeError during arithmetic operations.
            lambda: (self.instance.md or 0) - (self.instance.nmd or 0)
        )
    