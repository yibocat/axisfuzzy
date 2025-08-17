#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
axisfuzzy.core.base
===================

Abstract strategy base for single-element fuzzy-number implementations.

This module defines FuzznumStrategy, the core abstract base class that
concrete fuzzy-number implementations (mtypes) should inherit from.
"""

import json
from abc import ABC, abstractmethod
from typing import Optional, Set, Dict, Callable, Any, Union, List

from .cache import LruCache
from .triangular import OperationTNorm
from .operation import get_registry_operation
from ..config import get_config


class FuzznumStrategy(ABC):
    """
    Abstract base class for fuzzy-number strategies.

    Concrete implementations should declare their data attributes (via
    annotations or class-level defaults) and implement `report` / `str`.

    Parameters
    ----------
    q : int, optional
        q-rung parameter. If ``None``, subclasses or configuration defaults
        will be used.

    Attributes
    ----------
    mtype : str
        The registered fuzzy-number type string for this strategy class.
    q : int
        Effective q-rung for this instance.
    _declared_attributes : set
        Names of attributes declared for the strategy.
    _op_cache : LruCache
        Per-instance operation result cache.
    _attribute_validators : dict
        Mapping from attribute name to validator callable.
    _change_callbacks : dict
        Mapping from attribute name to change-callback callable.

    Examples
    --------
    Basic usage showing a minimal concrete strategy, adding a validator and a
    change callback, and changing an attribute value.

    >>> class MyStrategy(FuzznumStrategy):
    ...     # declare a public attribute via annotation
    ...     value: float = 0.5
    ...     def report(self) -> str:
    ...         return f"MyStrategy(value={self.value})"
    ...     def str(self) -> str:
    ...         return f"{self.value}"
    ...
    >>> s = MyStrategy(q=2)
    >>> # register a validator that enforces 0 <= value <= 1
    >>> s.add_attribute_validator('value', lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0)
    >>> # register a change callback to observe updates
    >>> s.add_change_callback('value', lambda name, old, new: print(f"{name} changed {old}->{new}"))
    >>> s.value = 0.8
    value changed 0.5->0.8

    Notes
    -----
    - Subclasses should call `super().__init__()` when overriding `__init__`.
    - The class enforces "strict attribute mode": assigning attributes not
      declared in `_declared_attributes` will raise AttributeError.
    """

    mtype: str = get_config().DEFAULT_MTYPE
    q: Optional[int] = get_config().DEFAULT_Q

    _declared_attributes: Set[str] = set()
    _op_cache: LruCache = LruCache()
    _attribute_validators: Dict[str, Callable[[Any], bool]] = {}
    _change_callbacks: Dict[str, Callable[[str, Any, Any], None]] = {}

    def __init__(self, q: Optional[int] = None):
        """
        Initialize base strategy internals.

        Parameters
        ----------
        q : int, optional
            q-rung value for this instance. If None, default will be used.

        Notes
        -----
        - Uses object.__setattr__ to avoid recursion through custom __setattr__.
        - Registers a basic validator for `q` and performs initial `_validate()`.
        """
        # Set 'q' attribute directly using object.__setattr__ to bypass custom __setattr__
        # during initialization, preventing potential recursion or validation issues
        # before the object is fully set up.
        object.__setattr__(self, 'q', q)

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
        """
        Hook executed when a subclass is defined.

        This collects declared attributes (from annotations and class-level
        assignments) and inherits declared attributes from parent strategies.

        Parameters
        ----------
        **kwargs
            Forwarded to base implementation.
        """
        """Hook for subclasses to declare their own attributes."""
        super().__init_subclass__(**kwargs)

        # 1. Inherit declared attributes from parent strategies
        base_attrs = set()
        for base in cls.__bases__:
            if issubclass(base, FuzznumStrategy):
                base_attrs.update(getattr(base, '_declared_attributes', set()))

        # 2. Collect attributes defined in the current class
        current_class_attrs = set()
        # 2a. Add attributes from type annotations (for instance state)
        current_class_attrs.update(
            attr for attr in cls.__annotations__ if not attr.startswith('_')
        )
        # 2b. Add attributes from class-level assignments (like mtype, q)
        # We inspect __dict__ to only get attributes from the current class
        for attr_name, attr_value in cls.__dict__.items():
            if not attr_name.startswith('_') and not callable(attr_value):
                current_class_attrs.add(attr_name)

        # 3. Combine inherited and current class attributes
        cls._declared_attributes = base_attrs.union(current_class_attrs)

        # 4. Initialize other necessary structures for the new subclass
        cls._attribute_validators = {}
        cls._change_callbacks = {}

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute with validation and optional change callbacks.

        The method enforces declared-attribute constraints, runs registered
        validators and calls change callbacks after successful assignment.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            New value to assign.

        Raises
        ------
        AttributeError
            If `name` is not declared in `_declared_attributes`.
        ValueError
            If a registered validator rejects the value or a callback explicitly rejects the change.
        RuntimeError
            If a change callback raises an unexpected exception.
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
        if name not in self._declared_attributes:
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
        """
        Register a validator for an attribute.

        This method allows registering a custom validation function for a specific
        attribute. The validator will be called whenever the attribute is set,
        ensuring data integrity.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to validate.
        validator : callable
            Callable that accepts the attribute value and returns True if valid.

        Examples
        --------
        >>> # assume `s` is an instance of a concrete FuzznumStrategy with attribute 'p'
        >>> s.add_attribute_validator('p', lambda v: isinstance(v, (int, float)) and 0 <= v <= 1)
        >>> s.p = 0.3   # valid assignment
        >>> try:
        ...     s.p = 1.5  # validator will reject this assignment
        ... except ValueError:
        ...     print('invalid')
        invalid
        """
        self._attribute_validators[attr_name] = validator

    def add_change_callback(self,
                            attr_name: str,
                            callback: Callable[[str, Any, Any], None]) -> None:
        """
        Register a change callback for an attribute.

        This method allows registering a function to be called whenever a specific
        attribute's value changes. Callbacks can be used to trigger side effects,
        update dependent properties, or perform additional validation after a change.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to monitor.
        callback : callable
            Callable with signature (attr_name, old_value, new_value).
        
        Examples
        --------
        >>> # assume `s` is an instance of a concrete FuzznumStrategy with attribute 'score'
        >>> def on_change(name, old, new):
        ...     print(f"{name} changed from {old} to {new}")
        ...
        >>> s.add_change_callback('score', on_change)
        >>> s.score = 0.7
        score changed from None to 0.7
        """
        self._change_callbacks[attr_name] = callback

    def _validate(self) -> None:
        """
        Perform internal consistency checks.

        This is a protected method intended to be overridden by subclasses to
        implement specific validation logic relevant to their fuzzy number type.
        It's typically used for complex constraints involving multiple attributes
        or for validation that cannot be performed at the individual attribute
        setting level.

        The default implementation performs a basic check on the `mtype` attribute.
        Subclasses should call `super()._validate()` to ensure base class validation
        is also performed.

        Raises
        ------
        ValueError
            If any core validation constraint is violated.
        """
        # Check if 'mtype' exists and is a non-empty string.
        # This is a fundamental property for all fuzzy number strategies.
        if hasattr(self, 'mtype') and (not isinstance(self.mtype, str) or not self.mtype.strip()):
            raise ValueError(f"mtype must be a non-empty string, got '{self.mtype}'")

    def _generate_cache_key(self,
                            op_name: str,
                            operands: Union['FuzznumStrategy', int, float],
                            tnorm: OperationTNorm) -> Optional[str]:
        """
        Generates a unique cache key for an operation.

        The key is based on the operation name, the state of the operands (including
        the current FuzznumStrategy instance and the other operand), and the
        parameters of the t-norm used. This ensures that identical operations
        with identical inputs and t-norm settings can retrieve cached results.

        Parameters
        ----------
        op_name : str
            Operation name (e.g., 'add', 'mul').
        operands : FuzznumStrategy or int or float
            The second operand.
        tnorm : OperationTNorm
            T-norm instance describing operator parameters.

        Returns
        -------
        str or None
            MD5 hex digest representing the key, or None if generation fails.
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
        """
        Executes a specified operation with another operand.

        This method acts as a central dispatcher for performing various fuzzy
        number operations (e.g., addition, multiplication, comparison, complement).
        It retrieves the appropriate operation handler from the global registry
        based on the operation name and the fuzzy number's `mtype`. It also
        manages caching of operation results.

        Parameters
        ----------
        op_name : str
            The name of the operation to execute (e.g., 'add', 'mul', 'complement').
        operand : Optional[Union[FuzznumStrategy, int, float]]
            The second operand for the operation. Can be another FuzznumStrategy instance,
            an integer, or a float, depending on the operation.

        Returns
        -------
            Dict[str, Any]
                A dictionary containing the result of the operation.
                The structure of the dictionary depends on the specific operation.

        Raises
        ------
            NotImplementedError
                If the requested operation is not supported for the current `mtype`.
            TypeError
                If the operand type is incompatible with the operation.
            ValueError
                If an unknown operation type is requested.
        """
        # Get the global operation registry.
        registry = get_registry_operation()
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

        Returns
        -------
        set
            Copy of the declared attributes set.
        """
        return self._declared_attributes.copy()

    def get_available_operations(self) -> List[str]:
        """Gets the names of all operations supported by the current fuzzy number type.

        This method queries the global operation registry to determine which
        operations are registered and available for the `mtype` of this
        `FuzznumStrategy` instance.

        Returns
        -------
        List[str]
            A list of strings, where each string is the name of an
            operation supported by this fuzzy number type.
        """
        operation = get_registry_operation()
        return operation.get_available_ops(self.mtype)

    def validate_all_attributes(self) -> Dict[str, Any]:
        """Validates all attributes of the FuzznumStrategy instance.

        This method provides a unified interface to trigger a comprehensive
        health check of the instance's state. It performs both the general
        `_validate()` checks and individual attribute validations.

        Returns:
        -------
        Dict[str, Any]
            A dictionary containing the validation results.
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

    # ======================== Representation & Properties ========================

    @abstractmethod
    def report(self) -> str:
        """
        Produce a detailed textual report of the fuzzy number.

        Returns
        -------
        str
            Multi-line detailed representation.

        Notes
        -----
        - Must be implemented by concrete strategies.
        """
        raise NotImplementedError

    @abstractmethod
    def str(self) -> str:
        """
        Produce a concise string representation.

        Returns
        -------
        str
            Short one-line representation.

        Notes
        -----
        - Subclasses may override; default can delegate to `report`.
        """
        return NotImplemented
