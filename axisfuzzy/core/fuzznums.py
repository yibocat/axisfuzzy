#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
High-level Fuzznum wrapper and factory utilities.

This module provides the :class:`Fuzznum` facade, which binds a concrete per-element
strategy (:class:`FuzznumStrategy`) implementation to a lightweight user-facing
object. It also provides the convenient factory function :func:`fuzznum`.

Overview
--------
- `Fuzznum` is the main user-facing class for single fuzzy numbers.
- It delegates all logic and data to a registered `FuzznumStrategy` subclass, determined by `mtype`.
- All mathematical operations, validation, and formatting are handled by the strategy.
- The facade exposes strategy attributes and methods as if they were native to `Fuzznum`.

Notes
-----
- The actual fuzzy number logic is implemented in strategy classes (see `axisfuzzy/core/base.py`).
- The registry system allows new fuzzy number types to be plugged in without modifying this module.
- Operator overloading is supported via the dispatcher system.

Examples
--------
.. code-block:: python

    from axisfuzzy.core.fuzznums import fuzznum

    # Create a default fuzzy number (mtype and q from config)
    a = fuzznum()
    b = fuzznum(mtype='qrofn', q=3, md=0.7, nmd=0.2)

    # Access attributes
    print(a.mtype, a.q)

    # Set fuzzy number attributes
    a.md = 0.5
    a.nmd = 0.3

    # Use operator overloading (requires operation registration)
    c = a + b

    # Serialize and deserialize
    d = fuzznum.from_dict(a.to_dict())
"""

import difflib

from typing import Optional, Any, Dict, Callable, Set, List, Tuple

from ..config import get_config

from .registry import get_registry_fuzztype
from .base import FuzznumStrategy


class Fuzznum:
    """
    Facade object representing a single fuzzy number.

    Fuzznum binds a concrete strategy implementation (subclass of
    :class:`axisfuzzy.core.base.FuzznumStrategy`) at construction time and
    exposes the strategy's attributes and methods on the Fuzznum instance.
    This design keeps per-element logic in strategy classes while providing
    a small, ergonomic high-level API.

    Parameters
    ----------
    mtype : str, optional
        Name of the fuzzy-number strategy (membership type). When omitted,
        the default is read from configuration.
    q : int, optional
        q-rung parameter used by some strategies. When omitted, the default
        is read from configuration.

    Attributes
    ----------
    mtype : str
        Configured membership type name.
    q : int
        Effective q-rung for this instance.

    Notes
    -----
    - The Fuzznum instance dynamically binds strategy methods and attributes
      into itself at initialization. Attempting to access strategy-backed
      attributes before initialization raises an AttributeError.
    - Internal attributes are prefixed with an underscore and are not
      forwarded to the strategy.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.core.fuzznums import fuzznum

        # Create a QROFN fuzzy number
        a = fuzznum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
        print(a.md, a.nmd)

        # Copy and modify
        b = a.copy()
        b.md = 0.8

        # Serialize/deserialize
        d = fuzznum.from_dict(a.to_dict())
    """

    __array_priority__ = 1.0
    _INTERNAL_ATTRS = {
        'mtype',
        '_initialized',
        '_strategy_instance',
        '_bound_strategy_methods',
        '_bound_strategy_attributes',
    }

    def __init__(self,
                 mtype: Optional[str] = None,
                 q: Optional[int] = None):

        object.__setattr__(self, '_initialized', False)
        if mtype is None:
            mtype = get_config().DEFAULT_MTYPE

        if q is None:
            q = get_config().DEFAULT_Q

        if not isinstance(mtype, str):
            raise TypeError(f"mtype must be a string type, got '{type(mtype).__name__}'")

        if not isinstance(q, int):
            raise TypeError(f"q must be an integer, got '{type(q).__name__}'")

        object.__setattr__(self, 'mtype', mtype)
        object.__setattr__(self, 'q', q)

        try:
            self._initialize()
            object.__setattr__(self, '_initialized', True)

        except Exception:
            self._cleanup_partial_initialization()
            raise

    def _initialize(self):
        self._configure_strategy()

    def _configure_strategy(self):
        registry = get_registry_fuzztype()

        if self.mtype not in registry.strategies:
            available_mtypes = ', '.join(registry.strategies.keys())
            raise ValueError(
                f"Unsupported strategy mtype: '{self.mtype}'. "
                f"Available mtypes: '{available_mtypes}'"
            )

        strategy_instance = registry.strategies[self.mtype](self.q)
        bound_methods, bound_attributes = self._bind_instance_members(
            strategy_instance, 'strategy'
        )

        object.__setattr__(self, '_strategy_instance', strategy_instance)
        object.__setattr__(self, '_bound_strategy_methods', bound_methods)
        object.__setattr__(self, '_bound_strategy_attributes', bound_attributes)

    def _bind_instance_members(self,
                               instance: Any,
                               instance_type: str) -> tuple[Dict[str, Callable[..., Any]], Set[str]]:
        bound_methods: Dict[str, Callable[..., Any]] = {}
        bound_attributes: Set[str] = set()

        exclude_attrs = {'mtype'}

        try:
            for attr_name in dir(instance):
                if attr_name.startswith('_') or attr_name in exclude_attrs:
                    continue

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
        cleanup_attrs = [
            '_strategy_instance',
            '_bound_strategy_methods',
            '_bound_strategy_attributes',
        ]

        for attr in cleanup_attrs:
            try:
                object.__delattr__(self, attr)
            except AttributeError:
                pass

    def _is_initialized(self) -> bool:
        try:
            return object.__getattribute__(self, '_initialized')
        except AttributeError:
            return False

    def _delegate_attribute_access(self, name: str) -> Any:
        try:
            bound_strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')
            if name in bound_strategy_attrs:
                value = getattr(self.get_strategy_instance(), name)
                return value
        except (AttributeError, RuntimeError):
            pass
        return self.__getattr__(name)

    def __getattribute__(self, name: str) -> Any:
        if name in Fuzznum._INTERNAL_ATTRS:
            return object.__getattribute__(self, name)

        if name.startswith('_') or name in ('__dict__', '__class__'):
            return object.__getattribute__(self, name)

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
        if not self._is_initialized():
            raise AttributeError(
                f"The '{self.__class__.__name__}' object has no attribute '{name}'."
                f"The Fuzznum is still initializing."
            )

        available_info = self._get_available_members_info()

        error_msg = f"'{self.__class__.__name__}' object has no attribute '{name}'."
        all_members = available_info['attributes'] + available_info['methods']
        suggestions = difflib.get_close_matches(name, all_members, n=3, cutoff=0.6)
        if suggestions:
            error_msg += f" Did you mean: {', '.join(suggestions)}?"

        if available_info['attributes']:
            error_msg += f"\nAvailable attributes: {', '.join(sorted(available_info['attributes']))}."

        if available_info['methods']:
            error_msg += f"\nAvailable methods: {', '.join(sorted(available_info['methods']))}."

        raise AttributeError(error_msg)

    def __setattr__(self, name: str, value: Any) -> None:

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
                    strategy_class = strategy_instance.__class__
                    attr_descriptor = getattr(strategy_class, name, None)

                    if isinstance(attr_descriptor, property):
                        if attr_descriptor.fset:
                            setattr(strategy_instance, name, value)
                            return
                        else:
                            raise AttributeError(f"The attribute '{name}' is read-only "
                                                 f"for the fuzzy number mtype '{self.mtype}'.")
                    else:
                        # 1. Set the attribute on the strategy instance.
                        # This will trigger the strategy's __setattr__, including any
                        # validators and, crucially, any transformers.
                        setattr(strategy_instance, name, value)

                        # 2. Read the attribute back from the strategy.
                        # This ensures we get the final, potentially transformed value
                        # (e.g., a list converted to an ndarray).
                        final_value = getattr(strategy_instance, name)

                        # 3. Set the final, corrected value on the Fuzznum instance itself.
                        # This maintains consistency between the facade and the strategy.
                        object.__setattr__(self, name, final_value)
                        return

                except AttributeError as e:
                    raise AttributeError(f"Cannot set property '{name}' on the strategy instance "
                                         f"(fuzzy number mtype '{self.mtype}'): {e}")
                except Exception as e:
                    raise RuntimeError(f"An unexpected error occurred while setting the property '{name}' "
                                       f"on the strategy instance (fuzzy number type '{self.mtype}'): {e}")

        except (AttributeError, Exception) as e:
            # If the attribute is not a bound strategy
            raise e

        object.__setattr__(self, name, value)

    def __del__(self) -> None:
        try:
            container = object.__getattribute__(self, '_bound_strategy_methods')
            if hasattr(container, 'clear'):
                container.clear()
        except AttributeError:
            pass

    def _get_available_members_info(self) -> Dict[str, List[str]]:
        try:
            strategy_methods = object.__getattribute__(self, '_bound_strategy_methods')
            strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')

            return {
                'attributes': list(strategy_attrs),
                'methods': list(strategy_methods.keys())
            }
        except AttributeError:
            return {'attributes': [], 'methods': []}

    def create(self, **kwargs) -> 'Fuzznum':
        """
        Create a new Fuzznum instance of the same mtype/q and set initial attributes.

        Parameters
        ----------
        **kwargs :
            Attribute names and values to assign on the created Fuzznum. Only
            attributes declared by the underlying strategy will be set.

        Returns
        -------
        Fuzznum
            A newly constructed Fuzznum instance with provided attributes applied.

        Raises
        ------
        AttributeError
            If an attribute provided in ``kwargs`` is not accepted by the strategy.
        """
        instance = Fuzznum(self.mtype, self.q)
        if kwargs:
            for key, value in kwargs.items():
                try:
                    setattr(instance, key, value)
                except Exception as e:
                    raise AttributeError(
                        f"The parameter '{key}' is invalid for the fuzzy number mtype '{self.mtype}': {e}"
                    ) from e
        return instance

    def copy(self) -> 'Fuzznum':
        """
        Produce a shallow copy of this Fuzznum preserving current strategy attributes.

        Returns
        -------
        Fuzznum
            New Fuzznum instance configured with the same mtype/q and strategy
            attribute values as this instance.

        Raises
        ------
        RuntimeError
            If called on an uninitialized Fuzznum.
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

        return self.create(**current_params)

    def get_strategy_instance(self) -> FuzznumStrategy:
        """
        Return the bound FuzznumStrategy instance.

        Returns
        -------
        FuzznumStrategy
            The strategy object that implements the per-element behavior.

        Raises
        ------
        RuntimeError
            If the strategy instance is not available (e.g., partially initialized).
        """
        try:
            strategy_instance = object.__getattribute__(self, '_strategy_instance')
            if strategy_instance is None:
                raise RuntimeError("Strategy instance not found.")
            return strategy_instance
        except AttributeError:
            raise RuntimeError("Strategy instance not found.")

    def get_strategy_attributes_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary of strategy attributes and their current values.

        Returns
        -------
        dict
            Dictionary mapping attribute names to their current values.
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
        Get basic information about the Fuzznum instance.

        Returns
        -------
        dict
            Dictionary containing basic information about the Fuzznum instance.
        """
        if not self._is_initialized():
            return {
                'mtype': getattr(self, 'mtype', 'unknown'),
                'status': 'not_initialized',
            }

        try:
            strategy_methods = object.__getattribute__(self, '_bound_strategy_methods')
            strategy_attributes = object.__getattribute__(self, '_bound_strategy_attributes')

            return {
                'mtype': self.mtype,
                'status': 'initialized',
                'binding_info': {
                    'bound_methods': sorted(list(strategy_methods.keys())),
                    'bound_attributes': sorted(list(strategy_attributes)),
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
        Validate internal consistency of the Fuzznum and its bound strategy.

        Returns
        -------
        dict
            Validation summary with keys: 'is_valid', 'issues', 'warnings'.

        Notes
        -----
        - Calls into the underlying strategy's validation method if available.
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
                '_strategy_instance',
                '_bound_strategy_methods',
                '_bound_strategy_attributes',
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

        except Exception as e:
            validation_result['issues'].append(f"An exception occurred during the verification process.: {e}")
            validation_result['is_valid'] = False

        return validation_result

    # ========================= Special Attributes ==============================

    @property
    def shape(self) -> Tuple[int, ...]:
        return ()

    @property
    def ndim(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    # ================== Specific calculation method (operator overloading) ===============

    def __add__(self, other):
        """Overloads the addition operator (+)."""
        from .dispatcher import operate
        return operate('add', self, other)

    def __sub__(self, other):
        """Overloads the subtraction operator (-)."""
        from .dispatcher import operate
        return operate('sub', self, other)

    def __mul__(self, other):
        """Overloads the multiplication operator (*)."""
        from .dispatcher import operate
        return operate('mul', self, other)

    def __rmul__(self, other):
        """Overloads the reverse multiplication operator (*)."""
        from .dispatcher import operate
        return operate('mul', self, other)

    def __truediv__(self, other):
        """Overloads the true division operator (/)."""
        from .dispatcher import operate
        return operate('div', self, other)

    def __pow__(self, power, modulo=None):
        """Overloads the power operator (**)."""
        from .dispatcher import operate
        return operate('pow', self, power)

    def __gt__(self, other):
        """Overloads the greater than operator (>)."""
        from .dispatcher import operate
        return operate('gt', self, other)

    def __lt__(self, other):
        """Overloads the less than operator (<)."""
        from .dispatcher import operate
        return operate('lt', self, other)

    def __ge__(self, other):
        """Overloads the greater than or equal to operator (>=)."""
        from .dispatcher import operate
        return operate('ge', self, other)

    def __le__(self, other):
        """Overloads the less than or equal to operator (<=)."""
        from .dispatcher import operate
        return operate('le', self, other)

    def __eq__(self, other):
        """Overloads the equality operator (==)."""
        from .dispatcher import operate
        return operate('eq', self, other)

    def __ne__(self, other):
        """Overloads the inequality operator (!=)."""
        from .dispatcher import operate
        return operate('ne', self, other)

    def __and__(self, other):
        """Overloads the and operator (&).
            intersection operation.
        """
        from .dispatcher import operate
        return operate('intersection', self, other)

    def __or__(self, other):
        from .dispatcher import operate
        return operate('union', self, other)

    def __invert__(self, other=None):
        """Overloads the invert operator (~).
            Complement operation.
        """
        from .dispatcher import operate
        return operate('complement', self, other)

    def __lshift__(self, other):
        """Overloads the left shift operator (<<).
            Denotes the left implication operation: self <- other
        """
        from .dispatcher import operate
        return operate('implication', other, self)

    def __rshift__(self, other):
        """Overloads the shift operator (>>).
            Denotes the right implication operation: self -> other
        """
        from .dispatcher import operate
        return operate('implication', self, other)

    def __xor__(self, other):
        """Overloads the xor operator (^).
            Denotes the symmetric difference operation.
        """
        from .dispatcher import operate
        return operate('symdiff', self, other)

    def equivalent(self, other):
        """
        Calculate the equivalence level between two fuzzy numbers

        Corresponding to the "if and only if" operation in classical logic,
        it represents the degree to which two fuzzy propositions are
        equivalent to each other.
        """
        from .dispatcher import operate
        return operate('equivalence', self, other)

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
        Serialize the Fuzznum to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary containing 'mtype' and 'attributes' describing the object.

        Raises
        ------
        RuntimeError
            If the object is not fully initialized.
        """
        if not self._is_initialized():
            raise RuntimeError("Unable to serialize uninitialized object")
        try:
            result = {
                'mtype': self.mtype,
                'attributes': {},
            }
            strategy_attrs = object.__getattribute__(self, '_bound_strategy_attributes')
            for attr_name in strategy_attrs:
                try:
                    result['attributes'][attr_name] = getattr(self, attr_name)
                except AttributeError:
                    pass

            return result
        except AttributeError as e:
            raise RuntimeError(f"Failed to serialize: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fuzznum':
        """
        Construct a Fuzznum from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Dictionary containing at minimum the 'mtype' key and, optionally,
            an 'attributes' mapping.

        Returns
        -------
        Fuzznum
            Reconstructed Fuzznum instance.

        Raises
        ------
        ValueError
            If 'mtype' is missing from the input dictionary.
        """
        if 'mtype' not in data:
            raise ValueError("The dictionary must contain the 'mtype' key.")

        instance = cls(data['mtype'])

        if 'attributes' in data:
            for attr_name, value in data['attributes'].items():
                try:
                    setattr(instance, attr_name, value)
                except AttributeError:
                    pass

        return instance

    def __getstate__(self) -> Dict[str, Any]:
        return self.to_dict()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if 'mtype' not in state:
            raise ValueError("Invalid state for Fuzznum: 'mtype' key is missing.")

        self.__init__(mtype=state['mtype'], q=state.get('q', 1))

        if 'attributes' in state:
            for attr_name, value in state['attributes'].items():
                try:
                    setattr(self, attr_name, value)
                except AttributeError:
                    pass

    # ================================ Type Conversion ===============================

    def __repr__(self) -> str:
        if hasattr(self, 'report') and callable(self.report):
            try:
                return self.get_strategy_instance().report()
            except ValueError:
                pass

        return f"Fuzznum[{getattr(self, 'mtype', 'unknown')}]"

    def __str__(self) -> str:
        if hasattr(self, 'str') and callable(self.str):
            try:
                return self.get_strategy_instance().str()
            except ValueError:
                pass
        return f"Fuzznum[{getattr(self, 'mtype', 'unknown')}]"

    def __bool__(self) -> bool:
        return True

    def __format__(self, format_spec: str) -> str:
        try:
            return self.get_strategy_instance().__format__(format_spec)
        except (RuntimeError, AttributeError):
            return format(str(self), format_spec)


# ================================= 工厂函数 =================================

def fuzznum(mtype: Optional[str] = None,
            q: Optional[int] = None,
            **kwargs: Any) -> Fuzznum:
    """
    Factory function to create a Fuzznum instance.

    Parameters
    ----------
    mtype : str, optional
        The type of fuzzy number strategy to use. If omitted, uses the default from config.
    q : int, optional
        The discretization level for the fuzzy number. If omitted, uses the default from config.
    **kwargs : dict
        Additional parameters specific to the chosen fuzzy number strategy.

    Returns
    -------
    Fuzznum
        An instance of Fuzznum configured with the specified strategy and parameters.

    Examples
    --------
    .. code-block:: python

        a = fuzznum(mtype='qrofn', q=3, md=0.7, nmd=0.2)
        print(a)
    """
    mtype = mtype or get_config().DEFAULT_MTYPE
    q = q or get_config().DEFAULT_Q

    instance = Fuzznum(mtype, q)
    if kwargs:
        return instance.create(**kwargs)
    return instance
