#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/5 16:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module provides a flexible function registry system.

It allows registering functions as instance methods for specific classes
and as top-level functions. The system supports dynamic injection of
these functions into target classes and module namespaces.
"""
import functools
from typing import Dict, Callable, Union, List, Any


class FunctionRegistry:
    """
    A registry for managing and injecting functions into classes and modules.

    This class provides decorators to register functions as instance methods
    for specific classes or as global top-level functions. It also handles
    the injection of these registered functions into their respective targets.

    Attributes:
        _instance_functions (Dict[str, Dict[str, Callable]]): Stores instance methods
            mapped by function name and then by target class name.
        _top_level_functions (Dict[str, Callable]): Stores top-level functions
            mapped by function name.
    """

    def __init__(self):
        self._instance_functions: Dict[str, Dict[str, Callable]] = {}
        self._top_level_functions: Dict[str, Callable] = {}

    def register_instance_function(self, name: str, target_class: str) -> Callable:
        """
        Decorator to register a function as an instance method for a specific class.

        The registered function will be called as a method of the target class
        (e.g., `self, *args, **kwargs`).

        Args:
            name (str): The name under which the function will be registered.
            target_class (str): The name of the class to which this method belongs.

        Returns:
            Callable: A decorator that registers the function.

        Raises:
            ValueError: If an instance method with the same name is already
                        registered for the target class.
        """

        def decorator(func: Callable) -> Callable:
            # Initialize the dictionary for the function name if it doesn't exist
            if name not in self._instance_functions:
                self._instance_functions[name] = {}
            # Check if the instance method for the target class is already registered
            if target_class in self._instance_functions[name]:
                raise ValueError(f"Instance method '{name}' for class '{target_class}' already registered.")
            # Register the function
            self._instance_functions[name][target_class] = func
            return func

        return decorator

    def register_top_level_function(self, name: str) -> Callable:
        """
        Decorator to register a function as a top-level function.

        The registered function will be called as a module-level function
        (e.g., `*args, **kwargs`).

        Args:
            name (str): The name under which the function will be registered.

        Returns:
            Callable: A decorator that registers the function.

        Raises:
            ValueError: If a top-level function with the same name is already registered.
        """

        def decorator(func: Callable) -> Callable:
            # Check if the top-level function is already registered
            if name in self._top_level_functions:
                raise ValueError(f"Top-level function '{name}' already registered.")
            # Register the function
            self._top_level_functions[name] = func
            return func

        return decorator

    def register(self, name: str, target_classes: Union[str, List[str]]) -> Callable:
        """
        Decorator to register a function as both an instance method and a top-level function.

        The decorated function must accept an instance as its first argument.
        If `target_classes` is a list, the instance method is registered for each class.
        The top-level function will dynamically call the appropriate instance method
        based on the type of the instance passed as the first argument.

        Args:
            name (str): The name under which the function will be registered.
            target_classes (Union[str, List[str]]): The name(s) of the class(es)
                                                    to which this method belongs.

        Returns:
            Callable: A decorator that registers the function.
        """
        # Ensure target_classes is always a list for uniform processing
        if isinstance(target_classes, str):
            target_classes = [target_classes]

        def decorator(func: Callable) -> Callable:
            # 1. Register as instance methods for each target class
            for cls_name in target_classes:
                self.register_instance_function(name, cls_name)(func)

            # 2. Register as a top-level function
            @functools.wraps(func)
            def top_level_wrapper(obj: Any, *args, **kwargs):
                """
                Wrapper function for the top-level function.

                This wrapper dynamically dispatches the call to the appropriate
                instance method based on the type of the `obj` argument.

                Args:
                    obj (Any): The instance on which the method should be called.
                    *args: Positional arguments to pass to the instance method.
                    **kwargs: Keyword arguments to pass to the instance method.

                Returns:
                    Any: The result of the instance method call.

                Raises:
                    TypeError: If the method is not supported for the given object type.
                """
                # Dynamically call the instance method
                if hasattr(obj, name) and callable(getattr(obj, name)):
                    return getattr(obj, name)(*args, **kwargs)
                else:
                    raise TypeError(f"'{name}' is not supported for type {type(obj).__name__}")

            self.register_top_level_function(name)(top_level_wrapper)
            return func

        return decorator

    def build_and_inject(self, class_map: Dict[str, type], module_namespace: Dict[str, Any]):
        """
        Builds dispatcher functions and injects them into target classes,
        and injects top-level functions into the module namespace.

        Args:
            class_map (Dict[str, type]): A dictionary mapping class names (strings)
                                         to actual class objects.
            module_namespace (Dict[str, Any]): The namespace (e.g., `globals()`)
                                               where top-level functions should be injected.
        """
        # Inject instance methods
        for name, implementations in self._instance_functions.items():
            @functools.wraps(list(implementations.values())[0])
            def dispatcher(self_obj, *args,
                           current_name=name,
                           current_implementations=implementations,
                           **kwargs):
                """
                A dispatcher function that routes calls to the correct instance method
                implementation based on the `self_obj`'s class.

                Args:
                    self_obj (Any): The instance on which the method is called.
                    *args: Positional arguments for the method.
                    current_name (str): The name of the function being dispatched.
                    current_implementations (Dict[str, Callable]): A dictionary of
                                                                   implementations
                                                                   for different classes.
                    **kwargs: Keyword arguments for the method.

                Returns:
                    Any: The result of the specific instance method implementation.

                Raises:
                    NotImplementedError: If the function is not implemented for the
                                         `self_obj`'s class.
                """
                # Get the class name of the current object
                classname = self_obj.__class__.__name__
                # Retrieve the specific implementation for this class
                implementation = current_implementations.get(classname)

                if implementation:
                    # Call the specific implementation
                    return implementation(self_obj, *args, **kwargs)

                # Raise an error if no implementation is found for the class
                raise NotImplementedError(
                    f"Function '{current_name}' is not implemented for class '{classname}'."
                )

            # Inject the dispatcher into each target class
            for class_name in implementations.keys():
                if class_name in class_map:
                    target_class = class_map[class_name]
                    setattr(target_class, name, dispatcher)

        # Inject top-level functions into the module namespace
        for name, func in self._top_level_functions.items():
            module_namespace[name] = func

    def get_top_level_function_names(self) -> List[str]:
        """
        Returns a list of names of all registered top-level functions.

        Returns:
            List[str]: A list of function names.
        """
        return list(self._top_level_functions.keys())


# Create a global instance of the registry
_registry = FunctionRegistry()


def get_mixin_registry():
    """
    Returns the global FunctionRegistry instance.

    This function provides singleton-like access to the registry,
    ensuring all mixin functions are registered with the same instance.

    Returns:
        FunctionRegistry: The global registry instance.
    """
    return _registry
