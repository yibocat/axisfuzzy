The FuzzLab extension system is an ingeniously designed and highly flexible mechanism that allows developers to dynamically add and manage functionalities for different types of fuzzy numbers (`mtype`). Its core idea is a `mtype`-based pluggable architecture, enabling FuzzLab to easily extend support for new fuzzy number types or provide specialized operations for existing ones, without modifying the core code.

## 1. Overall Architecture and Operational Mechanism
The FuzzLab extension system primarily consists of the following core components:

### 1. Registry (`ExtensionRegistry`)
- File: `registry.py`
- Role: It is the "brain" of the entire extension system, responsible for storing all registered functions and their metadata.
- Core Functions:
    - `register()`: Registers functions via the `@extension` decorator (see below). It supports registering specialized implementations for specific `mtype`s, as well as general default implementations.
    - `get_function()`: Retrieves the corresponding function implementation based on the function name and `mtype`. If a specialized implementation exists, it returns that; otherwise, it falls back to the default implementation.
    - `list_functions()`: Lists all registered functions and their detailed information.
- Features: Thread-safe, supports priority-based selection (higher priority implementations are preferred when multiple exist).

### 2. Decorators (`extension`, `batch_extension`)
- File: `decorators.py`
- Role: Provides concise syntactic sugar for registering functions with the `ExtensionRegistry`.
- Core Functions:
  - `@extension`: This is the most commonly used decorator for registering a single function. You can specify the function name (`name`), applicable `mtype`, target classes (`target_classes`, e.g., `Fuzznum` or `Fuzzarray`), injection type (`injection_type`, which can be `instance_method`, `top_level_function`, `instance_property` or `both`), whether it's a default implementation (`is_default`), and its `priority`.
  - `@batch_extension`: Used for batch registration of multiple functions, facilitating management.
- injection_type supports:
  - instance_method: injected as a dispatched instance method.
  - top_level_function: injected into fuzzlab module namespace.
  - both: combines the above two.
  - instance_property (NEW): injected as a lazily evaluated dispatched @property (read‑only).
    - Used for lightweight computed features (e.g. score, acc, ind).
    - Implementation function still receives the instance as the first argument.

Example (register a dispatched property):
```python
@extension(
    name='score',
    mtype='qrofn',
    target_classes=['Fuzznum', 'Fuzzarray'],
    injection_type='instance_property'
)
def qrofn_score_ext(obj):
    return obj.md ** obj.q - obj.nmd ** obj.q
```
After injection: obj.score

### 3. Dispatcher (`ExtensionDispatcher`)
- File: `dispatcher.py`
- Role: Responsible for dispatching method calls to the correct concrete implementation at runtime based on the object's `mtype`.
- Core Functions:
  - `create_instance_method()`: Creates a "proxy" instance method. When this method is called, it inspects the `mtype` of the calling object and then finds and invokes the correct registered extension implementation.
  - `create_top_level_function()`: Similarly, creates a "proxy" top-level function for handling `mtype` dispatch during top-level function calls.
- Operational Mechanism: It does not directly execute the function but generates a wrapper function. This wrapper function, when called, dynamically looks up and executes the most matching implementation from the registry.
- Notes:
    - Added create_instance_property(name) returning a property descriptor.
    - Property getter performs the same mtype dispatch as methods.
    - No __name__ on property objects; documentation is supplied via doc argument.

### 4. Injector (`ExtensionInjector`)
- File: `injector.py`
- Role: Dynamically injects functions defined in the registry into `Fuzznum` and `Fuzzarray` classes, or as top-level functions in the `fuzzlab` module, during program startup.
- Core Functions:
  - `inject_all()`: Iterates through all registered functions in the registry and, based on their `injection_type`, injects them into the specified classes (`Fuzznum`, `Fuzzarray`) or module namespace.
- Operational Mechanism: It uses `setattr()` or similar methods to bind the dispatcher-created proxy functions to the target classes or module, making the extensions callable like native methods or functions.
- Note:
  - inject_all now also detects registrations whose injection_type includes instance_property and attaches a dispatched property (if attribute not already present).

### 5. Utility Function (`call_extension`)
- File: `utils.py`
- Role: Provides a helper function that allows one extension function to call another internally, without concerns about circular dependencies or injection timing issues.
- Core Functions: `call_extension(func_name, obj, *args, **kwargs)`: Directly calls an extension function by name and object `mtype` from the registry.

### 6. Initialization (`apply_extensions`)
- File: `__init__.py` (in `fuzzlab.extension` and `fuzzlab` root)
- Role: This is the entry point to activate the entire extension system.
- Operational Mechanism: When the FuzzLab library is imported (via the `apply_extensions()` call in `fuzzlab/__init__.py`), the `apply_extensions()` function is executed. It obtains the `ExtensionInjector` instance and calls its `inject_all()` method, thereby completing the dynamic injection of all registered functions.

## 2. Core Philosophy
The overall philosophy of the FuzzLab extension system can be summarized as: "Registration - Dispatch - Injection". Registration – Dispatch – Injection now also covers attribute-style access for computed metrics.

- Registration: Developers use the simple `@extension` decorator to declare a function as a FuzzLab extension, specifying its name, applicable `mtype`, and injection method. This information is stored in the `ExtensionRegistry`.
- Dispatching: When a user calls an extension function (either as an instance method or a top-level function), the actual execution is handled by a proxy function created by the `ExtensionDispatcher`. This proxy function intelligently looks up and invokes the most appropriate concrete implementation from the `ExtensionRegistry` based on the `mtype` of the calling object.
- Injection: During FuzzLab library loading, the `ExtensionInjector` dynamically binds these proxy functions to the `Fuzznum` and `Fuzzarray` classes, or as top-level functions in the `fuzzlab` module. This allows users to call these extension functions just like regular methods or functions, without needing to understand the underlying `mtype` dispatch logic.

This design offers several significant advantages:

- Extensibility: Easily add new fuzzy number types and corresponding specialized functionalities without modifying the core code.
- Modularity: Separates the implementation of different `mtype`-specific functions, improving code maintainability.
- Flexibility: Supports both default and specialized implementations, along with a priority mechanism, to meet various requirements.
- Decoupling: The core `Fuzznum` and `Fuzzarray` classes are decoupled from specific function implementations; they only know how to call the dispatcher.
- User-Friendly: Users can call extension functions like regular methods, with the underlying complexity hidden.

### Example using `_func.py`
Let's look at the `qrofn_distance` function you wrote in `_func.py`:

```python
# ...existing code...
@extension(
    name='distance',
    mtype='qrofn',
    target_classes=['Fuzznum', 'Fuzzarray'],
    injection_type='both'
)
def qrofn_distance(fuzz1: Fuzznum, fuzz2: Fuzznum, p: int = 2) -> float:
    q = fuzz1.q

    md_diff = abs(fuzz1.md ** q - fuzz2.md ** q) ** p
    nmd_diff = abs(fuzz1.nmd ** q - fuzz2.nmd ** q) ** p
    return ((md_diff + nmd_diff) / 2) ** (1 / p)
```

Here's what happens:

1.  **`@extension(...)` Decorator**:
    *   `name='distance'`: This indicates that we are registering a function named `distance`.
    *   `mtype='qrofn'`: This `distance` function is specifically implemented for fuzzy numbers of type `qrofn`.
    *   `target_classes=['Fuzznum', 'Fuzzarray']`: This means the `distance` function will be injected into the `Fuzznum` and `Fuzzarray` classes as instance methods.
    *   `injection_type='both'`: This means the `distance` function will be injected as both an instance method of `Fuzznum` and `Fuzzarray`, and as a top-level function in the `fuzzlab` module.

2.  **Registration Process**:
    *   When the `_func.py` module is imported (typically during FuzzLab initialization), the `@extension` decorator executes.
    *   It calls `get_extension_registry().register(...)`, passing the `qrofn_distance` function and its metadata.
    *   The `ExtensionRegistry` stores this function and its associated metadata, making it available for lookup.

3.  **Injection Process**:
    *   When `apply_extensions()` in `fuzzlab.extension.__init__.py` is called (which is triggered by `fuzzlab/__init__.py` during library import):
    *   The `ExtensionInjector` retrieves information about the `distance` function from the `ExtensionRegistry`.
    *   Because `injection_type='both'`, the `ExtensionInjector` will:
        *   Call `ExtensionDispatcher.create_instance_method('distance')` to get a proxy function for instance methods. This proxy is then set as the `distance` method on both `Fuzznum` and `Fuzzarray` classes.
        *   Call `ExtensionDispatcher.create_top_level_function('distance')` to get a proxy function for top-level calls. This proxy is then set as `fuzzlab.distance` in the `fuzzlab` module's namespace.

4.  **Calling Process**:

    *   **As an Instance Method (e.g., `my_qrofn_fuzznum.distance(another_fuzznum)`)**:
        *   The call is intercepted by the proxy method (created by `ExtensionDispatcher`) that was injected into the `Fuzznum` class.
        *   This proxy method inspects `my_qrofn_fuzznum` to determine its `mtype` (e.g., `'qrofn'`).
        *   It then queries the `ExtensionRegistry` using the function name (`'distance'`) and the detected `mtype` (`'qrofn'`).
        *   The `ExtensionRegistry` returns the actual `qrofn_distance` function.
        *   The proxy method then executes the `qrofn_distance` function, passing `my_qrofn_fuzznum` and `another_fuzznum` as arguments.

    *   **As a Top-Level Function (e.g., `fuzzlab.distance(my_qrofn_fuzznum, another_fuzznum)`)**:
        *   The call is intercepted by the proxy function (created by `ExtensionDispatcher`) that was injected into the `fuzzlab` module's namespace.
        *   This proxy function inspects the first argument (`my_qrofn_fuzznum`) to determine its `mtype` (e.g., `'qrofn'`).
        *   It then queries the `ExtensionRegistry` using the function name (`'distance'`) and the detected `mtype` (`'qrofn'`).
        *   The `ExtensionRegistry` returns the actual `qrofn_distance` function.
        *   The proxy function then executes the `qrofn_distance` function, passing `my_qrofn_fuzznum` and `another_fuzznum` as arguments.

Through this mechanism, FuzzLab achieves high modularity and extensibility. You can define a `distance` function for any `mtype`, or even a generic default `distance` implementation, and the system will automatically select the most appropriate implementation based on the object's actual `mtype`.

## 3. New Example: Dispatched Properties (score / acc / ind)

For q-rung orthopair fuzzy numbers (QROFN):
- score = md^q - nmd^q
- acc (accuracy) = md^q + nmd^q
- ind (indeterminacy) = 1 - acc

Registered as:
```python
@extension(name='score', mtype='qrofn',
           target_classes=['Fuzznum','Fuzzarray'],
           injection_type='instance_property')
def qrofn_score_ext(x): return x.md ** x.q - x.nmd ** x.q
```

Usage:
```python
x.score   # dispatched, no parentheses
x.acc
x.ind
```
Performance:
- Implementations for arrays operate on backend SoA component arrays (vectorized power and arithmetic).
- Fuzznum path uses scalar computation.

## 4. Summary of injection_type (updated)

| injection_type       | Effect                                      |
|----------------------|---------------------------------------------|
| instance_method      | Inject as dispatched bound method           |
| top_level_function   | Inject into fuzzlab namespace               |
| both                 | Method + top-level                          |
| instance_property    | Read-only dispatched property               |

