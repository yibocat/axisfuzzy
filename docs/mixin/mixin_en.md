# FuzzLab Mixin System

The FuzzLab Mixin System is a powerful mechanism that dynamically extends the functionality of core classes like `Fuzznum` and `Fuzzarray`. It allows for adding new methods to these classes and creating corresponding top-level functions in the `fuzzlab` namespace without directly modifying the class definitions. This is achieved through a central registry that manages and injects these functions during the library's initialization.

## Core Concept: `MixinFunctionRegistry`

The heart of the system is the `MixinFunctionRegistry`. It provides a decorator-based approach to register functions and specify how they should be integrated into the FuzzLab ecosystem.

A function can be injected in three ways:
- **`instance_function`**: The function becomes an instance method of the target class (e.g., `my_fuzzarray.my_func()`).
- **`top_level_function`**: The function is available directly from the `fuzzlab` module (e.g., `fuzzlab.my_func(...)`).
- **`both`**: The function is available as both an instance method and a top-level function.

This design keeps the core class definitions clean while allowing for easy and organized extension of their capabilities.

## Available Mixin Functions

The following functions are available through the mixin system. They are grouped by their source module.

### Array Manipulation (`fuzzlab.mixin.function`)

These functions provide `numpy`-like array manipulation capabilities for `Fuzzarray` and `Fuzznum` objects.

| Function/Method | Injection Type | Description |
|---|---|---|
| `reshape` | both | Gives a new shape to an array without changing its data. |
| `flatten` | both | Returns a copy of the array collapsed into one dimension. |
| `squeeze` | both | Removes single-dimensional entries from the shape of an array. |
| `copy` | top_level_function | Returns a deep copy of the fuzzy object. |
| `ravel` | both | Returns a contiguous flattened array. A view is returned if possible. |
| `transpose` | top_level_function | Returns a view of the fuzzy object with axes transposed. |
| `broadcast_to` | both | Broadcasts the fuzzy object to a new shape. |
| `item` | both | Returns the scalar item of the fuzzy object. |
| `sort` | both | Returns a sorted copy of a fuzzy array. |
| `argsort` | both | Returns the indices that would sort a fuzzy array. |
| `argmax` | both | Returns the indices of the maximum values along an axis. |
| `argmin` | both | Returns the indices of the minimum values along an axis. |
| `concat` | both | Concatenates one or more `Fuzzarray`s along a specified axis. |
| `stack` | both | Stacks `Fuzzarray`s along a new axis. |
| `append` | both | Appends elements to an object. |
| `pop` | both | Removes and returns an element from a 1-D array. |


### Mathematical & Aggregation Operations (`fuzzlab.mixin.ops`)

These functions provide `numpy`-like mathematical and aggregation capabilities.

| Function/Method | Injection Type | Description |
|---|---|---|
| `sum` | both | Calculates the sum of all elements. |
| `mean` | both | Calculates the mean of all elements. |
| `max` | both | Finds the maximum element. |
| `min` | both | Finds the minimum element. |
| `prod` | both | Calculates the product of all elements. |
| `var` | both | Computes the variance. |
| `std` | both | Computes the standard deviation. |
