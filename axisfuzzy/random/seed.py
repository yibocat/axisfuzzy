#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Random seed management for reproducible fuzzy number generation.

This module provides a centralized random state management system that ensures
reproducible random generation across the entire AxisFuzzy library while
maintaining thread safety and high performance. It serves as the foundation
for all random fuzzy number generation operations.

The module implements a global singleton pattern for random state management,
ensuring that all random generators throughout the library share a consistent
and controllable random state. This is essential for scientific reproducibility
and deterministic behavior in fuzzy computations.

Architecture
------------
The random state management follows these key principles:

- **Global Consistency**: Single source of truth for random state across the library
- **Thread Safety**: All operations are protected by locks for concurrent access
- **Reproducibility**: Deterministic behavior through proper seed management
- **Independence**: Ability to spawn independent generators for parallel operations
- **Performance**: Efficient NumPy generator usage with minimal overhead

The system is built around NumPy's modern random number generation framework,
using `numpy.random.Generator` instances for high-performance random sampling.

Classes
-------
GlobalRandomState
    Thread-safe manager for global random state and generator instances.

Functions
---------
set_seed : Set global random seed for reproducible generation
get_rng : Get the global random number generator instance
spawn_rng : Create independent generator for parallel operations
get_seed : Retrieve the current global seed value

See Also
--------
axisfuzzy.random.api : High-level random generation API
axisfuzzy.random.base : Abstract base classes for random generators
numpy.random : NumPy's random number generation framework

Examples
--------
Basic seed management:

.. code-block:: python

    import axisfuzzy.random as fr

    # Set global seed for reproducibility
    fr.set_seed(42)

    # Generate random fuzzy numbers - results will be reproducible
    num1 = fr.rand('qrofn', q=2)
    arr1 = fr.rand('qrofn', shape=(100,), q=3)

    # Reset to same seed to get identical results
    fr.set_seed(42)
    num2 = fr.rand('qrofn', q=2)  # Identical to num1
    arr2 = fr.rand('qrofn', shape=(100,), q=3)  # Identical to arr1

Advanced usage with independent generators:

.. code-block:: python

    import axisfuzzy.random as fr
    import numpy as np

    # Set global seed
    fr.set_seed(123)

    # Get the global generator for custom use
    rng = fr.get_rng()
    custom_samples = rng.uniform(0, 1, size=10)

    # Spawn independent generator for parallel work
    independent_rng = fr.spawn_rng()
    parallel_samples = independent_rng.normal(0, 1, size=100)

Notes
-----
The module uses NumPy's modern random number generation system introduced
in NumPy 1.17+. This provides better performance, statistical properties,
and parallel support compared to the legacy `numpy.random` interface.

All generators are instances of `numpy.random.Generator`, which is the
recommended interface for new code. The module ensures thread safety
through appropriate locking mechanisms without significant performance
penalties for typical usage patterns.
"""

import threading
from typing import Optional, Union

import numpy as np


class GlobalRandomState:
    """
    Thread-safe global random state manager for AxisFuzzy.

    This class manages a single numpy.random.Generator instance that serves
    as the primary source of randomness throughout the AxisFuzzy library.
    It provides thread-safe access to random number generation capabilities
    while ensuring reproducible behavior through proper seed management.

    The class implements a singleton-like pattern where a single global
    instance manages the random state for the entire library, ensuring
    consistency across all random operations.

    Attributes
    ----------
    _lock : threading.Lock
        Thread synchronization lock for safe concurrent access.
    _rng : numpy.random.Generator
        The primary random number generator instance.
    _seed : int, numpy.random.SeedSequence, numpy.random.BitGenerator, or None
        The current seed used to initialize the generator.

    Notes
    -----
    This class uses NumPy's modern random number generation framework based
    on `numpy.random.Generator`. This provides better performance and
    statistical properties compared to the legacy `numpy.random` interface.

    All methods are thread-safe and can be called concurrently from multiple
    threads without data races or inconsistent state.

    The generator can be seeded with various types:
    - `int`: Simple integer seed
    - `numpy.random.SeedSequence`: Advanced seed sequence for better entropy
    - `numpy.random.BitGenerator`: Pre-configured bit generator

    Examples
    --------
    Basic usage (typically done through module functions):

    .. code-block:: python

        # Create and configure global state
        state = GlobalRandomState()
        state.set_seed(42)

        # Get generator for random operations
        rng = state.get_generator()
        values = rng.uniform(0, 1, size=100)

        # Spawn independent generator
        independent_rng = state.spawn_generator()
        parallel_values = independent_rng.normal(0, 1, size=50)
    """

    def __init__(self):
        """
        Initialize the global random state manager.

        Creates a new random state manager with default initialization.
        The generator is initialized with system entropy for unpredictable
        behavior until a specific seed is set.
        """
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()
        self._seed = None

    def set_seed(self, seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = None):
        """
        Set the global random seed for reproducible generation.

        Configures the internal random number generator with a specific seed,
        enabling reproducible random number generation across the entire
        AxisFuzzy library. This is essential for scientific reproducibility
        and debugging.

        Parameters
        ----------
        seed : int, numpy.random.SeedSequence, numpy.random.BitGenerator, optional
            The seed to use for random number generation. Different types provide
            different levels of control:

            - `int`: Simple integer seed (0 to 2^32-1)
            - `numpy.random.SeedSequence`: Advanced seed with entropy mixing
            - `numpy.random.BitGenerator`: Pre-configured bit generator
            - `None`: Use system entropy for non-reproducible behavior

        Notes
        -----
        Setting the seed affects all subsequent random operations in the library,
        including fuzzy number generation, random sampling, and stochastic
        operations. Once set, the sequence of random numbers becomes deterministic
        and reproducible.

        For maximum reproducibility across different platforms and NumPy versions,
        consider using `numpy.random.SeedSequence` instead of raw integers.

        Examples
        --------
        Different seeding approaches:

        .. code-block:: python

            # Simple integer seed
            state.set_seed(42)

            # Using SeedSequence for better entropy
            seed_seq = np.random.SeedSequence(12345)
            state.set_seed(seed_seq)

            # Using specific bit generator
            bit_gen = np.random.PCG64(67890)
            state.set_seed(bit_gen)

            # Reset to non-deterministic behavior
            state.set_seed(None)
        """
        with self._lock:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

    def get_generator(self) -> np.random.Generator:
        """
        Get the global random number generator instance.

        Returns the current Generator instance used for all random operations
        in the library. This generator can be used directly for custom random
        sampling operations that need to be consistent with the global random
        state.

        Returns
        -------
        numpy.random.Generator
            The current global random number generator instance.

        Notes
        -----
        The returned generator is the actual instance used internally,
        not a copy. This means that using it directly will advance the
        internal random state and affect subsequent library operations.

        For operations that should not interfere with the global state,
        consider using :meth:`spawn_generator` instead.

        Thread safety is maintained through internal locking, so the
        generator can be safely accessed from multiple threads.

        Examples
        --------
        Direct generator usage:

        .. code-block:: python

            # Get global generator
            rng = state.get_generator()

            # Use for custom sampling
            custom_values = rng.uniform(0, 1, size=1000)
            beta_values = rng.beta(2, 5, size=500)

            # This advances the global state, affecting subsequent
            # library operations
            fuzz_num = fr.rand('qrofn')  # Uses advanced state
        """
        with self._lock:
            return self._rng

    def get_seed(self) -> Union[int, np.random.SeedSequence, np.random.BitGenerator]:
        """
        Get the current global random seed.

        Returns the seed value that was used to initialize the current
        random number generator. This can be useful for debugging,
        logging, or re-creating the current random state.

        Returns
        -------
        int, numpy.random.SeedSequence, numpy.random.BitGenerator, or None
            The seed value used to initialize the current generator.
            Returns None if the generator was initialized with system entropy.

        Notes
        -----
        The returned value is the original seed passed to :meth:`set_seed`,
        not the current internal state of the generator. The generator's
        internal state changes with each random number drawn.

        To fully reproduce a sequence, you need both the original seed and
        knowledge of how many random numbers have been drawn since initialization.

        Examples
        --------
        .. code-block:: python

            # Set a seed and retrieve it
            state.set_seed(42)
            current_seed = state.get_seed()
            print(f"Current seed: {current_seed}")  # Output: Current seed: 42

            # Use for logging or debugging
            if current_seed is not None:
                print(f"Random state is deterministic with seed {current_seed}")
            else:
                print("Random state uses system entropy")
        """
        with self._lock:
            return self._seed

    def spawn_generator(self) -> np.random.Generator:
        """
        Create an independent random number generator.

        Spawns a new Generator instance that is statistically independent
        of the global generator. This is useful for parallel operations,
        isolated random processes, or when you need randomness that doesn't
        interfere with the global random state.

        Returns
        -------
        numpy.random.Generator
            A new, statistically independent random number generator.

        Notes
        -----
        The spawned generator is completely independent of the parent:
        - Drawing numbers from it doesn't affect the global state
        - It has its own internal state and sequence
        - It's suitable for parallel processing without interference

        Spawning uses NumPy's recommended approach for creating independent
        generators, ensuring proper statistical independence and avoiding
        correlation issues that can occur with naive approaches.

        This method is particularly useful for:
        - Parallel fuzzy number generation
        - Isolated experiments within larger computations
        - Monte Carlo simulations with independent streams

        Examples
        --------
        Independent random generation:

        .. code-block:: python

            # Set global state
            state.set_seed(42)

            # Spawn independent generator
            independent_rng = state.spawn_generator()

            # Use independent generator
            independent_values = independent_rng.uniform(0, 1, size=100)

            # Global state is unaffected
            global_values = state.get_generator().uniform(0, 1, size=100)

            # These sequences are statistically independent
            print("Independent and global sequences are uncorrelated")

        Parallel processing example:

        .. code-block:: python

            import concurrent.futures

            def generate_random_array(seed_offset):
                # Each worker gets independent generator
                worker_rng = state.spawn_generator()
                return worker_rng.normal(0, 1, size=1000)

            # Generate multiple independent arrays in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(generate_random_array, i)
                          for i in range(10)]
                results = [f.result() for f in futures]
        """
        with self._lock:
            # Use the generator's spawn method, which is the recommended way
            # to create statistically independent child generators.
            return self._rng.spawn(1)[0]


# Global instance
_global_random_state = GlobalRandomState()


# Public interface functions
def set_seed(seed: Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]] = None):
    """
    Set the global random seed for AxisFuzzy.

    This function provides the primary interface for controlling randomness
    throughout the AxisFuzzy library. Setting a seed ensures that all
    random operations (fuzzy number generation, random sampling, etc.)
    produce reproducible results.

    Parameters
    ----------
    seed : int, numpy.random.SeedSequence, numpy.random.BitGenerator, optional
        The seed to use for random number generation:

        - `int`: Simple integer seed (recommended for basic use)
        - `numpy.random.SeedSequence`: Advanced seed sequence for better control
        - `numpy.random.BitGenerator`: Pre-configured bit generator
        - `None`: Use system entropy for non-reproducible behavior

    See Also
    --------
    get_seed : Retrieve the current global seed
    get_rng : Get the global random number generator
    numpy.random.default_rng : NumPy's default generator function

    Examples
    --------
    Basic reproducible generation:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Set seed for reproducibility
        fr.set_seed(42)

        # Generate fuzzy numbers - results are reproducible
        num1 = fr.rand('qrofn', q=2)
        arr1 = fr.rand('qrofn', shape=(10,), q=3)

        # Reset seed to get identical results
        fr.set_seed(42)
        num2 = fr.rand('qrofn', q=2)  # Identical to num1
        arr2 = fr.rand('qrofn', shape=(10,), q=3)  # Identical to arr1

        print(num1 == num2)  # True
        print((arr1 == arr2).all())  # True

    Scientific workflow with reproducibility:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Set seed at start of experiment
        EXPERIMENT_SEED = 12345
        fr.set_seed(EXPERIMENT_SEED)

        # Generate datasets
        training_data = fr.rand('qrofn', shape=(1000, 10), q=2)
        test_data = fr.rand('qrofn', shape=(200, 10), q=2)

        # Results are now reproducible across runs
        print(f"Experiment run with seed {EXPERIMENT_SEED}")

    Different seed types:

    .. code-block:: python

        import numpy as np
        import axisfuzzy.random as fr

        # Simple integer seed
        fr.set_seed(123)

        # Advanced seed sequence
        seed_seq = np.random.SeedSequence(123, spawn_key=(1, 2, 3))
        fr.set_seed(seed_seq)

        # Custom bit generator
        bit_gen = np.random.PCG64(456)
        fr.set_seed(bit_gen)
    """
    _global_random_state.set_seed(seed)


def get_rng() -> np.random.Generator:
    """
    Get the global random number generator for AxisFuzzy.

    Returns the Generator instance used throughout the library for all
    random operations. This can be used for custom random sampling that
    needs to be consistent with the library's random state.

    Returns
    -------
    numpy.random.Generator
        The global random number generator instance.

    Notes
    -----
    The returned generator is the actual instance used internally by the
    library. Using it directly will advance the global random state and
    affect subsequent library operations.

    For independent random generation that doesn't interfere with the
    global state, use :func:`spawn_rng` instead.

    See Also
    --------
    spawn_rng : Create independent random generator
    set_seed : Set the global random seed
    numpy.random.Generator : NumPy generator documentation

    Examples
    --------
    Custom random sampling:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Set seed for reproducibility
        fr.set_seed(42)

        # Get global generator
        rng = fr.get_rng()

        # Use for custom sampling consistent with global state
        membership_values = rng.beta(2, 5, size=100)
        noise = rng.normal(0, 0.1, size=100)

        # Subsequent library operations use advanced state
        fuzz_array = fr.rand('qrofn', shape=(50,), q=2)

    Integration with NumPy operations:

    .. code-block:: python

        import axisfuzzy.random as fr
        import numpy as np

        # Get generator for NumPy compatibility
        rng = fr.get_rng()

        # Use NumPy methods with consistent state
        indices = rng.choice(1000, size=100, replace=False)
        weights = rng.dirichlet([1, 1, 1, 1])

        # Mix with AxisFuzzy operations
        fuzzy_samples = fr.rand('qrofn', shape=(len(indices),), q=3)
    """
    return _global_random_state.get_generator()


def spawn_rng() -> np.random.Generator:
    """
    Create an independent random number generator.

    Spawns a new Generator instance that is statistically independent
    of the global generator. This is ideal for parallel operations,
    isolated computations, or when you need random numbers without
    affecting the global random state.

    Returns
    -------
    numpy.random.Generator
        A new, statistically independent random number generator.

    Notes
    -----
    The spawned generator is completely independent:
    - Using it doesn't affect the global random state
    - It maintains its own internal state and sequence
    - It's safe for parallel processing without interference

    This function uses NumPy's recommended spawning mechanism to ensure
    proper statistical independence and avoid correlation issues.

    See Also
    --------
    get_rng : Get the global random generator
    set_seed : Set the global random seed
    numpy.random.Generator.spawn : NumPy's generator spawning method

    Examples
    --------
    Independent random operations:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Set global seed
        fr.set_seed(42)

        # Create independent generator
        independent_rng = fr.spawn_rng()

        # Use independent generator without affecting global state
        independent_data = independent_rng.uniform(0, 1, size=1000)

        # Global state remains unaffected
        global_fuzz = fr.rand('qrofn', q=2)  # Predictable based on seed 42

    Parallel processing with independent streams:

    .. code-block:: python

        import axisfuzzy.random as fr
        from concurrent.futures import ThreadPoolExecutor

        def worker_function(worker_id):
            # Each worker gets its own independent generator
            worker_rng = fr.spawn_rng()

            # Generate data independently
            data = worker_rng.normal(0, 1, size=100)
            fuzzy_data = []

            # Use worker's generator for fuzzy generation
            for _ in range(10):
                # This would typically use a custom generator
                # For demonstration, we'll use direct values
                md = worker_rng.uniform(0, 1)
                nmd = worker_rng.uniform(0, 1-md**2)**(1/2)  # q=2 constraint
                fuzzy_data.append((md, nmd))

            return worker_id, data, fuzzy_data

        # Run parallel workers with independent randomness
        fr.set_seed(123)  # Global seed for reproducibility
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker_function, range(4)))

        # Each worker produced independent results
        for worker_id, data, fuzzy_data in results:
            print(f"Worker {worker_id}: generated {len(data)} samples")

    Monte Carlo simulation with independent streams:

    .. code-block:: python

        import axisfuzzy.random as fr

        def monte_carlo_trial():
            # Independent generator for each trial
            trial_rng = fr.spawn_rng()

            # Generate random fuzzy scenario
            scenario_data = trial_rng.uniform(0, 1, size=100)

            # Simulate some process
            result = scenario_data.mean()
            return result

        # Run multiple independent trials
        fr.set_seed(456)
        trials = [monte_carlo_trial() for _ in range(1000)]

        # Analyze results
        import numpy as np
        print(f"Monte Carlo estimate: {np.mean(trials):.4f} ± {np.std(trials):.4f}")
    """
    return _global_random_state.spawn_generator()


def get_seed() -> Optional[Union[int, np.random.SeedSequence, np.random.BitGenerator]]:
    """
    Get the current global random seed.

    Returns the seed value that was used to initialize the global
    random number generator. This is useful for debugging, logging,
    experiment tracking, or reproducing specific random states.

    Returns
    -------
    int, numpy.random.SeedSequence, numpy.random.BitGenerator, or None
        The seed value used to initialize the current generator.
        Returns None if the generator was initialized with system entropy.

    Notes
    -----
    This function returns the original seed passed to :func:`set_seed`,
    not the current internal state of the generator. The generator's
    internal state evolves with each random number generated.

    To fully reproduce a random sequence, you need both the original
    seed and knowledge of the generation history (how many random
    numbers have been drawn).

    See Also
    --------
    set_seed : Set the global random seed
    get_rng : Get the global random generator

    Examples
    --------
    Basic seed retrieval:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Set and retrieve seed
        fr.set_seed(42)
        current_seed = fr.get_seed()
        print(f"Current seed: {current_seed}")  # Output: Current seed: 42

    Experiment logging:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Set up experiment
        experiment_seed = 12345
        fr.set_seed(experiment_seed)

        # Log experiment parameters
        print(f"Starting experiment with seed: {fr.get_seed()}")

        # Run experiment
        results = fr.rand('qrofn', shape=(100,), q=2)

        # Log for reproducibility
        print(f"Experiment completed with seed {fr.get_seed()}")
        print(f"To reproduce: fr.set_seed({fr.get_seed()})")

    Conditional behavior based on seed:

    .. code-block:: python

        import axisfuzzy.random as fr

        # Check if random behavior is deterministic
        if fr.get_seed() is not None:
            print("Random generation is reproducible")
            print(f"Seed: {fr.get_seed()}")
        else:
            print("Random generation uses system entropy")
            print("Results will not be reproducible")

        # Generate data
        data = fr.rand('qrofn', shape=(10,), q=3)
    """
    return _global_random_state.get_seed()

