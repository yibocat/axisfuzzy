#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 10:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
This module provides a high-performance factory function `fuzzyset` for
creating Fuzzarray instances. It serves as a versatile and efficient entry
point for Fuzzarray construction, accommodating various input types and
optimizing for performance where possible.
"""

from typing import Optional, Union, Tuple, List, Any
import numpy as np

from .fuzzarray import Fuzzarray
from .fuzznums import Fuzznum
from .backend import FuzzarrayBackend
from .registry import get_registry_fuzztype
from ..config import get_config


def fuzzynum(values: Optional[tuple] = None,
             mtype: Optional[str] = None,
             q: Optional[int] = None,
             **kwargs: Any) -> Fuzznum:
    """
    Factory function to create a Fuzznum instance.

    Parameters
    ----------
    values : tuple, optional
        Membership degree value tuple of fuzzy numbers, adapted to different mtypes.
        If none, convert to settings based on kwargs.
    mtype : str, optional
        The type of fuzzy number strategy to use. If omitted, uses the default from config.
    q : int, optional
        The q-rung value for the fuzzy number. If omitted, uses the default from config.
    kwargs : dict
        Additional parameters specific to the chosen fuzzy number strategy.

    Returns
    -------
    Fuzznum
        An instance of Fuzznum configured with the specified strategy and parameters.

    Examples
    --------

    .. code-block:: python

        a = fuzznum((0.5,0.3), mtype='qrofn')
        print(a)    # <0.5,0.3>

    .. code-block:: python

        a = fuzznum(mtype='qrofn', md=0.7, nmd=0.2)
        print(a)    # <0.7,0.2>
    """
    mtype = mtype or get_config().DEFAULT_MTYPE
    q = q or get_config().DEFAULT_Q

    if values is not None:
        from .registry import get_registry_fuzztype
        registry = get_registry_fuzztype()
        if mtype not in registry.strategies:
            raise ValueError(f"Unsupported mtype '{mtype}'. Available mtypes: {', '.join(registry.strategies.keys())}")
        strategy_cls = registry.strategies[mtype]
        attr_names = [a for a in strategy_cls().get_declared_attributes() if a != 'q' and a != 'mtype']
        attr_names = attr_names[:len(values)]
        tuple_kwargs = dict(zip(attr_names, values))
        tuple_kwargs.update(kwargs)
        instance = Fuzznum(mtype, q)
        return instance.create(**tuple_kwargs)
    else:
        instance = Fuzznum(mtype, q)
        if kwargs:
            return instance.create(**kwargs)
        return instance


def fuzzyset(
    data: Optional[Union[np.ndarray, List, tuple, Fuzznum, Fuzzarray]] = None,
    backend: Optional[FuzzarrayBackend] = None,
    mtype: Optional[str] = None,
    q: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> Fuzzarray:
    """
    Factory function to create a Fuzzarray instance efficiently.

    This function provides a flexible and optimized way to construct a Fuzzarray.
    It supports creation from raw NumPy arrays, existing Fuzznum/Fuzzarray
    objects, or a pre-configured backend.

    Parameters
    ----------
    data : array-like, Fuzznum, Fuzzarray, optional
        Input data for the Fuzzarray. Can be one of the following:
        - A NumPy array or nested list representing the components of fuzzy
          numbers (e.g., `[[m1, m2], [n1, n2]]` for q-ROFNs). This is the
          high-performance path.
        - A list, tuple, or array of Fuzznum objects.
        - A single Fuzznum object to be broadcast.
        - An existing Fuzzarray object (a copy will be made).
    backend : FuzzarrayBackend, optional
        A pre-constructed backend instance. If provided, `data` is ignored.
    mtype : str, optional
        The membership type (e.g., 'qrofn', 'qrohfn'). If not provided, it will
        be inferred from the data or default to the configured value.
    q : int, optional
        The q-rung for q-rung-based fuzzy types. Inferred or defaulted if not
        provided.
    shape : tuple of int, optional
        The desired shape of the array.
    **kwargs :
        Additional backend-specific keyword arguments.

    Returns
    -------
    Fuzzarray
        A new Fuzzarray instance.

    Raises
    ------
    ValueError
        If input data is inconsistent or required parameters are missing.
    TypeError
        If the data type is not supported.
    """
    # Path 1: Direct backend assignment (fastest)
    if backend is not None:
        return Fuzzarray(backend=backend)

    # Resolve mtype and q
    mtype, q = _resolve_mtype_and_q(data, mtype, q)
    
    # Get backend class from registry
    registry = get_registry_fuzztype()
    backend_cls = registry.get_backend(mtype)
    if backend_cls is None:
        raise ValueError(f"No backend registered for mtype '{mtype}'")

    # Path 2: High-performance creation from raw arrays
    if isinstance(data, (np.ndarray, list, tuple)) and not isinstance(data, Fuzzarray):
        # Heuristic to check if it's raw component data vs. a list of Fuzznums
        is_fuzznum_list = False
        if isinstance(data, (list, tuple)) and data:
            if isinstance(data[0], Fuzznum):
                is_fuzznum_list = True
        elif isinstance(data, np.ndarray) and data.dtype == object and data.size > 0:
            if isinstance(data.flat[0], Fuzznum):
                is_fuzznum_list = True

        if not is_fuzznum_list:
            try:
                return _create_from_raw_arrays(backend_cls, data, q, shape, **kwargs)
            except (ValueError, TypeError):
                # Fallback to standard Fuzzarray constructor if raw creation fails
                pass

    # Path 3: Standard creation from Fuzznum, Fuzzarray, or list of Fuzznums
    return Fuzzarray(data=data, mtype=mtype, q=q, shape=shape, **kwargs)


def _create_from_raw_arrays(backend_cls, data, q, shape, **kwargs) -> Fuzzarray:
    """
    Helper to create a Fuzzarray from raw component arrays.
    """
    data_arr = np.asarray(data)

    # Use backend metadata for validation
    be_instance_for_meta = backend_cls(shape=(), q=q)
    expected_comps = be_instance_for_meta.cmpnum
    expected_dtype = be_instance_for_meta.dtype

    if data_arr.ndim < 2:
        raise ValueError(
            f"Raw data must have at least 2 dimensions (components, ...), "
            f"but got {data_arr.ndim}"
        )

    if data_arr.shape[0] != expected_comps:
        raise ValueError(
            f"Expected {expected_comps} component arrays for mtype '{be_instance_for_meta.mtype}', "
            f"but got {data_arr.shape[0]}"
        )

    if shape is None:
        shape = data_arr.shape[1:]
    elif np.prod(shape) != np.prod(data_arr.shape[1:]):
        raise ValueError(
            f"Provided shape {shape} is not compatible with data shape {data_arr.shape[1:]}"
        )

    # Unpack component arrays and ensure correct dtype
    components = [
        np.asarray(data_arr[i], dtype=expected_dtype).reshape(shape)
        for i in range(expected_comps)
    ]

    # Use the backend's from_arrays factory
    new_backend = backend_cls.from_arrays(*components, q=q, **kwargs)
    return Fuzzarray(backend=new_backend)


def _resolve_mtype_and_q(data, mtype, q) -> Tuple[str, int]:
    """
    Helper to determine the final mtype and q.
    """
    # mtype resolution
    if mtype is None:
        if isinstance(data, (Fuzznum, Fuzzarray)):
            mtype = data.mtype
        elif isinstance(data, (list, tuple, np.ndarray)) and np.asarray(data).size > 0:
            if isinstance(np.asarray(data).flat[0], Fuzznum):
                mtype = np.asarray(data).flat[0].mtype
            else:
                mtype = get_config().DEFAULT_MTYPE
        else:
            mtype = get_config().DEFAULT_MTYPE
    
    # q resolution
    if q is None:
        if isinstance(data, (Fuzznum, Fuzzarray)):
            q = data.q
        elif isinstance(data, (list, tuple, np.ndarray)) and np.asarray(data).size > 0:
            if isinstance(np.asarray(data).flat[0], Fuzznum):
                q = np.asarray(data).flat[0].q
            else:
                q = get_config().DEFAULT_Q
        else:
            q = get_config().DEFAULT_Q
            
    return mtype.lower(), int(q)
