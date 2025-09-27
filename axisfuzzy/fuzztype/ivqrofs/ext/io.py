#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
IVQROFN I/O Extension Methods.

This module implements high-performance I/O operations for IVQROFN arrays,
including CSV, JSON, and NumPy binary format support with efficient
serialization and deserialization.
"""

import csv
import json
from typing import Any

import numpy as np

from ....core import Fuzzarray, get_registry_fuzztype


def _ivqrofn_to_csv(arr: Fuzzarray, path: str, **kwargs) -> None:
    """
    Export IVQROFN Fuzzarray to CSV file with high-performance backend array access.
    
    Parameters:
        arr: IVQROFN Fuzzarray to export
        path: Output CSV file path
        **kwargs: Additional CSV writer parameters
        
    Raises:
        TypeError: If arr is not an IVQROFN Fuzzarray
    """
    if arr.mtype != 'ivqrofn':
        raise TypeError(f"to_csv for mtype 'ivqrofn' cannot be called on Fuzzarray with mtype '{arr.mtype}'")

    # Get component arrays directly from backend for high performance
    mds, nmds = arr.backend.get_component_arrays()

    # Create efficient string representation for interval format
    # Format: <[md_lower,md_upper],[nmd_lower,nmd_upper]>
    str_data = np.char.add(
        np.char.add(
            np.char.add('<[', mds[..., 0].astype(str)),
            np.char.add(',', np.char.add(mds[..., 1].astype(str), '],['))
        ),
        np.char.add(
            np.char.add(nmds[..., 0].astype(str), ','),
            np.char.add(nmds[..., 1].astype(str), ']>')
        )
    )

    # Write to CSV file with efficient vectorized operations
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, **kwargs)
        if str_data.ndim == 1:
            writer.writerow(str_data)
        else:
            writer.writerows(str_data)


def _ivqrofn_from_csv(path: str, mtype: str = 'ivqrofn', q: int = 1, **kwargs) -> Fuzzarray:
    """
    Import IVQROFN Fuzzarray from CSV file with high-performance parsing.
    
    Parameters:
        path: Input CSV file path
        mtype: Fuzztype (should be 'ivqrofn')
        q: Q-rung parameter for the imported IVQROFNs
        **kwargs: Additional CSV reader parameters
        
    Returns:
        Fuzzarray: IVQROFN Fuzzarray loaded from CSV
    """
    # Filter out non-CSV parameters
    csv_kwargs = {k: v for k, v in kwargs.items() 
                  if k not in ('mtype', 'q')}
    
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, **csv_kwargs)
        str_data_list = list(reader)

    if not str_data_list:
        # Return empty array if no data
        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        backend = backend_cls(shape=(0,), q=q)
        return Fuzzarray(backend=backend)

    str_data = np.array(str_data_list, dtype=str)
    shape = str_data.shape

    # Parse IVQROFN strings efficiently using vectorized operations
    # Expected format: <[md_lower,md_upper],[nmd_lower,nmd_upper]>
    # Remove < and > characters
    clean_data = np.char.strip(np.char.strip(str_data, '<'), '>')
    
    # Split by ],[ to separate membership and non-membership intervals
    md_nmd_parts = np.char.split(clean_data, '],[', 1)
    
    # Extract and process membership intervals
    md_parts = np.array([parts[0] for parts in md_nmd_parts.flat]).reshape(shape)
    md_clean = np.char.strip(md_parts, '[')
    md_values = np.char.split(md_clean, ',', 1)
    md_lower = np.array([parts[0] for parts in md_values.flat]).reshape(shape).astype(float)
    md_upper = np.array([parts[1] for parts in md_values.flat]).reshape(shape).astype(float)
    
    # Extract and process non-membership intervals
    nmd_parts = np.array([parts[1] for parts in md_nmd_parts.flat]).reshape(shape)
    nmd_clean = np.char.strip(nmd_parts, ']')
    nmd_values = np.char.split(nmd_clean, ',', 1)
    nmd_lower = np.array([parts[0] for parts in nmd_values.flat]).reshape(shape).astype(float)
    nmd_upper = np.array([parts[1] for parts in nmd_values.flat]).reshape(shape).astype(float)

    # Construct interval arrays
    mds = np.stack([md_lower, md_upper], axis=-1)
    nmds = np.stack([nmd_lower, nmd_upper], axis=-1)

    # Create backend directly with parsed arrays
    backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
    new_backend = backend_cls.from_arrays(mds, nmds, q=q)
    return Fuzzarray(backend=new_backend)


def _ivqrofn_to_json(arr: Fuzzarray, path: str, **kwargs) -> None:
    """
    Export IVQROFN Fuzzarray to JSON file with structured data format.
    
    Parameters:
        arr: IVQROFN Fuzzarray to export
        path: Output JSON file path
        **kwargs: Additional JSON dump parameters
        
    Raises:
        TypeError: If arr is not an IVQROFN Fuzzarray
    """
    if arr.mtype != 'ivqrofn':
        raise TypeError(f"Expected IVQROFN Fuzzarray, got mtype '{arr.mtype}'")

    # Get component arrays directly from backend
    mds, nmds = arr.backend.get_component_arrays()

    # Create structured JSON data
    data = {
        'mtype': arr.mtype,
        'q': arr.q,
        'shape': list(arr.shape),
        'md_data': mds.tolist(),  # Convert intervals to nested lists
        'nmd_data': nmds.tolist()
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)


def _ivqrofn_from_json(path: str, mtype: str = 'ivqrofn', q: int = 1, **kwargs) -> Fuzzarray:
    """
    Import IVQROFN Fuzzarray from JSON file with structured data parsing.
    
    Parameters:
        path: Input JSON file path
        mtype: Fuzztype (should be 'ivqrofn')
        q: Q parameter for IVQROFN
        **kwargs: Additional JSON load parameters
        
    Returns:
        Fuzzarray: IVQROFN Fuzzarray loaded from JSON
    """
    # Filter out non-JSON parameters
    json_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ('mtype', 'q')}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f, **json_kwargs)

    q = data.get('q', q)  # Use file q or fallback to parameter q
    shape = tuple(data['shape'])
    mds = np.array(data['md_data'])
    nmds = np.array(data['nmd_data'])

    # Create backend directly with loaded arrays
    backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
    new_backend = backend_cls.from_arrays(mds, nmds, q=q)
    return Fuzzarray(backend=new_backend)


def _ivqrofn_to_npy(arr: Fuzzarray, path: str, **kwargs) -> None:
    """
    Export IVQROFN Fuzzarray to NumPy binary format using structured arrays.
    
    Parameters:
        arr: IVQROFN Fuzzarray to export
        path: Output .npy file path
        **kwargs: Additional numpy.save parameters
        
    Raises:
        TypeError: If arr is not an IVQROFN Fuzzarray
    """
    if arr.mtype != 'ivqrofn':
        raise TypeError(f"Expected IVQROFN Fuzzarray, got mtype '{arr.mtype}'")

    # Get component arrays directly from backend
    mds, nmds = arr.backend.get_component_arrays()

    # Create structured array for efficient binary storage
    # Each element contains both intervals and metadata
    dtype = [
        ('md_lower', 'f8'), ('md_upper', 'f8'),
        ('nmd_lower', 'f8'), ('nmd_upper', 'f8'),
        ('q', 'i4')
    ]
    
    structured_data = np.empty(arr.shape, dtype=dtype)
    structured_data['md_lower'] = mds[..., 0]
    structured_data['md_upper'] = mds[..., 1]
    structured_data['nmd_lower'] = nmds[..., 0]
    structured_data['nmd_upper'] = nmds[..., 1]
    structured_data['q'] = arr.q

    np.save(path, structured_data, **kwargs)


def _ivqrofn_from_npy(path: str, mtype: str = 'ivqrofn', q: int = 1, **kwargs) -> Fuzzarray:
    """
    Import IVQROFN Fuzzarray from NumPy binary format with structured data.
    
    Parameters:
        path: Input .npy file path
        mtype: Fuzztype (should be 'ivqrofn')
        q: Q parameter for IVQROFN (fallback if not in file)
        **kwargs: Additional numpy.load parameters
        
    Returns:
        Fuzzarray: IVQROFN Fuzzarray loaded from NumPy binary file
    """
    # Filter out non-numpy parameters
    numpy_kwargs = {k: v for k, v in kwargs.items() 
                    if k not in ('mtype', 'q')}
    
    structured_data = np.load(path, **numpy_kwargs)

    # Reconstruct interval arrays from structured data
    mds = np.stack([
        structured_data['md_lower'],
        structured_data['md_upper']
    ], axis=-1)
    
    nmds = np.stack([
        structured_data['nmd_lower'],
        structured_data['nmd_upper']
    ], axis=-1)
    
    # Extract q parameter (assume uniform q across all elements)
    q = int(structured_data['q'].flat[0])

    # Create backend directly with reconstructed arrays
    backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
    new_backend = backend_cls.from_arrays(mds, nmds, q=q)
    return Fuzzarray(backend=new_backend)