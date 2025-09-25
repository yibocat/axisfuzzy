#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) I/O Extension Methods.

This module implements high-performance I/O operations for classical fuzzy sets (FS),
providing efficient serialization and deserialization across multiple formats.
All methods leverage backend component arrays for optimal performance.

Supported Formats:
    - CSV: Human-readable comma-separated values
    - JSON: Structured JSON with metadata
    - NPY: NumPy binary format for high-performance storage

Mathematical Foundation:
    FS I/O operations preserve the single membership degree component,
    ensuring data integrity and mathematical consistency during
    serialization/deserialization cycles.
"""

import csv
import json

import numpy as np

from ....core import Fuzzarray, get_registry_fuzztype


def _fs_to_csv(arr: Fuzzarray, path: str, **kwargs) -> None:
    """
    Export FS Fuzzarray to CSV format using high-performance backend operations.
    
    The CSV format represents each FS value as '<md>' where md is the
    membership degree. This provides a human-readable representation
    while maintaining precision.
    
    Parameters:
        arr (Fuzzarray): FS Fuzzarray to export
        path (str): Output CSV file path
        **kwargs: Additional arguments passed to csv.writer
        
    Raises:
        TypeError: If arr is not an FS Fuzzarray
        
    Example Output:
        For a 2x2 FS array:
        <0.8>,<0.6>
        <0.9>,<0.3>
    """
    if arr.mtype != 'fs':
        raise TypeError(f"to_csv for mtype 'fs' cannot be called on Fuzzarray with mtype '{arr.mtype}'")

    # Get membership degrees directly from backend for efficiency
    mds, = arr.backend.get_component_arrays()

    # Create string representation efficiently using numpy char operations
    str_data = np.char.add(
        np.char.add('<', mds.astype(str)),
        '>'
    )

    # Write directly to CSV without external dependencies
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, **kwargs)
        if str_data.ndim == 1:
            writer.writerow(str_data)
        else:
            writer.writerows(str_data)


def _fs_from_csv(path: str, **kwargs) -> Fuzzarray:
    """
    Import FS Fuzzarray from CSV format using high-performance parsing.
    
    Reads CSV files containing FS values in '<md>' format and creates
    a Fuzzarray with proper backend initialization for optimal performance.
    
    Parameters:
        path (str): Input CSV file path
        **kwargs: Additional arguments passed to csv.reader
        
    Returns:
        Fuzzarray: FS Fuzzarray loaded from CSV
        
    Raises:
        ValueError: If CSV format is invalid or contains invalid FS values
        FileNotFoundError: If the CSV file doesn't exist
    """
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, **kwargs)
        str_data_list = list(reader)

    if not str_data_list:
        # Return empty FS array if file is empty
        backend_cls = get_registry_fuzztype().get_backend('fs')
        backend = backend_cls(shape=(0,))
        return Fuzzarray(backend=backend)

    str_data = np.array(str_data_list, dtype=str)

    # Parse strings efficiently using vectorized operations
    # Remove < and > characters
    clean_data = np.char.strip(np.char.strip(str_data, '<'), '>')

    # Convert to float arrays
    try:
        mds = clean_data.astype(float)
    except ValueError as e:
        raise ValueError(f"Invalid FS format in CSV: {e}")

    # Validate FS constraints: membership degrees must be in [0, 1]
    if np.any((mds < 0.0) | (mds > 1.0)):
        raise ValueError("FS membership degrees must be in [0, 1]")

    # Create backend directly with arrays
    backend_cls = get_registry_fuzztype().get_backend('fs')
    new_backend = backend_cls.from_arrays(mds=mds)
    return Fuzzarray(backend=new_backend)


def _fs_to_json(arr: Fuzzarray, path: str, **kwargs):
    """
    Export FS Fuzzarray to JSON format with metadata preservation.
    
    The JSON format includes complete metadata (mtype, shape) along with
    the membership degree data, enabling perfect reconstruction of the
    original FS Fuzzarray.
    
    Parameters:
        arr (Fuzzarray): FS Fuzzarray to export
        path (str): Output JSON file path
        **kwargs: Additional arguments passed to json.dump
        
    Raises:
        TypeError: If arr is not an FS Fuzzarray
        
    Example Output:
        {
          "mtype": "fs",
          "shape": [2, 2],
          "md_data": [[0.8, 0.6], [0.9, 0.3]]
        }
    """
    if arr.mtype != 'fs':
        raise TypeError(f"Expected FS Fuzzarray, got mtype '{arr.mtype}'")

    # Get membership degrees from backend
    mds, = arr.backend.get_component_arrays()

    data = {
        'mtype': arr.mtype,
        'shape': list(arr.shape),
        'md_data': mds.tolist()
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)


def _fs_from_json(path: str, **kwargs) -> Fuzzarray:
    """
    Import FS Fuzzarray from JSON format with metadata validation.
    
    Reads JSON files containing FS data with metadata and creates
    a Fuzzarray with proper type and shape validation.
    
    Parameters:
        path (str): Input JSON file path
        **kwargs: Additional arguments passed to json.load
        
    Returns:
        Fuzzarray: FS Fuzzarray loaded from JSON
        
    Raises:
        ValueError: If JSON format is invalid or data violates FS constraints
        KeyError: If required metadata fields are missing
        FileNotFoundError: If the JSON file doesn't exist
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f, **kwargs)

    # Validate required fields
    required_fields = ['mtype', 'shape', 'md_data']
    for field in required_fields:
        if field not in data:
            raise KeyError(f"Missing required field '{field}' in JSON data")

    # Validate mtype
    if data['mtype'] != 'fs':
        raise ValueError(f"Expected mtype 'fs', got '{data['mtype']}'")

    # Extract data
    shape = tuple(data['shape'])
    mds = np.array(data['md_data'])

    # Validate data shape consistency
    if mds.shape != shape:
        raise ValueError(f"Shape mismatch: expected {shape}, got {mds.shape}")

    # Validate FS constraints
    if np.any((mds < 0.0) | (mds > 1.0)):
        raise ValueError("FS membership degrees must be in [0, 1]")

    # Create backend directly with arrays
    backend_cls = get_registry_fuzztype().get_backend('fs')
    new_backend = backend_cls.from_arrays(mds=mds)
    return Fuzzarray(backend=new_backend)


def _fs_to_npy(arr: Fuzzarray, path: str, **kwargs):
    """
    Export FS Fuzzarray to NumPy binary format for high-performance storage.
    
    The NPY format provides the most efficient storage for FS arrays,
    preserving full precision and enabling fast I/O operations.
    Uses structured arrays to maintain data organization.
    
    Parameters:
        arr (Fuzzarray): FS Fuzzarray to export
        path (str): Output NPY file path
        **kwargs: Additional arguments passed to np.save
        
    Raises:
        TypeError: If arr is not an FS Fuzzarray
    """
    if arr.mtype != 'fs':
        raise TypeError(f"Expected FS Fuzzarray, got mtype '{arr.mtype}'")

    # Get membership degrees from backend
    mds, = arr.backend.get_component_arrays()

    # Create structured array with metadata
    dtype = [('md', 'f8'), ('mtype', 'U4')]  # f8 = float64, U4 = unicode string length 4
    structured_data = np.empty(arr.shape, dtype=dtype)
    structured_data['md'] = mds
    structured_data['mtype'] = 'fs'

    np.save(path, structured_data, **kwargs)


def _fs_from_npy(path: str, **kwargs) -> Fuzzarray:
    """
    Import FS Fuzzarray from NumPy binary format with validation.
    
    Reads NPY files containing FS structured array data and creates
    a Fuzzarray with proper validation and backend initialization.
    
    Parameters:
        path (str): Input NPY file path
        **kwargs: Additional arguments passed to np.load
        
    Returns:
        Fuzzarray: FS Fuzzarray loaded from NPY
        
    Raises:
        ValueError: If NPY data violates FS constraints or format
        KeyError: If required fields are missing from structured array
        FileNotFoundError: If the NPY file doesn't exist
    """
    structured_data = np.load(path, **kwargs)

    # Validate structured array format
    if not isinstance(structured_data, np.ndarray) or structured_data.dtype.names is None:
        raise ValueError("NPY file must contain a structured array")

    required_fields = ['md', 'mtype']
    for field in required_fields:
        if field not in structured_data.dtype.names:
            raise KeyError(f"Missing required field '{field}' in NPY structured array")

    # Extract data
    mds = structured_data['md']
    mtype_values = structured_data['mtype']

    # Validate mtype consistency
    unique_mtypes = np.unique(mtype_values)
    if len(unique_mtypes) != 1 or unique_mtypes[0] != 'fs':
        raise ValueError(f"Expected uniform mtype 'fs', got {unique_mtypes}")

    # Validate FS constraints
    if np.any((mds < 0.0) | (mds > 1.0)):
        raise ValueError("FS membership degrees must be in [0, 1]")

    # Create backend directly with arrays
    backend_cls = get_registry_fuzztype().get_backend('fs')
    new_backend = backend_cls.from_arrays(mds=mds)
    return Fuzzarray(backend=new_backend)