#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 15:48
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import csv
import json

import numpy as np

from ....core import Fuzzarray, get_fuzztype_backend


def _qrofn_to_csv(arr: Fuzzarray, path: str, **kwargs) -> None:
    """High-performance CSV export using backend arrays directly."""
    if arr.mtype != 'qrofn':
        raise TypeError(f"to_csv for mtype 'qrofn' cannot be called on Fuzzarray with mtype '{arr.mtype}'")

    # Get component arrays directly from backend
    mds, nmds = arr.backend.get_component_arrays()

    # Create string representation efficiently
    str_data = np.char.add(
        np.char.add('<', mds.astype(str)),
        np.char.add(',', np.char.add(nmds.astype(str), '>'))
    )

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, **kwargs)
        if str_data.ndim == 1:
            writer.writerow(str_data)
        else:
            writer.writerows(str_data)


def _qrofn_from_csv(path: str, q: int, **kwargs) -> Fuzzarray:
    """High-performance CSV import directly to backend."""
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, **kwargs)
        str_data_list = list(reader)

    if not str_data_list:
        return Fuzzarray([], mtype='qrofn', q=q)

    str_data = np.array(str_data_list, dtype=str)
    shape = str_data.shape

    # Parse strings efficiently using vectorized operations
    # Remove < and > characters
    clean_data = np.char.strip(np.char.strip(str_data, '<'), '>')

    # Split by comma and convert to float arrays
    md_strs, nmd_strs = np.char.split(clean_data, ',', 1).T
    mds = md_strs.astype(float)
    nmds = nmd_strs.astype(float)

    # Create backend directly with arrays
    backend_cls = get_fuzztype_backend('qrofn')
    new_backend = backend_cls.from_arrays(mds, nmds, q=q)
    return Fuzzarray(backend=new_backend)


def _qrofn_to_json(arr: Fuzzarray, path: str, **kwargs):
    """Export to JSON with backend optimization."""
    if arr.mtype != 'qrofn':
        raise TypeError(f"Expected QROFN Fuzzarray, got mtype '{arr.mtype}'")

    mds, nmds = arr.backend.get_component_arrays()

    data = {
        'mtype': arr.mtype,
        'q': arr.q,
        'shape': list(arr.shape),
        'md_data': mds.tolist(),
        'nmd_data': nmds.tolist()
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)


def _qrofn_from_json(path: str, **kwargs) -> Fuzzarray:
    """Import from JSON directly to backend."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f, **kwargs)

    q = data.get('q', 1)
    shape = tuple(data['shape'])
    mds = np.array(data['md_data'])
    nmds = np.array(data['nmd_data'])

    backend_cls = get_fuzztype_backend('qrofn')
    new_backend = backend_cls.from_arrays(mds, nmds, q=q)
    return Fuzzarray(backend=new_backend)


def _qrofn_to_npy(arr: Fuzzarray, path: str, **kwargs):
    """Export to NumPy format using structured arrays."""
    if arr.mtype != 'qrofn':
        raise TypeError(f"Expected QROFN Fuzzarray, got mtype '{arr.mtype}'")

    mds, nmds = arr.backend.get_component_arrays()

    # Create structured array
    dtype = [('md', 'f8'), ('nmd', 'f8'), ('q', 'i4')]
    structured_data = np.empty(arr.shape, dtype=dtype)
    structured_data['md'] = mds
    structured_data['nmd'] = nmds
    structured_data['q'] = arr.q

    np.save(path, structured_data, **kwargs)


def _qrofn_from_npy(path: str, **kwargs) -> Fuzzarray:
    """Import from NumPy format directly to backend."""
    structured_data = np.load(path, **kwargs)

    mds = structured_data['md']
    nmds = structured_data['nmd']
    q = int(structured_data['q'].flat[0])  # Assume uniform q

    backend_cls = get_fuzztype_backend('qrofn')
    new_backend = backend_cls.from_arrays(mds, nmds, q=q)
    return Fuzzarray(backend=new_backend)


# 未来扩展,可添加更多 I/O 方法
# `to_sqlite`, `from_sqlite`, `to_hdf5`, `from_hdf5` 等等
# 这些方法可以使用类似的模式实现，直接操作后端数组以提高性能
