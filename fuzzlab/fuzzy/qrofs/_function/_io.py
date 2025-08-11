#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 16:18
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import csv
import json
import sqlite3
import pickle

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import numpy as np

from ....core import Fuzzarray, Fuzznum

from ....extension import extension, call_extension
from ....utils import experimental


@extension(name='to_csv', mtype='qrofn', target_classes=['Fuzzarray'])
def to_csv_of_qrofn(arr: Fuzzarray, path: str, **kwargs):
    """
    Exports a q-rung orthopair Fuzzarray to a CSV file.

    Each Fuzznum element is converted to its string representation (e.g., "<md,nmd>")
    and then saved. This implementation uses Python's built-in csv module
    to correctly handle delimiters within the string representation.

    Args:
        arr (Fuzzarray): The Fuzzarray to export.
        path (str): The path to the output CSV file.
        **kwargs: Additional keyword arguments passed directly to `csv.writer`.
                  Common arguments include `delimiter`.
    """
    if arr.mtype != 'qrofn':
        raise TypeError(f"to_csv for mtype 'qrofn' cannot be called on Fuzzarray with mtype '{arr.mtype}'")

    # Vectorize the str() conversion for efficiency
    to_str_vec = np.vectorize(str)
    str_data = to_str_vec(arr.data)

    # Use the csv module to write the data
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, **kwargs)
        # If str_data is 1D, writerows expects a list of lists/tuples
        if str_data.ndim == 1:
            writer.writerow(str_data)
        else:
            writer.writerows(str_data)


@extension(name='read_csv', mtype='qrofn', injection_type='top_level_function')
def from_csv_for_qrofn(path: str, q: int = 1, **kwargs) -> Fuzzarray:
    """
    Imports a q-rung orthopair Fuzzarray from a CSV file.

    Each entry in the CSV is parsed as a string (e.g., "<md,nmd>") and converted
    into a QROFN. This implementation uses Python's built-in csv module to
    correctly handle delimiters.

    Args:
        path (str): The path to the input CSV file.
        q (int): The q-rung to be used for the created fuzzy numbers.
        **kwargs: Additional keyword arguments passed directly to `csv.reader`.
                  Common arguments include `delimiter`.

    Returns:
        Fuzzarray: A new Fuzzarray instance populated with data from the CSV file.
    """
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, **kwargs)
        # Read all rows into a list of lists
        str_data_list = list(reader)

    if not str_data_list:
        return Fuzzarray([], mtype='qrofn')

    str_data = np.array(str_data_list, dtype=str)

    # Use call_extension to invoke the 'str_to_fuzznum' function for 'qrofn'
    def converter(s):
        return call_extension('str_to_fuzznum', None, fuzznum_str=s, q=q, mtype='qrofn')

    # Vectorize the conversion function
    to_fuzznum_vec = np.vectorize(converter, otypes=[object])

    fuzznum_data = to_fuzznum_vec(str_data)

    return Fuzzarray(fuzznum_data, mtype='qrofn')


@extension(name='to_json', mtype='qrofn', target_classes=['Fuzzarray'])
def to_json_qrofn(arr: Fuzzarray, path: str, **kwargs):
    """Export Fuzzarray to JSON format with metadata."""
    data = {
        'mtype': arr.mtype,
        'q': arr.q,
        'shape': arr.shape,
        'data': []
    }

    # Convert to nested list structure
    str_data = np.vectorize(str)(arr.data)
    data['data'] = str_data.tolist()

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)


@extension(name='read_json', mtype='qrofn', injection_type='top_level_function')
def from_json_qrofn(path: str, **kwargs) -> Fuzzarray:
    """Import Fuzzarray from JSON format."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f, **kwargs)

    q = data.get('q', 1)
    str_data = np.array(data['data'], dtype=str)

    def converter(s):
        return call_extension('str_to_fuzznum', None, fuzznum_str=s, q=q, mtype='qrofn')

    to_fuzznum_vec = np.vectorize(converter, otypes=[object])
    fuzznum_data = to_fuzznum_vec(str_data)

    return Fuzzarray(fuzznum_data, mtype='qrofn')


@extension(name='to_npy', mtype='qrofn', target_classes=['Fuzzarray'])
def to_npy_qrofn(arr: Fuzzarray, path: str, **kwargs):
    """Export Fuzzarray to NumPy format."""
    # 创建结构化数组存储 md, nmd 和元数据
    dtype = [('md', 'f8'), ('nmd', 'f8'), ('q', 'i4')]
    structured_data = np.empty(arr.shape, dtype=dtype)

    for idx in np.ndindex(arr.shape):
        fuzz = arr.data[idx]
        structured_data[idx] = (fuzz.md, fuzz.nmd, fuzz.q)

    np.save(path, structured_data, **kwargs)


@extension(name='read_npy', mtype='qrofn', injection_type='top_level_function')
def from_npy_qrofn(path: str, **kwargs) -> Fuzzarray:
    """Import Fuzzarray from NumPy format."""
    structured_data = np.load(path, **kwargs)

    def converter(item):
        # 支持 record array 或 tuple 两种情况
        if isinstance(item, tuple):
            md, nmd, q = item
        else:
            md = item['md']
            nmd = item['nmd']
            q = int(item['q'])
        return Fuzznum('qrofn', q=q).create(md=md, nmd=nmd)

    to_fuzznum_vec = np.vectorize(converter, otypes=[object])
    fuzznum_data = to_fuzznum_vec(structured_data)

    return Fuzzarray(fuzznum_data, mtype='qrofn')


# # TODO: 必须安装 sqlite3 扩展才能使用 SQLite 导入导出功能
# #  后期如果要添加该方法,则需要采用 extension 中的条件扩展,目前尚在测试阶段
# @experimental
# @extension(name='to_sqlite', mtype='qrofn', target_classes=['Fuzzarray'])
# def to_sqlite_qrofn(arr: Fuzzarray, db_path: str, table_name: str = 'fuzzarray', **kwargs):
#     """Export Fuzzarray to SQLite database."""
#     conn = sqlite3.connect(db_path)
#     try:
#         # 创建表结构
#         conn.execute(f'''
#             CREATE TABLE IF NOT EXISTS {table_name} (
#                 id INTEGER PRIMARY KEY,
#                 shape TEXT,
#                 mtype TEXT,
#                 q INTEGER,
#                 data BLOB
#             )
#         ''')
#
#         # 序列化数据
#         serialized_data = pickle.dumps(arr.data)
#         shape_str = ','.join(map(str, arr.shape))
#
#         conn.execute(f'''
#             INSERT INTO {table_name} (shape, mtype, q, data)
#             VALUES (?, ?, ?, ?)
#         ''', (shape_str, arr.mtype, arr.q, serialized_data))
#
#         conn.commit()
#     finally:
#         conn.close()
#
#
# # TODO: 必须安装 sqlite3 扩展才能使用 SQLite 导入导出功能
# #  后期如果要添加该方法,则需要采用 extension 中的条件扩展,目前尚在测试阶段
# @experimental
# @extension(name='read_sqlite', mtype='qrofn', injection_type='top_level_function')
# def from_sqlite_qrofn(db_path: str, table_name: str = 'fuzzarray', record_id: int = -1) -> Fuzzarray:
#     """Import Fuzzarray from SQLite database."""
#     conn = sqlite3.connect(db_path)
#     try:
#         if record_id == -1:
#             cursor = conn.execute(f'SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1')
#         else:
#             cursor = conn.execute(f'SELECT * FROM {table_name} WHERE id = ?', (record_id,))
#
#         row = cursor.fetchone()
#         if not row:
#             raise ValueError(f"No data found in table {table_name}")
#
#         _, shape_str, mtype, q, serialized_data = row
#         shape = tuple(map(int, shape_str.split(',')))
#         data = pickle.loads(serialized_data)
#
#         return Fuzzarray(data, mtype=mtype)
#     finally:
#         conn.close()
#
#
# # TODO: 必须安装 h5py 扩展才能使用 HDF5 导入导出功能
# #  后期如果要添加该方法,则需要采用 extension 中的条件扩展,目前尚在测试阶段
# @experimental
# @extension(name='to_hdf5', mtype='qrofn', target_classes=['Fuzzarray'])
# def to_hdf5_qrofn(arr: Fuzzarray, path: str, dataset_name: str = 'fuzzarray', **kwargs):
#     """Export Fuzzarray to HDF5 format."""
#     if not HDF5_AVAILABLE:
#         raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
#
#     with h5py.File(path, 'w') as f:
#         # 存储元数据
#         f.attrs['mtype'] = arr.mtype
#         f.attrs['q'] = arr.q
#         f.attrs['shape'] = arr.shape
#
#         # 分别存储 md 和 nmd 数组
#         md_data = np.vectorize(lambda x: x.md)(arr.data)
#         nmd_data = np.vectorize(lambda x: x.nmd)(arr.data)
#
#         f.create_dataset(f'{dataset_name}/md', data=md_data, **kwargs)
#         f.create_dataset(f'{dataset_name}/nmd', data=nmd_data, **kwargs)
#
#
# # TODO: 必须安装 h5py 扩展才能使用 HDF5 导入导出功能
# #  后期如果要添加该方法,则需要采用 extension 中的条件扩展,目前尚在测试阶段
# @experimental
# @extension(name='read_hdf5', mtype='qrofn', injection_type='top_level_function')
# def from_hdf5_qrofn(path: str, dataset_name: str = 'fuzzarray', **kwargs) -> Fuzzarray:
#     """Import Fuzzarray from HDF5 format."""
#     with h5py.File(path, 'r') as f:
#         mtype = f.attrs.get('mtype', 'qrofn')
#         q = int(f.attrs.get('q', 1))
#         shape = tuple(f.attrs['shape'])
#         md_data = f[f'{dataset_name}/md'][()]
#         nmd_data = f[f'{dataset_name}/nmd'][()]
#         # 组装 Fuzznum 对象
#         fuzznum_data = np.empty(shape, dtype=object)
#         for idx in np.ndindex(shape):
#             fuzznum_data[idx] = call_extension('create', None, md=md_data[idx], nmd=nmd_data[idx], q=q, mtype=mtype)
#     return Fuzzarray(fuzznum_data, mtype=mtype)
#
#
# @experimental
# @extension(name='to_api', mtype='qrofn', target_classes=['Fuzzarray'])
# def to_api_qrofn(arr: Fuzzarray, url: str, **kwargs):
#     """Upload Fuzzarray to remote API endpoint."""
#     if not REQUESTS_AVAILABLE:
#         raise ImportError("requests is required for API support. Install with: pip install requests")
#
#     # 转换为 JSON 格式
#     str_data = np.vectorize(str)(arr.data)
#     payload = {
#         'mtype': arr.mtype,
#         'q': arr.q,
#         'shape': arr.shape,
#         'data': str_data.tolist()
#     }
#
#     response = requests.post(url, json=payload, **kwargs)
#     response.raise_for_status()
#     return response.json()
#
#
# @experimental
# @extension(name='read_api', mtype='qrofn', injection_type='top_level_function')
# def from_api_qrofn(url: str, **kwargs) -> Fuzzarray:
#     """Download Fuzzarray from remote API endpoint."""
#     response = requests.get(url, **kwargs)
#     response.raise_for_status()
#     data = response.json()
#     mtype = data.get('mtype', 'qrofn')
#     q = data.get('q', 1)
#     str_data = np.array(data['data'], dtype=str)
#
#     def converter(s):
#         return call_extension('str_to_fuzznum', None, fuzznum_str=s, q=q, mtype=mtype)
#
#     to_fuzznum_vec = np.vectorize(converter, otypes=[object])
#     fuzznum_data = to_fuzznum_vec(str_data)
#     return Fuzzarray(fuzznum_data, mtype=mtype)
