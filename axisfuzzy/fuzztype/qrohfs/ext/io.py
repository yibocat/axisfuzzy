#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 20:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import csv
import json
import pickle
from typing import Any, Dict, List

import numpy as np

from ....core import Fuzzarray, get_fuzztype_backend


def _qrohfn_to_csv(arr: Fuzzarray, path: str, **kwargs) -> None:
    """将 QROHFN Fuzzarray 导出为 CSV 格式"""
    if arr.mtype != 'qrohfn':
        raise TypeError(f"to_csv for mtype 'qrohfn' cannot be called on Fuzzarray with mtype '{arr.mtype}'")

    # 获取后端组件数组
    mds, nmds = arr.backend.get_component_arrays()

    # 创建字符串表示
    str_data = np.empty(arr.shape, dtype=object)

    # 使用 np.nditer 高效迭代
    with np.nditer([mds, nmds, str_data],
                   flags=['multi_index', 'refs_ok'],
                   op_flags=[['readonly'], ['readonly'], ['writeonly']]):

        for md_val, nmd_val, str_val in np.nditer([mds, nmds, str_data],
                                                  flags=['refs_ok'],
                                                  op_flags=[['readonly'], ['readonly'], ['writeonly']]):
            md_list = md_val.item()
            nmd_list = nmd_val.item()

            # 格式化为字符串
            if md_list is None or len(md_list) == 0:
                md_str = "[]"
            else:
                md_str = "[" + ",".join(f"{x:.6f}" for x in md_list) + "]"

            if nmd_list is None or len(nmd_list) == 0:
                nmd_str = "[]"
            else:
                nmd_str = "[" + ",".join(f"{x:.6f}" for x in nmd_list) + "]"

            str_val[...] = f"<{md_str},{nmd_str}>"

    # 写入 CSV
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, **kwargs)
        if str_data.ndim == 1:
            writer.writerow(str_data)
        else:
            for row in str_data:
                if row.ndim == 0:
                    writer.writerow([row.item()])
                else:
                    writer.writerow(row)


def _qrohfn_from_csv(path: str, q: int, **kwargs) -> Fuzzarray:
    """从 CSV 格式导入 QROHFN Fuzzarray"""
    from .string import _qrohfn_from_str

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, **kwargs)
        str_data_list = list(reader)

    if not str_data_list:
        return Fuzzarray([], mtype='qrohfn', q=q)

    str_data = np.array(str_data_list, dtype=str)
    shape = str_data.shape

    # 创建后端
    backend_cls = get_fuzztype_backend('qrohfn')
    new_backend = backend_cls(shape, q=q)

    # 解析每个字符串并填充后端
    for idx in np.ndindex(shape):
        fuzznum = _qrohfn_from_str(str_data[idx], q=q)
        new_backend.set_fuzznum_data(idx, fuzznum)

    return Fuzzarray(backend=new_backend)


def _qrohfn_to_json(arr: Fuzzarray, path: str, **kwargs) -> None:
    """将 QROHFN Fuzzarray 导出为 JSON 格式"""
    if arr.mtype != 'qrohfn':
        raise TypeError(f"Expected QROHFN Fuzzarray, got mtype '{arr.mtype}'")

    mds, nmds = arr.backend.get_component_arrays()

    def serialize_hesitant_sets(hesitant_arrays: np.ndarray) -> List[Any]:
        """将犹豫集数组序列化为列表结构"""
        result = []
        for idx in np.ndindex(hesitant_arrays.shape):
            hesitant_set = hesitant_arrays[idx]
            if hesitant_set is None:
                result.append([])
            else:
                result.append(hesitant_set.tolist())
        return np.array(result, dtype=object).reshape(hesitant_arrays.shape).tolist()

    data = {
        'mtype': arr.mtype,
        'q': arr.q,
        'shape': list(arr.shape),
        'md_data': serialize_hesitant_sets(mds),
        'nmd_data': serialize_hesitant_sets(nmds)
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)


def _qrohfn_from_json(path: str, **kwargs) -> Fuzzarray:
    """从 JSON 格式导入 QROHFN Fuzzarray"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f, **kwargs)

    q = data.get('q', 1)
    shape = tuple(data['shape'])
    md_data = data['md_data']
    nmd_data = data['nmd_data']

    # 创建后端
    backend_cls = get_fuzztype_backend('qrohfn')
    new_backend = backend_cls(shape, q=q)

    # 递归填充犹豫集数组
    def deserialize_to_arrays(data_list: List, target_array: np.ndarray) -> None:
        """将列表结构反序列化为犹豫集数组"""
        flat_data = np.array(data_list).flatten()
        flat_target = target_array.flatten()

        for i, hesitant_list in enumerate(flat_data):
            if hesitant_list:
                flat_target[i] = np.array(hesitant_list, dtype=np.float64)
            else:
                flat_target[i] = np.array([], dtype=np.float64)

    # 获取后端数组并填充
    mds, nmds = new_backend.get_component_arrays()
    deserialize_to_arrays(md_data, mds)
    deserialize_to_arrays(nmd_data, nmds)

    return Fuzzarray(backend=new_backend)


def _qrohfn_to_npy(arr: Fuzzarray, path: str, **kwargs) -> None:
    """将 QROHFN Fuzzarray 导出为 NumPy 格式（使用 pickle 序列化犹豫集）"""
    if arr.mtype != 'qrohfn':
        raise TypeError(f"Expected QROHFN Fuzzarray, got mtype '{arr.mtype}'")

    mds, nmds = arr.backend.get_component_arrays()

    # 由于犹豫集长度不定，使用字典格式保存
    save_data = {
        'mtype': arr.mtype,
        'q': arr.q,
        'shape': arr.shape,
        'mds': mds,
        'nmds': nmds
    }

    np.save(path, save_data, **kwargs)


def _qrohfn_from_npy(path: str, **kwargs) -> Fuzzarray:
    """从 NumPy 格式导入 QROHFN Fuzzarray"""
    save_data = np.load(path, allow_pickle=True, **kwargs).item()

    q = save_data['q']
    shape = save_data['shape']
    mds = save_data['mds']
    nmds = save_data['nmds']

    # 创建后端并直接设置数组
    backend_cls = get_fuzztype_backend('qrohfn')
    new_backend = backend_cls.from_arrays(mds, nmds, q=q)

    return Fuzzarray(backend=new_backend)


def _qrohfn_to_pickle(arr: Fuzzarray, path: str, **kwargs) -> None:
    """将 QROHFN Fuzzarray 导出为 Pickle 格式"""
    if arr.mtype != 'qrohfn':
        raise TypeError(f"Expected QROHFN Fuzzarray, got mtype '{arr.mtype}'")

    with open(path, 'wb') as f:
        pickle.dump(arr, f, **kwargs)


def _qrohfn_from_pickle(path: str, **kwargs) -> Fuzzarray:
    """从 Pickle 格式导入 QROHFN Fuzzarray"""
    with open(path, 'rb') as f:
        return pickle.load(f, **kwargs)


# 未来可扩展的高级 I/O 方法:
# - HDF5 格式支持 (适合大规模数据)
# - SQLite 数据库支持 (适合结构化查询)
# - Parquet 格式支持 (适合列式存储)
# - 自定义二进制格式 (最优性能)

# def _qrohfn_to_hdf5(arr: Fuzzarray, path: str, dataset_name: str = 'qrohfn_data', **kwargs):
#     """导出到 HDF5 格式 - 未来实现"""
#     pass

# def _qrohfn_from_hdf5(path: str, dataset_name: str = 'qrohfn_data', **kwargs) -> Fuzzarray:
#     """从 HDF5 格式导入 - 未来实现"""
#     pass









