#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 20:11
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import numpy as np

from axisfuzzy.core.fuzznums import fuzznum, Fuzznum
from axisfuzzy.core.fuzzarray import Fuzzarray
from axisfuzzy.mixin.factory import _append_factory


@pytest.fixture
def sample_fuzznums():
    a = fuzznum(mtype="qrofn").create(md=0.2, nmd=0.5)
    b = fuzznum(mtype="qrofn").create(md=0.7, nmd=0.3)
    return a, b


def test_append_fuzznum_to_fuzznum(sample_fuzznums):
    a, b = sample_fuzznums
    arr = _append_factory(a, b)
    assert isinstance(arr, Fuzzarray)
    assert arr.shape == (2,)
    assert np.allclose([arr[0].md, arr[1].md], [0.2, 0.7])


def test_append_fuzznum_to_fuzzarray(sample_fuzznums):
    a, b = sample_fuzznums
    arr = Fuzzarray(data=a, shape=(2,))
    arr2 = _append_factory(arr, b)
    assert arr2.shape == (3,)
    assert isinstance(arr2, Fuzzarray)


def test_append_list_of_fuzznum(sample_fuzznums):
    a, b = sample_fuzznums
    arr = Fuzzarray(data=a, shape=(1,))
    arr2 = _append_factory(arr, [a, b])
    assert arr2.shape == (3,)
    assert isinstance(arr2[0], Fuzznum)


def test_append_fuzzarray_to_fuzzarray(sample_fuzznums):
    a, _ = sample_fuzznums
    arr1 = Fuzzarray(data=a, shape=(2,))
    arr2 = Fuzzarray(data=a, shape=(3,))
    merged = _append_factory(arr1, arr2)
    assert merged.shape == (5,)
    assert np.isclose(merged[0].md, 0.2)


def test_append_list_of_fuzzarrays(sample_fuzznums):
    a, _ = sample_fuzznums
    arr1 = Fuzzarray(data=a, shape=(2,))
    arr2 = Fuzzarray(data=a, shape=(3,))
    merged = _append_factory(arr1, [arr2, arr1])
    assert merged.shape == (7,)
    assert isinstance(merged, Fuzzarray)


def test_append_with_axis_none(sample_fuzznums):
    a, _ = sample_fuzznums
    arr1 = Fuzzarray(data=a, shape=(2,))
    arr2 = Fuzzarray(data=a, shape=(2,))
    merged = _append_factory(arr1, arr2, axis=None)
    assert merged.shape == (4,)  # flattened


def test_append_with_axis_1(sample_fuzznums):
    a, _ = sample_fuzznums
    arr1 = Fuzzarray(data=a, shape=(2, 1))
    arr2 = Fuzzarray(data=a, shape=(2, 1))
    merged = _append_factory(arr1, arr2, axis=1)
    assert merged.shape == (2, 2)


def test_inplace_append_modifies_original(sample_fuzznums):
    a, b = sample_fuzznums
    arr = Fuzzarray(data=a, shape=(2,))
    _append_factory(arr, b, inplace=True)
    assert arr.shape == (3,)
    assert np.isclose(arr[-1].md, b.md)


def test_append_empty_container(sample_fuzznums):
    a, _ = sample_fuzznums
    arr = Fuzzarray(data=None, shape=(0,), mtype=a.mtype, q=a.q)
    arr2 = _append_factory(arr, a)
    assert arr2.shape == (1,)


def test_append_invalid_types_raises():
    a = fuzznum(mtype="qrofn").create(md=0.2, nmd=0.3)
    arr = Fuzzarray(data=a, shape=(1,))
    with pytest.raises(TypeError):
        _append_factory(arr, "invalid")  # unsupported item


if __name__ == "__main__":
    pytest.main([__file__])