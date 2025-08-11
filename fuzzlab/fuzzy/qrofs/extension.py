#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 10:40
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .factory import qrofn_empty, qrofn_poss, qrofn_negs, qrofn_full

from ...extension import extension


@extension(name='empty', mtype='qrofn', injection_type='top_level_function')
def qrofn_empty_ext(*args, **kwargs):
    """Create an empty (uninitialized) QROFN Fuzzarray."""
    return qrofn_empty(*args, **kwargs)


@extension(name='positive', mtype='qrofn', injection_type='top_level_function')
def qrofn_poss_ext(*args, **kwargs):
    """Create a QROFN Fuzzarray filled with ones (md=1, nmd=0)."""
    return qrofn_poss(*args, **kwargs)


@extension(name='negative', mtype='qrofn', injection_type='top_level_function')
def qrofn_negs_ext(*args, **kwargs):
    """Create a QROFN Fuzzarray filled with ones (md=0, nmd=1)."""
    return qrofn_negs(*args, **kwargs)


@extension(name='full', mtype='qrofn', injection_type='top_level_function')
def qrofn_full_ext(*args, **kwargs):
    """Create a QROFN Fuzzarray filled with a specific Fuzznum."""
    return qrofn_full(*args, **kwargs)
