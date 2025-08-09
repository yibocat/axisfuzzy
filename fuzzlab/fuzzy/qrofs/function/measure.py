#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/9 12:14
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union

import numpy as np

from ....extension import extension


@extension(name='distance', mtype='qrofn', )
def qrofn_distance(fuzz_1, fuzz_2, p_l=2, indeterminacy=True) -> Union[float, np.ndarray]:
    q = fuzz_1.q
    pi1 = fuzz_1.indeterminacy
    pi2 = fuzz_2.indeterminacy
    pi = np.fabs(pi1 ** q - pi2 ** q) ** p_l

    if indeterminacy:
        dis = (0.5 * (np.fabs(fuzz_1.md ** q - fuzz_2.md ** q) ** p_l +
                      np.fabs(fuzz_1.nmd ** q - fuzz_2.nmd ** q) ** p_l + pi)) ** (1 / p_l)
    else:
        dis = (0.5 * (np.fabs(fuzz_1.md ** q - fuzz_2.md ** q) ** p_l +
                      np.fabs(fuzz_1.nmd ** q - fuzz_2.nmd ** q) ** p_l)) ** (1 / p_l)

    return dis.item() if isinstance(dis, np.floating) else dis
