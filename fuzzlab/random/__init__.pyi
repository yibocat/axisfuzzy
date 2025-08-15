#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 17:18
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .api import (
    rand as rand,
    choice as choice,
    uniform as uniform,
    normal as normal,
    beta as beta,
    registry as registry,
    register as register,
    unregister as unregister,
    get_generator as get_generator,
    list_registered as list_registered,
    is_registered as is_registered,
    get_seed as get_seed,
    set_seed as set_seed,
    get_rng as get_rng,
    spawn_rng as spawn_rng,
)
