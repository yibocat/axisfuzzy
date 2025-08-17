#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import Optional, Union, List, Literal, Callable, Any

def extension(name: str,
              mtype: Optional[str] = ...,
              target_classes: Union[str, List[str]] = ...,
              injection_type: Literal['instance_method',
                                      'top_level_function',
                                      'instance_property',
                                      'both'] = ...,
              is_default: bool = ...,
              priority: int = ...,
              **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

def batch_extension(registrations: List[dict]) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
