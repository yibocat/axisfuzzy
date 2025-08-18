#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import overload, Literal, Any, Dict, Tuple, Type
from .base import MembershipFunction
from .function import (
    SigmoidMF,
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    SMF,
    ZMF,
    DoubleGaussianMF,
    GeneralizedBellMF,
    PiMF
)

def get_mf_class(name: str) -> Type[MembershipFunction]: ...

@overload
def create_mf(name: Literal["sigmoid"], **mf_kwargs: Any) -> Tuple[SigmoidMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["trimf"], **mf_kwargs: Any) -> Tuple[TriangularMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["trapmf"], **mf_kwargs: Any) -> Tuple[TrapezoidalMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["gaussmf"], **mf_kwargs: Any) -> Tuple[GaussianMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["smf"], **mf_kwargs: Any) -> Tuple[SMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["zmf"], **mf_kwargs: Any) -> Tuple[ZMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["gbellmf"], **mf_kwargs: Any) -> Tuple[GeneralizedBellMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["pimf"], **mf_kwargs: Any) -> Tuple[PiMF, Dict[str, Any]]: ...

@overload
def create_mf(name: Literal["gauss2mf"], **mf_kwargs: Any) -> Tuple[DoubleGaussianMF, Dict[str, Any]]: ...

# Fallback signature for dynamic or unknown names
def create_mf(name: str, **mf_kwargs: Any) -> Tuple[MembershipFunction, Dict[str, Any]]: ...