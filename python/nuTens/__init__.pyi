"""
Library to calculate neutrino oscillations
"""
from __future__ import annotations
from . import dtype
from . import units
from . import tensor
from . import autograd
from . import propagator
from . import testing
__all__: list[str] = ['autograd', 'dtype', 'propagator', 'tensor', 'testing', 'units']
__version__: str = '0.7.0'
