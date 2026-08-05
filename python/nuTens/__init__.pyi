"""
Library to calculate neutrino oscillations
"""
from __future__ import annotations
from . import autograd
from . import dtype
from . import propagator
from . import tensor
from . import testing
from . import units
__all__: list[str] = ['autograd', 'dtype', 'propagator', 'tensor', 'testing', 'units']
__version__: str = '0.6.1'
