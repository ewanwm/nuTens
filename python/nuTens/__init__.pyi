"""
Library to calculate neutrino oscillations
"""
from __future__ import annotations
from . import dtype
from . import propagator
from . import tensor
from . import testing
from . import units
__all__: list[str] = ['dtype', 'propagator', 'tensor', 'testing', 'units']
__version__: str = '0.5.0'
