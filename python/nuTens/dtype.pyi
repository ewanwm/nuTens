"""
This module defines various data types used in nuTens
"""
from __future__ import annotations
import enum
import typing
__all__: list[str] = ['device_type', 'scalar_type']
class device_type(enum.Enum):
    cpu: typing.ClassVar[device_type]  # value = <device_type.cpu: 0>
    gpu: typing.ClassVar[device_type]  # value = <device_type.gpu: 1>
class scalar_type(enum.Enum):
    complex_double: typing.ClassVar[scalar_type]  # value = <scalar_type.complex_double: 3>
    complex_float: typing.ClassVar[scalar_type]  # value = <scalar_type.complex_float: 2>
    double: typing.ClassVar[scalar_type]  # value = <scalar_type.double: 1>
    float: typing.ClassVar[scalar_type]  # value = <scalar_type.float: 0>
