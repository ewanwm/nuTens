"""
This module defines various data types used in nuTens
"""
from __future__ import annotations
import enum
import typing
__all__: list[str] = ['device_type', 'scalar_type']
class scalar_type(enum.Enum):
    complex_double: typing.ClassVar[scalar_type]
    complex_float: typing.ClassVar[scalar_type]
    double: typing.ClassVar[scalar_type]
    float: typing.ClassVar[scalar_type]
    @classmethod
    def __new__(cls, value):
        ...
class device_type(enum.Enum):
    cpu: typing.ClassVar[device_type]
    gpu: typing.ClassVar[device_type]
    @classmethod
    def __new__(cls, value):
        ...
