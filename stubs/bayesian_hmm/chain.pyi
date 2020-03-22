# Stubs for bayesian_hmm.chain (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

import numpy
import typing
from . import variables
from typing import Any

Numeric: Any
DictStrNum: Any
InitDict = DictStrNum
DictStrDictStrNum: Any
NestedInitDict = DictStrDictStrNum

class Chain:
    emission_sequence: Any = ...
    latent_sequence: Any = ...
    T: Any = ...
    def __init__(self, sequence: typing.Sequence[variables.State]) -> None: ...
    def __len__(self) -> int: ...
    @property
    def initialised_flag(self) -> bool: ...
    @initialised_flag.setter
    def initialised_flag(self, value: bool) -> None: ...
    def to_array(self) -> numpy.array: ...
    def initialise(self, states: typing.Set[variables.State]) -> None: ...
    def log_likelihood(self, emission_probabilities: NestedInitDict, transition_probabilities: NestedInitDict) -> float: ...
    def resample(self, states: typing.Set[variables.State], emission_probabilities: NestedInitDict, transition_probabilities: NestedInitDict) -> None: ...

def resample_latent_sequence(sequences: typing.Tuple[typing.List[variables.State], typing.List[variables.State]], states: typing.Set[variables.State], emission_probabilities: NestedInitDict, transition_probabilities: NestedInitDict) -> typing.List[variables.State]: ...
