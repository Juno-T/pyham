from typing import Union, NewType
from .wrappers.single_choice import SingleChoiceTypeEnv

InducedMDP = NewType("InducedMDP", SingleChoiceTypeEnv)
# InducedMDP = Union[SingleChoiceTypeEnv, MultiChoice]