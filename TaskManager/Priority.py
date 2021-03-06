from enum import Enum
from functools import total_ordering

@total_ordering
class Priority(Enum):
  low = 1
  medium = 2
  high = 3

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value
    return NotImplemented
