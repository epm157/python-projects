
from dataclasses import dataclass
from uuid import UUID

from Priority import Priority


@dataclass
class Task:
  pid: UUID
  priority: Priority
  isAlive: bool = True

  def kill(self):
    self.isAlive = False
