import uuid

from Priority import Priority
from Task import Task
from TaskManager import TaskManager


def test_run(max_queue_size: int=5):
  taskManager = TaskManager(max_queue_size=max_queue_size)

  task1 = Task(pid=uuid.uuid4().hex, priority=Priority.medium)
  task2 = Task(pid=uuid.uuid4().hex, priority=Priority.high)
  task3 = Task(pid=uuid.uuid4().hex, priority=Priority.low)
  task4 = Task(pid=uuid.uuid4().hex, priority=Priority.medium)
  task5 = Task(pid=uuid.uuid4().hex, priority=Priority.medium)
  task6 = Task(pid=uuid.uuid4().hex, priority=Priority.high)
  task7 = Task(pid=uuid.uuid4().hex, priority=Priority.low)

  taskManager.add(task=task1)
  taskManager.add(task=task2)
  taskManager.add(task=task3)
  taskManager.add(task=task4)
  taskManager.add(task=task5)
  taskManager.add(task=task6)

  taskManager.list_running_tasks()

  taskManager.add_fifo(task=task6)
  taskManager.list_running_tasks()

  taskManager.add_priority_based(task7)
  taskManager.list_running_tasks()

  # taskManager.kill_all_tasks()
  # taskManager.list_running_tasks()




if __name__ == "__main__":
  test_run()



'''
Data class?
Readme -> https://www.makeareadme.com/
Test?
__main__?
Reverse order?
type check?
commend line parameter?
default values
'''
