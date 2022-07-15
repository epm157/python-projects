from collections import deque

from Task import Task


class TaskManager:

  #max_queue_size: int
  #tasks_queue = deque()

  def __init__(self, max_queue_size: int=5):
   self.max_queue_size = max_queue_size
   self.tasks_queue = deque([], self.max_queue_size)


  def add(self, task: Task):
    '''
    Adds a task to the tasks queue. If the queue is full, an error message will be printed
    :param task: an instance of task class
    '''
    if len(self.tasks_queue) >= self.max_queue_size:
      error_message = "Queue has reached it's maximum size. Please remove some tasks before adding a new one."
      #raise Exception(error_message)
      print(error_message)
      return

    self.add_fifo(task=task)

  def add_fifo(self, task: Task):
    '''
   Adds a task to the tasks queue. If the queue is full, the oldest task will be removed
   :param task: an instance of task class
   '''
    self.tasks_queue.append(task)

  def add_priority_based(self, new_task: Task):
    '''
    Adds a task to the tasks queue. If the queue is full, the oldest task with lower priority will be removed.
    if none of the existing tasks have lower priority than the new one, an error message will be printed
    Complexity of this method is O(n). For more info please see:
    https://stackoverflow.com/questions/58152201/time-complexity-deleting-element-of-deque
    :param new_task:
    '''
    if len(self.tasks_queue) < self.max_queue_size:
      self.add_fifo(task=new_task)
      return

    index_to_remove = -1
    current_index = 0
    for task in self.tasks_queue:
      if new_task.priority > task.priority:
        index_to_remove = current_index
        break

      current_index = current_index + 1

    if index_to_remove == -1:
      print(f'Queue is full and there is no task with lower priority than {new_task.priority} to remove :(')
      return

    del self.tasks_queue[index_to_remove]
    self.add_fifo(new_task)

  def kill_all_tasks(self):
    '''
    Kills all tasks
    '''
    print("-----Killing all of the tasks queue-----")
    for task in self.tasks_queue:
      task.kill()

  def clear_tasks_queue(self):
    '''
    Clears tasks queue
    '''
    print("-----Clearing the tasks queue-----")
    self.tasks_queue.clear()

  def list_running_tasks(self):
    '''
    Prints list of running tasks
    '''
    print("-----Printing list of running tasks-----")
    if len(self.tasks_queue) == 0:
      print("Tasks queue is empty")
      return
    for task in self.tasks_queue:
      if task.isAlive:
        print(task)
