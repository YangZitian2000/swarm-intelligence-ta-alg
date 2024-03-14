class Task:
  def __init__(self, task_type, coordinates):
    self.task_type = task_type
    self.coordinates = coordinates


class ST_Task(Task):
  def __init__(self, task_type, coordinates, phase):
    super().__init__(task_type, coordinates)
    self.phase = phase


class SA_Task(Task):
  def __init__(self, task_type, coordinates, side_length):
    super().__init__(task_type, coordinates)
    self.side_length = side_length


class AT_Task(Task):
  def __init__(self, task_type, coordinates):
    super().__init__(task_type, coordinates)


# 示例用法
st_task = ST_Task(task_type="ST", coordinates=(10, 20), phase=1)
sa_task = SA_Task(task_type="SA", coordinates=(30, 40), side_length=5)
at_task = AT_Task(task_type="AT", coordinates=(50, 60))

print("ST Task Type:", st_task.task_type)
print("ST Coordinates:", st_task.coordinates)
print("ST Phase:", st_task.phase)

print("SA Task Type:", sa_task.task_type)
print("SA Coordinates:", sa_task.coordinates)
print("SA Side Length:", sa_task.side_length)

print("AT Task Type:", at_task.task_type)
print("AT Coordinates:", at_task.coordinates)
