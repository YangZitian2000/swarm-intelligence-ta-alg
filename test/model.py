# Task class
# ST:Surface Target
# AT:Air Target
# SA:Surface Area
class Task:
  def __init__(self, id, type):
    self.id = id
    self.type = type  # string:{"ST","AT","SA"}
    self.coordinates = (0, 0)  # tuple:(double,double)


class ST_Task(Task):
  def __init__(self, id, type, phase=1):
    super().__init__(id, type)
    self.phase = phase  # phase 1:探测任务 phase 2:打击任务


class SA_Task(Task):
  def __init__(self, id, type, side_length=0):
    super().__init__(id, type)
    self.side_length = side_length  # length of side of searching area


class AT_Task(Task):
  def __init__(self, id, type):
    super().__init__(id, type)


# Agent class
# UAV:Unmanned Air Vehicle
# USV:Unmanned Surface Vehicle
class Agent:
  def __init__(self, id, type):
    self.id = id
    self.type = type  # string:{"UAV","USV"}
    self.coordinate = (0, 0)  # tuple:(double,double)


class UAV(Agent):
  def __init__(self, id, type):
    super().__init__(id, type)


class USV(Agent):
  def __init__(self, id, type):
    super().__init__(id, type)
