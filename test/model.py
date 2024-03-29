import random


requirements = [18, 20, 25, 28, 30, 32, 35, 40]
yield_rate = 100
rate = 0.15


# Task class
class Task:
  def __init__(self, id, type):
    self.id = id
    self.type = type  # string:{"ST", "AT", "SA"}
    self.rec_coor = (0, 0)  # tuple:(float, float) rectangular coordinate system
    self.polar_coor = (0, 0)  # tuple:(float, float) polar coordinate system


# ST:Surface Target
class ST_Task(Task):
  def __init__(self, id, type, direction):
    super().__init__(id, type)
    self.speed = 55.56  # 30节 = 55.56km/h
    self.direction = direction  # 单位: °
    self.requirement = random.choice(requirements)  # int: 资源需求量
    self.profit = random.uniform(self.requirement * yield_rate * (1 - rate),
                                 self.requirement * yield_rate * (1 + rate))  # float: Task收益


# AT:Air Target
class AT_Task(Task):
  def __init__(self, id, type):
    super().__init__(id, type)
    self.speed = 1080  # 单位：km/h(300m/s)
    self.direction = 0  # 单位: °


# SA:Surface Area
class SA_Task(Task):
  def __init__(self, id, type, side_length=0):
    super().__init__(id, type)
    self.side_length = side_length  # length of side of searching area


# Agent class
class Agent:
  def __init__(self, id, type):
    self.id = id
    self.type = type  # string:{"UAV","USV"}
    self.rec_coor = (0, 0)  # tuple:(float, float) rectangular coordinate system
    self.polar_coor = (0, 0)  # tuple:(float, float) polar coordinate system


# 无人机
class SD_Agent(Agent):
  def __init__(self, id, type, group_id):
    super().__init__(id, type)
    self.group_id = group_id


# 无人艇
class KD_Agent(Agent):
  def __init__(self, id, type):
    super().__init__(id, type)
    self.res_angle = [0, 0]


# 有人艇
class QD_Agent(Agent):
  def __init__(self, id, type, group_id):
    super().__init__(id, type)
    self.group_id = group_id


# 编组类, 形状为圆形, 坐标为圆心
class Group:
  def __init__(self, id):
    self.id = id
    self.rec_coor = (0, 0)  # tuple:(float, float) rectangular coordinate system
    self.polar_coor = (0, 0)  # tuple:(float, float) polar coordinate system
    self.radius = 20  # 半径: km
    self.agents = []  # 编组中包含的Agent id
