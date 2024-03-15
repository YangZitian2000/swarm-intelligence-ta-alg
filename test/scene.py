from model import *
import random


class Scene:
  def __init__(self, agent_code: str, task_code: str, distribution="None"):
    self.agent_code = agent_code
    self.agent = Factory.generate_agents(agent_code)
    self.task_code = task_code
    self.task = Factory.generate_tasks(task_code)
    self.distribution = distribution


# 打击ST场景
class SD(Scene):
  def __init__(self, agent_code, task_code, distribution):
    super().__init__(agent_code, task_code, distribution)


# 打击AT场景
class KD(Scene):
  def __init__(self, agent_code, task_code, distribution):
    super().__init__(agent_code, task_code, distribution)


# 区域搜索场景
class QD(Scene):
  def __init__(self, agent_code, task_code, distribution):
    super().__init__(agent_code, task_code, distribution)


class Factory:
  @staticmethod
  # 根据Agent类型标识生成对应的Agent列表(不包含坐标)
  def generate_agents(agent_code: str):
    agents: list[Agent] = []
    code = int(agent_code[1:])
    if code > 10 or code < 1:
      raise Exception("Unknown Agent code!")
    # A1~A10
    for i in range(1, code * 10 + 1):
      if i <= code * 3:
        agents.append(Agent(id=i, type="UAV"))
      else:
        agents.append(Agent(id=i, type="USV"))
    return agents

  @staticmethod
  # 根据Task类型标识生成对应的Task列表(不包含坐标)
  def generate_tasks(task_code: str):
    tasks: list[Task] = []
    code = int(task_code[1:])
    if code <= 10:  # T1~T10
      for i in range(1, code * 10 + 1):
        tasks.append(ST_Task(id=i, type="ST"))
    elif code <= 20:  # T11~T20
      for i in range(1, (code - 10) * 30 + 1):
        tasks.append(AT_Task(id=i, type="AT"))
    elif code <= 27:
      if code <= 23:  # T21~T23
        for i in range(1, (code - 20) + 1):
          tasks.append(SA_Task(id=i, type="SA"))
      else:  # T24~T26
        for i in range(1, (code - 23) * 5 + 1):
          tasks.append(SA_Task(id=i, type="SA"))
    else:
      raise Exception("Unknown Task code!")
    return tasks

  @staticmethod
  # 根据测试数据类型标识生成对应的测试用例实例
  def generate_scene(scene_code):
    type = scene_code[:2]
    code = int(scene_code[2:])
    if type == "SD":
      if code == 1:
        return SD("A1", "T1")
      elif code == 2:
        return SD("A2", "T1")
      # 润洋TODO
    elif type == "KD":
      if code == 1:
        return KD("A1", "T1")
      elif code == 2:
        return KD("A2", "T1")
      # 润洋TODO
    elif type == "QD":
      if code == 1:
        return QD("A1", "T1")
      elif code == 2:
        return QD("A2", "T1")
      # 润洋TODO
    else:
      raise Exception("Unknown scene type!")
