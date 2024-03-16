from model import *
import random
import matplotlib.pyplot as plt


class Scene:
  def __init__(self, scene_code: str, agent_code: str, task_code: str, distribution: str):
    self.scene_code = scene_code
    self.agent_code = agent_code  # Agent类型标识
    self.agent = Factory.generate_agents(agent_code)
    self.task_code = task_code  # Task类型标识
    self.task = Factory.generate_tasks(task_code)
    # 均匀分布:"uniform"
    # 集群分布:"cluster"
    self.distribution = distribution  # Agent分布类型:{"uniform","cluster"}
    if self.distribution == "uniform":
      self.generate_agent_coordinate_uniform((0, 50), (50, 200), (0, 200), (0.3))
    elif self.distribution == "cluster":
      self.generate_agent_coordinate_cluster()
    else:
      raise Exception("Unknown distribution!")

  def generate_agent_coordinate_uniform(self, x_range_most: tuple, x_range_few: tuple,
                                        y_range: tuple, few_percentage: float):
    agents_most = int(len(self.agent) * few_percentage)
    agents_few = len(self.agent) - agents_most
    i = 0
    # 生成x坐标在x_range_most范围内的大部分agent
    for _ in range(agents_most):
      x = random.uniform(x_range_most[0], x_range_most[1])
      y = random.uniform(y_range[0], y_range[1])
      self.agent[i].coordinate = (x, y)
      i += 1

    # 生成x坐标在x_range_few范围内的少部分agent
    for _ in range(agents_few):
      x = random.uniform(x_range_few[0], x_range_few[1])
      y = random.uniform(y_range[0], y_range[1])
      self.agent[i].coordinate = (x, y)
      i += 1

  def generate_agent_coordinate_cluster(self):
    pass

  def generate_ST_task_coordinate(self, x_ranges: list[float], y_range: tuple):
    error_rate = 0.1
    for i in range(len(self.task)):
      x_range = random.choice(x_ranges)
      x = random.uniform(x_range * (1 - error_rate), x_range * (1 + error_rate))
      y = random.uniform(y_range[0], y_range[1])
      self.task[i].coordinate = (x, y)

  def generate_AT_task_coordinate(self):
    pass

  def generate_SA_task_coordinate(self):
    pass


# 打击ST场景
class SD(Scene):
  def __init__(self, scene_code, agent_code, task_code, distribution="uniform"):
    super().__init__(scene_code, agent_code, task_code, distribution)
    self.generate_ST_task_coordinate([300, 600, 1000], (0, 200))


# 打击AT场景
class KD(Scene):
  def __init__(self, scene_code, agent_code, task_code, distribution="uniform"):
    super().__init__(scene_code, agent_code, task_code, distribution)


# 区域搜索场景
class QD(Scene):
  def __init__(self, scene_code, agent_code, task_code, distribution="uniform"):
    super().__init__(scene_code, agent_code, task_code, distribution)


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
        return SD(scene_code=scene_code, agent_code="A1", task_code="T1")
      elif code == 2:
        return SD(scene_code=scene_code, agent_code="A2", task_code="T1")
      # 润洋TODO
    elif type == "KD":
      if code == 1:
        return KD(scene_code=scene_code, agent_code="A1", task_code="T11")
      elif code == 2:
        return KD(scene_code=scene_code, agent_code="A1", task_code="T12")
      # 润洋TODO
    elif type == "QD":
      if code == 1:
        return QD(scene_code=scene_code, agent_code="A1", task_code="T21")
      elif code == 2:
        return QD(scene_code=scene_code, agent_code="A2", task_code="T22")
      # 润洋TODO
    else:
      raise Exception("Unknown scene type!")


scene_list: list[Scene] = []
SD1 = Factory.generate_scene("SD1")
SD2 = Factory.generate_scene("SD2")
scene_list.append(SD1)
scene_list.append(SD2)

# 控制台输出信息
for scene in scene_list:
  print("Scene : ", scene.scene_code)
  print("\tAgents:")
  for agent in scene.agent:
    print(f"\t- ID:{agent.id}, Type: {agent.type}, Coordinates: {agent.coordinate}")
  print("\n\tTasks:")
  for task in scene.task:
    print(f"\t- ID:{task.id}, Type: {task.type}, Coordinates: {task.coordinate}")


# 绘图
for scene in scene_list:
  fig, ax = plt.subplots()
  ax.set_xlim(-60, 1200)
  ax.set_ylim(-10, 200)
  ax.scatter([agent.coordinate[0] for agent in scene.agent if agent.type == "UAV"],
             [agent.coordinate[1] for agent in scene.agent if agent.type == "UAV"], color="g", label="UAV", marker=".")  # 无人机
  ax.scatter([agent.coordinate[0] for agent in scene.agent if agent.type == "USV"],
             [agent.coordinate[1] for agent in scene.agent if agent.type == "USV"], color="b", label="USV", marker=".")  # 无人艇

  ax.scatter([task.coordinate[0] for task in scene.task if task.type == "ST"],
             [task.coordinate[1] for task in scene.task if task.type == "ST"], color="r", label="ST", marker="x")  # ST任务

  ax.legend()
  ax.set_title(scene.scene_code + " : " + scene.agent_code + " + " + scene.task_code)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.grid(True)
  plt.show()
