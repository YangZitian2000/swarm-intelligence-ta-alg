from model import *
import random
import math
import matplotlib.pyplot as plt
import json
import os

random.seed(0)

# SD settings
agent_num_per_group_SD = [[(3, 1), (4, 1)],  # 每组Agent数, 每组中MSV数
                          [(3, 1), (2, 0), (3, 1)],
                          [(3, 0), (3, 0), (4, 1), (2, 0)],
                          [(3, 0), (3, 0), (3, 0), (3, 0), (4, 1), (4, 1), (4, 1), (2, 0), (2, 0), (2, 0)],
                          [(4, 0) for _ in range(15)]]

task_num_SD = [2, 2, 4, 8, 24]  # 每个场景的任务数


class Scene:
  def __init__(self, scene_code: str):
    self.scene_code = scene_code
    # 创建组, 除坐标外其余均确定
    self.groups = Factory.generate_groups(scene_code=scene_code)
    # 创建Agent, 除坐标外其余均确定
    self.agents = Factory.generate_agents(scene_code=scene_code)
    # 创建Task, 除坐标外其余均确定
    self.tasks = Factory.generate_tasks(scene_code=scene_code)

  def export_to_json(self):
    self.json_data = {"agents": [], "tasks": []}
    # 将Agent信息添加到 JSON 数据中
    for agent in self.agents.values():
      agent_info = {
          "id": agent.id,
          "type": agent.type,
          "rectangular coordinate": agent.rec_coor,
          "polar coordinate": agent.polar_coor,
          "group id": agent.group_id
      }
      self.json_data["agents"].append(agent_info)

    # 将Task信息添加到 JSON 数据中
    for task in self.tasks.values():
      task_info = {
          "id": task.id,
          "type": task.type,
          "rectangular coordinate": task.rec_coor,
          "polar coordinate": task.polar_coor,
      }
      if isinstance(task, ST_Task):
        task_info["speed"] = task.speed
        task_info["direction"] = task.direction
        task_info["requirement"] = task.requirement
        task_info["profit"] = task.profit
      elif isinstance(task, AT_Task):
        pass
      elif isinstance(task, SA_Task):
        pass
      else:
        raise Exception("Invalid task type!")
      self.json_data["tasks"].append(task_info)

    # 将 JSON 数据写入文件
    with open(os.path.dirname(os.path.abspath(__file__)) + "\\\\" + self.scene_code + ".json", 'w') as f:
      json.dump(self.json_data, f, indent=4)


def angle_to_radian(angle: float):
  return (450 - angle) % 360 / 180 * math.pi


def radian_to_angle(radian: float):
  return (450 - radian / math.pi * 180) % 360


def rec_to_polar(point):
  x, y = point
  r = math.sqrt(x**2 + y**2)
  theta = math.atan2(y, x)
  a = radian_to_angle(theta)
  return r, a


def rec_distance(point1, point2):
  x1, y1 = point1
  x2, y2 = point2
  distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return distance


def polar_distance(polar1, polar2):
  r1, theta1 = polar1
  r2, theta2 = polar2
  theta1 = angle_to_radian(theta1)
  theta2 = angle_to_radian(theta2)
  distance = math.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * math.cos(theta2 - theta1))
  return distance


# 打击ST场景
class SD(Scene):
  def __init__(self, scene_code):
    code = int(scene_code[2:])
    if code > 5 or code < 1:
      raise Exception("Invalid SD scene code!")
    super().__init__(scene_code)
    # 确定组的坐标
    self.generate_group_coordinate_SD()
    # 确定任务坐标
    self.generate_task_coordinate_SD()
    # 确定Agent坐标
    self.generate_agent_coordinate_SD()

  def generate_group_coordinate_SD(self):
    code = int(self.scene_code[2:])
    # Group 1默认位于原点, 且包含指挥舰
    if code == 1:
      group_2 = self.groups[2]
      r = random.uniform(180, 220)
      a = random.uniform(0, 30)
      group_2.polar_coor = (r, a)
      x = r * math.cos(angle_to_radian(a))
      y = r * math.sin(angle_to_radian(a))
      group_2.rec_coor = (x, y)
      self.groups[2] = group_2
    elif code == 2:
      pass
    elif code == 3:
      pass
    elif code == 4:
      pass
    else:
      pass

  def generate_task_coordinate_SD(self):
    min_range, max_range = 270, 330
    code = int(self.scene_code[2:])
    task_num: int = task_num_SD[code - 1]
    if code == 1:
      avaliable_angle = []
      for angle in range(0, 181):
        distance_min = polar_distance((min_range, angle), self.groups[2].polar_coor)
        distance_max = polar_distance((max_range, angle), self.groups[2].polar_coor)
        if min_range < distance_min and distance_min < max_range or \
                min_range < distance_max and distance_max < max_range:
          avaliable_angle.append(angle)
      while True:
        [a1, a2] = random.sample(avaliable_angle, task_num)
        r1, r2 = random.uniform(300, 330), random.uniform(300, 330)
        if polar_distance((r1, a1), (r2, a2)) >= 50:
          x1, y1 = r1 * math.cos(angle_to_radian(a1)), r1 * math.sin(angle_to_radian(a1))
          x2, y2 = r2 * math.cos(angle_to_radian(a2)), r2 * math.sin(angle_to_radian(a2))
          self.tasks[1].polar_coor = (r1, a1)
          self.tasks[1].rec_coor = (x1, y1)
          self.tasks[2].polar_coor = (r2, a2)
          self.tasks[2].rec_coor = (x2, y2)
          break
    elif code == 2:
      pass
    elif code == 3:
      pass
    elif code == 4:
      pass
    else:
      pass

  def generate_agent_coordinate_SD(self):
    code = int(self.scene_code[2:])
    if code == 1:
      for group in self.groups.values():
        agent_num_in_group = len(group.agents)
        agent_polar_coors = []
        if group.id == 1:  # 如果是第一组，则包含指挥舰位于原点
          agent_polar_coors.append((0, 0))
        while len(agent_polar_coors) < agent_num_in_group:
          r = random.uniform(0, group.radius)
          a = random.uniform(0, 360)
          valid = True
          for cur_agent_polar_coor in agent_polar_coors:
            if polar_distance((r, a), cur_agent_polar_coor) < 5:
              valid = False
          if valid:
            agent_polar_coors.append((r, a))
        for i, agent_id in enumerate(group.agents):
          if agent_id == 1:
            continue
          r, a = agent_polar_coors[i]
          x, y = r * math.cos(angle_to_radian(a)), r * math.sin(angle_to_radian(a))
          self.agents[agent_id].rec_coor = (group.rec_coor[0] + x, group.rec_coor[1] + y)
          self.agents[agent_id].polar_coor = rec_to_polar(self.agents[agent_id].rec_coor)
    elif code == 2:
      pass
    elif code == 3:
      pass
    elif code == 4:
      pass
    else:
      pass


# 打击AT场景
class KD(Scene):
  def __init__(self, scene_code):
    super().__init__(scene_code)


# 区域搜索场景
class QD(Scene):
  def __init__(self, scene_code):
    super().__init__(scene_code)


class Factory:
  @staticmethod
  def generate_groups(scene_code: str):
    code = int(scene_code[2:])
    agent_per_group = agent_num_per_group_SD[code - 1]
    group_num = len(agent_per_group)
    groups: dict[int, Group] = {}
    agent_id: int = 1
    for i in range(group_num):
      group_id = i + 1
      group = Group(id=group_id)
      for _ in range(agent_per_group[i][0]):
        group.agents.append(agent_id)
        agent_id += 1
      groups[group_id] = group
    return groups

  @staticmethod
  # 根据场景类型标识生成对应的Agent列表(不包含坐标)
  def generate_agents(scene_code: str):
    type = scene_code[0:2]
    code = int(scene_code[2:])
    if type == "SD":
      agent_per_group = agent_num_per_group_SD[code - 1]
      agents: dict[int, Agent] = {}
      agent_id: int = 1
      for i in range(len(agent_per_group)):
        for _ in range(agent_per_group[i][1]):
          agent = Agent(id=agent_id, type="MSV", group_id=i + 1)
          agents[agent_id] = agent
          agent_id += 1
        for _ in range(agent_per_group[i][0] - agent_per_group[i][1]):
          agent = Agent(id=agent_id, type="USV", group_id=i + 1)
          agents[agent_id] = agent
          agent_id += 1
    elif type == "KD":
      pass
    elif type == "QD":
      pass
    else:
      raise Exception("Invalid scene type!")
    return agents

  @staticmethod
  # 根据场景类型标识生成对应的Task列表(不包含坐标)
  def generate_tasks(scene_code: str):
    type = scene_code[0:2]
    code = int(scene_code[2:])
    if type == "SD":
      tasks: dict[int, ST_Task] = {}
      task_num = task_num_SD[code - 1]
      task_id: int = 1
      direction = random.uniform(0, 360)
      for _ in range(task_num):
        task = ST_Task(id=task_id, type="ST", direction=direction)
        tasks[task_id] = task
        task_id += 1
    elif type == "KD":
      pass
    elif type == "QD":
      pass
    else:
      raise Exception("Invalid scene type!")
    return tasks

  @staticmethod
  # 根据测试数据类型标识生成对应的测试用例实例
  def generate_scene(scene_code: str):
    type = scene_code[:2]
    if type == "SD":
      return SD(scene_code=scene_code)
    elif type == "KD":
      return KD(scene_code=scene_code)
    elif type == "QD":
      return QD(scene_code=scene_code)
    else:
      raise Exception("Invalid scene type!")


scene_list: list[Scene] = []
SD1 = Factory.generate_scene("SD1")
# SD2 = Factory.generate_scene("SD2")
scene_list.append(SD1)
# scene_list.append(SD2)

# 控制台输出信息
for scene in scene_list:
  print("Scene : ", scene.scene_code)
  if isinstance(scene, SD):
    print("\tGroups:")
    for group in scene.groups:
      print(
          f"\t- ID:{group}, Rec Coordinates:{scene.groups[group].rec_coor}, Polar Coordinates:{scene.groups[group].polar_coor}, Agents:{scene.groups[group].agents}")
    print("\tAgents:")
    for agent in scene.agents:
      print(
          f"\t- ID:{agent}, Type: {scene.agents[agent].type}, Rec Coordinates: {scene.agents[agent].rec_coor}, Polar Coordinates:{scene.agents[agent].polar_coor}, Group id: {scene.agents[agent].group_id}")
    print("\tTasks:")
    for task in scene.tasks:
      print(
          f"\t- ID:{task}, Type: {scene.tasks[task].type}, Rec Coordinates: {scene.tasks[task].rec_coor}, Polar Coordinates:{scene.tasks[task].polar_coor}, Direction: {scene.tasks[task].direction}, Requirement: {scene.tasks[task].requirement}, Profit: {scene.tasks[task].profit}")
      print(f"\t distance to group 2: {rec_distance(scene.tasks[task].rec_coor,scene.groups[2].rec_coor)}")
  print(f"\t seed: {random.getstate()[1][0]}")
  scene.export_to_json()


# 绘图
for scene in scene_list:
  fig, ax = plt.subplots()
  type = scene.scene_code[0:2]
  code = int(scene.scene_code[2:])
  if type == "SD":
    if code == 1:
      ax.set_xlim(-100, 400)
      ax.set_ylim(-100, 300)
  ax.set_aspect('equal', adjustable='box')
  for group in scene.groups.values():
    circle = plt.Circle(group.rec_coor, group.radius, linestyle='dashed', edgecolor='black', facecolor='none')
    ax.add_artist(circle)
  ax.scatter([agent.rec_coor[0] for agent in scene.agents.values() if agent.type == "MSV"],
             [agent.rec_coor[1] for agent in scene.agents.values() if agent.type == "MSV"], color="r", label="MSV", marker="*")  # 有人艇
  ax.scatter([agent.rec_coor[0] for agent in scene.agents.values() if agent.type == "USV"],
             [agent.rec_coor[1] for agent in scene.agents.values() if agent.type == "USV"], color="r", label="USV", marker=".")  # 无人艇
  ax.scatter([task.rec_coor[0] for task in scene.tasks.values() if task.type == "ST"],
             [task.rec_coor[1] for task in scene.tasks.values() if task.type == "ST"], color="b", label="ST", marker="<")  # ST任务

  ax.legend()
  ax.set_title(scene.scene_code)

  # 设置坐标轴边界
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_linewidth(0.5)  # 设置左边框的线宽
  ax.spines['bottom'].set_linewidth(0.5)  # 设置底部边框的线宽
  # 仅显示 x=0 和 y=0 的刻度线
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')
  # 设置 x=0 和 y=0 的刻度线
  ax.spines['bottom'].set_position(('data', 0))
  ax.spines['left'].set_position(('data', 0))

  plt.xlabel('X')
  plt.ylabel('Y')
  # plt.grid(True)
  plt.show()
