from model import *
from settings import *
import random
import math
import json
import os


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


class Factory:
  @staticmethod
  # 根据场景类型标识生成对应的编组列表(不包含坐标)
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


# 打击ST场景
class SD(Scene):
  def __init__(self, scene_code):
    code = int(scene_code[2:])
    if code > 5 or code < 1:
      raise Exception("Invalid SD scene code!")
    super().__init__(scene_code)
    if code == 3:
      # 确定任务坐标
      self.generate_task_coordinate_SD()
      # 确定组的坐标
      self.generate_group_coordinate_SD()
    else:
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
      group_2 = self.groups[2]  # 生成Group 2位置
      r, a = random.uniform(180, 220), random.uniform(0, 30)
      x, y = r * math.cos(angle_to_radian(a)), r * math.sin(angle_to_radian(a))
      group_2.polar_coor = (r, a)
      group_2.rec_coor = (x, y)
      self.groups[2] = group_2
    elif code == 2:
      for group_id in range(2, 4):  # 生成Group 2,3位置
        group = self.groups[group_id]
        if group_id == 2:
          r, a = random.uniform(75, 90), random.uniform(10, 30)
        elif group_id == 3:
          r, a = random.uniform(75, 90), random.uniform(150, 170)
        x, y = r * math.cos(angle_to_radian(a)), r * math.sin(angle_to_radian(a))
        group.polar_coor = (r, a)
        group.rec_coor = (x, y)
        self.groups[group_id] = group
    elif code == 3:
      group_coor_list = []
      group_1_coor = self.tasks[1].polar_coor
      group_coor_list.append((group_1_coor[0], group_1_coor[1] + 180))  # 编组1的位置已有
      while len(group_coor_list) < 4:  # 再生成3个编组坐标, 满足相互距离大于100
        r, a = random.uniform(310, 330), random.uniform(0, 360)
        valid = True
        for cur_coor in group_coor_list:
          if polar_distance((r, a), cur_coor) < 200:
            valid = False
        if valid:
          group_coor_list.append((r, a))
      group_coor_list.pop(0)
      task_rec = self.tasks[1].rec_coor
      group_id = 2
      for r, a in group_coor_list:  # 将3个编组坐标录入
        x, y = r * math.cos(angle_to_radian(a)), r * math.sin(angle_to_radian(a))
        x_t, y_t = task_rec[0] + x, task_rec[1] + y
        self.groups[group_id].rec_coor = (x_t, y_t)
        self.groups[group_id].polar_coor = rec_to_polar((x_t, y_t))
        group_id += 1
    elif code == 4:
      pass
    else:
      pass

  def generate_task_coordinate_SD(self):
    min_range, max_range = 270, 330
    code = int(self.scene_code[2:])
    if code == 1:
      avaliable_angle = []
      for angle in range(0, 181):  # 寻找所有在满足距离G1 min~max range条件下，能够满足距离G2也在min~max range范围的角度
        distance_min = polar_distance((min_range, angle), self.groups[2].polar_coor)
        distance_max = polar_distance((max_range, angle), self.groups[2].polar_coor)
        if min_range < distance_min and distance_min < max_range or \
                min_range < distance_max and distance_max < max_range:
          avaliable_angle.append(angle)
      while True:  # 任选2个可行的角度生成任务，并满足任务间距离大于50
        [a1, a2] = random.sample(avaliable_angle, 2)
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
      r1, a1 = random.uniform(310, 330), random.uniform(80, 90)  # 生成任务1坐标
      x1, y1 = r1 * math.cos(angle_to_radian(a1)), r1 * math.sin(angle_to_radian(a1))
      self.tasks[1].polar_coor = (r1, a1)
      self.tasks[1].rec_coor = (x1, y1)
      while True:  # 生成任务2坐标，满足任务1和2距离在50~60
        r2, a2 = random.uniform(310, 330), random.uniform(90, 100)
        p_dis = polar_distance((r1, a1), (r2, a2))
        if p_dis >= 50 and p_dis <= 60:
          x2, y2 = r2 * math.cos(angle_to_radian(a2)), r2 * math.sin(angle_to_radian(a2))
          self.tasks[2].polar_coor = (r2, a2)
          self.tasks[2].rec_coor = (x2, y2)
          break
    elif code == 3:
      r, a = random.uniform(315, 325), random.uniform(40, 50)
      x, y = r * math.cos(angle_to_radian(a)), r * math.sin(angle_to_radian(a))
      width, length = [random.uniform(10, 15) for _ in range(2)]  # 目标矩阵的长和宽
      x1, y1 = x + length / 2, y + width / 2
      x2, y2 = x + length / 2, y - width / 2
      x3, y3 = x - length / 2, y - width / 2
      x4, y4 = x - length / 2, y + width / 2
      self.tasks[1].rec_coor = (x1, y1)
      self.tasks[2].rec_coor = (x2, y2)
      self.tasks[3].rec_coor = (x3, y3)
      self.tasks[4].rec_coor = (x4, y4)
      self.tasks[1].polar_coor = rec_to_polar((x1, y1))
      self.tasks[2].polar_coor = rec_to_polar((x2, y2))
      self.tasks[3].polar_coor = rec_to_polar((x3, y3))
      self.tasks[4].polar_coor = rec_to_polar((x4, y4))
    elif code == 4:
      pass
    else:
      pass

  def generate_agent_coordinate_SD(self):
    code = int(self.scene_code[2:])
    if code == 1 or code == 2 or code == 3:
      for group in self.groups.values():
        agent_num_in_group = len(group.agents)
        agent_polar_coors = []
        if group.id == 1:  # 如果是第一组，则包含指挥舰位于原点
          agent_polar_coors.append((0, 0))
        while len(agent_polar_coors) < agent_num_in_group:  # 在编组半径范围内随机生成坐标，保证所有坐标之间距离大于5
          r = random.uniform(0, group.radius)
          a = random.uniform(0, 360)
          valid = True
          for cur_agent_polar_coor in agent_polar_coors:
            if polar_distance((r, a), cur_agent_polar_coor) < 5:
              valid = False
          if valid:
            agent_polar_coors.append((r, a))
        for i, agent_id in enumerate(group.agents):  # 将可行的一组坐标作为该编组中Agent的坐标
          if agent_id == 1:
            continue
          r, a = agent_polar_coors[i]
          x, y = r * math.cos(angle_to_radian(a)), r * math.sin(angle_to_radian(a))
          self.agents[agent_id].rec_coor = (group.rec_coor[0] + x, group.rec_coor[1] + y)
          self.agents[agent_id].polar_coor = rec_to_polar(self.agents[agent_id].rec_coor)
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


# 角度转弧度
def angle_to_radian(angle: float):
  return (450 - angle) % 360 / 180 * math.pi


# 弧度转角度
def radian_to_angle(radian: float):
  return (450 - radian / math.pi * 180) % 360


# 直角坐标转极坐标
def rec_to_polar(point):
  x, y = point
  r = math.sqrt(x**2 + y**2)
  theta = math.atan2(y, x)
  a = radian_to_angle(theta)
  return (r, a)


# 直角坐标距离
def rec_distance(point1, point2):
  x1, y1 = point1
  x2, y2 = point2
  distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return distance


# 极坐标距离
def polar_distance(polar1, polar2):
  r1, theta1 = polar1
  r2, theta2 = polar2
  theta1 = angle_to_radian(theta1)
  theta2 = angle_to_radian(theta2)
  distance = math.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * math.cos(theta2 - theta1))
  return distance
