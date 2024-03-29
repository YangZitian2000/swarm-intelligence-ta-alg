from model import *
from settings import *
import random
import math
import json
import os


class Scene:
  def __init__(self, scene_code: str):
    self.scene_code = scene_code
    type = scene_code[0:2]
    if type != "KD":
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
          "polar coordinate": agent.polar_coor
      }
      if isinstance(agent, SD_Agent):
        agent_info["group id"] = agent.group_id
      elif isinstance(agent, KD_Agent):
        agent_info["responsible angle"] = agent.res_angle
      elif isinstance(agent, QD_Agent):
        agent_info["group id"] = agent.group_id
      else:
        raise Exception("Invalid agent type!")
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
        task_info["speed"] = task.speed
        task_info["direction"] = task.direction
      elif isinstance(task, SA_Task):
        task_info["side length"] = task.side_length
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
    type = scene_code[0:2]
    code = int(scene_code[2:])
    groups: dict[int, Group] = {}
    if type == "SD":
      agent_per_group = agent_num_per_group_SD[code - 1]
      group_num = len(agent_per_group)
      agent_id: int = 1
      for i in range(group_num):
        group_id = i + 1
        group = Group(id=group_id)
        for _ in range(agent_per_group[i][0]):
          group.agents.append(agent_id)
          agent_id += 1
        groups[group_id] = group
    elif type == "KD":
      pass
    elif type == "QD":
      group_num = group_num_QD[code - 1]
      agent_num = agent_num_QD[code - 1]
      agent_per_group = int(agent_num / group_num)
      agent_id: int = 1
      for group_id in range(1, group_num + 1):
        group = Group(id=group_id)
        for _ in range(agent_per_group):
          group.agents.append(agent_id)
          agent_id += 1
        if code == 1:
          group.radius = 20
        elif code == 2 or code == 3:
          group.radius = 30
        elif code == 4:
          group.radius = 50
        groups[group_id] = group

    return groups

  @staticmethod
  # 根据场景类型标识生成对应的Agent列表(不包含坐标)
  def generate_agents(scene_code: str):
    type = scene_code[0:2]
    code = int(scene_code[2:])
    if type == "SD":
      agent_per_group = agent_num_per_group_SD[code - 1]
      agents: dict[int, SD_Agent] = {}
      agent_id: int = 1
      for i in range(len(agent_per_group)):
        for _ in range(agent_per_group[i][1]):
          agent = SD_Agent(id=agent_id, type="MSV", group_id=i + 1)
          agents[agent_id] = agent
          agent_id += 1
        for _ in range(agent_per_group[i][0] - agent_per_group[i][1]):
          agent = SD_Agent(id=agent_id, type="USV", group_id=i + 1)
          agents[agent_id] = agent
          agent_id += 1
    elif type == "KD":
      agents: dict[int, KD_Agent] = {}
      if code == 1 or code == 2:
        for i in range(1, 8):
          if i in [2, 3, 5, 6]:
            agent = KD_Agent(id=i, type="D")
          elif i in [4, 7]:
            agent = KD_Agent(id=i, type="B")
          else:
            agent = KD_Agent(id=i, type="HM")
          agents[i] = agent
      else:
        for i in range(1, 10):
          if i in [3, 4, 6, 8]:
            agent = KD_Agent(id=i, type="D")
          elif i in [2, 5, 7, 9]:
            agent = KD_Agent(id=i, type="B")
          else:
            agent = KD_Agent(id=i, type="HM")
          agents[i] = agent
    elif type == "QD":
      agents: dict[int, QD_Agent] = {}
      agent_num = agent_num_QD[code - 1]
      group_num = group_num_QD[code - 1]
      agent_num_per_group = int(agent_num / group_num)
      for agent_id in range(1, agent_num + 1):
        if random.choice([True, False]):
          type = "UAV"
        else:
          type = "USV"
        agent = QD_Agent(id=agent_id, type=type, group_id=(agent_id - 1) // agent_num_per_group + 1)
        agents[agent_id] = agent
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
      tasks: dict[int, AT_Task] = {}
      if code == 1 or code == 3:
        task_num = 24
      elif code == 2:
        task_num = random.choice([5, 6, 7])
      elif code == 4:
        task_num = random.choice([11, 12, 13])
      for task_id in range(1, task_num + 1):
        task = AT_Task(id=task_id, type="AT")
        tasks[task_id] = task
    elif type == "QD":
      tasks: dict[int, SA_Task] = {}
      task_num = task_num_QD[code - 1]
      for task_id in range(1, task_num + 1):
        task = SA_Task(id=task_id, type="SA")
        tasks[task_id] = task
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
      x, y = polar_to_rec((r, a))
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
        x, y = polar_to_rec((r, a))
        group.polar_coor = (r, a)
        group.rec_coor = (x, y)
        self.groups[group_id] = group
    elif code == 3:
      group_coor_list = []
      task_vir_polar_coor = (random.uniform(315, 325), random.uniform(40, 50))  # 创造一个虚拟的任务位置，以此作为生成编组的依据
      task_vir_rec_coor = polar_to_rec(task_vir_polar_coor)
      group_coor_list.append((task_vir_polar_coor[0], task_vir_polar_coor[1] + 180))  # 编组1的位置已有
      while len(group_coor_list) < 4:  # 再生成3个编组坐标, 满足相互距离大于100
        r, a = random.uniform(310, 330), random.uniform(0, 360)
        valid = True
        for cur_coor in group_coor_list:
          if polar_distance((r, a), cur_coor) < 200:
            valid = False
        if valid:
          group_coor_list.append((r, a))
      group_id = 2
      for r, a in group_coor_list[1:]:  # 将3个编组坐标录入
        x, y = polar_to_rec((r, a))
        x_t, y_t = task_vir_rec_coor[0] + x, task_vir_rec_coor[1] + y
        self.groups[group_id].rec_coor = (x_t, y_t)
        self.groups[group_id].polar_coor = rec_to_polar((x_t, y_t))
        group_id += 1
    elif code == 4:
      # 先按SD3的方法生成4个编组的坐标，存入group
      group_coor_list = []
      task_vir_polar_coor = (random.uniform(315, 325), random.uniform(40, 50))  # 创造一个虚拟的任务位置，以此作为生成编组的依据
      task_vir_rec_coor = polar_to_rec(task_vir_polar_coor)
      group_coor_list.append((task_vir_polar_coor[0], task_vir_polar_coor[1] + 180))  # 编组1的位置已有
      while len(group_coor_list) < 6:  # 再生成5个编组坐标, 满足相互距离大于200,前3个用于SD3的编组位置，后2个用于SD2的中心编组位置
        r, a = random.uniform(310, 330), random.uniform(0, 360)
        valid = True
        for cur_coor in group_coor_list:
          if polar_distance((r, a), cur_coor) < 200:
            valid = False
        if valid:
          group_coor_list.append((r, a))
      group_id = 2
      for r, a in group_coor_list[1:4]:  # 将SD3的3个编组坐标录入
        x, y = polar_to_rec((r, a))
        x_t, y_t = task_vir_rec_coor[0] + x, task_vir_rec_coor[1] + y
        self.groups[group_id].rec_coor = (x_t, y_t)
        self.groups[group_id].polar_coor = rec_to_polar((x_t, y_t))
        group_id += 1
      # 再按SD2的方法生成2次3个编组的坐标
      for revo_r, revo_a in group_coor_list[4:]:  # SD2的3个编组绕SD3的任务中心点的公转角度(以group1相对于中心点的角度为0度)
        error_a = 20
        autorota = random.uniform(revo_a + 90 - error_a, revo_a + 90 + error_a)  # SD2的3个编组绕自己的指挥舰的自转角度
        if random.choice([True, False]):  # 两种情况:SD2和SD3的任务同侧/异侧
          autorota += 180
        for g_id in range(group_id, group_id + 3):  # 生成SD2的三个编组的位置
          if g_id == group_id:
            r, a = 0, 0
          elif g_id == group_id + 1:
            r, a = random.uniform(70, 80), random.uniform(10, 30)
          elif g_id == group_id + 2:
            r, a = random.uniform(70, 80), random.uniform(150, 170)
          a = rotate(a, autorota)
          r_t, a_t = polar_sum(polar_sum(task_vir_polar_coor, (revo_r, revo_a)), (r, a))
          self.groups[g_id].polar_coor = (r_t, a_t)
          self.groups[g_id].rec_coor = polar_to_rec((r_t, a_t))
        group_id += 3
    else:  # code==5
      # 先按SD3的方法生成4个编组的坐标，存入group
      group_coor_list = []
      task_vir_polar_coor = (random.uniform(315, 325), random.uniform(40, 50))  # 创造一个虚拟的任务位置，以此作为生成编组的依据
      task_vir_rec_coor = polar_to_rec(task_vir_polar_coor)
      group_coor_list.append((task_vir_polar_coor[0], task_vir_polar_coor[1] + 180))  # 编组1的位置已有
      while len(group_coor_list) < 8:  # 再生成5个编组坐标, 满足相互距离大于200,前3个用于SD3的编组位置，后2个用于SD2的中心编组位置
        r, a = random.uniform(310, 330), random.uniform(0, 360)
        valid = True
        for cur_coor in group_coor_list:
          if polar_distance((r, a), cur_coor) < 150:
            valid = False
        if valid:
          group_coor_list.append((r, a))
      group_id = 2
      for r, a in group_coor_list[1:4]:  # 将SD3的3个编组坐标录入
        x, y = polar_to_rec((r, a))
        x_t, y_t = task_vir_rec_coor[0] + x, task_vir_rec_coor[1] + y
        self.groups[group_id].rec_coor = (x_t, y_t)
        self.groups[group_id].polar_coor = rec_to_polar((x_t, y_t))
        group_id += 1
      # 再按SD2的方法生成2次3个编组的坐标
      for revo_r, revo_a in group_coor_list[4:]:  # SD2的3个编组绕SD3的任务中心点的公转角度(以group1相对于中心点的角度为0度)
        error_a = 20
        autorota = random.uniform(revo_a + 90 - error_a, revo_a + 90 + error_a)  # SD2的3个编组绕自己的指挥舰的自转角度
        if random.choice([True, True, False]):  # 两种情况:SD2和SD3的任务同侧/异侧
          autorota += 180
        for g_id in range(group_id, group_id + 3):  # 生成SD2的三个编组的位置
          if g_id == group_id:
            r, a = 0, 0
          elif g_id == group_id + 1:
            r, a = random.uniform(70, 80), random.uniform(10, 30)
          elif g_id == group_id + 2:
            r, a = random.uniform(70, 80), random.uniform(150, 170)
          a = rotate(a, autorota)
          r_t, a_t = polar_sum(polar_sum(task_vir_polar_coor, (revo_r, revo_a)), (r, a))
          self.groups[g_id].polar_coor = (r_t, a_t)
          self.groups[g_id].rec_coor = polar_to_rec((r_t, a_t))
        group_id += 3

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
          x1, y1 = polar_to_rec((r1, a1))
          x2, y2 = polar_to_rec((r2, a2))
          self.tasks[1].polar_coor = (r1, a1)
          self.tasks[1].rec_coor = (x1, y1)
          self.tasks[2].polar_coor = (r2, a2)
          self.tasks[2].rec_coor = (x2, y2)
          break
    elif code == 2:
      r1, a1 = random.uniform(310, 330), random.uniform(80, 90)  # 生成任务1坐标
      x1, y1 = polar_to_rec((r1, a1))
      self.tasks[1].polar_coor = (r1, a1)
      self.tasks[1].rec_coor = (x1, y1)
      while True:  # 生成任务2坐标，满足任务1和2距离在50~60
        r2, a2 = random.uniform(310, 330), random.uniform(90, 100)
        dis = polar_distance((r1, a1), (r2, a2))
        if dis >= 50 and dis <= 60:
          x2, y2 = polar_to_rec((r2, a2))
          self.tasks[2].polar_coor = (r2, a2)
          self.tasks[2].rec_coor = (x2, y2)
          break
    elif code == 3:
      r, a = random.uniform(315, 325), random.uniform(40, 50)
      x, y = polar_to_rec((r, a))
      width, length = [random.uniform(10, 15) for _ in range(2)]  # 目标矩阵的长和宽
      for t_id in range(1, 5):
        if t_id == 1:
          x_t, y_t = x + length / 2, y + width / 2
        elif t_id == 2:
          x_t, y_t = x + length / 2, y - width / 2
        elif t_id == 3:
          x_t, y_t = x - length / 2, y - width / 2
        elif t_id == 4:
          x_t, y_t = x - length / 2, y + width / 2
        self.tasks[t_id].rec_coor = (x_t, y_t)
        self.tasks[t_id].polar_coor = rec_to_polar((x_t, y_t))
    elif code == 4:
      r, a = random.uniform(315, 325), random.uniform(40, 50)  # 生成SD3对应的4个任务
      x, y = polar_to_rec((r, a))
      width, length = [random.uniform(10, 15) for _ in range(2)]  # 目标矩阵的长和宽
      for t_id in range(1, 5):
        if t_id == 1:
          x_t, y_t = x + length / 2, y + width / 2
        elif t_id == 2:
          x_t, y_t = x + length / 2, y - width / 2
        elif t_id == 3:
          x_t, y_t = x - length / 2, y - width / 2
        elif t_id == 4:
          x_t, y_t = x - length / 2, y + width / 2
        self.tasks[t_id].rec_coor = (x_t, y_t)
        self.tasks[t_id].polar_coor = rec_to_polar((x_t, y_t))
      group_id = 5
      task_id = 5
      for _ in range(2):  # 生成2个SD2对应的4个任务
        r1, a1 = rec_to_polar(
            tuple(x - y for x, y in zip(self.groups[group_id + 1].rec_coor, self.groups[group_id].rec_coor)))
        r2, a2 = rec_to_polar(
            tuple(x - y for x, y in zip(self.groups[group_id + 2].rec_coor, self.groups[group_id].rec_coor)))
        task_angle = (a1 + a2) / 2 if a2 > a1 else (a1 + a2 + 360) / 2 % 360
        while True:  # 生成第1个任务的坐标，与其他所有编组距离不小于200, 与其他所有任务距离不小于50
          r_t1, a_t1 = random.uniform(310, 330), random.uniform(task_angle, task_angle + 10)
          x_t1, y_t1 = polar_to_rec(polar_sum(self.groups[group_id].polar_coor, (r_t1, a_t1)))
          if min([rec_distance((x_t1, y_t1), group.rec_coor) for group in self.groups.values()]) > 200 \
                  and min([rec_distance((x_t1, y_t1), self.tasks[task].rec_coor) for task in range(1, task_id)]) > 20:
            self.tasks[task_id].rec_coor = (x_t1, y_t1)
            self.tasks[task_id].polar_coor = rec_to_polar((x_t1, y_t1))
            task_id += 1
            break
        while True:  # 生成第2个任务坐标，满足第1、2个任务距离在50~60，且与其他所有编组距离不小于200, 与其他所有任务距离不小于20
          r_t2, a_t2 = random.uniform(300, 330), random.uniform(task_angle - 10, task_angle)
          x_t2, y_t2 = polar_to_rec(polar_sum(self.groups[group_id].polar_coor, (r_t2, a_t2)))
          dis = rec_distance(self.tasks[task_id - 1].rec_coor, (x_t2, y_t2))
          if dis >= 50 and dis <= 60 and \
             min([rec_distance((x_t2, y_t2), group.rec_coor) for group in self.groups.values()]) > 200 \
             and min([rec_distance((x_t2, y_t2), self.tasks[task].rec_coor) for task in range(1, task_id)]) > 20:
            self.tasks[task_id].rec_coor = (x_t2, y_t2)
            self.tasks[task_id].polar_coor = rec_to_polar((x_t2, y_t2))
            task_id += 1
            break
        group_id += 3
    else:  # code==5
      r, a = random.uniform(315, 325), random.uniform(40, 50)  # 生成SD3对应的4个任务
      x, y = polar_to_rec((r, a))
      width, length = [random.uniform(10, 15) for _ in range(2)]  # 目标矩阵的长和宽
      for t_id in range(1, 5):
        if t_id == 1:
          x_t, y_t = x + length / 2, y + width / 2
        elif t_id == 2:
          x_t, y_t = x + length / 2, y - width / 2
        elif t_id == 3:
          x_t, y_t = x - length / 2, y - width / 2
        elif t_id == 4:
          x_t, y_t = x - length / 2, y + width / 2
        self.tasks[t_id].rec_coor = (x_t, y_t)
        self.tasks[t_id].polar_coor = rec_to_polar((x_t, y_t))
      group_id = 5
      task_id = 5
      for _ in range(4):  # 生成4个SD2对应的8个任务
        r1, a1 = rec_to_polar(
            tuple(x - y for x, y in zip(self.groups[group_id + 1].rec_coor, self.groups[group_id].rec_coor)))
        r2, a2 = rec_to_polar(
            tuple(x - y for x, y in zip(self.groups[group_id + 2].rec_coor, self.groups[group_id].rec_coor)))
        task_angle = (a1 + a2) / 2 if a2 > a1 else (a1 + a2 + 360) / 2 % 360
        while True:  # 生成第1个任务的坐标，与其他所有编组距离不小于200
          r_t1, a_t1 = random.uniform(310, 330), random.uniform(task_angle, task_angle + 10)
          x_t1, y_t1 = polar_to_rec(polar_sum(self.groups[group_id].polar_coor, (r_t1, a_t1)))
          if min([rec_distance((x_t1, y_t1), group.rec_coor) for group in self.groups.values()]) > 200 and \
             min([rec_distance((x_t1, y_t1), self.tasks[task].rec_coor) for task in range(1, task_id)]) > 20:
            self.tasks[task_id].rec_coor = (x_t1, y_t1)
            self.tasks[task_id].polar_coor = rec_to_polar((x_t1, y_t1))
            task_id += 1
            break
        while True:  # 生成第2个任务坐标，满足第1、2个任务距离在50~60，且与其他所有编组距离不小于200
          r_t2, a_t2 = random.uniform(300, 330), random.uniform(task_angle - 10, task_angle)
          x_t2, y_t2 = polar_to_rec(polar_sum(self.groups[group_id].polar_coor, (r_t2, a_t2)))
          dis = rec_distance(self.tasks[task_id - 1].rec_coor, (x_t2, y_t2))
          if dis >= 50 and dis <= 60 and \
             min([rec_distance((x_t2, y_t2), group.rec_coor) for group in self.groups.values()]) > 200 and \
             min([rec_distance((x_t2, y_t2), self.tasks[task].rec_coor) for task in range(1, task_id)]) > 20:
            self.tasks[task_id].rec_coor = (x_t2, y_t2)
            self.tasks[task_id].polar_coor = rec_to_polar((x_t2, y_t2))
            task_id += 1
            break
        group_id += 3
      for i in range(2):  # 生成剩余的8个任务，4个为一组呈矩形排布
        if i:
          r, a = random.uniform(500, 600), random.uniform(120, 130)
        else:
          r, a = random.uniform(300, 330), random.uniform(210, 220)
        x, y = polar_to_rec((r, a))
        width, length = [random.uniform(15, 20) for _ in range(2)]  # 目标矩阵的长和宽
        for t_id in range(4):
          if t_id == 0:
            x_t, y_t = x + length / 2, y + width / 2
          elif t_id == 1:
            x_t, y_t = x + length / 2, y - width / 2
          elif t_id == 2:
            x_t, y_t = x - length / 2, y - width / 2
          elif t_id == 3:
            x_t, y_t = x - length / 2, y + width / 2
          self.tasks[task_id + t_id].rec_coor = (x_t, y_t)
          self.tasks[task_id + t_id].polar_coor = rec_to_polar((x_t, y_t))
        task_id += 4

  def generate_agent_coordinate_SD(self):
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
        x, y = polar_to_rec((r, a))
        self.agents[agent_id].rec_coor = (group.rec_coor[0] + x, group.rec_coor[1] + y)
        self.agents[agent_id].polar_coor = rec_to_polar(self.agents[agent_id].rec_coor)


# 打击AT场景
class KD(Scene):
  def __init__(self, scene_code):
    code = int(scene_code[2:])
    if code > 4 or code < 1:
      raise Exception("Invalid KD scene code!")
    super().__init__(scene_code)
    # 确定Agent坐标
    self.generate_agent_coordinate_KD()
    # 确定任务坐标
    self.generate_task_coordinate_KD()

  def generate_agent_coordinate_KD(self):
    code = int(self.scene_code[2:])
    if code == 1:
      r, a = random.uniform(10 / math.sqrt(2), 11 / math.sqrt(2)), random.uniform(285, 345)
      for a_id in range(2, 6):
        self.agents[a_id].polar_coor = (r, a)
        self.agents[a_id].rec_coor = polar_to_rec((r, a))
        self.agents[a_id].res_angle = [rotate(a, -75), rotate(a, 75)]
        a = rotate(a, 90)
      self.agents[6].rec_coor = (70, 12)
      self.agents[7].rec_coor = (70, -12)
      self.agents[6].polar_coor = rec_to_polar((70, 12))
      self.agents[7].polar_coor = rec_to_polar((70, -12))
      self.agents[6].res_angle = self.agents[7].res_angle = [0, 180]
    elif code == 3:
      r, a = random.uniform(10, 11), random.uniform(310, 350)
      for a_id in range(2, 8):
        self.agents[a_id].polar_coor = (r, a)
        self.agents[a_id].rec_coor = polar_to_rec((r, a))
        self.agents[a_id].res_angle = [rotate(a, -85), rotate(a, 85)]
        a = rotate(a, 60)
      self.agents[8].rec_coor = (70, 12)
      self.agents[9].rec_coor = (70, -12)
      self.agents[8].polar_coor = rec_to_polar((70, 12))
      self.agents[9].polar_coor = rec_to_polar((70, -12))
      self.agents[8].res_angle = self.agents[9].res_angle = [0, 180]

  def generate_task_coordinate_KD(self):
    code = int(self.scene_code[2:])
    if code == 1 or code == 2:
      x, y = (98, 1.25)
      task_rec_coors = []
      for i in range(4):
        for j in range(6):
          task_rec_coors.append((x + i * 0.3, y - j * 0.5))
      task_num = len(self.tasks)
      task_rec_coors = random.sample(task_rec_coors, task_num)
      if code == 2:
        task_rec_coors = [(coor[0] - 70, coor[1]) for coor in task_rec_coors]
      for t_id in range(1, task_num + 1):
        self.tasks[t_id].rec_coor = task_rec_coors[t_id - 1]
        self.tasks[t_id].polar_coor = rec_to_polar(task_rec_coors[t_id - 1])
        self.tasks[t_id].direction = 270
    elif code == 3 or code == 4:
      x1, y1 = (-98, 0.25)
      x2, y2 = (98, 0.75)
      left_tasks = []
      right_tasks = []
      for i in range(4):
        for j in range(2):
          left_tasks.append((x1 - i * 0.3, y1 - j * 0.5))
      for i in range(4):
        for j in range(4):
          right_tasks.append((x2 + i * 0.3, y2 - j * 0.5))
      task_num = len(self.tasks)
      right_tasks = random.sample(right_tasks, task_num - 8)
      if code == 4:
        left_tasks = [(coor[0] + 70, coor[1]) for coor in left_tasks]
        right_tasks = [(coor[0] - 70, coor[1]) for coor in right_tasks]
      for t_id in range(1, task_num + 1):
        if t_id <= 8:
          self.tasks[t_id].rec_coor = left_tasks[t_id - 1]
          self.tasks[t_id].direction = 90
        else:
          self.tasks[t_id].rec_coor = right_tasks[t_id - 9]
          self.tasks[t_id].direction = 270
        self.tasks[t_id].polar_coor = rec_to_polar(self.tasks[t_id].rec_coor)


# 区域搜索场景
class QD(Scene):
  def __init__(self, scene_code):
    code = int(scene_code[2:])
    if code > 4 or code < 1:
      raise Exception("Invalid QD scene code!")
    super().__init__(scene_code)
    # 确定组的坐标
    self.generate_group_coordinate_QD()
    # 确定Agent坐标
    self.generate_agent_coordinate_QD()
    # 确定任务坐标
    self.generate_task_coordinate_QD()

  def generate_group_coordinate_QD(self):
    code = int(self.scene_code[2:])
    group_num = group_num_QD[code - 1]
    group_coors = [(0, 0)]
    if code == 1:
      g_dis = 50
    elif code == 2 or code == 3:
      g_dis = 70
    elif code == 4:
      g_dis = 110
    while len(group_coors) < group_num:
      if code == 1:
        x, y = random.uniform(-50, 50), random.uniform(-100, 100)
      elif code == 2:
        x, y = random.uniform(-100, 100), random.uniform(-200, 200)
      elif code == 3:
        x, y = random.uniform(-100, 100), random.uniform(-200, 200)
      elif code == 4:
        x, y = random.uniform(-150, 150), random.uniform(-300, 300)
      valid = True
      for coor in group_coors:
        if rec_distance((x, y), coor) < g_dis:
          valid = False
          break
      if valid:
        group_coors.append((x, y))
    for i, coor in enumerate(group_coors):
      if i == 0:
        continue
      self.groups[i + 1].rec_coor = coor
      self.groups[i + 1].polar_coor = rec_to_polar(coor)

  def generate_task_coordinate_QD(self):
    code = int(self.scene_code[2:])
    task_num = task_num_QD[code - 1]
    for task_id in range(1, task_num + 1):
      side_length = random.choice([80, 100, 100, 150])
      while True:
        if code == 1:
          x, y = random.uniform(-200, 200), random.uniform(-200, 200)
        elif code == 2:
          x, y = random.uniform(-300, 300), random.uniform(-300, 300)
        elif code == 3:
          x, y = random.uniform(-400, 400), random.uniform(-400, 400)
        elif code == 4:
          x, y = random.uniform(-500, 500), random.uniform(-500, 500)
        task_cover_agent = False
        for a_id in self.agents:
          if self.task_cover_agent(task_coor=(x, y), side_length=side_length, agent_id=a_id):
            task_cover_agent = True
            break
        task_cover_task = False
        for t_id in range(1, task_id):
          if self.task_cover_task(task_coor=(x, y), side_length=side_length, task_id=t_id):
            task_cover_task = True
            break
        if not task_cover_agent and not task_cover_task:
          self.tasks[task_id].side_length = side_length
          self.tasks[task_id].rec_coor = (x, y)
          self.tasks[task_id].polar_coor = rec_to_polar((x, y))
          break

  def generate_agent_coordinate_QD(self):
    code = int(self.scene_code[2:])
    if code == 1:
      a_dis = 5
    elif code == 2 or code == 3:
      a_dis = 8
    elif code == 4:
      a_dis = 10
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
          if polar_distance((r, a), cur_agent_polar_coor) < a_dis:
            valid = False
        if valid:
          agent_polar_coors.append((r, a))
      for i, agent_id in enumerate(group.agents):  # 将可行的一组坐标作为该编组中Agent的坐标
        if agent_id == 1:
          continue
        r, a = agent_polar_coors[i]
        x, y = polar_to_rec((r, a))
        self.agents[agent_id].rec_coor = (group.rec_coor[0] + x, group.rec_coor[1] + y)
        self.agents[agent_id].polar_coor = rec_to_polar(self.agents[agent_id].rec_coor)

  def task_cover_agent(self, task_coor, side_length, agent_id):
    agent: QD_Agent = self.agents[agent_id]
    x_l, x_r = task_coor[0] - side_length / 2, task_coor[0] + side_length / 2
    y_d, y_u = task_coor[1] - side_length / 2, task_coor[1] + side_length / 2
    if x_l <= agent.rec_coor[0] and agent.rec_coor[0] <= x_r and y_d <= agent.rec_coor[1] and agent.rec_coor[1] <= y_u:
      return True
    else:
      return False

  def task_cover_task(self, task_coor, side_length, task_id):
    x_l_1, x_r_1 = task_coor[0] - side_length / 2, task_coor[0] + side_length / 2
    x_l_2, x_r_2 = self.tasks[task_id].rec_coor[0] - self.tasks[task_id].side_length / \
        2, self.tasks[task_id].rec_coor[0] + self.tasks[task_id].side_length / 2
    y_d_1, y_u_1 = task_coor[1] - side_length / 2, task_coor[1] + side_length / 2
    y_d_2, y_u_2 = self.tasks[task_id].rec_coor[1] - self.tasks[task_id].side_length / \
        2, self.tasks[task_id].rec_coor[1] + self.tasks[task_id].side_length / 2
    if x_r_1 < x_l_2 or x_l_1 > x_r_2 or y_u_1 < y_d_2 or y_d_1 > y_u_2:
      return False
    else:
      return True


# 极坐标相加
def polar_sum(polar1, polar2):
  x1, y1 = polar_to_rec(polar1)
  x2, y2 = polar_to_rec(polar2)
  x_sum = x1 + x2
  y_sum = y1 + y2
  r_sum, theta_sum = rec_to_polar((x_sum, y_sum))
  return (r_sum, theta_sum)


# 将angle1旋转angle2角度
def rotate(angle1: float, angle2: float):
  return (angle1 + angle2) % 360


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


# 极坐标转直角坐标
def polar_to_rec(point):
  r, theta = point
  x = r * math.cos(angle_to_radian(theta))
  y = r * math.sin(angle_to_radian(theta))
  return (x, y)


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
