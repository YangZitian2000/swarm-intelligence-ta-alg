from scene import *
import matplotlib.pyplot as plt

random.seed(0)

scene_list: list[Scene] = []

for i in range(1, 6):
  scene_list.append(Factory.generate_scene("SD" + str(i)))

for i in range(1, 5):
  scene = Factory.generate_scene("KD" + str(i))
  if i == 2:
    scene.agents = scene_list[-1].agents
  elif i == 4:
    scene.agents = scene_list[-1].agents
  scene_list.append(scene)

for i in range(1, 5):
  scene_list.append(Factory.generate_scene("QD" + str(i)))

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
  elif isinstance(scene, QD):
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
          f"\t- ID:{task}, Type: {scene.tasks[task].type}, Rec Coordinates: {scene.tasks[task].rec_coor}, Polar Coordinates:{scene.tasks[task].polar_coor}, Side length: {scene.tasks[task].side_length}")
  scene.export_to_json()


# 绘图
for scene in scene_list:
  fig, ax = plt.subplots(figsize=(24, 18))
  code = int(scene.scene_code[2:])
  if isinstance(scene, SD):
    if code == 1:
      ax.set_xlim(-100, 400)
      ax.set_ylim(-100, 300)
    elif code == 2:
      ax.set_xlim(-100, 400)
      ax.set_ylim(-150, 150)
    elif code == 3:
      ax.set_xlim(-100, 600)
      ax.set_ylim(-100, 500)
    elif code == 4:
      ax.set_xlim(-100, 900)
      ax.set_ylim(-150, 600)
    elif code == 5:
      ax.set_xlim(-250, 800)
      ax.set_ylim(-450, 800)
    for group in scene.groups.values():
      circle = plt.Circle(group.rec_coor, group.radius, linestyle='dashed', edgecolor='black', facecolor='none')
      ax.add_artist(circle)
    msv = [agent for agent in scene.agents.values() if agent.type == "MSV"]
    msv_x = [agent.rec_coor[0] for agent in msv]
    msv_y = [agent.rec_coor[1] for agent in msv]
    ax.scatter(msv_x, msv_y, color="r", label="MSV", marker="*")  # 有人艇
    usv = [agent for agent in scene.agents.values() if agent.type == "USV"]
    usv_x = [agent.rec_coor[0] for agent in usv]
    usv_y = [agent.rec_coor[1] for agent in usv]
    ax.scatter(usv_x, usv_y, color="r", label="USV", marker=".")  # 无人艇
    st = [task for task in scene.tasks.values() if task.type == "ST"]
    st_x = [task.rec_coor[0] for task in st]
    st_y = [task.rec_coor[1] for task in st]
    ax.scatter(st_x, st_y, color="b", label="ST", marker="<")  # ST任务
    for agent in scene.agents.values():
      ax.text(agent.rec_coor[0] + 2, agent.rec_coor[1] + 4, agent.id, ha='center', va='center', fontsize=8)
    for task in scene.tasks.values():
      ax.text(task.rec_coor[0] + 2, task.rec_coor[1] + 4, task.id, ha='center', va='center', fontsize=8)
  elif isinstance(scene, KD):
    if code == 1:
      ax.set_xlim(-20, 120)
      ax.set_ylim(-20, 20)
    elif code == 2:
      ax.set_xlim(-20, 80)
      ax.set_ylim(-20, 20)
    elif code == 3:
      ax.set_xlim(-120, 120)
      ax.set_ylim(-20, 20)
    elif code == 4:
      ax.set_xlim(-40, 80)
      ax.set_ylim(-20, 20)
    ax.scatter([agent.rec_coor[0] for agent in scene.agents.values() if agent.type == "HM"],
               [agent.rec_coor[1] for agent in scene.agents.values() if agent.type == "HM"], color="r", label="HM", marker="*")  # HM
    ax.scatter([agent.rec_coor[0] for agent in scene.agents.values() if agent.type == "D"],
               [agent.rec_coor[1] for agent in scene.agents.values() if agent.type == "D"], color="orange", label="D", marker="d")  # D
    ax.scatter([agent.rec_coor[0] for agent in scene.agents.values() if agent.type == "B"],
               [agent.rec_coor[1] for agent in scene.agents.values() if agent.type == "B"], color="green", label="B", marker="d")  # B
    ax.scatter([task.rec_coor[0] for task in scene.tasks.values() if task.type == "AT"],
               [task.rec_coor[1] for task in scene.tasks.values() if task.type == "AT"], color="b", label="AT", marker="^")  # AT任务
    for agent in scene.agents.values():
      ax.text(agent.rec_coor[0] + 1, agent.rec_coor[1] + 2, agent.id, ha='center', va='center', fontsize=8)
    for task in scene.tasks.values():
      ax.text(task.rec_coor[0] + 0.1, task.rec_coor[1] + 0.2, task.id, ha='center', va='center', fontsize=8)
  elif isinstance(scene, QD):
    if code == 1:
      ax.set_xlim(-250, 250)
      ax.set_ylim(-250, 250)
    elif code == 2:
      ax.set_xlim(-350, 350)
      ax.set_ylim(-350, 350)
    elif code == 3:
      ax.set_xlim(-500, 500)
      ax.set_ylim(-500, 500)
    elif code == 4:
      ax.set_xlim(-600, 600)
      ax.set_ylim(-600, 600)
    for group in scene.groups.values():
      circle = plt.Circle(
          group.rec_coor,
          group.radius,
          linestyle='dashed',
          edgecolor='black',
          facecolor='none')
      ax.add_artist(circle)
    ax.scatter([agent.rec_coor[0] for agent in scene.agents.values() if agent.type == "UAV"],
               [agent.rec_coor[1] for agent in scene.agents.values() if agent.type == "UAV"], color="r", label="UAV", marker=".")  # 无人机
    ax.scatter([agent.rec_coor[0] for agent in scene.agents.values() if agent.type == "USV"],
               [agent.rec_coor[1] for agent in scene.agents.values() if agent.type == "USV"], color="r", label="USV", marker="*")  # 无人艇

    for task in scene.tasks.values():
      square = plt.Rectangle(
          (task.rec_coor[0] - task.side_length / 2,
           task.rec_coor[1] - task.side_length / 2),
          task.side_length,
          task.side_length,
          linestyle='dashed',
          edgecolor='blue',
          facecolor='none')
      ax.add_artist(square)
    ax.scatter([task.rec_coor[0] for task in scene.tasks.values() if task.type == "SA"],
               [task.rec_coor[1] for task in scene.tasks.values() if task.type == "SA"], color="b", label="SA", marker="x")  # SA任务
    for agent in scene.agents.values():
      ax.text(agent.rec_coor[0] + 2, agent.rec_coor[1] + 4, agent.id, ha='center', va='center', fontsize=8)
    for task in scene.tasks.values():
      ax.text(task.rec_coor[0] + 5, task.rec_coor[1] + 10, task.id, ha='center', va='center', fontsize=8)
  ax.set_aspect('equal', adjustable='box')
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
  plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '\\\\' + scene.scene_code + '.png', dpi=200)
  plt.show()
