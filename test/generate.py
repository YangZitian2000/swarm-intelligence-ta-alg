from scene import *
import matplotlib.pyplot as plt

random.seed(0)

scene_list: list[Scene] = []
code = 3

for i in range(1, code + 1):
  scene_list.append(Factory.generate_scene("SD" + str(i)))


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
  fig, ax = plt.subplots(figsize=(12, 6))
  type = scene.scene_code[0:2]
  code = int(scene.scene_code[2:])
  if type == "SD":
    if code == 1:
      ax.set_xlim(-100, 400)
      ax.set_ylim(-100, 300)
    elif code == 2:
      ax.set_xlim(-100, 400)
      ax.set_ylim(-150, 150)
    elif code == 3:
      ax.set_xlim(-100, 600)
      ax.set_ylim(-100, 500)

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
