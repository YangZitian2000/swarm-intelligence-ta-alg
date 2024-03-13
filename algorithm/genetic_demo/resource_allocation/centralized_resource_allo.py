import math

import numpy as np
import copy
import matplotlib.pyplot as plt
import time
from resource_allo_problem_setting import *


class Centralized_Agent:
  def __init__(self, task_num, agent_num, init_popu_num):
    # 初始时，选择任务列表均为0， 表示没有选择任何任务
    self.init_popu_num = init_popu_num
    self.agent_num = agent_num
    self.task_num = task_num
    # 为保证一致，完整解种群的大小需要参考分布式的情况
    self.intact_population_size = init_popu_num * SHARE_NUM  # pow(SHARE_NUM, agent_num - 1)
    # 解的形式为intact_population_size × agent_num × task_num的数组，初始时大小为init_popu_num
    # 每个agent的解用一个bool数组表示，数组长度为task_num的长度，
    # 一个数组元素代表一个基因，表示对该任务分配资源个数
    # agent_num个agent的解组成一个完整解
    self.intact_population = np.zeros((self.init_popu_num, agent_num, task_num), dtype=np.int8)
    self.intact_population_fit = []

  def get_init_population(self):
    # 为扩大搜索范围，尽量生成不相同的初始解
    # 根据自身能力约束形成任务选择方案，即认为是遗传算法的初始可行解
    i = 0
    while i < self.init_popu_num:
      popu = np.zeros((self.agent_num, self.task_num), dtype=np.int8)
      for j in range(self.agent_num):
        # 随机选择有效的任务index
        choose_task = np.random.randint(0, 2, self.task_num, dtype=np.int8)
        self.make_local_choose_valid(choose_task)
        # 在选择的任务里平均分配资源
        if np.sum(choose_task) > 0:
          popu[j] = choose_task * RESOURCES[j] / np.sum(choose_task)

      # 只要满足本地约束就是合法解，如果对每个任务的资源过多或不足，可以在fitness计算时进行处理
      self.intact_population[i] = popu
      i += 1

    # 计算种群fitness备用
    self.intact_population_fit = self.cal_fitness(self.intact_population)

  def selection(self, population, fitness):
    # 按照适应度选择（排序）种群
    new_popu = np.zeros_like(population, dtype=np.int8)  # 选择后的种群
    # 对适应度进行处理，保证轮盘赌选则时不会出现nan值
    min_fit = min(fitness)
    temp_fitness = [fit - min_fit for fit in fitness]
    # 存放适应度的总和  计算轮盘赌的概率  由于适应度值是全局收益 需要的就是收益最大的, sum要做除数，不能为0
    sum_fitness = sum(temp_fitness) + 1e-4
    P_value = temp_fitness / sum_fitness  # 将每一个适应度值取出来  做比例  得出每个适应度值的占比概率 存到列表中
    P_value = np.cumsum(P_value)  # 累加和 并排序 从小到大
    random_deci = np.sort(np.random.rand(population.shape[0]))  # 产生 i个随即小数 存到列表中 从小到大排序

    fitin = 0
    newin = 0
    while newin < population.shape[0] and fitin < population.shape[0]:  # 遍历每一个解个体
      if random_deci[newin] < P_value[fitin]:
        new_popu[newin] = population[fitin]
        newin += 1
      else:
        fitin += 1
    return new_popu

  def cross(self, parents, pc):
    children = np.zeros_like(parents, dtype=np.int8)
    len_parents = parents.shape[0]  # 先提取出父代的个数  因为要配对 奇数个的话会剩余一个
    parity = len_parents % 2  # 计算出长度奇偶性  parity= 1 说明是奇数个  则需要把最后一条个体直接加上 不交叉
    for i in range(0, len_parents - 1, 2):  # 每次取出两条个体 如果是奇数个则长度要减去 一  range函数不会取最后一个
      father = parents[i]  # 取出当前选中的两个父代中的第一个
      mother = parents[i + 1]  # 取出当前选中的两个父代中的第二个
      child_1 = np.zeros_like(father, dtype=np.int8)
      child_2 = np.zeros_like(father, dtype=np.int8)
      same_content = father - mother
      for j in range(father.shape[0]):  # 逐行交叉
        same_content_index = np.where(same_content[j] == 0)  # 重复的列索引
        if np.random.rand() < pc and same_content_index[0].size >= 1:
          index = np.random.randint(0, same_content_index[0].size)  # 随机选一个重复的地方
          child_1[j] = np.concatenate((father[j][0: same_content_index[0][index] + 1],
                                       mother[j][same_content_index[0][index] + 1:]), axis=0)
          child_2[j] = np.concatenate((mother[j][0: same_content_index[0][index] + 1],
                                       father[j][same_content_index[0][index] + 1:]), axis=0)
        else:
          child_1[j] = father[j]
          child_2[j] = mother[j]
      children[i] = child_1
      children[i + 1] = child_2
    if parity == 1:  # 如果是个奇数  为了保证种群规模不变 需要加上最后一条
      children[-1] = parents[len_parents - 1]

    return children

  def mutation(self, population, pm, cm):
    count = population.shape[0]  # 子代有多少个体
    new_popu = np.zeros_like(population, dtype=np.int8)
    muta_gen_count = int(population.shape[1] * population.shape[2] * cm)
    for c in range(count):  # 对每个个体
      individual = copy.deepcopy(population[c])
      muta_gen_index = np.random.randint(0, population.shape[1] * population.shape[2], muta_gen_count)
      for index in muta_gen_index:
        if np.random.rand() < pm:
          i = int(index / population.shape[2])  # 对应agent编号
          j = index % population.shape[2]  # 对应Task编号
          # 随机一个变化量, 使得变化后的值在[0, RESOURCES[i]]范围内
          delta = np.random.randint(0, RESOURCES[i] + 1) - individual[i, j]
          individual[i, j] += delta
      new_popu[c] = individual

    # 变异后的种群转化为可行解
    i = 0
    while i < new_popu.shape[0]:
      self.make_intact_popu_valid(new_popu[i])
      i += 1
    return new_popu

  def cal_fitness(self, population):
    """
    计算解的适应度
    :param population:
    :return:
    """
    fitness = []
    # 代价矩阵
    cost = np.array(COST)[:AGENT_NUM, :TASK_NUM]
    # 每个任务中资源的单位收益
    unit_profit = []
    for t in range(TASK_NUM):
      if REQUIREMENT[t] > 0:
        unit_profit.append(PROFIT[t] / REQUIREMENT[t])
      else:
        unit_profit.append(0.0)
    unit_profit = np.array(unit_profit)
    if TASK_PROFIT_INDEPENDENT:
      # 假设任务之间不影响
      for popu in population:
        fit = 0
        # 1.先计算收益
        # 每个任务分配的资源总数
        resource_per_task = np.sum(popu, axis=0)
        for t in range(TASK_NUM):
          # 如果资源数量不足或刚好够，按比例计算收益
          if resource_per_task[t] <= REQUIREMENT[t]:
            fit += resource_per_task[t] * unit_profit[t]
          # 如果资源过剩，需要惩罚
          else:
            fit += PROFIT[t] - (resource_per_task[t] - REQUIREMENT[t]) * unit_profit[t]

        # 2.再计算代价
        # 任务选择矩阵
        choose_task_matrix = np.array(popu, dtype=np.bool_)
        temp_cost = np.sum(cost * choose_task_matrix)
        fitness.append(fit - temp_cost)
    else:  # TODO
      # 假设任务之间有影响, 对于一个tool, choose a set of tasks
      # different component get different profit,
      # not simply sum of single profit
      # 计算对每个agent选择每种任务组合对应的组合收益表
      non_inde_profit = []
      for i in range(AGENT_NUM):
        local_non_inde_profit = []
        local_profit = PROFIT[i]
        for j in range(pow(2, self.task_num)):
          local_indi = (bin(j)[2:])[::-1]
          local_indi_ = np.zeros(self.task_num, dtype=np.bool_)
          for p in range(len(local_indi)):
            local_indi_[p] = bool(int(local_indi[p]))
          sum = np.sum(local_indi_)
          if sum == 0:
            local_non_inde_profit.append(0)
          elif sum == 1:
            local_non_inde_profit.append(np.sum(local_indi_ * local_profit))

          elif sum == 2:
            local_non_inde_profit.append(np.sum(local_indi_ * local_profit) * 0.8)

          elif sum == 3:
            local_non_inde_profit.append(np.sum(local_indi_ * local_profit) * 0.6)
          else:
            local_non_inde_profit.append(0)

        non_inde_profit.append(local_non_inde_profit)

      for i in range(population.shape[0]):
        fit = 0
        for j in range(population.shape[1]):
          # agnet j选择的任务集合用二进制表示
          task_set = ''.join(str(int(num)) for num in reversed(population[i, j]))
          tase_set = int(task_set, base=2)
          fit += non_inde_profit[j][tase_set]
        fitness.append(fit)
    return fitness

  def make_local_choose_valid(self, local_choose):
    """
    用于产生初始解的过程，使不满足局部约束的解满足约束，可加速产生初始解的过程
    :param local_choose:
    :return:
    """
    # 随机选择有效的任务index
    choose_task_num = np.sum(local_choose)
    choose_task_index = []
    for index, choose in enumerate(local_choose):
      if choose > 0:
        choose_task_index.append(index)
    while choose_task_num > MAX_TASK_NUM_PER_AGENT:
      # 撤销选择的任务个数，因为至少要选一个，所以最多可撤销choose_task_num-1个
      undo_num = np.random.randint(1, choose_task_num)
      # 随机选择要撤销的任务index
      undo_index = np.random.choice(choose_task_index, undo_num, replace=False)
      for u_i in undo_index:
        local_choose[u_i] = 0
        choose_task_index.remove(u_i)
      choose_task_num -= undo_num

  def make_intact_popu_valid(self, intact_popu):
    """
    用于变异后，使不满足局部约束的解满足约束，使解尽量有效
    对于每个agent，变异后每个基因范围在[0, RESOURCE]之间，带来的问题是
    1.基因之和大于RESOURCE，需要降低
    2.承担任务总数大于MAX_TASK_NUM_PER_AGENT，需要降低
    :param local_popu:
    :return:
    """
    for i in range(self.agent_num):
      # 先保证任务个数不超过agent约束
      choose_task_index = []
      for index, res in enumerate(intact_popu[i]):
        if res > 0:
          choose_task_index.append(index)
      while len(choose_task_index) > MAX_TASK_NUM_PER_AGENT:
        undo_index = np.random.choice(choose_task_index, 1, replace=False)
        intact_popu[i][undo_index] = 0
        choose_task_index.remove(undo_index)

      # 再保证消耗资源总量不超过agent约束
      allo_resource = np.sum(intact_popu[i])
      if allo_resource - RESOURCES[i] <= 0:
        continue
      # 每个任务中资源的单位收益
      unit_profit = []
      for t in range(TASK_NUM):
        if REQUIREMENT[t] > 0:
          unit_profit.append(PROFIT[t] / REQUIREMENT[t])
        else:
          unit_profit.append(0.0)
      # 等比例减少

      delta = np.ceil(intact_popu[i] / allo_resource * (allo_resource - RESOURCES[i])).astype(np.int8)
      while allo_resource > RESOURCES[i]:
        max_unit_profit = np.argmax(unit_profit)
        if intact_popu[i][max_unit_profit] - delta[max_unit_profit] > 0:
          intact_popu[i][max_unit_profit] -= delta[max_unit_profit]
          allo_resource -= delta[max_unit_profit]
        else:
          allo_resource -= intact_popu[i][max_unit_profit]
          intact_popu[i][max_unit_profit] = 0
        unit_profit[max_unit_profit] = 0

  def n_ints_summing_to_v(n, v):
    """
    随机生成一个大小为n的数组，数组中个元素的和为v
    :param v:
    :return:
    """
    elements = []
    for i in range(v):
      elements.append(np.arange(n) == np.random.randint(0, n))
    return np.sum(elements, axis=0)


def cen_evolution(evolu_times, absolute_time, at_valid=False):
  """
  每个agent的进化过程
  :param at_valid: 绝对时间限制是否启用，默认不启用
  :param absolute_time: 算法运行的绝对时间
  :param evolu_time: 进化次数
  :return:
  """
  serial_compute = 0
  # 串行段0
  time_stamp = time.time() * 1000  # 毫秒级时间戳

  start_time = time_stamp
  c_agent = Centralized_Agent(TASK_NUM, AGENT_NUM, INIT_POPU_NUM)
  c_agent.get_init_population()
  # 画图用
  avg_popu_fitness = []
  best_popu_fitness = []

  pre_gene_intact_popu = c_agent.intact_population
  pre_gen_intact_fit = c_agent.intact_population_fit

  # 时间起效
  if at_valid:
    serial_compute += time.time() * 1000 - time_stamp
    time_stamp = time.time() * 1000
    while serial_compute < absolute_time:
      new = c_agent.selection(pre_gene_intact_popu, pre_gen_intact_fit)
      # 交叉
      cross = c_agent.cross(new, PC)
      # 变异
      mutate = c_agent.mutation(cross, PM, CM)

      # 新旧两代一起排序，保证下一代不差于本代
      mutate_fit = c_agent.cal_fitness(mutate)
      total_fit = pre_gen_intact_fit + mutate_fit
      total_popu = np.concatenate((pre_gene_intact_popu, mutate), axis=0)
      temp_dict = dict(zip(total_fit, total_popu))
      sort_fit = sorted(temp_dict, reverse=True)

      # 新设一个array
      new_intact_popu_size = min(len(sort_fit), c_agent.intact_population_size)
      sorted_fit = sort_fit[:new_intact_popu_size]
      new_intact_popu = np.zeros((new_intact_popu_size, c_agent.agent_num, c_agent.task_num), dtype=np.int8)
      j = 0
      for ft in sorted_fit:
        # if ft <= 0.0:
        #     break
        new_intact_popu[j] = temp_dict[ft]
        j += 1
      pre_gene_intact_popu = new_intact_popu[: j]
      pre_gen_intact_fit = sorted_fit[: j]
      avg_popu_fitness.append(np.mean(pre_gen_intact_fit))
      best_popu_fitness.append([time.time() * 1000 - start_time, pre_gen_intact_fit[0]])

      serial_compute += time.time() * 1000 - time_stamp
      time_stamp = time.time() * 1000

  else:
    for i in range(evolu_times):
      # 选择
      new = c_agent.selection(pre_gene_intact_popu, pre_gen_intact_fit)
      # 交叉
      cross = c_agent.cross(new, PC)
      # 变异
      mutate = c_agent.mutation(cross, PM, CM)

      # 新旧两代一起排序，保证下一代不差于本代
      mutate_fit = c_agent.cal_fitness(mutate)
      total_fit = pre_gen_intact_fit + mutate_fit
      total_popu = np.concatenate((pre_gene_intact_popu, mutate), axis=0)
      temp_dict = dict(zip(total_fit, total_popu))
      sort_fit = sorted(temp_dict, reverse=True)
      # 新设一个array
      new_intact_popu_size = min(len(sort_fit), c_agent.intact_population_size)
      sorted_fit = sort_fit[:new_intact_popu_size]
      new_intact_popu = np.zeros((new_intact_popu_size, c_agent.agent_num, c_agent.task_num), dtype=np.int8)
      j = 0
      for ft in sorted_fit:
        # if ft <= 0.0:
        #     break
        new_intact_popu[j] = temp_dict[ft]
        j += 1
      pre_gene_intact_popu = new_intact_popu[: j]
      pre_gen_intact_fit = sorted_fit[: j]
      avg_popu_fitness.append(np.mean(pre_gen_intact_fit))
      best_popu_fitness.append([time.time() * 1000 - start_time, pre_gen_intact_fit[0]])
  c_agent.intact_population = pre_gene_intact_popu
  c_agent.intact_population_fit = pre_gen_intact_fit

  # 串行段0结束
  serial_compute += time.time() * 1000 - time_stamp

  # 问题复杂度高时，初始解十分耗时，导致best_popu_fitness为空
  if len(best_popu_fitness) == 0:
    avg_popu_fitness.append(np.mean(pre_gen_intact_fit))
    best_popu_fitness.append([serial_compute, np.max(pre_gen_intact_fit)])
    # print('最大收益：', best_popu_fitness[0][1],
    #       '最佳分配：\n', c_agent.intact_population[np.argmax(pre_gen_intact_fit)])
    # print('串行计算时间：%0.2f' % (serial_compute / 3000))
  else:
    pass
    # print('最大收益：', c_agent.intact_population_fit[0],
    #       '最佳分配：\n', c_agent.intact_population[0])
    # print('串行计算时间：%0.2f' % (serial_compute / 3000))

  return avg_popu_fitness, best_popu_fitness


if __name__ == '__main__':
  total_evolu_times = 500
  run_time = 5000
  ab_time = False
  avg_popu_fitness, best_popu_fitness = cen_evolution(total_evolu_times, run_time, ab_time)
  # 第一个图：折线图
  plt.subplot(2, 1, 1)
  plt.title("avg_fit")
  # 画集中式
  X = [i for i in range(1, len(avg_popu_fitness) + 1)]
  Y = avg_popu_fitness
  plt.plot(X, Y, c="r", label="central avg_fit")
  # 第2个图：折线图
  plt.subplot(2, 1, 2)
  plt.title("best_fit")
  # 画集中式
  X = [i[0] for i in best_popu_fitness]
  Y = [i[1] for i in best_popu_fitness]
  plt.plot(X, Y, c="g", label="central best_fit")  # marker='o', markeredgecolor='r', markersize='2'
  plt.show()
