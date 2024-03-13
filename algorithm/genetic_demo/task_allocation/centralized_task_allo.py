import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from task_allo_problem_setting import *


class Centralized_Agent:
  def __init__(self, task_num, agent_num, init_popu_num):
    # 初始时，选择任务列表均为0， 表示没有选择任何任务
    self.init_popu_num = init_popu_num
    self.agent_num = agent_num
    self.task_num = task_num
    # 为保证一致，完整解种群的大小需要参考分布式的情况
    self.intact_population_size = init_popu_num * SHARE_NUM  # pow(SHARE_NUM, agent_num - 1)
    # 解的形式为intact_population_size × agent_num × task_num的数组，
    # 每个agent的解用一个bool数组表示，数组长度为task_num的长度，
    # 一个数组元素代表一个基因，1表示选择该任务，0表示不选择该任务
    # agent_num个agent的解组成一个完整解
    self.intact_population = np.zeros((self.init_popu_num, agent_num, task_num), dtype=np.bool_)
    self.intact_population_fit = []

  def get_init_population(self):
    # 为扩大搜索范围，尽量生成不相同的初始解
    # 根据自身能力约束形成任务选择方案，即认为是遗传算法的初始可行解
    # 先假定每个人承担的任务不超过3个，并且一个任务仅分配一个平台，其他无限制
    i = 0
    # 为提高运行速度，一开始先生成少量初始解, 后面是全0的解
    while i < self.init_popu_num:
      temp = np.random.randint(0, 2, (self.agent_num, self.task_num))
      valid = True
      # 检查每个人承担的任务不超过3个
      task_per_tool = np.sum(temp, axis=1)
      if task_per_tool.max() > MAX_TASK_NUM_PER_AGENT:
        valid = False
      # 检查一个任务仅分配一个平台
      tool_per_task = np.sum(temp, axis=0)
      if tool_per_task.max() > 1:
        valid = False
      exist = False
      for k in range(i):
        if (self.intact_population[k] == temp).all():
          exist = True
          break
      if valid and not exist:
        self.intact_population[i] = temp
        i += 1

    # 计算种群fitness备用
    self.intact_population_fit = self.cal_fitness(self.intact_population)

  def selection(self, population, fitness):
    # 按照适应度选择（排序）种群
    new_popu = np.zeros_like(population, dtype=np.bool_)  # 选择后的种群
    sum_fitness = sum(fitness)  # 存放适应度的总和  计算轮盘赌的概率  由于适应度值是全局收益  将需要的就是收益最大的
    P_value = fitness / sum_fitness  # 将每一个适应度值取出来  做比例  得出每个适应度值的占比概率 存到列表中
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
    children = np.zeros_like(parents, dtype=np.bool_)
    len_parents = parents.shape[0]  # 先提取出父代的个数  因为要配对 奇数个的话会剩余一个
    parity = len_parents % 2  # 计算出长度奇偶性  parity= 1 说明是奇数个  则需要把最后一条个体直接加上 不交叉
    for i in range(0, len_parents - 1, 2):  # 每次取出两条个体 如果是奇数个则长度要减去 一  range函数不会取最后一个
      father = parents[i]  # 取出当前选中的两个父代中的第一个
      mother = parents[i + 1]  # 取出当前选中的两个父代中的第二个
      child_1 = np.zeros_like(father, dtype=np.bool_)
      child_2 = np.zeros_like(father, dtype=np.bool_)
      same_content = np.bitwise_xor(father, mother)
      for j in range(father.shape[0]):  # 逐行交叉
        same_content_index = np.where(same_content[j] == False)  # 重复的列索引
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
    new_popu = np.zeros_like(population, dtype=np.bool_)
    muta_gen_count = int(population.shape[1] * population.shape[2] * cm)
    for c in range(count):  # 对每个个体
      individual = copy.deepcopy(population[c])
      muta_gen_index = np.random.randint(0, population.shape[1] * population.shape[2], muta_gen_count)
      for index in muta_gen_index:
        if np.random.rand() < pm:
          i = int(index / population.shape[2])
          j = index % population.shape[2]
          individual[i, j] = 1 - individual[i, j]
      new_popu[c] = individual

    # 变异后的种群转化为可行解
    i = 0
    while i < new_popu.shape[0]:
      if not self.satisfy_coupling_constrains(new_popu[i]):
        new_popu = np.delete(new_popu, i, axis=0)
      else:
        i += 1
    return new_popu

  @staticmethod
  def satisfy_coupling_constrains(intact_popu):
    success = True
    # 检查每个人承担的任务不超过3个
    task_per_tool = np.sum(intact_popu, axis=1)
    # 检查一个任务仅分配一个平台
    tool_per_task = np.sum(intact_popu, axis=0)

    if task_per_tool.max() > MAX_TASK_NUM_PER_AGENT or tool_per_task.max() > 1:
      success = False
    return success

  def cal_fitness(self, population):
    """
    计算解的适应度
    :param population:
    :return:
    """
    fitness = []
    profit = np.array(PROFIT)[:AGENT_NUM, :TASK_NUM]
    if TASK_PROFIT_INDEPENDENT:
      # 假设任务之间不影响
      for i in range(population.shape[0]):
        fit = 0
        for j in range(population.shape[1]):
          for k in range(population.shape[2]):
            fit += population[i, j, k] * profit[j][k]
        fitness.append(fit)
    else:
      # 假设任务之间有影响, 对于一个tool, choose a set of tasks
      # different component get different profit,
      # not simply sum of single profit
      # 计算对每个agent选择每种任务组合对应的组合收益表
      non_inde_profit = []
      for i in range(AGENT_NUM):
        local_non_inde_profit = []
        local_profit = profit[i]
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


def cen_evolution(evolu_times, absolute_time=3000, at_valid=False):
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
      new_intact_popu = np.zeros((new_intact_popu_size, c_agent.agent_num, c_agent.task_num), dtype=np.bool_)
      j = 0
      for ft in sorted_fit:
        if ft <= 0.0 or j >= new_intact_popu_size:
          break
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
      sorted_fit = sort_fit[:c_agent.init_popu_num]
      # 新设一个array，为了尽量保证解的个数不变
      new_intact_popu = np.zeros((c_agent.init_popu_num, c_agent.agent_num, c_agent.task_num), dtype=np.bool_)
      j = 0
      for ft in sorted_fit:
        if j >= c_agent.init_popu_num or ft == float('-inf'):
          break
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
    #       '最佳分配\n：', c_agent.intact_population[np.argmax(pre_gen_intact_fit)])
    # print('串行计算时间：%0.2f' % (serial_compute / 3000))
  else:
    print('最大收益：', c_agent.intact_population_fit[0],
          '最佳分配\n：', c_agent.intact_population[0].astype(int))
    print('串行计算时间：%0.2f' % (serial_compute / 3000))

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
  return avg_popu_fitness, best_popu_fitness


if __name__ == '__main__':
  np.random.seed(0)
  total_evolu_times = 450
  run_time = 5000
  ab_time = True
  cen_evolution(total_evolu_times)
