import numpy as np
import copy
import matplotlib.pyplot as plt
import time
from task_allo_problem_setting import *
# from rand import *

debug = False


class Agent:
  def __init__(self, no, task_num, init_popu_num, coop_agent_num):
    # agent 编号
    self.no = no
    # 初始时，选择任务列表均为0， 表示没有选择任何任务
    self.init_popu_num = init_popu_num
    self.task_num = task_num
    # 初始解的形式为init_popu_num × task_num的数组，每个agent只产生本agent的解
    # 本agent的解用一个bool数组表示，数组长度为task_num的长度，
    # 一个数组元素代表一个基因，1表示选择该任务，0表示不选择该任务
    self.local_population = np.zeros((init_popu_num, task_num), dtype=np.bool_)
    self.local_population_fit = []
    # 来自其他agent的最优解形式为coop_agent_num × task_num的数组
    # coop_agent_num 代表其他合作agent的个数
    # 初始时没收到任何信息，最优解为本agent的随机满足耦合约束的赋值，此处赋值均为0
    self.best_remote_popu = np.zeros((coop_agent_num, task_num), dtype=np.bool_)

  def get_init_local_population(self):
    """
    为扩大搜索范围，尽量生成不相同的初始解
    :return:
    """
    i = 1  # 保留[0， 0， 0，0]
    while i < self.init_popu_num:
      # 生成随机解
      # temp = np.random.randint(0, 2, self.task_num)
      temp = [np.random.randint(0, 2) for i in range(self.task_num)]
      exist = False
      for k in range(i):
        if (self.local_population[k] == temp).all():
          exist = True
          break
      # 检查随机解是否满足局部约束 以及 该解是否已存在
      if self.satisfy_local_constrains(temp) and not exist:
        self.local_population[i] = temp
        i += 1
    # 计算种群fitness备用
    self.local_population_fit = self.cal_intact_popu_fitness(self.local_population, self.best_remote_popu)

  # 下列含local的函数为仅利用局部信息的单个agent的进化过程
  @staticmethod
  def selection_local(local_population, local_population_fit):
    """
    按照适应度选择（排序）种群, 方法为轮盘赌，不同的应用此函数基本不变
    选择后的种群与原种群中个体数量一致
    :param local_population:
    :param local_population_fit:
    :return:
    """
    new_local_popu = np.zeros_like(local_population, dtype=np.bool_)  # 选择后的种群
    sum_fitness = sum(local_population_fit)  # 存放适应度的总和  计算轮盘赌的概率  由于适应度值是全局收益  将需要的就是收益最大的
    P_value = local_population_fit / sum_fitness  # 将每一个适应度值取出来  做比例  得出每个适应度值的占比概率 存到列表中
    P_value = np.cumsum(P_value)  # 累加和 并排序 从小到大
    random_deci = np.sort(np.random.rand(local_population.shape[0]))  # 产生 i个随即小数 存到列表中 从小到大排序
    # random_deci = np.sort(np.array([randdouble(0, 1) for i in range(0, local_population.shape[0])]))
    fitin = 0
    newin = 0
    while newin < local_population.shape[0] and fitin < local_population.shape[0]:  # 遍历每一个解个体
      if random_deci[newin] < P_value[fitin]:
        new_local_popu[newin] = local_population[fitin]
        newin += 1
      else:
        fitin += 1
    return new_local_popu

  def cross_local(self, parents_local, pc):
    """
    上代种群交叉产生子代种群，交叉的方式需与问题解的建模对应，保证合理
    子代种群与原种群中个体数量不一定一致，因为可以再此处直接删除不满足约束的子代
    也可不删除，后面用适应度值来筛选
    :param parents_local:
    :param pc: 交叉概率
    :return:
    """
    children = np.zeros_like(parents_local, dtype=np.bool_)
    len_parents = len(parents_local)  # 先提取出父代的个数  因为要配对 奇数个的话会剩余一个
    parity = len_parents % 2  # 计算出长度奇偶性  parity= 1 说明是奇数个  则需要把最后一条个体直接加上 不交叉
    for i in range(0, len_parents - 1, 2):  # 每次取出两条个体 如果是奇数个则长度要减去 一  range函数不会取最后一个
      father = parents_local[i]  # 取出当前选中的两个父代中的第一个
      mother = parents_local[i + 1]  # 取出当前选中的两个父代中的第二个
      # 针对不同应用，主要修改此处，即交叉的方式
      same_content = np.bitwise_xor(father, mother)
      same_content_index = np.where(same_content == False)  # 找出两个解中重复的值对应的列索引
      if np.random.rand() < pc and same_content_index[0].size >= 1:
        index = np.random.randint(0, same_content_index[0].size)  # 随机选一个重复的地方
        # 以重复的位置分割成4份，两两组合成新的解
        child_1 = np.concatenate((father[0: same_content_index[0][index] + 1],
                                  mother[same_content_index[0][index] + 1:]))
        child_2 = np.concatenate((mother[0: same_content_index[0][index] + 1],
                                  father[same_content_index[0][index] + 1:]))
      else:
        child_1 = father
        child_2 = mother
      children[i] = child_1
      children[i + 1] = child_2
    if parity == 1:  # 如果是个奇数  为了保证种群规模不变 需要加上最后一条
      children[-1] = parents_local[-1]

    # 交叉后的种群转化为可行解，只需满足本地约束即可
    i = 0
    while i < children.shape[0]:
      if not self.satisfy_local_constrains(children[i]):
        children = np.delete(children, i, axis=0)
      else:
        i += 1
    return children

  def mutation_local(self, local_population, pm, cm):
    """
    对交叉产生的子代种群进行随机变异，以保证探索更多解空间
    变异的方式需与问题解的建模对应，保证合理
    变异后的种群与原种群中个体数量不一定一致，因为可以再此处直接删除不满足约束的变异后的解
    也可不删除，后面用适应度值来筛选
    :param local_population:
    :param pm: 变异概率
    :param cm: 每个个体变异的基因个数比例
    :return:
    """
    count = local_population.shape[0]  # 子代有多少个体
    new_local_popu = np.zeros_like(local_population, dtype=np.bool_)
    # 每个个体变异的基因个数
    muta_gen_count = int(local_population.shape[1] * cm)
    for c in range(count):  # 对每个个体
      individual = copy.deepcopy(local_population[c])
      # 针对不同应用，主要修改此处，即变异的方式
      # 随机生成变异的基因index
      muta_gen_index = np.array([np.random.randint(0, local_population.shape[1]) for i in range(muta_gen_count)])
      for index in muta_gen_index:
        if np.random.rand() < pm:
          individual[index] = 1 - individual[index]
      new_local_popu[c] = individual

    # 变异后的种群转化为可行解
    i = 0
    while i < new_local_popu.shape[0]:
      if not self.satisfy_local_constrains(new_local_popu[i]):
        new_local_popu = np.delete(new_local_popu, i, axis=0)
      else:
        i += 1
    return new_local_popu

  def cal_intact_popu_fitness(self, local_population, remote_popu):
    """
    计算解的适应度
    :param local_population: 本地agent的解种群
    :param remote_popu: 来自其他agent的解，个数为1
    :return:
    """
    fitness = []
    # 把本地解和其他agent的解组合成完整解
    agent_num = 1 + remote_popu.shape[0]  # 完整解对应的的agent个数，remote_popu中一个元素来自一个agent
    intact_popu = np.zeros((local_population.shape[0], agent_num, self.task_num), dtype=np.bool_)
    i = 0
    for local_individual in local_population:
      intact_individual = np.insert(remote_popu, self.no, np.array([local_individual]), axis=0)
      intact_popu[i] = intact_individual
      i += 1

    profit = np.array(PROFIT)[:AGENT_NUM, :TASK_NUM]

    if TASK_PROFIT_INDEPENDENT:
      # 假设任务之间不影响
      for i in range(intact_popu.shape[0]):
        fit = 0
        for j in range(intact_popu.shape[1]):
          for k in range(intact_popu.shape[2]):
            fit += intact_popu[i, j, k] * profit[j][k]
        fitness.append(fit)
    else:
      # 假设任务之间有影响, 对于一个tool, choose a set of tasks
      # different component get different profit,
      # not simply sum of single profit
      # 计算对每个agent选择每种任务组合对应的组合收益表
      non_inde_profit = []
      for i in range(agent_num):  # 3代表3个agent
        local_non_inde_profit = []
        local_profit = profit[i]
        for j in range(pow(2, self.task_num)):
          # 每个整数j转化为2进制之后代表一种任务选择
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

      for i in range(intact_popu.shape[0]):
        fit = 0
        valid = self.satisfy_coupling_constrains_intact(intact_popu[i])
        if valid:  # 如果完整解满足耦合约束，计算所有agent的收益
          for j in range(intact_popu.shape[1]):
            # agnet j选择的任务集合用二进制表示
            task_set = ''.join(str(int(num)) for num in reversed(intact_popu[i, j]))
            tase_set = int(task_set, base=2)
            fit += non_inde_profit[j][tase_set]
        else:  # 如果完整解不满足耦合约束，仅以本agent的收益作为全部收益
          task_set = ''.join(str(int(num)) for num in reversed(intact_popu[i, self.no]))
          tase_set = int(task_set, base=2)
          fit += non_inde_profit[self.no][tase_set]
        fitness.append(fit)
    return fitness

  def evolution(self, evolu_time):
    """
    每个agent的进化过程
    :param evolu_time: 进化次数
    :return:
    """
    pre_gene_local_popu = self.local_population
    pre_gen_local_fit = self.local_population_fit
    for i in range(evolu_time):
      if debug:
        print("Agent ", self.no, " popu before selection")
        _popu = np.array(pre_gene_local_popu, dtype=int)
        print(_popu)
      # 选择
      new = self.selection_local(pre_gene_local_popu, pre_gen_local_fit)
      if debug:
        print("Agent ", self.no, " popu after selection")
        _popu = np.array(new, dtype=int)
        print(_popu)
      # 交叉
      cross = self.cross_local(new, PC)
      if debug:
        print("Agent ", self.no, " popu after cross")
        _popu = np.array(cross, dtype=int)
        print(_popu)
      # 变异
      mutate = self.mutation_local(cross, PM, CM)
      if debug:
        print("Agent ", self.no, " popu after mutate:")
        _popu = np.array(mutate, dtype=int)
        print(_popu)

      # 新旧两代一起排序，保证下一代不差于本代
      mutate_fit = self.cal_intact_popu_fitness(mutate, self.best_remote_popu)

      if debug:
        print("Agent ", self.no, " best remote popu after evolution:")
        _popu = np.array(self.best_remote_popu, dtype=int)
        print(_popu)

      total_fit = pre_gen_local_fit + mutate_fit
      total_popu = np.concatenate((pre_gene_local_popu, mutate), axis=0)
      if debug:
        print("Agent ", self.no, " total popu after evolution:")
        _popu = np.array(total_popu, dtype=int)
        print(_popu)
      temp_dict = dict(zip(total_fit, total_popu))
      sort_fit = sorted(temp_dict, reverse=True)
      sorted_fit = sort_fit[:self.init_popu_num]
      if debug:
        print("Agent ", self.no, " sorted total fit after evolution")
        for f in sorted_fit:
          print("profit: ", f, ", ", np.array(temp_dict[f], dtype=int))
      # 新设一个array，为了尽量保证解的个数不变
      new_local_popu = np.zeros((self.init_popu_num, self.task_num), dtype=np.bool_)
      j = 0
      for i in sorted_fit:
        if j >= self.init_popu_num or i == float('inf'):
          break
        new_local_popu[j] = temp_dict[i]
        j += 1
      pre_gene_local_popu = new_local_popu[: j]
      pre_gen_local_fit = sorted_fit[: j]
    self.local_population = pre_gene_local_popu
    self.local_population_fit = pre_gen_local_fit

  # 的函数为利用全局信息的单个agent的进化过程
  def selection_for_share(self, share_num):
    """
    按照轮盘赌选择部分解进行共享
    :param local_population:
    :param share_num:
    :return:
    """
    # fitness = self.cal_fitness_single(local_population, True, self.best_other)  # 共享信息之前只能独立进化
    fitness = self.local_population_fit
    shared_local_popu = np.zeros((share_num, self.task_num), dtype=np.bool_)
    shared_local_popu_fit = np.zeros(share_num, dtype=np.float_)
    sum_fitness = sum(fitness)  # 存放适应度的总和  计算轮盘赌的概率  由于适应度值是全局收益  将需要的就是收益最大的
    P_value = fitness / sum_fitness  # 将每一个适应度值取出来  做比例  得出每个适应度值的占比概率 存到列表中
    P_value = np.cumsum(P_value)  # 累加和 并排序 从小到大
    random_deci = np.sort(np.random.rand(share_num))  # 产生 i个随即小数 存到列表中 从小到大排序

    fitin = 0
    newin = 0
    while newin < share_num and fitin < self.local_population.shape[0]:  # 遍历每一个解个体
      if random_deci[newin] < P_value[fitin]:
        shared_local_popu[newin] = self.local_population[fitin]
        shared_local_popu_fit[newin] = self.local_population_fit[fitin]
        newin += 1
      else:
        fitin += 1
    return shared_local_popu, shared_local_popu_fit

  def cooperation_all(self, other_shared_popu):
    """
    接收到别的agent共享的数据后，通过合作原则调整完整的解
    :param local_population: 本地解种群
    :param other_shared_popu: 其他所有agent组合的解种群
    :return:
    """
    # 可能有重复的解以及无效的解，因此用平均fitness代表
    max_fit_for_local_popu, self.best_remote_popu \
        = self.cal_fitness_after_share(self.local_population, other_shared_popu)

    temp_dict = dict(zip(max_fit_for_local_popu, self.local_population))
    sort_fit = sorted(temp_dict, reverse=True)
    # 选择适应度较好的
    sorted_fit = sort_fit[:self.init_popu_num]
    new_popu = np.zeros((self.init_popu_num, self.task_num), dtype=np.bool_)
    j = 0
    for i in sorted_fit:
      if j >= self.init_popu_num:
        break
      if i == float('-inf'):
        break
      new_popu[j] = temp_dict[i]
      j += 1

    self.local_population = new_popu[: j]
    self.local_population_fit = sorted_fit[: j]  # 缓存一下，便于取用

  def cal_fitness_after_share(self, local_population, other_shared_popu):
    """
    接收到别的agent共享的数据后，计算组合的完整解的适应度
    :param other_shared_popu:
    :param local_population:
    :return:
    """
    other = np.insert(other_shared_popu, -1, self.best_remote_popu, axis=0)
    other = np.unique(other, axis=0)  # 去重

    max_fit_for_local_popu = []
    for local_indi in local_population:
      fit = 0
      i = 0
      for remote_indi in other:
        if self.satisfy_coupling_constrains(local_indi, remote_indi):
          fit = max(fit,
                    self.cal_intact_popu_fitness(np.array([local_indi]), remote_indi)[0])
          i += 1
      if i > 0:
        max_fit_for_local_popu.append(fit)
      else:
        max_fit_for_local_popu.append(float('-inf'))

    max_fit_for_remote_popu = []
    for remote_indi in other:
      fit = 0
      i = 0
      for local_indi in local_population:
        if self.satisfy_coupling_constrains(local_indi, remote_indi):
          fit = max(fit,
                    self.cal_intact_popu_fitness(np.array([local_indi]), remote_indi)[0])
          i += 1
      if i > 0:
        max_fit_for_remote_popu.append(fit)
      else:
        max_fit_for_remote_popu.append(float('-inf'))

    return max_fit_for_local_popu, other[np.argmax(max_fit_for_remote_popu)]

  def update_self_by_best_intact_popu(self, best_intact_popu, best_fitness):
    """
    根据本轮共享后的适应度最好的完整解，更新本地解
    :param best_fitness:
    :param best_intact_popu:
    :return:
    """
    # 新增本地解 和 fitness
    # print(best_intact_popu.shape)
    self.local_population = np.insert(self.local_population, 0,
                                      best_intact_popu[self.no], axis=0)
    self.local_population_fit.insert(0, best_fitness)
    # 修改 best_remote
    self.best_remote_popu = np.delete(best_intact_popu, self.no, axis=0)

  @staticmethod
  def satisfy_local_constrains(individual_local):
    """
    检查本agent的单个解是否满足局部约束， 每个agent的局部约束不相同，为简化问题此处定义为一致
    即，每个人承担的任务不超过规定的最大数量MAX_TASK_NUM_PER_AGENT
    :param individual_local: 本地种群中的一个个体
    :return:
    """
    success = True
    task_num_per_agent = np.sum(individual_local, axis=0)
    if task_num_per_agent > MAX_TASK_NUM_PER_AGENT:
      success = False
    return success

  def satisfy_coupling_constrains(self, local_indi, remote_indi):
    """
    判断组合后的一个完整解是否满足耦合约束
    :param remote_indi:
    :param local_indi:
    :return:
    """
    intact_individual = np.insert(remote_indi, self.no, local_indi, axis=0)
    success = True
    # 检查每个人承担的任务不超过3个
    task_per_tool = np.sum(intact_individual, axis=1)
    # 检查一个任务仅分配一个平台
    tool_per_task = np.sum(intact_individual, axis=0)

    if task_per_tool.max() > MAX_TASK_NUM_PER_AGENT or tool_per_task.max() > 1:
      success = False
    return success

  @staticmethod
  def satisfy_coupling_constrains_intact(intact_popu):
    """
    判断组合后的一个完整解是否满足耦合约束
    :param intact_popu:
    :param single:
    :return:
    """
    success = True
    # 检查每个人承担的任务不超过3个
    task_per_tool = np.sum(intact_popu, axis=1)
    # 检查一个任务仅分配一个平台
    tool_per_task = np.sum(intact_popu, axis=0)

    if task_per_tool.max() > MAX_TASK_NUM_PER_AGENT or tool_per_task.max() > 1:
      success = False
    return success

  def get_best_intact_indi(self):
    return np.insert(self.best_remote_popu, self.no, self.local_population[0], axis=0).astype(int)


def get_remote_popu_from_share_pool(share_pool, agent_no):
  # 其他agent的解的个数, 比如一共3个agent，每个共享4个解，那么共组合出4^(3-1)=16种解
  remote_popu_num = pow(SHARE_NUM, AGENT_NUM - 1)
  # 每个解有AGENT_NUM - 1行，TASK_NUM列
  remote_popus = np.zeros((remote_popu_num, AGENT_NUM - 1, TASK_NUM), dtype=np.bool_)
  # 把共享池中自己的部分删掉
  valid_pool = np.delete(share_pool, agent_no, axis=0)
  p = 0
  # 一共要产生remote_popu_num个解
  while p < remote_popu_num:
    # 把p转换成以share_num为底的进制数，每一位对应一个index
    index = []
    tmp = p
    for j in range(AGENT_NUM - 1):
      index.append(int(tmp / pow(SHARE_NUM, AGENT_NUM - j - 2)))
      tmp = tmp % pow(SHARE_NUM, AGENT_NUM - j - 2)
    # 从valid_pool的每个元素中取一个
    remote_popu = np.zeros((AGENT_NUM - 1, TASK_NUM), dtype=np.bool_)
    for j in range(AGENT_NUM - 1):
      remote_popu[j] = valid_pool[j, index[j]]
    remote_popus[p] = remote_popu
    p += 1

  return remote_popus


def co_evolution(interact_time, independent_time, absolute_time, at_valid=False):
  agents = []
  # 画图用
  avg_popu_fitness_all = []
  best_popu_fitness_all = []

  paralle_compute = 0
  communication = 0

  # 并行段0
  time_stamp = time.time() * 1000  # 毫秒级时间戳
  start_time = time_stamp
  for i in range(AGENT_NUM):
    agent = Agent(i, TASK_NUM, INIT_POPU_NUM, AGENT_NUM - 1)
    agent.get_init_local_population()
    agents.append(agent)
    avg_popu_fitness_all.append([])
    best_popu_fitness_all.append([])
  # 并行段0结束
  paralle_compute += time.time() * 1000 - time_stamp

  # print("Agent popu after init ")
  # for i in range(AGENT_NUM):
  #     print("Agent ",i," popu:")
  #     local_popu = np.array(agents[i].local_population,dtype=int)
  #     print(local_popu)

  if at_valid:
    while paralle_compute / AGENT_NUM + communication < absolute_time:
      # print(paralle_compute + communication)
      # 并行段1
      time_stamp = time.time() * 1000
      # 解共享池
      share_pool = np.zeros((AGENT_NUM, SHARE_NUM, TASK_NUM), dtype=np.bool_)
      for i in range(AGENT_NUM):
        # 每个agent分别单独进化
        agents[i].evolution(independent_time)
        # evolution操作之后种群和fitness都是排好序的
        share_pool[i] = agents[i].selection_for_share(SHARE_NUM)

      # 假设采用广播机制，形成share_pool的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY
      paralle_compute += time.time() * 1000 - time_stamp
      time_stamp = time.time() * 1000

      for i in range(AGENT_NUM):
        # 每个agent从共享池中拿其他agent的解
        remote_popus = get_remote_popu_from_share_pool(share_pool, i)
        agents[i].cooperation_all(remote_popus)
        # coop操作之后种群和fitness都是排好序的
        if agents[i].local_population_fit:
          avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
          best_popu_fitness_all[i].append([paralle_compute / AGENT_NUM + communication,
                                           agents[i].local_population_fit[0]])

      # 假设采用广播机制，agent共享最优全局解的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY
      # 选出适应度最好的完整解，与大家共享
      best_agent_index = np.argmax([agents[i].local_population_fit[0] if agents[i].local_population_fit else -1.0
                                    for i in range(AGENT_NUM)])
      best_fitness = agents[best_agent_index].local_population_fit[0]
      best_intact_popu = np.insert(agents[best_agent_index].best_remote_popu, best_agent_index,
                                   agents[best_agent_index].local_population[0], axis=0)

      for i in range(AGENT_NUM):
        if i == best_agent_index:
          continue
        agents[i].update_self_by_best_intact_popu(best_intact_popu, best_fitness)
        # 被更新的
        # best_popu_fitness_all[i][-1][0] += COMMUNACATION_DELAY
      # 并行段3结束
      paralle_compute += time.time() * 1000 - time_stamp

  else:
    for it in range(interact_time):  # 交互次数
      print("=====Interact Iteration " + str(it) + "=====")
      # 并行段1
      time_stamp = time.time() * 1000
      # 解共享池
      share_pool = np.zeros((AGENT_NUM, SHARE_NUM, TASK_NUM), dtype=np.bool_)
      share_pool_fit = np.zeros((AGENT_NUM, SHARE_NUM), dtype=np.float_)
      # print("Agent popu before evolution ", it)
      # for i in range(AGENT_NUM):
      #     print("Agent ",i," popu:")
      #     local_popu = np.array(agents[i].local_population,dtype=int)
      #     print(local_popu)

      for i in range(AGENT_NUM):
        # 每个agent分别单独进化10次
        agents[i].evolution(independent_time)
        # evolution操作之后种群和fitness都是排好序的
        share_pool[i], share_pool_fit[i] = agents[i].selection_for_share(SHARE_NUM)

      # print("Agent popu after evolution ", it)
      # for i in range(AGENT_NUM):
      #     print("Agent ",i," popu:")
      #     local_popu = np.array(agents[i].local_population,dtype=int)
      #     print(local_popu)

      # 假设采用广播机制，形成share_pool的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY
      paralle_compute += time.time() * 1000 - time_stamp
      time_stamp = time.time() * 1000
      for i in range(AGENT_NUM):
        # 每个agent从共享池中拿其他agent的解
        remote_popus = get_remote_popu_from_share_pool(share_pool, i)
        agents[i].cooperation_all(remote_popus)
        # coop操作之后种群和fitness都是排好序的
        if agents[i].local_population_fit:
          avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
          best_popu_fitness_all[i].append([paralle_compute / AGENT_NUM + communication,
                                          agents[i].local_population_fit[0]])

      # 假设采用广播机制，agent共享最优全局解的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY

      # print("Agent popu after cooperation all ", it)
      # for i in range(AGENT_NUM):
      #     print("Agent ",i," popu:")
      #     local_popu = np.array(agents[i].local_population,dtype=int)
      #     print(local_popu)
      # 选出适应度最好的完整解，与大家共享
      best_agent_index = np.argmax([agents[i].local_population_fit[0] if agents[i].local_population_fit else -1.0
                                    for i in range(AGENT_NUM)])
      best_fitness = agents[best_agent_index].local_population_fit[0]
      best_intact_popu = np.insert(agents[best_agent_index].best_remote_popu, best_agent_index,
                                   agents[best_agent_index].local_population[0], axis=0)

      # print(best_fitness)
      for i in range(AGENT_NUM):
        if i == best_agent_index:
          continue
        agents[i].update_self_by_best_intact_popu(best_intact_popu, best_fitness)

      # print("Agent popu after update self ", it)
      # for i in range(AGENT_NUM):
      #     print("Agent ",i," popu:")
      #     local_popu = np.array(agents[i].local_population,dtype=int)
      #     print(local_popu)
      # 并行段3结束
      paralle_compute += time.time() * 1000 - time_stamp

  print('最大收益：%0.5f' % agents[0].local_population_fit[0],
        '\n最优分配方式：\n', agents[0].get_best_intact_indi())

  print('并行计算时间：%0.2f' % (paralle_compute / 3000),
        '\n通讯消耗时间：', communication / 1000)

  # i = 0
  # for avg_popu_fitness in avg_popu_fitness_all:
  #   plt.plot(avg_popu_fitness, color=cnames[i])
  #   i += 1
  # plt.show()
  if len(best_popu_fitness_all) == 0:
    for i in range(AGENT_NUM):
      avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
      best_popu_fitness_all[i].append([time.time() * 1000 - start_time + communication,
                                       np.max(agents[i].local_population_fit)])
  return avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication


def co_evolution_with_netlag(interact_time, independent_time, absolute_time, at_valid=False):
  """
  有通信延迟的情况下的协同进化过程
  :param interact_time: 交互次数
  :param independent_time: 独立进化次数
  :param absolute_time: 绝对时间
  :param at_valid: 绝对时间是否有效，无效时按照进化次数进行迭代
  :return:
  """
  agents = []
  # 画图用
  avg_popu_fitness_all = []
  best_popu_fitness_all = []

  paralle_compute = 0
  communication = 0

  # 并行段0
  time_stamp = time.time() * 1000  # 毫秒级时间戳
  start_time = time_stamp
  for i in range(AGENT_NUM):
    agent = Agent(i, TASK_NUM, INIT_POPU_NUM, AGENT_NUM - 1)
    agent.get_init_local_population()
    agents.append(agent)
    avg_popu_fitness_all.append([])
    best_popu_fitness_all.append([])
  # 并行段0结束
  paralle_compute += time.time() * 1000 - time_stamp

  if at_valid:
    # 交互迭代次数，为了跟踪当前是否可以通信
    it_time = 0
    while paralle_compute / AGENT_NUM + communication < absolute_time:
      # print(paralle_compute + communication)
      # 并行段1
      time_stamp = time.time() * 1000
      # 解共享池
      share_pool = np.zeros((AGENT_NUM, SHARE_NUM, TASK_NUM), dtype=np.bool_)
      share_pool_fit = np.zeros((AGENT_NUM, SHARE_NUM), dtype=np.float_)
      for i in range(AGENT_NUM):
        # 每个agent分别单独进化10次

        agents[i].evolution(independent_time)
        # evolution操作之后种群和fitness都是排好序的
        if NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
          continue  # 对于有延迟的agent，延迟周期内不能共享
        share_pool[i], share_pool_fit[i] = agents[i].selection_for_share(SHARE_NUM)

      # 假设采用广播机制，形成share_pool的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY
      paralle_compute += time.time() * 1000 - time_stamp
      time_stamp = time.time() * 1000

      for i in range(AGENT_NUM):
        # 对于有延迟的agent，延迟周期内不能获取其他agent的共享数据
        if NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
          if agents[i].local_population_fit:
            avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
            best_popu_fitness_all[i].append([paralle_compute / AGENT_NUM + communication,
                                             agents[i].local_population_fit[0]])
          continue
        # 每个agent从共享池中拿其他agent的解
        remote_popus = get_remote_popu_from_share_pool(share_pool, i)
        agents[i].cooperation_all(remote_popus)
        # coop操作之后种群和fitness都是排好序的
        if agents[i].local_population_fit:
          avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
          best_popu_fitness_all[i].append([paralle_compute / AGENT_NUM + communication,
                                           agents[i].local_population_fit[0]])

      # 假设采用广播机制，agent共享最优全局解的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY
      # 选出适应度最好的完整解，与大家共享
      temp_best_intact_popu_fit = []
      for i in range(AGENT_NUM):
        if NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
          # 对于有延迟的agent，延迟周期内不能与其他agent共享数据
          # 因此取共享池内的对应数据， 按照算法逻辑第一个是最好的
          temp_best_intact_popu_fit.append(share_pool_fit[i][0])
        elif agents[i].local_population_fit:
          temp_best_intact_popu_fit.append(agents[i].local_population_fit[0])
        else:
          temp_best_intact_popu_fit.append(0.0)
      best_agent_index = np.argmax(temp_best_intact_popu_fit)
      # 如果最优的刚好是延迟的agent
      if NETLOG_AGENT[best_agent_index] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
        # 对于有延迟的agent，延迟周期内不能与其他agent共享数据
        # 目前最优解是上一轮的延迟解，说明本轮各agent进化的很差
        # 直接进入下一轮
        # 并行段3结束
        paralle_compute += time.time() * 1000 - time_stamp
        continue
      else:
        best_fitness = agents[best_agent_index].local_population_fit[0]
        best_intact_popu = np.insert(agents[best_agent_index].best_remote_popu, best_agent_index,
                                     agents[best_agent_index].local_population[0], axis=0)

      for i in range(AGENT_NUM):
        if i == best_agent_index or (NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0):
          continue
        agents[i].update_self_by_best_intact_popu(best_intact_popu, best_fitness)
      # 并行段3结束
      paralle_compute += time.time() * 1000 - time_stamp
      it_time += 1
  else:
    for it_time in range(interact_time):  # 交互次数
      # 并行段1
      time_stamp = time.time() * 1000
      # 解共享池
      share_pool = np.zeros((AGENT_NUM, SHARE_NUM, TASK_NUM), dtype=np.bool_)
      share_pool_fit = np.zeros((AGENT_NUM, SHARE_NUM), dtype=np.float_)
      for i in range(AGENT_NUM):
        # 每个agent分别单独进化10次
        agents[i].evolution(independent_time)
        # evolution操作之后种群和fitness都是排好序的
        if NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
          continue  # 对于有延迟的agent，延迟周期内不能共享
        share_pool[i], share_pool_fit[i] = agents[i].selection_for_share(SHARE_NUM)

      # 假设采用广播机制，形成share_pool的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY
      paralle_compute += time.time() * 1000 - time_stamp
      time_stamp = time.time() * 1000
      for i in range(AGENT_NUM):
        # 对于有延迟的agent，延迟周期内不能获取其他agent的共享数据
        if NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
          if agents[i].local_population_fit:
            avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
            best_popu_fitness_all[i].append([paralle_compute / AGENT_NUM + communication,
                                             agents[i].local_population_fit[0]])
          continue
        # 每个agent从共享池中拿其他agent的解
        remote_popus = get_remote_popu_from_share_pool(share_pool, i)
        agents[i].cooperation_all(remote_popus)
        # coop操作之后种群和fitness都是排好序的
        if agents[i].local_population_fit:
          avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
          best_popu_fitness_all[i].append([paralle_compute / AGENT_NUM + communication,
                                           agents[i].local_population_fit[0]])

      # 假设采用广播机制，agent共享最优全局解的时间为端到端时延 50ms
      communication += COMMUNACATION_DELAY
      # 选出适应度最好的完整解，与大家共享
      temp_best_intact_popu_fit = []
      for i in range(AGENT_NUM):
        if NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
          # 对于有延迟的agent，延迟周期内不能与其他agent共享数据
          # 因此取共享池内的对应数据， 按照算法逻辑第一个是最好的
          temp_best_intact_popu_fit.append(share_pool_fit[i][0])
        elif agents[i].local_population_fit:
          temp_best_intact_popu_fit.append(agents[i].local_population_fit[0])
        else:
          temp_best_intact_popu_fit.append(0.0)
      best_agent_index = np.argmax(temp_best_intact_popu_fit)
      # 如果最优的刚好是延迟的agent
      if NETLOG_AGENT[best_agent_index] == 1 and (it_time + 1) % NETLOG_PERIOD != 0:
        # 对于有延迟的agent，延迟周期内不能与其他agent共享数据
        # 目前最优解是上一轮的延迟解，说明本轮各agent进化的很差
        # 直接进入下一轮
        # 并行段3结束
        paralle_compute += time.time() * 1000 - time_stamp
        continue
      else:
        best_fitness = agents[best_agent_index].local_population_fit[0]
        best_intact_popu = np.insert(agents[best_agent_index].best_remote_popu, best_agent_index,
                                     agents[best_agent_index].local_population[0], axis=0)

      for i in range(AGENT_NUM):
        if i == best_agent_index or (NETLOG_AGENT[i] == 1 and (it_time + 1) % NETLOG_PERIOD != 0):
          continue
        agents[i].update_self_by_best_intact_popu(best_intact_popu, best_fitness)
      # 并行段3结束
      paralle_compute += time.time() * 1000 - time_stamp

  # for k in range(AGENT_NUM):
  #     best = agents[k].get_best_intact_indi()
  #     print('agent', k, '的最大收益：%0.5f' % agents[k].local_population_fit[0],
  #           '\n最优分配方式：\n', best)
  #
  # print('并行计算时间：%0.2f' % (paralle_compute / 3000),
  #       '\n通讯消耗时间：', communication / 1000)

  # i = 0
  # for avg_popu_fitness in avg_popu_fitness_all:
  #     plt.plot(avg_popu_fitness, color=cnames[i])
  #     i += 1
  # plt.show()
  if len(best_popu_fitness_all) == 0:
    for i in range(AGENT_NUM):
      avg_popu_fitness_all[i].append(np.mean(agents[i].local_population_fit))
      best_popu_fitness_all[i].append([time.time() * 1000 - start_time + communication,
                                       np.max(agents[i].local_population_fit)])
  return avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication


if __name__ == '__main__':
  np.random.seed(0)
  avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication = \
      co_evolution(INTERACT_TIMES, INDEPENDENT_TIMES, 0, False)
  # 第一个图：折线图
  plt.subplot(2, 1, 1)
  plt.title("avg_fit")
  # 画分布式
  i = 0
  for avg_popu_fit in avg_popu_fitness_all:
    X = [i * INDEPENDENT_TIMES * AGENT_NUM for i in range(1, len(avg_popu_fit) + 1)]
    Y = avg_popu_fit
    plt.plot(X, Y, color=cnames[i], label="agent%d avg_fit" % i)
    i += 1
  plt.xlabel("popu iter times")
  plt.ylabel("popu average fitness")
  plt.legend()

  # 第2个图：折线图
  plt.subplot(2, 1, 2)
  plt.title("best_fit")
  # 画分布式
  i = 0
  for best_popu_fit in best_popu_fitness_all:
    X = [i[0] for i in best_popu_fit]
    Y = [i[1] for i in best_popu_fit]
    plt.plot(X, Y, color=cnames[i], label="agent%d best_fit" % i)
    i += 1
  plt.xlabel("run time")
  plt.ylabel("popu best fitness")
  plt.legend()

  plt.tight_layout(pad=1.08)
  plt.show()
