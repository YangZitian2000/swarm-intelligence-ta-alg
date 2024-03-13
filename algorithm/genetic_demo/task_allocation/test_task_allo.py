from ma_task_allo import *
from centralized_task_allo import *
import matplotlib.pyplot as plt
import numpy as np


def test_result_quality_in_same_iter_times():
  test_repeat_time = 5
  distributed_count = 0
  central_count = 0
  for i in range(test_repeat_time):
    total_evolu_times = INTERACT_TIMES * INDEPENDENT_TIMES * AGENT_NUM
    avg_popu_fitness, best_popu_fitness = cen_evolution(total_evolu_times, 0)
    avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication =\
        co_evolution(INTERACT_TIMES, INDEPENDENT_TIMES, 0)
    # 集中式最大收益
    central_best = best_popu_fitness[-1][1]
    # 分布式最大收益
    distributed_best = best_popu_fitness_all[-1][-1][1]
    if distributed_best > central_best:
      distributed_count += 1
    elif distributed_best < central_best:
      central_count += 1
  print('分布式好的比例：%0.2f' % (distributed_count / test_repeat_time))
  print('集中式好的比例：%0.2f' % (central_count / test_repeat_time))
  print('持平的比例：%0.2f' % ((test_repeat_time - distributed_count - central_count) / test_repeat_time))


def test_result_quality_in_same_absolute_time():
  test_repeat_time = 20
  run_time = 5000
  distributed_count = 0
  central_count = 0
  for i in range(test_repeat_time):
    total_evolu_times = INTERACT_TIMES * INDEPENDENT_TIMES * AGENT_NUM
    avg_popu_fitness, best_popu_fitness = cen_evolution(total_evolu_times, run_time, at_valid=True)
    avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication = \
        co_evolution(INTERACT_TIMES, INDEPENDENT_TIMES, run_time, at_valid=True)
    # 集中式最大收益
    central_best = best_popu_fitness[-1][1]
    # 分布式最大收益
    distributed_best = best_popu_fitness_all[-1][-1][1]
    if distributed_best > central_best:
      distributed_count += 1
    elif distributed_best < central_best:
      central_count += 1
  print('分布式好的比例：%0.2f' % (distributed_count / test_repeat_time))
  print('集中式好的比例：%0.2f' % (central_count / test_repeat_time))
  print('持平的比例：%0.2f' % ((test_repeat_time - distributed_count - central_count) / test_repeat_time))


def plot_result(ab_time, run_time):
  total_evolu_times = INTERACT_TIMES * INDEPENDENT_TIMES * AGENT_NUM

  avg_popu_fitness, best_popu_fitness = cen_evolution(total_evolu_times, run_time, ab_time)

  avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication = \
      co_evolution(INTERACT_TIMES, INDEPENDENT_TIMES, run_time, ab_time)
  # 第一个图：折线图
  plt.subplot(2, 1, 1)
  plt.title("avg_fit")
  # 画集中式
  X = [i for i in range(1, len(avg_popu_fitness) + 1)]
  Y = avg_popu_fitness
  plt.plot(X, Y, c="r", label="central avg_fit")
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
  # 画集中式
  X = [i[0] for i in best_popu_fitness]
  Y = [i[1] for i in best_popu_fitness]
  plt.plot(X, Y, c="g", label="central best_fit")
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


def test_time_consume_in_same_iter_times():
  test_repeat_time = 20
  central_compute_time = []
  distributed_compute_time = []
  distributed_communicate_time = []
  for i in range(test_repeat_time):
    total_evolu_times = INTERACT_TIMES * INDEPENDENT_TIMES * AGENT_NUM
    avg_popu_fitness, best_popu_fitness = cen_evolution(total_evolu_times, 0)

    avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication = \
        co_evolution(INTERACT_TIMES, INDEPENDENT_TIMES, 0)

    central_compute_time.append(best_popu_fitness[-1][0])
    distributed_compute_time.append(paralle_compute / AGENT_NUM)
    distributed_communicate_time.append(communication)

  print('分布式并行计算用时：%0.2f' % np.mean(distributed_compute_time))
  print('分布式通信用时：%0.2f' % np.mean(distributed_communicate_time))
  print('集中式计算用时：%0.2f' % np.mean(central_compute_time))


def test_time_consume_in_same_absolute_time():
  test_repeat_time = 20
  run_time = 10000
  central_compute_time = []
  distributed_compute_time = []
  distributed_communicate_time = []
  for i in range(test_repeat_time):
    total_evolu_times = INTERACT_TIMES * INDEPENDENT_TIMES * AGENT_NUM
    avg_popu_fitness, best_popu_fitness = cen_evolution(total_evolu_times, run_time, at_valid=True)
    avg_popu_fitness_all, best_popu_fitness_all, paralle_compute, communication = \
        co_evolution(INTERACT_TIMES, INDEPENDENT_TIMES, run_time, at_valid=True)

    central_compute_time.append(best_popu_fitness[-1][0])
    distributed_compute_time.append(paralle_compute / AGENT_NUM)
    distributed_communicate_time.append(communication)
  print('分布式并行计算用时：%0.2f' % np.mean(distributed_compute_time))
  print('分布式通信用时：%0.2f' % np.mean(distributed_communicate_time))
  print('集中式计算用时：%0.2f' % np.mean(central_compute_time))


def test_netlong_impact_in_same_iter_times():
  test_repeat_time = 40
  no_netlog = 0
  with_netlog = 0
  for i in range(test_repeat_time):
    # total_evolu_times = INTERACT_TIMES * INDEPENDENT_TIMES * AGENT_NUM
    # 无延迟
    avg_popu_fitness_all_0, best_popu_fitness_all_0, paralle_compute_0, communication_0 = \
        co_evolution(INTERACT_TIMES, INDEPENDENT_TIMES, 0)

    # 有延迟
    avg_popu_fitness_all_1, best_popu_fitness_all_1, paralle_compute_1, communication_1 = \
        co_evolution_with_netlag(INTERACT_TIMES, INDEPENDENT_TIMES, 0)
    # 分布式最大收益
    no_netlog_best = best_popu_fitness_all_0[-1][-1][1]
    with_netlog_best = best_popu_fitness_all_1[-1][-1][1]
    if no_netlog_best > with_netlog_best:
      no_netlog += 1
    elif no_netlog_best < with_netlog_best:
      with_netlog += 1
  print('无延迟好的比例：%0.2f' % (no_netlog / test_repeat_time))
  print('有延迟好的比例：%0.2f' % (with_netlog / test_repeat_time))
  print('持平的比例：%0.2f' % ((test_repeat_time - no_netlog - with_netlog) / test_repeat_time))


if __name__ == '__main__':
  plt.figure(figsize=(19.2, 9))
  plot_result(False, 10000)
  # test_result_quality_in_same_absolute_time()
  test_result_quality_in_same_iter_times()
  # test_time_consume_in_same_iter_times()
  # test_time_consume_in_same_absolute_time()
  # test_netlong_impact_in_same_iter_times()
