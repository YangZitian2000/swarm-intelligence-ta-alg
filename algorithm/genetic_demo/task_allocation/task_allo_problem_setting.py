import random
random.seed(0)
"""
任务分配问题的超参数设置
任务分配解决的是任务的0、1选择问题
并且每个任务仅需要一个Agent承担
每个Agent可承担多个任务
"""
cnames = [
    'antiquewhite', 'aqua', 'aquamarine',
    'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
    'sienna', 'silver', 'skyblue', 'slateblue', 'slategray',
    'snow', 'springgreen', 'steelblue', 'tan''teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
    'yellow', 'yellowgreen']

INIT_POPU_NUM = 8
SHARE_NUM = 2
MAX_TASK_NUM_PER_AGENT = 5  # 每个Agent能承担的最大任务数
TASK_PROFIT_INDEPENDENT = True  # 任务收益是否独立


TASK_NUM = 10
AGENT_NUM = 3
# 任务收益矩阵，行代表agent，列代表task
PROFIT = [[random.randint(0, 50) for j in range(TASK_NUM)] for i in range(AGENT_NUM)]

PC = 0.8
PM = 0.8
CM = 0.5
COMMUNACATION_DELAY = 50  # 每次共享数据的通信耗时


INTERACT_TIMES = 30  # 协同进化次数
INDEPENDENT_TIMES = 5  # 独立进化次数

# 通信延迟参数
NETLOG_PERIOD = 1  # 每个节点的通信延迟周期（按照迭代次数算），即该周期后一定能通联
NETLOG_AGENT = [0, 0, 0, 1, 0]  # 有通信延迟的agent， 1表示延迟，0表示无延迟
