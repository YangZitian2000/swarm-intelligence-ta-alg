import random
random.seed(0)
"""
资源分配问题的超参数设置
资源分配解决的是任务的资源个数选择问题
每个任务可能需要多个Agent承担
"""

INIT_POPU_NUM = 8
SHARE_NUM = 2
MAX_TASK_NUM_PER_AGENT = 1  # 每个Agent能承担的最大任务数
TASK_PROFIT_INDEPENDENT = True  # Agent承担多个任务时，收益是否独立

TASK_NUM = 4
AGENT_NUM = 10
# 任务收益矩阵，如果每个task给足够的资源能够获得的效益
PROFIT = [random.randint(3000, 5000) for i in range(TASK_NUM)]
# Agent 资源个数
RESOURCES = [random.randint(10, 30) for i in range(AGENT_NUM)]
# 任务所需资源个数
REQUIREMENT = [random.randint(30, 60) for i in range(TASK_NUM)]
# 代价矩阵，该Agent承担该任务需要付出的代价（除资源外）
COST = [[random.randint(0, 50) for j in range(TASK_NUM)] for i in range(AGENT_NUM)]

PC = 0.8
PM = 0.8
CM = 0.5

INTERACT_TIMES = 30  # 协同进化次数
INDEPENDENT_TIMES = 4  # 独立进化次数

# 通信延迟参数
COMMUNACATION_DELAY = 50  # 每次共享数据的通信耗时
NETLOG_PERIOD = 5  # 每个节点的通信延迟周期（按照迭代次数算），即该周期后一定能通联
NETLOG_AGENT = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1]  # 有通信延迟的agent， 1表示延迟，0表示无延迟

AGENT_LOC = [[35, 340],
             [30, 340],
             [35, 345],
             [35, 335],
             [410, 380],
             [415, 405],
             [430, 370],
             [210, 15],
             [195, 20],
             [210, 20]]
TASK_LOC = [[245.0, 260.0],
            [255.0, 260.0],
            [245.0, 250.0],
            [255.0, 250.0]]

cnames = [
    'antiquewhite', 'aqua', 'aquamarine',
    'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
    'sienna', 'silver', 'skyblue', 'slateblue', 'slategray',
    'snow', 'springgreen', 'steelblue', 'tan''teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
    'yellow', 'yellowgreen']
