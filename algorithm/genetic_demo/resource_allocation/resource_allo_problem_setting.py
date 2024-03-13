"""
资源分配问题的超参数设置
资源分配解决的是任务的资源个数选择问题
每个任务可能需要多个Agent承担
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
MAX_TASK_NUM_PER_AGENT = 1  # 每个Agent能承担的最大任务数
TASK_PROFIT_INDEPENDENT = True  # Agent承担多个任务时，收益是否独立
# 任务收益矩阵，如果每个task给足够的资源能够获得的效益
# PROFIT = [4000, 2000,  4000,  4000,  36,  81]
PROFIT = [4000, 5000, 4000, 4000]
# Agent 资源个数
# RESOURCES = [32, 8,  4,  26,  20, 10, 10, 10, 10, 10]
RESOURCES = [30, 10, 10, 10, 30, 10, 10, 10, 10, 10]
# 任务所需资源个数
# REQUIREMENT = [30, 40, 30, 30, 7, 14]
REQUIREMENT = [40, 40, 30, 30]
# 代价矩阵，该Agent承担该任务需要付出的代价（除资源外）
# COST = [[88.044, 90.505,  0,  6,  6,  1],
#         [76.693, 50.344,  9, 22,  0, 22],
#         [76.826, 38.524,  5, 12,  5, 12],
#         [99.293,  19.517,  2,  0,  7, 12],
#         [ 1, 10,  8, 13, 16,  6],
#         [10,  3,  0,  8,  6, 18],
#         [15,  9,  7,  3, 16,  8],
#         [ 8, 12, 17,  1,  7,  8],
#         [ 4, 11,  9,  2, 19, 17]]

PC = 1
PM = 1
CM = 0.5
COMMUNACATION_DELAY = 50  # 每次共享数据的通信耗时

TASK_NUM = 4
AGENT_NUM = 10
INTERACT_TIMES = 30  # 协同进化次数
INDEPENDENT_TIMES = 4  # 独立进化次数

# 通信延迟参数
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
