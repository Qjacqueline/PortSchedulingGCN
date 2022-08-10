#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：port_scheduling
@File    ：configs.py
@Author  ：JacQ
@Date    ：2021/12/1 16:56
"""
import logging
import os

import numpy as np
import torch

LOGGING_LEVEL = logging.INFO  # logging.WARNING/DEBUG

dataset = 'v2'
# 布局配置
QUAY_EXIT = np.array([280, 0])  # 岸桥操作后小车出口坐标（单位：m）
QUAYCRANE_EXIT_SPACE = 40  # 出口距离最重起重机间距（单位：m）
QUAYCRANE_CRANE_SPACE = 45  # 起重机与起重机间间距（单位：m）
A1_LOCATION = np.array([20, 128])  # A1箱区位置
SLOT_LENGTH = 6.5  # 槽位长度（单位：m）
SLOT_WIDTH = 2.44  # 槽位宽度（单位：m）
SLOT_NUM_X = 30  # x方向槽位数
SLOT_NUM_Y = 8  # y方向槽位数
LANE_X = 40  # x方向车道宽度
LANE_Y = 12  # y方向车道宽度
BLOCK_SPACE_X = SLOT_LENGTH * SLOT_NUM_X + LANE_X  # x方向堆场间距（单位：m）
BLOCK_SPACE_Y = SLOT_WIDTH * SLOT_NUM_Y + LANE_Y  # y方向堆场间距（单位：m）
S1_STATION_LOCATION = np.array([180, 64])  # 第一个锁站所在位置
LOCK_STATION_SPACE = 150  # 锁站间距 150 TODO
LOCK_STATION_BUFFER_SPACE = 5  # 等待区距离锁站的垂直间距
FIRST_BUFFER_TO_FIRST_LOCK_STATION = 96  # 第一个锁站到第一个缓冲区的距离
MISSION_NUM_ONE_QUAY_CRANE = 500  # 一个场桥对应的任务数 TODO
CRANE_NUM = 3  # 场桥个数
MISSION_NUM = CRANE_NUM * MISSION_NUM_ONE_QUAY_CRANE  # 任务个数
QUAY_BUFFER_SIZE = 5  # 岸桥缓冲区可存放个数
STAGE_NUM = 3  # lock_station+crossover+yard
CROSSOVER_NUM = 3
# YARD_CRANE_NUM = 5  # (300-4)

# 机器运行参数
QUAY_CRANE_RELEASE_TIME = 120  # 岸桥释放集装箱任务时间间隔 TODO 要160s释放
# QUAYCRANE_PROCESS_TIME = [38, 70]  # 岸桥放下并装载集装箱时间服从U(38,70)分布（单位：秒）
BUFFER_PROCESS_TIME = 60  # 缓冲区操作所需时间（单位：秒）
LOCK_STATION_NUM = 4  # 锁站个数（单位：辆）
# LOCK_STATION_CAPACITY = 1  # 锁站处理能力（单位：辆）
LOCK_STATION_HANDLING_TIME = [100, 150]  # 解锁所需时间（单位：s） [100, 150] TODO
WAIT_TIME_DELAY = [0, 0, 0, 0]  # 由于停留在锁站缓冲区所增加的等待时间[30, 32, 50, 52]
CROSSOVER_CAPACITY = 4  # 交叉口通行能力（单位：辆）
CROSSOVER_MAX_WAIT_TIME = 20  # 交叉口车辆最大等待时间（s）
YARDCRANE_SPEED_X = 2.17  # 场桥x方向移动速度（单位：m/s）2.17
YARDCRANE_SPEED_Y = 1.8  # 场桥y方向移动速度（单位：m/s）1.80
MAX_MOVE_TIME = BLOCK_SPACE_X / YARDCRANE_SPEED_X + BLOCK_SPACE_Y / YARDCRANE_SPEED_Y
YARDCRANE_HANDLING_TIME = [30, 30]  # 场桥放下并装载集装箱时间服从U(25, 35)分布（单位：秒）TODO
VEHICLE_SPEED = [0.24, 2.63]  # AGV运行速度服从U(6,9)分布（单位m/s）

# 算法参数
RANDOM_SEED = 10
INITIAL_EPSILON = 0.8
RL_CONFIG = 6
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

# 文件路径
ROOT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_FOLDER_PATH, 'data/data_' + dataset)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
OUTPUT_RESULT_PATH = os.path.join(ROOT_FOLDER_PATH, 'output_result')  # output_result文件底下
OUTPUT_PATH = os.path.join(OUTPUT_RESULT_PATH, 'output_' + str(MISSION_NUM_ONE_QUAY_CRANE) + '.json')
LAYOUT_PATH = os.path.join(OUTPUT_RESULT_PATH, 'layout.png')
MODEL_PATH = os.path.join(OUTPUT_RESULT_PATH, 'model')
LOSS_PLOT_PATH = os.path.join(OUTPUT_RESULT_PATH, 'loss_')
OUTPUT_SOLUTION_PATH = os.path.join(OUTPUT_RESULT_PATH, 'solution_' + dataset + '/')

# SA
T0 = 4000
TEND = 1e-3
RATE = 0.995
ACTION_NUM_SA = 4

# RL_LA1
N_STEPS_LA1 = 3000  # 每一个epoch里挑选action的step数
N_EPOCH_LA1 = 2  # 同一个实例train的epoch数
NUM_NODE_FEATURES_LA1 = 3  # node的特征个数
ACTION_NUM_LA1 = 4  # action的个数
POLICY_LR_LA1 = 1e-6  # actor网络的学习率
VF_LR_LA1 = 1e-5  # critic网络的学习率,应该比actor大
GAMMA_LA1 = 0.98  # 衰减率
# RL_LA2
N_STEPS_LA2 = MISSION_NUM  # 每一个epoch里挑选action的step数
N_EPOCH_LA2 = 200  # 同一个实例train的epoch数
FEATURE_SIZE_QUAY = 6  # RNN yard点特征数
MAX_LENGTH_QUAY = 6  # RNN yard序列最多点数
FEATURE_SIZE_MACHINE = 3  # RNN 其他机器点特征数
MAX_LENGTH_MACHINE = MISSION_NUM + 1  # RNN 其他机器序列最多点数
ACTION_NUM_LA2 = LOCK_STATION_NUM  # action的个数
HIDDEN_SIZE = 32  # RNN隐藏层元个数
N_LAYERS = 2  # RNN层数
POLICY_LR_LA2 = 1e-6  # actor网络的学习率
VF_LR_LA2 = 1e-5  # critic网络的学习率,应该比actor大
GAMMA_LA2 = 0.9  # 衰减率
ACCEPT_PROB_LA2 = 0.99  # 差解接受概率
# RL_LA3
N_STEPS_LA3 = MISSION_NUM  # 每一个epoch里挑选action的step数
N_EPOCH_LA3 = 100  # 同一个实例train的epoch数
BATCH_SIZE_LA3 = 32
LR_LA3 = 1e-2
GAMMA_LA3 = 0.90
EPSILON_LA3 = 0.8
MEMORY_CAPACITY_LA3 = 2000
Q_NETWORK_ITERATION_LA3 = 100
NUM_NODE_FEATURES_LA3 = 3  # node的特征个数
ACTION_NUM_LA3 = LOCK_STATION_NUM  # action的个数
# RL_LA4
N_STEPS_LA4 = MISSION_NUM  # 每一个epoch里挑选action的step数
N_EPOCH_LA4 = 100  # 同一个实例train的epoch数
NUM_NODE_FEATURES_LA4 = 3  # node的特征个数
ACTION_NUM_LA4 = LOCK_STATION_NUM  # action的个数4
POLICY_LR_LA4 = 1e-7  # actor网络的学习率
VF_LR_LA4 = 1e-6  # critic网络的学习率,应该比actor大
GAMMA_LA4 = 0.98  # 衰减率
# RL_LA5
N_STEPS_LA5 = MISSION_NUM  # 每一个epoch里挑选action的step数
N_EPOCH_LA5 = 300  # 同一个实例train的epoch数
BATCH_SIZE_LA5 = 10
LR_LA5 = 0.0001
GAMMA_LA5 = 0.90
EPSILON_LA5 = 0.8
MEMORY_CAPACITY_LA5 = 2000
Q_NETWORK_ITERATION_LA5 = 100
NUM_NODE_FEATURES_LA5 = 3  # node的特征个数
ACTION_NUM_LA5 = LOCK_STATION_NUM  # action的个数
# RL_LA6
N_STEPS_LA6 = MISSION_NUM  # 每一个epoch里挑选action的step数
N_EPOCH_LA6 = 60  # 同一个实例train的epoch数
BATCH_SIZE_LA6 = 40
LR_LA6 = 1e-4
GAMMA_LA6 = 0.90
EPSILON_LA6 = 0.8
MEMORY_CAPACITY_LA6 = 2000
Q_NETWORK_ITERATION_LA6 = 100
NUM_NODE_FEATURES_LA6 = 3  # node的特征个数
ACTION_NUM_LA6 = LOCK_STATION_NUM  # action的个数
TARGET_NETWORK_REPLACE_FREQ = 10  # 目标网络替代频率
UPDATE_NUM_LA6 = 5
# RL_UA1
TRAIN_BATCH_SIZE_UA1 = 200
EMBEDDING_SIZE_UA1 = 128
HIDDENS_UA1 = 512
NOF_LSTMS_UA1 = 2
DROPOUTS_UA1 = 0.
BIDIR_UA1 = True
MISSION_ATTRIBUTE_NUM_UA1 = 7
N_EPOCH_UA1 = 1000
# RL_UA2
N_EPOCH_UA2 = 100
ACTION_NUM_UA2 = QUAY_BUFFER_SIZE * CRANE_NUM
MISSION_ATTRIBUTE_NUM_UA2 = 7
POLICY_LR_UA2 = 1e-6  # actor网络的学习率
VF_LR_UA2 = 1e-5  # critic网络的学习率,应该比actor大
GAMMA_UA2 = 0.9  # 衰减率
UPDATE_NUM_UA2 = 5
MEMORY_CAPACITY_UA2 = 2000
BATCH_SIZE_UA2 = 32
