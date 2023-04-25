#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New_Version
@File    ：boxplot.py
@Author  ：JacQ
@Date    ：2022/5/25 15:07
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.figure(figsize=(8, 6))
# plt.style.use('classic')  # seaborn-dark-palette
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.grid(linestyle='--')
plt.tick_params(labelsize=12)
datafile = u"/Users/jacq/Desktop/PortScheduling/实验结果(已自动还原).xlsx"
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
font_label = {'family': 'Times New Roman',
              'weight': 'medium',
              'color': 'black',
              'size': 13
              }
# 任务10
mission_num = 1000
data = pd.read_excel(datafile, sheet_name='Z2_' + str(mission_num) + 'p')
plt.ylabel('Makespan', fontdict=font_label)
# .plot.box(grid=True, flierprops=fliers)
f = data.boxplot(patch_artist=True, return_type='dict')
for box in f['boxes']:
    # 箱体边框颜色
    box.set(color='#7570b3', linewidth=2)
    # 箱体内部填充颜色
    box.set(facecolor='#1b9e77', alpha=0.8)  # '#1b9e77'
for whisker in f['whiskers']:
    whisker.set(color='r', linestyle='--', linewidth=1.8)
for cap in f['caps']:
    cap.set(color='darkgreen', linewidth=2)
for median in f['medians']:
    median.set(color='DarkBlue', linewidth=2)
for flier in f['fliers']:
    flier.set(marker='o', color='b', markeredgecolor='black', alpha=0.5, markersize=2.0)
plt.show()
# 任务10
# mission_num = 100
# data = pd.read_excel(datafile, sheet_name='Z2_' + str(mission_num) + 'p')
# axes[1].set_ylabel('Makespan')
# data.plot.box(grid=True, ax=axes[1], flierprops=fliers)

# # 任务10
# mission_num = 500
# data = pd.read_excel(datafile, sheet_name=str(mission_num))
#
# axes[1][0].set_title(str(mission_num) + " containers", fontsize=9)  # 标题，并设定字号大小
# axes[1][0].set_ylabel('makespan')
# data = data[:][0:5]
# color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
# data.plot.box(grid=True,
#               color=color,
#               ax=axes[1][0])
# # 任务10
# mission_num = 1000
# data = pd.read_excel(datafile, sheet_name=str(mission_num))

# axes[1][1].set_title(str(mission_num) + " containers", fontsize=9)  # 标题，并设定字号大小
# axes[1][1].set_ylabel('makespan')
# data = data[:][0:5]
# color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
# data.plot.box(grid=True,
#               color=color,
#               ax=axes[1][1])
# # 任务10
# mission_num = 100
# data = pd.read_excel(datafile, sheet_name=str(mission_num))
# box_1, box_2, box_3, box_4, box_5 = data['RC'], data['LW'], data['LN'], data['LD'], data['LA']
# box_6, box_7, box_8, box_9 = data['SA-10'], data['SA-60'], data['SA-180'], data['SA'],
#
# axes[0][1].set_title(str(mission_num * 3) + " containers", fontsize=15)  # 标题，并设定字号大小
# labels = 'RC', 'LW', 'LN', 'LD', 'LA'  # 图例
# axes[0][1].boxplot([box_1, box_2, box_3, box_4, box_5], vert=True, labels=labels,
#                    patch_artist=True)  # grid=False：代表不显示背景中的网格线
#
# # 任务500
# mission_num = 10
# data = pd.read_excel(datafile, sheet_name=str(mission_num))
# box_1, box_2, box_3, box_4, box_5 = data['RC'], data['LW'], data['LN'], data['LD'], data['LA']
# box_6, box_7, box_8, box_9 = data['SA-10'], data['SA-60'], data['SA-180'], data['SA'],
#
# axes[1][0].set_title(str(mission_num * 3) + " containers", fontsize=15)  # 标题，并设定字号大小
# labels = 'RC', 'LW', 'LN', 'LD', 'LA'  # 图例
# axes[1][0].boxplot([box_1, box_2, box_3, box_4, box_5], vert=True, labels=labels,
#                    patch_artist=True)  # grid=False：代表不显示背景中的网格线
#
# # 任务1000
# mission_num = 100
# data = pd.read_excel(datafile, sheet_name=str(mission_num))
# box_1, box_2, box_3, box_4, box_5 = data['RC'], data['LW'], data['LN'], data['LD'], data['LA']
# box_6, box_7, box_8, box_9 = data['SA-10'], data['SA-60'], data['SA-180'], data['SA'],
#
# axes[1][1].set_title(str(mission_num * 3) + " containers", fontsize=15)  # 标题，并设定字号大小
# labels = 'RC', 'LW', 'LN', 'LD', 'LA'  # 图例
# axes[1][1].boxplot([box_1, box_2, box_3, box_4, box_5], vert=True, labels=labels,
#                    patch_artist=True)  # grid=False：代表不显示背景中的网格线

#
# data = pd.read_excel(datafile, sheet_name='line')
# plt.figure(dpi=110)
# sns.set(style="whitegrid", font_scale=1.2)
# g = sns.regplot(x='instance', y='RC', data=data,
#                 marker='.',
#                 order=4,  # 默认为1，越大越弯曲
#                 scatter_kws={'s': 60, 'color': '#016392', 'alpha': 0 },  # 设置散点属性，参考plt.scatter
#                 line_kws={'linestyle': '--', 'color': '#01a2d9'}  # 设置线属性，参考 plt.plot
#                 )
# g2 = sns.regplot(x='instance', y='LW', data=data,
#                  marker='.',
#                  order=4,  # 默认为1，越大越弯曲
#                  scatter_kws={'s': 60, 'color': 'white', 'alpha': 0},  # 设置散点属性，参考plt.scatter
#                  line_kws={'linestyle': '--', 'color': '#c72e29'}  # 设置线属性，参考 plt.plot
#                  )
# g3 = sns.regplot(x='instance', y='LN', data=data,
#                  marker='.',
#                  order=4,  # 默认为1，越大越弯曲
#                  scatter_kws={'s': 60, 'color': '#016392', },  # 设置散点属性，参考plt.scatter
#                  line_kws={'linestyle': '--', 'color': 'green'}  # 设置线属性，参考 plt.plot
#                  )
# g4 = sns.regplot(x='instance', y='LA', data=data,
#                  marker='.',
#                  order=4,  # 默认为1，越大越弯曲
#                  scatter_kws={'s': 60, 'color': '#016392', },  # 设置散点属性，参考plt.scatter
#                  line_kws={'linestyle': '--', 'color': '#c72e29'}  # 设置线属性，参考 plt.plot
#                  )
