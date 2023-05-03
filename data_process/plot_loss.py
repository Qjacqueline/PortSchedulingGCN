import pandas as pd
from pandas import DataFrame
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns


def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val


# 平滑处理，类似tensorboard的smoothing函数。
def smooth(read_path, save_path, file_name, x='timestep', y='reward', weight=0.75):
    data = pd.read_csv(read_path + file_name)
    scalar = data[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({x: data[x].values, y: smoothed})
    save.to_csv(save_path + 'smooth_' + file_name)


def draw_loss(vals, weight):
    plt.figure()
    plt.style.use('classic')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    rc = {'font.sans-serif': ['Times New Roman']}
    plt.xlim((0, 10000))
    plt.rcParams.update({'font.size': 16})
    x = [[i.step * 20 for i in vals[0]], [i.step * 2 for i in vals[1]]]
    y = [[j.value / 50.0 for j in val] for val in vals]
    save = []
    tl = 293
    for yy in y:
        last = yy[0]
        smoothed = []
        for point in yy[0:tl]:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        save.append(DataFrame({'Training episodes': x[0][0:tl], 'Loss': smoothed}))
    save1, save2 = save[0], save[1]
    original1, original2 = DataFrame({'Training episodes': x[0][0:tl], 'Loss': y[0][0:tl]}), \
                           DataFrame({'Training episodes': x[0][0:tl], 'Loss': y[1][0:tl]})
    df1, df2 = pd.concat([save1, original1], ignore_index=True), pd.concat([save2, original2], ignore_index=True)

    fig = sns.lineplot(data=df1, x="Training episodes", y="Loss", label='Vector state')
    sns.lineplot(data=df2, x="Training episodes", y="Loss", label='Graph state')
    fig.legend(loc='upper right', fontsize=12)
    return df1, df2


def draw_reward(vals, weight):
    plt.figure()
    plt.style.use('classic')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.legend(loc='lower right')
    rc = {'font.sans-serif': ['Times New Roman']}
    plt.xlim((0, 10000))
    plt.rcParams.update({'font.size': 16})
    x, y = [[i.step * 200 for i in val] for val in vals], []
    y.append([j.value / 44 for j in vals[0]])
    y.append([j.value / 40 for j in vals[1]])
    save = []
    tl = 293
    for yy in y:
        last = yy[0]
        smoothed = []
        for point in yy[0:tl]:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        save.append(DataFrame({'Training episodes': x[0][0:tl], 'Reward': smoothed}))
    save1, save2 = save[0], save[1]
    original1, original2 = DataFrame({'Training episodes': x[0][0:tl], 'Reward': y[0][0:tl]}), \
                           DataFrame({'Training episodes': x[0][0:tl], 'Reward': y[1][0:tl]})
    df1, df2 = pd.concat([save1, original1], ignore_index=True), pd.concat([save2, original2], ignore_index=True)

    sns.set(font_scale=2, rc=rc)
    sns.set_style("whitegrid")
    fig = sns.lineplot(data=df1, x="Training episodes", y="Reward", label='Graph state')
    sns.lineplot(data=df2, x="Training episodes", y="Reward", label='Vector state')
    fig.legend(loc='lower right', fontsize=12)
    return df1, df2


def draw_from_data():
    l_1 = pd.read_csv("D:\\wangqi\\PortSchedulingGCN\\output_result\\loss\\loss1_p.csv")
    l_2 = pd.read_csv("D:\\wangqi\\PortSchedulingGCN\\output_result\\loss\\loss2_p.csv")
    r_1 = pd.read_csv("D:\\wangqi\\PortSchedulingGCN\\output_result\\loss\\reward1_p.csv")
    r_2 = pd.read_csv("D:\\wangqi\\PortSchedulingGCN\\output_result\\loss\\reward2_p.csv")
    plt.figure()
    plt.style.use('classic')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    rc = {'font.sans-serif': ['Times New Roman']}
    plt.xlim((0, 10000))
    fig = sns.lineplot(data=l_1, x="Training episodes", y="Loss", label='Vector state')
    sns.lineplot(data=l_2, x="Training episodes", y="Loss", label='Graph state')
    fig.legend(loc='upper right', fontsize=12)


def draw_plt(vals, val_names):
    color = ['dodgerblue', 'g', 'r', 'gold', 'c', 'm', 'k', 'darkviolet', 'blue']
    color = sns.color_palette("Spectral", 10)  # sns.color_palette("deep", 18)
    plt.figure()
    plt.style.use('classic')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    for i in range(len(vals)):
        val = vals[i]
        val_name = val_names[i]
        plt.plot([i.step for i in val], [j.value for j in val], linewidth=0.4,
                 label=val_name)
        # plt.fill_between([i.step for i in val], y1, y2,  # 上限，下限
        #                  facecolor='green',  # 填充颜色
        #                  edgecolor='red',  # 边界颜色
        #                  alpha=0.3)
    """横坐标是step，迭代次数 纵坐标是变量值"""
    fs = 13
    plt.ylim(0, 2)
    plt.xlim(0, 10000)
    plt.xlabel('Step', fontsize=fs)
    plt.ylabel('Loss', fontsize=fs)
    plt.legend(val_names, fontsize=11)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # plt.grid(True)
    # plt.grid(color='w', linestyle='--', linewidth=0.3)
    # plt.axes().set_facecolor("grey")
    plt.show()


if __name__ == "__main__":
    # 作多图
    # tensorboard_path_A = 'D:\wangqi\PortSchedulingGCN\runss\A2_02_06_14_14\events.out.tfevents.1675664084.DESKTOP-LUMKFTK'#D:\\wangqi\\PortSchedulingGCN\\runss\\A2\\01_17_19_33\\events.out.tfevents.1673955197.DESKTOP-LUMKFTK'
    # tensorboard_path_B = 'D:\wangqi\PortSchedulingGCN\runss\B2_02_07_07_27\events.out.tfevents.1675726073.DESKTOP-LUMKFTK'#SD:\\wangqi\\PortSchedulingGCN\\runss\\B2\\01_17_11_29\\events.out.tfevents.1673926172.DESKTOP-LUMKFTK'
    # tensorboard_path_C = 'D:\\wangqi\\PortSchedulingGCN\\runss\\C2\\01_16_13_15\\events.out.tfevents.1673846126.DESKTOP-LUMKFTK'
    # tensorboard_path_D = 'D:\\wangqi\\PortSchedulingGCN\\runss\\D2\\01_16_08_57\\events.out.tfevents.1673830664.DESKTOP-LUMKFTK'
    # tensorboard_path_E = 'D:\\wangqi\\PortSchedulingGCN\\runss\\E2\\01_15_22_44\\events.out.tfevents.1673793875.DESKTOP-LUMKFTK'
    # tensorboard_path_F = 'D:\\wangqi\\PortSchedulingGCN\\runss\\F2\\01_15_17_45\\events.out.tfevents.1673775907.DESKTOP-LUMKFTK'
    # tensorboard_path_G = 'D:\\wangqi\\PortSchedulingGCN\\runss\\G2\\01_15_13_27\\events.out.tfevents.1673760440.DESKTOP-LUMKFTK'
    # tensorboard_path_H = 'D:\\wangqi\\PortSchedulingGCN\\runss\\H2\\01_14_07_51\\events.out.tfevents.1673653888.DESKTOP-LUMKFTK'
    # tensorboard_path_Z = 'D:\\wangqi\\PortSchedulingGCN\\runss\\Z2_02_14_09_32\\events.out.tfevents.1676338360.DESKTOP-LUMKFTK'
    # profiles = [chr(i + 65) for i in range(8)]
    # profiles.append('Z')
    # profiles = ['Z']
    # vars = []
    # for profile in profiles:
    #     vars.append(read_tensorboard_data(locals()[f'tensorboard_path_{profile}'], 'l_train/loss'))
    # profiles = ['Profile ' + chr(i + 65) for i in range(8)]
    # profiles.append('Profile ' + 'I')
    # # profiles = ['Profile ' + 'Z']
    # draw_plt(vars, profiles)

    # 读取数据并画图
    path_A = 'D:\\wangqi\\PortSchedulingGCN\\runss\\A2N_04_28_15_32\\events.out.tfevents.1682667137.DESKTOP-LUMKFTK'
    path_B = 'D:\\wangqi\\PortSchedulingGCN\\runss\\A2N_04_28_15_31\\events.out.tfevents.1682667117.DESKTOP-LUMKFTK'
    # path_A = 'D:\\wangqi\\PortSchedulingGCN\\runss\\A2_02_06_14_14\\events.out.tfevents.1675664084.DESKTOP-LUMKFTK'
    # path_B = 'D:\\wangqi\\PortSchedulingGCN\\runss\\A2N_04_20_10_00\\events.out.tfevents.1681956049.DESKTOP-LUMKFTK'
    # path_A = 'D:\\wangqi\\PortSchedulingGCN\\runss\\Z2N_04_26_17_28\\events.out.tfevents.1682501322.DESKTOP-LUMKFTK'
    vars_loss = [read_tensorboard_data(path_A, 'l_train/loss'),
                 read_tensorboard_data(path_B, 'l_train/loss')]
    x, y = draw_loss(vars_loss, 0.8)
    vars_reward = [read_tensorboard_data(path_A, 'l_train_r/reward51'),
                   read_tensorboard_data(path_B, 'l_train_r/reward51')]
    xx, yy = draw_reward(vars_reward, 0.8)

    # 已有数据画图
    draw_from_data()
