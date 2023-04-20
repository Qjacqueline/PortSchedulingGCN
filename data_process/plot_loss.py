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


def draw_compare(vals, val_names, weight):
    plt.figure()
    plt.style.use('classic')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    x = [[i.step for i in val] for val in vals]
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
        save.append(DataFrame({'timestep': x[0][0:tl], 'loss': smoothed}))
    save1, save2 = save[0], save[1]
    original1, original2 = DataFrame({'timestep': x[0][0:tl], 'loss': y[0][0:tl]}), \
                           DataFrame({'timestep': x[0][0:tl], 'loss': y[1][0:tl]})
    df1, df2 = pd.concat([save1, original1], ignore_index=True), pd.concat([save2, original2], ignore_index=True)

    sns.lineplot(data=df1, x="timestep", y="loss", label='Vector state')
    sns.lineplot(data=df2, x="timestep", y="loss", label='Graph state')


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

    path_A = 'D:\\wangqi\\PortSchedulingGCN\\runss\\A2_02_06_14_14\\events.out.tfevents.1675664084.DESKTOP-LUMKFTK'
    path_B = 'D:\\wangqi\\PortSchedulingGCN\\runss\\Z2_1000N_04_17_20_22\\events.out.tfevents.1681734133.DESKTOP-LUMKFTK'

    vars = []
    vars.append(read_tensorboard_data(path_A, 'l_train_r/reward51'))
    vars.append(read_tensorboard_data(path_B, 'l_train_r/reward51'))
    draw_compare(vars, ['Vector state', 'Graph state'], 0.9)
