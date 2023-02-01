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


def draw_plt(vals, val_names):
    color = ['dodgerblue', 'g', 'r', 'gold', 'c', 'm', 'k', 'darkviolet', 'blue']
    color = sns.color_palette("Spectral", 10)  # sns.color_palette("deep", 18)
    plt.figure()
    plt.style.use('classic')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    for i in range(len(vals)):
        val = vals[i]
        val_name = val_names[i]
        plt.plot([i.step for i in val[0:5000]], [j.value for j in val[0:5000]], linewidth=0.4,
                 label=val_name)
    """横坐标是step，迭代次数 纵坐标是变量值"""
    fs = 13
    plt.ylim(0, 2)
    plt.xlim(0, 8000)
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
    tensorboard_path_A = 'D:\\wangqi\\PortSchedulingGCN\\runss\\A2\\01_17_19_33\\events.out.tfevents.1673955197.DESKTOP-LUMKFTK'
    tensorboard_path_B = 'D:\\wangqi\\PortSchedulingGCN\\runss\\B2\\01_17_11_29\\events.out.tfevents.1673926172.DESKTOP-LUMKFTK'
    tensorboard_path_C = 'D:\\wangqi\\PortSchedulingGCN\\runss\\C2\\01_16_13_15\\events.out.tfevents.1673846126.DESKTOP-LUMKFTK'
    tensorboard_path_D = 'D:\\wangqi\\PortSchedulingGCN\\runss\\D2\\01_16_08_57\\events.out.tfevents.1673830664.DESKTOP-LUMKFTK'
    tensorboard_path_E = 'D:\\wangqi\\PortSchedulingGCN\\runss\\E2\\01_15_22_44\\events.out.tfevents.1673793875.DESKTOP-LUMKFTK'
    tensorboard_path_F = 'D:\\wangqi\\PortSchedulingGCN\\runss\\F2\\01_15_17_45\\events.out.tfevents.1673775907.DESKTOP-LUMKFTK'
    tensorboard_path_G = 'D:\\wangqi\\PortSchedulingGCN\\runss\\G2\\01_15_13_27\\events.out.tfevents.1673760440.DESKTOP-LUMKFTK'
    tensorboard_path_H = 'D:\\wangqi\\PortSchedulingGCN\\runss\\H2\\01_14_07_51\\events.out.tfevents.1673653888.DESKTOP-LUMKFTK'
    tensorboard_path_Z = 'D:\\wangqi\\PortSchedulingGCN\\runss\\Z2\\01_12_19_40\\events.out.tfevents.1673523629.DESKTOP-LUMKFTK'
    profiles = [chr(i + 65) for i in range(8)]
    profiles.append('Z')
    # profiles = ['A']
    vars = []
    for profile in profiles:
        vars.append(read_tensorboard_data(locals()[f'tensorboard_path_{profile}'], 'l_train/loss'))
    profiles = ['Profile '+chr(i + 65) for i in range(8)]
    profiles.append('Profile '+'Z')
    draw_plt(vars, profiles)
