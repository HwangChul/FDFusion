import numpy as np
import matplotlib.pyplot as plt

def draw_epoch_metric(input, metric_list, best, best2, savepath=None):
    """
    绘制训练过程中的指标变化曲线。

    参数:
        input (ndarray): 包含每个 epoch 各指标的二维数组，形状为 (epochnum, metricnum)。
        metric_list (list): 指标名称列表，与 input 的列一一对应。
        best (list): 每个指标的最佳值，用于绘制红色水平线。
        best2 (list): 每个指标的次佳值，用于绘制蓝色水平线。
        savepath (str 或 None): 图像保存路径。如果为 None，则显示图像而不保存。
    """
    epochnum, metricnum = input.shape  # 获取 epoch 数和指标数
    x = np.linspace(1, epochnum, epochnum)  # 为 x 轴生成从 1 到 epochnum 的均匀分布点
    plt.figure(figsize=(16, 8))  # 设置图像大小

    if metricnum <= 12:  # 如果指标数量不超过 12，则绘制 3x4 的子图网格
        for row in range(3):  # 遍历每一行
            for column in range(4):  # 遍历每一列
                index = 4 * row + column  # 计算当前指标的索引
                if index >= metricnum:  # 如果超出指标数量，提前退出
                    break
                plt.subplot(3, 4, index + 1)  # 创建 3x4 网格的第 (index+1) 个子图
                plt.plot(x, input[:, index])  # 绘制当前指标的变化曲线
                plt.title(metric_list[index])  # 设置当前子图的标题为对应指标名称
                plt.axhline(best[index], color='red')  # 绘制红色水平线表示最佳值
                plt.axhline(best2[index], color='blue')  # 绘制蓝色水平线表示次佳值

    if savepath is None:  # 如果没有指定保存路径
        plt.show()  # 显示图像
    else:  # 如果指定了保存路径
        plt.savefig(savepath)  # 将图像保存到指定路径
    plt.close('all')  # 关闭所有打开的图像，释放内存
