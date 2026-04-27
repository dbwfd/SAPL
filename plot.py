import matplotlib.pyplot as plt

def plot_blue_line(x, y, title='', xlabel='X', ylabel='Y', save_path=None):
    """绘制蓝色折线统计图。

    Args:
        x (list或np.array): 横坐标数据。
        y (list或np.array): 纵坐标数据。
        title (str): 图表标题。
        xlabel (str): x 轴标签。
        ylabel (str): y 轴标签。
        save_path (str或None): 若提供路径，则保存图片，否则展示。
    """
    plt.figure(figsize=(6, 5))
    plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(65, 73)
    plt.grid(True, linestyle='--', alpha=0.4)
    





    plt.legend(['IoU'], loc='upper right')



    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
        print(f'已保存折线图: {save_path}')
    else:
        plt.show()


if __name__ == '__main__':
    # 示例数据
    x = ["4", "6", "8", "12", "16", "24"]
    #y = [97.01, 96.74, 96.87, 96.92, 96.78, 96.65]
    y = [69.54, 69.85, 70.29, 69.37, 68.99, 69.54]
    plot_blue_line(x, y, xlabel='$M$', ylabel='IoU',save_path='i1k.png')
