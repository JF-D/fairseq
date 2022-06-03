import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm


def myplot_animation(data_list, name, ylabel='Iteration'):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    datas = [act.reshape(-1) for act in data_list]
    iterations = np.arange(len(data_list))
    positions = np.arange(datas[0].size)
    positions, iterations = np.meshgrid(positions, iterations)
    acts = np.vstack(datas)
    surf = ax.plot_surface(positions, iterations, acts, rstride=1, cstride=1, cmap=cm.viridis)
    ax.set_xlabel('Tensor')
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(surf, shrink=1, aspect=30)
    def data_gen():
        for angle in range(0, 360, 10):
            yield angle
    def run(angle):
        ax.view_init(30, angle)
    ani = animation.FuncAnimation(fig, run, data_gen, interval=350, repeat=True)
    ani.save(name, writer='pillow')


def curve_line(steps, datas, labels, xticks=None, xlabel='', ylabel='', title='', figname=None):
    fig, ax = plt.subplots(dpi=300)
    for i, data in enumerate(datas):
        ax.plot(steps, data, label=labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if figname is not None:
        fig.savefig(figname)
