import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
from scipy import io as sio

def Draw_comparison_figure(strain, predictions, labels, save_fig):
    '''
    全局设置字体及大小，设置公式字体即可，若要修改刻度字体，可在此修改全局字体
    '''
    config = {
        "mathtext.fontset":'stix',
        "font.family":'Times New Roman',
        # "font.serif": ['SimSun'],
        # "font.size": 15,
    }
    rcParams.update(config)

    if not os.path.exists(save_fig):
        os.makedirs(save_fig)
    # 
    num_sample = np.arange(strain.shape[0])
    # 
    for i in num_sample:
        plt.rcParams['figure.figsize'] = (6.4, 4.8)
        # 作图数据
        x = np.diagonal(strain[i, :, :])
        x = np.insert(x, 0, 0)
        y1 = np.insert(predictions[i, :], 0, 1)
        y2 = np.insert(labels[i, :], 0, 1)
        # 作图代码
        plt.rcParams['figure.figsize'] = (6.4, 4.8)
        fig,ax = plt.subplots()
        plt.plot(x, y1, 'r-o', lw=1.5, markersize=8, label='predictions')
        plt.plot(x, y2, 'b--s', lw=1.5, markersize=8, label='labels')
        # 设置图片边框
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        # 设置图例
        plt.legend(loc='best', frameon=True, fontsize='12')
        # 设置图片标题
        plt.title('Comparison of predictions with labels for sample-{0}'.format(i+1), fontsize='12')
        # 设置坐标轴标题
        plt.xlabel(r'$\epsilon_\mathrm{a}$ (%)', fontsize='14')
        plt.ylabel(r'$\mathit{f}/\mathit{f}_0$', fontsize='14')
        # 设置坐标刻度
        plt.xlim(0, 10)
        plt.ylim(bottom=1)
        x_major_locator = MultipleLocator(1)            # 把x轴的主刻度设置为1的倍数
        ax.xaxis.set_major_locator(x_major_locator)
        ax.tick_params(direction='in', width='1.5')
        # 设置栅格
        plt.grid(True, linestyle='--', color='black')
        # 设置等轴
        ax.set_aspect(1/ax.get_data_ratio())
        # 保存图片
        plt.savefig('%s/test_%d.jpg'%(save_fig, i+1), bbox_inches='tight', dpi=100)

        plt.close()

def Draw_loss_figure(epochs, training_loss, validation_loss, folder_name):
    '''
    全局设置字体及大小，设置公式字体即可，若要修改刻度字体，可在此修改全局字体
    '''
    config = {
        "mathtext.fontset":'stix',
        "font.family":'Times New Roman',
        # "font.serif": ['SimSun'],
        # "font.size": 15,
    }
    rcParams.update(config)

    x = np.arange(epochs)

    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    fig,ax = plt.subplots()
    plt.plot(x, training_loss, color='r', ls='-', lw='1.5', marker='o', markerfacecolor='none', markeredgewidth='1.5', label='Training loss')
    plt.plot(x, validation_loss, color='b', ls='--', lw='1.5', marker='s', markerfacecolor='none', markeredgewidth='1.5', label='Validation loss')
    # 设置图片边框
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    # 设置图例
    plt.legend(loc='best', frameon=True, fontsize='12')
    # 设置坐标轴标题
    plt.xlabel('Epoch', fontsize='12')
    plt.ylabel('Loss', fontsize='12')
    # 设置坐标刻度
    plt.xlim(0, epochs)
    # plt.ylim(bottom=0)
    x_major_locator = MultipleLocator(epochs / 10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.tick_params(direction='in', width='1.5')
    # 设置等轴
    ax.set_aspect(1/ax.get_data_ratio())
    # 保存图片
    plt.savefig('%s/loss_evolution.jpg'%(folder_name), bbox_inches='tight', dpi=300)

    plt.close()