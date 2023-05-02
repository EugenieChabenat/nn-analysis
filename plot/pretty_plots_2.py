import os
import math

import numpy as np
import matplotlib.pyplot as plt

import plots
import config
import constants
from plots import visualize  as vi

def savefig(fig, filename):
    folder = os.path.split(filename)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(filename)

def subplots(n_rows, n_cols, height_per_plot=4, width_per_plot=5, polar=False, **kwargs):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*width_per_plot, n_rows*height_per_plot), subplot_kw={'polar': polar}, facecolor='white', **kwargs)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape((1,-1))
    elif n_cols == 1:
        axes = axes.reshape((-1,1))
    else:
        pass
    return fig, axes

def get_colors(N):
    if N <= 10:
        return [f'C{i}' for i in range(N)]
    return plt.cm.jet(np.linspace(0,1,N))

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
    
def grid_plot(scores, params_i, params_j, params_k, names=['model','metric','epoch'], score_name='score', permute=[0,1,2], grid=False, cols=4):
    # 3D plot
    params = [params_i, params_j, params_k]
    params_i, params_j, params_k = [params[permute[n]] for n in range(3)]
    names = [names[permute[n]] for n in range(3)]
    scores = np.transpose(scores, permute)
    
    colors = get_colors(len(params_i))
    rows = int(math.ceil(len(params_j)/cols))
    fig, axes = subplots(rows, cols, height_per_plot=4, width_per_plot=5)
    
    for ax in axes.reshape(-1):
        ax.set_axis_off()
    
    for j, param_j in enumerate(params_j):
        m, n = np.unravel_index(j, (rows, cols))
        axes[m,n].set_axis_on()
        for i, param_i in enumerate(params_i):
            axes[m,n].plot(params_k, scores[i,j,:], color=colors[i], label=param_i)
        axes[m,n].set_xlabel(names[2])
        axes[m,n].set_ylabel(score_name)
        axes[m,n].set_title(f"{param_j}")
        if grid:
            axes[m,n].grid(b=True)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.tight_layout()
    return fig

def plot_results(results, mode, n_per_row=4, grid=False, metric_labels=None):
#     print([results[model_name][metric].shape for model_name in results.keys() for metric in results[model_name].keys()])
    assert all_equal([results[model_name][metric].shape for model_name in results.keys() for metric in results[model_name].keys()])
    
    model_names = list(results.keys())
    metrics = list(results[model_names[0]].keys())
    if metric_labels is None:
        metric_labels = metrics
    data = np.stack([np.stack([results[model_name][metric] for metric in metrics],axis=0) for model_name in model_names],axis=0)
    epochs, layers = results[model_names[0]][metrics[0]].shape
    epochs, layers = np.arange(epochs), np.arange(layers)
    
    if mode == 'layer-wise':
        assert len(epochs) == 1
        return grid_plot(data.squeeze(), model_names, metric_labels, layers, names=['model','metric','layers'], permute=[0,1,2], grid=grid, cols=n_per_row)
    elif mode == 'epoch-wise':
        assert len(layers) == 1
        return grid_plot(data.squeeze(), model_names, metric_labels, epochs, names=['model','metric','epoch'], permute=[0,1,2], grid=grid, cols=n_per_row)
    
def grouped_bar(ax, xs, ys, width=0.2, sep=0.3):
    assert len(xs) == len(ys)
    total = 0.0
    all_xticks = []
    all_xlabels = []
    for i, y in enumerate(ys):
        xticks = np.linspace(0.0,len(y)*width,num=len(y))+total
        ax.bar(xticks, y, width=width)
        total += (len(y)+1.5)*width + sep
        all_xticks += list(xticks)
        all_xlabels += xs[i]
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xlabels, rotation=45, ha='right')
    
def grouped_bar_2(ax, x, ys, labels, group_width=1.0, ys_err=None):
    assert len(x) == len(ys[0])
    assert all_equal([len(y) for y in ys])
    width=group_width/(len(ys)+1)
    for i, y in enumerate(ys):
        ax.bar(np.arange(len(y))+i*width, y, yerr=ys_err[i], width=width, label=labels[i])
    ax.set_xticks(np.arange(len(ys[0]))+(len(ys)/2-0.5)*width)
    ax.set_xticklabels(x, rotation=45, ha='right')
    ax.legend()
