import numpy as np
import matplotlib.pyplot as plt
import torch

import os

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
  
def r_plot(ax, x, y, color=None, label=None, **kwargs):
    N = len(x)
    if N == 0:
        angles, values = [], []
    else:
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles = angles + angles[:1]
        values = list(y)
        values = values + values[:1]
    ax.set_rlabel_position(2.0)
    line1, = ax.plot(angles, values, linewidth=1, linestyle='solid', color=color, label=label, **kwargs)
#     line2, = ax.fill(angles, values, alpha=0.1, color=color, **kwargs)
    return line1,

def r_xticks(ax, x, x_offset=0.15, y_offset=0.1, size=11, color="grey", **kwargs):
    N = len(x)
    angles = [n / float(N) * 360 for n in range(N)]
    _, labels = ax.set_thetagrids(angles, x, color=color, size=size)
    for i, label in enumerate(labels):
        x, y = label.get_position()
        label.set_position((x,y-x_offset*np.cos(angles[i]*np.pi/180)**2-y_offset*(np.exp(-np.abs(angles[i]*np.pi/180-3*np.pi/2))+np.exp(-np.abs(angles[i]*np.pi/180-np.pi/2)))**5))

        
def r_yticks(ax, min=0.0, max=1.0, steps=4):
    ax.set_rgrids(np.linspace(min, max, steps+1)[1:], color="grey", size=9)
    ax.set_ylim((min, max))
def r_legend(ax, loc=(1.0, 0.6), **kwargs):
    ax.legend(loc=loc, **kwargs)
