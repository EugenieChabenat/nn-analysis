import numpy as np
import matplotlib.pyplot as plt
import torch
from nn_analysis import metrics as me
from nn_analysis import utils
from nn_analysis import plot as pt

def load_data(metric, model_name, epoch, layers):
    layer_names = utils.get_layer_names(model_name, layers)
    if isinstance(layer_names, list):
        return [me.utils.load_data(model_name, epoch, layer_name, metric[0], metric[1]) for layer_name in layer_names]
    else:
        return me.utils.load_data(model_name, epoch, layer_names, metric[0], metric[1])
      
"""epoch = 49
layers = np.arange(17)
metric = ["curve", 1]
metric_types = ["x_cam_trans", "y_cam_trans", "x_cam_rot", "y_cam_rot"]
# metric_types = ["x_cam_rot", "x_focus_pan", "x_cam_pan"]
model_names = [
    "moco_control",
    "moco_CD",
    "moco_CF",
    "moco_CDF",
    "barlow_control",
    "barlow_CD",
    "barlow_CF",
    "barlow_CDF",
    "barlow_P",
    "barlow_PF",
]

fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
        axes[0,i].plot(layers, scores, label=model_name)
    scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
    axes[0,i].plot(layers, scores, label='identity')
    axes[0,i].set_title(metric_type)
    axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('curvature')
fig.tight_layout()
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/plot1.png')


epoch = 82
layers = slice(None)
metric = ["curve",1]
metric_types = ["x_cam_trans", "y_cam_trans", "x_cam_rot", "y_cam_rot"]
# metric_types = ["x_focus_pan", "x_cam_pan"]
model_names = [
    "barlow_P",
    "barlow_P_projector",
]

fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        data = load_data(metric, model_name, epoch, layers)
        scores = [datum[metric_type] for datum in data]
        axes[0,i].plot(np.arange(len(scores)), scores, label=model_name)
    # scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
    # axes[0,i].plot(layers, scores, label='identity')
    axes[0,i].set_title(metric_type)
    axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('curvature')
fig.tight_layout()
plt.show()

plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/plot2.png')

epoch = 54
layers = slice(None)
metric = ["curve",1]
metric_types = ["x_cam_trans", "y_cam_trans", "x_cam_rot", "y_cam_rot"]
# metric_types = ["x_focus_pan", "x_cam_pan"]
model_names = [
    "barlow_control",
    "barlow_control_projector",
]

fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        data = load_data(metric, model_name, epoch, layers)
        scores = [datum[metric_type] for datum in data]
        axes[0,i].plot(np.arange(len(scores)), scores, label=model_name)
    # scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
    # axes[0,i].plot(layers, scores, label='identity')
    axes[0,i].set_title(metric_type)
    axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('curvature')
fig.tight_layout()
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/plot3.png')"""

epoch = 29
layer = 0
metric = ["trajectory", 0]
metric_types = ["x_pan", "y_pan", "x_focus_pan_0", "y_focus_pan"]
model_names = [
    #"identity",
    "barlow_v1_inj"
    #"barlow_control",
    #"barlow_v2"
]

fig, axes = pt.core.subplots(len(model_names), len(metric_types), size=(5,4))
for i, model_name in enumerate(model_names):
    for j, metric_type in enumerate(metric_types):
        if model_name == 'identity':
            scores = load_data(metric, model_name, None, 0)[metric_type]
            evr = load_data(metric, model_name, None, 0)[f'{metric_type}-evr']
        else:
            scores = load_data(metric, model_name, epoch, layer)[metric_type]
            evr = load_data(metric, model_name, epoch, layer)[f'{metric_type}-evr']
        axes[i,j].plot(scores[:,0], scores[:,1])
        axes[i,j].scatter(scores[:,0], scores[:,1])
        axes[i,j].set_title(f'{model_name} - {metric_type}: {evr[:2]}')
fig.supxlabel('PC 1')
fig.supylabel('PC 2')
fig.tight_layout()

plt.savefig(f'/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/figures/{epoch}_PCs_layer_{layer}.pdf')
plt.show()
