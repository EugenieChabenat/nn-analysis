import numpy as np
import matplotlib.pyplot as plt
import torch
from nn_analysis import metrics as me
from nn_analysis import utils
from nn_analysis import plot as pt

import os

def load_data(metric, model_name, epoch, layers):
    layer_names = utils.get_layer_names(model_name, layers)
    if isinstance(layer_names, list):
        return [me.utils.load_data(model_name, epoch, layer_name, metric[0], metric[1]) for layer_name in layer_names]
    else:
        return me.utils.load_data(model_name, epoch, layer_names, metric[0], metric[1])

metric_dict = {'obj_class': 'Object Class', 
               'cam_pos_x': 'Camera Position - x', 
               'cam_pos_y': 'Camera Position - y', 
               'cam_scale': 'Camera Scale', 
               'cam_pos': 'Camera Position', 
               'brightness': 'Brightness', 
               'contrast': 'Contrast', 
               'saturation': 'Saturation',
               'hue': 'Hue', 
               'color': 'Color', 
               'lighting': 'Lighting', 
               'obj_pos_x': 'Object Position - x', 
               'obj_pos_y': 'Object Position - y', 
               'obj_scale': 'Object Scale', 
               'obj_pos': 'Object Position', 
               'obj_pose_x': 'Object Pose - x', 
               'obj_pose_y': 'Object Pose - y', 
               'obj_pose_z': 'Object Pose - z', 
               'obj_pose': 'Object Pose'
               }


# --- 
epoch = 29
layers = np.arange(2)
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
one_layer = 0
metric = ["decode", 0]

metric_types = ['obj_class']
                
metric_types = ['cam_pos_x', 'cam_pos_y', 'cam_scale', 'cam_pos']
                
#metric_types = ['brightness', 'contrast', 'saturation', 'hue', 'color', 'lighting']
                
#metric_types = ['obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pos'] 

#metric_types = ['obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'obj_pose']

list_metrics = {
    "Object Class" : ['obj_class'], 
    "Camera Pos & Scale" : ['cam_pos_x', 'cam_pos_y', 'cam_scale', 'cam_pos'], 
    "Color & Lightning": ['brightness', 'contrast', 'saturation', 'hue', 'color', 'lighting'],
    "Object Pos & Scale" : ['obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pos'], 
    "Object Pose": ['obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'obj_pose']
}
model_names = [
    "injection_v1",
    "injection_v4",
    "resnet50_untrained", 
    "barlow_twins_50epochs"
]

"""model_names = [
    "barlow_faces_texture",
    #"identity", 
    "barlow_faces_notexture", 
    "barlow_faces_control", 
    "faces_pretrained_notexture"
    #"barlow_control"
]"""


for key, metric_types in list_metrics.items(): 
    
    fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
    for i, metric_type in enumerate(metric_types):
        for model_name in model_names:
            scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            if model_name == "barlow_v1_inj_b": 
                axes[0,i].plot(layers, scores, label="barlow_v3_inj")
            else: 
                axes[0,i].plot(layers, scores, label=model_name)
        #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        #axes[0,i].plot(layers, scores, label='identity')
        axes[0,i].set_title(key)
        axes[0,i].legend()
    fig.supxlabel('layers')
    fig.supylabel('decode')
    fig.tight_layout()
    plt.show()
    plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/{}.png'.format(key))



# ------------------------------------------------------------------------------------
# ROUND PLOT 
# ------------------------------------------------------------------------------------

"""one_layer = 3
metric = ["decode", 0]

metric_types = ['obj_class', 'cam_pos_x', 'cam_pos_y', 'cam_scale', 'cam_pos', 'brightness', 'contrast', 'saturation', 'hue', 'color', 'lighting', 
                'obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pos', 'obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'obj_pose']
model_names = [
    "barlow_faces_texture",
    #"identity", 
    "barlow_faces_notexture", 
    "barlow_control"
]   
model_names = [
    "barlow_v1_inj",
    #"identity", 
    "barlow_v2_inj",
    "barlow_v1_inj_b", 
    "barlow_control"
]

fig, axes = pt.round_plot.subplots(1, 1, height_per_plot=7.5, width_per_plot=7.5, polar=True)
ax = axes[0,0]

x = metric_types
for i, model_name in enumerate(model_names):
    #y = np.array([results[model_name][metric][-1,0] for metric in metrics])
    y = np.array([load_data(metric, model_name, epoch, one_layer)[metric_type] for metric_type in metric_types])
    if model_name == "barlow_v1_inj_b": 
        pt.round_plot.r_plot(ax, x, y, label="barlow_v3_inj")
    else: 
        pt.round_plot.r_plot(ax, x, y, label=model_names[i])
pt.round_plot.r_xticks(ax, x, x_offset=0.3, y_offset=0.3, size=11, color="grey")
pt.round_plot.r_yticks(ax, min=0.0, max=1.0, steps=4)
pt.round_plot.r_legend(ax, loc=(1.0, 1.0))
fig.tight_layout()
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/FINAL_rond_layer6.png')
fig.show()"""

# ------------------------------------------------------------------------------------
# HISTOGRAM PLOT 
# ------------------------------------------------------------------------------------
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
    
    
metricss = [
            ['obj_class', 'cam_pos_x', 'cam_pos_y', 'cam_scale', 'cam_pos'], 
            ['brightness', 'contrast', 'saturation', 'hue', 'color', 'lighting'], 
            ['obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pos', 'obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'obj_pose']]

baseline_model = "barlow_control"
model_names = [
    "barlow_v1_inj",
    "barlow_v2_inj",
    "barlow_v1_inj_b", 
]

one_layer = 13

"""fig, axes = pt.round_plot.subplots(1,3,height_per_plot=6,width_per_plot=6)
for i, model_name in enumerate(model_names):
    #ys = [[results[model_name][metric][-1,0]-results[baseline_model_name][metric][-1,0] for metric in metrics] for metrics in metricss]
    ys = [[load_data(metric, model_name, epoch, one_layer)[metric_type] - load_data(metric, baseline_model, epoch, one_layer)[metric_type] for metric_type in metric_types] for metric_types in metricss]
    #xs = metricss
    xs = [[metric_dict[metric_type] for metric_type in metrics] for metrics in metricss]
    grouped_bar(axes[0,i], xs, ys)
    if model_name == "barlow_v1_inj_b": 
        axes[0, i].set_title("barlow_v3_inj")
    else: 
        axes[0,i].set_title(model_name)
    axes[0,i].set_ylabel('Score (relative to baseline)')
#     axes[0,i].set_ylim(-0.25,0.27)
y_lim_min = min([axes[0,i].get_ylim()[0] for i in range(len(model_names))])
y_lim_max = max([axes[0,i].get_ylim()[1] for i in range(len(model_names))])
for i in range(len(model_names)):
    axes[0,i].set_ylim(y_lim_min, y_lim_max)
fig.tight_layout()
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/FINAL_hist_layer13.png')
fig.show()"""



