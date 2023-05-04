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
# --
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


# --- 
epoch = 29
layers = np.arange(2)
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
one_layer = 0
metric = ["decode", 0]

metric_types = ['obj_class']
                
#metric_types = ['cam_pos_x', 'cam_pos_y', 'cam_scale', 'cam_pos']
                
#metric_types = ['brightness', 'contrast', 'saturation', 'hue', 'color', 'lighting']
                
#metric_types = ['obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pos'] 

#metric_types = ['obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'obj_pose']
    
model_names = [
    "barlow_v1_inj",
    #"identity", 
    "barlow_v2_inj",
    "barlow_v1_inj_b", 
    "barlow_control"
]

model_names = [
    "barlow_faces_texture",
    #"identity", 
    "barlow_faces_notexture", 
    "barlow_control"
]

"""fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        print('model: ', model_name)
        print('layer: ', layers)
                
        scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
        if model_name == "barlow_v1_inj_b": 
            axes[0,i].plot(layers, scores, label="barlow_v3_inj")
        else: 
            axes[0,i].plot(layers, scores, label=model_name)
    #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
    #axes[0,i].plot(layers, scores, label='identity')
    axes[0,i].set_title(metric_type)
    axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('decode')
fig.tight_layout()
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/FACES_inj_vs_control_obj_class.png')"""



# ---
from plot.plots import visualize  as vi

one_layer = 16
metric = ["decode", 0]

metric_types = ['obj_class', 'cam_pos_x', 'cam_pos_y', 'cam_scale', 'cam_pos', 'brightness', 'contrast', 'saturation', 'hue', 'color', 'lighting', 
                'obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pos', 'obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'obj_pose']
    
model_names = [
    "barlow_v1_inj",
    #"identity", 
    "barlow_v2_inj",
    "barlow_v1_inj_b", 
    "barlow_control"
]
fig, axes = subplots(1, 1, height_per_plot=7.5, width_per_plot=7.5, polar=True)
ax = axes[0,0]

x = metric_types
for i, model_name in enumerate(model_names):
    #y = np.array([results[model_name][metric][-1,0] for metric in metrics])
    y = load_data(metric, model_name, epoch, one_layer)[metric_type]
    r_plot(ax, x, y, label=model_names[i])
r_xticks(ax, x, x_offset=0.3, y_offset=0.3, size=11, color="grey")
r_yticks(ax, min=0.0, max=1.0, steps=4)
r_legend(ax, loc=(1.0, 1.0))
fig.tight_layout()
savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/rond.png')
fig.show()
# -- 



"""epoch = 29
layers = np.arange(2)
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]
metric = ["decode", 0]
metric_types = ['obj_class', 'cam_pos_x', 'cam_pos_y', 'cam_scale', #'brightness', 'contrast', 'saturation', 'hue', 
                'obj_pos_x', 'obj_pos_y', 'obj_scale']#, 
                #'obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'cam_pos', 'obj_pos', 'color', 'lighting', 'obj_pose']# metric_types = ["x_cam_rot", "x_focus_pan", "x_cam_pan"]
model_names = [
    #"barlow_v1_inj",
    #"identity", 
    "barlow_v3_equi", 
    "barlow_control"
]

fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        print('model: ', model_name)
        print('layer: ', layers)
        
        #print('keys: ', load_data(metric, model_name, epoch, one_layer).keys())
        
        scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
        axes[0,i].plot(layers, scores, label=model_name)
    #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
    #axes[0,i].plot(layers, scores, label='identity')
    axes[0,i].set_title(metric_type)
    axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('neural fits')
fig.tight_layout()
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/plot2.png')"""

