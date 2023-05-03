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

fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
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
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/FINAL_inj_vs_control_objclass.png')

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

