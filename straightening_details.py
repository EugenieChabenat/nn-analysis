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

    
    
metric_dict = {
            'x_pan': 'Pan - x',
            'y_pan': 'Pan - y ', 
            'z_pan': 'Pan - z', 
            'x_focus_pan': 'Focus Pan - x', 
            'y_focus_pan': 'Focus Pan - y', 
            'z_focus_pan': 'Focus Pan - z', 
            'x_cam_pan': 'Camera Pan - x', 
            'yz_cam_pan': 'Camera Pan - yz', 
            'x_cam_rot': 'Camera Rotation - x', 
            'y_cam_rot': 'Camera Rotation - y', 
            'x_cam_trans': 'Camera Translation - x', 
            'y_cam_trans': 'Camera Translation - y', 
            'z_cam_trans': 'Camera Translation - z', 
            'x_obj_rot': 'Object Rotation - x', 
            'y_obj_rot': 'Object Rotation - y'
             }


epoch = 29
#layers = np.arange(12)
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
metric = ["curve", 1]


list_metrics = {
    "Camera Translation & Rotation" : ["x_cam_trans", "y_cam_trans", "z_cam_trans", "x_cam_rot", "y_cam_rot"], 
    "x, y & z Pan" : ['x_pan-detailed', 'x_pan', 'y_pan-detailed', 'y_pan', 'z_pan-detailed', 'z_pan'], 
    "x-focus Pan" : ['x_focus_pan-detailed', 'x_focus_pan'], 
    "y-focus Pan" : ['y_focus_pan-detailed', 'y_focus_pan'], 
    "z-focus Pan" : ['z_focus_pan-detailed', 'z_focus_pan'], 
    "Camera Pan" : ['x_cam_pan-detailed', 'x_cam_pan', 'yz_cam_pan-detailed','yz_cam_pan'], 
    "Object Rotation" : ['x_obj_rot', 'y_obj_rot'], 
}

list_labels = {
    "barlow_twins_50epochs_pc1": "max PCs",
    "barlow_twins_50epochs_pc2": "300 PCs", 
    "barlow_twins_50epochs_pc3": "300 RCs",
    "barlow_twins_50epochs_pc4": "500 RCs", 
    "barlow_twins_50epochs_pc5": "1000 RCs", 
    "barlow_twins_50epochs_pc6": "2048 RCs"
}

dict_color = {
    "barlow_twins_50epochs_pc1" : "red",
    "barlow_twins_50epochs_pc2": "green", 
    "barlow_twins_50epochs_pc3": "violet",
    "barlow_twins_50epochs_pc4": "purple", 
    "barlow_twins_50epochs_pc5": "indigo", 
    "barlow_twins_50epochs_pc6": "plum"
}


model_names = [
    "barlow_twins_50epochs_pc1",
    "barlow_twins_50epochs_pc2", 
    "barlow_twins_50epochs_pc3",
    "barlow_twins_50epochs_pc4", 
    "barlow_twins_50epochs_pc5", 
    "barlow_twins_50epochs_pc6", 
    #"barlow_twins_50epochs_pc7"
]







# ------------------------------------------------------------------------------------
# LAYERS PLOT 
# ------------------------------------------------------------------------------------
for key, metric_types in list_metrics.items(): 

    fig, axes = pt.core.subplots(1, len(metric_types), size=(10,8), sharex=True)
    for i, metric_type in enumerate(metric_types):
        for model_name in model_names:
            print('model: ', model_name)
            print('layer: ', layers)
            scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            if model_name[-1] == "1" or model_name[-1] == "2": 
                axes[0,i].plot(layers, scores, label=list_labels[model_name], color = dict_color[model_name], ls = '--')
            else: 
                axes[0,i].plot(layers, scores, label=list_labels[model_name], color = dict_color[model_name])
        scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        axes[0,i].plot(layers, scores, label='identity', color = 'black')
        
        axes[0,i].axvline(x = 3, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 6, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 10, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 16, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 19, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 20, color = 'grey', alpha = 0.5, ls = 'dotted')
        
        axes[0,i].set_title(metric_type)
        axes[0,i].set_xticks([0, 3, 6, 10, 16, 19, 20])
        axes[0,i].set_xticklabels(['', '', 'inj v1', 'inj v2', 'inj v4', 'inj IT', 20], rotation=45, ha='right')
        
        axes[0,i].text(4.5, 0.2, "Block V1", ha="center", va="center", size=12)
        axes[0,i].text(8, 0.2, "Block V2", ha="center", va="center", size=12)
        axes[0,i].text(13, 0.2, "Block V4", ha="center", va="center", size=12)
        axes[0,i].text(17.5, 0.2, "Block IT", ha="center", va="center", size=12)
        axes[0,i].set_ylim(0.0, 1.)
        axes[0,i].legend(loc='lower left')
    fig.supxlabel('layers')
    fig.supylabel('curvature')
    fig.tight_layout()
    plt.show()
    plt.savefig('/home/ec3731/issa_analysis/nn-analysis/pcs_injection_bt50_{}.png'.format(key))
    plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/PCs_vs_RCs_injection_bt50_{}.png'.format(key))
    #plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/PCs_vs_RCs/injection_v1.png')




# ------------------------------------------------------------------------------------
# ROUND PLOT 
# ------------------------------------------------------------------------------------
model_names = [
    "injection_v2_pc1",
    "injection_v2_pc2", 
    "injection_v2_pc3",
    "injection_v2_pc4", 
    "injection_v2_pc5"
]
model_names = [
    "resnet50_untrained_pc1",
    "resnet50_untrained_pc2", 
    "resnet50_untrained_pc3",
    "resnet50_untrained_pc5", 
    "resnet50_untrained_pc6"
]

model_names = [
    "barlow_twins_50epochs_pc1",
    "barlow_twins_50epochs_pc2", 
    "barlow_twins_50epochs_pc3",
    "barlow_twins_50epochs_pc4", 
    "barlow_twins_50epochs_pc5"
]

model_names = [
    "injection_v4_pc1",
    "injection_v4_pc2", 
    "injection_v4_pc3",
    "injection_v4_pc4", 
    "injection_v4_pc5"
]
# ------------------------------------------------------------------------------------
# HISTOGRAM PLOT 
# ------------------------------------------------------------------------------------


