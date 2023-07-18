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

dict_color = {
    "noprojector_control_v1":  ["black", '-'], 
    "noprojector_control_v2":  ["black", '-'], 
    
    # no projector linear 
    "noprojector_linear_v1":  ["brown", '-'], 
    "noprojector_linear_v2":  ["brown", '-'], 
    "noprojector_linear_v4":  ["brown", '--'], 
    "noprojector_linear_IT":  ["brown", '--'], 
    
    # no projector conv
    "noprojector_conv_v1": ["gold", '-'], 
    "noprojector_conv_v2": ["gold", '-'], 
    "noprojector_conv_v4": ["gold", '-'], 
    "noprojector_conv_IT": ["gold", '-'], 
    
    # random injection 
    "injection_v1" : ["orange", ':'],
    "injection_v2": ["orange", ':'], 
    "injection_v4": ["orange", ':'],
    "injection_IT": ["orange", ':'],
    
    # convolution injection
    "injection_conv_v1": ["lightblue", ':'], 
    "injection_conv_v2": ["lightblue", ':'], 
    "injection_conv_v4": ["lightblue", ':'], 
    "injection_conv_IT": ["lightblue", ':'], 
    
    #"injection_conv_v1": ["red", '-'], 
    #"injection_conv_v2": ["blue", '-'], 
    #"injection_conv_v4": ["orange", '-'], 
    #"injection_conv_IT": ["green", '-'], 
        
    # unfreeze convolution injection 
    "unfreeze_injection_v1": ["green", '-'], 
    "unfreeze_injection_v2": ["green", '-'], 
    "unfreeze_injection_v4": ["green", '-'], 
    "unfreeze_injection_IT": ["green", '-'], 
    
    #"unfreeze_injection_v1": ["red", '-'], 
    #"unfreeze_injection_v2": ["blue", '-'], 
    #"unfreeze_injection_v4": ["orange", '-'], 
    #"unfreeze_injection_IT": ["green", '-'], 

    # subset injection 
    "subset_injection_v1": ["blue", '-'], 
    "subset_injection_v2": ["blue", '-'], 
    "subset_injection_v4": ["blue", '-'], 
    "subset_injection_IT": ["blue", '-'], 

    # separate injection 
    "injection_separate_v1": ["purple", '-'], 
    "injection_separate_v2": ["purple", '-'], 
    "injection_separate_v4": ["purple", '-'], 
    "injection_separate_IT": ["purple", '-'], 
    
    # conv subset injection 
    "injection_conv_subset_v1": ["lime", '-'], 
    "injection_conv_subset_v2": ["lime", '-'], 
    "injection_conv_subset_v4": ["lime", '-'], 
    "injection_conv_subset_IT": ["lime", '-'],

    
    # control models 
    # control models 
    "v1_no_injection": ["red", '--'], 
    "v2_no_injection": ["red", '--'], 
    "v4_no_injection": ["red", '--'], 
    "IT_no_injection": ["red", '--'], 

    "resnet50_untrained": ["pink", '--'], 
    "barlow_twins_50epochs": ["grey", '--'], 
    "barlow_fact_no_injection": ["black", '--']
}
# --- 
epoch = 29
layers = np.arange(2)
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]#, 21, 22, 23, 24, 25, 26, 27]
one_layer = 0
metric = ["decode", 0]

dict_metric_names = {
    'obj_class': "Object Class" , 
    'cam_pos_x': "Camera Position - x", 
    'cam_pos_y': "Camera Position - y", 
    'cam_pos': "Camera Position", 
    'cam_scale': "Camera Scale", 
    'brightness': "Brightness", 
    'lighting': "Lighting",
    'contrast': "Contrast", 
    'saturation': "Saturation",
    'hue': "Hue", 
    'color': "Color",
    'obj_pos_x': "Object Position - x",
    'obj_pos_y': "Object Position - y", 
    'obj_pos': "Object Position", 
    'obj_scale': "Object Scale",
    'obj_pose_x': "Object Pose - x", 
    'obj_pose_y': "Object Pose - y", 
    'obj_pose_z':"Object Pose - z", 
    'obj_pose': "Object Pose"
}

"""list_metrics = {
    "Object Class" : ['obj_class'], 
    "Camera Pos" : ['cam_pos_x', 'cam_pos_y'], 
    "Camera Scale" : ['cam_scale', 'cam_pos'], 
    "Lightning": ['brightness', 'lighting'],
    "Contrast": ['contrast', 'saturation'],
    "Color": ['hue', 'color'],
    "Object Pos" : ['obj_pos_x', 'obj_pos_y'], 
    "Object Scale" : ['obj_scale', 'obj_pos'], 
    "Object Pose": ['obj_pose_x', 'obj_pose_y'], 
    "Object Pose 2": ['obj_pose_z', 'obj_pose']
}"""

list_metrics = {
    #0 : ['obj_scale'], 
    0 : ['obj_class', 'obj_pos', 'obj_scale'], 
    1 : ['obj_pose', 'cam_pos_x', 'cam_pos_y'], 
   2 : ['cam_scale',  'lighting', 'color']
}

dict_model_names = {
    "injection_v1": "Random linear injection at V1",
    "injection_separate_v1": "Trained linear injection at V1" , 
    "injection_conv_v1": "Random convolutional injection at V1" ,
    "unfreeze_injection_v1": "Trained convolutional injection at V1" , 
    "subset_injection_v1": "Random linear injection of spatial information at V1", 
    "injection_conv_subset_v1": "Random convolutional injection of spatial information at V1" ,
    "noprojector_linear_v1": "Random linear injection at V1 - no projector" ,
    "noprojector_conv_v1": "Random convolutional injection of subset at V1 - no projector" ,
    "noprojector_control_v1": "Evaluation at V1, no injection - no projector", 
    

    "injection_v2": "Random linear injection at V2",
    "injection_separate_v2": "Trained linear injection at V2" , 
    "injection_conv_v2": "Random convolutional injection at V2" ,
    "unfreeze_injection_v2": "Trained convolutional injection at V2" , 
    "subset_injection_v2": "Random linear injection of spatial information at V2", 
    "injection_conv_subset_v2": "Random convolutional injection of spatial information at V2" ,
    "noprojector_linear_v2": "Random linear injection at V2 - no projector",
    "noprojector_conv_v2": "Random convolutional injection at V2 - no projector" ,
    "noprojector_control_v2": "Evaluation at V2, no injection - no projector", 

    "injection_v4": "Random linear injection at V4",
    "injection_separate_v4": "Trained linear injection at V4" , 
    "injection_conv_v4": "Random convolutional injection at V4" ,
    "unfreeze_injection_v4": "Trained convolutional injection at V4" , 
    "subset_injection_v4": "Random linear injection of spatial information at V4", 
    "injection_conv_subset_v4": "Random convolutional injection of spatial information at V4",
    "noprojector_linear_v4": "Random linear injection at V4 - no projector",
    "noprojector_conv_v4": "Random convolutional injection at V4 - no projector",

    "injection_IT": "Random linear injection at IT",
    "injection_separate_IT": "Trained linear injection at IT" , 
    "injection_conv_IT": "Random convolutional injection at IT" ,
    "unfreeze_injection_IT": "Trained convolutional injection at IT" , 
    "subset_injection_IT": "Random linear injection of spatial information at IT", 
    "injection_conv_subset_IT": "Random convolutional injection of spatial information at IT", 
    "noprojector_linear_IT": "Random linear injection at IT - no projector", 
    "noprojector_conv_IT": "Random convolutional injection at IT - no projector", 
    
    "v1_no_injection": "Evaluation at V1, no injection", 
    "v2_no_injection": "Evaluation at V2, no injection", 
    "v4_no_injection": "Evaluation at V4, no injection", 
    "IT_no_injection": "Evaluation at IT, no injection", 

    "resnet50_untrained": "ResNet50 untrained", 
    "barlow_twins_50epochs": "Vanilla Barlow Twins", 
}
model_names = [
    # no projector - control 
    #"noprojector_control_v1", 
    #"noprojector_control_v2", 
    
    # random linear no projector
    #"noprojector_linear_v1", 
    #"noprojector_linear_v2",
    "noprojector_linear_v4", 
    #"noprojector_linear_IT", 
    
    # random convolution no projector 
    #"noprojector_conv_v1", 
    #"noprojector_conv_v2",
    "noprojector_conv_v4", 
    #"noprojector_conv_IT", 
    
    # random injection models  
    #"injection_v1",
    #"injection_v2", 
    "injection_v4",
    #"injection_IT",
    
    # convolution injection models 
    #"injection_conv_v1", 
    #"injection_conv_v2", 
    "injection_conv_v4", 
    #"injection_conv_IT", 
    
    # unfreeze convolution injection models 
    #"unfreeze_injection_v1", 
    #"unfreeze_injection_v2", 
    #"unfreeze_injection_v4", 
    #"unfreeze_injection_IT", 

    # subset 
    #"subset_injection_v1", 
    #"subset_injection_v2", 
    #"subset_injection_v4", 
    #"subset_injection_IT",

    # conv subset injection 
    #"injection_conv_subset_v1", 
    #"injection_conv_subset_v2", 
    #"injection_conv_subset_v4", 
    #"injection_conv_subset_IT",

    # separate learning of weights 
    #"injection_separate_v1", 
    #"injection_separate_v2", 
    #"injection_separate_v4", 
    #"injection_separate_IT",
    
    # control models 
    #"v1_no_injection", 
    #"v2_no_injection", 
    "v4_no_injection", 
    #"IT_no_injection", 

    "resnet50_untrained", 
    "barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
]



#fig, axes = pt.core.subplots(2, 5, size=(40, 40), sharex=True)
fig, axes = pt.core.subplots(3, 3, size=(10, 10), sharex=True)
for key, metric_types in list_metrics.items(): 
    
    #fig, axes = pt.core.subplots(1, len(metric_types), size=(15,8), sharex=True)
    
    for i, metric_type in enumerate(metric_types):
        for model_name in model_names:
            scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            
            for j in range(len(scores)): 
                if scores[j] <0: 
                    scores[j] = 0
            axes[key,i].plot(layers, scores, label=dict_model_names[model_name], color = dict_color[model_name][0], ls = dict_color[model_name][1])
        #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        #axes[0,i].plot(layers, scores, label='identity')
        
        # blocks 
        axes[key,i].axvline(x = 3, color = 'grey',  ls = 'dotted')
        axes[key,i].axvline(x = 6, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 10, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 16, color = 'red',  ls = 'dotted', linewidth=4)
        axes[key,i].axvline(x = 19, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 20, color = 'grey' , ls = 'dotted')
        
        axes[key,i].set_title(dict_metric_names[metric_type], fontsize=18)#, fontsize =60)
        axes[key,i].set_xticks([0, 3, 6, 10, 16, 19, 20])
        axes[key,i].set_xticklabels(['1st convolution', 'maxpool', 'v1 injection', 'v2 injection', 'v4 injection', 'IT injection', 'avgpool'], rotation=45, ha='right', fontsize=16)#, fontsize=60)
        
        axes[key,i].text(4.5, 0.95, "Block V1", ha="center", va="center", size=14)#, size=60)
        axes[key,i].text(8, 0.95, "Block V2", ha="center", va="center", size=14)#, size=60)
        axes[key,i].text(13, 0.95, "Block V4", ha="center", va="center", size=14)#, size=60)
        axes[key,i].text(17.5, 0.95, "Block IT", ha="center", va="center", size=14)#, size=60)
        #axes[0,i].text(23.5, 0.95, "Projector", ha="center", va="center", size=12)#, size=10)
        axes[key,i].set_ylim(0.0, 1.)
        axes[key,i].tick_params(axis='y', labelsize=14)
        #axes[0,i].legend()#loc='center left')
        #if i == len(metric_types)-2 and key == 2: 
            #axes[key,i].legend(loc='lower center', bbox_to_anchor=(1.75, 0.5), fontsize=18)#, fontsize=60)
fig.supxlabel('layers')#, fontsize=60)
fig.supylabel('decode')#, fontsize=60)
fig.tight_layout()
plt.show()
plt.savefig('/home/ec3731/issa_analysis/nn-analysis/f-decode-v4-{}.png'.format(key))
#plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/v4-decode_{}.png'.format(key))
#plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/thesis_plots/nolegends_title/V1_decode_{}.png'.format(key))
    
   



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
            ['cam_pos_x', 'cam_pos_y', 'cam_scale', 'cam_pos'], 
            ['brightness', 'contrast', 'saturation', 'hue', 'color', 'lighting'], 
            ['obj_class', 'obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pos', 'obj_pose_x', 'obj_pose_y', 'obj_pose_z', 'obj_pose']]

baseline_model = {"injection_conv_v1": "injection_v1",
                  "injection_conv_v2": "injection_v2",
                  "injection_conv_v4": "injection_v4", 
                  "injection_conv_IT": "injection_IT" }

# if plotting at injection site 
one_layer = {"injection_conv_v1": 6,
                  "injection_conv_v2": 10,
                  "injection_conv_v4": 16, 
                  "injection_conv_IT": 19}
# if plotting at last layer
"""one_layer = {"injection_conv_v1": 20,
                  "injection_conv_v2": 20,
                  "injection_conv_v4": 20, 
                   "injection_conv_IT": 20 }"""

model_names = [
    #"injection_v1",
    #"injection_v2", 
    #"injection_v4",
    #"injection_IT",
    "injection_conv_v1", 
    "injection_conv_v2", 
    "injection_conv_v4",
    "injection_conv_IT", 
    #"v4_no_injection", 
    #"resnet50_untrained", 
    #"barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
]




"""fig, axes = pt.round_plot.subplots(1,4,height_per_plot=6,width_per_plot=6)
for i, model_name in enumerate(model_names):
    #ys = [[results[model_name][metric][-1,0]-results[baseline_model_name][metric][-1,0] for metric in metrics] for metrics in metricss]
    ys = [[load_data(metric, model_name, epoch, one_layer[model_name])[metric_type] - load_data(metric, baseline_model[model_name], epoch, one_layer[model_name])[metric_type] for metric_type in metric_types] for metric_types in metricss]
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

fig.suptitle('Comparison in decoding performance between Random and Convolution injection models at injection site')
fig.tight_layout()
#pt.round_plot.savefig(fig, '/home/ec3731/issa_analysis/nn-analysis/essai1.png')
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/decode/compare_random_conv_injection_site_2.png')
fig.show()"""



