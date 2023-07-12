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

dict_color = {
    # no projector linear 
    "noprojector_linear_v1":  ["brown", '--'], 
    "noprojector_linear_v2":  ["brown", '--'], 
    "noprojector_linear_v1_nm3":  ["brown", '--'], 
    "noprojector_linear_v4":  ["brown", '--'], 
    
    'injection_conv_subset_v1_proj':["black", '-'],
    
    # no projector conv
    "noprojector_conv_v2": ["gold", '--'], 
    "noprojector_conv_v4": ["gold", '--'], 
    "noprojector_conv_IT": ["gold", '--'], 
    
    
    # random injection 
    "injection_v1_af" : ["orange", '-'],
    "injection_v2_af": ["orange", '-'], 
    "injection_v4_af": ["orange", '-'],
    "injection_IT_af": ["orange", '-'],
    
    # convolution injection
    "injection_conv_v1_af": ["lightblue", '-'], 
    "injection_conv_v2_af": ["lightblue", '-'], 
    "injection_conv_v4_af": ["lightblue", '-'], 
    "injection_conv_IT_af": ["lightblue", '-'],  
    
    #"injection_conv_v1_af": ["red", '-'], 
    #"injection_conv_v2_af": ["blue", '-'], 
    #"injection_conv_v4_af": ["orange", '-'], 
    #"injection_conv_IT_af": ["green", '-'], 
        
    # unfreeze convolution injection 
    "unfreeze_injection_v1_af": ["green", '-'], 
    "unfreeze_injection_v2_af": ["green", '-'], 
    "unfreeze_injection_v4_af": ["green", '-'], 
    "unfreeze_injection_IT_af": ["green", '-'], 
    
    #"unfreeze_injection_v1_af": ["red", '-'], 
    #"unfreeze_injection_v2_af": ["blue", '-'], 
    #"unfreeze_injection_v4_af": ["orange", '-'], 
    #"unfreeze_injection_IT_af": ["green", '-'], 

    # subset injection 
    "subset_injection_v1": ["blue", '-'], 
    "subset_injection_v2": ["blue", '-'], 
    "subset_injection_v4": ["blue", '-'], 
    "subset_injection_IT": ["blue", '-'],

    # conv subset injection 
    "injection_conv_subset_v1": ["lime", '-'], 
    "injection_conv_subset_v2": ["lime", '-'], 
    "injection_conv_subset_v4": ["lime", '-'], 
    "injection_conv_subset_IT": ["lime", '-'],

    # separate injection 
    "injection_separate_v1": ["purple", '-'], 
    "injection_separate_v2": ["purple", '-'], 
    "injection_separate_v4": ["purple", '-'], 
    "injection_separate_IT": ["purple", '-'], 
    
    # control models 
    #"v1_no_injection": ["red", '--'], 
    #"v2_no_injection": ["blue", '--'], 
    #"v4_no_injection": ["green", '--'], 
    #"IT_no_injection": ["orange", '--'],

    "v1_no_injection": ["red", '--'], 
    "v2_no_injection": ["red", '--'], 
    "v4_no_injection": ["red", '--'], 
    "IT_no_injection": ["red", '--'], 
    
    "resnet50_allfeatures": ["pink", '--'], 
    "bt_allfeatures": ["grey", '--'], 
    #"barlow_fact_no_injection": ["black", '--']
}

epoch = 29
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]#, 21, 22, 23, 24, 25, 26, 27]
metric = ["curve", 1]

dict_metric_names = {
    "x_cam_trans": "Camera Translation - x", 
    "y_cam_trans": "Camera Translation - y", 
    "z_cam_trans": "Camera Translation - z", 
    "x_cam_rot": "Camera Rotation - x", 
    "y_cam_rot": "Camera Rotation - y", 
    'x_cam_pan': "Camera Pan - x", 
    'yz_cam_pan': "Camera Pan - yz", 
    'x_obj_rot': "Object Rotation - x" ,
    'y_obj_rot': "Object Rotation - y" , 
}

"""list_metrics = {
    "Camera Translation" : ["x_cam_trans", "y_cam_trans", "z_cam_trans"], 
    "Camera Rotation" : ["x_cam_rot", "y_cam_rot"], 
    "Camera Pan" : ['x_cam_pan', 'yz_cam_pan'], 
    #"Object Rotation" : ['x_obj_rot', 'y_obj_rot'], 
}"""

list_metrics = {
    #0 : ["x_cam_trans", "y_cam_trans", "z_cam_trans"], 
    0 : ["x_cam_rot", "y_cam_rot"], 
    #0 : ['x_cam_pan', 'yz_cam_pan']
}

dict_model_names = {
    "injection_conv_subset_v1_proj": "Random convolutional injection after projector at V1",  
    
    "injection_v1_af": "Random linear injection at V1",
    "injection_separate_v1": "Trained linear injection at V1" , 
    "injection_conv_v1_af": "Random convolutional injection at V1" ,
    "unfreeze_injection_v1_af": "Trained convolutional injection at V1" , 
    "subset_injection_v1": "Random linear injection of subset at V1", 
    "injection_conv_subset_v1": "Random convolutional injection of subset at V1" ,
    "noprojector_linear_v1": "Random linear injection at V1 - no projector" , 

    "injection_v2_af": "Random linear injection at V2",
    "injection_separate_v2": "Trained linear injection at V2" , 
    "injection_conv_v2_af": "Random convolutional injection at V2" ,
    "unfreeze_injection_v2_af": "Trained convolutional injection at V2" , 
    "subset_injection_v2": "Random linear injection of subset at V2", 
    "injection_conv_subset_v2": "Random convolutional injection of subset at V2" ,
    "noprojector_linear_v2": "Random linear injection at V2 - no projector",
    "noprojector_conv_v2": "Random convolutional injection at V2 - no projector" ,
    
    "injection_v4_af": "Random linear injection at V4",
    "injection_separate_v4": "Trained linear injection at V4" , 
    "injection_conv_v4_af": "Random convolutional injection at V4" ,
    "unfreeze_injection_v4_af": "Trained convolutional injection at V4" , 
    "subset_injection_v4": "Random linear injection of subset at V4", 
    "injection_conv_subset_v4": "Random convolutional injection of subset at V4" ,
    "noprojector_linear_v4": "Random linear injection at V4 - no projector",
    "noprojector_conv_v4": "Random convolutional injection at V4 - no projector",

    "injection_IT_af": "Random linear injection at IT",
    "injection_separate_IT": "Trained linear injection at IT" , 
    "injection_conv_IT_af": "Random convolutional injection at IT" ,
    "unfreeze_injection_IT_af": "Trained convolutional injection at IT" , 
    "subset_injection_IT": "Random linear injection of subset at IT", 
    "injection_conv_subset_IT": "Random convolutional injection of subset at IT", 
    "noprojector_conv_IT": "Random convolutional injection of subset at IT - no projector", 
    
    "v1_no_injection": "Evaluation at V1, no injection", 
    "v2_no_injection": "Evaluation at V2, no injection", 
    "v4_no_injection": "Evaluation at V4, no injection", 
    "IT_no_injection": "Evaluation at IT, no injection", 
    
    
    "resnet50_allfeatures": "ResNet50 untrained", 
    "bt_allfeatures": "Vanilla Barlow Twins", 
}
model_names = [
    #"injection_conv_subset_v1_proj", 
    
    # no projector linear 
    #"noprojector_linear_v1",
    #"noprojector_linear_v1_nm3",
    #"noprojector_linear_v2", 
    "noprojector_linear_v4", 
    
    # random conv no projector 
    #"noprojector_conv_v2", 
    "noprojector_conv_v4",
    #"noprojector_conv_IT", 
    
    # random injection models  
    #"injection_v1_af",
    #"injection_v2_af", 
    "injection_v4_af",
    #"injection_IT_af",
    
    # convolution injection models 
    #"injection_conv_v1_af", 
    #"injection_conv_v2_af", 
    "injection_conv_v4_af", 
    #"injection_conv_IT_af", 
    
    # unfreeze convolution injection models 
    #"unfreeze_injection_v1_af", 
    #"unfreeze_injection_v2_af", 
    "unfreeze_injection_v4_af", 
    #"unfreeze_injection_IT_af", 

    # subset 
    #"subset_injection_v1", 
    #"subset_injection_v2", 
    "subset_injection_v4", 
    #"subset_injection_IT",
    
    # conv subset injection 
    #"injection_conv_subset_v1", 
    #"injection_conv_subset_v2", 
    "injection_conv_subset_v4", 
    #"injection_conv_subset_IT",
    
    # separate
    #"injection_separate_v1", 
    #"injection_separate_v2", 
    "injection_separate_v4", 
    #"injection_separate_IT", 

    #"v1_no_injection", 
    #"v2_no_injection", 
    "v4_no_injection", 
    #"IT_no_injection",
    
    "resnet50_allfeatures", 
    "bt_allfeatures", 
    #"barlow_fact_no_injection"
]
# ------------------------------------------------------------------------------------
# LAYERS PLOT 
# ------------------------------------------------------------------------------------
fig, axes = pt.core.subplots(1, 2, size=(10, 8), sharex=True)
for key, metric_types in list_metrics.items(): 

    #fig, axes = pt.core.subplots(1, len(metric_types), size=(10,8), sharex=True)
    for i, metric_type in enumerate(metric_types):
        for model_name in model_names:
            
            scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            axes[key,i].plot(layers, scores, label=dict_model_names[model_name], color = dict_color[model_name][0], ls = dict_color[model_name][1])
        scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        axes[key,i].plot(layers, scores, label='identity', color = 'black')
        
        axes[key,i].axvline(x = 3, color = 'grey',  ls = 'dotted')
        axes[key,i].axvline(x = 6, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 10, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 16, color = 'grey',  ls = 'dotted')
        axes[key,i].axvline(x = 19, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 20, color = 'grey' , ls = 'dotted')
        
        axes[key,i].set_title(dict_metric_names[metric_type], fontsize=18)
        axes[key,i].set_xticks([0, 3, 6, 10, 16, 19, 20])
        axes[key,i].set_xticklabels(['1st convolution', 'maxpool', 'v1 injection', 'v2 injection', 'v4 injection', 'IT injection', 'avgpool'], rotation=45, ha='right', fontsize=16)
        
        axes[key,i].text(4.5, 0.95, "Block V1", ha="center", va="center", size=14)#, size=60)
        axes[key,i].text(8, 0.95, "Block V2", ha="center", va="center", size=14)#, size=60)
        axes[key,i].text(13, 0.95, "Block V4", ha="center", va="center", size=14)#, size=60)
        axes[key,i].text(17.5, 0.95, "Block IT", ha="center", va="center", size=14)#, size=60)
        #axes[0,i].text(23.5, 0.9, "Projector", ha="center", va="center", size=10)
        axes[key,i].set_ylim(0.0, 1.)
        #if i == len(metric_types)-1 and key==0: 
            #axes[0,i].legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
fig.supxlabel('layers')
fig.supylabel('curvature')
fig.tight_layout()
plt.show()
plt.savefig('/home/ec3731/issa_analysis/nn-analysis/v4-camrot-{}.png'.format(key))
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/v4_camrot_{}.png'.format(key))
#plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/thesis_plots/nolegends_title/V1_straightening_{}.png'.format(key))
    




# ------------------------------------------------------------------------------------
# ROUND PLOT 
# ------------------------------------------------------------------------------------
one_layer = 16
fig, axes = pt.round_plot.subplots(1, 1, height_per_plot=7.5, width_per_plot=7.5, polar=True)
ax = axes[0,0]

model_names = [
    "barlow_v1_inj",
    "barlow_v2_inj", 
    "barlow_v1_inj_b",
    "barlow_control"
]
"""model_names = [
    "barlow_faces_texture",
    "barlow_faces_notexture",
    "barlow_control"
]"""
metric_types = ['x_pan-detailed', 'x_pan', 'y_pan-detailed', 'y_pan', 'z_pan-detailed', 'z_pan', 'x_focus_pan-detailed', 'x_focus_pan', 
                'y_focus_pan-detailed', 'y_focus_pan', 'z_focus_pan-detailed', 'z_focus_pan', 'x_cam_pan-detailed', 'x_cam_pan', 'yz_cam_pan-detailed',
                'yz_cam_pan', 'x_cam_rot', 'y_cam_rot', 'x_cam_trans', 'y_cam_trans', 'z_cam_trans', 
                'x_obj_rot', 'y_obj_rot']

"""x = metric_types
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
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/FINAL_rond_layer16.png')
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
            ['x_pan',  'y_pan', 'z_pan', 'x_focus_pan', 'y_focus_pan', 'z_focus_pan'], 
            ['x_cam_pan', 'yz_cam_pan', 'x_cam_rot', 'y_cam_rot', 'x_cam_trans', 'y_cam_trans', 'z_cam_trans'], 
            ['x_obj_rot', 'y_obj_rot']]

baseline_model = {"injection_conv_v1": "injection_v1",
                  "injection_conv_v2": "injection_v2",
                  "injection_conv_v4": "injection_v4", 
                 "injection_conv_IT": "injection_IT"}

one_layer = {"injection_conv_v1": 6,
                  "injection_conv_v2": 10,
                  "injection_conv_v4": 16, 
                "injection_conv_IT": 19 }

one_layer = {"injection_conv_v1": 20,
                  "injection_conv_v2": 20,
                  "injection_conv_v4" :20, 
                "injection_conv_IT": 20}
            
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
fig.suptitle('Comparison in straightening performance between Random and Convolution injection models at the last layer')
fig.tight_layout()

pt.round_plot.savefig(fig, '/home/ec3731/issa_analysis/nn-analysis/essai2.png')
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/compare_random_conv_last_layer.png')
fig.show()"""


