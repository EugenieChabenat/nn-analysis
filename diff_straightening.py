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
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
metric = ["curve", 1]

list_metrics = {
    "Camera Translation" : ["x_cam_trans", "y_cam_trans", "z_cam_trans"], 
    "Camera Rotation" : ["x_cam_rot", "y_cam_rot"], 
    "Camera Pan" : ['x_cam_pan', 'yz_cam_pan'], 
    "Object Rotation" : ['x_obj_rot', 'y_obj_rot'], 
}

model_names = [
    # random injection models  
    #"injection_v1_af",
    #"injection_v2_af", 
    #"injection_v4_af",
    "injection_IT_af",
    
    # convolution injection models 
    #"injection_conv_v1_af", 
    #"injection_conv_v2_af", 
    #"injection_conv_v4_af", 
    "injection_conv_IT_af", 
    
    # unfreeze convolution injection models 
    #"unfreeze_injection_v1_af", 
    #"unfreeze_injection_v2_af", 
    #"unfreeze_injection_v4_af", 
    "unfreeze_injection_IT_af", 

    # subset 
    #"subset_injection_v1", 
    #"subset_injection_v2", 
    #"subset_injection_v4", 
    "subset_injection_IT",
    
    # conv subset injection 
    #"injection_conv_subset_v1", 
    #"injection_conv_subset_v2", 
    #"injection_conv_subset_v4", 
    "injection_conv_subset_IT",
    
    # separate
    #"injection_separate_v1", 
    #"injection_separate_v2", 
    #"injection_separate_v4", 
    "injection_separate_IT", 

    #"v1_no_injection", 
    #"v2_no_injection", 
    #"v4_no_injection", 
    "IT_no_injection",
    
    "resnet50_allfeatures", 
    "bt_allfeatures", 
    #"barlow_fact_no_injection"
]
# ------------------------------------------------------------------------------------
# LAYERS PLOT 
# ------------------------------------------------------------------------------------
"""for key, metric_types in list_metrics.items(): 

    fig, axes = pt.core.subplots(1, len(metric_types), size=(10,8), sharex=True)
    for i, metric_type in enumerate(metric_types):
        scores_id = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        scores_id = [i * 180 for i in scores_id]
        for model_name in model_names:
            print('model: ', model_name)
            print('layer: ', layers)
            #scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            scores = [i * 180 for i in scores]
            
            scores = [element1 - element2 for (element1, element2) in zip(scores, scores_id)]
            
            axes[0,i].plot(layers, scores, label=model_name, color = dict_color[model_name][0], ls = dict_color[model_name][1])
        #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        #axes[0,i].plot(layers, scores, label='identity', color = 'black')
        axes[0,i].axhline(y=0, label='identity', color = 'black')
        
        axes[0,i].axvline(x = 3, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 6, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 10, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 16, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 19, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 20, color = 'grey', alpha = 0.5, ls = 'dotted')
        
        axes[0,i].set_title(metric_type)
        axes[0,i].set_xticks([0, 3, 6, 10, 16, 19, 20])
        axes[0,i].set_xticklabels(['convolution', 'maxpool', 'inj v1', 'inj v2', 'inj v4', 'inj IT', 'avgpool'], rotation=45, ha='right')
        
        axes[0,i].text(4.5, 0.2, "Block V1", ha="center", va="center", size=12)
        axes[0,i].text(8, 0.2, "Block V2", ha="center", va="center", size=12)
        axes[0,i].text(13, 0.2, "Block V4", ha="center", va="center", size=12)
        axes[0,i].text(17.5, 0.2, "Block IT", ha="center", va="center", size=12)
        #axes[0,i].set_ylim(0.0, 1.)
        axes[0,i].legend(loc='lower left')
    fig.supxlabel('layers')
    fig.supylabel('curvature in degrees')
    fig.tight_layout()
    plt.show()
    #plt.savefig('/home/ec3731/issa_analysis/nn-analysis/lookv1_{}.png'.format(key))
    plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/final_metrics/degrees/IT+no_injection_{}.png'.format(key))
"""



# ------------------------------------------------------------------------------------
# HISTOGRAM PLOT 
# ------------------------------------------------------------------------------------

    
list_metrics = {
    "Camera Translation" : ["x_cam_trans", "y_cam_trans", "z_cam_trans"], 
    "Camera Rotation" : ["x_cam_rot", "y_cam_rot"], 
    "Camera Pan" : ['x_cam_pan', 'yz_cam_pan'], 
    "Object Rotation" : ['x_obj_rot', 'y_obj_rot'], 
}
    
metricss = [
           ["x_cam_trans", "y_cam_trans", "z_cam_trans"], 
            ["x_cam_rot", "y_cam_rot"], 
            ['x_cam_pan', 'yz_cam_pan']]

baseline_model = {"injection_v1_af": "v1_no_injection",
                  "injection_conv_v1_af" : "v1_no_injection",
                  "unfreeze_injection_v1_af": "v1_no_injection", 
                  "subset_injection_v1": "v1_no_injection", 
                  "injection_conv_subset_v1": "v1_no_injection" ,
                 "injection_separate_v1": "v1_no_injection", 
                 
                 "injection_v2_af": "v2_no_injection",
                  "injection_conv_v2_af" : "v2_no_injection",
                  "unfreeze_injection_v2_af": "v2_no_injection", 
                  "subset_injection_v2": "v2_no_injection", 
                  "injection_conv_subset_v2": "v2_no_injection" ,
                 "injection_separate_v2": "v2_no_injection", 
                 
                 "injection_v4_af": "v4_no_injection",
                  "injection_conv_v4_af" : "v4_no_injection",
                  "unfreeze_injection_v4_af": "v4_no_injection", 
                  "subset_injection_v4": "v4_no_injection", 
                  "injection_conv_subset_v4": "v4_no_injection" ,
                 "injection_separate_v4": "v4_no_injection", 
                 
                 "injection_IT_af": "IT_no_injection",
                  "injection_conv_IT_af" : "IT_no_injection",
                  "unfreeze_injection_IT_af": "IT_no_injection", 
                  "subset_injection_IT": "IT_no_injection", 
                  "injection_conv_subset_IT": "IT_no_injection" ,
                 "injection_separate_IT": "IT_no_injection"}

one_layer = {"injection_v1_af": [6, 20],
                  "injection_conv_v1_af" : [6, 20],
                  "unfreeze_injection_v1_af": [6, 20], 
                  "subset_injection_v1": [6, 20], 
                  "injection_conv_subset_v1": [6, 20],
                 "injection_separate_v1": [6, 20], 

             "injection_v2_af": [10, 20],
                  "injection_conv_v2_af" : [10, 20],
                  "unfreeze_injection_v2_af": [10, 20], 
                  "subset_injection_v2": [10, 20], 
                  "injection_conv_subset_v2": [10, 20],
                 "injection_separate_v2": [10, 20], 

             "injection_v4_af": [16, 20],
                  "injection_conv_v4_af" : [16, 20],
                  "unfreeze_injection_v4_af": [16, 20], 
                  "subset_injection_v4": [16, 20], 
                  "injection_conv_subset_v4": [16, 20],
                 "injection_separate_v4": [16, 20], 

             "injection_IT_af": [19, 20],
                  "injection_conv_IT_af" : [19, 20],
                  "unfreeze_injection_IT_af": [19, 20], 
                  "subset_injection_IT": [19, 20], 
                  "injection_conv_subset_IT": [19, 20],
                 "injection_separate_IT": [19, 20], 
             
}


dict_model_names = {
    "injection_v1_af": "Random linear injection at V1",
    "injection_separate_v1": "Trained linear injection at V1" , 
    "injection_conv_v1_af": "Random convolutional injection at V1" ,
    "unfreeze_injection_v1_af": "Trained convolutional injection at V1" , 
    "subset_injection_v1": "Random linear injection of subset at V1", 
    "injection_conv_subset_v1": "Random convolutional injection of subset at V1" ,

    "injection_v2_af": "Random linear injection at V2",
    "injection_separate_v2": "Trained linear injection at V2" , 
    "injection_conv_v2_af": "Random convolutional injection at V2" ,
    "unfreeze_injection_v2_af": "Trained convolutional injection at V2" , 
    "subset_injection_v2": "Random linear injection of subset at V2", 
    "injection_conv_subset_v2": "Random convolutional injection of subset at V2" ,

    "injection_v4_af": "Random linear injection at V4",
    "injection_separate_v4": "Trained linear injection at V4" , 
    "injection_conv_v4_af": "Random convolutional injection at V4" ,
    "unfreeze_injection_v4_af": "Trained convolutional injection at V4" , 
    "subset_injection_v4": "Random linear injection of subset at V4", 
    "injection_conv_subset_v4": "Random convolutional injection of subset at V4" ,

    "injection_IT_af": "Random linear injection at IT",
    "injection_separate_IT": "Trained linear injection at IT" , 
    "injection_conv_IT_af": "Random convolutional injection at IT" ,
    "unfreeze_injection_IT_af": "Trained convolutional injection at IT" , 
    "subset_injection_IT": "Random linear injection of subset at IT", 
    "injection_conv_subset_IT": "Random convolutional injection of subset at IT" ,
    

    #"v4_no_injection", 
    #"resnet50_untrained", 
    #"barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
}    
model_names = [
    "injection_v1_af",
    "injection_separate_v1", 
    "subset_injection_v1", 
    "injection_conv_v1_af",
    "unfreeze_injection_v1_af", 
    "injection_conv_subset_v1",
    
    #"injection_v2_af",
    #"injection_separate_v2", 
    #"subset_injection_v2", 
    #"injection_conv_v2_af",
    #"unfreeze_injection_v2_af", 
    #"injection_conv_subset_v2",

    #"injection_v4_af",
    #"injection_separate_v4", 
    #"subset_injection_v4", 
    #"injection_conv_v4_af",
    #"unfreeze_injection_v4_af", 
    #"injection_conv_subset_v4",

    #"injection_IT_af",
    #"injection_separate_IT", 
    #"subset_injection_IT", 
    #"injection_conv_IT_af",
    #"unfreeze_injection_IT_af", 
    #"injection_conv_subset_IT",
    
    #"v4_no_injection", 
    #"resnet50_untrained", 
    #"barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
]
alphas = 1 # 0.5#, 0.5]
#colors = ["darkblue", "blue", "lightblue"]
edge_colors = "black"#, "black"]
colors =  ["darkblue", "lightblue"]
def grouped_bar(ax, xs, ys, ys_, alpha, colors, edgecolor, width=0.2, sep=0.3):
    assert len(xs) == len(ys)
    total = 0.0
    all_xticks = []
    all_xlabels = []
    fig2 = plt.subplots()
    for i, (y, y_) in enumerate(zip(ys, ys_)):
        xticks = np.linspace(0.0,len(y)*width,num=len(y))+total
        
        p1 = ax.bar(xticks, y, width=width, alpha = alpha, color =  colors[0], edgecolor = edgecolor)
        p2 = ax.bar(xticks, y_, width=width/2, alpha = alpha, color = colors[1], edgecolor = edgecolor)

        
        total += (len(y)+1.5)*width + sep
        all_xticks += list(xticks)
        all_xlabels += xs[i]
    ax.axhline(y=0, label='identity', color = 'grey', ls = '--')
    ax.legend((p1[0], p2[0]), ('injection site', 'last layer'), loc="lower left")
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xlabels, rotation=45, ha='right')
    
    
fig, axes = pt.round_plot.subplots(2,3,height_per_plot=6,width_per_plot=6)
k = 0 
j = 0 
for i, model_name in enumerate(model_names):

    ys = [[ (load_data(metric, model_name, epoch, one_layer[model_name][0])[metric_type] - load_data(metric, baseline_model[model_name], epoch, one_layer[model_name][0])[metric_type])\
           *100/load_data(metric, baseline_model[model_name], epoch, one_layer[model_name][0])[metric_type] for metric_type in metric_types] for metric_types in metricss]

    ys_ = [[ (load_data(metric, model_name, epoch, one_layer[model_name][1])[metric_type] - load_data(metric, baseline_model[model_name], epoch, one_layer[model_name][1])[metric_type])\
           *100/load_data(metric, baseline_model[model_name], epoch, one_layer[model_name][1])[metric_type] for metric_type in metric_types] for metric_types in metricss]
  
    xs = [[metric_dict[metric_type] for metric_type in metrics] for metrics in metricss]
    
    grouped_bar(axes[k,j], xs, ys, ys_, alphas, colors, edge_colors)
    #grouped_bar(axes[k, j], xs, ys_, alphas[1], colors[1], edge_colors[1])
    
    axes[k, j].set_title(dict_model_names[model_name])
    axes[k, j].set_ylabel('Difference relative to baseline, in %')
    j += 1 
    if j >2: 
        k += 1 
        j = 0 

#y_lim_min = min([axes[0, i].get_ylim()[0] for i in range(len(model_names))])
#y_lim_max = max([axes[0, i].get_ylim()[1] for i in range(len(model_names))])

y_lim_min = min(min([axes[0, i].get_ylim()[0] for i in range(3)]), min([axes[1, i].get_ylim()[0] for i in range(3)]))
y_lim_max = max(max([axes[0, i].get_ylim()[1] for i in range(3)]), max([axes[1, i].get_ylim()[1] for i in range(3)])) 

for i in range(3):
    axes[0, i].set_ylim(y_lim_min, y_lim_max)
    axes[1, i].set_ylim(y_lim_min, y_lim_max)
#fig.suptitle('Comparison in straightening performance between injection V1 models and control (no injection) at injection site', fontweight='bold')
fig.tight_layout()

pt.round_plot.savefig(fig, '/home/ec3731/issa_analysis/nn-analysis/no-title_straight_v1.png')
#pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/hist-compare_straightening_v1.png')
fig.show()
