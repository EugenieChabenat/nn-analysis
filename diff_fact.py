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
    

metric_dict = {'ss_inv-background': 'SS Invariance - Background', 
               'ss_inv-obj_motion': 'SS Invariance - Object Motion', 
               'ss_inv-crop': 'SS Invariance - Crop',
               'ss_inv-color': 'SS Invariance - Color',
               'inv-background': 'Invariance - Background', 
               'inv-obj_motion': 'Invariance - Object Motion', 
               'inv-crop': 'Invariance - Crop',
               'inv-color': 'Invariance - Color',
               'fact-background': 'Factorization - Background', 
               'fact-obj_motion': 'Factorization - Object Motion', 
               'fact-crop': 'Factorization - Crop', 
               'fact-color': 'Factorization - Color'
              }
dict_color = {
    # random injection 
    "injection_v1" : ["orange", '-'],
    "injection_v2": ["orange", '-'], 
    "injection_v4": ["orange", '-'],
    "injection_IT": ["orange", '-'],
    
    # convolution injection
    "injection_conv_v1": ["lightblue", '-'], 
    "injection_conv_v2": ["lightblue", '-'], 
    "injection_conv_v4": ["lightblue", '-'], 
    "injection_conv_IT": ["lightblue", '-'], 
    
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
    "v1_no_injection": ["red", '--'], 
    "v2_no_injection": ["red", '--'], 
    "v4_no_injection": ["red", '--'], 
    "IT_no_injection": ["red", '--'], 

    "resnet50_untrained": ["pink", '--'], 
    "barlow_twins_50epochs": ["grey", '--'], 
    "barlow_fact_no_injection": ["black", '--']
}
  
epoch = 29
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
one_layer = 0
metric = ["fact", 0]

metric_types = ['ss_inv-background', 'ss_inv-obj_motion', 'ss_inv-crop','ss_inv-color']
                
#metric_types = ['inv-background', 'inv-obj_motion', 'inv-crop', 'inv-color']
                
#metric_types = ['fact-background', 'fact-obj_motion', 'fact-crop', 'fact-color']

list_metrics = {
    "Subspace Invariance": ['ss_inv-background', 'ss_inv-obj_motion'], 
    "Subspace Invariance 2": ['ss_inv-crop','ss_inv-color'], 
    "Invariance": ['inv-background', 'inv-obj_motion'], 
    "Invariance 2": ['inv-crop', 'inv-color'], 
    "Factorization": ['fact-background', 'fact-obj_motion'],
    "Factorization 2": [ 'fact-crop', 'fact-color']
}


model_names = [
    # random injection models  
    #"injection_v1",
    #"injection_v2", 
    #"injection_v4",
    "injection_IT",
    
    # convolution injection models 
    #"injection_conv_v1", 
    #"injection_conv_v2", 
    #"injection_conv_v4", 
    "injection_conv_IT", 
    
    # unfreeze convolution injection models 
    #"unfreeze_injection_v1", 
    #"unfreeze_injection_v2", 
    #"unfreeze_injection_v4", 
    "unfreeze_injection_IT", 

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

    "resnet50_untrained", 
    "barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
]

"""model_names = [
    "barlow_faces_texture",
    "barlow_faces_notexture", 
    "barlow_faces_control",
    "faces_pretrained_notexture"
    #"barlow_control"
]"""
# ------------------------------------------------------------------------------------
# LAYERS PLOT 
# ------------------------------------------------------------------------------------
"""for key, metric_types in list_metrics.items(): 
    fig, axes = pt.core.subplots(1, len(metric_types), size=(10, 8), sharex=True)
    for i, metric_type in enumerate(metric_types):
        for model_name in model_names:
            print('model: ', model_name)
            print('layer: ', layers)

            scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            #axes[0,i].plot(layers, scores, label=model_name)
            axes[0,i].plot(layers, scores, label=model_name, color = dict_color[model_name][0], ls = dict_color[model_name][1])
        #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        #axes[0,i].plot(layers, scores, label='identity')
        
        axes[0,i].axvline(x = 3, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 6, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 10, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 16, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 19, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].axvline(x = 20, color = 'grey', alpha = 0.5, ls = 'dotted')
        axes[0,i].set_xticks([0, 3, 6, 10, 16, 19, 20])
        axes[0,i].set_xticklabels(['', 'maxpool', 'inj v1', 'inj v2', 'inj v4', 'inj IT', 'avgpool'], rotation=45, ha='right')
        
        axes[0,i].set_title(metric_type)
        
        axes[0,i].text(4.5, 0.2, "Block V1", ha="center", va="center", size=12)
        axes[0,i].text(8, 0.2, "Block V2", ha="center", va="center", size=12)
        axes[0,i].text(13, 0.2, "Block V4", ha="center", va="center", size=12)
        axes[0,i].text(17.5, 0.2, "Block IT", ha="center", va="center", size=12)
        axes[0,i].set_ylim(0.0, 1.0)
        
        axes[0,i].legend(loc='center left')
    fig.supxlabel('layers')
    fig.supylabel('fact')
    fig.tight_layout()
    plt.show()
    #plt.savefig('/home/ec3731/issa_analysis/nn-analysis/fact_v2_{}.png'.format(key))
    plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/fact/IT+_no_injection_{}.png'.format(key))"""


# ------------------------------------------------------------------------------------
# ROUND PLOT 
# ------------------------------------------------------------------------------------

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
           # ['ss_inv-background', 'ss_inv-obj_motion', 'ss_inv-crop','ss_inv-color'],
            ['inv-background', 'inv-obj_motion', 'inv-crop', 'inv-color'],
            ['fact-background', 'fact-obj_motion', 'fact-crop', 'fact-color']]

baseline_model = {"injection_v1": "v1_no_injection",
                  "injection_conv_v1" : "v1_no_injection",
                  "unfreeze_injection_v1": "v1_no_injection", 
                  "subset_injection_v1": "v1_no_injection", 
                  "injection_conv_subset_v1": "v1_no_injection" ,
                 "injection_separate_v1": "v1_no_injection"}

one_layer = {"injection_v1": [6, 20],
                  "injection_conv_v1" : [6, 20],
                  "unfreeze_injection_v1": [6, 20], 
                  "subset_injection_v1": [6, 20], 
                  "injection_conv_subset_v1": [6, 20],
                 "injection_separate_v1": [6, 20], 
}

model_names = [
    "injection_v1",
    "injection_separate_v1", 
    "subset_injection_v1", 
    "injection_conv_v1",
    "unfreeze_injection_v1", 
    "injection_conv_subset_v1",
    
    #"injection_v1",
    #"injection_v2", 
    #"injection_v4",
    #"injection_IT",
    #"injection_conv_v1", 
    #"injection_conv_v2", 
    #"injection_conv_v4", 
    #"injection_conv_IT", 
    #"v4_no_injection", 
    #"resnet50_untrained", 
    #"barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
]
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

y_lim_min = min(min([axes[0, i].get_ylim()[0] for i in range(3)]), min([axes[1, i].get_ylim()[0] for i in range(3)]))
y_lim_max = max(max([axes[0, i].get_ylim()[1] for i in range(3)]), max([axes[1, i].get_ylim()[1] for i in range(3)])) 

for i in range(3):
    axes[0, i].set_ylim(y_lim_min, y_lim_max)
    axes[1, i].set_ylim(y_lim_min, y_lim_max)
fig.suptitle('Comparison in factorization performance between injection V1 models and control (no injection) at injection site', fontweight='bold')
fig.tight_layout()

pt.round_plot.savefig(fig, '/home/ec3731/issa_analysis/nn-analysis/factper1.png')
#pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/compare_random_conv_last_layer.png')
fig.show()

