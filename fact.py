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
    # trained without projector 
    "noprojector_linear_v1":  ["brown", '--'], 
    
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

dict_metric_names = {
    'inv-background': "Background Invariance", 
    'inv-obj_motion': "Object Motion Invariance" , 
    'inv-crop': "Crop Invariance",
    'inv-color': "Color Invariance", 
    'fact-background': "Background Factorization" ,
    'fact-obj_motion': "Object Motion Factorization",
    'fact-crop': "Crop Factorization", 
    'fact-color': "Color Factorization"
}

"""list_metrics = {
    #"Subspace Invariance": ['ss_inv-background', 'ss_inv-obj_motion'], 
    #"Subspace Invariance 2": ['ss_inv-crop','ss_inv-color'], 
    "Invariance": ['inv-background', 'inv-obj_motion'], 
    "Invariance 2": ['inv-crop', 'inv-color'], 
    "Factorization": ['fact-background', 'fact-obj_motion'],
    "Factorization 2": [ 'fact-crop', 'fact-color']
}"""

list_metrics = {
    0: ['fact-background', 'fact-obj_motion'],
    1: ['fact-crop', 'fact-color']
}

dict_model_names = {
    "injection_v1": "Random linear injection at V1",
    "injection_separate_v1": "Trained linear injection at V1" , 
    "injection_conv_v1": "Random convolutional injection at V1" ,
    "unfreeze_injection_v1": "Trained convolutional injection at V1" , 
    "subset_injection_v1": "Random linear injection of subset at V1", 
    "injection_conv_subset_v1": "Random convolutional injection of subset at V1" ,
    "noprojector_linear_v1": "Random linear injection at V1 - no projector", 
    
    "injection_v2": "Random linear injection at V2",
    "injection_separate_v2": "Trained linear injection at V2" , 
    "injection_conv_v2": "Random convolutional injection at V2" ,
    "unfreeze_injection_v2": "Trained convolutional injection at V2" , 
    "subset_injection_v2": "Random linear injection of subset at V2", 
    "injection_conv_subset_v2": "Random convolutional injection of subset at V2" ,

    "injection_v4": "Random linear injection at V4",
    "injection_separate_v4": "Trained linear injection at V4" , 
    "injection_conv_v4": "Random convolutional injection at V4" ,
    "unfreeze_injection_v4": "Trained convolutional injection at V4" , 
    "subset_injection_v4": "Random linear injection of subset at V4", 
    "injection_conv_subset_v4": "Random convolutional injection of subset at V4" ,

    "injection_IT": "Random linear injection at IT",
    "injection_separate_IT": "Trained linear injection at IT" , 
    "injection_conv_IT": "Random convolutional injection at IT" ,
    "unfreeze_injection_IT": "Trained convolutional injection at IT" , 
    "subset_injection_IT": "Random linear injection of subset at IT", 
    "injection_conv_subset_IT": "Random convolutional injection of subset at IT", 
    
    "v1_no_injection": "Evaluation at V1, no injection", 
    "v2_no_injection": "Evaluation at V2, no injection", 
    "v4_no_injection": "Evaluation at V4, no injection", 
    "IT_no_injection": "Evaluation at IT, no injection", 

    "resnet50_untrained": "ResNet50 untrained", 
    "barlow_twins_50epochs": "Vanilla Barlow Twins", 
}
model_names = [
    # trained without projector 
    #"noprojector_linear_v1", 
    
    # random injection models  
    "injection_v1",
    #"injection_v2", 
    #"injection_v4",
    #"injection_IT",
    
    # convolution injection models 
    "injection_conv_v1", 
    #"injection_conv_v2", 
    #"injection_conv_v4", 
    #"injection_conv_IT", 
    
    # unfreeze convolution injection models 
    "unfreeze_injection_v1", 
    #"unfreeze_injection_v2", 
    #"unfreeze_injection_v4", 
    #"unfreeze_injection_IT", 

    # subset 
    "subset_injection_v1", 
    #"subset_injection_v2", 
    #"subset_injection_v4", 
    #"subset_injection_IT",

    # conv subset injection 
    "injection_conv_subset_v1", 
    #"injection_conv_subset_v2", 
    #"injection_conv_subset_v4", 
    #"injection_conv_subset_IT",

    # separate 
    "injection_separate_v1", 
    #"injection_separate_v2", 
    #"injection_separate_v4", 
    #"injection_separate_IT",
    
    "v1_no_injection", 
    #"v2_no_injection", 
    #"v4_no_injection", 
    #"IT_no_injection", 

    "resnet50_untrained", 
    "barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
]


# ------------------------------------------------------------------------------------
# LAYERS PLOT 
# ------------------------------------------------------------------------------------
fig, axes = pt.core.subplots(2,2 , size=(10, 10), sharex=True)
for key, metric_types in list_metrics.items(): 
    #fig, axes = pt.core.subplots(1, len(metric_types), size=(10, 8), sharex=True)
    for i, metric_type in enumerate(metric_types):
        for model_name in model_names:
        

            scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
            #axes[0,i].plot(layers, scores, label=model_name)
            axes[key,i].plot(layers, scores, label=dict_model_names[model_name], color = dict_color[model_name][0], ls = dict_color[model_name][1])
        #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
        #axes[0,i].plot(layers, scores, label='identity')
        
        axes[key,i].axvline(x = 3, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 6, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 10, color = 'grey',  ls = 'dotted')
        axes[key,i].axvline(x = 16, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 19, color = 'grey', ls = 'dotted')
        axes[key,i].axvline(x = 20, color = 'grey', ls = 'dotted')
        axes[key,i].set_xticks([0, 3, 6, 10, 16, 19, 20])
        axes[key,i].set_xticklabels(['1st convolution', 'maxpool', 'v1 injection', 'v2 injection', 'v4 injection', 'IT injection', 'avgpool'], rotation=45, ha='right',fontsize=16)
        
        axes[0,i].set_title(dict_metric_names[metric_type])
        
        axes[0,i].text(4.5, 0.2, "Block V1", ha="center", va="center", size=14)
        axes[0,i].text(8, 0.2, "Block V2", ha="center", va="center", size=14)
        axes[0,i].text(13, 0.2, "Block V4", ha="center", va="center", size=14)
        axes[0,i].text(17.5, 0.2, "Block IT", ha="center", va="center", size=14)
        axes[0,i].set_ylim(0.0, 1.0)
        axes[key,i].tick_params(axis='y', labelsize=14)
        
        axes[0,i].legend()#loc='center left')
    fig.supxlabel('layers')
    fig.supylabel('factorization')
    fig.tight_layout()
    plt.show()
    plt.savefig('/home/ec3731/issa_analysis/nn-analysis/1v1_fact_{}.png'.format(key))
    #plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/thesis_plots/nolegends_title/V1_fact_{}.png'.format(key))
    #plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/fact/IT+_no_injection_{}.png'.format(key))


# ------------------------------------------------------------------------------------
# ROUND PLOT 
# ------------------------------------------------------------------------------------

"""one_layer = 16 
fig, axes = pt.round_plot.subplots(1, 1, height_per_plot=7.5, width_per_plot=7.5, polar=True)
ax = axes[0,0]

model_names = [
    "barlow_v1_inj",
    "barlow_v2_inj", 
    "barlow_v1_inj_b",
    "barlow_control"
]

metric_types = ['ss_inv-background', 'ss_inv-obj_motion', 'ss_inv-crop','ss_inv-color','inv-background', 'inv-obj_motion', 'inv-crop', 'inv-color',
                'fact-background', 'fact-obj_motion', 'fact-crop', 'fact-color']

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
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/fact/FINAL_rond.png')
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
           # ['ss_inv-background', 'ss_inv-obj_motion', 'ss_inv-crop','ss_inv-color'],
            ['inv-background', 'inv-obj_motion', 'inv-crop', 'inv-color'],
            ['fact-background', 'fact-obj_motion', 'fact-crop', 'fact-color']]

baseline_model = {"injection_conv_v1": "injection_v1",
                  "injection_conv_v2": "injection_v2",
                  "injection_conv_v4": "injection_v4", 
                  "injection_conv_IT": "injection_IT" }

one_layer = {"injection_conv_v1": 6,
                  "injection_conv_v2": 10,
                  "injection_conv_v4": 16, 
                  "injection_conv_IT": 19 }
one_layer = {"injection_conv_v1": 20,
                  "injection_conv_v2": 20,
                  "injection_conv_v4": 20, 
            "injection_conv_IT": 20
            }

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
fig.suptitle('Comparison in factorization performance between Random and Convolution injection models at the last layer')
fig.tight_layout()
#pt.round_plot.savefig(fig, '/home/ec3731/issa_analysis/nn-analysis/essai3.png')
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/fact/compare_random_conv_last_layer.png')
fig.show()"""




