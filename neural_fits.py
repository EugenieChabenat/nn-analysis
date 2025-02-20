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

dict_color = {
    # new architectures 
    "inj_v1_evaluate_IT": ["magenta", '-'], 
    "inj_v2_evaluate_IT": ["forestgreen", '-'], 
    
    # no projector control 
    "noprojector_control_v1":  ["black", '-'], 
    "noprojector_control_v2":  ["black", '-'], 
    "noprojector_control_v4":  ["black", '-'], 
    "noprojector_control_IT":  ["black", '-'], 
    
    # no projector linear 
    "noprojector_linear_v1":  ["brown", '-'], 
    "noprojector_linear_v2":  ["brown", '-'], 
    "noprojector_linear_v4":  ["brown", '-'], 
    "noprojector_linear_IT":  ["brown", '-'], 
    
    # no projector conv
    "noprojector_conv_v1": ["gold", '-'], 
    "noprojector_conv_v2": ["gold", '-'], 
    "noprojector_conv_v4": ["gold", '-'], 
    "noprojector_conv_IT": ["gold", '-'], 

    # multiplicative injection 
    "multiplicative_model_v1" : ["darkblue", '-'], # '-'],
    "multiplicative_model_v2": ["darkblue", '-'], # '-'],
    "multiplicative_model_v4": ["darkblue",  '-'], # '-'],
    "multiplicative_model_IT": ["darkblue",  '-'], # '-'],

    "multiplicative_separate_v2": ["purple", '-'], # '-'],
    "multiplicative_linear_v2": ["lightblue", '-'], # '-'],
    "multiplicative_unfreeze_v2": ["purple", '-'], # '-'],
    "multiplicative_afterproj": ["yellow", '-'], # '-'],

    # multiplicative injection 
    "injection_avgpool_v1" : ["forestgreen", '-'], # '-'],
    "injection_avgpool_v2": ["forestgreen", '-'], # '-'],
    "injection_avgpool_v4": ["forestgreen",  '-'], # '-'],
    "injection_avgpool_IT": ["forestgreen",  '-'], # '-'],

    # random injection 
    "injection_v1" : ["orange", ':'], # '-'],
    "injection_v2": ["orange", ':'], # '-'],
    "injection_v4": ["orange",  ':'], # '-'],
    "injection_IT": ["orange",  ':'], # '-'],
    
    # convolution injection
    "injection_conv_v1": ["lightblue", ':'], # '-'],
    "injection_conv_v2": ["lightblue",':'], # '-'],
    "injection_conv_v4": ["lightblue", ':'], # '-'],
    "injection_conv_IT": ["lightblue", ':'], # '-'],
    
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
epoch = 29
layers = np.arange(2)
#layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#layers =[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
layers =[3, 6,  10, 16, 19, 20]
metric = ["neural_fits", 0]
metric_types = ['hvm-V4', 'hvm-IT', 'rust-V4', 'rust-IT']

model_names = [
    # new architectures 
    #"inj_v1_evaluate_IT", 
    #"inj_v2_evaluate_IT", 
    
    
    # no projector - control 
    #"noprojector_control_v1", 
    #"noprojector_control_v2", 
    #"noprojector_control_v4", 
    #"noprojector_control_IT", 
    
    # random linear no projector
    #"noprojector_linear_v1", 
    #"noprojector_linear_v2",
    #"noprojector_linear_v4", 
    #"noprojector_linear_IT", 
    
    # random convolution no projector 
    #"noprojector_conv_v1", 
    #"noprojector_conv_v2",
    #"noprojector_conv_v4", 
    #"noprojector_conv_IT", 

    # multiplicative models, 
    #"multiplicative_model_v1", 
    "multiplicative_model_v2", 
    #"multiplicative_model_v4", 
    #"multiplicative_model_IT", 

    "multiplicative_separate_v2",
    "multiplicative_linear_v2",
    "multiplicative_unfreeze_v2",
    "multiplicative_afterproj",

    # injection into avgpool
    #"injection_avgpool_v1", 
    #"injection_avgpool_v2", 
    #"injection_avgpool_v4", 
    #"injection_avgpool_IT", 
    
    # random injection models  
    #"injection_v1",
    #"injection_v2", 
    #"injection_v4",
    #"injection_IT",
    
    # convolution injection models 
    #"injection_conv_v1", 
    #"injection_conv_v2", 
    #"injection_conv_v4", 
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
    #"v4_no_injection", 
    #"IT_no_injection", 

    #"resnet50_untrained", 
    "barlow_twins_50epochs", 
    #"barlow_fact_no_injection"
]


fig, axes = pt.core.subplots(1, len(metric_types), size=(10, 10), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        
        scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
        axes[0,i].plot(layers, scores, label=model_name, color = dict_color[model_name][0])
    #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]

    axes[0,i].axvline(x = 6, color = 'grey', ls = 'dotted')#, linewidth=4)
    axes[0,i].axvline(x = 10, color = 'red', ls = 'dotted', linewidth=4)
    axes[0,i].axvline(x = 16, color = 'grey',  ls = 'dotted')
    axes[0,i].axvline(x = 19, color = 'grey', ls = 'dotted')#, linewidth=4)
    axes[0,i].axvline(x = 20, color = 'grey' , ls = 'dotted')

    axes[0,i].text(4.5, 0.95, "Block V1", ha="center", va="center", size=14)#, size=60)
    axes[0,i].text(8, 0.95, "Block V2", ha="center", va="center", size=14)#, size=60)
    axes[0,i].text(13, 0.95, "Block V4", ha="center", va="center", size=14)#, size=60)
    axes[0,i].text(17.5, 0.95, "Block IT", ha="center", va="center", size=14)#, size=60)
    
    axes[0,i].set_title(metric_type, fontsize = 20)
    axes[0,i].set_ylim(0.0, 1.)
    axes[0,i].set_xticks([6, 10, 16, 19, 20])
    axes[0,i].set_xticklabels(['v1 injection', 'v2 injection', 'v4 injection', 'IT injection', 'avgpool'], rotation=45, ha='right', fontsize=16)#, fontsize=60)
        
    #axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('neural fits')
fig.tight_layout()
plt.show()
plt.savefig('/home/ec3731/issa_analysis/nn-analysis/neuralfits_blocks_v2-mul.png')
#plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/neural_fits/plot1.png')

# -- 
"""epoch = 29
layers = np.arange(2)
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]
metric = ["neural_fits", 0]
metric_types = ['hvm-V4', 'hvm-IT', 'rust-V4', 'rust-IT']


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
plt.savefig('/home/ec3731/issa_analysis/nn-analysis/neuralfits_test2.png')
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/neural_fits/plot2.png')"""


"""epoch = 82
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

"""epoch = 29
layer = 16
metric = ["neural_fits", 0]
metric_types = ["hvm", "rust"]
model_names = [
    #"identity",
    "barlow_v1_inj", 
    "barlow_v2_inj", 
    "barlow_control"
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

plt.savefig(f'/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/figures/{epoch}_PCs_layer_{layer}_1.pdf')
plt.show()"""
