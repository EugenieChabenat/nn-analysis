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
   
epoch = 29
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
one_layer = 0
metric = ["fact", 0]

#metric_types = ['ss_inv-background', 'ss_inv-obj_motion', 'ss_inv-crop','ss_inv-color']
                
metric_types = ['inv-background', 'inv-obj_motion', 'inv-crop', 'inv-color']
                
#metric_types = ['fact-background', 'fact-obj_motion', 'fact-crop', 'fact-color']

model_names = [
    "barlow_v1_inj",
    #"identity", 
    "barlow_v2_inj", 
    "barlow_v1_inj_b",
    "barlow_before_projector", 
    "barlow_control"
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
fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        print('model: ', model_name)
        print('layer: ', layers)
        
        #print('keys: ', load_data(metric, model_name, epoch, one_layer).keys())
        
        scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
        #axes[0,i].plot(layers, scores, label=model_name)
        if model_name == "barlow_v1_inj_b": 
            axes[0,i].plot(layers, scores, label="barlow_v3_inj")
        else: 
            axes[0,i].plot(layers, scores, label=model_name)
    #scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
    #axes[0,i].plot(layers, scores, label='identity')
    axes[0,i].set_title(metric_type)
    axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('fact')
fig.tight_layout()
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/fact/control_projector/plot2.png')


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
            ['ss_inv-background', 'ss_inv-obj_motion', 'ss_inv-crop','ss_inv-color'],
            ['inv-background', 'inv-obj_motion', 'inv-crop', 'inv-color'],
            ['fact-background', 'fact-obj_motion', 'fact-crop', 'fact-color']]

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
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/fact/FINAL_hist_layer13.png')
fig.show()"""




