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
             }



epoch = 29
#layers = np.arange(12)
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
metric = ["curve", 3]

                
metric_types = ["artificial_movie_labels", "natural_movie_labels"]


"""model_names = [
    "barlow_control", 
    "barlow_v1_inj", 
    "barlow_v2_inj", 
    "barlow_v1_inj_b", 
    "resnet50_untrained", 
    "barlow_twins_50epochs"
]

model_names = [
    "barlow_twins_50epochs_nm1", 
    "resnet50_untrained_nm1", 
    "injection_v1_nm1", 
    "injection_v2_nm1", 
    "injection_v4_nm1"
]"""

dict_color = {
    # random injection 
    "injection_v1" : ["orange", '-'],
    "injection_v2": ["orange", '-'], 
    "injection_v4": ["orange", '-'],
    "injection_IT": ["orange", '-'],
    
    # convolution injection
    "injection_conv_v1": ["red", '-'], 
    "injection_conv_v2": ["blue", '-'], 
    "injection_conv_v4": ["orange", '-'], 
    "injection_conv_IT": ["green", '-'], 
    
    # unfreeze convolution injection 
    "unfreeze_injection_v1": ["red", '-'], 
    "unfreeze_injection_v2": ["blue", '-'], 
    "unfreeze_injection_v4": ["orange", '-'], 
    "unfreeze_injection_IT": ["green", '-'], 
    
    # control models 
    "v4_no_injection": ["purple", '--'], 
    "resnet50_untrained": ["pink", '--'], 
    "barlow_twins_50epochs": ["grey", '--'], 
    "barlow_fact_no_injection": ["black", '--'], 

    "resnet50_untrained_bootstrap5": ["black", '--'], 
    "resnet50_allfeatures": ["blue", '--'], 
    "bt_bootstrap": ["red", '--'],
    "bt_allfeatures": ["green", '--']
}

model_names = [
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
    
    #"v4_no_injection", 
    #"resnet50_untrained", 
    #"barlow_twins_50epochs", 
    #"barlow_fact_no_injection"

    "resnet50_untrained_bootstrap5", 
    "resnet50_allfeatures", 
    "bt_bootstrap",
    "bt_allfeatures"
]

# ------------------------------------------------------------------------------------
# LAYERS PLOT 
# ------------------------------------------------------------------------------------
fig, axes = pt.core.subplots(1, len(metric_types), size=(10,8), sharex=True)
for i, metric_type in enumerate(metric_types):
    for model_name in model_names:
        print('model: ', model_name)
        print('layer: ', layers)
        scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
        if model_name == "barlow_v1_inj_b": 
            axes[0,i].plot(layers, scores, label="barlow_v3_inj")
        else: 
            axes[0,i].plot(layers, scores, label=model_name)
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
    axes[0,i].set_ylim(0.0, 0.8)
    axes[0,i].legend(loc='lower left')
    
fig.supxlabel('layers')
fig.supylabel('curvature')
fig.tight_layout()
plt.show()
plt.savefig('/home/ec3731/issa_analysis/nn-analysis/controls_nm.png')
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/natural_movies/controls_nm.png')

#plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/natural_movies/sanity_control/10frames_metrics_plot9.png')




# ------------------------------------------------------------------------------------
# ROUND PLOT 
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# HISTOGRAM PLOT 
# ------------------------------------------------------------------------------------
"""def grouped_bar(ax, xs, ys, width=0.2, sep=0.3):
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

baseline_model = "barlow_control"
model_names = [
    "barlow_v1_inj",
    "barlow_v2_inj",
    "barlow_v1_inj_b", 
]

fig, axes = pt.round_plot.subplots(1,3,height_per_plot=6,width_per_plot=6)
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
pt.round_plot.savefig(fig, '/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/FINAL_hist.png')
fig.show()"""



