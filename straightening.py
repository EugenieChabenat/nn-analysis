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
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
metric = ["curve", 1]
#metric_types = ["x_cam_trans", "y_cam_trans", "x_cam_rot", "y_cam_rot"]
#metric_types = ['x_pan-detailed', 'x_pan', 'y_pan-detailed', 'y_pan', 'z_pan-detailed', 'z_pan']

#metric_types = ['x_focus_pan-detailed', 'x_focus_pan', 
 #               'y_focus_pan-detailed', 'y_focus_pan', 
 #              'z_focus_pan-detailed', 'z_focus_pan']

#metric_types = ['x_cam_pan-detailed', 'x_cam_pan', 
#                  'yz_cam_pan-detailed','yz_cam_pan']
                
#metric_types = ['x_focus_pan_0-detailed', 'x_focus_pan_0', 'x_focus_pan_1-detailed', 'x_focus_pan_1', 'x_focus_pan_2-detailed', 
#               'x_focus_pan_2', 'x_focus_pan_3-detailed', 'x_focus_pan_3', 'x_focus_pan_4-detailed', 'x_focus_pan_4', 'x_focus_pan_5-detailed', 
#               'x_focus_pan_5', 'x_focus_pan_6-detailed', 'x_focus_pan_6', 'x_focus_pan_7-detailed', 'x_focus_pan_7']
                
#metric_types = ['x_camel_rotate-detailed', 'x_camel_rotate',
#               'y_camel_rotate-detailed', 'y_camel_rotate']
                
metric_types = ['x_obj_rot', 'y_obj_rot', 'z_cam_trans']

"""metric_types = ['x_pan-detailed', 'x_pan', 'y_pan-detailed', 'y_pan', 'z_pan-detailed', 'z_pan', 'x_focus_pan-detailed', 'x_focus_pan', 
                'y_focus_pan-detailed', 'y_focus_pan', 'z_focus_pan-detailed', 'z_focus_pan', 'x_cam_pan-detailed', 'x_cam_pan', 'yz_cam_pan-detailed',
                'yz_cam_pan', 'x_focus_pan_0-detailed', 'x_focus_pan_0', 'x_focus_pan_1-detailed', 'x_focus_pan_1', 'x_focus_pan_2-detailed', 
                'x_focus_pan_2', 'x_focus_pan_3-detailed', 'x_focus_pan_3', 'x_focus_pan_4-detailed', 'x_focus_pan_4', 'x_focus_pan_5-detailed', 
                'x_focus_pan_5', 'x_focus_pan_6-detailed', 'x_focus_pan_6', 'x_focus_pan_7-detailed', 'x_focus_pan_7', 'x_camel_rotate-detailed', 
                'x_camel_rotate', 'y_camel_rotate-detailed', 'y_camel_rotate', 'x_cam_rot', 'y_cam_rot', 'x_cam_trans', 'y_cam_trans', 'z_cam_trans', 
                'x_obj_rot', 'y_obj_rot']"""


"""model_names = [
    "barlow_v1_inj",
    #"identity", 
    "barlow_v2_inj", 
    "barlow_v1_inj_b",
    "barlow_control", 
    #"barlow_v1_equi", 
    #"barlow_v3_equi"
]"""

model_names = [
    "barlow_faces_texture",
    "barlow_faces_notexture",
    "barlow_faces_control", 
    "faces_pretrained_notexture"
    #"barlow_control"
]



# ------------------------------------------------------------------------------------
# LAYERS PLOT 
# ------------------------------------------------------------------------------------
"""fig, axes = pt.core.subplots(1, len(metric_types), size=(5,4), sharex=True)
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
    axes[0,i].plot(layers, scores, label='identity')
    axes[0,i].set_title(metric_type)
    axes[0,i].legend()
fig.supxlabel('layers')
fig.supylabel('curvature')
fig.tight_layout()
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/straightening/3FACES_vs_control_straight5.png')"""




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
fig.show()




