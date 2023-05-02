import numpy as np
import matplotlib.pyplot as plt

from plots import visualize as vi

__all__ = ['plot_results', 'filter_nans', 'filter_models', 'filter_scores', 'compare_results', 'compare_best_results', 'compare_layer_results', 'radar_plot']

def plot_results(results, mode, n_per_row=4, grid=False):
    assert all_equal([results[model_name][metric].shape for model_name in results.keys() for metric in results[model_name].keys()])
    
    model_names = list(results.keys())
    metrics = list(results[model_names[0]].keys())
    data = np.stack([np.stack([results[model_name][metric] for metric in metrics],axis=0) for model_name in model_names],axis=0)
    epochs, layers = results[model_names[0]][metrics[0]].shape
    epochs, layers = np.arange(epochs), np.arange(layers)
    
    for i in range(0,len(metrics),n_per_row):
        if mode == 'layer-wise':
            assert len(epochs) == 1
            vi.grid_plot(data[:,i:i+n_per_row], model_names, metrics[i:i+n_per_row], epochs, layers, names=['model','metric','epoch','layers'], permute=[0,2,1,3], grid=grid)
        elif mode == 'epoch-wise':
            assert len(layers) == 1
            vi.grid_plot(data[:,i:i+n_per_row], model_names, metrics[i:i+n_per_row], epochs, layers, names=['model','metric','epoch','layers'], permute=[0,3,1,2], grid=grid)
            
def filter_nans(results, suppress=False):
    filtered_results = {}
    for model_name, result in results.items():
        if all([not np.isnan(scores).any() for scores in result.values()]):
            filtered_results[model_name] = result
        else:
            if not suppress:
                print(model_name)
    return filtered_results

def filter_models(results, condition, **kwargs):
    return {model_name: result for model_name, result in results.items() if eval(condition, globals(), {**locals(),**kwargs})}

def filter_scores(results, func):
    return {model_name: {metric: func(scores) for metric, scores in result.items()} for model_name, result in results.items()}

def compare_results(results_1, results_2, mode, n_per_row=4):    
    model_names = [model_name for model_name in results_1.keys() if model_name in results_2.keys()]
    metrics = [metric for metric in results_1[model_names[0]].keys() if metric in results_2[model_names[0]].keys()]
    
    results = {model_name: {metric: results_2[model_name][metric] - results_1[model_name][metric] for metric in metrics} for model_name in model_names}
    
    plot_results(results, mode, n_per_row=n_per_row)
    
def compare_best_results(results_1, results_2):
    model_names = [model_name for model_name in results_1.keys() if model_name in results_2.keys()]
    metrics = [metric for metric in results_1[model_names[0]].keys() if metric in results_2[model_names[0]].keys()]
    
    results = {model_name: {metric: results_2[model_name][metric].max() - results_1[model_name][metric].max() for metric in metrics} for model_name in model_names}
    
    x = np.arange(len(metrics))
    plt.figure(figsize=(10,6))
    for model_name in model_names:
        plt.scatter(x, [results[model_name][metric] for metric in metrics], label=model_name)
    plt.plot(x, np.zeros(len(metrics)))
    plt.legend()
    plt.xticks(ticks=x, labels=metrics, rotation=90.0)
    plt.ylabel('Difference')
    plt.show()
    
def compare_layer_results(results_1, results_2, layer):
    model_names = [model_name for model_name in results_1.keys() if model_name in results_2.keys()]
    metrics = [metric for metric in results_1[model_names[0]].keys() if metric in results_2[model_names[0]].keys()]
    
    results = {model_name: {metric: results_2[model_name][metric][:,layer].max() - results_1[model_name][metric][:,layer].max() for metric in metrics} for model_name in model_names}
    
    x = np.arange(len(metrics))
    plt.figure(figsize=(10,6))
    for model_name in model_names:
        plt.scatter(x, [results[model_name][metric] for metric in metrics], label=model_name)
#     plt.scatter(x, [results_1['moco_v1_cam_color_decode2_v2'][metric][:,layer].max() - results_1['moco_v1_pretrained'][metric][:,layer].max() for metric in metrics], label='from scratch')
#     plt.scatter(x, [results_1['moco_v1_factorize2_v3'][metric][:,layer].max() - results_1['moco_v1_pretrained'][metric][:,layer].max() for metric in metrics], label='fact + contrastive')
    plt.scatter(x, [results_1['moco_v1_factorize_cam_color_decode2_v2'][metric][:,layer].max() - results_1['moco_v1_pretrained'][metric][:,layer].max() for metric in metrics], label='fact + decode + contrastive')
    plt.plot(x, np.zeros(len(metrics)))
    plt.legend()
    plt.xticks(ticks=x, labels=metrics, rotation=90.0)
    plt.ylabel('Difference')
    plt.show()
    
def radar_plot(ax, x, y, color=None, label=None, label_loc=(0.95,0.95)):
    vi.r_plot(ax, x, y, color=color, label=label)
    vi.r_xticks(ax, x)
    vi.r_yticks(ax)
    vi.r_legend(ax, loc=label_loc)
#     plt.show()
    
# utils

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
