import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import tables
import pickle
from collections import OrderedDict
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge

from plots import visualize as vi
import config

def filter_nans(results, suppress=False):
    filtered_results = {}
    for model_name, result in results.items():
        if all([not np.isnan(scores).any() for scores in result.values()]):
            filtered_results[model_name] = result
        else:
            if not suppress:
                print(model_name)
    return filtered_results

def fit(results, test_models, excluded_metrics=[], alpha=10.0):
    train_X, test_Xs = [], {}
    train_y, test_ys = [], {}
    pred_ys = {}
    target_metrics = ['neural fits - V4', 'neural fits - IT']
    target_labels = ['V4', 'IT']
    excluded_metrics = target_metrics + excluded_metrics
    feature_metrics = [metric for metric in config.METRICS if metric not in excluded_metrics]
    
    for model_name, result in results.items():
        row_X = []
        for metric in feature_metrics:
            row_X.append(result[metric].flatten()) # Add layers and epochs data
        if model_name in test_models:
            test_Xs[model_name] = np.stack(row_X, axis=1)
        else:
            train_X.append(np.stack(row_X, axis=1))
        row_y = []
        for metric in target_metrics:
            row_y.append(result[metric].flatten()) # Add layers and epochs data
        if model_name in test_models:
            test_ys[model_name] = np.stack(row_y, axis=1)
        else:
            train_y.append(np.stack(row_y, axis=1))
    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    weights = {}
    
    if isinstance(alpha, list) or isinstance(alpha, np.ndarray):
        reg = RidgeCV(alphas=alpha, alpha_per_target=True)
    else:
        reg = Ridge(alpha=alpha)

    reg.fit(train_X,train_y)
    for model_name, test_X in test_Xs.items():
        pred_ys[model_name] = reg.predict(test_X)
    assert len(pred_ys) == len(test_ys) == len(test_Xs)
    
    fig, axes = vi.subplots(1,2)
    for i, target_label in enumerate(target_labels):
        axes[0,i].scatter(reg.predict(train_X)[:,i], train_y[:,i], label='train')
        for model_name in test_models:
            axes[0,i].scatter(pred_ys[model_name][:,i], test_ys[model_name][:,i], label=model_name)
        axes[0,i].set_title(f'{target_label}: {r2(reg.predict(train_X)[:,i], train_y[:,i]):.4f}')
        axes[0,i].legend()
        axes[0,i].grid()
        
        weights[target_label] = {metric: reg.coef_[i][j] for j, metric in enumerate(feature_metrics)}
#         weights[target_label] = {metric: reg[1].coef_[i][j] for j, metric in enumerate(feature_metrics)}

    plt.show()
    
    if isinstance(alpha, list) or isinstance(alpha, np.ndarray):
        print(reg.alpha_)
#         print(reg[1].alpha_)

    return weights

def plot_weights(weights):
    fig, axes = vi.subplots(1,3,height_per_plot=12, width_per_plot=8)
    for i, (target_label, metric_weights) in enumerate(weights.items()):
        x, y = list(metric_weights.keys()), list(metric_weights.values())
        x, y = zip(*sorted(zip(x,y),key=lambda pair: pair[1]))
        axes[0,i].barh(x,y)
        axes[0,i].set_title(target_label)
        xlabels = axes[0,i].get_xticklabels()
        for xlabel in xlabels:
            xlabel.set_rotation(90)
            
    ranks = []
    for target_label, metric_weights in weights.items():
        x, y = list(metric_weights.keys()), list(metric_weights.values())
        sorted_x, _ = zip(*sorted(zip(x,y),key=lambda pair: pair[1]))
        ranks.append(np.array([sorted_x.index(metric) for metric in x]))
    rank_diffs = ranks[1] - ranks[0]
    x, y = x, rank_diffs
    x, y = zip(*sorted(zip(x,y),key=lambda pair: pair[1]))
    axes[0,2].barh(x,y)
    axes[0,2].set_title('IT rank minus V4 rank')        
    
    plt.tight_layout()
    plt.show()
    
def correlation_scatter_plots(results, target_metric, categories):
    fig, axes = vi.subplots(8,4)
    colors = vi.get_colors(len(categories))
    other_metrics = [metric for metric in config.METRICS if metric != target_metric]
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            index = i*axes.shape[1]+j
            if index >= len(other_metrics):
                break
            metric = other_metrics[index]
            for k, (category, model_names) in enumerate(categories.items()):
                for model_name in model_names:
                    if model_name in results:
                        axes[i,j].scatter(results[model_name][metric].flatten(), results[model_name][target_metric].flatten(), color=colors[k], alpha=0.3, label=category)
            handles, labels = axes[i,j].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[i,j].legend(by_label.values(), by_label.keys())
            axes[i,j].set_title(metric)
    plt.show()
    
def correlation_per_layer_scatter_plots(results, target_metric, layers):
    fig, axes = vi.subplots(8,4)
    colors = vi.get_colors(len(layers))
    other_metrics = [metric for metric in config.METRICS if metric != target_metric]
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            index = i*axes.shape[1]+j
            if index >= len(other_metrics):
                break
            metric = other_metrics[index]
            for model_name, result in results.items():
                for k, layer in enumerate(layers):
                    axes[i,j].scatter(result[metric][:,layer].flatten(), result[target_metric][:,layer].flatten(), color=colors[k], alpha=0.3, label=f"layer {layer}")
            handles, labels = axes[i,j].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[i,j].legend(by_label.values(), by_label.keys())
            axes[i,j].set_title(metric)
    plt.show()
    
def filter_models(results, condition, **kwargs):
    return {model_name: result for model_name, result in results.items() if eval(condition, globals(), {**locals(),**kwargs})}

def filter_scores(results, func):
    return {model_name: {metric: func(scores) for metric, scores in result.items()} for model_name, result in results.items()}
    
def compute_correlations(results, method='pearson'):
    data = []
    for model_name, result in results.items():
        sub_data = []
        for metric, scores in result.items():
#             print(scores)
            sub_data.append(scores.flatten())
        sub_data = np.stack(sub_data, axis=0)
        data.append(sub_data)
    data = np.concatenate(data, axis=1)
    print(f"Data shape: {data.shape}")
    if method == 'pearson':
        return np.corrcoef(data)
    elif method == 'spearman':
        return spearmanr(data, axis=1)[0]
    else:
        raise RuntimeError("method must be either 'pearson' or 'spearman'.")
    
def plot_column(correlation_matrix, target_metric, sort='name'):
    x = [metric for metric in config.METRICS if metric != target_metric]
    if sort == 'name':
        x = sorted(x, key=lambda string: string.split('-')[-1] if '-' in string else string)
    row_indices = np.array([config.METRICS.index(metric) for metric in x])
    column_index = config.METRICS.index(target_metric)
    y = correlation_matrix[row_indices][:,column_index]
    
    if sort == 'value':
        x, y = zip(*sorted(zip(x,y),key=lambda pair: pair[1]))
    
    plt.figure(figsize=(6,10))
    plt.barh(x, y)
    plt.show()
    
def plot_matrix(correlation_matrix):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    im = ax.matshow(correlation_matrix)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(config.METRICS)))
    ax.set_xticklabels(config.METRICS, rotation=90)
    ax.set_yticks(np.arange(len(config.METRICS)))
    ax.set_yticklabels(config.METRICS)
    plt.show()
    
def partial_corr(x, z, y=None, mode='pearson'):
    """
    x, y, z - (n_features, n_samples) or (n_samples)
    Test code (Same as the example at https://en.wikipedia.org/wiki/Partial_correlation):
        x = np.array([2,4,15,20],dtype=float)
        y = np.array([1,2,3,4],dtype=float)
        z = np.array([0,0,1,1],dtype=float)
        print(np.corrcoef(x, y)[0,1]) # 0.9695015519208121
        print(partial_corr(x, z, y, mode='pearson')[0,1]) # 0.9191450300180579

        x = np.array([[2,4,15,20],[1,2,3,4]],dtype=float)
        z = np.array([[0,0,1,1],[0,0,0,0]],dtype=float)
        print(np.corrcoef(x)[0,1])  # 0.9695015519208121
        print(partial_corr(x, z, mode='pearson')[0,1])  # 0.9191450300180579
    """
    # This is code for computing partial correlation when x, y, z are all 1-dimensional, using
    # Kendall's Advanced Theory of Statistics Volume 2 p. 318 Eq. 27.5
#     assert mode == 'pearson'
#     if len(x.shape) == 1 and len(y.shape) == 1 and len(z.shape) == 1:
#         rho_xz = np.corrcoef(x, z)[0,1]
#         rho_yz = np.corrcoef(y, z)[0,1]
#         rho_xy = np.corrcoef(x, y)[0,1]
#         return (rho_xy - rho_xz*rho_yz)/((1-rho_xz**2)*(1-rho_yz**2))**0.5
    
    # This is general code for computing partial correlation based on wikipedia article
    # https://en.wikipedia.org/wiki/Partial_correlation. Results empirically agree with previous code
    # on data where x, y, z are all 1-dimensional
    if len(z.shape) == 1:
        z = z.reshape(1,-1) # (n_features=1, n_samples)
    if len(x.shape) == 1:
        x = x.reshape(1,-1) # (n_features=1, n_samples)
    if y is not None and len(y.shape) == 1:
        y = y.reshape(1,-1) # (n_features=1, n_samples)
    assert (y is None and x.shape[1] == z.shape[1]) or (x.shape[1] == y.shape[1] == z.shape[1])
    reg = LinearRegression()
    reg.fit(z.T, x.T) # reg.fit((n_samples, n_features_z), (n_samples, n_targets=n_features_x))
    err_x = x - reg.predict(z.T).T # (n_features_x, n_samples)
    if y is not None:
        reg.fit(z.T, y.T)
        err_y = y - reg.predict(z.T).T # (n_features_y, n_samples)
    else:
        err_y = None
    if mode == 'pearson':
        return np.corrcoef(err_x, y=err_y)
    if mode == 'spearman':
        return spearmanr(err_x, b=err_y, axis=1)
    raise RuntimeError("mode must be 'pearson' or 'spearman'")
    
def semipartial_corr(x, z, mode='pearson'):
    if len(z.shape) == 1:
        z = z.reshape(1,-1) # (n_features=1, n_samples)
    if len(x.shape) == 1:
        x = x.reshape(1,-1) # (n_features=1, n_samples)
    assert x.shape[1] == z.shape[1]
    reg = LinearRegression()
    reg.fit(z.T, x.T) # reg.fit((n_samples, n_features_z), (n_samples, n_targets=n_features_x))
    err_x = x - reg.predict(z.T).T # (n_features_x, n_samples)
    if mode == 'pearson':
        return np.corrcoef(x, y=err_x)[:x.shape[0],x.shape[0]:]
    if mode == 'spearman':
        return spearmanr(x, b=err_x, axis=1)[:x.shape[0],x.shape[0]:]
    raise RuntimeError("mode must be 'pearson' or 'spearman'")
    
def subcategorybar(X, vals, labels, width=0.6):
    plt.figure(figsize=(10,6))
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge", label=labels[i])
    plt.legend()
    plt.xticks(_X, X, rotation=90)
    

# utils

def prod(size):
    return torch.Tensor(list(size)).prod().int()

def numpy_to_torch(func):
    """
    Converts all numpy arugments to torch.
    In current implementation, if there is a mix of torch and numpy arguments,
    the torch arguments must be on CPU.
    """
    @wraps(func)
    def decorated_func(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[i] = torch.from_numpy(arg).float()
        args = tuple(args)
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                kwargs[k] = torch.from_numpy(v).float()
        return func(*args, **kwargs)
    return decorated_func

@numpy_to_torch
def cc(pred_y, y, weights=None):
    # pred_y, y - (n_samples, n_targets) or (n_samples)
    # return (n_targets)
    if weights is not None:
        raise NotImplementedError()
    v1, v2 = y - y.mean(dim=0), pred_y - pred_y.mean(dim=0)
    return (v1*v2).sum(dim=0)/(v1.norm(dim=0)*v2.norm(dim=0))

@numpy_to_torch
def r2(pred_y, y, weights=None):
    # pred_y, y - (n_samples, n_targets) or (n_samples)
    # return (n_targets)
    if weights is not None:
        raise NotImplementedError()
    return 1 - ((y - pred_y).norm(dim=0)**2)/((y - y.mean(dim=0)).norm(dim=0)**2)

@numpy_to_torch
def mse(pred_y, y, weights=None):
    # pred_y, y - (n_samples, n_targets) or (n_samples)
    # return (n_targets)
    if weights is not None:
        return (weights*(y - pred_y)**2).mean(dim=0)
    return ((y - pred_y)**2).mean(dim=0)
