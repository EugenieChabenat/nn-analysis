import os
import tables
import pickle
from collections import OrderedDict
import numpy as np
import torch

import config

__all__ = ['query_results', 'save_results', 'load_results']

def _query_results(filename, where, keys, table='data', order=None, suppress=False, atol=2.5e-3, **kwargs):
    table = None
    """
    For python version >= 3.6, if order = None then the dimensions of the results array will follow the order
    of the keyword arguments
    """
    list_kwargs = [(k, v) for k, v in kwargs.items() if isinstance(v, list) or isinstance(v, np.ndarray)]
    non_list_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, list) and not isinstance(v, np.ndarray)}
    if order is not None:
        list_kwargs.sort(key=lambda tup: order.index(tup[0]))
    shape = tuple([len(v) for k, v in list_kwargs]) if len(list_kwargs) > 0 else (1,)
    list_kwargs = OrderedDict(list_kwargs)
    
    results = {}
    filename = '/mnt/smb/locker/issa-locker/users/Eugénie/data/results/'
    with tables.open_file(filename, mode='r') as f:
        try:
            table = f.get_node(where, name=table)
        except Exception:
            raise RuntimeError(f"WHERE must be one of {', '.join(f.list_nodes('/'))}")
            
        for key in keys:
            results[key] = np.zeros((*shape,*table.col(key).shape[1:]))
        
        for i in range(prod(shape)):
            multi_index = np.unravel_index(i, shape)
            list_kwargs_i = {k: v[multi_index[j]] for j, (k, v) in enumerate(list_kwargs.items())}
            kwargs_i = {}
            kwargs_i.update(list_kwargs_i)
            kwargs_i.update(non_list_kwargs)
            
            condition = '&'.join([f'({k} == b"{v}")' if isinstance(v, str) else f'({k} == {v})' for k, v in kwargs_i.items()])
            potential_results = [{key: row[key] for key in keys} for row in table.where(condition)]
            if len(potential_results) == 0:
                print(kwargs_i)
                raise RuntimeError("No results returned.")
            if len(potential_results) > 1:
                new_potential_results = {key: [row[key] for row in potential_results if row[key] != table.description._v_dflts[key]] for key in keys}
                rows = [{colname: row[colname] for colname in table.colnames} for row in table.where(condition)]
                query_keys = list(kwargs_i.keys())
                if all([len(new_potential_results[key]) == 1 for key in keys]):
                    # only one non-default value is found for each key
                    potential_results = [{key: new_potential_results[key][0] for key in keys}]
                    pass
                elif all(all([abs(row[key]-rows[0][key]) < atol for key in keys]) for row in rows):
                    # Results are all nearly identical
                    pass
#                     if not suppress:
#                         row_indices = [row.nrow for row in table.where(condition)]
#                         print("Identical rows: ", row_indices, ". Using first entry.")
                elif all(all([row[colname] == rows[0][colname] for colname in table.colnames if colname in query_keys]) for row in rows):
                    # Results are not all nearly identical, but query keys are identical
                    if not suppress:
                        conflicting_results = {key: [row[key] for row in rows] for key in keys}
                        print(f"Conflicting entries: {conflicting_results}. Using first entry.")
                else:
                    # Results are not all nearly identical and have non-identical query keys
                    combined_rows = {colname: list(set([row[colname] for row in rows])) for colname in table.colnames}
                    differences = {colname: combined_rows[colname] for colname in table.colnames if len(combined_rows[colname]) > 1 and colname in query_keys}
                    print(f"Differences: {differences}")
                    raise RuntimeError("More than one result returned for query, try being more specific")
            for key in keys:
                results[key][multi_index] = potential_results[0][key]
    
    return results, list(list_kwargs.keys())

def query_results(filename, metrics, suppress=False, **kwargs):
    results = {}
    for metric in metrics:
        configs = config.metric_configs[metric]
        where, key = configs['where'], configs['key']
        other_kwargs = {k: v for k, v in configs.items() if k not in ['where', 'metric']}
        new_results, _ = _query_results(
            filename,
            where,
            [key],
            table='data',
            suppress=suppress,
            **kwargs,
            **other_kwargs,
        )
        results.update({metric: new_results[key]})
    return results

def save_results(results, filename, directory='data'):
    filename = os.path.join(directory, filename)
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
        
def load_results(filename, directory='data'):
    filename = os.path.join(directory, filename)
    with open(filename, 'rb') as f:
        results = pickle.load(f)
        return results
    
# utils

def prod(size):
    return torch.Tensor(list(size)).prod().int()
