import pandas as pd
import numpy as np
import os
import feature_extract


def data_clean(dict_x, dict_id_name):
    # if dataset not founded in id_to_name table, then remove the field.
    array_datasetName = dict_id_name['Dataset_name'].values.tolist()
    array_dictx = dict_x.copy().columns.values.tolist()
    b_find = False
    for item in array_dictx:
        for match in array_datasetName:
            if (item == match.strip()):
                b_find = True
                pass
            else:
                pass
            pass
        if (b_find == False):
            dict_x.drop(item, axis=1, inplace=True)
        else:
            b_find = False
    pass


def increase_Val(dict_kb):
    n_min = dict_kb.min().as_matrix().min()
    dict_kb = dict_kb + abs(n_min)
    return (dict_kb, n_min)
    pass


def convert(cell):
    if (str(cell).find('(') > 0):
        return cell[0:cell.find('(')]
        pass
    else:
        return cell
    return cell


def get_machineNames():
    dict_kb = pd.read_csv('data/kb.csv')
    result = dict_kb[dict_kb.columns[1]].values
    return result
    pass


def get_data(n_features=20):
    dict_kb = pd.read_csv('data/kb.csv')
    dict_pa = pd.read_csv('data/PA.csv')
    dict_rmse = pd.read_csv('data/rmse.csv')
    dict_id_to_name = pd.read_excel('data/ID_to_name.xlsx', "Sheet2", converters={
        'Dataset_ID ': convert,
        'Dataset_name': convert,
        'Dataset_run': convert,
        'Dataset_NOI': convert,
        'Dataset_NOF': convert
    })
    data_clean(dict_kb, dict_id_to_name)
    dict_kb.to_excel("data/input_kb.xlsx")
    (dict_kb, n_minkb) = increase_Val(dict_kb)
    data_clean(dict_pa, dict_id_to_name)
    data_clean(dict_rmse, dict_id_to_name)
    dict_pa.to_excel("data/input_pa.xlsx")
    dict_rmse.to_excel("data/input_rmse.xlsx")
    array_kb = dict_kb.as_matrix()
    array_pa = dict_pa.as_matrix()
    array_rmse = dict_rmse.as_matrix()
    panda_dataFeatures = feature_extract.extract_features('dataset_features', n_features)
    # get data features.

    return (dict_kb, dict_pa, dict_rmse, n_minkb, panda_dataFeatures)
    pass

# python extract_input.py
