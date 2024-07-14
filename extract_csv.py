import numpy as np
import pandas as pd

def load_carbon_csv_content(dir: str):
    csv_obj = pd.read_csv(dir)
    data = np.array((csv_obj['campaign_E'], csv_obj['resistance_E']))
    data = data.T
    target = np.array(csv_obj['target_cycle'])
    target = class_enumerate(target)
    return data, target

def class_enumerate(target):
    for i in range(len(target)):
        if target[i] == 'normal': target[i] = 1
        elif target[i] == 'alerta': target[i] = 2
        elif target[i] == 'anômalo': target[i] = 3
    return target
