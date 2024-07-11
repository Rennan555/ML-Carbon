import numpy as np
import pandas as pd

def load_carbon_csv_content(dir: str):
    csv_obj = pd.read_csv(dir)
    data = np.array((csv_obj['campaign_E'], csv_obj['resistance_E']))
    target = np.array(csv_obj['target_cycle'])
    return data, target
