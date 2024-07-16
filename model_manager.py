import tensorflow as tf
import pickle as pk
import os

def load_model_acc(acc_path: str):
    if not os.path.isfile(acc_path):
        with open(acc_path, 'wb+') as dir:
            nulled = 0.0
            pk.dump(nulled, dir)
            return float(nulled)
    else:
        with open(acc_path, 'rb+') as dir: return pk.load(dir)

def compare(prev_acc, cur_acc, acc_path):
    if cur_acc > prev_acc:
        dir_acc = open(acc_path, 'wb+')
        pk.dump(cur_acc, dir_acc)
        print('Modelo salvo!')
        return True
    else: return False
