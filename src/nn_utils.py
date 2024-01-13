
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import random
sys.path.append(os.path.abspath('./src/'))
from ActivationFunctions import *
from NeuralNetwork import *

from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 12345

# PLOT
def multy_plot(datas, labels, title, scale='linear'):
    x = np.arange(0, len(datas[0])).tolist()

    for i, el in enumerate(datas):
        plt.plot(x, el, label=labels[i])


    plt.title(title)
    plt.grid()
    plt.legend()
    plt.yscale(scale)
    plt.show()


# create data structures
def create_dataset(n_items, n_input, input_range, output_functions, seed):
    random.seed(seed)

    n_output = len(output_functions)
    x = np.ndarray((n_items, n_input + n_output))

    for i in range(n_items):
        for l in range(n_input):
            x[i,l] = random.uniform(input_range[0], input_range[1])

        for l, fun in enumerate(output_functions):
            
            x[i, n_input + l] = fun(x[i][:n_input])
            #print(x[i][:n_input], fun(x[i][:n_input]), x[i, l])

    return pd.DataFrame(x, columns = ['input_' + str(i + 1) for i in range(n_input)] + ['output_' + str(i + 1) for i in range(n_output)])


def create_stratified_topology(layers, act_args = None):

    orig_layers_len = len(layers)
    if act_args == None:
        if orig_layers_len > 2:
            act_args = [[]] * layers[0] + [[1]] * sum(layers[1:-1]) + [[]] * layers[-1]
        else:
            act_args = [[]] * layers[0] + [[]] * layers[-1]


    index_unit = 0
    layers.append(0)
    top = {}

    print(act_args)
    for i in range(orig_layers_len):

        if i == 0: 
            unit_type = 'input_'
            unit_act_fun = None
        elif i == orig_layers_len - 1: 
            unit_type = 'output_'
            unit_act_fun = 'identity'
        else: 
            unit_type = 'hidden_'
            unit_act_fun = 'sigmoid'

        succ = [index_unit + layers[i] + l for l in range(layers[i + 1])]

        for j in range(layers[i]):
            top[index_unit + j] = [unit_type + str(i), unit_act_fun, act_args[index_unit + j], succ]
        index_unit += layers[i]

    return top
