
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
            x[i,l] = random.randrange(input_range[0], input_range[1], input_range[2])

        for l, fun in enumerate(output_functions):
            
            x[i, n_input + l] = fun(x[i][:n_input])
            #print(x[i][:n_input], fun(x[i][:n_input]), x[i, l])

    return pd.DataFrame(x, columns = ['input_' + str(i + 1) for i in range(n_input)] + ['output_' + str(i + 1) for i in range(n_output)])


def create_topology(n_input, hidden_layers, n_output, act_vals = None):
    top = {}
    tot_unit = 0
    for i in range(n_input):
        top[i] = ['input', 'None', [], [l + n_input for l in range(hidden_layers[0])]]
        tot_unit += 1

    n_hidden_unit = sum(hidden_layers)
    if act_vals == None: act_vals = [[0.5]] * n_hidden_unit
    n_layer = len(hidden_layers)
    index_unit = n_input

    for i, layer in enumerate(hidden_layers):
        if i < n_layer - 1:
            succ = [index_unit + hidden_layers[i] + l for l in range(hidden_layers[i + 1])]
        else:
            succ = [index_unit + hidden_layers[i] + l for l in range(n_output)]

        for j in range(layer):
            top[index_unit + j] = ['hidden_' + str(i), 'sigmoid', act_vals[index_unit + j - n_input], succ]
        
        index_unit += hidden_layers[i]



    for i in range(n_output):
        top[index_unit + i] = ['output', 'identity', [], []]
        tot_unit += 1

    return top