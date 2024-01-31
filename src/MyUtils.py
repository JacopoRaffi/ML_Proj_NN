
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import random
import matplotlib.colors as mcolors
import plotly.express as px
import pathlib

sys.path.append(os.path.abspath('./'))
from ActivationFunctions import *
from NeuralNetwork import *

'''
The collections of useful function used in then project and useful global variables
'''

RANDOM_STATE = 549123
COLUMNS_ORDER = ['topology', 'stats',
 'batch_size',
 'min_epochs',
 'max_epochs',
 'patience',
 'error_increase_tolerance',
 'lambda_tikhonov',
 
 'learning_rate',
 'alpha_momentum',
 'lr_decay_tau',
 
 'adamax',
 'adamax_learning_rate',
 'exp_decay_rate_1',
 'exp_decay_rate_2',
 
 'mean_mean_euclidean_error',
 'mean_mean_squared_error',
 'var_mean_euclidean_error',
 'var_mean_squared_error',
 'mean_accuracy',
 'var_accuracy',
 'mean_best_validation_training_error']

# -- TRAIN -- 

def train_from_index(df:pd.DataFrame, tr_set:np.ndarray, val_set:np.ndarray, index:int, topologies_dict:dict, early_stop:bool=False):
    '''
    Train a model described in a row of a dataframe in a give index
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe that contains the model's description
    tr_set: np.ndarray
        The training set
    val_set: np.ndarray
        the validation set
    index: int
        The index of the model to be trained
    topologies_dict: dict
        Dictionary that contains desctiption of the topologies contained in the dataframe
    early_stop: bool
        Whether to use or not the value contained in df to set the early stopping based on tr_error
        
    Returns
    -------
    return: [NeuralNetwork, dict]
        The model trained
        The model's training stats
    '''
        
    default_values =  {
            'range_min' : -0.75,
            'range_max' : 0.75,
            'fan_in' : True,
            'random_state' : RANDOM_STATE,

            'lambda_tikhonov' : 0.0,
            
            'learning_rate' : 0.1,
            'alpha_momentum' : 0.0,
            'lr_decay_tau' : 0,
            'nesterov' : False,
            
            'adamax': False,
            'adamax_learning_rate': 0.002,
            'exp_decay_rate_1':0.9,
            'exp_decay_rate_2':0.999,
            
            'batch_size' : 1,
            'min_epochs' : 0,
            'max_epochs' : 512,
            'patience' : 5,
            'error_increase_tolerance' : math.inf,
            'retrainig_es_error': -1,

            'metrics':[ErrorFunctions.mean_squared_error, ErrorFunctions.mean_euclidean_error],      
            'topology': {}, # must be inizialized

            'collect_data':True, 
            'collect_data_batch':False, 
            'verbose':True
            }
    
    for i in df.columns:
        if i in default_values:
            default_values[i] = df.iloc[index][i]
    
    # set critical values
    default_values['learning_rate'] = default_values['learning_rate'] / default_values['batch_size']
    default_values['adamax_learning_rate'] = default_values['adamax_learning_rate'] / default_values['batch_size']
    default_values['eta_tau'] = default_values['learning_rate']/100 # eta tau more or less 1% of eta_0
    default_values['topology'] = topologies_dict[df.iloc[index]['topology']]
    
    default_values['training_set'] = tr_set
    default_values['validation_set'] = val_set
    
    if early_stop:
        default_values['retraing_es_error'] = df.iloc[index]['mean_best_validation_training_error']
    
    
    train_args = [default_values[key] for key in NeuralNetwork.train_input] 
    
    # create a new model
    NN = NeuralNetwork(default_values['topology'], default_values['range_min'], default_values['range_max'], default_values['fan_in'], default_values['random_state'])
    # train the model
    stats = NN.train(*train_args)
    return [NN, stats]
    
    
# -- PLOT --
def multy_plot(datas, labels, title=None, scale='linear', ax=None, legend=True, style=False, font_size=14):
    '''
    Display a multy scatter plot using the given data ad parameters, and using matplotlib   
    '''
    x = np.arange(0, len(datas[0])).tolist()
    plt.rcParams.update({'font.size': font_size})
    styles = ['-.', '-']
    if ax != None: plt.sca(ax=ax)
    if style:
        for i, el in enumerate(datas):
            plt.plot(x, el, label=labels[i], linestyle=styles[i])
    else:
        for i, el in enumerate(datas):
            plt.plot(x, el, label=labels[i])
    plt.title(title)
    plt.grid()
    if legend:
        plt.legend()
    plt.yscale(scale)
    plt.xlabel('epochs')
    if ax == None: plt.show()

def multy_plot_3d(x, y, z, label, title):
    '''
    Display a 3D scatter plot using the given data ad parameters, and using matplotlib   
    '''
    print('Tot points:', len(x[0]))
    color_list = list(mcolors.TABLEAU_COLORS)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle(title)

    for i in range(len(x)):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        # For each set of style and range settings, plot n random points in the box
        # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
        ax.scatter(x[i], y[i], z[i], marker='o', color=color_list[i])
        ax.set_xlabel(label[i][0])
        ax.set_ylabel(label[i][1])
        ax.set_zlabel(label[i][2])

    return fig

def interactive_3d_plot(dataframe, x_col, y_col, z_col, color_col, size_col=None, max_size=None, symbol_col=None):
    '''
    Display a 3D scatter plot using the given data ad parameters, and using Plotly Express
    '''
    print('Tot points:', len(dataframe))
    fig = px.scatter_3d(dataframe, x=x_col, y=y_col, z=z_col,
                color=color_col, opacity=0.7, size=size_col, size_max=max_size, symbol=symbol_col)

    fig.update_layout(
        margin=dict(l=100, r=100, t=10, b=10),
        paper_bgcolor="LightSteelBlue",
    )
    return fig

# -- create data structures --
def create_dataset(n_items, n_input, input_range, output_functions, seed):
    '''
    Create a dummy dataset for early training tests given the inputs
    '''
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


def create_stratified_topology(layers, act_fun_list = None):
    '''
    Create a stratified topology from the number of layers and the sctivation functions list,
    used to describe a neural network structure
    '''
    orig_layers_len = len(layers)
    if act_fun_list == None:
        if orig_layers_len > 2:
            act_fun_list = [[None, []]] * layers[0] + [['sigmoid',[1]]] * sum(layers[1:-1]) + [['identity', []]] * layers[-1]
        else:
            act_fun_list = [[None, []]] * layers[0] + [['identity', []]] * layers[-1]

    index_unit = 0
    layers.append(0)
    top = {}

    for i in range(orig_layers_len):
        if i == 0: 
            unit_type = 'input_'
        elif i == orig_layers_len - 1: 
            unit_type = 'output_'
        else: 
            unit_type = 'hidden_'

        succ = [index_unit + layers[i] + l for l in range(layers[i + 1])]

        for j in range(layers[i]):
            unit_act_fun = act_fun_list[index_unit + j][0]
            act_args = act_fun_list[index_unit + j][1]

            top[index_unit + j] = [unit_type + str(i), unit_act_fun, act_args, succ]
        index_unit += layers[i]

    return top

def monk_to_csv():
    '''
    used to convert the monk datasets in csv
    '''
    for j in range(1,4):
        datas_tr = {'input_1':[],
                'input_2':[],
                'input_3':[],
                'input_4':[],
                'input_5':[],
                'input_6':[],
                'output_1':[]}
        datas_ts = {'input_1':[],
                'input_2':[],
                'input_3':[],
                'input_4':[],
                'input_5':[],
                'input_6':[],
                'output_1':[]}
        
        for line in open(pathlib.Path(__file__).parent.parent.joinpath('data\\monks\\monks-' + str(j) + '.test')):
            line_divided = line.split(' ')[1:]
            datas_ts['output_1'].append(line_divided[0])
            for i in range(1,7):
                datas_ts['input_' + str(i)].append(line_divided[i])

        for line in open(pathlib.Path(__file__).parent.parent.joinpath('data\\monks\\monks-' + str(j) + '.train')):
            line_divided = line.split(' ')[1:]
            datas_tr['output_1'].append(line_divided[0])
            for i in range(1,7):
                datas_tr['input_' + str(i)].append(line_divided[i])

        df = pd.DataFrame(datas_tr)
        df.to_csv(pathlib.Path(__file__).parent.parent.joinpath('data\\monks_csv\\monks_tr_' + str(j) + '.csv'))
        df = pd.DataFrame(datas_ts)
        df.to_csv(pathlib.Path(__file__).parent.parent.joinpath('data\\monks_csv\\monks_ts_' + str(j) + '.csv'))

